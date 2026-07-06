# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""
SIMT cross-workitem all-reduce.

All-reduce ops are emitted **inline** at the current insertion point.
Three reducer variants: ``simt_allreduce_sum``, ``simt_allreduce_max``, ``simt_allreduce_min``.

Dispatch tree (compile-time, since *threads* / *scale* are Python ints)::

    threads <= scale                                       →  identity
    threads ≤ 32,  pow2(threads), pow2(scale)              →  warp_reduce
    threads ≤ 32                                           →  ub_reduce
    threads > 32,  pow2(threads), scale≤32, pow2(scale)   →  cross_warp_reduce
    otherwise                                              →  ub_reduce (fallback)
"""

from __future__ import annotations

from . import scalar
from ._control_flow import if_, for_
from ._ops import const as _const, get_laneid, get_tid_x, redux_add, redux_max, redux_min, shuffle_bfly, syncthreads
from ._surface_values import unwrap_surface_value
from ._types import _resolve, float16 as _f16_dtype, float32 as _f32_dtype

from mlir.ir import F16Type, F32Type


# ── helpers ────────────────────────────────────────────────────────────────────

def _is_pow2(n: int) -> bool:
    """Compile-time power-of-two check."""
    return n > 0 and (n & (n - 1)) == 0


# ── reducer dispatch tables ────────────────────────────────────────────────────

_REDUCER_IDENTITY = {
    "sum": {"f32": 0.0, "f16": 0.0},
    "max": {"f32": float("-inf"), "f16": float("-inf")},
    "min": {"f32": float("inf"), "f16": float("inf")},
}

_REDUCER_COMBINE = {
    "sum": lambda a, b: a + b,
    "max": scalar.max,
    "min": scalar.min,
}

_REDUCER_REDUX = {
    "sum": redux_add,
    "max": redux_max,
    "min": redux_min,
}


# ── butterfly  ──────────────────────────────────────────────────────────────────

def _emit_butterfly(v, *, threads: int, scale: int, reducer: str):
    """Unrolled butterfly shuffle reduce."""
    combine = _REDUCER_COMBINE[reducer]
    cur = threads
    while cur > scale:
        offset = cur // 2
        v = combine(v, shuffle_bfly(v, offset))
        cur //= 2
    return v


# ── warp_hw_reduce  ────────────────────────────────────────────────────────────

def _emit_warp_hw_reduce(x, *, threads: int, lane_in_warp, dtype: str, reducer: str):
    """Warp-level hardware reduce with group masking."""
    redux_fn = _REDUCER_REDUX[reducer]
    groups = 32 // threads

    if groups == 1:
        return redux_fn(x)

    c_identity = _const(
        _REDUCER_IDENTITY[reducer][dtype],
        dtype=_resolve(_f32_dtype if dtype == "f32" else _f16_dtype),
    )
    my_group = lane_in_warp // threads

    for g in range(groups):
        in_group = my_group == g
        masked = scalar.select(in_group, x, c_identity)
        reduced = redux_fn(masked)
        x = scalar.select(in_group, reduced, x)
    return x


# ── warp_reduce  ───────────────────────────────────────────────────────────────

def _emit_warp_reduce(x, *,
                      dtype, threads, scale, thread_offset, reducer):
    """Single-warp all-reduce."""
    extent = threads // scale
    if extent <= 1:
        return x

    if thread_offset:
        lane_in_warp = (get_tid_x() - thread_offset) & 31
    else:
        lane_in_warp = get_laneid()

    if extent >= 16 and scale == 1:
        return _emit_warp_hw_reduce(
            x, threads=threads,
            lane_in_warp=lane_in_warp, dtype=dtype, reducer=reducer,
        )
    return _emit_butterfly(x, threads=threads, scale=scale, reducer=reducer)


# ── cross_warp_reduce  ─────────────────────────────────────────────────────────

def _emit_cross_warp_reduce(x, scratch, *,
                            dtype, threads, scale, thread_offset, reducer):
    """Cross-warp all-reduce (threads > 32)."""
    num_warps = threads // 32
    c_identity = _const(
        _REDUCER_IDENTITY[reducer][dtype],
        dtype=_resolve(_f32_dtype if dtype == "f32" else _f16_dtype),
    )
    combine = _REDUCER_COMBINE[reducer]
    redux_fn = _REDUCER_REDUX[reducer]

    # ── thread indexing ──────────────────────────────────────────────────
    tid_x = get_tid_x()
    if thread_offset:
        tx = tid_x - thread_offset
        wid = tx // 32
        lid = tx & 31
    else:
        tx = tid_x
        wid = tx // 32
        lid = get_laneid()

    # ── per-warp reduce ──────────────────────────────────────────────────
    if scale == 1:
        warp_val = redux_fn(x)
    else:
        warp_val = _emit_butterfly(x, threads=32, scale=scale, reducer=reducer)

    # ── warp leaders write partial results ───────────────────────────────
    is_writer = lid < scale
    with if_(is_writer) as br:
        with br.then_:
            slot = wid * scale + lid
            scalar.store(warp_val, scratch, scalar.index_cast(slot))

    syncthreads()

    # ── leader warp reduces partial sums ─────────────────────────────────
    is_leader_warp = tx < 32
    with if_(is_leader_warp) as br:
        with br.then_:
            if scale == 1:
                need_load = lid < num_warps
                with if_(need_load) as inner_br:
                    with inner_br.then_:
                        tmp = scalar.load(scratch, scalar.index_cast(lid))
                        inner_br.assign(loaded=tmp)
                    with inner_br.else_:
                        inner_br.assign(loaded=c_identity)
                loaded = inner_br.loaded
                stage4_result = redux_fn(loaded)
            elif scale * num_warps <= 32:
                total = scale * num_warps
                need_load = lid < total
                with if_(need_load) as inner_br:
                    with inner_br.then_:
                        tmp = scalar.load(scratch, scalar.index_cast(lid))
                        inner_br.assign(loaded=tmp)
                    with inner_br.else_:
                        inner_br.assign(loaded=c_identity)
                loaded = inner_br.loaded
                stage4_result = _emit_butterfly(
                    loaded, threads=total, scale=scale, reducer=reducer,
                )
            else:
                is_reducer = lid < scale
                reduced = c_identity
                my_slot = lid % scale
                for w in range(num_warps):
                    idx_val = w * scale + my_slot
                    loaded_v = scalar.load(scratch, scalar.index_cast(idx_val))
                    reduced = combine(reduced, loaded_v)
                stage4_result = scalar.select(is_reducer, reduced, c_identity)

            br.assign(stage4_result=stage4_result)
        with br.else_:
            br.assign(stage4_result=c_identity)

    partial_reduced = br.stage4_result

    # ── global leader writes result ──────────────────────────────────────
    is_global_leader = tx < scale
    with if_(is_global_leader) as br5:
        with br5.then_:
            scalar.store(partial_reduced, scratch, scalar.index_cast(tx))

    # ── broadcast ────────────────────────────────────────────────────────
    syncthreads()
    result = scalar.load(scratch, scalar.index_cast(tx % scale))
    syncthreads()

    return result


# ── ub_reduce  ─────────────────────────────────────────────────────────────────

def _emit_ub_reduce(x, scratch, *,
                    dtype, threads, scale, thread_offset, reducer):
    """UB-scratch all-reduce (fallback for non-pow2 or general case)."""
    combine = _REDUCER_COMBINE[reducer]

    # ── thread indexing ──────────────────────────────────────────────────
    tid_x = get_tid_x()
    tx = (tid_x - thread_offset) if thread_offset else tid_x
    group = tx // threads
    lane = tx % threads

    # ── each lane writes x → scratch[tx] ─────────────────────────────────
    scalar.store(x, scratch, scalar.index_cast(tx))
    syncthreads()

    # ── reducers sequentially combine ────────────────────────────────────
    is_reducer = lane < scale
    with if_(is_reducer) as br:
        with br.then_:
            group_offset = group * threads
            first_elem = group_offset + lane
            acc = scalar.load(scratch, scalar.index_cast(first_elem))

            carry_loop = for_(scale, threads, step=scale).carry(acc=acc)
            with carry_loop:
                prev = carry_loop.acc
                elem = first_elem + carry_loop.iv
                loaded = scalar.load(scratch, elem)
                carry_loop.update(acc=combine(prev, loaded))
            acc = carry_loop.final("acc")

            br.assign(flag=acc)
        with br.else_:
            br.assign(flag=x)

    flag = br.flag
    syncthreads()

    # ── per-class leader writes back ─────────────────────────────────────
    is_leader = lane < scale
    with if_(is_leader) as br5:
        with br5.then_:
            scalar.store(flag, scratch, scalar.index_cast(group * threads + lane))

    # ── broadcast ────────────────────────────────────────────────────────
    syncthreads()
    result = scalar.load(scratch, scalar.index_cast(group * threads + (tx % scale)))
    syncthreads()

    return result


# ── public API  ────────────────────────────────────────────────────────────────

def _check_params(*, threads, scale, thread_offset):
    """Validate allreduce parameters (compile-time checks)."""
    for name, val in (("threads", threads), ("scale", scale),
                       ("thread_offset", thread_offset)):
        if not isinstance(val, int):
            raise ValueError(
                f"all_reduce: '{name}' must be a Python int, "
                f"got {type(val).__name__}"
            )
    if threads < 1:
        raise ValueError(f"all_reduce: threads must be >= 1, got {threads}")
    if scale < 1:
        raise ValueError(f"all_reduce: scale must be >= 1, got {scale}")
    if thread_offset < 0:
        raise ValueError(
            f"all_reduce: thread_offset must be >= 0, got {thread_offset}"
        )
    if threads % scale != 0:
        raise ValueError(
            f"all_reduce requires threads % scale == 0; "
            f"got threads={threads}, scale={scale}"
        )


def _simt_allreduce(value, *, threads, scale, thread_offset, scratch, reducer):
    """Unified allreduce dispatch tree."""
    _check_params(threads=threads, scale=scale, thread_offset=thread_offset)

    if threads <= scale:
        return value

    raw_value = unwrap_surface_value(value)
    if raw_value.type == F32Type.get():
        dtype = "f32"
    elif raw_value.type == F16Type.get():
        dtype = "f16"
    else:
        raise NotImplementedError(f"all_reduce: unsupported dtype {raw_value.type}")

    args = dict(dtype=dtype, threads=threads, scale=scale,
                thread_offset=thread_offset, reducer=reducer)

    if threads <= 32 and _is_pow2(threads) and _is_pow2(scale):
        return _emit_warp_reduce(value, **args)

    if scratch is None:
        raise ValueError(
            f"all_reduce {reducer}/{dtype}/t{threads}/s{scale}/o{thread_offset} "
            "requires a UB scratch buffer"
        )

    if threads <= 32:
        return _emit_ub_reduce(value, scratch, **args)

    if scale <= 32 and _is_pow2(threads) and _is_pow2(scale):
        return _emit_cross_warp_reduce(value, scratch, **args)

    return _emit_ub_reduce(value, scratch, **args)


def simt_allreduce_sum(value, *, threads, scale=1, thread_offset=0, scratch=None):
    """Sum reduce across SIMT work-items."""
    return _simt_allreduce(value, threads=threads, scale=scale,
                           thread_offset=thread_offset, scratch=scratch, reducer="sum")


def simt_allreduce_max(value, *, threads, scale=1, thread_offset=0, scratch=None):
    """Max reduce across SIMT work-items."""
    return _simt_allreduce(value, threads=threads, scale=scale,
                           thread_offset=thread_offset, scratch=scratch, reducer="max")


def simt_allreduce_min(value, *, threads, scale=1, thread_offset=0, scratch=None):
    """Min reduce across SIMT work-items."""
    return _simt_allreduce(value, threads=threads, scale=scale,
                           thread_offset=thread_offset, scratch=scratch, reducer="min")


__all__ = [
    "simt_allreduce_sum",
    "simt_allreduce_max",
    "simt_allreduce_min",
]
