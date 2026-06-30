# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""
SIMT cross-workitem all-reduce.

All-reduce ops are emitted **inline** at the current insertion point
(no ``func.func`` outline or ``func.call``).  Three reducer variants
are exposed: ``simt_allreduce_sum``, ``simt_allreduce_max``, ``simt_allreduce_min``.

Dispatch tree (mirrors the C++ compile-time dispatch in ``reduce.h``)::

    threads <= scale                                    → identity
    threads ≤ 32,  pow2(threads), pow2(scale)           → warp_reduce
    threads ≤ 32                                        → ub_reduce
    threads > 32, pow2(threads), scale ≤ 32, pow2(scale) → cross_warp_reduce
    otherwise                                           → ub_reduce
"""

from __future__ import annotations

from . import scalar
from ._control_flow import if_, for_
from ._ops import const as _const, get_laneid, get_tid_x, redux_add, redux_max, redux_min, shuffle_bfly, syncthreads
from ._surface_values import unwrap_surface_value, wrap_surface_value
from ._tracing.active import current_session
from ._types import (
    _resolve,
    _restore_integer_signedness,
    _strip_integer_signedness,
    float16 as _f16_dtype,
    float32 as _f32_dtype,
)

from mlir.dialects import pto as _pto
from mlir.ir import F16Type, F32Type, IntegerType, UnitAttr


# ═══════════════════════════════════════════════════════════════════════════════
# helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _as_ui32(sv):
    """Re-tag an i32 RuntimeValue as ui32 (zero-cost for non-negative values)."""
    raw = unwrap_surface_value(sv)
    return wrap_surface_value(_restore_integer_signedness(raw, IntegerType.get_unsigned(32)))


def _f32_const(value: float):
    """Emit an f32 constant, returning a RuntimeValue."""
    return _const(value, dtype=_resolve(_f32_dtype))


def _f16_const(value: float):
    """Emit an f16 constant, returning a RuntimeValue."""
    return _const(value, dtype=_resolve(_f16_dtype))


def _index_cast(value):
    """``scalar.index_cast`` that tolerates signed/unsigned integer types."""
    raw = unwrap_surface_value(value)
    raw = _strip_integer_signedness(raw)
    return scalar.index_cast(wrap_surface_value(raw))


def _is_pow2(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def _dtype_to_str(mlir_type) -> str:
    """Map an MLIR scalar type to a canonical dtype string."""
    if mlir_type == F32Type.get():
        return "f32"
    if mlir_type == F16Type.get():
        return "f16"
    raise NotImplementedError(
        f"all_reduce: unsupported dtype {mlir_type}"
    )


# ── reducer dispatch tables ──────────────────────────────────────────────────

_REDUCER_IDENTITY = {
    "sum": {"f32": 0.0, "f16": 0.0},
    "max": {"f32": float("-inf"), "f16": float("-inf")},
    "min": {"f32": float("inf"), "f16": float("inf")},
}
"""Identity element per reducer and dtype."""


_REDUCER_COMBINE = {
    "sum": lambda a, b: a + b,
    "max": scalar.max,
    "min": scalar.min,
}
"""Element-wise combine function per reducer."""


_REDUCER_REDUX = {
    "sum": redux_add,
    "max": redux_max,
    "min": redux_min,
}
"""Hardware redux op per reducer."""


# ── scratch validation ────────────────────────────────────────────────────

def _validate_scratch(scratch, expected_mlir_type, *, context: str):
    """Verify *scratch* is a ``!pto.ptr<expected_mlir_type, ub>`` buffer."""
    raw_scratch = unwrap_surface_value(scratch)
    try:
        ptr_type = _pto.PtrType(raw_scratch.type)
    except Exception:
        raise TypeError(
            f"all_reduce {context}: scratch must be a !pto.ptr buffer, "
            f"got {raw_scratch.type}"
        ) from None
    vec_attr = _pto.AddressSpaceAttr.get(_pto.AddressSpace.VEC)
    if ptr_type.memory_space != vec_attr:
        raise TypeError(
            f"all_reduce {context}: scratch must be in UB memory space, "
            f"got {ptr_type.memory_space}"
        )
    if ptr_type.element_type != expected_mlir_type:
        raise TypeError(
            f"all_reduce {context}: scratch element type mismatch: "
            f"expected {expected_mlir_type}, got {ptr_type.element_type}"
        )


# ── shared inline-emission utility ──────────────────────────────────────────

def _emit_inline(emit_fn, *args):
    """Call *emit_fn* at the current insertion point and return its result.

    Inline SIMT allreduce emits ``pto.syncthreads``, which requires the
    containing function to carry ``pto.simt_entry``.  We attach the attribute
    here (idempotently) so that callers inside ``with pto.simt():`` do not
    need to manage the attribute themselves.
    """
    result = emit_fn(*args)

    session = current_session()
    if session is not None:
        parent_func = session.current_function
        parent_func.attributes["pto.simt_entry"] = UnitAttr.get()

    return result


# ── reduction operator application ─────────────────────────────────────────

def _emit_store(buffer, offset, value):
    """Emit ``scalar.store`` via PTODSL."""
    scalar.store(value, buffer, offset)


def _emit_load(buffer, offset):
    """Emit ``scalar.load`` via PTODSL."""
    return scalar.load(buffer, offset)


def _emit_butterfly(v, *, threads: int, scale: int, reducer: str):
    """Emit unrolled butterfly shuffle reduce.

    Implements::

        cur = threads
        while cur > scale:
            x = combine(x, shfl_bfly(x, mask=cur/2))
            cur //= 2

    All loops are unrolled at emission time.  Caller must have set the
    insertion point.
    """
    combine = _REDUCER_COMBINE[reducer]
    cur = threads
    while cur > scale:
        offset = cur // 2
        mask = offset
        v = combine(v, shuffle_bfly(v, mask))
        cur //= 2
    return v


def _emit_warp_hw_reduce(x, *, threads: int,
                         lane_in_warp, c_identity, reducer: str):
    """Emit warp-level hardware reduce.

    When *threads* == 32 ("groups" == 1): a single ``pto.redux_*``.
    When *threads* < 32 ("groups" > 1): one ``pto.redux_*`` per group,
    with identity masking for lanes outside the group.

    Caller must have set the insertion point.
    """
    redux_fn = _REDUCER_REDUX[reducer]
    groups = 32 // threads

    if groups == 1:
        return redux_fn(x)

    my_group = lane_in_warp // threads  # unsigned div on ui32

    for g in range(groups):
        in_group = my_group == g
        masked = scalar.select(in_group, x, c_identity)
        reduced = redux_fn(masked)
        x = scalar.select(in_group, reduced, x)
    return x


# ═══════════════════════════════════════════════════════════════════════════════
# public API
# ═══════════════════════════════════════════════════════════════════════════════

def simt_allreduce_sum(value, *,
               threads: int,
               scale: int = 1,
               thread_offset: int = 0,
               scratch=None):
    """Cross-workitem all-reduce for SIMT VF context.

    Dispatch logic mirrors the compile-time tree in
    ``AscendAllReduce<Reducer, threads, scale, thread_offset>::run()``.

    Args:
        value: Lane-local scalar (f32 or f16).
        threads: Number of workitems.  Must satisfy ``threads % scale == 0``.
        scale: Scale factor (must divide *threads*).  Defaults to 1.
        thread_offset: Thread offset.  Defaults to 0.
        scratch: UB scratch buffer (``!pto.ptr<dtype, ub>``).  Required for
            ``cross_warp_reduce`` and ``ub_reduce`` paths.  Defaults to None.

    Returns:
        Lane-uniform scalar (same type as *value*) — the reduced sum.
    """
    return _dispatch_allreduce_helper(
        value, scratch=scratch,
        threads=threads, scale=scale, thread_offset=thread_offset,
        reducer="sum",
    )


def simt_allreduce_max(value, *,
               threads: int,
               scale: int = 1,
               thread_offset: int = 0,
               scratch=None):
    """Cross-workitem all-reduce **max** for SIMT VF context.

    Dispatch logic mirrors the compile-time tree in
    ``AscendAllReduce<MaxOp, threads, scale, thread_offset>::run()``.

    Args:
        value: Lane-local scalar (f32 or f16).
        threads: Number of workitems.  Must satisfy ``threads % scale == 0``.
        scale: Scale factor (must divide *threads*).  Defaults to 1.
        thread_offset: Thread offset.  Defaults to 0.
        scratch: UB scratch buffer (``!pto.ptr<dtype, ub>``).  Required for
            ``cross_warp_reduce`` and ``ub_reduce`` paths.  Defaults to None.

    Returns:
        Lane-uniform scalar (same type as *value*) — the element-wise maximum.
    """
    return _dispatch_allreduce_helper(
        value, scratch=scratch,
        threads=threads, scale=scale, thread_offset=thread_offset,
        reducer="max",
    )


def simt_allreduce_min(value, *,
               threads: int,
               scale: int = 1,
               thread_offset: int = 0,
               scratch=None):
    """Cross-workitem all-reduce **min** for SIMT VF context.

    Dispatch logic mirrors the compile-time tree in
    ``AscendAllReduce<MinOp, threads, scale, thread_offset>::run()``.

    Args:
        value: Lane-local scalar (f32 or f16).
        threads: Number of workitems.  Must satisfy ``threads % scale == 0``.
        scale: Scale factor (must divide *threads*).  Defaults to 1.
        thread_offset: Thread offset.  Defaults to 0.
        scratch: UB scratch buffer (``!pto.ptr<dtype, ub>``).  Required for
            ``cross_warp_reduce`` and ``ub_reduce`` paths.  Defaults to None.

    Returns:
        Lane-uniform scalar (same type as *value*) — the element-wise minimum.
    """
    return _dispatch_allreduce_helper(
        value, scratch=scratch,
        threads=threads, scale=scale, thread_offset=thread_offset,
        reducer="min",
    )


def _dispatch_allreduce_helper(value, *, scratch,
                                threads, scale, thread_offset, reducer):
    # ── parameter validation (before identity shortcut) ───────────────────
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

    # ── Path 0: identity ──────────────────────────────────────────────────
    if threads <= scale:
        return value

    # ── dtype validation ─────────────────────────────────────────────────
    raw_value = unwrap_surface_value(value)
    dtype = _dtype_to_str(raw_value.type)
    if dtype not in ("f32", "f16"):
        raise NotImplementedError(
            f"all_reduce only supports f32/f16, got {dtype}"
        )

    args = dict(dtype=dtype, threads=threads, scale=scale,
                thread_offset=thread_offset, reducer=reducer)

    # ── Path 1: warp_reduce ───────────────────────────────────────────────
    if threads <= 32 and _is_pow2(threads) and _is_pow2(scale):
        return _emit_inline(
            lambda x: _emit_warp_reduce(x, **args),
            value,
        )

    # ── All paths below require a scratch buffer ──────────────────────────
    if scratch is None:
        raise ValueError(
            f"all_reduce {reducer}/{dtype}/t{threads}/s{scale}/o{thread_offset} "
            "requires a UB scratch buffer"
        )
    _validate_scratch(
        scratch, raw_value.type,
        context=f"{reducer}/{dtype}/t{threads}/s{scale}/o{thread_offset}",
    )

    # ── Path 2: ub_reduce (threads ≤ 32, non-pow2) ──────────────────────
    if threads <= 32:
        return _emit_inline(
            lambda x, s: _emit_ub_reduce(x, s, **args),
            value, scratch,
        )

    # ── Path 3: cross_warp_reduce ────────────────────────────────────────
    if scale <= 32 and _is_pow2(threads) and _is_pow2(scale):
        return _emit_inline(
            lambda x, s: _emit_cross_warp_reduce(x, s, **args),
            value, scratch,
        )

    # ── Path 4: ub_reduce fallback (threads > 32, anything else) ─────────
    return _emit_inline(
        lambda x, s: _emit_ub_reduce(x, s, **args),
        value, scratch,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# emitter: warp_reduce  (Path 1: threads ≤ 32, pow2, pow2 scale)
# ═══════════════════════════════════════════════════════════════════════════════

def _emit_warp_reduce(x, *,
                      dtype, threads, scale, thread_offset, reducer):
    """Emit inline single-warp all-reduce at the current insertion point.

    Dispatches to:

    * ``warp_hw_reduce`` when ``extent >= 16`` and ``scale == 1``
      (fast hardware redux, with group masking for threads < 32).
    * ``butterfly`` otherwise (software shuffle via ``pto.shuffle_bfly``).
    """
    extent = threads // scale
    identity_val = _REDUCER_IDENTITY[reducer][dtype]
    const_f = _f32_const if dtype == "f32" else _f16_const

    c_identity = const_f(identity_val)

    if thread_offset:
        # lane_in_warp = (tid_x - offset) & 31
        tx = _as_ui32(get_tid_x()) - thread_offset
        lane_in_warp = tx & 31
    else:
        lane_in_warp = _as_ui32(get_laneid())

    if extent >= 16 and scale == 1:
        return _emit_warp_hw_reduce(
            x, threads=threads,
            lane_in_warp=lane_in_warp, c_identity=c_identity, reducer=reducer,
        )
    else:
        return _emit_butterfly(
            x, threads=threads, scale=scale, reducer=reducer,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# emitter: cross_warp_reduce  (Path 3: threads > 32)
# ═══════════════════════════════════════════════════════════════════════════════

def _emit_cross_warp_reduce(x, scratch, *,
                            dtype, threads, scale, thread_offset, reducer):
    """Emit inline cross-warp all-reduce at the current insertion point.

    Algorithm overview:

    1. *num_warps* subgroups of 32 lanes each do a per-warp reduce.
    2. Warp leaders (lid < scale) write → scratch[wid * scale + lid].
    3. ``pto.syncthreads``.
    4. Leader warp (lanes with ``tx < 32``) reduces the partial sums:
       - scale == 1:  ``hw_reduce`` across leader warp.
       - scale * num_warps ≤ 32:  ``butterfly<total, scale, 0>``.
       - otherwise:  manual loop over warps.
    5. Global leader (tx < scale) writes result → scratch[tx].
    6. ``pto.syncthreads`` + broadcast: each lane reads scratch[tx % scale].
    7. Extra ``pto.syncthreads`` to fence scratch reuse.
    """
    num_warps = threads // 32
    identity_val = _REDUCER_IDENTITY[reducer][dtype]
    const_f = _f32_const if dtype == "f32" else _f16_const
    combine = _REDUCER_COMBINE[reducer]
    redux_fn = _REDUCER_REDUX[reducer]

    c_identity = const_f(identity_val)

    # ── thread indexing ──────────────────────────────────────────────
    tid_x = get_tid_x()
    if thread_offset:
        tx = _as_ui32(tid_x) - thread_offset
        wid = tx // 32    # unsigned div on ui32
        lid = tx & 31
    else:
        tx = _as_ui32(tid_x)
        wid = tx // 32
        lid = _as_ui32(get_laneid())

    # ── Stage 1: per-warp reduce ─────────────────────────────────────
    if scale == 1:
        warp_val = redux_fn(x)
    else:
        warp_val = _emit_butterfly(
            x, threads=32, scale=scale, reducer=reducer,
        )

    # ── Stage 2: warp leaders write partial results ──────────────────
    is_writer = lid < scale  # unsigned cmp on ui32
    with if_(is_writer) as br:
        with br.then_:
            slot = wid * scale + lid
            slot_idx = _index_cast(slot)
            _emit_store(scratch, slot_idx, warp_val)

    # ── Stage 3: sync before reading partial results ─────────────────
    syncthreads()

    # ── Stage 4: leader warp reduces partial sums ────────────────────
    is_leader_warp = tx < 32
    with if_(is_leader_warp) as br:
        with br.then_:
            if scale == 1:
                # ── scale == 1: hw_reduce across leader warp ────────
                need_load = lid < num_warps
                with if_(need_load) as inner_br:
                    with inner_br.then_:
                        lid_idx = _index_cast(lid)
                        tmp = _emit_load(scratch, lid_idx)
                        inner_br.assign(loaded=tmp)
                    with inner_br.else_:
                        inner_br.assign(loaded=c_identity)
                loaded = inner_br.loaded
                stage4_result = redux_fn(loaded)
            elif scale * num_warps <= 32:
                # ── scale > 1, fits in one warp: butterfly ──────────
                total = scale * num_warps
                need_load = lid < total
                with if_(need_load) as inner_br:
                    with inner_br.then_:
                        lid_idx = _index_cast(lid)
                        tmp = _emit_load(scratch, lid_idx)
                        inner_br.assign(loaded=tmp)
                    with inner_br.else_:
                        inner_br.assign(loaded=c_identity)
                loaded = inner_br.loaded
                stage4_result = _emit_butterfly(
                    loaded,
                    threads=total, scale=scale, reducer=reducer,
                )
            else:
                # ── manual loop: lid < scale lanes each reduce num_warps
                is_reducer = lid < scale
                reduced = c_identity
                my_slot = lid % scale  # unsigned rem on ui32
                for w in range(num_warps):
                    idx_val = w * scale + my_slot
                    slot_idx = _index_cast(idx_val)
                    loaded_v = _emit_load(scratch, slot_idx)
                    reduced = combine(reduced, loaded_v)
                stage4_result = scalar.select(is_reducer, reduced, c_identity)

            br.assign(stage4_result=stage4_result)
        with br.else_:
            br.assign(stage4_result=c_identity)

    partial_reduced = br.stage4_result

    # ── Stage 5: global leader writes result to scratch ──────────────
    is_global_leader = tx < scale
    with if_(is_global_leader) as br5:
        with br5.then_:
            tx_idx = _index_cast(tx)
            _emit_store(scratch, tx_idx, partial_reduced)

    # ── Stage 6: sync + broadcast load scratch[tx % scale] ───────────
    syncthreads()
    my_slot = tx % scale  # unsigned rem on ui32
    load_idx = _index_cast(my_slot)
    result = _emit_load(scratch, load_idx)

    # ── Stage 7: extra sync to fence scratch reuse ───────────────────
    syncthreads()

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# emitter: ub_reduce  (Paths 2 & 4: fallback via UB scratch)
# ═══════════════════════════════════════════════════════════════════════════════

def _emit_ub_reduce(x, scratch, *,
                    dtype, threads, scale, thread_offset, reducer):
    """Emit inline UB-scratch all-reduce at the current insertion point.

    Algorithm:

    1. Each lane writes x → scratch[tx].
    2. ``pto.syncthreads``.
    3. Lanes with ``lane < scale`` sequentially reduce scratch slots.
    4. ``pto.syncthreads``.
    5. Per-class leader writes back.
    6. ``pto.syncthreads`` + broadcast: each lane reads scratch[tx % scale].
    7. ``pto.syncthreads`` to fence scratch reuse.
    """
    combine = _REDUCER_COMBINE[reducer]

    # ── thread indexing ──────────────────────────────────────────────
    tid_x = get_tid_x()
    if thread_offset:
        tx = _as_ui32(tid_x) - thread_offset
    else:
        tx = _as_ui32(tid_x)
    group = tx // threads    # unsigned div on ui32
    lane = tx % threads       # unsigned rem on ui32

    # ── Stage 1: each lane writes x → scratch[tx] ──
    tx_idx = _index_cast(tx)
    _emit_store(scratch, tx_idx, x)

    # ── Stage 2: sync ────────────────────────────────────────────────
    syncthreads()

    # ── Stage 3: reducers sequentially combine ───────────────────────
    is_reducer = lane < scale
    with if_(is_reducer) as br:
        with br.then_:
            # initial: load scratch[group * threads + lane]
            group_offset = group * threads
            first_elem = group_offset + lane
            first_idx = _index_cast(first_elem)
            acc = _emit_load(scratch, first_idx)

            # pto.for_ i = scale to threads step scale  (for_ coerce_index handles ui32)
            lb = scale
            ub = threads
            step = scale
            carry_loop = for_(lb, ub, step=step).carry(acc=acc)
            with carry_loop:
                prev = carry_loop.acc
                i = carry_loop.iv
                elem = first_idx + i
                loaded = _emit_load(scratch, elem)
                new_acc = combine(prev, loaded)
                carry_loop.update(acc=new_acc)
            acc = carry_loop.final("acc")

            br.assign(flag=acc)
        with br.else_:
            br.assign(flag=x)

    flag = br.flag

    # ── Stage 4: sync ────────────────────────────────────────────────
    syncthreads()

    # ── Stage 5: per-class leader writes reduced value ───────────────
    is_leader = lane < scale
    with if_(is_leader) as br5:
        with br5.then_:
            dst_offset = group * threads + lane
            dst_idx = _index_cast(dst_offset)
            _emit_store(scratch, dst_idx, flag)

    # ── Stage 6: sync + broadcast scratch[group*threads + tx%scale] ──
    syncthreads()
    my_slot = group * threads + (tx % scale)
    load_idx = _index_cast(my_slot)
    result = _emit_load(scratch, load_idx)

    # ── Stage 7: extra sync to fence scratch reuse ───────────────────
    syncthreads()

    return result


__all__ = [
    "simt_allreduce_sum",
    "simt_allreduce_max",
    "simt_allreduce_min",
]
