# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""
SIMT cross-workitem all-reduce helpers.

Implements ``AscendAllReduce<Reducer, threads, scale, thread_offset>::run()``
as PTO IR helper functions that are lazily emitted into the trace module.

Public entry point: ``all_reduce(x, scratch, *, op, threads, scale, thread_offset)``,
callable from within a ``@pto.simt`` context.

Dispatch tree (mirrors the C++ compile-time dispatch in ``reduce.h``)::

    threads <= scale                                    → identity
    threads ≤ 32,  pow2(threads), pow2(scale)           → warp_reduce
    threads ≤ 32                                        → ub_reduce
    threads > 32, pow2(threads), scale ≤ 32, pow2(scale) → cross_warp_reduce
    otherwise                                           → ub_reduce
"""

from __future__ import annotations

from ._surface_values import unwrap_surface_value, wrap_surface_value
from ._tracing.active import require_active_session
from ._tracing.session import HelperFunctionSpec

from mlir.dialects import arith, func, scf
from mlir.dialects import pto as _pto
from mlir.ir import F16Type, F32Type, IndexType, InsertionPoint, IntegerType, Operation, UnitAttr


# ═══════════════════════════════════════════════════════════════════════════════
# helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _is_pow2(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def _helper_name(dtype: str, threads: int, scale: int, thread_offset: int) -> str:
    """Canonical helper symbol name for a specific all-reduce instance.

    Example: ``__tl_allreduce_sum_f32_t128_s1_o0``.
    """
    return f"__tl_allreduce_sum_{dtype}_t{threads}_s{scale}_o{thread_offset}"


def _dtype_to_str(mlir_type) -> str:
    """Map an MLIR scalar type to a canonical dtype string."""
    if mlir_type == F32Type.get():
        return "f32"
    if mlir_type == F16Type.get():
        return "f16"
    raise NotImplementedError(
        f"all_reduce: unsupported dtype {mlir_type}"
    )


def _mlir_scalar_type(dtype: str):
    """Map a canonical dtype string back to an MLIR scalar type."""
    if dtype == "f32":
        return F32Type.get()
    if dtype == "f16":
        return F16Type.get()
    raise NotImplementedError(
        f"all_reduce: unsupported dtype {dtype!r}"
    )


# ── compile-time parameter tables ──────────────────────────────────────────

_IDENTITY = {
    "f32": 0.0,
    "f16": 0.0,
}
"""Identity element for sum reduction (0.0 for both f32 and f16)."""

_REDUX_OP = _pto.ReduxAddOp
"""Reduction operator (hardware redux_add)."""


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


# ── shared helper-emission utility ─────────────────────────────────────────

def _invoke_helper(helper_name, emit_fn, *surface_args):
    """Look up or lazily create *helper_name*, then ``func.call`` it.

    *emit_fn(helper_fn)* is called exactly once per trace session — on the
    first invocation for this *helper_name*.
    """
    session = require_active_session("simt_allreduce_sum")
    raw_args = [unwrap_surface_value(a) for a in surface_args]
    arg_types = tuple(a.type for a in raw_args)

    helper_spec = HelperFunctionSpec(
        symbol_name=helper_name,
        arg_types=arg_types,
        result_types=(arg_types[0],),
        attributes=(("pto.simt_entry", UnitAttr.get()),),
    )
    helper_fn, created = session.get_or_create_helper_function(helper_spec)
    if created:
        emit_fn(helper_fn)
    call = func.CallOp(helper_fn, raw_args)
    return wrap_surface_value(call.result)


# ── reduction operator application ─────────────────────────────────────────

def _emit_store(buffer, offset, value):
    """Emit ``pto.store`` — accepts Ptr and any MemRef (including UB/VEC).

    Unlike ``pto.store_scalar`` (which rejects VEC memrefs), ``pto.store``
    uses ``PTO_BufferLikeType`` and survives the Ptr→MemRef type conversion
    pass during lowering.
    """
    Operation.create(
        "pto.store",
        operands=[buffer, offset, value],
    )


def _emit_load(result_type, buffer, offset):
    """Emit ``pto.load`` — accepts Ptr and any MemRef (including UB/VEC).

    Counterpart to ``_emit_store``.  Returns the loaded SSA value.
    """
    return Operation.create(
        "pto.load",
        results=[result_type],
        operands=[buffer, offset],
    ).results[0]


def _apply_sum(a, b):
    """Emit ``a = a + b`` (float addition)."""
    return arith.AddFOp(a, b).result


def _emit_butterfly(v, *, threads: int, scale: int):
    """Emit unrolled butterfly shuffle reduce.

    Implements::

        cur = threads
        while cur > scale:
            x = op(x, shfl_xor(x, cur/2))
            cur /= 2

    All loops are unrolled at emission time.  Caller must have set the
    insertion point.
    """
    i32 = IntegerType.get_signless(32)
    cur = threads
    while cur > scale:
        offset = cur // 2
        c_offset = arith.ConstantOp(i32, offset).result
        shfl = _pto.ShuffleBflyOp(v, c_offset).result
        v = _apply_sum(v, shfl)
        cur //= 2
    return v


def _emit_warp_hw_reduce(x, *, threads: int,
                         lane_in_warp, c_identity, i32):
    """Emit warp-level hardware reduce.

    When *threads* == 32 ("groups" == 1): a single ``pto.redux_*``.
    When *threads* < 32 ("groups" > 1): one ``pto.redux_*`` per group,
    with identity masking for lanes outside the group.

    Caller must have set the insertion point.
    """
    groups = 32 // threads

    if groups == 1:
        return _REDUX_OP(x).result

    c_threads = arith.ConstantOp(i32, threads).result
    my_group = arith.DivUIOp(lane_in_warp, c_threads).result

    for g in range(groups):
        c_g = arith.ConstantOp(i32, g).result
        in_group = arith.CmpIOp(arith.CmpIPredicate.eq, my_group, c_g).result
        masked = arith.SelectOp(in_group, x, c_identity).result
        reduced = _REDUX_OP(masked).result
        x = arith.SelectOp(in_group, reduced, x).result
    return x


# ═══════════════════════════════════════════════════════════════════════════════
# public API
# ═══════════════════════════════════════════════════════════════════════════════

def simt_allreduce_sum(value, *,
               threads: int,
               scale: int = 1,
               thread_offset: int = 0,
               scratch=None,
               scratch_offset: int = 0):
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
        scratch_offset: Element offset into *scratch*.  Defaults to 0.

    Returns:
        Lane-uniform scalar (same type as *value*) — the reduced sum.
    """
    return _dispatch_allreduce_helper(
        value, scratch=scratch, scratch_offset=scratch_offset,
        threads=threads, scale=scale, thread_offset=thread_offset,
    )


def _dispatch_allreduce_helper(value, *, scratch, scratch_offset,
                                threads, scale, thread_offset):
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

    name = _helper_name(dtype, threads, scale, thread_offset)
    args = dict(dtype=dtype, threads=threads, scale=scale,
                thread_offset=thread_offset, scratch_offset=scratch_offset)

    # ── Path 1: warp_reduce ───────────────────────────────────────────────
    if threads <= 32 and _is_pow2(threads) and _is_pow2(scale):
        return _invoke_helper(
            name,
            lambda hf: _emit_warp_reduce(hf, **args),
            value,
        )

    # ── All paths below require a scratch buffer ──────────────────────────
    if scratch is None:
        raise ValueError(
            f"all_reduce sum/{dtype}/t{threads}/s{scale}/o{thread_offset} "
            "requires a UB scratch buffer"
        )
    _validate_scratch(
        scratch, raw_value.type,
        context=f"sum/{dtype}/t{threads}/s{scale}/o{thread_offset}",
    )

    # ── Path 2: ub_reduce (threads ≤ 32, non-pow2) ──────────────────────
    if threads <= 32:
        return _invoke_helper(
            name,
            lambda hf: _emit_ub_reduce(hf, **args),
            value, scratch,
        )

    # ── Path 3: cross_warp_reduce ────────────────────────────────────────
    if scale <= 32 and _is_pow2(threads) and _is_pow2(scale):
        return _invoke_helper(
            name,
            lambda hf: _emit_cross_warp_reduce(hf, **args),
            value, scratch,
        )

    # ── Path 4: ub_reduce fallback (threads > 32, anything else) ─────────
    return _invoke_helper(
        name,
        lambda hf: _emit_ub_reduce(hf, **args),
        value, scratch,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# emitter: warp_reduce  (Path 1: threads ≤ 32, pow2, pow2 scale)
# ═══════════════════════════════════════════════════════════════════════════════

def _emit_warp_reduce(helper_fn, *,
                      dtype, threads, scale, thread_offset,
                      scratch_offset):
    """Build the body of a single-warp all-reduce helper.

    Dispatches to:

    * ``warp_hw_reduce`` when ``extent >= 16`` and ``scale == 1``
      (fast hardware redux, with group masking for threads < 32).
    * ``butterfly`` otherwise (software shuffle via ``pto.shuffle_bfly``).
    """
    extent = threads // scale
    scalar_t = _mlir_scalar_type(dtype)
    identity_val = _IDENTITY[dtype]
    i32 = IntegerType.get_signless(32)

    entry = helper_fn.add_entry_block()
    with InsertionPoint(entry):
        x = entry.arguments[0]

        c_offset = arith.ConstantOp(i32, thread_offset).result
        c_identity = arith.ConstantOp(scalar_t, identity_val).result

        if thread_offset:
            # lane_in_warp = (tid_x - offset) & 31
            tid_x = _pto.GetTidXOp().result
            tx = arith.SubIOp(tid_x, c_offset).result
            lane_in_warp = arith.AndIOp(tx, arith.ConstantOp(i32, 31).result).result
        else:
            lane_in_warp = _pto.GetLaneIdOp().result

        if extent >= 16 and scale == 1:
            result = _emit_warp_hw_reduce(
                x, threads=threads,
                lane_in_warp=lane_in_warp, c_identity=c_identity, i32=i32,
            )
        else:
            result = _emit_butterfly(
                x, threads=threads, scale=scale,
            )

        func.ReturnOp([result])


# ═══════════════════════════════════════════════════════════════════════════════
# emitter: cross_warp_reduce  (Path 3: threads > 32)
# ═══════════════════════════════════════════════════════════════════════════════

def _emit_cross_warp_reduce(helper_fn, *,
                            dtype, threads, scale, thread_offset,
                            scratch_offset):
    """Build the body of a cross-warp all-reduce helper.

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
    scalar_t = _mlir_scalar_type(dtype)
    identity_val = _IDENTITY[dtype]

    i32 = IntegerType.get_signless(32)
    idx_t = IndexType.get()

    entry = helper_fn.add_entry_block()
    with InsertionPoint(entry):
        x = entry.arguments[0]
        scratch = entry.arguments[1]

        # ── constants ────────────────────────────────────────────────────
        c0_i32 = arith.ConstantOp(i32, 0).result
        c5_i32 = arith.ConstantOp(i32, 5).result
        c31_i32 = arith.ConstantOp(i32, 31).result
        c32_i32 = arith.ConstantOp(i32, 32).result
        c_scale = arith.ConstantOp(i32, scale).result
        c_num_warps = arith.ConstantOp(i32, num_warps).result
        c_offset = arith.ConstantOp(i32, thread_offset).result
        c_scratch_off = arith.ConstantOp(idx_t, scratch_offset).result
        c_identity = arith.ConstantOp(scalar_t, identity_val).result

        # ── thread indexing ──────────────────────────────────────────────
        tid_x = _pto.GetTidXOp().result
        if thread_offset:
            tx = arith.SubIOp(tid_x, c_offset).result
            wid = arith.ShRUIOp(tx, c5_i32).result
            lid = arith.AndIOp(tx, c31_i32).result
        else:
            tx = tid_x
            wid = arith.ShRUIOp(tx, c5_i32).result
            lid = _pto.GetLaneIdOp().result

        # ── Stage 1: per-warp reduce ─────────────────────────────────────
        if scale == 1:
            warp_val = _REDUX_OP(x).result
        else:
            warp_val = _emit_butterfly(
                x, threads=32, scale=scale,
            )

        # ── Stage 2: warp leaders write partial results ──────────────────
        is_writer = arith.CmpIOp(arith.CmpIPredicate.ult, lid, c_scale).result
        write_if = scf.IfOp(is_writer, hasElse=False)
        with InsertionPoint(write_if.then_block):
            slot = arith.AddIOp(
                arith.MulIOp(wid, c_scale).result, lid).result
            slot_idx = arith.IndexCastOp(idx_t, slot).result
            if scratch_offset:
                slot_idx = arith.AddIOp(slot_idx, c_scratch_off).result
            _emit_store(scratch, slot_idx, warp_val)
            scf.YieldOp([])

        # ── Stage 3: sync before reading partial results ─────────────────
        _pto.SyncthreadsOp()

        # ── Stage 4: leader warp reduces partial sums ────────────────────
        is_leader_warp = arith.CmpIOp(
            arith.CmpIPredicate.ult, tx, c32_i32).result
        outer_if = scf.IfOp(is_leader_warp, [scalar_t], hasElse=True)

        with InsertionPoint(outer_if.then_block):
            if scale == 1:
                # ── scale == 1: hw_reduce across leader warp ────────────
                need_load = arith.CmpIOp(
                    arith.CmpIPredicate.ult, lid, c_num_warps).result
                inner_if = scf.IfOp(need_load, [scalar_t], hasElse=True)
                with InsertionPoint(inner_if.then_block):
                    lid_idx = arith.IndexCastOp(idx_t, lid).result
                    tmp = _emit_load(scalar_t, scratch, lid_idx)
                    scf.YieldOp([tmp])
                with InsertionPoint(inner_if.else_block):
                    scf.YieldOp([c_identity])
                loaded = inner_if.results[0]
                stage4_result = _REDUX_OP(loaded).result
            elif scale * num_warps <= 32:
                # ── scale > 1, fits in one warp: butterfly ──────────────
                total = scale * num_warps
                c_total = arith.ConstantOp(i32, total).result
                need_load = arith.CmpIOp(
                    arith.CmpIPredicate.ult, lid, c_total).result
                inner_if = scf.IfOp(need_load, [scalar_t], hasElse=True)
                with InsertionPoint(inner_if.then_block):
                    lid_idx = arith.IndexCastOp(idx_t, lid).result
                    if scratch_offset:
                        lid_idx = arith.AddIOp(lid_idx, c_scratch_off).result
                    tmp = _emit_load(scalar_t, scratch, lid_idx)
                    scf.YieldOp([tmp])
                with InsertionPoint(inner_if.else_block):
                    scf.YieldOp([c_identity])
                loaded = inner_if.results[0]
                stage4_result = _emit_butterfly(
                    loaded,
                    threads=total, scale=scale,
                )
            else:
                # ── manual loop: lid < scale lanes each reduce num_warps
                is_reducer = arith.CmpIOp(
                    arith.CmpIPredicate.ult, lid, c_scale).result
                result = c_identity
                my_slot = arith.RemUIOp(lid, c_scale).result
                for w in range(num_warps):
                    c_w = arith.ConstantOp(i32, w).result
                    idx_val = arith.AddIOp(
                        arith.MulIOp(c_w, c_scale).result, my_slot).result
                    slot_idx = arith.IndexCastOp(idx_t, idx_val).result
                    if scratch_offset:
                        slot_idx = arith.AddIOp(slot_idx, c_scratch_off).result
                    loaded_v = _emit_load(
                        scalar_t, scratch, slot_idx)
                    result = _apply_sum(result, loaded_v)
                stage4_result = arith.SelectOp(
                    is_reducer, result, c_identity).result

            scf.YieldOp([stage4_result])

        with InsertionPoint(outer_if.else_block):
            scf.YieldOp([c_identity])

        partial_reduced = outer_if.results[0]

        # ── Stage 5: global leader writes result to scratch ──────────────
        is_global_leader = arith.CmpIOp(
            arith.CmpIPredicate.ult, tx, c_scale).result
        write_result_if = scf.IfOp(is_global_leader, hasElse=False)
        with InsertionPoint(write_result_if.then_block):
            tx_idx = arith.IndexCastOp(idx_t, tx).result
            if scratch_offset:
                tx_idx = arith.AddIOp(tx_idx, c_scratch_off).result
            _emit_store(scratch, tx_idx, partial_reduced)
            scf.YieldOp([])

        # ── Stage 6: sync + broadcast load scratch[tx % scale] ───────────
        _pto.SyncthreadsOp()
        my_slot = arith.RemUIOp(tx, c_scale).result
        load_idx = arith.IndexCastOp(idx_t, my_slot).result
        if scratch_offset:
            load_idx = arith.AddIOp(load_idx, c_scratch_off).result
        result = _emit_load(scalar_t, scratch, load_idx)

        # ── Stage 7: extra sync to fence scratch reuse ───────────────────
        _pto.SyncthreadsOp()

        func.ReturnOp([result])


# ═══════════════════════════════════════════════════════════════════════════════
# emitter: ub_reduce  (Paths 2 & 4: fallback via UB scratch)
# ═══════════════════════════════════════════════════════════════════════════════

def _emit_ub_reduce(helper_fn, *,
                    dtype, threads, scale, thread_offset,
                    scratch_offset):
    """Build the body of a UB-scratch all-reduce helper.

    Algorithm:

    1. Each lane writes x → scratch[tx].
    2. ``pto.syncthreads``.
    3. Lanes with ``lane % scale == 0`` sequentially reduce scratch slots.
    4. ``pto.syncthreads``.
    5. Global leader (lane % scale == 0, lane / scale == 0) writes back.
    6. ``pto.syncthreads`` + broadcast: each lane reads scratch[tx % scale].
    7. ``pto.syncthreads`` to fence scratch reuse.
    """
    scalar_t = _mlir_scalar_type(dtype)
    i32 = IntegerType.get_signless(32)
    idx_t = IndexType.get()

    entry = helper_fn.add_entry_block()
    with InsertionPoint(entry):
        x = entry.arguments[0]
        scratch = entry.arguments[1]

        # ── constants ────────────────────────────────────────────────────
        c0_i32 = arith.ConstantOp(i32, 0).result
        c_threads = arith.ConstantOp(i32, threads).result
        c_scale = arith.ConstantOp(i32, scale).result
        c_offset = arith.ConstantOp(i32, thread_offset).result
        c_scratch_off = arith.ConstantOp(idx_t, scratch_offset).result

        # ── thread indexing ──────────────────────────────────────────────
        tid_x = _pto.GetTidXOp().result
        tx = arith.SubIOp(tid_x, c_offset).result if thread_offset else tid_x
        group = arith.DivUIOp(tx, c_threads).result
        lane = arith.RemUIOp(tx, c_threads).result
        lane_mod = arith.RemUIOp(lane, c_scale).result

        # ── Stage 1: each lane writes x → scratch[scratch_offset + tx] ──
        tx_idx = arith.IndexCastOp(idx_t, tx).result
        if scratch_offset:
            tx_idx = arith.AddIOp(tx_idx, c_scratch_off).result
        _emit_store(scratch, tx_idx, x)

        # ── Stage 2: sync ────────────────────────────────────────────────
        _pto.SyncthreadsOp()

        # ── Stage 3: reducers sequentially combine ───────────────────────
        # lane < scale gives exactly one reducer per residue class
        is_reducer = arith.CmpIOp(
            arith.CmpIPredicate.ult, lane, c_scale).result
        reduce_if = scf.IfOp(is_reducer, [scalar_t], hasElse=True)

        with InsertionPoint(reduce_if.then_block):
            # initial: load scratch[scratch_offset + group * threads + lane]
            group_offset = arith.MulIOp(group, c_threads).result
            first_elem = arith.AddIOp(group_offset, lane).result
            first_idx = arith.IndexCastOp(idx_t, first_elem).result
            if scratch_offset:
                first_idx = arith.AddIOp(first_idx, c_scratch_off).result
            acc = _emit_load(scalar_t, scratch, first_idx)

            # scf.for i = scale to threads step scale
            lb = arith.ConstantOp(idx_t, scale).result
            ub = arith.ConstantOp(idx_t, threads).result
            step = arith.ConstantOp(idx_t, scale).result
            for_op = scf.ForOp(lb, ub, step, [acc])
            with InsertionPoint(for_op.body):
                i = for_op.induction_variable
                prev = for_op.inner_iter_args[0]
                elem = arith.AddIOp(first_idx, i).result
                loaded = _emit_load(
                    scalar_t, scratch, elem)
                new_acc = _apply_sum(prev, loaded)
                scf.YieldOp([new_acc])
            scf.YieldOp([for_op.results[0]])

        with InsertionPoint(reduce_if.else_block):
            scf.YieldOp([x])

        flag = reduce_if.results[0]

        # ── Stage 4: sync ────────────────────────────────────────────────
        _pto.SyncthreadsOp()

        # ── Stage 5: per-class leader writes reduced value ───────────────
        # leader lanes 0..scale-1 each write their residue class result
        is_leader = arith.CmpIOp(
            arith.CmpIPredicate.ult, lane, c_scale).result
        write_if = scf.IfOp(is_leader, hasElse=False)
        with InsertionPoint(write_if.then_block):
            dst_offset = arith.AddIOp(
                arith.MulIOp(group, c_threads).result, lane).result
            dst_idx = arith.IndexCastOp(idx_t, dst_offset).result
            if scratch_offset:
                dst_idx = arith.AddIOp(dst_idx, c_scratch_off).result
            _emit_store(scratch, dst_idx, flag)
            scf.YieldOp([])

        # ── Stage 6: sync + broadcast scratch[scratch_offset + group*threads + tx%scale] ──
        _pto.SyncthreadsOp()
        my_slot = arith.AddIOp(
            arith.MulIOp(group, c_threads).result,
            arith.RemUIOp(tx, c_scale).result).result
        load_idx = arith.IndexCastOp(idx_t, my_slot).result
        if scratch_offset:
            load_idx = arith.AddIOp(load_idx, c_scratch_off).result
        result = _emit_load(scalar_t, scratch, load_idx)

        # ── Stage 7: extra sync to fence scratch reuse ───────────────────
        _pto.SyncthreadsOp()

        func.ReturnOp([result])


__all__ = [
    "simt_allreduce_sum",
]
