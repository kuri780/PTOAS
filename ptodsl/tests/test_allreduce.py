#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "ptodsl"))

from ptodsl import pto


def expect(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main():
    from ptodsl._allreduce import _helper_name, simt_allreduce_sum

    # ══════════════════════════════════════════════════════════════════════════
    # helper name format
    # ══════════════════════════════════════════════════════════════════════════
    expect(
        _helper_name("f32", 128, 1, 0) == "__tl_allreduce_sum_f32_t128_s1_o0",
        "helper name format (sum/f32/t128/s1/o0)",
    )
    expect(
        _helper_name("f16", 32, 2, 4) == "__tl_allreduce_sum_f16_t32_s2_o4",
        "helper name format (f16/t32/s2/o4)",
    )

    # ══════════════════════════════════════════════════════════════════════════
    # Path 0: identity (threads <= scale)
    # ══════════════════════════════════════════════════════════════════════════
    expect(
        simt_allreduce_sum(1.0, threads=1, scale=1) == 1.0,
        "identity: threads == scale",
    )
    expect(
        simt_allreduce_sum(1.0, threads=2, scale=2) == 1.0,
        "identity: threads == scale (alt)",
    )

    # ══════════════════════════════════════════════════════════════════════════
    # validation errors
    # ══════════════════════════════════════════════════════════════════════════

    # threads % scale != 0  (validation now runs before identity shortcut)
    try:
        simt_allreduce_sum(1.0, threads=3, scale=2)
        raise AssertionError("expected ValueError for threads % scale != 0")
    except ValueError:
        pass


    # threads < 1
    try:
        simt_allreduce_sum(1.0, threads=0, scale=1)
        raise AssertionError("expected ValueError for threads < 1")
    except ValueError:
        pass

    # validation runs before identity: bad params not bypassed by threads<=scale
    try:
        simt_allreduce_sum(1.0, threads=1, scale=2)
        raise AssertionError("expected ValueError for threads%scale!=0 (before identity)")
    except ValueError:
        pass

    # i32 dtype rejected — need a real JIT kernel so we get an MLIR i32 value
    @pto.jit(target="a5")
    def kernel_i32(scratch_gm: pto.ptr(pto.f32, "gm")):
        with pto.simt():
            x = pto.const(1, dtype=pto.i32)
            _result = pto.simt_allreduce_sum(x, threads=32, scale=1)

    try:
        kernel_i32.compile()
        raise AssertionError("expected NotImplementedError for i32")
    except NotImplementedError:
        pass

    # ══════════════════════════════════════════════════════════════════════════
    # Path 1a: warp_reduce — hardware redux, groups == 1 (threads=32)
    # ══════════════════════════════════════════════════════════════════════════

    @pto.jit(target="a5")
    def kernel_warp(scratch_gm: pto.ptr(pto.f32, "gm")):
        zero_u64 = pto.const(0, dtype=pto.ui64)
        _ub_scratch = pto.castptr(zero_u64, pto.ptr(pto.f32, "ub"))
        with pto.simt():
            x = pto.const(1.0, dtype=pto.f32)
            _result = pto.simt_allreduce_sum(x, threads=32, scale=1)

    compiled_warp = kernel_warp.compile()
    mlir_warp = compiled_warp.mlir_text()
    expect("func.func @__tl_allreduce_sum_f32_t32_s1_o0" in mlir_warp,
           "IR: warp_reduce helper name")
    expect("pto.redux_add" in mlir_warp,
           "IR: redux_add in warp_reduce helper")
    expect("pto.syncthreads" not in mlir_warp,
           "IR: warp_reduce has no syncthreads")
    expect("pto.shuffle_bfly" not in mlir_warp,
           "IR: warp_reduce (groups=1) has no shuffle_bfly")
    compiled_warp.verify()

    # ══════════════════════════════════════════════════════════════════════════
    # Path 1b: warp_reduce — hardware redux, groups > 1 (threads=16, scale=1)
    # ══════════════════════════════════════════════════════════════════════════

    @pto.jit(target="a5")
    def kernel_warp_t16(scratch_gm: pto.ptr(pto.f32, "gm")):
        zero_u64 = pto.const(0, dtype=pto.ui64)
        _ub_scratch = pto.castptr(zero_u64, pto.ptr(pto.f32, "ub"))
        with pto.simt():
            x = pto.const(1.0, dtype=pto.f32)
            _result = pto.simt_allreduce_sum(x, threads=16, scale=1)

    compiled_warp_t16 = kernel_warp_t16.compile()
    mlir_warp_t16 = compiled_warp_t16.mlir_text()
    expect("func.func @__tl_allreduce_sum_f32_t16_s1_o0" in mlir_warp_t16,
           "IR: warp_reduce t=16 helper name")
    expect("pto.redux_add" in mlir_warp_t16,
           "IR: redux_add for groups>1")
    expect("arith.select" in mlir_warp_t16,
           "IR: arith.select for group masking")
    expect("pto.syncthreads" not in mlir_warp_t16,
           "IR: warp_reduce (groups=2) has no syncthreads")
    compiled_warp_t16.verify()

    # ══════════════════════════════════════════════════════════════════════════
    # Path 1c: warp_reduce — butterfly shuffle (threads=8, scale=1)
    # ══════════════════════════════════════════════════════════════════════════

    @pto.jit(target="a5")
    def kernel_warp_t8(scratch_gm: pto.ptr(pto.f32, "gm")):
        zero_u64 = pto.const(0, dtype=pto.ui64)
        _ub_scratch = pto.castptr(zero_u64, pto.ptr(pto.f32, "ub"))
        with pto.simt():
            x = pto.const(1.0, dtype=pto.f32)
            _result = pto.simt_allreduce_sum(x, threads=8, scale=1)

    compiled_warp_t8 = kernel_warp_t8.compile()
    mlir_warp_t8 = compiled_warp_t8.mlir_text()
    expect("func.func @__tl_allreduce_sum_f32_t8_s1_o0" in mlir_warp_t8,
           "IR: warp_reduce t=8 butterfly helper name (sum)")
    expect("pto.shuffle_bfly" in mlir_warp_t8,
           "IR: shuffle_bfly for butterfly path")
    expect("pto.redux_add" not in mlir_warp_t8,
           "IR: butterfly has no hardware redux")
    expect("pto.syncthreads" not in mlir_warp_t8,
           "IR: butterfly has no syncthreads")
    compiled_warp_t8.verify()

    # ══════════════════════════════════════════════════════════════════════════
    # Path 1d: warp_reduce — butterfly with scale > 1 (threads=32, scale=2)
    # ══════════════════════════════════════════════════════════════════════════

    @pto.jit(target="a5")
    def kernel_warp_s2(scratch_gm: pto.ptr(pto.f32, "gm")):
        zero_u64 = pto.const(0, dtype=pto.ui64)
        _ub_scratch = pto.castptr(zero_u64, pto.ptr(pto.f32, "ub"))
        with pto.simt():
            x = pto.const(1.0, dtype=pto.f32)
            _result = pto.simt_allreduce_sum(x, threads=32, scale=2)

    compiled_warp_s2 = kernel_warp_s2.compile()
    mlir_warp_s2 = compiled_warp_s2.mlir_text()
    expect("func.func @__tl_allreduce_sum_f32_t32_s2_o0" in mlir_warp_s2,
           "IR: warp_reduce s=2 butterfly helper name (sum)")
    expect("pto.shuffle_bfly" in mlir_warp_s2,
           "IR: shuffle_bfly for butterfly (scale>1)")
    expect("pto.redux_add" not in mlir_warp_s2,
           "IR: butterfly (scale>1) has no hardware redux")
    compiled_warp_s2.verify()

    # ── warp_reduce: sum, f32, t=16, s=1, o=4 (non-zero thread_offset) ────────
    @pto.jit(target="a5")
    def kernel_warp_o4(scratch_gm: pto.ptr(pto.f32, "gm")):
        zero_u64 = pto.const(0, dtype=pto.ui64)
        _ub_scratch = pto.castptr(zero_u64, pto.ptr(pto.f32, "ub"))
        with pto.simt():
            x = pto.const(1.0, dtype=pto.f32)
            _result = pto.simt_allreduce_sum(x, threads=16, scale=1, thread_offset=4)

    compiled_warp_o4 = kernel_warp_o4.compile()
    mlir_warp_o4 = compiled_warp_o4.mlir_text()
    expect("func.func @__tl_allreduce_sum_f32_t16_s1_o4" in mlir_warp_o4,
           "IR: warp_reduce o=4 helper name")
    expect("pto.get_tid_x" in mlir_warp_o4,
           "IR: warp_reduce o=4 uses get_tid_x (not raw get_laneid)")
    expect("arith.subi" in mlir_warp_o4,
           "IR: warp_reduce o=4 uses subi for tx = tid_x - offset")
    expect("arith.andi" in mlir_warp_o4,
           "IR: warp_reduce o=4 uses andi to extract lane_in_warp")
    compiled_warp_o4.verify()

    # ══════════════════════════════════════════════════════════════════════════
    # Path 2: ub_reduce — threads ≤ 32, non-power-of-2 (threads=6, scale=1)
    # ══════════════════════════════════════════════════════════════════════════

    @pto.jit(target="a5")
    def kernel_ub6(scratch_gm: pto.ptr(pto.f32, "gm")):
        zero_u64 = pto.const(0, dtype=pto.ui64)
        ub_scratch = pto.castptr(zero_u64, pto.ptr(pto.f32, "ub"))
        with pto.simt():
            x = pto.const(1.0, dtype=pto.f32)
            _result = pto.simt_allreduce_sum(x, scratch=ub_scratch, threads=6, scale=1)

    compiled_ub6 = kernel_ub6.compile()
    mlir_ub6 = compiled_ub6.mlir_text()
    expect("func.func @__tl_allreduce_sum_f32_t6_s1_o0" in mlir_ub6,
           "IR: ub_reduce t=6 helper name")
    expect("pto.syncthreads" in mlir_ub6,
           "IR: ub_reduce has syncthreads")
    expect("pto.store" in mlir_ub6,
           "IR: ub_reduce has store (write to scratch)")
    expect("pto.load" in mlir_ub6,
           "IR: ub_reduce has load (read from scratch)")
    syncthreads_count = mlir_ub6.count("pto.syncthreads")
    expect(syncthreads_count == 4,
           f"IR: ub_reduce has 4 syncthreads, got {syncthreads_count}")
    compiled_ub6.verify()

    # ── ub_reduce: sum, f32, t=6, s=2 (scale > 1, non-pow2 threads) ─────────
    @pto.jit(target="a5")
    def kernel_ub6s2(scratch_gm: pto.ptr(pto.f32, "gm")):
        zero_u64 = pto.const(0, dtype=pto.ui64)
        ub_scratch = pto.castptr(zero_u64, pto.ptr(pto.f32, "ub"))
        with pto.simt():
            x = pto.const(1.0, dtype=pto.f32)
            _result = pto.simt_allreduce_sum(x, scratch=ub_scratch, threads=6, scale=2)

    compiled_ub6s2 = kernel_ub6s2.compile()
    mlir_ub6s2 = compiled_ub6s2.mlir_text()
    expect("func.func @__tl_allreduce_sum_f32_t6_s2_o0" in mlir_ub6s2,
           "IR: ub_reduce t=6 s=2 helper name")
    expect("pto.syncthreads" in mlir_ub6s2,
           "IR: ub_reduce t=6 s=2 has syncthreads")
    expect("pto.store" in mlir_ub6s2,
           "IR: ub_reduce t=6 s=2 has store")
    expect("pto.load" in mlir_ub6s2,
           "IR: ub_reduce t=6 s=2 has load")
    expect("scf.for" in mlir_ub6s2,
           "IR: ub_reduce t=6 s=2 has scf.for (sequential reduce loop)")
    expect("pto.redux_add" not in mlir_ub6s2,
           "IR: ub_reduce t=6 s=2 has no hardware redux")
    expect("pto.shuffle_bfly" not in mlir_ub6s2,
           "IR: ub_reduce t=6 s=2 has no butterfly shuffle")
    # scale>1 fixes: reducer uses lane < scale (ult), not lane_mod == 0
    expect("arith.cmpi ult" in mlir_ub6s2,
           "IR: ub_reduce t=6 s=2 reducer uses ult (lane < scale)")
    compiled_ub6s2.verify()

    # ── ub_reduce: sum, f32, t=6, s=1, o=4 (non-zero thread_offset) ─────────
    @pto.jit(target="a5")
    def kernel_ub_o4(scratch_gm: pto.ptr(pto.f32, "gm")):
        zero_u64 = pto.const(0, dtype=pto.ui64)
        ub_scratch = pto.castptr(zero_u64, pto.ptr(pto.f32, "ub"))
        with pto.simt():
            x = pto.const(1.0, dtype=pto.f32)
            _result = pto.simt_allreduce_sum(x, scratch=ub_scratch, threads=6, scale=1,
                                     thread_offset=4)

    compiled_ub_o4 = kernel_ub_o4.compile()
    mlir_ub_o4 = compiled_ub_o4.mlir_text()
    expect("func.func @__tl_allreduce_sum_f32_t6_s1_o4" in mlir_ub_o4,
           "IR: ub_reduce o=4 helper name")
    expect("arith.subi" in mlir_ub_o4,
           "IR: ub_reduce o=4 uses subi for tx = tid_x - offset")
    compiled_ub_o4.verify()

    # ══════════════════════════════════════════════════════════════════════════
    # Path 3a: cross_warp_reduce — sum, f32, t=128, s=1, o=0 (baseline)
    # ══════════════════════════════════════════════════════════════════════════

    @pto.jit(target="a5")
    def kernel_128(scratch_gm: pto.ptr(pto.f32, "gm")):
        zero_u64 = pto.const(0, dtype=pto.ui64)
        ub_scratch = pto.castptr(zero_u64, pto.ptr(pto.f32, "ub"))
        with pto.simt():
            x = pto.const(1.0, dtype=pto.f32)
            _result = pto.simt_allreduce_sum(x, scratch=ub_scratch, threads=128, scale=1)

    compiled = kernel_128.compile()
    mlir = compiled.mlir_text()

    expect("func.func @__tl_allreduce_sum_f32_t128_s1_o0" in mlir,
           "IR: helper function definition")
    expect("pto.simt_entry" in mlir,
           "IR: helper carries pto.simt_entry")
    expect("call @__tl_allreduce_sum_f32_t128_s1_o0" in mlir,
           "IR: func.call to helper")

    for op_name in (
        "pto.redux_add", "pto.syncthreads", "pto.store", "pto.load",
        "pto.get_tid_x", "pto.get_laneid", "arith.shrui", "scf.if",
    ):
        expect(op_name in mlir, f"IR: expected '{op_name}' in helper body")

    syncthreads_count = mlir.count("pto.syncthreads")
    expect(syncthreads_count == 3,
           f"IR: expected 3 syncthreads, got {syncthreads_count}")

    compiled.verify()

    # ── cross_warp: sum, f32, t=64 (2 warps) ────────────────────────────────
    @pto.jit(target="a5")
    def kernel_64(scratch_gm: pto.ptr(pto.f32, "gm")):
        zero_u64 = pto.const(0, dtype=pto.ui64)
        ub_scratch = pto.castptr(zero_u64, pto.ptr(pto.f32, "ub"))
        with pto.simt():
            x = pto.const(1.0, dtype=pto.f32)
            _result = pto.simt_allreduce_sum(x, scratch=ub_scratch, threads=64, scale=1)

    compiled_64 = kernel_64.compile()
    mlir_64 = compiled_64.mlir_text()
    expect("func.func @__tl_allreduce_sum_f32_t64_s1_o0" in mlir_64,
           "IR: helper for t=64")
    compiled_64.verify()

    # ── cross_warp: sum, f32, t=256 (8 warps) ───────────────────────────────
    @pto.jit(target="a5")
    def kernel_256(scratch_gm: pto.ptr(pto.f32, "gm")):
        zero_u64 = pto.const(0, dtype=pto.ui64)
        ub_scratch = pto.castptr(zero_u64, pto.ptr(pto.f32, "ub"))
        with pto.simt():
            x = pto.const(1.0, dtype=pto.f32)
            _result = pto.simt_allreduce_sum(x, scratch=ub_scratch, threads=256, scale=1)

    compiled_256 = kernel_256.compile()
    mlir_256 = compiled_256.mlir_text()
    expect("func.func @__tl_allreduce_sum_f32_t256_s1_o0" in mlir_256,
           "IR: helper for t=256")
    compiled_256.verify()

    # ══════════════════════════════════════════════════════════════════════════
    # Path 3b: cross_warp_reduce — scale > 1, scale*num_warps ≤ 32
    #           (threads=128, scale=2, num_warps=4, total=8 ≤ 32)
    # ══════════════════════════════════════════════════════════════════════════

    @pto.jit(target="a5")
    def kernel_cw_s2(scratch_gm: pto.ptr(pto.f32, "gm")):
        zero_u64 = pto.const(0, dtype=pto.ui64)
        ub_scratch = pto.castptr(zero_u64, pto.ptr(pto.f32, "ub"))
        with pto.simt():
            x = pto.const(1.0, dtype=pto.f32)
            _result = pto.simt_allreduce_sum(x, scratch=ub_scratch, threads=128, scale=2)

    compiled_cw_s2 = kernel_cw_s2.compile()
    mlir_cw_s2 = compiled_cw_s2.mlir_text()
    expect("func.func @__tl_allreduce_sum_f32_t128_s2_o0" in mlir_cw_s2,
           "IR: cross_warp s=2 helper name")
    expect("pto.shuffle_bfly" in mlir_cw_s2,
           "IR: cross_warp s=2 has shuffle_bfly (butterfly for per-warp + leader)")
    expect("pto.syncthreads" in mlir_cw_s2,
           "IR: cross_warp s=2 has syncthreads")
    # scale > 1: per-warp uses butterfly, not hardware redux
    compiled_cw_s2.verify()

    # ══════════════════════════════════════════════════════════════════════════
    # Path 3c: cross_warp_reduce — scale > 1, scale*num_warps > 32 (manual, sum)
    #           (threads=128, scale=16, num_warps=4, total=64 > 32)
    # ══════════════════════════════════════════════════════════════════════════

    @pto.jit(target="a5")
    def kernel_cw_s16(scratch_gm: pto.ptr(pto.f32, "gm")):
        zero_u64 = pto.const(0, dtype=pto.ui64)
        ub_scratch = pto.castptr(zero_u64, pto.ptr(pto.f32, "ub"))
        with pto.simt():
            x = pto.const(1.0, dtype=pto.f32)
            _result = pto.simt_allreduce_sum(x, scratch=ub_scratch, threads=128, scale=16)

    compiled_cw_s16 = kernel_cw_s16.compile()
    mlir_cw_s16 = compiled_cw_s16.mlir_text()
    expect("func.func @__tl_allreduce_sum_f32_t128_s16_o0" in mlir_cw_s16,
           "IR: cross_warp s=16 manual helper name")
    expect("pto.syncthreads" in mlir_cw_s16,
           "IR: cross_warp s=16 has syncthreads")
    compiled_cw_s16.verify()

    # ── cross_warp: sum, f32, t=128, s=1, o=4 (non-zero thread_offset) ─────
    @pto.jit(target="a5")
    def kernel_cw_o4(scratch_gm: pto.ptr(pto.f32, "gm")):
        zero_u64 = pto.const(0, dtype=pto.ui64)
        ub_scratch = pto.castptr(zero_u64, pto.ptr(pto.f32, "ub"))
        with pto.simt():
            x = pto.const(1.0, dtype=pto.f32)
            _result = pto.simt_allreduce_sum(x, scratch=ub_scratch, threads=128, scale=1,
                                     thread_offset=4)

    compiled_cw_o4 = kernel_cw_o4.compile()
    mlir_cw_o4 = compiled_cw_o4.mlir_text()
    expect("func.func @__tl_allreduce_sum_f32_t128_s1_o4" in mlir_cw_o4,
           "IR: cross_warp o=4 helper name")
    expect("pto.get_tid_x" in mlir_cw_o4,
           "IR: cross_warp o=4 uses get_tid_x")
    expect("arith.subi" in mlir_cw_o4,
           "IR: cross_warp o=4 uses subi for tx = tid_x - offset")
    compiled_cw_o4.verify()

    # ══════════════════════════════════════════════════════════════════════════
    # Path 4: ub_reduce fallback — threads > 32, non-power-of-2
    #          (threads=48, scale=1)
    # ══════════════════════════════════════════════════════════════════════════

    @pto.jit(target="a5")
    def kernel_ub48(scratch_gm: pto.ptr(pto.f32, "gm")):
        zero_u64 = pto.const(0, dtype=pto.ui64)
        ub_scratch = pto.castptr(zero_u64, pto.ptr(pto.f32, "ub"))
        with pto.simt():
            x = pto.const(1.0, dtype=pto.f32)
            _result = pto.simt_allreduce_sum(x, scratch=ub_scratch, threads=48, scale=1)

    compiled_ub48 = kernel_ub48.compile()
    mlir_ub48 = compiled_ub48.mlir_text()
    expect("func.func @__tl_allreduce_sum_f32_t48_s1_o0" in mlir_ub48,
           "IR: ub_reduce fallback t=48 helper name")
    expect("pto.syncthreads" in mlir_ub48,
           "IR: ub_reduce fallback has syncthreads")
    expect("pto.store" in mlir_ub48,
           "IR: ub_reduce fallback has store")
    expect("pto.load" in mlir_ub48,
           "IR: ub_reduce fallback has load")
    compiled_ub48.verify()

    # ══════════════════════════════════════════════════════════════════════════
    # helper deduplication across multiple calls
    # ══════════════════════════════════════════════════════════════════════════

    @pto.jit(target="a5")
    def kernel_reuse(scratch_gm: pto.ptr(pto.f32, "gm")):
        zero_u64 = pto.const(0, dtype=pto.ui64)
        ub_scratch = pto.castptr(zero_u64, pto.ptr(pto.f32, "ub"))
        with pto.simt():
            x1 = pto.const(1.0, dtype=pto.f32)
            _r1 = pto.simt_allreduce_sum(x1, scratch=ub_scratch, threads=128, scale=1)
            x2 = pto.const(2.0, dtype=pto.f32)
            _r2 = pto.simt_allreduce_sum(x2, scratch=ub_scratch, threads=128, scale=1)

    compiled2 = kernel_reuse.compile()
    mlir2 = compiled2.mlir_text()

    definitions = mlir2.count("func.func @__tl_allreduce_sum_f32_t128_s1_o0")
    expect(definitions == 1,
           f"IR: helper defined {definitions} times, expected 1")
    calls = mlir2.count("call @__tl_allreduce_sum_f32_t128_s1_o0")
    expect(calls == 2, f"IR: expected 2 call sites, got {calls}")
    compiled2.verify()


    # ══════════════════════════════════════════════════════════════════════════
    # scratch required for ub_reduce and cross_warp paths
    # ══════════════════════════════════════════════════════════════════════════

    # cross_warp requires scratch — use a real JIT kernel so the error
    # originates from _dispatch_allreduce_helper, not from a bare Python float.
    @pto.jit(target="a5")
    def kernel_no_scratch_cw(scratch_gm: pto.ptr(pto.f32, "gm")):
        with pto.simt():
            x = pto.const(1.0, dtype=pto.f32)
            _result = pto.simt_allreduce_sum(x, scratch=None, threads=128, scale=1)

    try:
        kernel_no_scratch_cw.compile()
        raise AssertionError("expected ValueError for missing scratch (cross_warp)")
    except ValueError as e:
        expect("requires a UB scratch buffer" in str(e),
               f"error message should mention scratch (cross_warp), got: {e}")

    # ub_reduce (non-pow2) requires scratch
    @pto.jit(target="a5")
    def kernel_no_scratch_ub(scratch_gm: pto.ptr(pto.f32, "gm")):
        with pto.simt():
            x = pto.const(1.0, dtype=pto.f32)
            _result = pto.simt_allreduce_sum(x, scratch=None, threads=6, scale=1)

    try:
        kernel_no_scratch_ub.compile()
        raise AssertionError("expected ValueError for missing scratch (ub_reduce)")
    except ValueError as e:
        expect("requires a UB scratch buffer" in str(e),
               f"error message should mention scratch (ub_reduce), got: {e}")

    # scratch must be a pto.ptr type
    try:
        simt_allreduce_sum(1.0, scratch="not_a_ptr", threads=6, scale=1)
        raise AssertionError("expected TypeError for non-ptr scratch")
    except (TypeError, AttributeError):
        pass

    # cross_warp: gm scratch (wrong memory space) should be rejected
    @pto.jit(target="a5")
    def kernel_gm_scratch(scratch_gm: pto.ptr(pto.f32, "gm")):
        with pto.simt():
            x = pto.const(1.0, dtype=pto.f32)
            _result = pto.simt_allreduce_sum(x, scratch=scratch_gm, threads=128, scale=1)

    try:
        kernel_gm_scratch.compile()
        raise AssertionError("expected TypeError for gm scratch")
    except TypeError as e:
        expect("UB" in str(e).upper() or "memory space" in str(e).lower(),
               f"gm scratch error should mention memory space, got: {e}")

    # cross_warp: i32 scratch with f32 x (dtype mismatch) should be rejected
    @pto.jit(target="a5")
    def kernel_dtype_mismatch(scratch_gm: pto.ptr(pto.f32, "gm")):
        zero_u64 = pto.const(0, dtype=pto.ui64)
        ub_i32 = pto.castptr(zero_u64, pto.ptr(pto.i32, "ub"))
        with pto.simt():
            x = pto.const(1.0, dtype=pto.f32)
            _result = pto.simt_allreduce_sum(x, scratch=ub_i32, threads=128, scale=1)

    try:
        kernel_dtype_mismatch.compile()
        raise AssertionError("expected TypeError for dtype mismatch scratch")
    except TypeError as e:
        err = str(e)
        expect("element type" in err.lower() or "mismatch" in err.lower(),
               f"dtype mismatch should mention element type, got: {e}")

    print("ptodsl_allreduce: PASS")


if __name__ == "__main__":
    main()
