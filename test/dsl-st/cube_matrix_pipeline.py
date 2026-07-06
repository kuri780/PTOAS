#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import numpy as np

from common import auto_main, golden_output_case
from ptodsl import pto


M = 16
K = 32
N = 48

L1_A_ADDR = 0
L1_B_ADDR = 4096
UB_O_ADDR = 0
L0A_ADDR = 0
L0B_ADDR = 0
L0C_ADDR = 0


@pto.cube
def cube_gemm_tile(a_mat, b_mat, o_tile, a_l0a, b_l0b, o_acc):
    m = a_mat.valid_shape[0]
    k = a_mat.valid_shape[1]
    n = b_mat.valid_shape[1]

    pto.mte_l1_l0a(a_mat.as_ptr(), a_l0a.as_ptr(), m, k)
    pto.mte_l1_l0b(b_mat.as_ptr(), b_l0b.as_ptr(), k, n)
    pto.set_flag(pto.Pipe.MTE1, pto.Pipe.M, event_id=0)
    pto.wait_flag(pto.Pipe.MTE1, pto.Pipe.M, event_id=0)
    pto.mad(
        a_l0a.as_ptr(),
        b_l0b.as_ptr(),
        o_acc.as_ptr(),
        m,
        n,
        k,
        unit_flag=pto.MadUnitFlagMode.CHECK_ONLY,
        sat=pto.SatMode.OFF,
    )
    pto.set_flag(pto.Pipe.M, pto.Pipe.FIX, event_id=1)
    pto.wait_flag(pto.Pipe.M, pto.Pipe.FIX, event_id=1)
    pto.mte_l0c_ub(
        o_acc.as_ptr(),
        o_tile.as_ptr(),
        m,
        n,
        n,
        n,
    )


@pto.jit(
    name="cube_matrix_pipeline_kernel",
    target="a5",
    mode="explicit",
    insert_sync=False,
)
def cube_matrix_pipeline_kernel(
    a_ptr: pto.ptr(pto.f32, "gm"),
    b_ptr: pto.ptr(pto.f32, "gm"),
    o_ptr: pto.ptr(pto.f32, "gm"),
):
    a_view = pto.make_tensor_view(a_ptr, shape=[M, K], strides=[K, 1])
    b_view = pto.make_tensor_view(b_ptr, shape=[K, N], strides=[N, 1])
    o_view = pto.make_tensor_view(o_ptr, shape=[M, N], strides=[N, 1])

    a_part = pto.partition_view(a_view, offsets=[0, 0], sizes=[M, K])
    b_part = pto.partition_view(b_view, offsets=[0, 0], sizes=[K, N])
    o_part = pto.partition_view(o_view, offsets=[0, 0], sizes=[M, N])

    a_mat = pto.alloc_tile(
        shape=[M, K],
        dtype=pto.f32,
        memory_space=pto.MemorySpace.MAT,
        addr=L1_A_ADDR,
        valid_shape=[M, K],
    )
    b_mat = pto.alloc_tile(
        shape=[K, N],
        dtype=pto.f32,
        memory_space=pto.MemorySpace.MAT,
        addr=L1_B_ADDR,
        valid_shape=[K, N],
    )
    o_tile = pto.alloc_tile(
        shape=[M, N],
        dtype=pto.f32,
        addr=UB_O_ADDR,
        valid_shape=[M, N],
    )
    a_l0a = pto.alloc_tile(
        shape=[M, K],
        dtype=pto.f32,
        memory_space=pto.MemorySpace.LEFT,
        addr=L0A_ADDR,
        valid_shape=[M, K],
    )
    b_l0b = pto.alloc_tile(
        shape=[K, N],
        dtype=pto.f32,
        memory_space=pto.MemorySpace.RIGHT,
        addr=L0B_ADDR,
        valid_shape=[K, N],
    )
    o_acc = pto.alloc_tile(
        shape=[M, N],
        dtype=pto.f32,
        memory_space=pto.MemorySpace.ACC,
        addr=L0C_ADDR,
        valid_shape=[M, N],
    )

    pto.tile.load(a_part, a_mat)
    pto.tile.load(b_part, b_mat)
    pto.set_flag(pto.Pipe.MTE2, pto.Pipe.MTE1, event_id=0)
    pto.wait_flag(pto.Pipe.MTE2, pto.Pipe.MTE1, event_id=0)
    cube_gemm_tile(a_mat, b_mat, o_tile, a_l0a, b_l0b, o_acc)
    pto.set_flag(pto.Pipe.FIX, pto.Pipe.MTE3, event_id=2)
    pto.wait_flag(pto.Pipe.FIX, pto.Pipe.MTE3, event_id=2)
    pto.tile.store(o_tile, o_part)


def make_inputs():
    a = (np.arange(M * K, dtype=np.float32).reshape(M, K) % 7) - 3.0
    b = (np.arange(K * N, dtype=np.float32).reshape(K, N) % 5) - 2.0
    return [a, b]


def make_expected(a, b):
    return a @ b


CASES = [
    golden_output_case(
        "cube_matrix_pipeline_gemm",
        cube_matrix_pipeline_kernel,
        inputs=make_inputs,
        expected=make_expected,
        rtol=1e-4,
        atol=1e-4,
    ),
]


auto_main(globals())
