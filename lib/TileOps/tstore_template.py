# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""`pto.tstore` 的 TileLang DSL 模板"""

import tilelang_dsl as pto

def _match_store_tile_layout(src, *, row_major: bool, s_layout) -> bool:
    b_layout_ok = (
        src.config.b_layout == pto.BLayout.ROW_MAJOR
        if row_major
        else src.config.b_layout != pto.BLayout.ROW_MAJOR
    )
    return b_layout_ok and src.config.s_layout == s_layout


def _check_store_bounds(src, dst, *, logical_rows, logical_cols, stride_axis=None) -> bool:
    if dst.rank != 5:
        return False
    if stride_axis is not None and dst.strides[stride_axis] != 1:
        return False
    if src.valid_shape[0] != logical_rows:
        return False
    if src.valid_shape[1] != logical_cols:
        return False
    if src.valid_shape[0] > src.shape[0]:
        return False
    if src.valid_shape[1] > src.shape[1]:
        return False
    return True


def _tstore_preconditions_nd(src, dst) -> bool:
    logical_rows = dst.shape[0] * dst.shape[1] * dst.shape[2] * dst.shape[3]
    logical_cols = dst.shape[4]
    return _match_store_tile_layout(
        src, row_major=True, s_layout=pto.SLayout.NONE_BOX
    ) and _check_store_bounds(
        src, dst, logical_rows=logical_rows, logical_cols=logical_cols, stride_axis=4
    )
    
def _tstore_preconditions_dn(src, dst) -> bool:
    logical_rows = dst.shape[3]
    logical_cols = dst.shape[0] * dst.shape[1] * dst.shape[2] * dst.shape[4]
    return _match_store_tile_layout(
        src, row_major=False, s_layout=pto.SLayout.NONE_BOX
    ) and _check_store_bounds(
        src, dst, logical_rows=logical_rows, logical_cols=logical_cols, stride_axis=3
    )

def _tstore_preconditions_nz(src, dst) -> bool:
    logical_rows = dst.shape[2] * dst.shape[3]
    logical_cols = dst.shape[0] * dst.shape[1] * dst.shape[4]
    return _match_store_tile_layout(
        src, row_major=False, s_layout=pto.SLayout.ROW_MAJOR
    ) and _check_store_bounds(
        src, dst, logical_rows=logical_rows, logical_cols=logical_cols
    )

@pto.vkernel(
    target="a5",
    op="pto.tstore",
    advanced=True,
    constraints=[_tstore_preconditions_nd],
)
def template_tstore_nd(src: pto.Tile, dst: pto.PartitionTensorView):
    dtype = src.element_type
    elem_bytes = pto.bytewidth(dtype)

    g0, g1, g2, g3, g4 = dst.shape
    s0, s1, s2, s3, s4 = dst.strides

    valid_rows, valid_cols = src.valid_shape
    ub_rows, ub_cols = src.shape

    # These preconditions are expressed through the descriptor-level constraint
    # callable above, using direct `src.*` / `dst.*` metadata syntax.

    n_burst = g3
    len_burst = valid_cols * elem_bytes
    ub_stride = ub_cols * elem_bytes
    gm_stride = s3 * elem_bytes

    src_stride2 = g3 * ub_cols
    src_stride1 = g2 * src_stride2
    src_stride0 = g1 * src_stride1

    loop1 = g2
    loop2 = g1
    loop1_src_stride = src_stride2 * elem_bytes
    loop1_dst_stride = s2 * elem_bytes
    loop2_src_stride = src_stride1 * elem_bytes
    loop2_dst_stride = s1 * elem_bytes

    ub_ptr = src.as_ptr()
    gm_ptr = dst.as_ptr()

    if loop1 != 1 or loop2 != 1:
        pto.set_loop2_stride_ubtoout(
            src_stride=loop2_src_stride, dst_stride=loop2_dst_stride
        )
        pto.set_loop1_stride_ubtoout(
            src_stride=loop1_src_stride, dst_stride=loop1_dst_stride
        )
        pto.set_loop_size_ubtoout(loop1=loop1, loop2=loop2)

    for i in range(0, g0, 1):
        src_i = pto.addptr(ub_ptr, i * src_stride0)
        dst_i = pto.addptr(gm_ptr, i * s0)
        pto.copy_ubuf_to_gm(
            dst=dst_i,
            src=src_i,
            n_burst=n_burst,
            len_burst=len_burst,
            gm_stride=gm_stride,
            ub_stride=ub_stride,
        )

    if loop1 != 1 or loop2 != 1:
        pto.set_loop_size_ubtoout(loop1=1, loop2=1)
    return

@pto.vkernel(
    target="a5",
    op="pto.tstore",
    advanced=True,
    constraints=[_tstore_preconditions_dn],
)
def template_tstore_dn(src: pto.Tile, dst: pto.PartitionTensorView):
    dtype = src.element_type
    elem_bytes = pto.bytewidth(dtype)

    g0, g1, g2, g3, g4 = dst.shape
    s0, s1, s2, s3, s4 = dst.strides

    valid_rows, valid_cols = src.valid_shape
    ub_rows, ub_cols = src.shape

    n_burst = g4
    len_burst = valid_rows * elem_bytes
    gm_stride = s4 * elem_bytes
    ub_stride = ub_rows * elem_bytes

    # UB 源 tile 是列高 `ub_rows` 的紧凑 col-major 布局，
    # 与 `TStoreVecDN` 一样由 `g4` / `g2` / `g1` 递推出三级 stride。
    src_stride2 = ub_rows * g4
    src_stride1 = g2 * src_stride2
    src_stride0 = g1 * src_stride1

    loop1 = g2
    loop2 = g1
    loop1_src_stride = src_stride2 * elem_bytes
    loop1_dst_stride = s2 * elem_bytes
    loop2_src_stride = src_stride1 * elem_bytes
    loop2_dst_stride = s1 * elem_bytes

    ub_ptr = src.as_ptr()
    gm_ptr = dst.as_ptr()

    if loop1 != 1 or loop2 != 1:
        pto.set_loop2_stride_ubtoout(
            src_stride=loop2_src_stride, dst_stride=loop2_dst_stride
        )
        pto.set_loop1_stride_ubtoout(
            src_stride=loop1_src_stride, dst_stride=loop1_dst_stride
        )
        pto.set_loop_size_ubtoout(loop1=loop1, loop2=loop2)

    for i in range(0, g0, 1):
        src_i = pto.addptr(ub_ptr, i * src_stride0)
        dst_i = pto.addptr(gm_ptr, i * s0)
        pto.copy_ubuf_to_gm(
            dst=dst_i,
            src=src_i,
            n_burst=n_burst,
            len_burst=len_burst,
            gm_stride=gm_stride,
            ub_stride=ub_stride,        
        )

    if loop1 != 1 or loop2 != 1:
        pto.set_loop_size_ubtoout(loop1=1, loop2=1)
    return

@pto.vkernel(
    target="a5",
    op="pto.tstore",
    advanced=True,
    constraints=[_tstore_preconditions_nz],
)
def template_tstore_nz(src: pto.Tile, dst: pto.PartitionTensorView):
    dtype = src.element_type
    elem_bytes = pto.bytewidth(dtype)

    g0, g1, g2, g3, g4 = dst.shape
    s0, s1, s2, s3, s4 = dst.strides

    valid_rows, valid_cols = src.valid_shape
    ub_rows, ub_cols = src.shape

    # 对应 C++ `C0_SIZE_BYTE`。NZ 每个 burst 始终写一个完整 C0 block。
    c0_size_bytes = 32
    n_burst = g1
    len_burst = valid_rows * c0_size_bytes
    gm_stride = s1 * elem_bytes
    ub_stride = ub_rows * c0_size_bytes

    # 每个 g0 block 在 UB 中由 `g1` 个 NZ block 串接组成。
    tile_stride = g1 * ub_rows * g4

    ub_ptr = src.as_ptr()
    gm_ptr = dst.as_ptr()

    # NZ path 本身不使用 loop1/loop2，主动切回 normal mode 避免继承旧状态。
    pto.set_loop_size_ubtoout(loop1=1, loop2=1)
    for i in range(0, g0, 1):
        src_i = pto.addptr(ub_ptr, i * tile_stride)
        dst_i = pto.addptr(gm_ptr, i * s0)
        pto.copy_ubuf_to_gm(
            dst=dst_i,
            src=src_i,
            n_burst=n_burst,
            len_burst=len_burst,
            gm_stride=gm_stride,
            ub_stride=ub_stride,            
        )
    return
