# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""PTODSL TileLib templates for ``pto.tinsert``."""

from ptodsl import pto
import ptodsl.tilelib as tilelib

from ._common import NUMERIC_DTYPES


BLOCK_BYTE_SIZE = 32


def _acc_to_mat(src_kind, dst_kind, src_memory_space, dst_memory_space, src_config, dst_config, **_):
    return (
        src_kind == "tile"
        and dst_kind == "tile"
        and src_memory_space == "acc"
        and dst_memory_space == "mat"
        and src_config.b_layout == "col_major"
        and src_config.s_layout == "row_major"
        and dst_config.b_layout == "col_major"
        and dst_config.s_layout == "row_major"
    )


def _vec_to_vec_nd(src_memory_space, dst_memory_space, src_config, dst_config, src_valid_shape, src_dtype, dst_dtype, **_):
    return (
        src_memory_space == "ub"
        and dst_memory_space == "ub"
        and src_config.b_layout == "row_major"
        and src_config.s_layout == "none_box"
        and dst_config.b_layout == "row_major"
        and dst_config.s_layout == "none_box"
        and src_dtype == dst_dtype
        and src_valid_shape != (1, 1)
    )


def _vec_to_vec_nd_scalar(src_memory_space, dst_memory_space, src_config, dst_config, src_valid_shape, src_dtype, dst_dtype, **_):
    return (
        src_memory_space == "ub"
        and dst_memory_space == "ub"
        and src_config.b_layout == "row_major"
        and src_config.s_layout == "none_box"
        and dst_config.b_layout == "row_major"
        and dst_config.s_layout == "none_box"
        and src_dtype == dst_dtype
        and src_valid_shape == (1, 1)
    )


_DTYPES = [(dtype, "i32", "i32", dtype) for dtype in NUMERIC_DTYPES]


@tilelib.tile_template(
    op="pto.tinsert",
    target="a5",
    name="template_tinsert_acc_to_mat_basic",
    dtypes=(
        ("f32", "i32", "i32", "f16"),
        ("f32", "i32", "i32", "bf16"),
        ("f32", "i32", "i32", "f32"),
        ("i32", "i32", "i32", "i32"),
    ),
    iteration_axis="none",
    op_engine="other",
    op_class="movement",
    constraints=[_acc_to_mat],
    priority=1,
    id=2,
    loop_depth=0,
    is_post_update=False,
    tags=("insert", "acc", "mat"),
)
def template_tinsert_acc_to_mat_basic(
    src: pto.Tile,
    index_row: pto.i32,
    index_col: pto.i32,
    dst: pto.Tile,
):
    elem_bytes = pto.bytewidth(dst.dtype)
    c0_size = BLOCK_BYTE_SIZE // elem_bytes
    valid_rows, valid_cols = src.valid_shape
    n_size = (valid_cols + c0_size - 1) // c0_size * c0_size

    col_block = index_col // c0_size
    col_mod = index_col - col_block * c0_size
    dst_offset = dst.shape[0] * c0_size * col_block + index_row * c0_size + col_mod

    pto.mte_l0c_l1(
        src.as_ptr(),
        pto.addptr(dst.as_ptr(), dst_offset),
        valid_rows,
        n_size,
        src.shape[0] * pto.bytewidth(src.dtype),
        dst.shape[0] * c0_size * elem_bytes,
    )


@tilelib.tile_template(
    op="pto.tinsert",
    target="a5",
    name="template_tinsert_vec_to_vec_nd_basic",
    dtypes=_DTYPES,
    iteration_axis="none",
    op_engine="vector",
    op_class="movement",
    constraints=[_vec_to_vec_nd],
    id=0,
    loop_depth=2,
    is_post_update=False,
    tags=("move", "insert", "ub"),
)
def template_tinsert_vec_to_vec_nd_basic(
    src: pto.Tile,
    index_row: pto.i32,
    index_col: pto.i32,
    dst: pto.Tile,
):
    dtype = dst.dtype
    valid_rows, valid_cols = src.valid_shape
    lanes = pto.elements_per_vreg(dtype)
    for row in range(0, valid_rows, 1):
        remained = valid_cols
        for col in range(0, valid_cols, lanes):
            mask, remained = pto.make_mask(dtype, remained)
            data = pto.vlds(src[row, col:])
            pto.vsts(data, dst[index_row + row, index_col + col:], mask)


@tilelib.tile_template(
    op="pto.tinsert",
    target="a5",
    name="template_tinsert_vec_to_vec_nd_scalar_basic",
    dtypes=_DTYPES,
    iteration_axis="none",
    op_engine="vector",
    op_class="movement",
    constraints=[_vec_to_vec_nd_scalar],
    priority=1,
    id=1,
    loop_depth=0,
    is_post_update=False,
    tags=("move", "insert", "ub", "scalar"),
)
def template_tinsert_vec_to_vec_nd_scalar_basic(
    src: pto.Tile,
    index_row: pto.i32,
    index_col: pto.i32,
    dst: pto.Tile,
):
    value = pto.load_scalar(src.as_ptr(), 0)
    pto.store_scalar(dst.as_ptr(), index_row * dst.shape[1] + index_col, value)
