# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Shared PTODSL templates for row/column expand binary TileOps."""

from ptodsl import pto
import ptodsl.tilelib as tilelib


NUMERIC_SIGNATURES = [
    ("i8", "i8", "i8"),
    ("i16", "i16", "i16"),
    ("i32", "i32", "i32"),
    ("f16", "f16", "f16"),
    ("bf16", "bf16", "bf16"),
    ("f32", "f32", "f32"),
]

FLOAT_SIGNATURES = [
    ("f16", "f16", "f16"),
    ("f32", "f32", "f32"),
]


def _ub_or_vec_row_major(operand_memory_spaces, operand_b_layouts, operand_s_layouts, **_):
    return (
        all(space in {"ub", "vec"} for space in operand_memory_spaces)
        and all(layout == "row_major" for layout in operand_b_layouts)
        and all(layout == "none_box" for layout in operand_s_layouts)
    )


def register_row_expand_binary(*, op, name, vector_op, dtypes):
    @tilelib.tile_template(
        op=op,
        target="a5",
        name=name,
        dtypes=dtypes,
        constraints=[
            _ub_or_vec_row_major,
            _valid_row_expand_binary,
        ],
        id=0,
        loop_depth=2,
        is_post_update=False,
        tags=("row_expand", "binary"),
    )
    def template(src0: pto.Tile, src1: pto.Tile, dst: pto.Tile):
        _emit_row_expand_body(src0, src1, dst, vector_op)

    return template


def register_column_expand_binary(*, op, name, vector_op, dtypes):
    @tilelib.tile_template(
        op=op,
        target="a5",
        name=name,
        dtypes=dtypes,
        constraints=[
            _ub_or_vec_row_major,
            _valid_column_expand_binary,
        ],
        id=0,
        loop_depth=2,
        is_post_update=False,
        tags=("column_expand", "binary"),
    )
    def template(src0: pto.Tile, src1: pto.Tile, dst: pto.Tile):
        _emit_column_expand_body(src0, src1, dst, vector_op)

    return template


def register_row_expand_expdif():
    @tilelib.tile_template(
        op="pto.trowexpandexpdif",
        target="a5",
        name="template_trowexpandexpdif_f32",
        dtypes=[("f32", "f32", "f32")],
        constraints=[
            _ub_or_vec_row_major,
            _valid_row_expand_binary,
        ],
        id=0,
        loop_depth=2,
        is_post_update=False,
        tags=("row_expand", "expdif"),
    )
    def template_f32(src0: pto.Tile, src1: pto.Tile, dst: pto.Tile):
        _emit_row_expand_body(
            src0,
            src1,
            dst,
            lambda lhs, rhs, mask: pto.vexpdif(lhs, rhs, mask, pto.VcvtPartMode.EVEN),
        )

    @tilelib.tile_template(
        op="pto.trowexpandexpdif",
        target="a5",
        name="template_trowexpandexpdif_f16",
        dtypes=[("f16", "f16", "f16")],
        constraints=[
            _ub_or_vec_row_major,
            _valid_row_expand_binary,
        ],
        id=0,
        loop_depth=2,
        is_post_update=False,
        tags=("row_expand", "expdif"),
    )
    def template_f16(src0: pto.Tile, src1: pto.Tile, dst: pto.Tile):
        _emit_row_expand_body(
            src0,
            src1,
            dst,
            lambda lhs, rhs, mask: pto.vexp(pto.vsub(lhs, rhs, mask), mask),
        )

    return template_f32, template_f16


def register_column_expand_expdif():
    @tilelib.tile_template(
        op="pto.tcolexpandexpdif",
        target="a5",
        name="template_tcolexpandexpdif_f32",
        dtypes=[("f32", "f32", "f32")],
        constraints=[
            _ub_or_vec_row_major,
            _valid_column_expand_binary,
        ],
        id=0,
        loop_depth=2,
        is_post_update=False,
        tags=("column_expand", "expdif"),
    )
    def template_f32(src0: pto.Tile, src1: pto.Tile, dst: pto.Tile):
        _emit_column_expand_body(
            src0,
            src1,
            dst,
            lambda lhs, rhs, mask: pto.vexpdif(lhs, rhs, mask, pto.VcvtPartMode.ODD),
        )

    @tilelib.tile_template(
        op="pto.tcolexpandexpdif",
        target="a5",
        name="template_tcolexpandexpdif_f16",
        dtypes=[("f16", "f16", "f16")],
        constraints=[
            _ub_or_vec_row_major,
            _valid_column_expand_binary,
        ],
        id=0,
        loop_depth=2,
        is_post_update=False,
        tags=("column_expand", "expdif"),
    )
    def template_f16(src0: pto.Tile, src1: pto.Tile, dst: pto.Tile):
        _emit_column_expand_body(
            src0,
            src1,
            dst,
            lambda lhs, rhs, mask: pto.vexp(pto.vsub(lhs, rhs, mask), mask),
        )

    return template_f32, template_f16


def _emit_row_expand_body(src0, src1, dst, vector_op):
    dtype = dst.dtype
    valid_rows, valid_cols = dst.valid_shape
    lanes = pto.elements_per_vreg(dtype)

    with pto.for_(0, valid_rows, step=1) as row:
        col_loop = pto.for_(0, valid_cols, step=lanes).carry(remained=valid_cols)
        with col_loop:
            col = col_loop.iv
            mask, remained = pto.make_mask(dtype, col_loop.remained)
            lhs = pto.vlds(src0[row, col:])
            scalar_vec = pto.vlds(src1[row, :])
            rhs = pto.vdup(scalar_vec, mask)
            result = vector_op(lhs, rhs, mask)
            pto.vsts(result, dst[row, col:], mask)
            col_loop.update(remained=remained)


def _emit_column_expand_body(src0, src1, dst, vector_op):
    dtype = dst.dtype
    valid_rows, valid_cols = dst.valid_shape
    lanes = pto.elements_per_vreg(dtype)

    with pto.for_(0, valid_rows, step=1) as row:
        col_loop = pto.for_(0, valid_cols, step=lanes).carry(remained=valid_cols)
        with col_loop:
            col = col_loop.iv
            mask, remained = pto.make_mask(dtype, col_loop.remained)
            lhs = pto.vlds(src0[row, col:])
            rhs = pto.vlds(src1[0, col:])
            result = vector_op(lhs, rhs, mask)
            pto.vsts(result, dst[row, col:], mask)
            col_loop.update(remained=remained)


def _valid_row_expand_binary(src0_valid_shape=(), src1_valid_shape=(), dst_valid_shape=(), **_):
    return (
        len(src0_valid_shape) == 2
        and len(src1_valid_shape) == 2
        and len(dst_valid_shape) == 2
        and src0_valid_shape == dst_valid_shape
        and src1_valid_shape[0] == dst_valid_shape[0]
        and src1_valid_shape[1] >= 1
    )


def _valid_column_expand_binary(src0_valid_shape=(), src1_valid_shape=(), dst_valid_shape=(), **_):
    return (
        len(src0_valid_shape) == 2
        and len(src1_valid_shape) == 2
        and len(dst_valid_shape) == 2
        and src0_valid_shape == dst_valid_shape
        and src1_valid_shape[0] >= 1
        and src1_valid_shape[1] == dst_valid_shape[1]
    )


__all__ = [
    "FLOAT_SIGNATURES",
    "NUMERIC_SIGNATURES",
    "register_column_expand_binary",
    "register_column_expand_expdif",
    "register_row_expand_binary",
    "register_row_expand_expdif",
]
