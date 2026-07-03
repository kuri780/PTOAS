# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Shared helpers for PTODSL TileLib load/store template ports."""

from ._common import NUMERIC_DTYPES


LOAD_STORE_DTYPES = tuple((dtype, dtype) for dtype in NUMERIC_DTYPES)
MAT_LOAD_DTYPES = (("f16", "f16"), ("bf16", "bf16"), ("f32", "f32"))
ACC_STORE_DTYPES = (
    ("f32", "f32"),
    ("f32", "f16"),
    ("f32", "bf16"),
    ("i32", "i32"),
)


def _known_eq(lhs, rhs) -> bool:
    return lhs is None or rhs is None or lhs == rhs


def _known_le(lhs, rhs) -> bool:
    return lhs is None or rhs is None or lhs <= rhs


def _shape_size(shape):
    result = 1
    for dim in shape:
        result *= dim
    return result


def _view_rank(shape):
    return len(shape) if shape is not None else None


def _stride_at(strides, index):
    if strides is None:
        return None
    return strides[index]


def _is_tile_layout(config, *, row_major: bool, s_layout: str) -> bool:
    if config is None:
        return False
    if row_major:
        return config.b_layout == "row_major" and config.s_layout == s_layout
    return config.b_layout != "row_major" and config.s_layout == s_layout


def _check_load_bounds(src_shape, src_strides, dst_shape, dst_valid_shape, *, logical_rows, logical_cols=None, stride_axis=None):
    if _view_rank(src_shape) != 5:
        return False
    if stride_axis is not None and not _known_eq(_stride_at(src_strides, stride_axis), 1):
        return False
    if not _known_le(dst_valid_shape[0], logical_rows):
        return False
    if not _known_le(logical_rows, dst_shape[0]):
        return False
    if not _known_le(dst_valid_shape[0], dst_shape[0]):
        return False
    if logical_cols is not None:
        if not _known_le(dst_valid_shape[1], logical_cols):
            return False
        if not _known_le(logical_cols, dst_shape[1]):
            return False
    if not _known_le(dst_valid_shape[1], dst_shape[1]):
        return False
    return True


def _check_store_bounds(src_shape, src_valid_shape, dst_shape, dst_strides, *, logical_rows, logical_cols, stride_axis=None):
    if _view_rank(dst_shape) != 5:
        return False
    if stride_axis is not None and not _known_eq(_stride_at(dst_strides, stride_axis), 1):
        return False
    if not _known_eq(src_valid_shape[0], logical_rows):
        return False
    if not _known_eq(src_valid_shape[1], logical_cols):
        return False
    if not _known_le(src_valid_shape[0], src_shape[0]):
        return False
    if not _known_le(src_valid_shape[1], src_shape[1]):
        return False
    return True


def tload_nd2nd_constraint(src_kind, src_shape, src_strides, src_memory_space, dst_kind, dst_shape, dst_valid_shape, dst_memory_space, dst_config, **_):
    if src_kind != "view" or dst_kind != "tile" or src_memory_space != "gm" or dst_memory_space not in {"ub", "vec"}:
        return False
    logical_rows = _shape_size(src_shape[:4])
    logical_cols = src_shape[4]
    return _is_tile_layout(dst_config, row_major=True, s_layout="none_box") and _check_load_bounds(
        src_shape,
        src_strides,
        dst_shape,
        dst_valid_shape,
        logical_rows=logical_rows,
        logical_cols=logical_cols,
        stride_axis=4,
    )


def tload_dn2dn_constraint(src_kind, src_shape, src_strides, src_memory_space, dst_kind, dst_shape, dst_valid_shape, dst_memory_space, dst_config, **_):
    if src_kind != "view" or dst_kind != "tile" or src_memory_space != "gm" or dst_memory_space not in {"ub", "vec"}:
        return False
    logical_rows = src_shape[3]
    logical_cols = src_shape[0] * src_shape[1] * src_shape[2] * src_shape[4]
    return _is_tile_layout(dst_config, row_major=False, s_layout="none_box") and _check_load_bounds(
        src_shape,
        src_strides,
        dst_shape,
        dst_valid_shape,
        logical_rows=logical_rows,
        logical_cols=logical_cols,
        stride_axis=3,
    )


def tload_nz2nz_constraint(src_kind, src_shape, src_memory_space, dst_kind, dst_shape, dst_valid_shape, dst_memory_space, dst_config, **_):
    if src_kind != "view" or dst_kind != "tile" or src_memory_space != "gm" or dst_memory_space not in {"ub", "vec"}:
        return False
    logical_rows = src_shape[2]
    return _is_tile_layout(dst_config, row_major=False, s_layout="row_major") and _check_load_bounds(
        src_shape,
        None,
        dst_shape,
        dst_valid_shape,
        logical_rows=logical_rows,
    )


def tstore_nd_constraint(src_kind, src_shape, src_valid_shape, src_memory_space, src_config, dst_kind, dst_shape, dst_strides, dst_memory_space, **_):
    if src_kind != "tile" or dst_kind != "view" or src_memory_space not in {"ub", "vec"} or dst_memory_space != "gm":
        return False
    logical_rows = _shape_size(dst_shape[:4])
    logical_cols = dst_shape[4]
    return _is_tile_layout(src_config, row_major=True, s_layout="none_box") and _check_store_bounds(
        src_shape,
        src_valid_shape,
        dst_shape,
        dst_strides,
        logical_rows=logical_rows,
        logical_cols=logical_cols,
        stride_axis=4,
    )


def tstore_dn_constraint(src_kind, src_shape, src_valid_shape, src_memory_space, src_config, dst_kind, dst_shape, dst_strides, dst_memory_space, **_):
    if src_kind != "tile" or dst_kind != "view" or src_memory_space not in {"ub", "vec"} or dst_memory_space != "gm":
        return False
    logical_rows = dst_shape[3]
    logical_cols = dst_shape[0] * dst_shape[1] * dst_shape[2] * dst_shape[4]
    return _is_tile_layout(src_config, row_major=False, s_layout="none_box") and _check_store_bounds(
        src_shape,
        src_valid_shape,
        dst_shape,
        dst_strides,
        logical_rows=logical_rows,
        logical_cols=logical_cols,
        stride_axis=3,
    )


def tstore_nz_constraint(src_kind, src_shape, src_valid_shape, src_memory_space, src_config, dst_kind, dst_shape, dst_memory_space, **_):
    if src_kind != "tile" or dst_kind != "view" or src_memory_space not in {"ub", "vec"} or dst_memory_space != "gm":
        return False
    logical_rows = dst_shape[2] * dst_shape[3]
    logical_cols = dst_shape[0] * dst_shape[1] * dst_shape[4]
    return _is_tile_layout(src_config, row_major=False, s_layout="row_major") and _check_store_bounds(
        src_shape,
        src_valid_shape,
        dst_shape,
        None,
        logical_rows=logical_rows,
        logical_cols=logical_cols,
    )


def tload_mat_nd2nz_constraint(src_kind, src_shape, src_memory_space, dst_kind, dst_valid_shape, dst_memory_space, dst_config, dst_dtype, **_):
    if src_kind != "view" or dst_kind != "tile" or src_memory_space != "gm" or dst_memory_space != "mat":
        return False
    if dst_config.b_layout != "col_major" or dst_config.s_layout != "row_major":
        return False
    if dst_dtype not in {"f16", "bf16", "f32"}:
        return False
    return _view_rank(src_shape) != 5 or _known_eq(src_shape[4], dst_valid_shape[1])


def tload_mat_dn2nz_constraint(src_kind, src_shape, src_memory_space, dst_kind, dst_valid_shape, dst_memory_space, dst_config, dst_dtype, **_):
    if src_kind != "view" or dst_kind != "tile" or src_memory_space != "gm" or dst_memory_space != "mat":
        return False
    if dst_config.b_layout != "col_major" or dst_config.s_layout != "row_major":
        return False
    if dst_dtype not in {"f16", "bf16", "f32"}:
        return False
    return _view_rank(src_shape) != 5 or _known_eq(src_shape[4], dst_valid_shape[0])


def tstore_acc_base(src_kind, src_memory_space, src_dtype, dst_kind, dst_memory_space, **_):
    return (
        src_kind == "tile"
        and dst_kind == "view"
        and src_memory_space == "acc"
        and dst_memory_space == "gm"
        and src_dtype in {"f32", "i32", "si32"}
    )


def tstore_acc_nz2nd_constraint(dst_shape, dst_layout, **context):
    if not tstore_acc_base(**context):
        return False
    return dst_layout in {None, "nd", "row_major"} and _view_rank(dst_shape) == 5


def tstore_acc_nz2dn_constraint(dst_shape, dst_layout, **context):
    if not tstore_acc_base(**context):
        return False
    return dst_layout in {"dn", "col_major"}


def tstore_acc_nz2nz_constraint(dst_shape, dst_layout, **context):
    if not tstore_acc_base(**context):
        return False
    return dst_layout in {"nz", "fractal"}


def tstore_fp_constraint(src_kind, src_memory_space, src_dtype, fp_kind, fp_memory_space, dst_kind, dst_memory_space, **_):
    return (
        src_kind == "tile"
        and fp_kind == "tile"
        and dst_kind == "view"
        and src_memory_space == "acc"
        and fp_memory_space in {"scaling", "ub", "vec"}
        and dst_memory_space == "gm"
        and src_dtype == "f32"
    )


def dma_pad_for(tile):
    if str(getattr(tile, "pad_value", "Null")).lower() == "null":
        return None
    return (0.0, 0, 0)
