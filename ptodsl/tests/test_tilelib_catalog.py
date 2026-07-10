# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Table-driven selection and render coverage for the PTODSL TileLib catalog."""

import unittest

import ptodsl.tilelib as tilelib
from ptodsl.tilelib import ScalarSpec, ScalarType, TileSpec, VectorSpec, ViewSpec, select


# op -> (template name, rendered op, parameter names, representative dtype[, candidate id])
CATALOG = {
    "pto.tabs": ("template_tabs", "pto.vabs", ("src", "dst"), "f32"),
    "pto.tadd": (
        "template_tadd_2d_no_post_update",
        "pto.vadd",
        ("src0", "src1", "dst"),
        "f32",
        "template_tadd_2d_no_post_update",
    ),
    "pto.tand": ("template_tand", "pto.vand", ("src0", "src1", "dst"), "i32"),
    "pto.tands": ("template_tands", "pto.vand", ("src", "scalar", "dst"), "i32"),
    "pto.tcmp": ("template_tcmp", "pto.vcmp", ("src0", "src1", "dst"), "f32"),
    "pto.tcmps": ("template_tcmps", "pto.vcmps", ("src", "scalar", "dst"), "f32"),
    "pto.tcolexpand": ("template_tcolexpand", "pto.vlds", ("src", "dst"), "f32"),
    "pto.tcolexpandadd": ("template_tcolexpandadd", "pto.vadd", ("src0", "src1", "dst"), "f32"),
    "pto.tcolexpanddiv": ("template_tcolexpanddiv", "pto.vdiv", ("src0", "src1", "dst"), "f32"),
    "pto.tcolexpandexpdif": (
        "template_tcolexpandexpdif_f32",
        "pto.vexpdif",
        ("src0", "src1", "dst"),
        "f32",
    ),
    "pto.tcolexpandmax": ("template_tcolexpandmax", "pto.vmax", ("src0", "src1", "dst"), "f32"),
    "pto.tcolexpandmin": ("template_tcolexpandmin", "pto.vmin", ("src0", "src1", "dst"), "f32"),
    "pto.tcolexpandmul": ("template_tcolexpandmul", "pto.vmul", ("src0", "src1", "dst"), "f32"),
    "pto.tcolexpandsub": ("template_tcolexpandsub", "pto.vsub", ("src0", "src1", "dst"), "f32"),
    "pto.tcolargmax": ("template_tcolargmax_f32_to_i32", "pto.vcmp", ("src", "tmp", "dst"), "f32"),
    "pto.tcolargmin": ("template_tcolargmin_f32_to_i32", "pto.vcmp", ("src", "tmp", "dst"), "f32"),
    "pto.tcolmax": ("template_tcolmax", "pto.vmax", ("src", "dst"), "f32"),
    "pto.tcolmin": ("template_tcolmin", "pto.vmin", ("src", "dst"), "f32"),
    "pto.tcolprod": ("template_tcolprod", "pto.vmul", ("src", "dst"), "f32"),
    "pto.tcolsum": ("template_tcolsum", "pto.vadd", ("src", "dst"), "f32"),
    "pto.texpands": ("template_texpands", "pto.vdup", ("scalar", "dst"), "f32"),
    "pto.textract": ("template_textract_vec2vec_nd", "pto.vlds", ("src", "index_row", "index_col", "dst"), "f32"),
    "pto.textract_fp": (
        "template_textract_fp_f32_f16",
        "pto.mte_l0c_l1",
        ("src", "fp", "index_row", "index_col", "dst"),
        "f32",
        "template_textract_fp_f32_f16",
    ),
    "pto.tlrelu": ("template_tlrelu", "pto.vlrelu", ("src", "slope", "dst"), "f32"),
    "pto.tlog": ("template_tlog", "pto.vln", ("src", "dst"), "f32"),
    "pto.tdiv": ("template_tdiv", "pto.vdiv", ("src0", "src1", "dst"), "f32"),
    "pto.tdivs": (
        "template_tdivs_tile_scalar",
        "pto.vdiv",
        ("src", "scalar", "dst"),
        "f32",
        "template_tdivs_tile_scalar",
    ),
    "pto.tcvt": ("template_tcvt_f32_to_i32", "pto.vcvt", ("src", "dst"), "f32"),
    "pto.texp": ("template_texp", "pto.vexp", ("src", "dst"), "f32"),
    "pto.tfmod": ("template_tfmod", "pto.vtrc", ("src0", "src1", "dst"), "f32"),
    "pto.tfmods": ("template_tfmods", "pto.vtrc", ("src", "scalar", "dst"), "f32"),
    "pto.tfillpad": ("template_tfillpad", "pto.vsts", ("src", "dst"), "f32"),
    "pto.tfillpad_expand": ("template_tfillpad_expand", "pto.vsts", ("src", "dst"), "f32"),
    "pto.tfillpad_inplace": ("template_tfillpad_inplace", "pto.vdup", ("src", "dst"), "f32"),
    "pto.tgemv": ("template_tgemv", "pto.mad", ("lhs", "rhs", "acc"), "f16"),
    "pto.tgemv.acc": ("template_tgemv_acc", "pto.mad_acc", ("acc_in", "lhs", "rhs", "dst"), "f16"),
    "pto.tgemv.bias": ("template_tgemv_bias", "pto.mad_bias", ("lhs", "rhs", "bias", "dst"), "f16"),
    "pto.tgemv.mx": (
        "template_tgemv_mx",
        "pto.mad_mx",
        ("lhs", "lhs_scale", "rhs", "rhs_scale", "dst"),
        "f8e4m3",
    ),
    "pto.tgemv.mx.acc": (
        "template_tgemv_mx_acc",
        "pto.mad_mx_acc",
        ("acc_in", "lhs", "lhs_scale", "rhs", "rhs_scale", "dst"),
        "f8e4m3",
    ),
    "pto.tgemv.mx.bias": (
        "template_tgemv_mx_bias",
        "pto.mad_mx_bias",
        ("lhs", "lhs_scale", "rhs", "rhs_scale", "bias", "dst"),
        "f8e4m3",
    ),
    "pto.tinsert": ("template_tinsert_vec_to_vec_nd_basic", "pto.vsts", ("src", "index_row", "index_col", "dst"), "f32"),
    "pto.tload": (
        "template_tload_nd2nd",
        "pto.mte_gm_ub",
        ("src", "dst"),
        "f32",
        "template_tload_nd2nd",
    ),
    "pto.tmatmul": ("template_tmatmul", "pto.mad", ("lhs", "rhs", "acc"), "f16"),
    "pto.tmatmul.acc": ("template_tmatmul_acc", "pto.mad_acc", ("acc_in", "lhs", "rhs", "dst"), "f16"),
    "pto.tmatmul.bias": ("template_tmatmul_bias", "pto.mad_bias", ("lhs", "rhs", "bias", "dst"), "f16"),
    "pto.tmatmul.mx": (
        "template_tmatmul_mx",
        "pto.mad_mx",
        ("lhs", "lhs_scale", "rhs", "rhs_scale", "dst"),
        "f8e4m3",
    ),
    "pto.tmatmul.mx.acc": (
        "template_tmatmul_mx_acc",
        "pto.mad_mx_acc",
        ("acc_in", "lhs", "lhs_scale", "rhs", "rhs_scale", "dst"),
        "f8e4m3",
    ),
    "pto.tmatmul.mx.bias": (
        "template_tmatmul_mx_bias",
        "pto.mad_mx_bias",
        ("lhs", "lhs_scale", "rhs", "rhs_scale", "bias", "dst"),
        "f8e4m3",
    ),
    "pto.tmax": ("template_tmax", "pto.vmax", ("src0", "src1", "dst"), "f32"),
    "pto.tneg": ("template_tneg", "pto.vneg", ("src", "dst"), "f32"),
    "pto.tmin": ("template_tmin", "pto.vmin", ("src0", "src1", "dst"), "f32"),
    "pto.tmov": ("template_tmov_basic", "pto.vsts", ("src", "dst"), "f32"),
    "pto.tnot": ("template_tnot", "pto.vnot", ("src", "dst"), "i32"),
    "pto.tor": ("template_tor", "pto.vor", ("src0", "src1", "dst"), "i32"),
    "pto.tors": ("template_tors", "pto.vor", ("src", "scalar", "dst"), "i32"),
    "pto.tpartadd": ("template_tpartadd", "pto.vadd", ("src0", "src1", "dst"), "f32"),
    "pto.tpartmax": ("template_tpartmax", "pto.vmax", ("src0", "src1", "dst"), "f32"),
    "pto.tpartmin": ("template_tpartmin", "pto.vmin", ("src0", "src1", "dst"), "f32"),
    "pto.tpartmul": ("template_tpartmul", "pto.vmul", ("src0", "src1", "dst"), "f32"),
    "pto.tprelu": ("template_tprelu", "pto.vprelu", ("src0", "src1", "tmp", "dst"), "f32"),
    "pto.trandom": ("template_trandom", "pto.vmull", ("key0", "key1", "counter0", "counter1", "counter2", "counter3", "dst"), "ui32"),
    "pto.trelu": ("template_trelu", "pto.vrelu", ("src", "dst"), "f32"),
    "pto.trecip": ("template_trecip", "pto.vdiv", ("src", "dst"), "f32"),
    "pto.trem": ("template_trem", "pto.vtrc", ("src0", "src1", "tmp", "dst"), "f32"),
    "pto.trems": ("template_trems", "pto.vtrc", ("src", "scalar", "tmp", "dst"), "f32"),
    "pto.trsqrt": ("template_trsqrt", "pto.vsqrt", ("src", "dst"), "f32"),
    "pto.trowargmax": ("template_trowargmax", "pto.vdintlv", ("src", "tmp", "dst"), "f32"),
    "pto.trowargmin": ("template_trowargmin", "pto.vdintlv", ("src", "tmp", "dst"), "f32"),
    "pto.trowexpand": ("template_trowexpand", "pto.vdup", ("src", "dst"), "f32"),
    "pto.trowexpandadd": ("template_trowexpandadd", "pto.vadd", ("src0", "src1", "dst"), "f32"),
    "pto.trowexpanddiv": ("template_trowexpanddiv", "pto.vdiv", ("src0", "src1", "dst"), "f32"),
    "pto.trowexpandexpdif": (
        "template_trowexpandexpdif_f32",
        "pto.vexpdif",
        ("src0", "src1", "dst"),
        "f32",
    ),
    "pto.trowexpandmax": ("template_trowexpandmax", "pto.vmax", ("src0", "src1", "dst"), "f32"),
    "pto.trowexpandmin": ("template_trowexpandmin", "pto.vmin", ("src0", "src1", "dst"), "f32"),
    "pto.trowexpandmul": ("template_trowexpandmul", "pto.vmul", ("src0", "src1", "dst"), "f32"),
    "pto.trowexpandsub": ("template_trowexpandsub", "pto.vsub", ("src0", "src1", "dst"), "f32"),
    "pto.trowmax": ("template_trowmax", "pto.vcmax", ("src", "tmp", "dst"), "f32"),
    "pto.trowmin": ("template_trowmin", "pto.vcmin", ("src", "tmp", "dst"), "f32"),
    "pto.trowprod": ("template_trowprod", "pto.vintlv", ("src", "tmp", "dst"), "f32"),
    "pto.trowsum": ("template_trowsum", "pto.vcadd", ("src", "tmp", "dst"), "f32"),
    "pto.tsel": ("template_tsel", "pto.vsel", ("mask", "src0", "src1", "tmp", "dst"), "f32"),
    "pto.tsels": ("template_tsels", "pto.vsel", ("mask", "src", "tmp", "scalar", "dst"), "f32"),
    "pto.tshl": ("template_tshl", "pto.vshl", ("src0", "src1", "dst"), "i32"),
    "pto.tshls": ("template_tshls", "pto.vshls", ("src", "scalar", "dst"), "i32"),
    "pto.tshr": ("template_tshr", "pto.vshr", ("src0", "src1", "dst"), "i32"),
    "pto.tshrs": ("template_tshrs", "pto.vshrs", ("src", "scalar", "dst"), "i32"),
    "pto.tmrgsort": ("template_tmrgsort_multi_list2", "pto.vmrgsort4", ("src0", "src1", "tmp", "dst", "ex_vec"), "f32"),
    "pto.tsort32": ("template_tsort32", "pto.vbitsort", ("src", "idx", "dst"), "f32"),
    "pto.tstore": (
        "template_tstore_nd",
        "pto.mte_ub_gm",
        ("src", "dst"),
        "f32",
        "template_tstore_nd",
    ),
    "pto.tstore_fp": (
        "template_tstore_fp_acc_to_gm",
        "pto.mte_l0c_gm",
        ("src", "fp", "dst"),
        "f32",
        "template_tstore_fp_acc_to_gm",
    ),
    "pto.tadds": ("template_tadds", "pto.vadds", ("src", "scalar", "dst"), "f32"),
    "pto.tmaxs": ("template_tmaxs", "pto.vmaxs", ("src", "scalar", "dst"), "f32"),
    "pto.tmins": ("template_tmins", "pto.vmins", ("src", "scalar", "dst"), "f32"),
    "pto.tmuls": ("template_tmuls", "pto.vmuls", ("src", "scalar", "dst"), "f32"),
    "pto.tmul": (
        "template_tmul_2d_no_post_update",
        "pto.vmul",
        ("src0", "src1", "dst"),
        "f32",
        "template_tmul_2d_no_post_update",
    ),
    "pto.txor": (
        "template_txor",
        "pto.vxor",
        ("src0", "src1", "tmp", "dst"),
        "i32",
    ),
    "pto.txors": ("template_txors", "pto.vxor", ("src", "scalar", "tmp", "dst"), "i32"),
    "pto.tsubs": ("template_tsubs", "pto.vadds", ("src", "scalar", "dst"), "f32"),
    "pto.tsub": ("template_tsub", "pto.vsub", ("src0", "src1", "dst"), "f32"),
    "pto.tsqrt": ("template_tsqrt", "pto.vsqrt", ("src", "dst"), "f32"),
}

CUBE_OPS = {
    "pto.tgemv",
    "pto.tgemv.acc",
    "pto.tgemv.bias",
    "pto.tgemv.mx",
    "pto.tgemv.mx.acc",
    "pto.tgemv.mx.bias",
    "pto.tmatmul",
    "pto.tmatmul.acc",
    "pto.tmatmul.bias",
    "pto.tmatmul.mx",
    "pto.tmatmul.mx.acc",
    "pto.tmatmul.mx.bias",
}

COLUMN_REDUCTIONS = {"pto.tcolmax", "pto.tcolmin", "pto.tcolprod", "pto.tcolsum"}
ARG_COLUMN_REDUCTIONS = {"pto.tcolargmax", "pto.tcolargmin"}
ROW_REDUCTIONS = {
    "pto.trowargmax",
    "pto.trowargmin",
    "pto.trowmax",
    "pto.trowmin",
    "pto.trowprod",
    "pto.trowsum",
}
SPECIAL_VALID_SHAPES = {
    ("pto.tcolexpand", "src"): (1, 64),
    ("pto.trowexpand", "src"): (8, 1),
}
for _op in (
    "pto.trowexpandadd",
    "pto.trowexpanddiv",
    "pto.trowexpandexpdif",
    "pto.trowexpandmax",
    "pto.trowexpandmin",
    "pto.trowexpandmul",
    "pto.trowexpandsub",
):
    SPECIAL_VALID_SHAPES[(_op, "src1")] = (8, 1)
for _op in (
    "pto.tcolexpandadd",
    "pto.tcolexpanddiv",
    "pto.tcolexpandexpdif",
    "pto.tcolexpandmax",
    "pto.tcolexpandmin",
    "pto.tcolexpandmul",
    "pto.tcolexpandsub",
):
    SPECIAL_VALID_SHAPES[(_op, "src1")] = (1, 64)
SHARED_RENDERED_OPS = (
    "pto.tile_buf_addr",
    "memref.subview",
    "scf.for",
    "pto.vsts",
    "pto.tilelang.instance",
)
OPS_WITHOUT_TILE_LOAD = {"pto.texpands"}
OPS_WITHOUT_TILE_LOAD = OPS_WITHOUT_TILE_LOAD | {"pto.trandom", "pto.tsort32", "pto.tload", "pto.tstore", "pto.tstore_fp", "pto.textract_fp"}
OPS_WITHOUT_TILE_LOAD = OPS_WITHOUT_TILE_LOAD | {"pto.tfillpad_inplace"}
OPS_WITHOUT_TILE_LOAD = OPS_WITHOUT_TILE_LOAD | CUBE_OPS
OPS_WITHOUT_VECTOR_STORE = {"pto.tcmp", "pto.tcmps", "pto.tsort32"}
OPS_WITHOUT_VECTOR_STORE = OPS_WITHOUT_VECTOR_STORE | {"pto.tload", "pto.tstore", "pto.tstore_fp", "pto.textract_fp"}
OPS_WITHOUT_VECTOR_STORE = OPS_WITHOUT_VECTOR_STORE | CUBE_OPS
OPS_WITHOUT_MEMREF_SUBVIEW = {"pto.tsort32"}
OPS_WITHOUT_MEMREF_SUBVIEW = OPS_WITHOUT_MEMREF_SUBVIEW | {"pto.tload", "pto.tstore", "pto.tstore_fp", "pto.textract_fp"}
OPS_WITHOUT_MEMREF_SUBVIEW = OPS_WITHOUT_MEMREF_SUBVIEW | CUBE_OPS
OPS_WITHOUT_LOOP = {"pto.tmrgsort"}
OPS_WITHOUT_LOOP = OPS_WITHOUT_LOOP | {"pto.tstore_fp", "pto.textract_fp"}
OPS_WITHOUT_LOOP = OPS_WITHOUT_LOOP | CUBE_OPS
OPS_ALLOWING_CASTPTR = {"pto.tsel", "pto.tsels"}
SCALAR_OPERANDS = {
    "scalar",
    "slope",
    "index_row",
    "index_col",
    "key0",
    "key1",
    "counter0",
    "counter1",
    "counter2",
    "counter3",
    "block_len",
}
SPECIAL_SCALAR_DTYPES = {
    ("pto.tshls", "scalar"): "i16",
    ("pto.tshrs", "scalar"): "i16",
    ("pto.textract", "index_row"): "i32",
    ("pto.textract", "index_col"): "i32",
    ("pto.textract_fp", "index_row"): "i32",
    ("pto.textract_fp", "index_col"): "i32",
    ("pto.tinsert", "index_row"): "i32",
    ("pto.tinsert", "index_col"): "i32",
    ("pto.trandom", "key0"): "i32",
    ("pto.trandom", "key1"): "i32",
    ("pto.trandom", "counter0"): "i32",
    ("pto.trandom", "counter1"): "i32",
    ("pto.trandom", "counter2"): "i32",
    ("pto.trandom", "counter3"): "i32",
}
SPECIAL_OPERAND_DTYPES = {
    ("pto.tcmp", "dst"): "i8",
    ("pto.tcmps", "dst"): "ui8",
    ("pto.trandom", "dst"): "ui32",
    ("pto.tsort32", "idx"): "ui32",
    ("pto.textract_fp", "fp"): "f32",
    ("pto.textract_fp", "dst"): "f16",
    ("pto.tstore_fp", "fp"): "f16",
    ("pto.tstore_fp", "dst"): "f16",
    ("pto.trowargmax", "dst"): "i32",
    ("pto.trowargmin", "dst"): "i32",
}
SPECIAL_VECTOR_OPERANDS = {
    ("pto.tmrgsort", "ex_vec"): ("i16", (4,)),
}
for _op in CUBE_OPS:
    SPECIAL_OPERAND_DTYPES[(_op, "acc")] = "f32"
    SPECIAL_OPERAND_DTYPES[(_op, "acc_in")] = "f32"
    SPECIAL_OPERAND_DTYPES[(_op, "bias")] = "f32"
    SPECIAL_OPERAND_DTYPES[(_op, "dst")] = "f32"
    SPECIAL_OPERAND_DTYPES[(_op, "lhs_scale")] = "f16"
    SPECIAL_OPERAND_DTYPES[(_op, "rhs_scale")] = "f16"
for _op in ("pto.tgemv", "pto.tgemv.acc", "pto.tgemv.bias", "pto.tmatmul", "pto.tmatmul.acc", "pto.tmatmul.bias"):
    SPECIAL_OPERAND_DTYPES[(_op, "lhs")] = "f16"
    SPECIAL_OPERAND_DTYPES[(_op, "rhs")] = "f16"
for _op in (
    "pto.tgemv.mx",
    "pto.tgemv.mx.acc",
    "pto.tgemv.mx.bias",
    "pto.tmatmul.mx",
    "pto.tmatmul.mx.acc",
    "pto.tmatmul.mx.bias",
):
    SPECIAL_OPERAND_DTYPES[(_op, "lhs")] = "f8e4m3"
    SPECIAL_OPERAND_DTYPES[(_op, "rhs")] = "f8e4m3"

SPECIAL_MEMORY_SPACES = {}
VIEW_OPERANDS = {
    ("pto.tload", "src"),
    ("pto.tstore", "dst"),
    ("pto.tstore_fp", "dst"),
}
VIEW_SHAPES = {
    ("pto.tload", "src"): (1, 1, 1, 8, 64),
    ("pto.tstore", "dst"): (1, 1, 1, 8, 64),
    ("pto.tstore_fp", "dst"): (1, 1, 1, 8, 64),
}
VIEW_STRIDES = {
    ("pto.tload", "src"): (512, 512, 512, 64, 1),
    ("pto.tstore", "dst"): (512, 512, 512, 64, 1),
    ("pto.tstore_fp", "dst"): (512, 512, 512, 64, 1),
}
for _op in CUBE_OPS:
    SPECIAL_MEMORY_SPACES[(_op, "lhs")] = "left"
    SPECIAL_MEMORY_SPACES[(_op, "rhs")] = "right"
    SPECIAL_MEMORY_SPACES[(_op, "acc")] = "acc"
    SPECIAL_MEMORY_SPACES[(_op, "acc_in")] = "acc"
    SPECIAL_MEMORY_SPACES[(_op, "dst")] = "acc"
    SPECIAL_MEMORY_SPACES[(_op, "bias")] = "bias"
    SPECIAL_MEMORY_SPACES[(_op, "lhs_scale")] = "scaling"
    SPECIAL_MEMORY_SPACES[(_op, "rhs_scale")] = "scaling"
SPECIAL_MEMORY_SPACES[("pto.textract_fp", "src")] = "acc"
SPECIAL_MEMORY_SPACES[("pto.textract_fp", "fp")] = "scaling"
SPECIAL_MEMORY_SPACES[("pto.textract_fp", "dst")] = "mat"
SPECIAL_MEMORY_SPACES[("pto.tstore_fp", "src")] = "acc"
SPECIAL_MEMORY_SPACES[("pto.tstore_fp", "fp")] = "scaling"

for _op in ("pto.tgemv", "pto.tgemv.acc", "pto.tgemv.bias", "pto.tgemv.mx", "pto.tgemv.mx.acc", "pto.tgemv.mx.bias"):
    SPECIAL_VALID_SHAPES[(_op, "lhs")] = (1, 64)
FLOAT_REMAINDER_OPS = {"pto.tfmod", "pto.tfmods", "pto.trem", "pto.trems"}
FLOAT_REMAINDER_DTYPES = {"f16", "bf16", "f32"}


def _entry_parts(entry):
    if len(entry) == 4:
        name, rendered_op, parameter_names, dtype_name = entry
        return name, rendered_op, parameter_names, dtype_name, None
    name, rendered_op, parameter_names, dtype_name, candidate_id = entry
    return name, rendered_op, parameter_names, dtype_name, candidate_id


def _tile_spec_for(op, operand, dtype_name):
    valid_shape = SPECIAL_VALID_SHAPES.get((op, operand), (8, 64))
    return TileSpec(
        shape=(8, 64),
        dtype=ScalarType(dtype_name),
        memory_space=SPECIAL_MEMORY_SPACES.get((op, operand), "ub"),
        valid_shape=valid_shape,
    )


def _view_spec_for(op, operand, dtype_name):
    return ViewSpec(
        shape=VIEW_SHAPES.get((op, operand), (1, 1, 1, 8, 64)),
        dtype=ScalarType(dtype_name),
        memory_space=SPECIAL_MEMORY_SPACES.get((op, operand), "gm"),
        strides=VIEW_STRIDES.get((op, operand), (512, 512, 512, 64, 1)),
    )


def _specs(op, parameter_names, dtype_name):
    specs = {}
    for name in parameter_names:
        if name in SCALAR_OPERANDS:
            scalar_dtype = SPECIAL_SCALAR_DTYPES.get((op, name), dtype_name)
            specs[name] = ScalarSpec(dtype=ScalarType(scalar_dtype), value=1)
            continue
        operand_dtype = SPECIAL_OPERAND_DTYPES.get(
            (op, name),
            "i8" if name == "mask" else dtype_name,
        )
        if (op, name) in VIEW_OPERANDS:
            specs[name] = _view_spec_for(op, name, operand_dtype)
            continue
        if (op, name) in SPECIAL_VECTOR_OPERANDS:
            vector_dtype, vector_shape = SPECIAL_VECTOR_OPERANDS[(op, name)]
            specs[name] = VectorSpec(shape=vector_shape, dtype=ScalarType(vector_dtype))
            continue
        valid_shape = SPECIAL_VALID_SHAPES.get((op, name), (8, 64))
        if op in COLUMN_REDUCTIONS and name == "dst":
            valid_shape = (1, 64)
        if op in ARG_COLUMN_REDUCTIONS and name == "dst":
            operand_dtype = "i32"
            valid_shape = (1, 64)
        if op == "pto.tcvt" and name == "dst":
            operand_dtype = "i32"
        if op in ROW_REDUCTIONS and name == "dst":
            valid_shape = (8, 1)
        specs[name] = _tile_spec_for(op, name, operand_dtype)
        if valid_shape != specs[name].valid_shape:
            specs[name] = TileSpec(
                shape=(8, 64),
                dtype=ScalarType(operand_dtype),
                memory_space=SPECIAL_MEMORY_SPACES.get((op, name), "ub"),
                valid_shape=valid_shape,
            )
    return specs


def _expected_rendered_op(op, signature):
    if op in FLOAT_REMAINDER_OPS and not any(
        dtype in FLOAT_REMAINDER_DTYPES for dtype in signature
    ):
        return "pto.vdiv"
    return CATALOG[op][1]


class TileLibCatalogTest(unittest.TestCase):
    def test_tilelib_does_not_duplicate_the_public_ptodsl_surface(self):
        for name in (
            "Tile",
            "PostUpdate",
            "get_lanes",
            "make_mask",
            "vlds",
            "vadd",
            "vsts",
        ):
            with self.subTest(name=name):
                self.assertFalse(hasattr(tilelib, name))

    def test_each_catalog_entry_selects_and_renders(self):
        for op, entry in CATALOG.items():
            with self.subTest(op=op):
                name, vector_op, parameter_names, dtype_name, candidate_id = _entry_parts(entry)
                specs = _specs(op, parameter_names, dtype_name)
                descriptor = select(op, "a5", specs, candidate_id=candidate_id)
                self.assertEqual(descriptor.name, name)

                mlir = descriptor.specialize(**specs).mlir_text()
                self.assertIn(vector_op, mlir)
                shared_ops = SHARED_RENDERED_OPS
                if op in OPS_WITHOUT_VECTOR_STORE:
                    shared_ops = tuple(
                        shared_op for shared_op in shared_ops
                        if shared_op != "pto.vsts"
                    )
                if op in OPS_WITHOUT_MEMREF_SUBVIEW:
                    shared_ops = tuple(
                        shared_op for shared_op in shared_ops
                        if shared_op != "memref.subview"
                    )
                if op in OPS_WITHOUT_LOOP:
                    shared_ops = tuple(
                        shared_op for shared_op in shared_ops
                        if shared_op != "scf.for"
                    )
                for shared_op in shared_ops:
                    self.assertIn(shared_op, mlir)
                if op not in OPS_WITHOUT_TILE_LOAD:
                    self.assertIn("pto.vlds", mlir)
                if op not in OPS_ALLOWING_CASTPTR:
                    self.assertNotIn("pto.castptr", mlir)

    def test_declared_dtype_signatures_are_selectable(self):
        for op, entry in CATALOG.items():
            _, _, parameter_names, representative_dtype, candidate_id = _entry_parts(entry)
            first_specs = _specs(op, parameter_names, representative_dtype)
            descriptor = select(op, "a5", first_specs, candidate_id=candidate_id)
            for signature in descriptor.metadata.dtypes:
                with self.subTest(op=op, signature=signature):
                    specs = {}
                    for operand, dtype_name in zip(parameter_names, signature):
                        if operand in SCALAR_OPERANDS:
                            specs[operand] = ScalarSpec(
                                dtype=ScalarType(dtype_name),
                                value=1,
                            )
                            continue
                        valid_shape = SPECIAL_VALID_SHAPES.get(
                            (op, operand),
                            (8, 64),
                        )
                        if op in COLUMN_REDUCTIONS and operand == "dst":
                            valid_shape = (1, 64)
                        if op in ARG_COLUMN_REDUCTIONS and operand == "dst":
                            valid_shape = (1, 64)
                        if op in ROW_REDUCTIONS and operand == "dst":
                            valid_shape = (8, 1)
                        if (op, operand) in VIEW_OPERANDS:
                            specs[operand] = _view_spec_for(op, operand, dtype_name)
                            continue
                        specs[operand] = _tile_spec_for(op, operand, dtype_name)
                        if valid_shape != specs[operand].valid_shape:
                            specs[operand] = TileSpec(
                                shape=(8, 64),
                                dtype=ScalarType(dtype_name),
                                memory_space=SPECIAL_MEMORY_SPACES.get((op, operand), "ub"),
                                valid_shape=valid_shape,
                            )
                    selected = select(op, "a5", specs, candidate_id=candidate_id)
                    self.assertEqual(selected.name, descriptor.name)
                    self.assertIn(
                        _expected_rendered_op(op, signature),
                        selected.specialize(**specs).mlir_text(),
                    )

    def test_simple_elementwise_vec_smoke_shapes_render(self):
        cases = (
            ("pto.tabs", ("src", "dst"), "f16", "pto.vabs"),
            ("pto.tand", ("src0", "src1", "dst"), "i32", "pto.vand"),
            ("pto.tands", ("src", "scalar", "dst"), "i16", "pto.vand"),
            ("pto.tnot", ("src", "dst"), "ui8", "pto.vnot"),
            ("pto.tor", ("src0", "src1", "dst"), "i32", "pto.vor"),
            ("pto.tors", ("src", "scalar", "dst"), "i16", "pto.vor"),
            ("pto.tneg", ("src", "dst"), "f16", "pto.vneg"),
            ("pto.tmin", ("src0", "src1", "dst"), "i32", "pto.vmin"),
        )
        for op, parameter_names, dtype_name, expected_op in cases:
            with self.subTest(op=op):
                specs = {}
                for name in parameter_names:
                    if name == "scalar":
                        specs[name] = ScalarSpec(dtype=ScalarType(dtype_name), value=1)
                    else:
                        specs[name] = TileSpec(
                            shape=(8, 64),
                            dtype=ScalarType(dtype_name),
                            memory_space="vec",
                        )
                selected = select("pto.tmin" if op == "pto.tmin" else op, "a5", specs)
                self.assertIn(expected_op, selected.specialize(**specs).mlir_text())

    def test_tdivs_scalar_tile_candidate_renders(self):
        specs = _specs("pto.tdivs", ("scalar", "src", "dst"), "f32")
        selected = select(
            "pto.tdivs",
            "a5",
            specs,
            candidate_id="template_tdivs_scalar_tile",
        )
        self.assertEqual(selected.param_names, ("scalar", "src", "dst"))
        self.assertIn("pto.vdiv", selected.specialize(**specs).mlir_text())

    def test_column_vec_smoke_shapes_render(self):
        cases = (
            (
                "pto.tcolexpand",
                {"src": (1, 128, (1, 63)), "dst": (8, 128, (8, 63))},
                "f32",
                "pto.vsts",
            ),
            (
                "pto.tcolexpandmax",
                {"src0": (10, 64, (10, 64)), "src1": (1, 64, (1, 64)), "dst": (10, 64, (10, 64))},
                "f16",
                "pto.vmax",
            ),
            (
                "pto.tcolexpandmin",
                {"src0": (10, 64, (10, 64)), "src1": (1, 64, (1, 64)), "dst": (10, 64, (10, 64))},
                "f16",
                "pto.vmin",
            ),
            (
                "pto.tcolexpandmul",
                {"src0": (10, 64, (10, 64)), "src1": (1, 64, (1, 64)), "dst": (10, 64, (10, 64))},
                "f16",
                "pto.vmul",
            ),
            (
                "pto.tcolmax",
                {"src": (1, 256, (1, 255)), "dst": (1, 256, (1, 255))},
                "f32",
                "pto.vmax",
            ),
            (
                "pto.tcolmin",
                {"src": (1, 256, (1, 255)), "dst": (1, 256, (1, 255))},
                "f32",
                "pto.vmin",
            ),
            (
                "pto.tlrelu",
                {"src": (32, 64, (32, 64)), "dst": (32, 64, (32, 64))},
                "f32",
                "pto.vlrelu",
            ),
            (
                "pto.tmov",
                {"src": (32, 32, (32, 32)), "dst": (32, 32, (32, 32))},
                "f32",
                "pto.vsts",
            ),
        )
        for op, tile_shapes, dtype_name, expected_op in cases:
            with self.subTest(op=op):
                specs = {}
                for name, (rows, cols, valid_shape) in tile_shapes.items():
                    specs[name] = TileSpec(
                        shape=(rows, cols),
                        dtype=ScalarType(dtype_name),
                        memory_space="vec",
                        valid_shape=valid_shape,
                    )
                if op == "pto.tlrelu":
                    specs["slope"] = ScalarSpec(dtype=ScalarType(dtype_name), value=1)
                selected = select(op, "a5", specs)
                self.assertIn(expected_op, selected.specialize(**specs).mlir_text())

    def test_tcmp_vec_tiles_render_packed_mask_paths(self):
        cases = (
            ("f32", "pto.pdintlv_b8", "PK"),
            ("f16", "pto.pbitcast", "PK"),
            ("i8", "pto.vcmp", "NORM"),
        )
        for dtype_name, expected_op, expected_dist in cases:
            with self.subTest(dtype=dtype_name):
                specs = {
                    "src0": TileSpec(
                        shape=(8, 64),
                        dtype=ScalarType(dtype_name),
                        memory_space="vec",
                    ),
                    "src1": TileSpec(
                        shape=(8, 64),
                        dtype=ScalarType(dtype_name),
                        memory_space="vec",
                    ),
                    "dst": TileSpec(
                        shape=(8, 64),
                        dtype=ScalarType("i8"),
                        memory_space="vec",
                    ),
                }
                selected = select("pto.tcmp", "a5", specs)
                self.assertEqual(selected.name, "template_tcmp")
                mlir = selected.specialize(**specs).mlir_text()
                self.assertIn(expected_op, mlir)
                self.assertIn(expected_dist, mlir)

    def test_tcmps_vec_tiles_render_packed_mask_paths(self):
        cases = (
            ("f32", "pto.pdintlv_b8", "PK"),
            ("f16", "pto.vcmps", "PK"),
            ("i8", "pto.vcmps", "NORM"),
        )
        for dtype_name, expected_op, expected_dist in cases:
            with self.subTest(dtype=dtype_name):
                specs = {
                    "src": TileSpec(
                        shape=(8, 64),
                        dtype=ScalarType(dtype_name),
                        memory_space="vec",
                    ),
                    "scalar": ScalarSpec(dtype=ScalarType(dtype_name), value=1),
                    "dst": TileSpec(
                        shape=(8, 64),
                        dtype=ScalarType("ui8"),
                        memory_space="vec",
                    ),
                }
                selected = select("pto.tcmps", "a5", specs)
                self.assertEqual(selected.name, "template_tcmps")
                mlir = selected.specialize(context_attrs={"cmp_mode": "lt"}, **specs).mlir_text()
                self.assertIn(expected_op, mlir)
                self.assertIn(expected_dist, mlir)
                self.assertIn('"lt"', mlir)

    def test_tlrelu_accepts_f16_tiles_with_f32_slope(self):
        specs = {
            "src": TileSpec(
                shape=(63, 64),
                dtype=ScalarType("f16"),
                memory_space="vec",
                valid_shape=(63, 64),
            ),
            "slope": ScalarSpec(dtype=ScalarType("f32"), value=1.0),
            "dst": TileSpec(
                shape=(63, 128),
                dtype=ScalarType("f16"),
                memory_space="vec",
                valid_shape=(63, 64),
            ),
        }
        selected = select("pto.tlrelu", "a5", specs)
        self.assertEqual(selected.name, "template_tlrelu")
        mlir = selected.specialize(**specs).mlir_text()
        self.assertIn("pto.vlrelu", mlir)

    def test_tmov_accepts_ui8_vec_tiles(self):
        specs = {
            "src": TileSpec(shape=(64, 64), dtype=ScalarType("ui8"), memory_space="vec"),
            "dst": TileSpec(shape=(64, 64), dtype=ScalarType("ui8"), memory_space="vec"),
        }
        selected = select("pto.tmov", "a5", specs)
        self.assertEqual(selected.name, "template_tmov_basic")
        self.assertIn("pto.vsts", selected.specialize(**specs).mlir_text())

    def test_tcvt_additional_rowwise_versions_render(self):
        signatures = {
            ("i32", "f32"): "template_tcvt_i32_to_f32",
            ("i16", "f16"): "template_tcvt_i16_to_f16",
            ("f16", "i16"): "template_tcvt_f16_to_i16",
            ("bf16", "f16"): "template_tcvt_bf16_to_f16",
            ("f32", "f16"): "template_tcvt_f32_to_f16",
            ("f32", "bf16"): "template_tcvt_f32_to_bf16",
            ("f16", "i32"): "template_tcvt_f16_to_i32",
            ("f16", "f32"): "template_tcvt_f16_to_f32",
            ("bf16", "i32"): "template_tcvt_bf16_to_i32",
            ("ui8", "ui16"): "template_tcvt_ui8_to_ui16",
        }
        for (src_dtype, dst_dtype), expected_name in signatures.items():
            with self.subTest(signature=(src_dtype, dst_dtype)):
                specs = {
                    "src": TileSpec(shape=(8, 64), dtype=ScalarType(src_dtype)),
                    "dst": TileSpec(shape=(8, 64), dtype=ScalarType(dst_dtype)),
                }
                selected = select("pto.tcvt", "a5", specs)
                self.assertEqual(selected.name, expected_name)
                self.assertIn("pto.vcvt", selected.specialize(**specs).mlir_text())

    def test_tcolexpanddiv_i32_uses_float_divide_path(self):
        specs = {
            "src0": TileSpec(shape=(8, 64), dtype=ScalarType("i32")),
            "src1": TileSpec(shape=(1, 64), dtype=ScalarType("i32")),
            "dst": TileSpec(shape=(8, 64), dtype=ScalarType("i32")),
        }
        selected = select("pto.tcolexpanddiv", "a5", specs)
        self.assertEqual(selected.name, "template_tcolexpanddiv_i32")
        mlir = selected.specialize(**specs).mlir_text()
        self.assertIn("pto.vcvt", mlir)
        self.assertIn("pto.vdiv", mlir)

    def test_tmrgsort_multi_list3_and_4_render(self):
        cases = (
            (
                ("src0", "src1", "src2", "tmp", "dst", "ex_vec"),
                "template_tmrgsort_multi_list3",
            ),
            (
                ("src0", "src1", "src2", "src3", "tmp", "dst", "ex_vec"),
                "template_tmrgsort_multi_list4",
            ),
        )
        for parameter_names, expected_name in cases:
            with self.subTest(expected_name=expected_name):
                specs = _specs("pto.tmrgsort", parameter_names, "f32")
                selected = select("pto.tmrgsort", "a5", specs)
                self.assertEqual(selected.name, expected_name)
                self.assertIn("pto.vmrgsort4", selected.specialize(**specs).mlir_text())

    def test_tsort32_unaligned_tmp_version_renders(self):
        specs = {
            "src": TileSpec(shape=(8, 64), dtype=ScalarType("f32"), valid_shape=(8, 63)),
            "idx": TileSpec(shape=(8, 64), dtype=ScalarType("i32"), valid_shape=(8, 63)),
            "tmp": TileSpec(shape=(1, 64), dtype=ScalarType("f32"), valid_shape=(1, 64)),
            "dst": TileSpec(shape=(8, 64), dtype=ScalarType("f32"), valid_shape=(8, 63)),
        }
        selected = select("pto.tsort32", "a5", specs)
        self.assertEqual(selected.name, "template_tsort32_with_tmp")
        mlir = selected.specialize(**specs).mlir_text()
        self.assertIn("pto.copy_ubuf_to_ubuf", mlir)
        self.assertIn("pto.vbitsort", mlir)

    def test_colarg_additional_dtype_versions_render(self):
        for op in ("pto.tcolargmax", "pto.tcolargmin"):
            for dtype in ("f16", "ui16", "i8", "ui8"):
                with self.subTest(op=op, dtype=dtype):
                    specs = {
                        "src": TileSpec(shape=(8, 64), dtype=ScalarType(dtype)),
                        "tmp": TileSpec(shape=(8, 64), dtype=ScalarType(dtype)),
                        "dst": TileSpec(shape=(8, 64), dtype=ScalarType("i32"), valid_shape=(1, 64)),
                    }
                    selected = select(op, "a5", specs)
                    self.assertIn(dtype, selected.name)
                    self.assertIn("pto.vcmp", selected.specialize(**specs).mlir_text())

    def test_tload_versions_render(self):
        cases = (
            (
                "template_tload_dn2dn",
                {
                    "src": ViewSpec(
                        shape=(1, 1, 1, 8, 64),
                        dtype=ScalarType("f32"),
                        strides=(512, 512, 512, 1, 8),
                    ),
                    "dst": TileSpec(
                        shape=(8, 64),
                        dtype=ScalarType("f32"),
                        b_layout="col_major",
                    ),
                },
                "pto.mte_gm_ub",
            ),
            (
                "template_tload_nz2nz",
                {
                    "src": ViewSpec(
                        shape=(1, 1, 8, 1, 64),
                        dtype=ScalarType("f32"),
                        strides=(512, 512, 64, 64, 1),
                    ),
                    "dst": TileSpec(
                        shape=(8, 64),
                        dtype=ScalarType("f32"),
                        b_layout="col_major",
                        s_layout="row_major",
                    ),
                },
                "pto.mte_gm_ub",
            ),
            (
                "template_tload_gm_to_mat_nd2nz",
                {
                    "src": ViewSpec(
                        shape=(1, 1, 1, 8, 64),
                        dtype=ScalarType("f16"),
                        strides=(512, 512, 512, 64, 1),
                    ),
                    "dst": TileSpec(
                        shape=(8, 64),
                        dtype=ScalarType("f16"),
                        memory_space="mat",
                        b_layout="col_major",
                        s_layout="row_major",
                    ),
                },
                "pto.mte_gm_l1_frac",
            ),
            (
                "template_tload_gm_to_mat_dn2nz",
                {
                    "src": ViewSpec(
                        shape=(1, 1, 1, 64, 8),
                        dtype=ScalarType("f16"),
                        strides=(512, 512, 512, 1, 64),
                    ),
                    "dst": TileSpec(
                        shape=(8, 64),
                        dtype=ScalarType("f16"),
                        memory_space="mat",
                        b_layout="col_major",
                        s_layout="row_major",
                    ),
                },
                "pto.mte_gm_l1_frac",
            ),
        )
        for expected_name, specs, expected_op in cases:
            with self.subTest(expected_name=expected_name):
                selected = select("pto.tload", "a5", specs)
                self.assertEqual(selected.name, expected_name)
                self.assertIn(expected_op, selected.specialize(**specs).mlir_text())

    def test_tload_accepts_numeric_pad_value_metadata(self):
        specs = {
            "src": ViewSpec(
                shape=(1, 1, 1, 60, 60),
                dtype=ScalarType("i32"),
                strides=(3600, 3600, 3600, 60, 1),
            ),
            "dst": TileSpec(
                shape=(64, 64),
                dtype=ScalarType("i32"),
                memory_space="vec",
                valid_shape=(60, 60),
                pad_value="0x2",
            ),
        }
        selected = select("pto.tload", "a5", specs)
        self.assertEqual(selected.name, "template_tload_nd2nd")
        self.assertIn("pto.mte_gm_ub", selected.specialize(**specs).mlir_text())

    def test_tstore_versions_render(self):
        cases = (
            (
                "template_tstore_dn",
                {
                    "src": TileSpec(
                        shape=(8, 64),
                        dtype=ScalarType("f32"),
                        b_layout="col_major",
                    ),
                    "dst": ViewSpec(
                        shape=(1, 1, 1, 8, 64),
                        dtype=ScalarType("f32"),
                        strides=(512, 512, 512, 1, 8),
                    ),
                },
                "pto.mte_ub_gm",
            ),
            (
                "template_tstore_nz",
                {
                    "src": TileSpec(
                        shape=(8, 64),
                        dtype=ScalarType("f32"),
                        b_layout="col_major",
                        s_layout="row_major",
                    ),
                    "dst": ViewSpec(
                        shape=(1, 1, 8, 1, 64),
                        dtype=ScalarType("f32"),
                        strides=(512, 512, 64, 64, 1),
                    ),
                },
                "pto.mte_ub_gm",
            ),
            (
                "template_tstore_acc_to_gm_nz2nd",
                {
                    "src": TileSpec(
                        shape=(8, 64),
                        dtype=ScalarType("f32"),
                        memory_space="acc",
                    ),
                    "dst": ViewSpec(
                        shape=(1, 1, 1, 8, 64),
                        dtype=ScalarType("f32"),
                        layout="nd",
                    ),
                },
                "pto.mte_l0c_gm",
            ),
            (
                "template_tstore_acc_to_gm_nz2dn",
                {
                    "src": TileSpec(
                        shape=(8, 64),
                        dtype=ScalarType("f32"),
                        memory_space="acc",
                    ),
                    "dst": ViewSpec(
                        shape=(1, 1, 1, 8, 64),
                        dtype=ScalarType("f32"),
                        layout="dn",
                    ),
                },
                "pto.mte_l0c_gm",
            ),
            (
                "template_tstore_acc_to_gm_nz2nz",
                {
                    "src": TileSpec(
                        shape=(8, 64),
                        dtype=ScalarType("f32"),
                        memory_space="acc",
                    ),
                    "dst": ViewSpec(
                        shape=(1, 1, 1, 8, 64),
                        dtype=ScalarType("f32"),
                        layout="nz",
                    ),
                },
                "pto.mte_l0c_gm",
            ),
        )
        for expected_name, specs, expected_op in cases:
            with self.subTest(expected_name=expected_name):
                selected = select("pto.tstore", "a5", specs)
                self.assertEqual(selected.name, expected_name)
                self.assertIn(expected_op, selected.specialize(**specs).mlir_text())

    def test_textract_fp_versions_render(self):
        signatures = {
            ("f32", "f32", "i32", "i32", "si8"): "template_textract_fp_f32_si8",
            ("f32", "f32", "i32", "i32", "ui8"): "template_textract_fp_f32_ui8",
            ("f32", "f32", "i32", "i32", "f16"): "template_textract_fp_f32_f16",
            ("f32", "f32", "i32", "i32", "bf16"): "template_textract_fp_f32_bf16",
            ("f32", "f32", "i32", "i32", "f32"): "template_textract_fp_f32_f32",
            ("i32", "f32", "i32", "i32", "si8"): "template_textract_fp_si32_si8",
            ("i32", "f32", "i32", "i32", "ui8"): "template_textract_fp_si32_ui8",
            ("i32", "f32", "i32", "i32", "f16"): "template_textract_fp_si32_f16",
            ("i32", "f32", "i32", "i32", "bf16"): "template_textract_fp_si32_bf16",
        }
        for signature, expected_name in signatures.items():
            with self.subTest(signature=signature):
                src_dtype, fp_dtype, index_row_dtype, index_col_dtype, dst_dtype = signature
                specs = {
                    "src": TileSpec(
                        shape=(8, 64),
                        dtype=ScalarType(src_dtype),
                        memory_space="acc",
                    ),
                    "fp": TileSpec(
                        shape=(8, 64),
                        dtype=ScalarType(fp_dtype),
                        memory_space="scaling",
                    ),
                    "index_row": ScalarSpec(dtype=ScalarType(index_row_dtype), value=1),
                    "index_col": ScalarSpec(dtype=ScalarType(index_col_dtype), value=1),
                    "dst": TileSpec(
                        shape=(8, 64),
                        dtype=ScalarType(dst_dtype),
                        memory_space="mat",
                    ),
                }
                selected = select("pto.textract_fp", "a5", specs)
                self.assertEqual(selected.name, expected_name)
                self.assertIn("pto.mte_l0c_l1", selected.specialize(**specs).mlir_text())

    def test_tinsert_acc_to_mat_basic_renders(self):
        for dst_dtype in ("f16", "bf16"):
            with self.subTest(dst_dtype=dst_dtype):
                specs = {
                    "src": TileSpec(
                        shape=(16, 16),
                        dtype=ScalarType("f32"),
                        memory_space="acc",
                        valid_shape=(16, 16),
                        b_layout="col_major",
                        s_layout="row_major",
                    ),
                    "index_row": ScalarSpec(dtype=ScalarType("i32"), value=0),
                    "index_col": ScalarSpec(dtype=ScalarType("i32"), value=0),
                    "dst": TileSpec(
                        shape=(16, 16),
                        dtype=ScalarType(dst_dtype),
                        memory_space="mat",
                        valid_shape=(16, 16),
                        b_layout="col_major",
                        s_layout="row_major",
                    ),
                }
                selected = select("pto.tinsert", "a5", specs)
                self.assertEqual(selected.name, "template_tinsert_acc_to_mat_basic")
                self.assertIn("pto.mte_l0c_l1", selected.specialize(**specs).mlir_text())

    def test_tpart_add_mul_partial_source_versions_render(self):
        for op, expected_op in (("pto.tpartadd", "pto.vadd"), ("pto.tpartmul", "pto.vmul")):
            with self.subTest(op=op):
                specs = {
                    "src0": TileSpec(shape=(8, 64), dtype=ScalarType("f32")),
                    "src1": TileSpec(
                        shape=(8, 64),
                        dtype=ScalarType("f32"),
                        valid_shape=(8, 32),
                    ),
                    "dst": TileSpec(shape=(8, 64), dtype=ScalarType("f32")),
                }
                selected = select(op, "a5", specs)
                mlir = selected.specialize(**specs).mlir_text()
                self.assertIn(expected_op, mlir)
                self.assertIn("pto.vlds", mlir)
                self.assertIn("pto.vsts", mlir)

                vec_specs = {
                    name: TileSpec(
                        shape=spec.shape,
                        dtype=spec.dtype,
                        memory_space="vec",
                        valid_shape=spec.valid_shape,
                    )
                    for name, spec in specs.items()
                }
                self.assertEqual(select(op, "a5", vec_specs).name, selected.name)

    def test_tpart_extreme_allows_both_sources_partial(self):
        for op, expected_op, expected_pad in (
            ("pto.tpartmax", "pto.vmax", "0xFF800000"),
            ("pto.tpartmin", "pto.vmin", "0x7F800000"),
        ):
            with self.subTest(op=op):
                specs = {
                    "src0": TileSpec(
                        shape=(8, 64),
                        dtype=ScalarType("f32"),
                        valid_shape=(4, 64),
                    ),
                    "src1": TileSpec(
                        shape=(8, 64),
                        dtype=ScalarType("f32"),
                        valid_shape=(8, 32),
                    ),
                    "dst": TileSpec(shape=(8, 64), dtype=ScalarType("f32")),
                }
                selected = select(op, "a5", specs)
                mlir = selected.specialize(**specs).mlir_text()
                self.assertIn(expected_op, mlir)
                self.assertIn("pto.mem_bar", mlir)
                self.assertIn(expected_pad, mlir)

    def test_trowprod_uses_dtype_specific_reduction_depth(self):
        for dtype_name, expected_stages in (("f32", 6), ("f16", 7)):
            with self.subTest(dtype=dtype_name):
                specs = {
                    "src": TileSpec(
                        shape=(8, 128),
                        dtype=ScalarType(dtype_name),
                        memory_space="vec",
                    ),
                    "tmp": TileSpec(
                        shape=(8, 128),
                        dtype=ScalarType(dtype_name),
                        memory_space="vec",
                    ),
                    "dst": TileSpec(
                        shape=(8, 8),
                        dtype=ScalarType(dtype_name),
                        memory_space="vec",
                        valid_shape=(8, 1),
                    ),
                }
                selected = select("pto.trowprod", "a5", specs)
                mlir = selected.specialize(**specs).mlir_text()
                self.assertEqual(selected.name, "template_trowprod")
                self.assertEqual(mlir.count("pto.vintlv"), expected_stages)


if __name__ == "__main__":
    unittest.main()
