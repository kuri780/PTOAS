# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Tracing-time authored scalar operator lowering for runtime values."""

from __future__ import annotations

from ._types import (
    _integer_signedness,
    _materialize_integer_literal,
    _restore_integer_signedness,
    _signless_integer_type,
    _strip_integer_signedness,
)

from mlir.dialects import arith, math
from mlir.ir import BF16Type, F16Type, F32Type, FloatAttr, IndexType, IntegerType


_FLOAT_BINARY_OPS = {
    "add": arith.AddFOp,
    "sub": arith.SubFOp,
    "mul": arith.MulFOp,
    "truediv": arith.DivFOp,
}


def emit_runtime_binary_op(op_name: str, lhs, rhs):
    """Lower one authored runtime scalar binary operator."""
    lhs, rhs, kind = normalize_runtime_binary_operands(lhs, rhs)
    if kind in {"index", "integer"}:
        op_cls = _integer_binary_op(op_name, lhs.type)
        if op_cls is None:
            raise TypeError(f"runtime scalar operator '{op_name}' is not supported for integer/index values")
        authored_type = lhs.type
        if kind == "integer":
            lhs = _strip_integer_signedness(lhs)
            rhs = _strip_integer_signedness(rhs)
        result = op_cls(lhs, rhs).result
        if kind == "index":
            return result
        return _restore_runtime_integer_result(result, authored_type)
    if kind == "float":
        op_cls = _FLOAT_BINARY_OPS.get(op_name)
        if op_cls is None:
            raise TypeError(f"runtime scalar operator '{op_name}' is not supported for floating-point values")
        return op_cls(lhs, rhs).result
    raise TypeError(f"unsupported runtime scalar operand category '{kind}'")


def emit_runtime_max(lhs, rhs):
    """Lower one authored runtime scalar max operation."""
    lhs, rhs, kind = normalize_runtime_binary_operands(lhs, rhs)
    if kind == "float":
        return arith.MaximumFOp(lhs, rhs).result
    if kind == "integer":
        signedness = _integer_signedness(lhs.type)
        signless_lhs = _strip_integer_signedness(lhs)
        signless_rhs = _strip_integer_signedness(rhs)
        if signedness == "unsigned":
            result = arith.MaxUIOp(signless_lhs, signless_rhs).result
        else:
            result = arith.MaxSIOp(signless_lhs, signless_rhs).result
        return _restore_integer_signedness(result, lhs.type)
    if kind == "index":
        cond = arith.CmpIOp(arith.CmpIPredicate.sge, lhs, rhs).result
        return arith.SelectOp(cond, lhs, rhs).result
    raise TypeError(f"unsupported runtime scalar operand category '{kind}'")


def emit_runtime_min(lhs, rhs):
    """Lower one authored runtime scalar min operation."""
    lhs, rhs, kind = normalize_runtime_binary_operands(lhs, rhs)
    if kind == "float":
        return arith.MinimumFOp(lhs, rhs).result
    if kind == "integer":
        signedness = _integer_signedness(lhs.type)
        signless_lhs = _strip_integer_signedness(lhs)
        signless_rhs = _strip_integer_signedness(rhs)
        if signedness == "unsigned":
            result = arith.MinUIOp(signless_lhs, signless_rhs).result
        else:
            result = arith.MinSIOp(signless_lhs, signless_rhs).result
        return _restore_integer_signedness(result, lhs.type)
    if kind == "index":
        cond = arith.CmpIOp(arith.CmpIPredicate.sle, lhs, rhs).result
        return arith.SelectOp(cond, lhs, rhs).result
    raise TypeError(f"unsupported runtime scalar operand category '{kind}'")


def normalize_runtime_binary_operands(lhs, rhs):
    lhs_is_value = _is_mlir_value(lhs)
    rhs_is_value = _is_mlir_value(rhs)

    if not lhs_is_value and not rhs_is_value:
        raise TypeError("runtime scalar operators require at least one traced runtime operand")

    if lhs_is_value and rhs_is_value:
        return _reconcile_typed_operands(lhs, rhs)

    anchor_type = lhs.type if lhs_is_value else rhs.type
    lhs = lhs if lhs_is_value else _materialize_literal(lhs, anchor_type)
    rhs = rhs if rhs_is_value else _materialize_literal(rhs, anchor_type)
    return _reconcile_typed_operands(lhs, rhs)


def _reconcile_typed_operands(lhs, rhs):
    lhs_type = lhs.type
    rhs_type = rhs.type

    if lhs_type == rhs_type:
        return lhs, rhs, classify_runtime_scalar_type(lhs_type)

    if IndexType.isinstance(lhs_type) and IntegerType.isinstance(rhs_type):
        rhs = arith.IndexCastOp(IndexType.get(), _strip_integer_signedness(rhs)).result
        return lhs, rhs, "index"

    if IntegerType.isinstance(lhs_type) and IndexType.isinstance(rhs_type):
        lhs = arith.IndexCastOp(IndexType.get(), _strip_integer_signedness(lhs)).result
        return lhs, rhs, "index"

    if IntegerType.isinstance(lhs_type) and IntegerType.isinstance(rhs_type):
        lhs_width = IntegerType(lhs_type).width
        rhs_width = IntegerType(rhs_type).width
        target_type = lhs_type if lhs_width >= rhs_width else rhs_type
        lhs = _coerce_runtime_integer_like(lhs, target_type)
        rhs = _coerce_runtime_integer_like(rhs, target_type)
        return lhs, rhs, "integer"

    raise TypeError(
        "runtime scalar operators require matching scalar types or an index/integer pair; "
        f"got {lhs_type} and {rhs_type}"
    )


def _materialize_literal(value, anchor_type):
    if isinstance(value, bool):
        raise TypeError("runtime scalar operators do not accept bool literals")

    kind = classify_runtime_scalar_type(anchor_type)
    if kind == "float":
        return arith.ConstantOp(anchor_type, FloatAttr.get(anchor_type, float(value))).result
    if kind == "index":
        return arith.ConstantOp(anchor_type, int(value)).result

    if isinstance(value, float):
        raise TypeError(
            "runtime scalar operators cannot materialize a floating-point literal "
            f"against non-floating operand type {anchor_type}"
        )

    return _materialize_integer_literal(anchor_type, value)


def _coerce_runtime_integer_like(raw_value, target_type):
    if IndexType.isinstance(raw_value.type):
        signless_target = _signless_integer_type(target_type)
        adapted = arith.IndexCastOp(signless_target, raw_value).result
        return _restore_integer_signedness(adapted, target_type)

    source_type = raw_value.type
    source_width = IntegerType(source_type).width
    target_width = IntegerType(target_type).width
    signless_source = _strip_integer_signedness(raw_value)
    signless_target = _signless_integer_type(target_type)

    if source_width < target_width:
        source_signedness = _integer_signedness(source_type)
        if source_signedness == "unsigned":
            widened = arith.ExtUIOp(signless_target, signless_source).result
        else:
            widened = arith.ExtSIOp(signless_target, signless_source).result
        return _restore_integer_signedness(widened, target_type)
    if source_width > target_width:
        truncated = arith.TruncIOp(signless_target, signless_source).result
        return _restore_integer_signedness(truncated, target_type)
    return _restore_integer_signedness(signless_source, target_type)


def classify_runtime_scalar_type(type_obj):
    if IndexType.isinstance(type_obj):
        return "index"
    if IntegerType.isinstance(type_obj):
        return "integer"
    if any(cls.isinstance(type_obj) for cls in (BF16Type, F16Type, F32Type)):
        return "float"
    raise TypeError(f"runtime scalar operators only support index/int/float values, got {type_obj}")


def _is_mlir_value(value) -> bool:
    return not isinstance(value, (bool, int, float)) and hasattr(value, "type")


def _restore_runtime_integer_result(result, authored_type):
    if IndexType.isinstance(authored_type):
        return result
    if not IntegerType.isinstance(authored_type):
        return result
    return _restore_integer_signedness(result, authored_type)


def emit_runtime_compare(op_name: str, lhs, rhs):
    """Lower one authored runtime scalar comparison operator."""
    lhs, rhs, kind = normalize_runtime_binary_operands(lhs, rhs)

    if kind == "float":
        predicate = {
            "lt": arith.CmpFPredicate.OLT,
            "le": arith.CmpFPredicate.OLE,
            "gt": arith.CmpFPredicate.OGT,
            "ge": arith.CmpFPredicate.OGE,
            "eq": arith.CmpFPredicate.OEQ,
            "ne": arith.CmpFPredicate.ONE,
        }.get(op_name)
        if predicate is None:
            raise TypeError(f"runtime scalar comparison '{op_name}' is not supported for floating-point values")
        return arith.CmpFOp(predicate, lhs, rhs).result

    if kind == "index":
        predicate = {
            "lt": arith.CmpIPredicate.slt,
            "le": arith.CmpIPredicate.sle,
            "gt": arith.CmpIPredicate.sgt,
            "ge": arith.CmpIPredicate.sge,
            "eq": arith.CmpIPredicate.eq,
            "ne": arith.CmpIPredicate.ne,
        }.get(op_name)
        if predicate is None:
            raise TypeError(f"runtime scalar comparison '{op_name}' is not supported for index values")
        return arith.CmpIOp(predicate, lhs, rhs).result

    if kind == "integer":
        signedness = _integer_signedness(lhs.type)
        signed_predicates = {
            "lt": arith.CmpIPredicate.slt,
            "le": arith.CmpIPredicate.sle,
            "gt": arith.CmpIPredicate.sgt,
            "ge": arith.CmpIPredicate.sge,
            "eq": arith.CmpIPredicate.eq,
            "ne": arith.CmpIPredicate.ne,
        }
        unsigned_predicates = {
            "lt": arith.CmpIPredicate.ult,
            "le": arith.CmpIPredicate.ule,
            "gt": arith.CmpIPredicate.ugt,
            "ge": arith.CmpIPredicate.uge,
            "eq": arith.CmpIPredicate.eq,
            "ne": arith.CmpIPredicate.ne,
        }
        predicate = (unsigned_predicates if signedness == "unsigned" else signed_predicates).get(op_name)
        if predicate is None:
            raise TypeError(f"runtime scalar comparison '{op_name}' is not supported for integer values")
        return arith.CmpIOp(predicate, _strip_integer_signedness(lhs), _strip_integer_signedness(rhs)).result

    raise TypeError(f"unsupported runtime scalar operand category '{kind}'")


def emit_runtime_bitwise_op(op_name: str, lhs, rhs):
    """Lower one authored runtime scalar bitwise operator."""
    lhs, rhs, kind = normalize_runtime_binary_operands(lhs, rhs)
    if kind != "integer":
        raise TypeError(
            f"runtime scalar bitwise operator '{op_name}' expects integer-like operands, got {lhs.type} and {rhs.type}"
        )

    op_cls = {
        "and": arith.AndIOp,
        "or": arith.OrIOp,
        "xor": arith.XOrIOp,
    }.get(op_name)
    if op_cls is None:
        raise TypeError(f"unsupported runtime scalar bitwise operator '{op_name}'")

    authored_type = lhs.type
    result = op_cls(_strip_integer_signedness(lhs), _strip_integer_signedness(rhs)).result
    return _restore_integer_signedness(result, authored_type)


def emit_runtime_abs(value):
    """Lower one authored runtime scalar absolute-value operation."""
    kind = classify_runtime_scalar_type(value.type)
    if kind == "float":
        return math.AbsFOp(value).result
    if kind == "index":
        return value
    if kind == "integer":
        signedness = _integer_signedness(value.type)
        if signedness == "unsigned":
            return value
        result = math.AbsIOp(_strip_integer_signedness(value)).result
        return _restore_integer_signedness(result, value.type)
    raise TypeError(f"unsupported runtime scalar operand category '{kind}'")


def _integer_binary_op(op_name: str, authored_type):
    if IndexType.isinstance(authored_type):
        return {
            "add": arith.AddIOp,
            "sub": arith.SubIOp,
            "mul": arith.MulIOp,
            "floordiv": arith.FloorDivSIOp,
            "mod": arith.RemSIOp,
        }.get(op_name)

    signedness = _integer_signedness(authored_type)
    if op_name in {"add", "sub", "mul"}:
        return {
            "add": arith.AddIOp,
            "sub": arith.SubIOp,
            "mul": arith.MulIOp,
        }[op_name]
    if op_name == "floordiv":
        if signedness == "unsigned":
            return arith.DivUIOp
        return arith.FloorDivSIOp
    if op_name == "mod":
        if signedness == "unsigned":
            return arith.RemUIOp
        return arith.RemSIOp
    return None


__all__ = [
    "classify_runtime_scalar_type",
    "emit_runtime_abs",
    "emit_runtime_binary_op",
    "emit_runtime_compare",
    "emit_runtime_bitwise_op",
    "emit_runtime_max",
    "emit_runtime_min",
    "normalize_runtime_binary_operands",
]
