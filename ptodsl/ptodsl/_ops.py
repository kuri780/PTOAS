# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""
PTO operation wrappers.

Every function in this module emits one or more MLIR operations at the
active insertion point and returns the primary SSA result(s).

Design rules:
- Vector math ops infer the result type from the first operand's type.
- ``vlds(tile[row, col:])`` and ``vlds(ptr, offset)`` infer the result
  ``vreg`` type from the source element type. ``vbrc_load`` still requires an
  explicit result ``vreg`` type because broadcast widths are authored
  explicitly in the current surface.
- ``make_tensor_view`` infers the TensorViewType from ``len(shape)`` and the
  pointer's element type.
- ``partition_view`` infers the PartitionTensorViewType from the source type.
"""

from functools import wraps

from ._bootstrap import make_context  # noqa: F401 – ensure MLIR on sys.path
from ._diagnostics import (
    explicit_mode_required_with_context_error,
    make_tensor_view_missing_metadata_error,
    tile_row_alignment_error,
)
from ._host_tensors import resolve_tensor_data_entry
from ._scalar_coercion import coerce_scalar_to_type, materialize_scalar_literal
from ._runtime_scalar_ops import classify_runtime_scalar_type, emit_runtime_binary_op
from ._surface_values import (
    MaskResultValue,
    PartitionTensorViewValue,
    TensorViewValue,
    TileSliceValue,
    TileValue,
    _coerce_index_value,
    _static_index_dims,
    _unwrap_sequence,
    compose_partition_spec,
    emit_as_ptr,
    infer_tile_element_type,
    parse_tile_type_metadata,
    unwrap_surface_value,
    wrap_surface_value,
)
from ._types import (
    _isinstance_pto_type,
    _integer_signedness,
    _materialize_integer_literal,
    _resolve,
    _strip_integer_signedness,
    mask_type,
    part_tensor_view_type,
    part_tensor_view_type_from_dims,
    tensor_view_type,
    tensor_view_type_from_dims,
    vreg_type,
)

from mlir.dialects import arith, pto as _pto
from mlir.ir import (
    Attribute,
    BF16Type,
    F16Type,
    F32Type,
    Float8E4M3FNType,
    Float8E5M2Type,
    FloatAttr,
    IndexType,
    IntegerType,
    MemRefType,
    Type,
)

# Pipe name shorthands → canonical PIPE_* names
_PIPE_ALIASES = {
    "MTE1": "PIPE_MTE1",
    "MTE2": "PIPE_MTE2",
    "MTE3": "PIPE_MTE3",
    "MTE4": "PIPE_MTE4",
    "V":    "PIPE_V",
    "M":    "PIPE_M",
    "S":    "PIPE_S",
    "ALL":  "PIPE_ALL",
}


def _pipe_attr(name: str):
    if not isinstance(name, str):
        return _pto.PipeAttr.get(name)
    canonical = _PIPE_ALIASES.get(name, name)
    if not canonical.startswith("PIPE_"):
        canonical = "PIPE_" + canonical
    return _pto.PipeAttr.get(getattr(_pto.PIPE, canonical))


def _event_attr(event_id: int):
    return getattr(_pto, f"EVENT_ID{event_id}")


def _canonical_pipe_token(pipe):
    if isinstance(pipe, str):
        canonical = _PIPE_ALIASES.get(pipe, pipe)
        if not canonical.startswith("PIPE_"):
            canonical = "PIPE_" + canonical
        return canonical

    for canonical in (
        "PIPE_FIX", "PIPE_MTE1", "PIPE_MTE2", "PIPE_MTE3", "PIPE_MTE4",
        "PIPE_V", "PIPE_M", "PIPE_S", "PIPE_V2", "PIPE_ALL",
    ):
        pipe_attr = getattr(_pto.PIPE, canonical, None)
        if pipe_attr is not None and pipe == pipe_attr:
            return canonical
    return None


def _validate_static_event_id(event_id, *, context: str):
    if isinstance(event_id, int) and not 0 <= event_id <= 7:
        raise ValueError(f"{context} expects static event_id in [0, 7], got {event_id}")


def _validate_sync_pipe(pipe, *, context: str, allowed: tuple[str, ...]):
    canonical = _canonical_pipe_token(pipe)
    if canonical is None:
        raise TypeError(f"{context} expects a concrete Pipe value, got {pipe!r}")
    if canonical not in allowed:
        expected = ", ".join(f"<{name}>" for name in allowed)
        raise ValueError(f"{context} expects pipe to be one of {expected}, got <{canonical}>")


def _require_explicit_mode(surface: str):
    try:
        from ._tracing.active import current_session
        session = current_session()
    except Exception:
        session = None
    if session is None:
        return
    current_mode = getattr(session.module_spec, "mode", None)
    if current_mode != "explicit":
        raise explicit_mode_required_with_context_error(surface, session.module_spec)


def _explicit_mode_only(surface: str):
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            _require_explicit_mode(surface)
            return fn(*args, **kwargs)

        return wrapper

    return decorator


# ── Constants ────────────────────────────────────────────────────────────────

def const(value: int, *, dtype=None):
    """
    Emit an ``arith.constant``.

    ``dtype`` is a ``_DType`` descriptor or a concrete ``mlir.ir.Type``.
    Defaults to ``index`` when omitted.
    """
    from ._types import index as _idx_dtype
    mlir_type = _resolve(dtype) if dtype is not None else _resolve(_idx_dtype)
    if any(cls.isinstance(mlir_type) for cls in (F16Type, BF16Type, F32Type)):
        return wrap_surface_value(arith.ConstantOp(mlir_type, FloatAttr.get(mlir_type, value)).result)
    if IntegerType.isinstance(mlir_type):
        return wrap_surface_value(_materialize_integer_literal(mlir_type, value))
    return wrap_surface_value(arith.ConstantOp(mlir_type, value).result)


# ── Pointer ops ───────────────────────────────────────────────────────────────

def castptr(int_addr, result_ptr_type):
    """``pto.castptr`` – cast an integer address to a typed PTO pointer."""
    return wrap_surface_value(
        _pto.CastPtrOp(_resolve(result_ptr_type), unwrap_surface_value(int_addr)).result
    )


def addptr(base_ptr, index_offset):
    """``pto.addptr`` – advance a pointer by an index offset."""
    return wrap_surface_value(
        _pto.AddPtrOp(
            unwrap_surface_value(base_ptr),
            _coerce_index(index_offset, context="addptr(ptr, offset)"),
        ).result
    )


# ── Vector load / store ───────────────────────────────────────────────────────

_VLOAD_DIST_TOKENS = {
    "NORM",
    "UNPK_B8", "UNPK_B16", "UNPK_B32",
    "BRC_B8", "BRC_B16", "BRC_B32",
    "US_B8", "US_B16",
    "DS_B8", "DS_B16",
}


def vlds(src_ptr, offset=None, result_vreg_type=None, *, dist=None):
    """``pto.vlds`` – vector load from a tile slice or from *src_ptr* at *offset*."""
    if isinstance(src_ptr, TileSliceValue):
        if offset is not None or result_vreg_type is not None:
            raise TypeError("vlds(tile[row, col:]) infers its memref slice and vreg type; do not pass offset/result_vreg_type")
        kwargs = {}
        if dist is not None:
            kwargs["dist"] = _normalize_dist_token(
                dist,
                allowed=_VLOAD_DIST_TOKENS,
                context="vlds(..., dist)",
            )
        return wrap_surface_value(_pto.VldsOp(
            _infer_vreg_type_from_tile_slice(src_ptr),
            unwrap_surface_value(src_ptr),
            _index_zero(),
            **kwargs,
        ).result)

    if offset is None:
        raise TypeError("vlds(ptr, offset, result_vreg_type=None) requires an explicit offset")
    if result_vreg_type is None:
        result_vreg_type = _infer_vreg_type_from_address_source(src_ptr)
    kwargs = {}
    if dist is not None:
        kwargs["dist"] = _normalize_dist_token(
            dist,
            allowed=_VLOAD_DIST_TOKENS,
            context="vlds(..., dist)",
        )
    return wrap_surface_value(_pto.VldsOp(
        _resolve(result_vreg_type),
        unwrap_surface_value(src_ptr),
        unwrap_surface_value(offset),
        **kwargs,
    ).result)


def vldas(source):
    """``pto.vldas`` – prime alignment state for a following unaligned load stream."""
    if isinstance(source, TileSliceValue):
        source = _tile_slice_ptr(source)
    return wrap_surface_value(
        _pto.VldasOp(
            _pto.AlignType.get(),
            unwrap_surface_value(source),
        ).result
    )


def vldus(source, align):
    """``pto.vldus`` – unaligned vector load threaded through alignment state."""
    result_type = (
        _infer_vreg_type_from_tile_slice(source)
        if isinstance(source, TileSliceValue)
        else _infer_vreg_type_from_address_source(source)
    )
    if isinstance(source, TileSliceValue):
        source = _tile_slice_ptr(source)
    op = _pto.VldusOp(
        result_type,
        _pto.AlignType.get(),
        unwrap_surface_value(source),
        unwrap_surface_value(align),
    )
    return wrap_surface_value(op.result), wrap_surface_value(op.updated_align)


_DEINTERLEAVE_DIST_TOKENS = {"DINTLV_B8", "DINTLV_B16", "DINTLV_B32", "BDINTLV"}
_INTERLEAVE_DIST_TOKENS = {"INTLV_B8", "INTLV_B16", "INTLV_B32"}
_VSTORE_DIST_TOKENS = {
    "NORM_B8", "NORM_B16", "NORM_B32",
    "1PT_B8", "1PT_B16", "1PT_B32",
    "PK_B16", "PK_B32", "PK_B64", "PK4_B32",
    "MRG4CHN_B8", "MRG2CHN_B8", "MRG2CHN_B16",
}


def _normalize_dist_token(dist, *, allowed: set[str], context: str):
    token = dist
    if not isinstance(token, str):
        token = str(token)
        if "." in token:
            token = token.rsplit(".", 1)[-1]
    normalized = token.strip().upper()
    if normalized.startswith("_"):
        normalized = normalized[1:]
    if normalized not in allowed:
        expected = ", ".join(sorted(allowed))
        raise ValueError(f"{context} does not support dist {dist!r}; expected one of {expected}")
    return normalized


def vldsx2(source, offset_or_dist, dist=None):
    """``pto.vldsx2`` – dual vector load with deinterleave."""
    if isinstance(source, TileSliceValue):
        if dist is not None:
            raise TypeError("vldsx2(tile[row, col:], dist) does not accept a separate offset argument")
        result_type = _infer_vreg_type_from_tile_slice(source)
        op = _pto.Vldsx2Op(
            result_type,
            result_type,
            unwrap_surface_value(source),
            _index_zero(),
            _normalize_dist_token(
                offset_or_dist,
                allowed=_DEINTERLEAVE_DIST_TOKENS,
                context="vldsx2(..., dist)",
            ),
        )
        return wrap_surface_value(op.low), wrap_surface_value(op.high)

    if dist is None:
        raise TypeError("vldsx2(ptr, offset, dist) requires an explicit offset and dist")
    result_type = _infer_vreg_type_from_address_source(source)
    op = _pto.Vldsx2Op(
        result_type,
        result_type,
        unwrap_surface_value(source),
        _coerce_index(offset_or_dist, context="vldsx2(ptr, offset, dist)"),
        _normalize_dist_token(
            dist,
            allowed=_DEINTERLEAVE_DIST_TOKENS,
            context="vldsx2(..., dist)",
        ),
    )
    return wrap_surface_value(op.low), wrap_surface_value(op.high)


def vbitcast(vector_value, to_dtype):
    """``pto.vbitcast`` – reinterpret one vector register as a different element type."""
    target_elem = _resolve(to_dtype)
    target_type = _resolve(vreg_type(_elements_per_vreg(target_elem), target_elem))
    return wrap_surface_value(
        _pto.VbitcastOp(
            target_type,
            unwrap_surface_value(vector_value),
        ).result
    )


def pbitcast(mask_value, to_type):
    """``pto.pbitcast`` – reinterpret one mask register at a different granularity."""
    return wrap_surface_value(
        _pto.PbitcastOp(
            _resolve(to_type),
            unwrap_surface_value(mask_value),
        ).result
    )


def vsts(val, dst_ptr, offset, mask=None, *, dist=None):
    """``pto.vsts`` – vector store to a tile slice or to *dst_ptr* at *offset*."""
    if isinstance(dst_ptr, TileSliceValue):
        if mask is not None:
            raise TypeError("vsts(vec, tile[row, col:], mask) does not accept a separate offset argument")
        kwargs = {}
        if dist is not None:
            kwargs["dist"] = _normalize_dist_token(
                dist,
                allowed=_VSTORE_DIST_TOKENS,
                context="vsts(..., dist)",
            )
        _pto.VstsOp(
            unwrap_surface_value(val),
            unwrap_surface_value(dst_ptr),
            _index_zero(),
            unwrap_surface_value(offset),
            **kwargs,
        )
        return

    if mask is None:
        raise TypeError("vsts(vec, ptr, offset, mask) requires an explicit mask")
    kwargs = {}
    if dist is not None:
        kwargs["dist"] = _normalize_dist_token(
            dist,
            allowed=_VSTORE_DIST_TOKENS,
            context="vsts(..., dist)",
        )
    _pto.VstsOp(
        unwrap_surface_value(val),
        unwrap_surface_value(dst_ptr),
        unwrap_surface_value(offset),
        unwrap_surface_value(mask),
        **kwargs,
    )


def vstsx2(low, high, dst_ptr, offset_or_dist, dist_or_mask=None, mask=None):
    """``pto.vstsx2`` – dual interleaving vector store."""
    if isinstance(dst_ptr, TileSliceValue):
        if mask is not None:
            raise TypeError("vstsx2(low, high, tile[row, col:], dist, mask) does not accept a separate offset argument")
        _pto.Vstsx2Op(
            unwrap_surface_value(low),
            unwrap_surface_value(high),
            unwrap_surface_value(dst_ptr),
            _index_zero(),
            _normalize_dist_token(
                offset_or_dist,
                allowed=_INTERLEAVE_DIST_TOKENS,
                context="vstsx2(..., dist)",
            ),
            unwrap_surface_value(dist_or_mask),
        )
        return

    if mask is None:
        raise TypeError("vstsx2(low, high, ptr, offset, dist, mask) requires an explicit offset, dist, and mask")
    _pto.Vstsx2Op(
        unwrap_surface_value(low),
        unwrap_surface_value(high),
        unwrap_surface_value(dst_ptr),
        _coerce_index(offset_or_dist, context="vstsx2(ptr, offset, dist, mask)"),
        _normalize_dist_token(
            dist_or_mask,
            allowed=_INTERLEAVE_DIST_TOKENS,
            context="vstsx2(..., dist)",
        ),
        unwrap_surface_value(mask),
    )


def vgather2(buf, offsets, mask, result_vreg_type=None):
    """``pto.vgather2`` – indexed gather from UB."""
    rt = result_vreg_type if result_vreg_type is not None else _infer_vreg_type_from_address_source(buf)
    return wrap_surface_value(
        _pto.Vgather2Op(
            _resolve(rt),
            unwrap_surface_value(buf),
            unwrap_surface_value(offsets),
            unwrap_surface_value(mask),
        ).result
    )


def vgather2_bc(buf, offsets, mask, result_vreg_type=None):
    """``pto.vgather2_bc`` – indexed gather from UB with masked zero-fill."""
    rt = result_vreg_type if result_vreg_type is not None else _infer_vreg_type_from_address_source(buf)
    return wrap_surface_value(
        _pto.Vgather2BcOp(
            _resolve(rt),
            unwrap_surface_value(buf),
            unwrap_surface_value(offsets),
            unwrap_surface_value(mask),
        ).result
    )


def vgatherb(buf, offsets, mask, result_vreg_type=None):
    """``pto.vgatherb`` – block gather from UB using byte offsets."""
    rt = result_vreg_type if result_vreg_type is not None else _infer_vreg_type_from_address_source(buf)
    return wrap_surface_value(
        _pto.VgatherbOp(
            _resolve(rt),
            unwrap_surface_value(buf),
            unwrap_surface_value(offsets),
            unwrap_surface_value(mask),
        ).result
    )


def vscatter(value, destination, offsets, mask):
    """``pto.vscatter`` – indexed scatter to UB."""
    _pto.VscatterOp(
        unwrap_surface_value(value),
        unwrap_surface_value(destination),
        unwrap_surface_value(offsets),
        unwrap_surface_value(mask),
    )


def _coerce_i16(value, *, context: str):
    raw_value = unwrap_surface_value(value)
    i16_type = IntegerType.get_signless(16)
    if isinstance(raw_value, bool):
        raise TypeError(f"{context} does not accept bool values")
    if isinstance(raw_value, int):
        return _materialize_integer_literal(i16_type, raw_value)
    kind = classify_runtime_scalar_type(raw_value.type)
    if kind == "float":
        raise TypeError(f"{context} expects an integer-like scalar, got {raw_value.type}")
    if kind == "index":
        return arith.IndexCastOp(i16_type, raw_value).result
    signless_value = _strip_integer_signedness(raw_value)
    if signless_value.type == i16_type:
        return signless_value
    width = IntegerType(raw_value.type).width
    if width < 16:
        if _integer_signedness(raw_value.type) == "unsigned":
            return arith.ExtUIOp(i16_type, signless_value).result
        return arith.ExtSIOp(i16_type, signless_value).result
    if width > 16:
        return arith.TruncIOp(i16_type, signless_value).result
    return signless_value


def vsldb(source, block_stride, repeat_stride, mask):
    """``pto.vsldb`` – block-strided load."""
    result_type = (
        _infer_vreg_type_from_tile_slice(source)
        if isinstance(source, TileSliceValue)
        else _infer_vreg_type_from_address_source(source)
    )
    return wrap_surface_value(
        _pto.VsldbOp(
            result_type,
            unwrap_surface_value(source),
            _coerce_i16(block_stride, context="vsldb(..., block_stride, repeat_stride, mask)"),
            _coerce_i16(repeat_stride, context="vsldb(..., block_stride, repeat_stride, mask)"),
            unwrap_surface_value(mask),
        ).result
    )


def vsstb(value, destination, block_stride, repeat_stride, mask):
    """``pto.vsstb`` – block-strided store."""
    _pto.VsstbOp(
        unwrap_surface_value(value),
        unwrap_surface_value(destination),
        _coerce_i16(block_stride, context="vsstb(..., block_stride, repeat_stride, mask)"),
        _coerce_i16(repeat_stride, context="vsstb(..., block_stride, repeat_stride, mask)"),
        unwrap_surface_value(mask),
    )


# ── Mask / predicate ops ──────────────────────────────────────────────────────

_MASK_PATTERN_TOKENS = {
    "PAT_ALL",
    "PAT_ALLF",
    "PAT_H",
    "PAT_Q",
    "PAT_M3",
    "PAT_M4",
    *(f"PAT_VL{count}" for count in range(1, 129)),
}

_CMP_MODE_TOKENS = {"eq", "ne", "lt", "le", "gt", "ge"}
_PREDICATE_PART_TOKENS = {"LOWER", "HIGHER"}
_PREDICATE_LOAD_DIST_TOKENS = {"NORM", "US", "DS"}
_PREDICATE_STORE_DIST_TOKENS = {"NORM", "PK"}
_POST_UPDATE_TOKENS = {"NO_POST_UPDATE", "POST_UPDATE"}


def _normalize_mask_pattern(pattern):
    token = pattern
    if not isinstance(token, str):
        token = str(token)
        if "." in token:
            token = token.rsplit(".", 1)[-1]
    token = token.strip().upper()
    normalized = token if token.startswith("PAT_") else f"PAT_{token}"
    if normalized not in _MASK_PATTERN_TOKENS:
        raise ValueError(
            f"unsupported mask pattern {pattern!r}; expected one of PAT_ALL, PAT_ALLF, "
            "PAT_H, PAT_Q, PAT_VL1..PAT_VL128, PAT_M3, PAT_M4"
        )
    return normalized


def _normalize_cmp_mode(cmp_mode):
    token = cmp_mode
    if not isinstance(token, str):
        token = str(token)
        if "." in token:
            token = token.rsplit(".", 1)[-1]
    normalized = token.strip().lower()
    if normalized not in _CMP_MODE_TOKENS:
        raise ValueError(
            f"unsupported cmp_mode {cmp_mode!r}; expected one of EQ, NE, LT, LE, GT, GE"
        )
    return normalized


def _cmp_mode_attr(cmp_mode):
    return Attribute.parse(f"#pto<cmp {_normalize_cmp_mode(cmp_mode)}>")


def _normalize_predicate_part(part):
    token = part
    if not isinstance(token, str):
        token = str(token)
        if "." in token:
            token = token.rsplit(".", 1)[-1]
    normalized = token.strip().upper()
    if normalized not in _PREDICATE_PART_TOKENS:
        raise ValueError(f"unsupported predicate part {part!r}; expected LOWER or HIGHER")
    return normalized


def _normalize_predicate_dist(dist, *, allowed: set[str], context: str):
    token = dist
    if not isinstance(token, str):
        token = str(token)
        if "." in token:
            token = token.rsplit(".", 1)[-1]
    normalized = token.strip().upper()
    if normalized not in allowed:
        expected = ", ".join(sorted(allowed))
        raise ValueError(f"{context} does not support dist {dist!r}; expected one of {expected}")
    return normalized


def _normalize_post_update_mode(mode, *, context: str):
    token = mode
    if not isinstance(token, str):
        token = str(token)
        if "." in token:
            token = token.rsplit(".", 1)[-1]
    normalized = token.strip().upper()
    if normalized in {"OFF", "NO_POST_UPDATE"}:
        return "NO_POST_UPDATE"
    if normalized in {"ON", "POST_UPDATE"}:
        return "POST_UPDATE"
    expected = ", ".join(sorted(_POST_UPDATE_TOKENS))
    raise ValueError(f"{context} does not support mode {mode!r}; expected one of ON/OFF ({expected})")


def _mask_type_from_bits(mask_bits: int):
    return _resolve(mask_type(f"b{mask_bits}"))


def _resolve_mask_result_type(result_type, *, context: str):
    if result_type is None:
        return None
    resolved = _resolve(result_type)
    try:
        _pto.MaskType(resolved)
    except Exception as exc:
        raise TypeError(f"{context} expects to_type to resolve to a PTO mask type, got {resolved}") from exc
    return resolved


def _infer_mask_metadata(mask_value, *, context: str):
    raw_type = unwrap_surface_value(mask_value).type
    try:
        mask_ty = _pto.MaskType(raw_type)
    except Exception as exc:
        raise TypeError(f"{context} expects a PTO mask value, got {raw_type}") from exc
    granularity = mask_ty.granularity
    return int(granularity[1:]), raw_type


def _require_same_mask_types(values, *, context: str):
    raw_types = [unwrap_surface_value(value).type for value in values]
    first = raw_types[0]
    for other in raw_types[1:]:
        if other != first:
            raise TypeError(f"{context} expects masks of the same granularity, got {first} and {other}")
    return first


def _pointer_element_type(ptr_value, *, context: str):
    raw_type = unwrap_surface_value(ptr_value).type
    try:
        return _pto.PtrType(raw_type).element_type
    except Exception:
        try:
            return MemRefType(raw_type).element_type
        except Exception as exc:
            raise TypeError(f"{context} expects a PTO pointer or memref-backed address, got {raw_type}") from exc


def _coerce_index(value, *, context: str):
    raw_value = unwrap_surface_value(value)
    index_type = IndexType.get()
    if isinstance(raw_value, bool):
        raise TypeError(f"{context} does not accept bool values")
    if isinstance(raw_value, int):
        return arith.ConstantOp(index_type, raw_value).result
    kind = classify_runtime_scalar_type(raw_value.type)
    if kind == "float":
        raise TypeError(f"{context} expects an index-like scalar, got {raw_value.type}")
    if IndexType.isinstance(raw_value.type):
        return raw_value
    if IntegerType.isinstance(raw_value.type):
        return arith.IndexCastOp(index_type, _strip_integer_signedness(raw_value)).result
    raise TypeError(f"{context} expects an index-like scalar, got {raw_value.type}")


def init_align():
    """``pto.init_align`` – materialize the initial alignment state."""
    return wrap_surface_value(_pto.InitAlignOp(_pto.AlignType.get()).result)


def _plt_impl(mask_bits: int, scalar):
    plt_op = _plt_op_for_mask_bits(mask_bits)(
        _mask_type_from_bits(mask_bits),
        IntegerType.get_signless(32),
        _coerce_i32(scalar, context=f"plt_b{mask_bits}(scalar)"),
    )
    return wrap_surface_value(plt_op.mask), wrap_surface_value(plt_op.scalar_out)


def plt_b8(scalar):
    """``pto.plt_b8`` – predicate-load from a 32-bit scalar into a b8 mask."""
    return _plt_impl(8, scalar)


def plt_b16(scalar):
    """``pto.plt_b16`` – predicate-load from a 32-bit scalar into a b16 mask."""
    return _plt_impl(16, scalar)


def plt_b32(scalar):
    """
    ``pto.plt_b32`` – predicate-load from a 32-bit scalar.

    Returns ``(mask_value, scalar_out)``.  ``scalar_out`` is often unused
    and can be discarded with ``_``.
    """
    return _plt_impl(32, scalar)


def _pset_impl(mask_bits: int, pattern):
    return wrap_surface_value(
        _pset_op_for_mask_bits(mask_bits)(
            _mask_type_from_bits(mask_bits),
            _normalize_mask_pattern(pattern),
        ).result
    )


def pset_b8(pattern):
    """``pto.pset_b8(pattern)`` → ``!pto.mask<b8>``."""
    return _pset_impl(8, pattern)


def pset_b16(pattern):
    """``pto.pset_b16(pattern)`` → ``!pto.mask<b16>``."""
    return _pset_impl(16, pattern)


def pset_b32(pattern):
    """``pto.pset_b32(pattern)`` → ``!pto.mask<b32>``."""
    return _pset_impl(32, pattern)


def _pge_op_for_mask_bits(mask_bits: int):
    return {
        8: _pto.PgeB8Op,
        16: _pto.PgeB16Op,
        32: _pto.PgeB32Op,
    }[mask_bits]


def _pge_impl(mask_bits: int, pattern):
    return wrap_surface_value(
        _pge_op_for_mask_bits(mask_bits)(
            _mask_type_from_bits(mask_bits),
            _normalize_mask_pattern(pattern),
        ).result
    )


def pge_b8(pattern):
    """``pto.pge_b8(pattern)`` → ``!pto.mask<b8>``."""
    return _pge_impl(8, pattern)


def pge_b16(pattern):
    """``pto.pge_b16(pattern)`` → ``!pto.mask<b16>``."""
    return _pge_impl(16, pattern)


def pge_b32(pattern):
    """``pto.pge_b32(pattern)`` → ``!pto.mask<b32>``."""
    return _pge_impl(32, pattern)


def pand(src0, src1, mask):
    """``pto.pand`` – gated mask AND."""
    result_type = _require_same_mask_types((src0, src1, mask), context="pand(src0, src1, mask)")
    return wrap_surface_value(
        _pto.PandOp(
            result_type,
            unwrap_surface_value(src0),
            unwrap_surface_value(src1),
            unwrap_surface_value(mask),
        ).result
    )


def por(src0, src1, mask):
    """``pto.por`` – gated mask OR."""
    result_type = _require_same_mask_types((src0, src1, mask), context="por(src0, src1, mask)")
    return wrap_surface_value(
        _pto.PorOp(
            result_type,
            unwrap_surface_value(src0),
            unwrap_surface_value(src1),
            unwrap_surface_value(mask),
        ).result
    )


def pxor(src0, src1, mask):
    """``pto.pxor`` – gated mask XOR."""
    result_type = _require_same_mask_types((src0, src1, mask), context="pxor(src0, src1, mask)")
    return wrap_surface_value(
        _pto.PxorOp(
            result_type,
            unwrap_surface_value(src0),
            unwrap_surface_value(src1),
            unwrap_surface_value(mask),
        ).result
    )


def pnot(src, mask):
    """``pto.pnot`` – gated mask NOT."""
    result_type = _require_same_mask_types((src, mask), context="pnot(src, mask)")
    return wrap_surface_value(
        _pto.PnotOp(
            result_type,
            unwrap_surface_value(src),
            unwrap_surface_value(mask),
        ).result
    )


def psel(src0, src1, sel):
    """``pto.psel`` – per-lane mask select."""
    result_type = _require_same_mask_types((src0, src1, sel), context="psel(src0, src1, sel)")
    return wrap_surface_value(
        _pto.PselOp(
            result_type,
            unwrap_surface_value(src0),
            unwrap_surface_value(src1),
            unwrap_surface_value(sel),
        ).result
    )


def ppack(mask_value, part, to_type=None):
    """``pto.ppack`` – pack predicate bits into the selected half."""
    _, inferred_type = _infer_mask_metadata(mask_value, context="ppack(mask, part)")
    result_type = _resolve_mask_result_type(to_type, context="ppack(mask, part, to_type=...)")
    if result_type is None:
        result_type = inferred_type
    return wrap_surface_value(
        _pto.PpackOp(
            result_type,
            unwrap_surface_value(mask_value),
            _normalize_predicate_part(part),
        ).result
    )


def punpack(mask_value, part, to_type=None):
    """``pto.punpack`` – unpack predicate bits from the selected half."""
    _, inferred_type = _infer_mask_metadata(mask_value, context="punpack(mask, part)")
    result_type = _resolve_mask_result_type(to_type, context="punpack(mask, part, to_type=...)")
    if result_type is None:
        result_type = inferred_type
    return wrap_surface_value(
        _pto.PunpackOp(
            result_type,
            unwrap_surface_value(mask_value),
            _normalize_predicate_part(part),
        ).result
    )


def _pintlv_op_for_mask_bits(mask_bits: int):
    return {
        8: _pto.PintlvB8Op,
        16: _pto.PintlvB16Op,
        32: _pto.PintlvB32Op,
    }[mask_bits]


def _pdintlv_op_for_mask_bits(mask_bits: int):
    return {
        8: _pto.PdintlvB8Op,
        16: _pto.PdintlvB16Op,
        32: _pto.PdintlvB32Op,
    }[mask_bits]


def _mask_pair_op(op_resolver, lhs, rhs, *, expected_mask_bits: int, context: str):
    mask_bits, result_type = _infer_mask_metadata(lhs, context=context)
    if mask_bits != expected_mask_bits:
        raise TypeError(f"{context} expects mask_b{expected_mask_bits} operands, got mask_b{mask_bits}")
    _require_same_mask_types((lhs, rhs), context=context)
    op = op_resolver(mask_bits)(
        result_type,
        result_type,
        unwrap_surface_value(lhs),
        unwrap_surface_value(rhs),
    )
    return wrap_surface_value(op.low), wrap_surface_value(op.high)


def pintlv_b8(lhs, rhs):
    """``pto.pintlv_b8`` – interleave two b8 masks."""
    return _mask_pair_op(
        _pintlv_op_for_mask_bits,
        lhs,
        rhs,
        expected_mask_bits=8,
        context="pintlv_b8(lhs, rhs)",
    )


def pintlv_b16(lhs, rhs):
    """``pto.pintlv_b16`` – interleave two b16 masks."""
    return _mask_pair_op(
        _pintlv_op_for_mask_bits,
        lhs,
        rhs,
        expected_mask_bits=16,
        context="pintlv_b16(lhs, rhs)",
    )


def pintlv_b32(lhs, rhs):
    """``pto.pintlv_b32`` – interleave two b32 masks."""
    return _mask_pair_op(
        _pintlv_op_for_mask_bits,
        lhs,
        rhs,
        expected_mask_bits=32,
        context="pintlv_b32(lhs, rhs)",
    )


def pdintlv_b8(lhs, rhs):
    """``pto.pdintlv_b8`` – deinterleave two b8 masks."""
    return _mask_pair_op(
        _pdintlv_op_for_mask_bits,
        lhs,
        rhs,
        expected_mask_bits=8,
        context="pdintlv_b8(lhs, rhs)",
    )


def pdintlv_b16(lhs, rhs):
    """``pto.pdintlv_b16`` – deinterleave two b16 masks."""
    return _mask_pair_op(
        _pdintlv_op_for_mask_bits,
        lhs,
        rhs,
        expected_mask_bits=16,
        context="pdintlv_b16(lhs, rhs)",
    )


def pdintlv_b32(lhs, rhs):
    """``pto.pdintlv_b32`` – deinterleave two b32 masks."""
    return _mask_pair_op(
        _pdintlv_op_for_mask_bits,
        lhs,
        rhs,
        expected_mask_bits=32,
        context="pdintlv_b32(lhs, rhs)",
    )


def vcmp(src0, src1, seed_mask, cmp_mode):
    """``pto.vcmp`` – vector/vector comparison producing a predicate mask."""
    _, elem_type = _infer_vreg_metadata(src0)
    result_type = _mask_type_from_bits(_mask_bits_for_dtype(elem_type))
    seed_type = unwrap_surface_value(seed_mask).type
    if seed_type != result_type:
        raise TypeError(
            f"vcmp(src0, src1, seed_mask, cmp_mode) expects seed_mask {result_type}, got {seed_type}"
        )
    return wrap_surface_value(
        _pto.VcmpOp(
            result_type,
            unwrap_surface_value(src0),
            unwrap_surface_value(src1),
            unwrap_surface_value(seed_mask),
            _normalize_cmp_mode(cmp_mode),
        ).result
    )


def vcmps(src, scalar, seed_mask, cmp_mode):
    """``pto.vcmps`` – vector/scalar comparison producing a predicate mask."""
    _, elem_type = _infer_vreg_metadata(src)
    result_type = _mask_type_from_bits(_mask_bits_for_dtype(elem_type))
    seed_type = unwrap_surface_value(seed_mask).type
    if seed_type != result_type:
        raise TypeError(
            f"vcmps(src, scalar, seed_mask, cmp_mode) expects seed_mask {result_type}, got {seed_type}"
        )
    scalar_value = _coerce_scalar_like_vector_element(src, scalar, context="vcmps")
    return wrap_surface_value(
        _pto.VcmpsOp(
            result_type,
            unwrap_surface_value(src),
            unwrap_surface_value(scalar_value),
            unwrap_surface_value(seed_mask),
            _normalize_cmp_mode(cmp_mode),
        ).result
    )


def plds(buf, offset, *, dist="NORM"):
    """``pto.plds`` – load a predicate mask from UB memory."""
    elem_type = _pointer_element_type(buf, context="plds(buf, offset)")
    result_type = _mask_type_from_bits(_mask_bits_for_dtype(elem_type))
    return wrap_surface_value(
        _pto.PldsOp(
            result_type,
            unwrap_surface_value(buf),
            _coerce_index(offset, context="plds(buf, offset)"),
            _normalize_predicate_dist(
                dist,
                allowed=_PREDICATE_LOAD_DIST_TOKENS,
                context="plds(..., dist)",
            ),
        ).result
    )


def psts(mask_value, buf, offset, *, dist="NORM"):
    """``pto.psts`` – store a predicate mask to UB memory."""
    _infer_mask_metadata(mask_value, context="psts(mask, buf, offset)")
    _pto.PstsOp(
        unwrap_surface_value(mask_value),
        unwrap_surface_value(buf),
        _coerce_index(offset, context="psts(mask, buf, offset)"),
        _normalize_predicate_dist(
            dist,
            allowed=_PREDICATE_STORE_DIST_TOKENS,
            context="psts(..., dist)",
        ),
    )


def pstu(align_in, mask_value, buf):
    """``pto.pstu`` – unaligned predicate store with threaded alignment state."""
    mask_bits, _ = _infer_mask_metadata(mask_value, context="pstu(align_in, mask, buf)")
    if mask_bits not in {16, 32}:
        raise TypeError("pstu(align_in, mask, buf) currently supports only mask_b16 and mask_b32")
    elem_type = _pointer_element_type(buf, context="pstu(align_in, mask, buf)")
    expected_bytes = mask_bits // 8
    actual_bytes = _element_bytewidth(elem_type)
    if actual_bytes != expected_bytes:
        raise TypeError(
            f"pstu(align_in, mask, buf) expects a {expected_bytes}-byte pointer element for mask_b{mask_bits}, "
            f"got {elem_type}"
        )
    align_type = _pto.AlignType.get()
    base_type = unwrap_surface_value(buf).type
    op = _pto.PstuOp(
        align_type,
        base_type,
        unwrap_surface_value(align_in),
        unwrap_surface_value(mask_value),
        unwrap_surface_value(buf),
    )
    return wrap_surface_value(op.align_out), wrap_surface_value(op.base_out)


def vstar(align, destination):
    """``pto.vstar`` – flush alignment-buffered tail bytes to the destination base."""
    _pto.VstarOp(
        unwrap_surface_value(align),
        unwrap_surface_value(destination),
    )


def vstas(align, destination, offset):
    """``pto.vstas`` – flush alignment-buffered tail bytes with an explicit offset."""
    _pto.VstasOp(
        unwrap_surface_value(align),
        unwrap_surface_value(destination),
        _coerce_i32(offset, context="vstas(align, destination, offset)"),
    )


def vstur(align_in, value, base, mode="NO_POST_UPDATE"):
    """``pto.vstur`` – unaligned vector store that updates only alignment state."""
    return wrap_surface_value(
        _pto.VsturOp(
            _pto.AlignType.get(),
            unwrap_surface_value(align_in),
            unwrap_surface_value(value),
            unwrap_surface_value(base),
            _normalize_post_update_mode(mode, context="vstur(..., mode)"),
        ).align_out
    )


def vstus(align_in, offset, value, base):
    """``pto.vstus`` – scalar-offset unaligned vector store that updates alignment state."""
    return wrap_surface_value(
        _pto.VstusOp(
            _pto.AlignType.get(),
            unwrap_surface_value(align_in),
            _coerce_i32(offset, context="vstus(align, offset, value, base)"),
            unwrap_surface_value(value),
            unwrap_surface_value(base),
        ).align_out
    )


# ── Vector math (result type inferred from first operand) ─────────────────────

def vbr(value):
    """``pto.vbr`` – broadcast one scalar value to all vector lanes."""
    raw_value = unwrap_surface_value(value)
    if isinstance(raw_value, bool):
        raise TypeError("vbr(value) does not accept bool values")

    if hasattr(raw_value, "type"):
        scalar_kind = classify_runtime_scalar_type(raw_value.type)
        if scalar_kind == "index":
            raise TypeError("vbr(value) does not support index scalars")
        scalar_value = raw_value
        elem_type = raw_value.type
    else:
        if isinstance(raw_value, float):
            elem_type = F32Type.get()
        elif isinstance(raw_value, int):
            elem_type = IntegerType.get_signless(32)
        else:
            raise TypeError("vbr(value) expects a runtime scalar or one Python int/float literal")
        scalar_value = materialize_scalar_literal(raw_value, elem_type, context="vbr(value)")

    try:
        result_type = _resolve(vreg_type(_elements_per_vreg(elem_type), elem_type))
    except TypeError as exc:
        raise TypeError(f"vbr(value) does not support scalar type {elem_type}") from exc

    return wrap_surface_value(_pto.VbrOp(result_type, scalar_value).result)


def _emit_unary_vec_op(op_ctor, inp, mask):
    return wrap_surface_value(
        op_ctor(
            unwrap_surface_value(inp).type,
            unwrap_surface_value(inp),
            unwrap_surface_value(mask),
        ).result
    )


def _emit_binary_vec_op(op_ctor, lhs, rhs, mask):
    return wrap_surface_value(
        op_ctor(
            unwrap_surface_value(lhs).type,
            unwrap_surface_value(lhs),
            unwrap_surface_value(rhs),
            unwrap_surface_value(mask),
        ).result
    )


def _emit_vec_scalar_masked_op(op_ctor, inp, scalar, mask, *, context: str):
    scalar_value = _coerce_scalar_like_vector_element(inp, scalar, context=context)
    return wrap_surface_value(
        op_ctor(
            unwrap_surface_value(inp).type,
            unwrap_surface_value(inp),
            unwrap_surface_value(scalar_value),
            unwrap_surface_value(mask),
        ).result
    )


def vadd(lhs, rhs, mask, result_type=None):
    """``pto.vadd`` – element-wise add."""
    rt = result_type if result_type is not None else lhs.type
    return wrap_surface_value(
        _pto.VaddOp(
            _resolve(rt),
            unwrap_surface_value(lhs),
            unwrap_surface_value(rhs),
            unwrap_surface_value(mask),
        ).result
    )


def vsub(lhs, rhs, mask):
    """``pto.vsub`` – element-wise subtract."""
    return _emit_binary_vec_op(_pto.VsubOp, lhs, rhs, mask)


def vmul(lhs, rhs, mask):
    """``pto.vmul`` – element-wise multiply."""
    return _emit_binary_vec_op(_pto.VmulOp, lhs, rhs, mask)


def vmax(lhs, rhs, mask):
    """``pto.vmax`` – element-wise maximum."""
    return _emit_binary_vec_op(_pto.VmaxOp, lhs, rhs, mask)


def vmin(lhs, rhs, mask):
    """``pto.vmin`` – element-wise minimum."""
    return _emit_binary_vec_op(_pto.VminOp, lhs, rhs, mask)


def vand(lhs, rhs, mask):
    """``pto.vand`` – element-wise bitwise and."""
    return _emit_binary_vec_op(_pto.VandOp, lhs, rhs, mask)


def vor(lhs, rhs, mask):
    """``pto.vor`` – element-wise bitwise or."""
    return _emit_binary_vec_op(_pto.VorOp, lhs, rhs, mask)


def vxor(lhs, rhs, mask):
    """``pto.vxor`` – element-wise bitwise xor."""
    return _emit_binary_vec_op(_pto.VxorOp, lhs, rhs, mask)


def vdiv(lhs, rhs, mask):
    """``pto.vdiv`` – element-wise divide."""
    return _emit_binary_vec_op(_pto.VdivOp, lhs, rhs, mask)


def vshl(lhs, rhs, mask):
    """``pto.vshl`` – element-wise shift left."""
    return _emit_binary_vec_op(_pto.VshlOp, lhs, rhs, mask)


def vshr(lhs, rhs, mask):
    """``pto.vshr`` – element-wise shift right."""
    return _emit_binary_vec_op(_pto.VshrOp, lhs, rhs, mask)


def vcmax(v, mask):
    """``pto.vcmax`` – cross-lane maximum reduction."""
    return _emit_unary_vec_op(_pto.VcmaxOp, v, mask)


def vcadd(v, mask):
    """``pto.vcadd`` – cross-lane add (sum reduction)."""
    return _emit_unary_vec_op(_pto.VcaddOp, v, mask)


def vcmin(v, mask):
    """``pto.vcmin`` – cross-lane minimum reduction."""
    return _emit_unary_vec_op(_pto.VcminOp, v, mask)


def vdup(v, mask, *, position=None):
    """``pto.vdup`` – duplicate a lane value into all lanes.

    Pass ``position="LOWEST"`` to broadcast the lowest (lane-0) element.
    """
    return wrap_surface_value(
        _pto.VdupOp(
            unwrap_surface_value(v).type,
            unwrap_surface_value(v),
            unwrap_surface_value(mask),
            position=position,
        ).result
    )


def vln(inp, mask):
    """``pto.vln`` – element-wise natural logarithm."""
    return _emit_unary_vec_op(_pto.VlnOp, inp, mask)


def vsqrt(inp, mask):
    """``pto.vsqrt`` – element-wise square root."""
    return _emit_unary_vec_op(_pto.VsqrtOp, inp, mask)


def vabs(inp, mask):
    """``pto.vabs`` – element-wise absolute value."""
    return _emit_unary_vec_op(_pto.VabsOp, inp, mask)


def vneg(inp, mask):
    """``pto.vneg`` – element-wise negation."""
    return _emit_unary_vec_op(_pto.VnegOp, inp, mask)


def vrelu(inp, mask):
    """``pto.vrelu`` – element-wise ReLU."""
    return _emit_unary_vec_op(_pto.VreluOp, inp, mask)


def vnot(inp, mask):
    """``pto.vnot`` – element-wise bitwise/logical not."""
    return _emit_unary_vec_op(_pto.VnotOp, inp, mask)


def vexpdif(inp, ref, mask, part: str = "ODD"):
    """``pto.vexpdif`` – ``exp(inp - ref)`` selecting ODD or EVEN lanes."""
    return wrap_surface_value(
        _pto.VexpdifOp(
            unwrap_surface_value(inp).type,
            unwrap_surface_value(inp),
            unwrap_surface_value(ref),
            unwrap_surface_value(mask),
            part,
        ).result
    )


def vexp(inp, mask):
    """``pto.vexp`` – element-wise exponential."""
    return _emit_unary_vec_op(_pto.VexpOp, inp, mask)


def vrec(inp, mask):
    """``pto.vrec`` – reciprocal, surfaced as ``1 / inp``."""
    zero_vec = vmuls(inp, 0, mask)
    one_vec = vadds(zero_vec, 1, mask)
    return vdiv(one_vec, inp, mask)


def vrsqrt(inp, mask):
    """``pto.vrsqrt`` – inverse square root, surfaced as ``1 / sqrt(inp)``."""
    sqrt_vec = vsqrt(inp, mask)
    return vrec(sqrt_vec, mask)


def vcgmax(v, mask):
    """``pto.vcgmax`` – group maximum reduction, surfaced as the lowest-lane scalar."""
    reduced = _pto.VcgmaxOp(
        unwrap_surface_value(v).type,
        unwrap_surface_value(v),
        unwrap_surface_value(mask),
    ).result
    return _extract_lowest_lane_scalar(reduced, mask)


def vcgadd(v, mask):
    """``pto.vcgadd`` – group sum reduction, surfaced as the lowest-lane scalar."""
    reduced = _pto.VcgaddOp(
        unwrap_surface_value(v).type,
        unwrap_surface_value(v),
        unwrap_surface_value(mask),
    ).result
    return _extract_lowest_lane_scalar(reduced, mask)


def vcgmin(v, mask):
    """``pto.vcgmin`` – group minimum reduction, surfaced as the lowest-lane scalar."""
    reduced = _pto.VcgminOp(
        unwrap_surface_value(v).type,
        unwrap_surface_value(v),
        unwrap_surface_value(mask),
    ).result
    return _extract_lowest_lane_scalar(reduced, mask)


def vcpadd(v, mask):
    """``pto.vcpadd`` – inclusive prefix sum."""
    return _emit_unary_vec_op(_pto.VcpaddOp, v, mask)


def vadds(inp, scalar, mask):
    """``pto.vadds`` – vector plus scalar under mask."""
    return _emit_vec_scalar_masked_op(_pto.VaddsOp, inp, scalar, mask, context="vadds")


def vsubs(inp, scalar, mask):
    """``pto.vsubs`` – vector minus scalar under mask."""
    raw_scalar = _coerce_scalar_like_vector_element(inp, scalar, context="vsubs")
    neg_scalar = _negate_runtime_scalar(raw_scalar)
    return wrap_surface_value(
        _pto.VaddsOp(
            unwrap_surface_value(inp).type,
            unwrap_surface_value(inp),
            neg_scalar,
            unwrap_surface_value(mask),
        ).result
    )


def vmuls(inp, scalar, mask):
    """``pto.vmuls`` – vector times scalar under mask."""
    return _emit_vec_scalar_masked_op(_pto.VmulsOp, inp, scalar, mask, context="vmuls")


def vmaxs(inp, scalar, mask):
    """``pto.vmaxs`` – vector/scalar maximum under mask."""
    return _emit_vec_scalar_masked_op(_pto.VmaxsOp, inp, scalar, mask, context="vmaxs")


def vmins(inp, scalar, mask):
    """``pto.vmins`` – vector/scalar minimum under mask."""
    return _emit_vec_scalar_masked_op(_pto.VminsOp, inp, scalar, mask, context="vmins")


def vlrelu(inp, alpha, mask):
    """``pto.vlrelu`` – vector leaky ReLU under mask."""
    return _emit_vec_scalar_masked_op(_pto.VlreluOp, inp, alpha, mask, context="vlrelu")


def vaddrelu(lhs, rhs, mask):
    """``pto.vaddrelu`` – add, then apply ReLU."""
    return vrelu(vadd(lhs, rhs, mask), mask)


def vsubrelu(lhs, rhs, mask):
    """``pto.vsubrelu`` – subtract, then apply ReLU."""
    return vrelu(vsub(lhs, rhs, mask), mask)


def vaxpy(alpha, x, y, mask):
    """``pto.vaxpy`` – fused ``alpha * x + y``."""
    alpha_value = _coerce_scalar_like_vector_element(x, alpha, context="vaxpy")
    return wrap_surface_value(
        _pto.VaxpyOp(
            unwrap_surface_value(x).type,
            unwrap_surface_value(x),
            unwrap_surface_value(y),
            unwrap_surface_value(alpha_value),
            unwrap_surface_value(mask),
        ).result
    )


def vsel(true_v, false_v, mask):
    """``pto.vsel`` – element-wise select under a predicate mask."""
    return wrap_surface_value(
        _pto.VselOp(
            unwrap_surface_value(true_v).type,
            unwrap_surface_value(true_v),
            unwrap_surface_value(false_v),
            unwrap_surface_value(mask),
        ).result
    )


# ── Tile-domain operations ────────────────────────────────────────────────────

def make_tensor_view(ptr, *, shape=None, strides=None):
    """
    ``pto.make_tensor_view`` – wrap a pointer as a tensor view.

    Type is inferred: rank from ``len(shape)``, element type from ``ptr``.
    """
    if shape is None or strides is None:
        raise make_tensor_view_missing_metadata_error(ptr)
    ptr = resolve_tensor_data_entry(ptr)
    rank = len(shape)
    raw_ptr = unwrap_surface_value(ptr)
    elem = _pto.PtrType(raw_ptr.type).element_type
    normalized_shape = [
        _coerce_index(dim, context="make_tensor_view(shape=...)")
        for dim in shape
    ]
    normalized_strides = [
        _coerce_index(dim, context="make_tensor_view(strides=...)")
        for dim in strides
    ]
    static_dims = _static_index_dims(normalized_shape)
    tv_type = (
        tensor_view_type_from_dims(static_dims, elem)
        if static_dims is not None
        else tensor_view_type(rank, elem)
    )
    value = _pto.MakeTensorViewOp(
        tv_type,
        raw_ptr,
        _unwrap_sequence(normalized_shape),
        _unwrap_sequence(normalized_strides),
    ).result
    return TensorViewValue(value, shape=tuple(shape), strides=tuple(strides))


def _normalize_static_tile_shape(shape):
    static_shape = []
    for dim in shape:
        if isinstance(dim, bool) or not isinstance(dim, int):
            raise TypeError(
                "alloc_tile(shape=...) currently requires a static physical tile shape. "
                "Use constexpr/static integers for shape and place runtime metadata in valid_shape."
            )
        static_shape.append(dim)
    return tuple(static_shape)


def _authored_tile_physical_shape(shape):
    if len(shape) == 1:
        return (1, shape[0])
    return tuple(shape)


def _split_valid_shape(shape, valid_shape):
    logical_rank = len(shape)
    if valid_shape is None:
        return _authored_tile_physical_shape(shape), None, None, tuple(shape)

    if len(valid_shape) != logical_rank:
        raise TypeError(
            f"alloc_tile(valid_shape=...) rank mismatch: expected {logical_rank} dims, got {len(valid_shape)}"
        )

    surface_valid_shape = []
    if logical_rank == 1:
        dim = valid_shape[0]
        surface_valid_shape.append(dim)
        if isinstance(dim, bool):
            raise TypeError("alloc_tile(valid_shape=...) does not accept bool dimensions")
        if isinstance(dim, int):
            return (1, dim), None, None, tuple(surface_valid_shape)
        return (-1, -1), 1, dim, tuple(surface_valid_shape)

    type_valid_shape = []
    valid_row = None
    valid_col = None
    for index, dim in enumerate(valid_shape):
        surface_valid_shape.append(dim)
        if isinstance(dim, bool):
            raise TypeError("alloc_tile(valid_shape=...) does not accept bool dimensions")
        if isinstance(dim, int):
            type_valid_shape.append(dim)
            continue
        type_valid_shape.append(-1)
        if index == 0:
            valid_row = dim
            continue
        if index == 1:
            valid_col = dim
            continue
        raise TypeError(
            "alloc_tile(valid_shape=...) currently only supports dynamic runtime metadata "
            "for the first two dimensions"
        )
    return tuple(type_valid_shape), valid_row, valid_col, tuple(surface_valid_shape)


def _uses_row_major_none_box_layout(blayout, slayout) -> bool:
    return str(blayout).lower() == "rowmajor" and str(slayout).lower() == "nonebox"


def _validate_authored_tile_row_alignment(shape, dtype, *, blayout, slayout):
    if not _uses_row_major_none_box_layout(blayout, slayout):
        return
    if not shape:
        return
    elem_bytewidth = _element_bytewidth(_resolve(dtype))
    row_bytes = shape[-1] * elem_bytewidth
    required_alignment = 32
    if row_bytes % required_alignment == 0:
        return
    raise tile_row_alignment_error(
        shape=shape,
        dtype=str(_resolve(dtype)),
        row_bytes=row_bytes,
        required_alignment=required_alignment,
    )


def partition_view(tv, *, offsets, sizes):
    """
    ``pto.partition_view`` – slice a tensor view.

    Type is inferred from the source tensor-view type.
    """
    spec = compose_partition_spec(tv, offsets=offsets, sizes=sizes)
    if spec is not None:
        source = spec.root_tensor_view
        offsets = spec.offsets
        sizes = spec.sizes
    else:
        source = tv

    raw_source = unwrap_surface_value(source)
    src_type = _pto.TensorViewType(raw_source.type)
    rank = src_type.rank
    elem = src_type.element_type
    normalized_offsets = [
        _coerce_index(offset, context="partition_view(offsets=...)")
        for offset in offsets
    ]
    normalized_sizes = [
        _coerce_index(size, context="partition_view(sizes=...)")
        for size in sizes
    ]
    static_dims = _static_index_dims(normalized_sizes)
    ptv_type = (
        part_tensor_view_type_from_dims(static_dims, elem)
        if static_dims is not None
        else part_tensor_view_type(rank, elem)
    )
    value = _pto.PartitionViewOp(
        ptv_type,
        raw_source,
        _unwrap_sequence(normalized_offsets),
        _unwrap_sequence(normalized_sizes),
    ).result
    return wrap_surface_value(
        value,
        root_tensor_view=source if spec is None else spec.root_tensor_view,
        offsets=tuple(offsets),
        sizes=tuple(sizes),
    )


def alloc_tile(
    tile_type=None,
    *,
    shape=None,
    dtype=None,
    memory_space="ub",
    valid_shape=None,
    blayout: str = "RowMajor",
    slayout: str = "NoneBox",
    fractal_size: int = 512,
    pad: str = "Null",
    addr=None,
    valid_row=None,
    valid_col=None,
):
    """
    ``pto.alloc_tile``.

    Accepts either the authored surface form:

    ``alloc_tile(shape=[...], dtype=..., memory_space=..., valid_shape=..., addr=...)``

    or the low-level explicit-type form:

    ``alloc_tile(tile_type, addr=..., valid_row=..., valid_col=...)``.
    """
    if tile_type is not None and shape is not None:
        raise TypeError("alloc_tile() accepts either tile_type or shape=/dtype=, not both")

    if tile_type is None:
        if shape is None or dtype is None:
            raise TypeError("alloc_tile() requires either tile_type or both shape= and dtype=")
        if valid_row is not None or valid_col is not None:
            raise TypeError(
                "alloc_tile(shape=..., dtype=...) uses the authored surface form; "
                "use valid_shape=... instead of valid_row=/valid_col="
            )
        logical_shape = _normalize_static_tile_shape(shape)
        physical_shape = _authored_tile_physical_shape(logical_shape)
        _validate_authored_tile_row_alignment(physical_shape, dtype, blayout=blayout, slayout=slayout)
        type_valid_shape, valid_row, valid_col, surface_valid_shape = _split_valid_shape(logical_shape, valid_shape)
        from ._types import tile_buf_type
        tile_type = tile_buf_type(
            physical_shape,
            dtype,
            type_valid_shape,
            blayout=blayout,
            address_space=memory_space,
            slayout=slayout,
            fractal_size=fractal_size,
            pad=pad,
        )
        shape = logical_shape
    else:
        physical_shape = None
        surface_valid_shape = None

    value = _pto.AllocTileOp(
        _resolve(tile_type),
        addr=_coerce_i64(addr, context="alloc_tile(addr)") if addr is not None else None,
        valid_row=_coerce_index(valid_row, context="alloc_tile(valid_row)") if valid_row is not None else None,
        valid_col=_coerce_index(valid_col, context="alloc_tile(valid_col)") if valid_col is not None else None,
    ).result
    if tile_type is not None and (valid_row is not None or valid_col is not None):
        parsed_tile_type = parse_tile_type_metadata(_resolve(tile_type))
        rank = len(shape) if shape is not None else len(parsed_tile_type["shape_dims"])
        surface_valid_shape = [None] * rank
        if rank >= 1:
            surface_valid_shape[0] = valid_row
        if rank >= 2:
            surface_valid_shape[1] = valid_col
        surface_valid_shape = tuple(surface_valid_shape)
    return wrap_surface_value(
        value,
        tile_metadata={
            "shape": shape,
            "physical_shape": physical_shape,
            "dtype": dtype,
            "memory_space": memory_space,
            "valid_shape": surface_valid_shape,
        },
    )


def set_tile_valid_shape(tile, valid_shape):
    """Update the runtime valid-shape metadata of an authored dynamic tile."""
    parsed_tile_type = parse_tile_type_metadata(unwrap_surface_value(tile).type)
    if parsed_tile_type is None:
        raise TypeError("tile.valid_shape assignment expects a tile_buf-backed value")
    if len(parsed_tile_type["shape_dims"]) != 2:
        raise TypeError("tile.valid_shape assignment currently only supports rank-2 tiles")
    logical_rank = len(tile.shape) if getattr(tile, "shape", None) is not None else 2
    if logical_rank == 1:
        if len(valid_shape) != 1:
            raise TypeError("rank-1 tile.valid_shape assignment expects exactly one dimension")
        if parsed_tile_type["valid_dims"] != (None, None):
            raise TypeError(
                "rank-1 tile.valid_shape assignment requires a tile allocated with "
                "valid_shape=[...] so the physical valid row/col metadata remain dynamic"
            )
        valid_row = _coerce_index_value(1)
        valid_col = _coerce_index(valid_shape[0], context="tile.valid_shape assignment")
    else:
        if len(valid_shape) != 2:
            raise TypeError("tile.valid_shape assignment currently expects exactly two dimensions")
        if parsed_tile_type["valid_dims"] != (None, None):
            raise TypeError(
                "tile.valid_shape assignment requires a tile allocated with fully dynamic "
                "valid_shape=[..., ...]"
            )
        valid_row = _coerce_index(valid_shape[0], context="tile.valid_shape assignment")
        valid_col = _coerce_index(valid_shape[1], context="tile.valid_shape assignment")
    _pto.SetValidShapeOp(
        unwrap_surface_value(tile),
        valid_row,
        valid_col,
    )


def tload(part, tile):
    """``pto.tload ins(part) outs(tile)``."""
    _pto.TLoadOp(None, unwrap_surface_value(part), unwrap_surface_value(tile))


def tstore(tile, part):
    """``pto.tstore ins(tile) outs(part)``."""
    _pto.TStoreOp(None, unwrap_surface_value(tile), unwrap_surface_value(part))


def tmov(src, dst):
    """``pto.tmov ins(src) outs(dst)`` – move data between tile domains."""
    _pto.TMovOp(None, unwrap_surface_value(src), unwrap_surface_value(dst))


def _coerce_tile_scalar_operand(tile, scalar, *, context: str):
    return _constant_like(scalar, infer_tile_element_type(wrap_surface_value(tile)))


def tadd(src0, src1, dst):
    """``pto.tadd ins(src0, src1) outs(dst)``."""
    _pto.tadd(
        unwrap_surface_value(src0),
        unwrap_surface_value(src1),
        unwrap_surface_value(dst),
    )


def tsub(src0, src1, dst):
    """``pto.tsub ins(src0, src1) outs(dst)``."""
    _pto.tsub(
        unwrap_surface_value(src0),
        unwrap_surface_value(src1),
        unwrap_surface_value(dst),
    )


def tmul(src0, src1, dst):
    """``pto.tmul ins(src0, src1) outs(dst)``."""
    _pto.tmul(
        unwrap_surface_value(src0),
        unwrap_surface_value(src1),
        unwrap_surface_value(dst),
    )


def tdiv(src0, src1, dst, *, div_precision=None):
    """``pto.tdiv ins(src0, src1) outs(dst)``."""
    _pto.tdiv(
        unwrap_surface_value(src0),
        unwrap_surface_value(src1),
        unwrap_surface_value(dst),
        precision_type=div_precision,
    )


def tmax(src0, src1, dst):
    """``pto.tmax ins(src0, src1) outs(dst)``."""
    _pto.tmax(
        unwrap_surface_value(src0),
        unwrap_surface_value(src1),
        unwrap_surface_value(dst),
    )


def tmin(src0, src1, dst):
    """``pto.tmin ins(src0, src1) outs(dst)``."""
    _pto.tmin(
        unwrap_surface_value(src0),
        unwrap_surface_value(src1),
        unwrap_surface_value(dst),
    )


def tadds(src, scalar, dst):
    """``pto.tadds ins(src, scalar) outs(dst)``."""
    _pto.tadds(
        unwrap_surface_value(src),
        _coerce_tile_scalar_operand(src, scalar, context="tadds"),
        unwrap_surface_value(dst),
    )


def tsubs(src, scalar, dst):
    """``pto.tsubs ins(src, scalar) outs(dst)``."""
    _pto.tsubs(
        unwrap_surface_value(src),
        _coerce_tile_scalar_operand(src, scalar, context="tsubs"),
        unwrap_surface_value(dst),
    )


def tmuls(src, scalar, dst):
    """``pto.tmuls ins(src, scalar) outs(dst)``."""
    _pto.tmuls(
        unwrap_surface_value(src),
        _coerce_tile_scalar_operand(src, scalar, context="tmuls"),
        unwrap_surface_value(dst),
    )


def tdivs(src, scalar, dst, *, div_precision=None):
    """``pto.tdivs ins(src, scalar) outs(dst)``."""
    _pto.tdivs(
        unwrap_surface_value(src),
        _coerce_tile_scalar_operand(src, scalar, context="tdivs"),
        unwrap_surface_value(dst),
        precision_type=div_precision,
    )


def tmaxs(src, scalar, dst):
    """``pto.tmaxs ins(src, scalar) outs(dst)``."""
    _pto.tmaxs(
        unwrap_surface_value(src),
        _coerce_tile_scalar_operand(src, scalar, context="tmaxs"),
        unwrap_surface_value(dst),
    )


def tmins(src, scalar, dst):
    """``pto.tmins ins(src, scalar) outs(dst)``."""
    _pto.tmins(
        unwrap_surface_value(src),
        _coerce_tile_scalar_operand(src, scalar, context="tmins"),
        unwrap_surface_value(dst),
    )


def texp(src, dst, *, exp_precision=None):
    """``pto.texp ins(src) outs(dst)``."""
    _pto.texp(
        unwrap_surface_value(src),
        unwrap_surface_value(dst),
        precision_type=exp_precision,
    )


def tlog(src, dst, *, log_precision=None):
    """``pto.tlog ins(src) outs(dst)``."""
    _pto.tlog(
        unwrap_surface_value(src),
        unwrap_surface_value(dst),
        precision_type=log_precision,
    )


def tsqrt(src, dst, *, sqrt_precision=None):
    """``pto.tsqrt ins(src) outs(dst)``."""
    _pto.tsqrt(
        unwrap_surface_value(src),
        unwrap_surface_value(dst),
        precision_type=sqrt_precision,
    )


def trsqrt(src, dst, *, tmp=None, rsqrt_precision=None):
    """``pto.trsqrt ins(src, tmp?) outs(dst)``."""
    _pto.trsqrt(
        unwrap_surface_value(src),
        unwrap_surface_value(dst),
        tmp=None if tmp is None else unwrap_surface_value(tmp),
        precision_type=rsqrt_precision,
    )


def trecip(src, dst, *, recip_precision=None):
    """``pto.trecip ins(src) outs(dst)``."""
    _pto.trecip(
        unwrap_surface_value(src),
        unwrap_surface_value(dst),
        precision_type=recip_precision,
    )


def tabs(src, dst):
    """``pto.tabs ins(src) outs(dst)``."""
    _pto.tabs(
        unwrap_surface_value(src),
        unwrap_surface_value(dst),
    )


def tneg(src, dst):
    """``pto.tneg ins(src) outs(dst)``."""
    _pto.tneg(
        unwrap_surface_value(src),
        unwrap_surface_value(dst),
    )


def trelu(src, dst):
    """``pto.trelu ins(src) outs(dst)``."""
    _pto.trelu(
        unwrap_surface_value(src),
        unwrap_surface_value(dst),
    )


def tlrelu(src, slope, dst):
    """``pto.tlrelu ins(src, slope) outs(dst)``."""
    _pto.tlrelu(
        unwrap_surface_value(src),
        _coerce_tile_scalar_operand(src, slope, context="tlrelu"),
        unwrap_surface_value(dst),
    )


def trowsum(src, tmp, dst):
    """``pto.trowsum ins(src, tmp) outs(dst)``."""
    _pto.trowsum(
        unwrap_surface_value(src),
        unwrap_surface_value(tmp),
        unwrap_surface_value(dst),
    )


def trowmax(src, tmp, dst):
    """``pto.trowmax ins(src, tmp) outs(dst)``."""
    _pto.trowmax(
        unwrap_surface_value(src),
        unwrap_surface_value(tmp),
        unwrap_surface_value(dst),
    )


def trowmin(src, tmp, dst):
    """``pto.trowmin ins(src, tmp) outs(dst)``."""
    _pto.trowmin(
        unwrap_surface_value(src),
        unwrap_surface_value(tmp),
        unwrap_surface_value(dst),
    )


def trowprod(src, tmp, dst):
    """``pto.trowprod ins(src, tmp) outs(dst)``."""
    _pto.trowprod(
        unwrap_surface_value(src),
        unwrap_surface_value(tmp),
        unwrap_surface_value(dst),
    )


def trowargmax(src, tmp, dst):
    """``pto.trowargmax ins(src, tmp) outs(dst)``."""
    _pto.trowargmax(
        unwrap_surface_value(src),
        unwrap_surface_value(tmp),
        unwrap_surface_value(dst),
    )


def trowargmin(src, tmp, dst):
    """``pto.trowargmin ins(src, tmp) outs(dst)``."""
    _pto.trowargmin(
        unwrap_surface_value(src),
        unwrap_surface_value(tmp),
        unwrap_surface_value(dst),
    )


def tcolsum(src, dst, *, tmp=None, is_binary=None):
    """``pto.tcolsum ins(src, tmp?) outs(dst)``."""
    _pto.tcolsum(
        unwrap_surface_value(src),
        unwrap_surface_value(dst),
        tmp=None if tmp is None else unwrap_surface_value(tmp),
        is_binary=is_binary,
    )


def tcolmax(src, dst):
    """``pto.tcolmax ins(src) outs(dst)``."""
    _pto.tcolmax(
        unwrap_surface_value(src),
        unwrap_surface_value(dst),
    )


def tcolmin(src, dst):
    """``pto.tcolmin ins(src) outs(dst)``."""
    _pto.tcolmin(
        unwrap_surface_value(src),
        unwrap_surface_value(dst),
    )


def tcolprod(src, dst):
    """``pto.tcolprod ins(src) outs(dst)``."""
    _pto.tcolprod(
        unwrap_surface_value(src),
        unwrap_surface_value(dst),
    )


def tcolargmax(src, tmp, dst):
    """``pto.tcolargmax ins(src, tmp) outs(dst)``."""
    _pto.tcolargmax(
        unwrap_surface_value(src),
        unwrap_surface_value(tmp),
        unwrap_surface_value(dst),
    )


def tcolargmin(src, tmp, dst):
    """``pto.tcolargmin ins(src, tmp) outs(dst)``."""
    _pto.tcolargmin(
        unwrap_surface_value(src),
        unwrap_surface_value(tmp),
        unwrap_surface_value(dst),
    )


def tcmp(src0, src1, dst, *, cmp_mode=None):
    """``pto.tcmp ins(src0, src1) outs(dst)``."""
    _pto.tcmp(
        unwrap_surface_value(src0),
        unwrap_surface_value(src1),
        unwrap_surface_value(dst),
        cmp_mode=None if cmp_mode is None else _cmp_mode_attr(cmp_mode),
    )


def tcmps(src, scalar, dst, *, cmp_mode=None):
    """``pto.tcmps ins(src, scalar) outs(dst)``."""
    _pto.tcmps(
        unwrap_surface_value(src),
        _coerce_tile_scalar_operand(src, scalar, context="tcmps"),
        unwrap_surface_value(dst),
        cmp_mode=None if cmp_mode is None else _cmp_mode_attr(cmp_mode),
    )


def texpands(scalar, dst):
    """``pto.texpands ins(scalar) outs(dst)``."""
    _pto.texpands(
        _coerce_tile_scalar_operand(dst, scalar, context="texpands"),
        unwrap_surface_value(dst),
    )


def trowexpand(src, dst):
    """``pto.trowexpand ins(src) outs(dst)``."""
    _pto.trowexpand(
        unwrap_surface_value(src),
        unwrap_surface_value(dst),
    )


def tcolexpand(src, dst):
    """``pto.tcolexpand ins(src) outs(dst)``."""
    _pto.tcolexpand(
        unwrap_surface_value(src),
        unwrap_surface_value(dst),
    )


def trowexpandadd(src0, src1, dst):
    """``pto.trowexpandadd ins(src0, src1) outs(dst)``."""
    _pto.trowexpandadd(
        unwrap_surface_value(src0),
        unwrap_surface_value(src1),
        unwrap_surface_value(dst),
    )


def trowexpandsub(src0, src1, dst, *, tmp=None):
    """``pto.trowexpandsub ins(src0, src1, tmp?) outs(dst)``."""
    _pto.trowexpandsub(
        unwrap_surface_value(src0),
        unwrap_surface_value(src1),
        unwrap_surface_value(dst),
        tmp=None if tmp is None else unwrap_surface_value(tmp),
    )


def trowexpandmul(src0, src1, dst, *, tmp=None):
    """``pto.trowexpandmul ins(src0, src1, tmp?) outs(dst)``."""
    _pto.trowexpandmul(
        unwrap_surface_value(src0),
        unwrap_surface_value(src1),
        unwrap_surface_value(dst),
        tmp=None if tmp is None else unwrap_surface_value(tmp),
    )


def trowexpanddiv(src0, src1, dst, *, tmp=None, div_precision=None):
    """``pto.trowexpanddiv ins(src0, src1, tmp?) outs(dst)``."""
    _pto.trowexpanddiv(
        unwrap_surface_value(src0),
        unwrap_surface_value(src1),
        unwrap_surface_value(dst),
        tmp=None if tmp is None else unwrap_surface_value(tmp),
        precision_type=div_precision,
    )


def trowexpandmax(src0, src1, dst, *, tmp=None):
    """``pto.trowexpandmax ins(src0, src1, tmp?) outs(dst)``."""
    _pto.trowexpandmax(
        unwrap_surface_value(src0),
        unwrap_surface_value(src1),
        unwrap_surface_value(dst),
        tmp=None if tmp is None else unwrap_surface_value(tmp),
    )


def trowexpandmin(src0, src1, dst, *, tmp=None):
    """``pto.trowexpandmin ins(src0, src1, tmp?) outs(dst)``."""
    _pto.trowexpandmin(
        unwrap_surface_value(src0),
        unwrap_surface_value(src1),
        unwrap_surface_value(dst),
        tmp=None if tmp is None else unwrap_surface_value(tmp),
    )


def trowexpandexpdif(src0, src1, dst, *, tmp=None):
    """``pto.trowexpandexpdif ins(src0, src1, tmp?) outs(dst)``."""
    _pto.trowexpandexpdif(
        unwrap_surface_value(src0),
        unwrap_surface_value(src1),
        unwrap_surface_value(dst),
        tmp=None if tmp is None else unwrap_surface_value(tmp),
    )


def tcolexpandadd(src0, src1, dst):
    """``pto.tcolexpandadd ins(src0, src1) outs(dst)``."""
    _pto.tcolexpandadd(
        unwrap_surface_value(src0),
        unwrap_surface_value(src1),
        unwrap_surface_value(dst),
    )


def tcolexpandsub(src0, src1, dst):
    """``pto.tcolexpandsub ins(src0, src1) outs(dst)``."""
    _pto.tcolexpandsub(
        unwrap_surface_value(src0),
        unwrap_surface_value(src1),
        unwrap_surface_value(dst),
    )


def tcolexpandmul(src0, src1, dst):
    """``pto.tcolexpandmul ins(src0, src1) outs(dst)``."""
    _pto.tcolexpandmul(
        unwrap_surface_value(src0),
        unwrap_surface_value(src1),
        unwrap_surface_value(dst),
    )


def tcolexpanddiv(src0, src1, dst, *, div_precision=None):
    """``pto.tcolexpanddiv ins(src0, src1) outs(dst)``."""
    _pto.tcolexpanddiv(
        unwrap_surface_value(src0),
        unwrap_surface_value(src1),
        unwrap_surface_value(dst),
        precision_type=div_precision,
    )


def tcolexpandmax(src0, src1, dst):
    """``pto.tcolexpandmax ins(src0, src1) outs(dst)``."""
    _pto.tcolexpandmax(
        unwrap_surface_value(src0),
        unwrap_surface_value(src1),
        unwrap_surface_value(dst),
    )


def tcolexpandmin(src0, src1, dst):
    """``pto.tcolexpandmin ins(src0, src1) outs(dst)``."""
    _pto.tcolexpandmin(
        unwrap_surface_value(src0),
        unwrap_surface_value(src1),
        unwrap_surface_value(dst),
    )


def tcolexpandexpdif(src0, src1, dst):
    """``pto.tcolexpandexpdif ins(src0, src1) outs(dst)``."""
    _pto.tcolexpandexpdif(
        unwrap_surface_value(src0),
        unwrap_surface_value(src1),
        unwrap_surface_value(dst),
    )


def _resolve_selection_tmp(dst, tmp, *, context: str):
    if tmp is not None:
        return tmp

    session = None
    try:
        from ._tracing.active import current_session
        session = current_session()
    except Exception:
        session = None

    if session is not None and getattr(session.module_spec, "target_arch", None) == "a5":
        return dst

    return alloc_tile(tile_type=unwrap_surface_value(dst).type)


def tsel(mask, src0, src1, dst, *, tmp=None):
    """``pto.tsel ins(mask, src0, src1, tmp) outs(dst)`` with synthesized scratch when omitted."""
    resolved_tmp = tmp if tmp is not None else _resolve_selection_tmp(dst, tmp, context="tsel")
    _pto.tsel(
        unwrap_surface_value(mask),
        unwrap_surface_value(src0),
        unwrap_surface_value(src1),
        unwrap_surface_value(resolved_tmp),
        unwrap_surface_value(dst),
    )


def tsels(mask, src, scalar, dst, *, tmp=None):
    """``pto.tsels ins(mask, src, tmp, scalar) outs(dst)`` with synthesized scratch when omitted."""
    resolved_tmp = tmp if tmp is not None else _resolve_selection_tmp(dst, tmp, context="tsels")
    _pto.tsels(
        unwrap_surface_value(mask),
        unwrap_surface_value(src),
        unwrap_surface_value(resolved_tmp),
        _coerce_tile_scalar_operand(src, scalar, context="tsels"),
        unwrap_surface_value(dst),
    )


def tcvt(src, dst, *, tmp=None, rmode=None, sat_mode=None):
    """``pto.tcvt ins(src, tmp?) outs(dst)``."""
    _pto.tcvt(
        unwrap_surface_value(src),
        unwrap_surface_value(dst),
        tmp=None if tmp is None else unwrap_surface_value(tmp),
        rmode=rmode,
        sat_mode=sat_mode,
    )


def tnot(src, dst):
    """``pto.tnot ins(src) outs(dst)``."""
    _pto.tnot(
        unwrap_surface_value(src),
        unwrap_surface_value(dst),
    )


def tand(src0, src1, dst):
    """``pto.tand ins(src0, src1) outs(dst)``."""
    _pto.tand(
        unwrap_surface_value(src0),
        unwrap_surface_value(src1),
        unwrap_surface_value(dst),
    )


def tands(src, scalar, dst):
    """``pto.tands ins(src, scalar) outs(dst)``."""
    _pto.tands(
        unwrap_surface_value(src),
        _coerce_tile_scalar_operand(src, scalar, context="tands"),
        unwrap_surface_value(dst),
    )


def tor(src0, src1, dst):
    """``pto.tor ins(src0, src1) outs(dst)``."""
    _pto.tor(
        unwrap_surface_value(src0),
        unwrap_surface_value(src1),
        unwrap_surface_value(dst),
    )


def tors(src, scalar, dst):
    """``pto.tors ins(src, scalar) outs(dst)``."""
    _pto.tors(
        unwrap_surface_value(src),
        _coerce_tile_scalar_operand(src, scalar, context="tors"),
        unwrap_surface_value(dst),
    )


def txor(src0, src1, tmp, dst):
    """``pto.txor ins(src0, src1, tmp) outs(dst)``."""
    _pto.txor(
        unwrap_surface_value(src0),
        unwrap_surface_value(src1),
        unwrap_surface_value(tmp),
        unwrap_surface_value(dst),
    )


def txors(src, scalar, tmp, dst):
    """``pto.txors ins(src, scalar, tmp) outs(dst)``."""
    _pto.txors(
        unwrap_surface_value(src),
        _coerce_tile_scalar_operand(src, scalar, context="txors"),
        unwrap_surface_value(tmp),
        unwrap_surface_value(dst),
    )


def tshl(src0, src1, dst):
    """``pto.tshl ins(src0, src1) outs(dst)``."""
    _pto.tshl(
        unwrap_surface_value(src0),
        unwrap_surface_value(src1),
        unwrap_surface_value(dst),
    )


def tshls(src, scalar, dst):
    """``pto.tshls ins(src, scalar) outs(dst)``."""
    _pto.tshls(
        unwrap_surface_value(src),
        _coerce_tile_scalar_operand(src, scalar, context="tshls"),
        unwrap_surface_value(dst),
    )


def tshr(src0, src1, dst):
    """``pto.tshr ins(src0, src1) outs(dst)``."""
    _pto.tshr(
        unwrap_surface_value(src0),
        unwrap_surface_value(src1),
        unwrap_surface_value(dst),
    )


def tshrs(src, scalar, dst):
    """``pto.tshrs ins(src, scalar) outs(dst)``."""
    _pto.tshrs(
        unwrap_surface_value(src),
        _coerce_tile_scalar_operand(src, scalar, context="tshrs"),
        unwrap_surface_value(dst),
    )


def tpartadd(src0, src1, dst):
    """``pto.tpartadd ins(src0, src1) outs(dst)``."""
    _pto.tpartadd(
        unwrap_surface_value(src0),
        unwrap_surface_value(src1),
        unwrap_surface_value(dst),
    )


def tpartmul(src0, src1, dst):
    """``pto.tpartmul ins(src0, src1) outs(dst)``."""
    _pto.tpartmul(
        unwrap_surface_value(src0),
        unwrap_surface_value(src1),
        unwrap_surface_value(dst),
    )


def tpartmax(src0, src1, dst):
    """``pto.tpartmax ins(src0, src1) outs(dst)``."""
    _pto.tpartmax(
        unwrap_surface_value(src0),
        unwrap_surface_value(src1),
        unwrap_surface_value(dst),
    )


def tpartmin(src0, src1, dst):
    """``pto.tpartmin ins(src0, src1) outs(dst)``."""
    _pto.tpartmin(
        unwrap_surface_value(src0),
        unwrap_surface_value(src1),
        unwrap_surface_value(dst),
    )


def tfillpad(src, dst):
    """``pto.tfillpad ins(src) outs(dst)``."""
    _pto.tfillpad(
        unwrap_surface_value(src),
        unwrap_surface_value(dst),
    )


def tfillpad_expand(src, dst):
    """``pto.tfillpad_expand ins(src) outs(dst)``."""
    _pto.tfillpad_expand(
        unwrap_surface_value(src),
        unwrap_surface_value(dst),
    )


def tfillpad_inplace(src, dst):
    """``pto.tfillpad_inplace ins(src) outs(dst)``."""
    _pto.tfillpad_inplace(
        unwrap_surface_value(src),
        unwrap_surface_value(dst),
    )


def as_ptr(value):
    """Materialize a typed pointer from a tile or tensor-view descriptor."""
    wrapped = wrap_surface_value(value)
    return emit_as_ptr(wrapped)


def _constant_like(value, mlir_type):
    value = unwrap_surface_value(value)
    if hasattr(value, "type"):
        return value
    if isinstance(value, float):
        return arith.ConstantOp(mlir_type, FloatAttr.get(mlir_type, value)).result
    if IntegerType.isinstance(mlir_type):
        return _materialize_integer_literal(mlir_type, value)
    return arith.ConstantOp(mlir_type, value).result


def _index_zero():
    return arith.ConstantOp(IndexType.get(), 0).result


def _tile_slice_linear_offset(tile_slice: TileSliceValue):
    offsets = tile_slice.offsets
    if len(offsets) == 1:
        return offsets[0]
    if len(offsets) != 2:
        raise RuntimeError("tile slice pointer lowering only supports rank-1 or rank-2 offsets")

    physical_shape = getattr(tile_slice.tile, "physical_shape", None)
    if physical_shape is None or len(physical_shape) != 2 or physical_shape[1] is None:
        raise RuntimeError("tile slice pointer lowering requires static physical column shape metadata")

    row, col = offsets
    stride = physical_shape[1]
    if isinstance(row, int) and isinstance(col, int):
        return row * stride + col

    row_value = _coerce_index(row, context="tile slice pointer lowering")
    row_stride = arith.MulIOp(row_value, arith.ConstantOp(IndexType.get(), stride).result).result
    col_value = _coerce_index(col, context="tile slice pointer lowering")
    return arith.AddIOp(row_stride, col_value).result


def _tile_slice_ptr(tile_slice: TileSliceValue):
    base_ptr = emit_as_ptr(tile_slice.tile)
    linear_offset = _tile_slice_linear_offset(tile_slice)
    if isinstance(linear_offset, int) and linear_offset == 0:
        return base_ptr
    return addptr(base_ptr, _coerce_index(linear_offset, context="tile slice pointer lowering"))


def _infer_vreg_type_from_tile_slice(tile_slice: TileSliceValue):
    memref_type = MemRefType(tile_slice.type)
    elem_type = memref_type.element_type
    lanes = _elements_per_vreg(elem_type)
    return _resolve(vreg_type(lanes, elem_type))


def _infer_vreg_type_from_address_source(src_ptr):
    raw_source = unwrap_surface_value(src_ptr)
    source_type = raw_source.type
    try:
        elem_type = _pto.PtrType(source_type).element_type
    except Exception:
        try:
            elem_type = MemRefType(source_type).element_type
        except Exception as exc:
            raise TypeError(
                f"vlds(ptr, offset) cannot infer a vector-register type from source {source_type}; "
                "pass result_vreg_type= explicitly"
            ) from exc
    lanes = _elements_per_vreg(elem_type)
    return _resolve(vreg_type(lanes, elem_type))


def _elements_per_vreg(elem_type):
    try:
        bytewidth = _element_bytewidth(elem_type)
    except TypeError as exc:
        raise TypeError(f"vlds/vsts tile-slice sugar does not support element type {elem_type}")
    return 256 // bytewidth


def _infer_vreg_metadata(vector_value):
    raw_type = unwrap_surface_value(vector_value).type
    try:
        vreg_type = _pto.VRegType(raw_type)
        return vreg_type.lanes, vreg_type.element_type
    except Exception:
        text = str(raw_type)
        if not text.startswith("!pto.vreg<") or "x" not in text:
            raise TypeError(f"expected PTO vector-register type, got {raw_type}")
        body = text[len("!pto.vreg<"):-1]
        lanes_text, elem_text = body.split("x", 1)
        return int(lanes_text), Type.parse(elem_text)


def _extract_lowest_lane_scalar(vector_value, mask):
    lanes, elem_type = _infer_vreg_metadata(vector_value)
    tmp_tile = alloc_tile(shape=[1, lanes], dtype=elem_type, valid_shape=[1, 1])
    vsts(vector_value, tmp_tile.as_ptr(), _index_zero(), mask, dist="1PT_B32")
    from . import scalar as _scalar
    return _scalar.load(tmp_tile[0, 0])


def _element_bytewidth(elem_type):
    if F32Type.isinstance(elem_type):
        return 4
    if any(cls.isinstance(elem_type) for cls in (F16Type, BF16Type)):
        return 2
    if Float8E4M3FNType.isinstance(elem_type) or Float8E5M2Type.isinstance(elem_type):
        return 1
    if any(_isinstance_pto_type(elem_type, name) for name in ("HiF8Type", "F4E1M2x2Type", "F4E2M1x2Type")):
        return 1
    if IntegerType.isinstance(elem_type):
        width = IntegerType(elem_type).width
        if width % 8 != 0:
            raise TypeError(f"unsupported sub-byte integer element type {elem_type}")
        return width // 8
    raise TypeError(f"unsupported element type {elem_type}")


def bytewidth(dtype):
    """Return the size in bytes of one element of *dtype*."""
    return _element_bytewidth(_resolve(dtype))


def elements_per_vreg(dtype):
    """Return how many elements of *dtype* fit in one 256-byte vector register."""
    return _elements_per_vreg(_resolve(dtype))


def _mask_bits_for_dtype(dtype):
    elem_type = _resolve(dtype)
    bytewidth = _element_bytewidth(elem_type)
    if bytewidth == 4:
        return 32
    if bytewidth == 2:
        return 16
    if bytewidth == 1:
        return 8
    raise TypeError(f"make_mask(...) does not support dtype {elem_type}")


def _pset_op_for_mask_bits(mask_bits: int):
    return {
        8: _pto.PsetB8Op,
        16: _pto.PsetB16Op,
        32: _pto.PsetB32Op,
    }[mask_bits]


def _plt_op_for_mask_bits(mask_bits: int):
    return {
        8: _pto.PltB8Op,
        16: _pto.PltB16Op,
        32: _pto.PltB32Op,
    }[mask_bits]


def _coerce_i32(value, *, context: str):
    raw_value = unwrap_surface_value(value)
    i32_type = IntegerType.get_signless(32)
    if isinstance(raw_value, bool):
        raise TypeError(f"{context} does not accept bool values")
    if isinstance(raw_value, int):
        return _materialize_integer_literal(i32_type, raw_value)
    kind = classify_runtime_scalar_type(raw_value.type)
    if kind == "float":
        raise TypeError(f"{context} expects an integer-like scalar, got {raw_value.type}")
    if kind == "index":
        return arith.IndexCastOp(i32_type, raw_value).result
    signless_value = _strip_integer_signedness(raw_value)
    if signless_value.type == i32_type:
        return signless_value
    width = IntegerType(raw_value.type).width
    if width < 32:
        if _integer_signedness(raw_value.type) == "unsigned":
            return arith.ExtUIOp(i32_type, signless_value).result
        return arith.ExtSIOp(i32_type, signless_value).result
    if width > 32:
        return arith.TruncIOp(i32_type, signless_value).result
    return signless_value


def _coerce_i64(value, *, context: str):
    raw_value = unwrap_surface_value(value)
    i64_type = IntegerType.get_signless(64)
    if isinstance(raw_value, bool):
        raise TypeError(f"{context} does not accept bool values")
    if isinstance(raw_value, int):
        return _materialize_integer_literal(i64_type, raw_value)
    kind = classify_runtime_scalar_type(raw_value.type)
    if kind == "float":
        raise TypeError(f"{context} expects an integer-like scalar, got {raw_value.type}")
    if kind == "index":
        return arith.IndexCastOp(i64_type, raw_value).result
    signless_value = _strip_integer_signedness(raw_value)
    if signless_value.type == i64_type:
        return signless_value
    width = IntegerType(raw_value.type).width
    if width < 64:
        if _integer_signedness(raw_value.type) == "unsigned":
            return arith.ExtUIOp(i64_type, signless_value).result
        return arith.ExtSIOp(i64_type, signless_value).result
    if width > 64:
        return arith.TruncIOp(i64_type, signless_value).result
    return signless_value


def _i64_zero():
    return arith.ConstantOp(IntegerType.get_signless(64), 0).result


def _coerce_scalar_like_vector_element(vector_value, scalar_value, *, context: str):
    _, elem_type = _infer_vreg_metadata(vector_value)
    return coerce_scalar_to_type(scalar_value, elem_type, context=f"{context}(...)")


def _negate_runtime_scalar(value):
    raw_value = unwrap_surface_value(value)
    kind = classify_runtime_scalar_type(raw_value.type)
    zero = materialize_scalar_literal(0.0 if kind == "float" else 0, raw_value.type, context="_negate_runtime_scalar(...)")
    return emit_runtime_binary_op("sub", zero, raw_value)


def _mul_bytes(value, elem_type):
    factor = _element_bytewidth(_resolve(elem_type))
    raw_value = unwrap_surface_value(value)
    if isinstance(raw_value, int):
        return raw_value * factor
    return emit_runtime_binary_op("mul", raw_value, factor)


def _membar_attr(kind: str):
    normalized = str(kind)
    supported = {
        "VV_ALL",
        "VST_VLD",
        "VLD_VST",
        "VST_VST",
        "VS_ALL",
        "VST_LD",
        "VLD_ST",
        "VST_ST",
        "SV_ALL",
        "ST_VLD",
        "LD_VST",
        "ST_VST",
        "SS_ALL",
        "ST_LD",
        "LD_ST",
        "ST_ST",
    }
    if normalized not in supported:
        raise ValueError(f"unsupported mem_bar kind {kind!r}")
    return Attribute.parse(f"#pto.membar<{normalized}>")


def _acc_store_ub_dst_mode_attr(mode):
    normalized = {
        0: "single",
        1: "split_m",
        2: "split_n",
        "single": "single",
        "split_m": "split_m",
        "split_n": "split_n",
    }.get(mode if isinstance(mode, int) else str(mode).lower())
    if normalized is None:
        raise ValueError(f"unsupported mte_l0c_ub dst_mode {mode!r}")
    return Attribute.parse(f"#pto<acc_store_ub_dst_mode {normalized}>")


def _infer_dma_partition_row_stride(partition: PartitionTensorViewValue):
    if partition.shape is None or partition.strides is None:
        raise TypeError("mte_load/mte_store require partition view shape/stride metadata")
    outer_dims = list(partition.shape[:-1])
    non_unit = [i for i, dim in enumerate(outer_dims) if dim != 1]
    if len(non_unit) > 1:
        raise TypeError(
            "mte_load/mte_store currently only support partitions with at most one non-unit "
            "dimension before the contiguous innermost dimension"
        )
    if not non_unit:
        return 1, 0
    dim_index = non_unit[0]
    return partition.shape[dim_index], partition.strides[dim_index]


def _infer_dma_tile_geometry(tile: TileValue):
    if tile.shape is None:
        raise TypeError("mte_load/mte_store require tile shape metadata")
    if len(tile.shape) == 1:
        valid_cols = tile.valid_shape[0]
        return 1, valid_cols, tile.shape[0]
    if len(tile.shape) == 2:
        return tile.valid_shape[0], tile.valid_shape[1], tile.shape[1]
    raise TypeError("mte_load/mte_store currently only support rank-1 or rank-2 tiles")


def _infer_dma_2d_copy_signature(partition, tile, *, direction: str):
    row_count, src_row_stride = _infer_dma_partition_row_stride(partition)
    tile_rows, valid_cols, physical_cols = _infer_dma_tile_geometry(tile)
    if direction == "gm_to_ub":
        return row_count, valid_cols, _mul_bytes(src_row_stride, infer_tile_element_type(tile)), physical_cols * _element_bytewidth(infer_tile_element_type(tile))
    return row_count, valid_cols, physical_cols * _element_bytewidth(infer_tile_element_type(tile)), _mul_bytes(src_row_stride, infer_tile_element_type(tile))


def fill_tile(tile, value):
    """Broadcast a scalar into an entire tile."""
    wrapped_tile = wrap_surface_value(tile)
    scalar_value = _constant_like(value, infer_tile_element_type(wrapped_tile))
    _pto.TExpandsOp(scalar_value, unwrap_surface_value(wrapped_tile))


def make_mask(dtype, value):
    """Create a predicate mask matching *dtype* granularity."""
    mask_bits = _mask_bits_for_dtype(dtype)
    result_type = _mask_type_from_bits(mask_bits)

    if isinstance(value, str):
        return wrap_surface_value(
            _pset_op_for_mask_bits(mask_bits)(result_type, _normalize_mask_pattern(value)).result
        )

    raw_value = unwrap_surface_value(value)
    authored_scalar_type = raw_value.type if hasattr(raw_value, "type") else IntegerType.get_signless(32)
    raw_value = _coerce_i32(raw_value, context="make_mask(..., value)")
    plt_op = _plt_op_for_mask_bits(mask_bits)(result_type, IntegerType.get_signless(32), raw_value)
    next_value = coerce_scalar_to_type(
        plt_op.scalar_out,
        authored_scalar_type,
        context="make_mask(..., value) result",
    )
    return MaskResultValue(plt_op.mask, next_value)


# ── Hardware / sync ───────────────────────────────────────────────────────────

def _require_pto_ptr_operand(value, *, context: str):
    raw_value = unwrap_surface_value(value)
    try:
        _pto.PtrType(raw_value.type)
    except Exception as exc:
        raise TypeError(f"{context} expects PTO ptr operands, got {raw_value.type}") from exc
    return raw_value


@_explicit_mode_only("pto.mte_load(...)")
def mte_load(source, destination, l2_cache_ctl, len_burst, *, nburst, loops=None, pad=None):
    """
    Ptr-based GM->UB DMA wrapper aligned with the underlying ``pto.dma_load`` surface.

    This wrapper intentionally accepts only explicit pointer operands. It does
    not infer burst shape or strides from TensorView / PartitionTensorView /
    Tile metadata.
    """
    n_burst, nburst_src_stride, nburst_dst_stride = _normalize_dma_group(
        "nburst",
        nburst,
        context="mte_load(...)",
    )
    loop_counts, loop_src_strides, loop_dst_strides = _normalize_dma_loops(
        loops,
        context="mte_load(...)",
    )
    pad_value, left_padding_count, right_padding_count = _normalize_dma_pad(
        pad,
        context="mte_load(...)",
    )
    _pto.MteGmUbOp(
        _require_pto_ptr_operand(source, context="mte_load(...)"),
        _require_pto_ptr_operand(destination, context="mte_load(...)"),
        _coerce_i64(l2_cache_ctl, context="mte_load l2_cache_ctl"),
        _coerce_i64(len_burst, context="mte_load len_burst"),
        n_burst,
        nburst_src_stride,
        nburst_dst_stride,
        loop_counts,
        loop_src_strides,
        loop_dst_strides,
        pad_value=pad_value,
        left_padding_count=left_padding_count,
        right_padding_count=right_padding_count,
    )


@_explicit_mode_only("pto.mte_store(...)")
def mte_store(source, destination, len_burst, *, nburst, loops=None):
    """Ptr-based UB->GM DMA wrapper aligned with the underlying ``pto.dma_store`` surface."""
    n_burst, nburst_src_stride, nburst_dst_stride = _normalize_dma_group(
        "nburst",
        nburst,
        context="mte_store(...)",
    )
    loop_counts, loop_src_strides, loop_dst_strides = _normalize_dma_loops(
        loops,
        context="mte_store(...)",
    )
    _pto.MteUbGmOp(
        _require_pto_ptr_operand(source, context="mte_store(...)"),
        _require_pto_ptr_operand(destination, context="mte_store(...)"),
        _coerce_i64(len_burst, context="mte_store len_burst"),
        n_burst,
        nburst_src_stride,
        nburst_dst_stride,
        loop_counts,
        loop_src_strides,
        loop_dst_strides,
    )


def _normalize_dma_group(name, triple, *, context: str):
    if not isinstance(triple, tuple) or len(triple) != 3:
        raise TypeError(f"{context} expects {name}=(count, src_stride, dst_stride)")
    count, src_stride, dst_stride = triple
    return (
        _coerce_i64(count, context=f"{context} {name}[0]"),
        _coerce_i64(src_stride, context=f"{context} {name}[1]"),
        _coerce_i64(dst_stride, context=f"{context} {name}[2]"),
    )


def _normalize_dma_loops(loops, *, context: str):
    if loops is None:
        return [], [], []
    if not isinstance(loops, (list, tuple)):
        raise TypeError(f"{context} expects loops to be a list[tuple[int, int, int]] or None")
    counts = []
    src_strides = []
    dst_strides = []
    for i, loop in enumerate(loops):
        count, src_stride, dst_stride = _normalize_dma_group(
            f"loops[{i}]",
            loop,
            context=context,
        )
        counts.append(count)
        src_strides.append(src_stride)
        dst_strides.append(dst_stride)
    return counts, src_strides, dst_strides


def _normalize_dma_pad(pad, *, context: str):
    if pad is None:
        return None, None, None
    if not isinstance(pad, tuple):
        raise TypeError(f"{context} expects pad to be tuple[ScalarType] or tuple[ScalarType, int, int]")
    if len(pad) == 1:
        pad_value = pad[0]
        left_count = 0
        right_count = 0
    elif len(pad) == 3:
        pad_value, left_count, right_count = pad
    else:
        raise TypeError(f"{context} expects pad to have length 1 or 3")
    return (
        materialize_scalar_literal(pad_value, F32Type.get(), context=f"{context} pad[0]")
        if not hasattr(pad_value, "type") else unwrap_surface_value(pad_value),
        _coerce_i64(left_count, context=f"{context} pad[1]"),
        _coerce_i64(right_count, context=f"{context} pad[2]"),
    )


@_explicit_mode_only("pto.mte_gm_ub(...)")
def mte_gm_ub(source, destination, l2_cache_ctl, len_burst, *, nburst, loops=None, pad=None):
    """``pto.mte_gm_ub`` – grouped GM-to-UB DMA surface."""
    n_burst, nburst_src_stride, nburst_dst_stride = _normalize_dma_group(
        "nburst",
        nburst,
        context="mte_gm_ub(...)",
    )
    loop_counts, loop_src_strides, loop_dst_strides = _normalize_dma_loops(
        loops,
        context="mte_gm_ub(...)",
    )
    pad_value, left_padding_count, right_padding_count = _normalize_dma_pad(
        pad,
        context="mte_gm_ub(...)",
    )
    _pto.MteGmUbOp(
        unwrap_surface_value(source),
        unwrap_surface_value(destination),
        _coerce_i64(l2_cache_ctl, context="mte_gm_ub l2_cache_ctl"),
        _coerce_i64(len_burst, context="mte_gm_ub len_burst"),
        n_burst,
        nburst_src_stride,
        nburst_dst_stride,
        loop_counts,
        loop_src_strides,
        loop_dst_strides,
        pad_value=pad_value,
        left_padding_count=left_padding_count,
        right_padding_count=right_padding_count,
    )


@_explicit_mode_only("pto.mte_ub_gm(...)")
def mte_ub_gm(source, destination, len_burst, *, nburst, loops=None):
    """``pto.mte_ub_gm`` – grouped UB-to-GM DMA surface."""
    n_burst, nburst_src_stride, nburst_dst_stride = _normalize_dma_group(
        "nburst",
        nburst,
        context="mte_ub_gm(...)",
    )
    loop_counts, loop_src_strides, loop_dst_strides = _normalize_dma_loops(
        loops,
        context="mte_ub_gm(...)",
    )
    _pto.MteUbGmOp(
        unwrap_surface_value(source),
        unwrap_surface_value(destination),
        _coerce_i64(len_burst, context="mte_ub_gm len_burst"),
        n_burst,
        nburst_src_stride,
        nburst_dst_stride,
        loop_counts,
        loop_src_strides,
        loop_dst_strides,
    )


@_explicit_mode_only("pto.mte_ub_ub(...)")
def mte_ub_ub(source, destination, len_burst, *, nburst):
    """``pto.mte_ub_ub`` – grouped UB-to-UB DMA surface."""
    n_burst, src_stride, dst_stride = _normalize_dma_group(
        "nburst",
        nburst,
        context="mte_ub_ub(...)",
    )
    _pto.MteUbUbOp(
        unwrap_surface_value(source),
        unwrap_surface_value(destination),
        n_burst,
        _coerce_i64(len_burst, context="mte_ub_ub len_burst"),
        src_stride,
        dst_stride,
    )


@_explicit_mode_only("pto.mte_ub_l1(...)")
def mte_ub_l1(source, destination, len_burst, *, nburst):
    """``pto.mte_ub_l1`` – grouped UB-to-L1 DMA surface."""
    n_burst, src_stride, dst_stride = _normalize_dma_group(
        "nburst",
        nburst,
        context="mte_ub_l1(...)",
    )
    _pto.MteUbL1Op(
        unwrap_surface_value(source),
        unwrap_surface_value(destination),
        n_burst,
        _coerce_i64(len_burst, context="mte_ub_l1 len_burst"),
        src_stride,
        dst_stride,
    )


def mem_bar(barrier_type):
    """``pto.mem_bar`` with a small authored enum surface."""
    barrier_name = getattr(barrier_type, "value", barrier_type)
    _pto.MemBarOp(kind=_membar_attr(barrier_name))


@_explicit_mode_only("pto.mte_l1_l0a(...)")
def mte_l1_l0a(source, destination, m, k, *, transpose=False):
    """``pto.mte_l1_l0a`` – cube-side LEFT staging."""
    _pto.MteL1L0aOp(
        unwrap_surface_value(source),
        unwrap_surface_value(destination),
        _coerce_i64(m, context="mte_l1_l0a m"),
        _coerce_i64(k, context="mte_l1_l0a k"),
        transpose=transpose,
    )


@_explicit_mode_only("pto.mte_l1_l0b(...)")
def mte_l1_l0b(source, destination, k, n, *, transpose=False):
    """``pto.mte_l1_l0b`` – cube-side RIGHT staging."""
    _pto.MteL1L0bOp(
        unwrap_surface_value(source),
        unwrap_surface_value(destination),
        _coerce_i64(k, context="mte_l1_l0b k"),
        _coerce_i64(n, context="mte_l1_l0b n"),
        transpose=transpose,
    )


@_explicit_mode_only("pto.mte_l0c_ub(...)")
def mte_l0c_ub(source, destination, m, n, src_stride, dst_stride, sub_blockid=0, *, dst_mode="single"):
    """``pto.mte_l0c_ub`` – ACC to UB store."""
    _pto.MteL0cUbOp(
        unwrap_surface_value(source),
        unwrap_surface_value(destination),
        _coerce_i64(m, context="mte_l0c_ub m"),
        _coerce_i64(n, context="mte_l0c_ub n"),
        _coerce_i64(src_stride, context="mte_l0c_ub src_stride"),
        _coerce_i64(dst_stride, context="mte_l0c_ub dst_stride"),
        _acc_store_ub_dst_mode_attr(dst_mode),
        sub_blockid=_coerce_i64(sub_blockid, context="mte_l0c_ub sub_blockid"),
    )


def mad(lhs, rhs, dst, m, n, k):
    """``pto.mad`` – cube matmul accumulate."""
    _pto.MadOp(
        unwrap_surface_value(lhs),
        unwrap_surface_value(rhs),
        unwrap_surface_value(dst),
        _coerce_i64(m, context="mad m"),
        _coerce_i64(n, context="mad n"),
        _coerce_i64(k, context="mad k"),
    )


def mad_acc(lhs, rhs, dst, m, n, k):
    """``pto.mad_acc`` – cube matmul accumulate into an existing accumulator."""
    _pto.MadAccOp(
        unwrap_surface_value(lhs),
        unwrap_surface_value(rhs),
        unwrap_surface_value(dst),
        _coerce_i64(m, context="mad_acc m"),
        _coerce_i64(n, context="mad_acc n"),
        _coerce_i64(k, context="mad_acc k"),
    )


def mad_bias(lhs, rhs, dst, bias, m, n, k):
    """``pto.mad_bias`` – cube matmul initialized from a bias buffer."""
    _pto.MadBiasOp(
        unwrap_surface_value(lhs),
        unwrap_surface_value(rhs),
        unwrap_surface_value(dst),
        unwrap_surface_value(bias),
        _coerce_i64(m, context="mad_bias m"),
        _coerce_i64(n, context="mad_bias n"),
        _coerce_i64(k, context="mad_bias k"),
    )


def mad_mx(lhs, rhs, dst, m, n, k):
    """``pto.mad_mx`` – MX-format cube matmul."""
    _pto.MadMxOp(
        unwrap_surface_value(lhs),
        unwrap_surface_value(rhs),
        unwrap_surface_value(dst),
        _coerce_i64(m, context="mad_mx m"),
        _coerce_i64(n, context="mad_mx n"),
        _coerce_i64(k, context="mad_mx k"),
    )


def mad_mx_acc(lhs, rhs, dst, m, n, k):
    """``pto.mad_mx_acc`` – MX-format cube matmul accumulate."""
    _pto.MadMxAccOp(
        unwrap_surface_value(lhs),
        unwrap_surface_value(rhs),
        unwrap_surface_value(dst),
        _coerce_i64(m, context="mad_mx_acc m"),
        _coerce_i64(n, context="mad_mx_acc n"),
        _coerce_i64(k, context="mad_mx_acc k"),
    )


def mad_mx_bias(lhs, rhs, dst, bias, m, n, k):
    """``pto.mad_mx_bias`` – MX-format cube matmul initialized from a bias buffer."""
    _pto.MadMxBiasOp(
        unwrap_surface_value(lhs),
        unwrap_surface_value(rhs),
        unwrap_surface_value(dst),
        unwrap_surface_value(bias),
        _coerce_i64(m, context="mad_mx_bias m"),
        _coerce_i64(n, context="mad_mx_bias n"),
        _coerce_i64(k, context="mad_mx_bias k"),
    )

def get_block_idx():
    """``pto.get_block_idx`` → i64 block index."""
    return wrap_surface_value(_pto.GetBlockIdxOp().result)


def get_block_num():
    """``pto.get_block_num`` → i64 block count."""
    return wrap_surface_value(_pto.GetBlockNumOp().result)


def get_subblock_idx():
    """``pto.get_subblock_idx`` → i64 subblock index."""
    return wrap_surface_value(_pto.GetSubBlockIdxOp().result)


def get_subblock_num():
    """``pto.get_subblock_num`` → i64 subblock count."""
    return wrap_surface_value(_pto.GetSubBlockNumOp().result)


def store_vfsimt_info(dim_z, dim_y, dim_x):
    """``pto.store_vfsimt_info`` – configure the SIMT VF launch descriptor."""
    _pto.StoreVfSimtInfoOp(
        unwrap_surface_value(dim_z),
        unwrap_surface_value(dim_y),
        unwrap_surface_value(dim_x),
    )


def get_tid_x():
    """``pto.get_tid_x`` → i32 SIMT lane X coordinate."""
    return wrap_surface_value(_pto.GetTidXOp().result)


def get_tid_y():
    """``pto.get_tid_y`` → i32 SIMT lane Y coordinate."""
    return wrap_surface_value(_pto.GetTidYOp().result)


def get_tid_z():
    """``pto.get_tid_z`` → i32 SIMT lane Z coordinate."""
    return wrap_surface_value(_pto.GetTidZOp().result)


def pipe_barrier(pipe):
    """``pto.pipe_barrier(pipe)`` – drain the specified hardware pipeline."""
    _pto.BarrierOp(_pipe_attr(pipe))


def get_buf(pipe, buf_id, mode=0):
    """``pto.get_buf(pipe, buf_id, mode=0)`` – acquire a buffer token."""
    _pto.GetBufOp(
        _pipe_attr(pipe),
        buf_id,
        mode=mode,
    )


def rls_buf(pipe, buf_id, mode=0):
    """``pto.rls_buf(pipe, buf_id, mode=0)`` – release a buffer token."""
    _pto.RlsBufOp(
        _pipe_attr(pipe),
        buf_id,
        mode=mode,
    )


def _sync_event_id_operand(event_id, *, context: str):
    _validate_static_event_id(event_id, context=context)
    return event_id if isinstance(event_id, int) else unwrap_surface_value(event_id)


def _flag_event_id_operand(event_id, *, context: str):
    if isinstance(event_id, int):
        _validate_static_event_id(event_id, context=context)
        return event_id, True
    return _coerce_index(event_id, context=context), False


def set_cross_flag(pipe, event_id):
    """``pto.set_cross_flag(pipe, event_id)`` – cross-core sync facade for ``pto.sync.set``."""
    _validate_sync_pipe(pipe, context="set_cross_flag(pipe, event_id)", allowed=("PIPE_FIX",))
    event_operand = _sync_event_id_operand(event_id, context="set_cross_flag(..., event_id=...)")
    _pto.sync_set(_pipe_attr(pipe), event_operand)


def wait_cross_flag(pipe, event_id):
    """``pto.wait_cross_flag(pipe, event_id)`` – cross-core sync facade for ``pto.sync.wait``."""
    _validate_sync_pipe(pipe, context="wait_cross_flag(pipe, event_id)", allowed=("PIPE_FIX",))
    event_operand = _sync_event_id_operand(event_id, context="wait_cross_flag(..., event_id=...)")
    _pto.sync_wait(_pipe_attr(pipe), event_operand)


def set_intra_flag(pipe, event_id):
    """``pto.set_intra_flag(pipe, event_id)`` – intra-block sync facade for ``pto.sync.set``."""
    _validate_sync_pipe(pipe, context="set_intra_flag(pipe, event_id)", allowed=("PIPE_MTE3",))
    event_operand = _sync_event_id_operand(event_id, context="set_intra_flag(..., event_id=...)")
    _pto.sync_set(_pipe_attr(pipe), event_operand)


def wait_intra_flag(pipe, event_id):
    """``pto.wait_intra_flag(pipe, event_id)`` – intra-block sync facade for ``pto.sync.wait``."""
    _validate_sync_pipe(pipe, context="wait_intra_flag(pipe, event_id)", allowed=("PIPE_V",))
    event_operand = _sync_event_id_operand(event_id, context="wait_intra_flag(..., event_id=...)")
    _pto.sync_wait(_pipe_attr(pipe), event_operand)


def set_flag(src: str, dst: str, *, event_id: int = 0):
    """``pto.set_flag[src, dst, event_id]``.

    Accepts short pipe names (``"MTE2"``, ``"V"``, …) or full ``"PIPE_MTE2"``
    names.  Static ``event_id`` values in ``[0, 7]`` lower to ``pto.set_flag``;
    runtime index-like values lower to ``pto.set_flag_dyn``.
    """
    event_operand, is_static = _flag_event_id_operand(
        event_id,
        context="set_flag(..., event_id=...)",
    )
    if is_static:
        _pto.set_flag(_pipe_attr(src), _pipe_attr(dst), _event_attr(event_operand))
        return
    _pto.set_flag_dyn(_pipe_attr(src), _pipe_attr(dst), event_operand)


def wait_flag(src: str, dst: str, *, event_id: int = 0):
    """``pto.wait_flag[src, dst, event_id]``.

    Static ``event_id`` values in ``[0, 7]`` lower to ``pto.wait_flag``;
    runtime index-like values lower to ``pto.wait_flag_dyn``.
    """
    event_operand, is_static = _flag_event_id_operand(
        event_id,
        context="wait_flag(..., event_id=...)",
    )
    if is_static:
        _pto.wait_flag(_pipe_attr(src), _pipe_attr(dst), _event_attr(event_operand))
        return
    _pto.wait_flag_dyn(_pipe_attr(src), _pipe_attr(dst), event_operand)


__all__ = [
    "const",
    "castptr", "addptr",
    "vlds", "vldas", "vldus", "vldsx2", "vsts", "vstsx2",
    "init_align",
    "plt_b8", "plt_b16", "plt_b32",
    "pset_b8", "pset_b16", "pset_b32",
    "pge_b8", "pge_b16", "pge_b32",
    "make_mask",
    "pand", "por", "pxor", "pnot", "psel",
    "pbitcast", "ppack", "punpack",
    "pintlv_b8", "pintlv_b16", "pintlv_b32",
    "pdintlv_b8", "pdintlv_b16", "pdintlv_b32",
    "vgather2", "vgather2_bc", "vgatherb", "vscatter", "vsldb", "vsstb",
    "vcmp", "vcmps",
    "plds", "psts", "pstu", "vstar", "vstas", "vstur", "vstus",
    "vbitcast",
    "vbr",
    "vadd", "vsub", "vmul", "vdiv", "vmax", "vmin",
    "vand", "vor", "vxor", "vshl", "vshr",
    "vcmax", "vcadd", "vcmin", "vdup", "vexpdif",
    "vexp", "vln", "vsqrt", "vabs", "vneg", "vrec", "vrsqrt", "vrelu", "vnot",
    "vcgmax", "vcgadd", "vcgmin", "vcpadd",
    "vadds", "vsubs", "vmuls", "vmaxs", "vmins", "vlrelu",
    "vaxpy", "vaddrelu", "vsubrelu",
    "vsel",
    "make_tensor_view", "partition_view",
    "alloc_tile",
    "tload", "tstore", "tmov",
    "tadd", "tsub", "tmul", "tdiv", "tmax", "tmin",
    "tadds", "tsubs", "tmuls", "tdivs", "tmaxs", "tmins",
    "texp", "tlog", "tsqrt", "trsqrt", "trecip", "tabs", "tneg",
    "trelu", "tlrelu",
    "trowsum", "trowmax", "trowmin", "trowprod", "trowargmax", "trowargmin",
    "tcolsum", "tcolmax", "tcolmin", "tcolprod", "tcolargmax", "tcolargmin",
    "tcmp", "tcmps",
    "texpands", "trowexpand", "tcolexpand",
    "trowexpandadd", "trowexpandsub", "trowexpandmul", "trowexpanddiv", "trowexpandmax", "trowexpandmin", "trowexpandexpdif",
    "tcolexpandadd", "tcolexpandsub", "tcolexpandmul", "tcolexpanddiv", "tcolexpandmax", "tcolexpandmin", "tcolexpandexpdif",
    "tsel", "tsels", "tcvt",
    "tnot", "tand", "tands", "tor", "tors", "txor", "txors", "tshl", "tshls", "tshr", "tshrs",
    "tpartadd", "tpartmul", "tpartmax", "tpartmin",
    "tfillpad", "tfillpad_expand", "tfillpad_inplace",
    "as_ptr",
    "mte_load", "mte_store", "mte_gm_ub", "mte_ub_gm", "mte_ub_ub", "mte_ub_l1", "mem_bar",
    "mte_l1_l0a", "mte_l1_l0b", "mte_l0c_ub",
    "mad", "mad_acc", "mad_bias", "mad_mx", "mad_mx_acc", "mad_mx_bias",
    "get_block_idx", "get_block_num", "get_subblock_idx", "get_subblock_num",
    "store_vfsimt_info", "get_tid_x", "get_tid_y", "get_tid_z",
    "pipe_barrier", "get_buf", "rls_buf",
    "set_cross_flag", "wait_cross_flag", "set_intra_flag", "wait_intra_flag",
    "set_flag", "wait_flag",
]
