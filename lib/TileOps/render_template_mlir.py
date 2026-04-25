# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

#!/usr/bin/env python3
"""Materialize a TileLang DSL library template to authoring-form MLIR.

Examples:
  python3 lib/TileOps/render_template_mlir.py lib/TileOps/tload_template.py
  python3 lib/TileOps/render_template_mlir.py lib/TileOps/tadd_template.py --tile dst=8x64@ub --tile src0=8x64@ub --tile src1=8x64@ub
  python3 lib/TileOps/render_template_mlir.py lib/TileOps/tload_template.py --dtypes f16,f16 -o /tmp/tload.mlir
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from types import ModuleType


REPO_ROOT = Path(__file__).resolve().parents[2]
TILELANG_PYTHON_DIR = REPO_ROOT / "tilelang-dsl" / "python"
if str(TILELANG_PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(TILELANG_PYTHON_DIR))

import tilelang_dsl as pto


_DTYPE_BY_NAME = {
    "i1": pto.i1,
    "i8": pto.i8,
    "i16": pto.i16,
    "i32": pto.i32,
    "i64": pto.i64,
    "f16": pto.f16,
    "bf16": pto.bf16,
    "f32": pto.f32,
}
_MEMORY_SPACE_BY_NAME = {
    "gm": pto.MemorySpace.GM,
    "ub": pto.MemorySpace.UB,
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load a TileLang DSL template file and emit its corresponding MLIR text.",
    )
    parser.add_argument("template", help="Path to the template Python file")
    parser.add_argument(
        "--kernel",
        help="Descriptor symbol name inside the module when the file defines multiple @pto.vkernel templates",
    )
    parser.add_argument(
        "--op",
        help="Concrete op to bind when the descriptor matches multiple ops; defaults to the first match op",
    )
    parser.add_argument(
        "--dtypes",
        help="Concrete operand dtypes as a comma-separated list, for example: f32,f32 or f16,f16,f16",
    )
    parser.add_argument(
        "--tile",
        action="append",
        default=[],
        metavar="PARAM=SHAPE[@SPACE][:VALID]",
        help=(
            "Tile specialization override, for example: dst=16x32@ub or "
            "dst=16x32@ub:8x32. May be repeated."
        ),
    )
    parser.add_argument(
        "--default-tile-shape",
        default="16x32",
        help="Default shape for every bare Tile parameter when no --tile override is given",
    )
    parser.add_argument(
        "--default-tile-space",
        default="ub",
        choices=sorted(_MEMORY_SPACE_BY_NAME),
        help="Default memory space for every bare Tile parameter",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Optional output path; defaults to stdout",
    )
    return parser.parse_args()


def _load_module(template_path: Path) -> ModuleType:
    module_name = f"_tileops_template_{template_path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, template_path)
    if spec is None or spec.loader is None:
        raise ValueError(f"failed to load Python module from {template_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _find_descriptors(module: ModuleType) -> dict[str, pto.VKernelDescriptor]:
    descriptors: dict[str, pto.VKernelDescriptor] = {}
    for name, value in vars(module).items():
        if isinstance(value, pto.VKernelDescriptor):
            descriptors[name] = value
    return descriptors


def _select_descriptor(
    descriptors: dict[str, pto.VKernelDescriptor],
    kernel_name: str | None,
) -> tuple[str, pto.VKernelDescriptor]:
    if not descriptors:
        raise ValueError("no @pto.vkernel descriptor found in the template module")
    if kernel_name is not None:
        descriptor = descriptors.get(kernel_name)
        if descriptor is None:
            available = ", ".join(sorted(descriptors))
            raise ValueError(
                f"kernel {kernel_name!r} was not found in the template module; available descriptors: {available}"
            )
        return kernel_name, descriptor
    if len(descriptors) == 1:
        return next(iter(descriptors.items()))
    available = ", ".join(sorted(descriptors))
    raise ValueError(
        "the template module defines multiple @pto.vkernel descriptors; "
        f"please pass --kernel. Available descriptors: {available}"
    )


def _parse_dtype_list(text: str) -> tuple[pto.ScalarType, ...]:
    parts = [part.strip() for part in text.split(",") if part.strip()]
    if not parts:
        raise ValueError("--dtypes must contain at least one dtype")
    try:
        return tuple(_DTYPE_BY_NAME[part] for part in parts)
    except KeyError as exc:
        available = ", ".join(sorted(_DTYPE_BY_NAME))
        raise ValueError(
            f"unsupported dtype {exc.args[0]!r}; available dtypes: {available}"
        ) from exc


def _default_concrete_dtype(pattern: object) -> pto.ScalarType:
    if isinstance(pattern, pto.ScalarType):
        return pattern
    if isinstance(pattern, pto.WildcardType):
        if pattern.name in {"AnyType", "AnyFloat"}:
            return pto.f32
        if pattern.name == "AnyInt":
            return pto.i32
        if pattern.name == "AnyMask":
            return pto.i1
        raise ValueError(f"unsupported wildcard dtype pattern {pattern!r}")
    if isinstance(pattern, pto.TypeVariable):
        return pto.f32
    raise ValueError(f"unsupported dtype pattern {pattern!r}")


def _default_operand_types(descriptor: pto.VKernelDescriptor) -> tuple[pto.ScalarType, ...]:
    if not descriptor.dtypes:
        raise ValueError("descriptor does not declare any dtype signatures")
    prototype = descriptor.dtypes[0]
    typevar_bindings: dict[str, pto.ScalarType] = {}
    concrete: list[pto.ScalarType] = []
    for pattern in prototype:
        if isinstance(pattern, pto.TypeVariable):
            bound = typevar_bindings.get(pattern.name)
            if bound is None:
                bound = pto.f32
                typevar_bindings[pattern.name] = bound
            concrete.append(bound)
            continue
        concrete.append(_default_concrete_dtype(pattern))
    return tuple(concrete)


def _bind_descriptor(
    descriptor: pto.VKernelDescriptor,
    *,
    op_name: str | None,
    operand_types: tuple[pto.ScalarType, ...] | None,
) -> pto.VKernelDescriptor:
    concrete_op = op_name
    if concrete_op is None:
        if descriptor.selected_op is not None:
            concrete_op = descriptor.selected_op
        elif len(descriptor.match_ops) == 1:
            concrete_op = descriptor.match_ops[0]
        else:
            available = ", ".join(descriptor.match_ops)
            raise ValueError(
                f"descriptor matches multiple ops; pass --op. Available ops: {available}"
            )

    concrete_operand_types = operand_types
    if concrete_operand_types is None:
        if descriptor._selected_dtype_signature is not None:
            concrete_operand_types = descriptor._selected_dtype_signature
        else:
            concrete_operand_types = _default_operand_types(descriptor)

    registry = pto.KernelRegistry((descriptor,))
    return pto.select_kernel(
        target=descriptor.target,
        op=concrete_op,
        operand_types=concrete_operand_types,
        registry=registry,
    )


def _parse_shape(text: str) -> tuple[int, ...]:
    dims = []
    for part in text.split("x"):
        part = part.strip()
        if not part:
            raise ValueError(f"invalid shape {text!r}")
        value = int(part)
        if value <= 0:
            raise ValueError(f"shape dimensions must be positive integers, got {text!r}")
        dims.append(value)
    if not dims:
        raise ValueError(f"invalid shape {text!r}")
    return tuple(dims)


def _parse_tile_override(spec_text: str) -> tuple[str, pto.TileSpecialization]:
    if "=" not in spec_text:
        raise ValueError(
            f"invalid --tile value {spec_text!r}; expected PARAM=SHAPE[@SPACE][:VALID]"
        )
    param_name, payload = spec_text.split("=", 1)
    param_name = param_name.strip()
    payload = payload.strip()
    if not param_name:
        raise ValueError(f"invalid --tile value {spec_text!r}; missing parameter name")

    valid_shape = None
    if ":" in payload:
        payload, valid_text = payload.split(":", 1)
        valid_shape = _parse_shape(valid_text.strip())

    memory_space = pto.MemorySpace.UB
    if "@" in payload:
        shape_text, memory_space_text = payload.split("@", 1)
        memory_space_key = memory_space_text.strip().lower()
        try:
            memory_space = _MEMORY_SPACE_BY_NAME[memory_space_key]
        except KeyError as exc:
            available = ", ".join(sorted(_MEMORY_SPACE_BY_NAME))
            raise ValueError(
                f"unsupported memory space {memory_space_text!r}; available spaces: {available}"
            ) from exc
    else:
        shape_text = payload

    shape = _parse_shape(shape_text.strip())
    if valid_shape is not None and len(valid_shape) != len(shape):
        raise ValueError(
            f"valid_shape rank {len(valid_shape)} does not match shape rank {len(shape)} for {param_name!r}"
        )
    return (
        param_name,
        pto.TileSpecialization(
            shape=shape,
            memory_space=memory_space,
            valid_shape=valid_shape,
        ),
    )


def _default_tile_specialization(
    *,
    shape: tuple[int, ...],
    memory_space: pto.MemorySpace,
) -> pto.TileSpecialization:
    return pto.TileSpecialization(shape=shape, memory_space=memory_space)


def _specialize_tiles(
    descriptor: pto.VKernelDescriptor,
    *,
    tile_overrides: dict[str, pto.TileSpecialization],
    default_shape: tuple[int, ...],
    default_memory_space: pto.MemorySpace,
) -> pto.VKernelDescriptor:
    if not descriptor.tile_parameters:
        return descriptor

    specializations: dict[str, pto.TileSpecialization] = {}
    for param in descriptor.tile_parameters:
        specializations[param.name] = tile_overrides.get(
            param.name,
            _default_tile_specialization(
                shape=default_shape,
                memory_space=default_memory_space,
            ),
        )
    return descriptor.specialize(**specializations)


def _emit_output(text: str, output_path: str | None) -> None:
    if output_path is None:
        sys.stdout.write(text)
        if not text.endswith("\n"):
            sys.stdout.write("\n")
        return
    path = Path(output_path)
    path.write_text(text, encoding="utf-8")


def main() -> int:
    args = _parse_args()
    template_path = Path(args.template).resolve()
    if not template_path.is_file():
        print(f"error: template file not found: {template_path}", file=sys.stderr)
        return 1

    try:
        module = _load_module(template_path)
        _, descriptor = _select_descriptor(_find_descriptors(module), args.kernel)
        operand_types = None if args.dtypes is None else _parse_dtype_list(args.dtypes)
        bound = _bind_descriptor(
            descriptor,
            op_name=args.op,
            operand_types=operand_types,
        )
        tile_overrides = dict(_parse_tile_override(spec_text) for spec_text in args.tile)
        specialized = _specialize_tiles(
            bound,
            tile_overrides=tile_overrides,
            default_shape=_parse_shape(args.default_tile_shape),
            default_memory_space=_MEMORY_SPACE_BY_NAME[args.default_tile_space],
        )
        _emit_output(specialized.mlir_text(), args.output)
        return 0
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
