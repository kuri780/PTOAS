# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Launch handles and ctypes dispatch for compiled PTODSL kernels."""

from __future__ import annotations

import ctypes
from typing import TYPE_CHECKING

from .._host_tensors import (
    inspect_host_tensor_metadata,
    looks_like_host_tensor,
)
from .._kernel_signature import DeviceParameterSpec, TensorSpecParameterSpec
from .native_build import build_native_library

if TYPE_CHECKING:
    from .._kernel_compilation import CompiledKernelHandle


def _normalize_stream_ptr(stream):
    if stream is None:
        try:
            import torch
        except ImportError as exc:
            raise ImportError(
                "stream=None requires torch; install torch and torch_npu for default-stream launch"
            ) from exc
        return torch.npu.current_stream()._as_parameter_  # noqa: SLF001

    if isinstance(stream, ctypes.c_void_p):
        return stream
    if isinstance(stream, int):
        return ctypes.c_void_p(stream)
    if hasattr(stream, "value"):
        return ctypes.c_void_p(int(stream.value))
    return stream


def _as_void_ptr(value):
    if isinstance(value, ctypes.c_void_p):
        return value
    if hasattr(value, "data_ptr"):
        return ctypes.c_void_p(value.data_ptr())
    if isinstance(value, int):
        return ctypes.c_void_p(value)
    raise TypeError(f"expected a pointer-like launch argument, got {type(value)!r}")


def _marshal_launch_args(kernel_signature, args):
    if len(args) != len(kernel_signature.positional_parameters):
        raise TypeError(
            f"expected {len(kernel_signature.positional_parameters)} launch argument(s), "
            f"got {len(args)}"
        )

    marshaled = []
    for param, value in zip(kernel_signature.positional_parameters, args):
        if isinstance(param, DeviceParameterSpec):
            marshaled.append(_as_void_ptr(value))
            continue
        if isinstance(param, TensorSpecParameterSpec):
            if not looks_like_host_tensor(value):
                raise TypeError(
                    f"launch argument '{param.name}' expects a Python-native tensor-like object"
                )
            meta = inspect_host_tensor_metadata(value)
            marshaled.append(_as_void_ptr(meta.data_handle))
            for dim in meta.shape:
                marshaled.append(ctypes.c_int64(dim))
            for dim in meta.strides:
                marshaled.append(ctypes.c_int64(dim))
            continue
        raise TypeError(f"unsupported launch parameter spec: {param!r}")
    return marshaled


class LaunchHandle:
    """Callable launch binding returned by ``compiled[grid, stream]``."""

    def __init__(self, compiled: CompiledKernelHandle, grid: int, stream):
        if not isinstance(grid, int) or grid <= 0:
            raise ValueError("launch grid must be a positive integer")
        self._compiled = compiled
        self._grid = grid
        self._stream = stream
        self._launch_fn = None
        self._launch_symbol = None

    def _ensure_launch_fn(self):
        if self._launch_fn is not None:
            return

        lib_path, launch_symbol = build_native_library(
            py_name=self._compiled._py_name,
            module_spec=self._compiled._module_spec,
            kernel_signature=self._compiled._kernel_signature,
            mlir_text=self._compiled.mlir_text(),
            specialization_key=self._compiled.specialization_key,
        )
        lib = ctypes.CDLL(str(lib_path))
        fn = getattr(lib, launch_symbol)
        fn.argtypes = _launch_argtypes(self._compiled._kernel_signature)
        fn.restype = None
        self._launch_fn = fn
        self._launch_symbol = launch_symbol

    def __call__(self, *args):
        self._ensure_launch_fn()
        marshaled = _marshal_launch_args(self._compiled._kernel_signature, args)
        self._launch_fn(
            ctypes.c_uint32(self._grid),
            _normalize_stream_ptr(self._stream),
            *marshaled,
        )


def _launch_argtypes(kernel_signature):
    argtypes = [ctypes.c_uint32, ctypes.c_void_p]
    for param in kernel_signature.positional_parameters:
        if isinstance(param, DeviceParameterSpec):
            argtypes.append(ctypes.c_void_p)
            continue
        if isinstance(param, TensorSpecParameterSpec):
            argtypes.append(ctypes.c_void_p)
            rank = param.tensor_spec.rank
            argtypes.extend([ctypes.c_int64] * (rank + rank))
            continue
        raise TypeError(f"unsupported launch parameter spec: {param!r}")
    return argtypes


def parse_launch_spec(launch_spec) -> tuple[int, object]:
    if not isinstance(launch_spec, tuple) or len(launch_spec) != 2:
        raise TypeError(
            "compiled launch syntax expects compiled[grid, stream]; "
            f"got {type(launch_spec)!r} with length "
            f"{len(launch_spec) if isinstance(launch_spec, tuple) else 'n/a'}"
        )
    grid, stream = launch_spec
    return grid, stream


__all__ = [
    "LaunchHandle",
    "parse_launch_spec",
]
