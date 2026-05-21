# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""``@pto.jit`` decorator and compiled-kernel handles."""

from __future__ import annotations

from ._kernel_compilation import CompiledKernelHandle, KernelCompiler
from ._kernel_signature import parse_jit_kernel_signature
from ._tracing import (
    KernelModuleSpec,
    ModuleArtifact,
    ModuleStyle,
)

from mlir.ir import InsertionPoint


_MODULE_ATTRS = ("pto.target_arch", "pto.kernel_kind")


def _module_attr_map(module):
    attrs = module.operation.attributes
    return {name: str(attrs[name]) for name in _MODULE_ATTRS if name in attrs}


def merge_jit_modules(*kernels: KernelHandle):
    """
    Merge multiple ``@pto.jit`` flat-module kernels into one MLIR module.

    Each handle must have been compiled with the same ``target`` and
    ``kernel_kind`` module attributes.  Function order follows *kernels*.
    """
    if not kernels:
        raise ValueError("merge_jit_modules() requires at least one kernel handle")

    merged = kernels[0].build()
    expected_attrs = _module_attr_map(merged)

    for kernel in kernels[1:]:
        module = kernel.build()
        actual_attrs = _module_attr_map(module)
        if actual_attrs != expected_attrs:
            raise ValueError(
                "merge_jit_modules() requires compatible module attributes; "
                f"expected {expected_attrs}, got {actual_attrs}"
            )
        with InsertionPoint(merged.body):
            for op in module.body.operations:
                op.operation.clone()

    merged.operation.verify()
    return merged


def jit(
    name=None,
    *,
    target: str = "a5",
    kernel_kind: str = "vector",
    func_attr: str = None,
):
    """
    Decorator that wraps a Python function as a PTODSL JIT kernel template.

    Parameters
    ----------
    name:        IR function name (defaults to the Python function name).
    target:      Target architecture string, e.g. ``"a5"``.
    kernel_kind: ``"vector"`` or ``"cube"`` – sets ``pto.kernel_kind``.
    func_attr:   Optional function attribute.  Pass ``"pto.aicore"`` to
                 select the flat-module structure with the aicore attribute.

    The decorated function is replaced by a :class:`KernelHandle` that:

    - supports ``my_kernel.compile(**constexprs)`` specialization,
    - prints as the default-specialization MLIR text,
    - exposes ``my_kernel.mlir_module()`` / ``verify()`` / ``emit()`` on the
      default specialization for convenience.
    """

    def decorator(fn):
        fn_name = name or fn.__name__
        kernel_signature = parse_jit_kernel_signature(fn)
        module_style = (
            ModuleStyle.FLAT_AICORE
            if func_attr == "pto.aicore"
            else ModuleStyle.NESTED
        )
        compiler = KernelCompiler(
            fn.__name__,
            KernelModuleSpec(
                function_name=fn_name,
                target_arch=target,
                kernel_kind=kernel_kind,
                module_style=module_style,
            ),
            kernel_signature,
            fn,
        )
        return KernelHandle(fn.__name__, compiler)

    return decorator


class KernelHandle(ModuleArtifact):
    """
    Represents a JIT kernel template plus its compiled specializations.

    ``handle.compile(**constexprs)`` returns one compiled specialization.
    ``print(handle)`` emits the default-specialization MLIR module text.
    """

    def __init__(self, py_name: str, compiler: KernelCompiler):
        self._compiler = compiler
        super().__init__(py_name, module_factory=self._build_default_module)

    def compile(self, **constexpr_bindings) -> CompiledKernelHandle:
        return self._compiler.compile(**constexpr_bindings)

    def cached_specializations(self):
        return self._compiler.cached_specializations()

    def _build_default_module(self):
        return self.compile().build()


__all__ = ["CompiledKernelHandle", "jit", "KernelHandle", "merge_jit_modules"]
