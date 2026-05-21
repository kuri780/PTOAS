# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Kernel specialization and compilation helpers for ``@pto.jit``."""

from __future__ import annotations

from ._runtime.launch import LaunchHandle, parse_launch_spec
from ._tracing import ModuleArtifact, SignatureTracingRuntime


class CompiledKernelHandle(ModuleArtifact):
    """One compiled ``@pto.jit`` specialization."""

    def __init__(
        self,
        py_name: str,
        *,
        specialization_key,
        constexpr_bindings,
        module_factory,
        module_spec,
        kernel_signature,
    ):
        super().__init__(py_name, module_factory=module_factory)
        self._specialization_key = specialization_key
        self._constexpr_bindings = dict(constexpr_bindings)
        self._module_spec = module_spec
        self._kernel_signature = kernel_signature

    @property
    def specialization_key(self):
        return self._specialization_key

    @property
    def constexpr_bindings(self):
        return dict(self._constexpr_bindings)

    @property
    def ir_function_name(self):
        return self._module_spec.function_name

    def __getitem__(self, launch_spec):
        grid, stream = parse_launch_spec(launch_spec)
        return LaunchHandle(self, grid, stream)


class KernelCompiler:
    """Per-kernel specialization cache and module builder."""

    def __init__(self, py_name: str, module_spec, kernel_signature, callback):
        self._py_name = py_name
        self._module_spec = module_spec
        self._kernel_signature = kernel_signature
        self._callback = callback
        self._kernel_identity = id(callback)
        self._compiled_cache = {}

    def compile(self, **constexpr_bindings):
        normalized_bindings = self._kernel_signature.bind_constexpr_bindings(constexpr_bindings)
        specialization_key = self._kernel_signature.specialization_key(
            self._kernel_identity,
            normalized_bindings,
        )

        cached = self._compiled_cache.get(specialization_key)
        if cached is not None:
            return cached

        runtime = SignatureTracingRuntime(
            self._module_spec,
            self._kernel_signature,
            self._callback,
            constexpr_bindings=normalized_bindings,
        )
        compiled = CompiledKernelHandle(
            self._py_name,
            specialization_key=specialization_key,
            constexpr_bindings=normalized_bindings,
            module_factory=runtime.build_module,
            module_spec=self._module_spec,
            kernel_signature=self._kernel_signature,
        )
        compiled.build()
        self._compiled_cache[specialization_key] = compiled
        return compiled

    def cached_specializations(self):
        return tuple(self._compiled_cache.values())


__all__ = [
    "CompiledKernelHandle",
    "KernelCompiler",
]
