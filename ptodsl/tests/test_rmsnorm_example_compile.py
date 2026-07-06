#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import re
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "ptodsl"))

from mlir.ir import Module
from ptodsl._bootstrap import make_context


def expect(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def expect_raises(exc_type, func, message_substring: str | None = None) -> Exception:
    try:
        func()
    except exc_type as exc:
        if message_substring is not None and message_substring not in str(exc):
            raise AssertionError(
                f"expected {exc_type.__name__} containing {message_substring!r}, got {exc!r}"
            ) from exc
        return exc
    except Exception as exc:
        raise AssertionError(
            f"expected {exc_type.__name__}, got {exc.__class__.__name__}: {exc}"
        ) from exc
    raise AssertionError(f"expected {exc_type.__name__} to be raised")


def expect_parse_roundtrip_and_verify(text: str, label: str) -> None:
    with make_context() as ctx:
        parsed = Module.parse(text, ctx)
        parsed.operation.verify()
        roundtrip_text = str(parsed)
    expect(
        roundtrip_text == text,
        f"{label} should survive Module.parse(...) round-trip without textual drift",
    )


def load_rmsnorm_example():
    example_path = REPO_ROOT / "ptodsl" / "examples" / "rms_norm" / "rmsnorm_alloc_buffer_simt.py"
    expect(example_path.is_file(), f"RMSNorm example is missing: {example_path}")

    spec = spec_from_file_location("ptodsl_rmsnorm_alloc_buffer_simt", example_path)
    expect(spec is not None and spec.loader is not None, f"unable to create import spec for {example_path}")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check_variant(compiled, *, label: str, vector_type: str) -> None:
    compiled.verify()
    text = compiled.mlir_text()
    expect_parse_roundtrip_and_verify(text, f"RMSNorm {label} MLIR")

    expect("func.func @rmsnorm_4096_alloc_buffer_simt_context_kernel" in text, f"{label}: missing entry")
    expect("scf.for" in text, f"{label}: tokens_per_core loop should lower to scf.for")
    expect("pto.mte_gm_ub" in text, f"{label}: missing GM->UB transfer")
    expect("pto.mte_ub_gm" in text, f"{label}: missing UB->GM transfer")
    expect("pto.simt_launch @rmsnorm_simt_token_body__simt_" in text,
           f"{label}: indexed SIMT call should lower to an explicit token simt_launch op")
    expect("pto.simt_launch @inline_simt_" not in text,
           f"{label}: token SIMT body should be emitted as the named helper, not an inline helper")
    expect("pto.store_vfsimt_info" not in text,
           f"{label}: explicit simt_launch dims should not emit caller-side store_vfsimt_info")
    expect("pto.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID3>]" in text,
           f"{label}: W load should signal completion before token processing")
    expect("pto.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID3>]" in text,
           f"{label}: token processing should start after the W load completes")
    expect("pto.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]" in text,
           f"{label}: missing V->MTE2 ping-pong priming flag")
    expect("pto.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID1>]" in text,
           f"{label}: missing MTE3->V pong priming flag")
    expect("pto.set_flag_dyn" in text, f"{label}: token loop should lower dynamic set_flag ops")
    expect("pto.wait_flag_dyn" in text, f"{label}: token loop should lower dynamic wait_flag ops")
    expect(vector_type in text, f"{label}: missing contiguous vector access type {vector_type}")
    expect("__tl_allreduce_sum" not in text,
           f"{label}: PR3 allreduce should inline the reduce sequence into the SIMT body")
    expect("pto.redux_add" in text, f"{label}: PR3 inline allreduce should use redux_add")
    expect("pto.syncthreads" in text, f"{label}: PR3 inline allreduce should synchronize through UB scratch")
    expect("pto.sqrt" in text, f"{label}: RMSNorm runtime sqrt should lower through the PTO SIMT sqrt op")
    expect("math.sqrt" not in text, f"{label}: RMSNorm SIMT helper should not leave math.sqrt in the MLIR")

    expect("w_frag" not in text, f"{label}: W should be read directly from UB, not from a local fragment")
    expect(
        re.search(
            r"func\.func @rmsnorm_simt_token_body__simt_[^{]+\{(?:(?!func\.func @).)*"
            r"llvm\.alloca(?:(?!func\.func @).)*llvm\.alloca",
            text,
            re.S,
        )
        is not None,
        f"{label}: x_frag and sum_sq should be allocated inside the token SIMT helper",
    )
    expect(
        re.search(
            rf"llvm\.insertelement .* : {re.escape(vector_type)}(?:(?!func\.func @).)*"
            rf"arith\.mulf .* : {re.escape(vector_type)}(?:(?!func\.func @).)*"
            rf"arith\.mulf .* : {re.escape(vector_type)}(?:(?!func\.func @).)*"
            rf"llvm\.store .* : {re.escape(vector_type)}",
            text,
            re.S,
        )
        is not None,
        f"{label}: y = x * rstd * w should lower as vector broadcast/mul/store",
    )


def main() -> None:
    example = load_rmsnorm_example()

    expect(hasattr(example, "build_x128"), "RMSNorm example should export build_x128()")
    expect(hasattr(example, "build_x64"), "RMSNorm example should export build_x64()")
    expect_raises(
        AssertionError,
        lambda: example.rmsnorm_4096_alloc_buffer_simt_context_kernel.compile(
            threads=128,
            rounds=16,
            lanes=2,
            hidden_size=4097,
        ),
        "threads * rounds * lanes must equal hidden_size",
    )

    check_variant(
        example.build_x128(),
        label="x128",
        vector_type="vector<4xf32>",
    )
    check_variant(
        example.build_x64(),
        label="x64",
        vector_type="vector<4xf32>",
    )

    print("ptodsl_rmsnorm_example_compile: PASS")


if __name__ == "__main__":
    main()
