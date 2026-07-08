#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# Test wheel installation by verifying the installed Python contract works.
#
# Usage: ./test_wheel_imports.sh
#
# This script tests that the installed wheel can import:
#   - mlir.ir
#   - mlir.dialects.pto
#   - ptodsl
#   - from ptodsl import pto, scalar
# and that a minimal PTODSL compile-only probe succeeds.

set -euo pipefail

if [[ -n "${PYTHON:-}" ]]; then
  PYTHON_BIN="${PYTHON}"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="python3"
else
  PYTHON_BIN="python"
fi

echo "Testing wheel imports..."

# Test in a clean directory to avoid local imports
cd /tmp

echo "Testing mlir.ir import..."
"$PYTHON_BIN" -c "import mlir.ir; print('mlir.ir imported successfully')"

echo "Testing pto dialect import..."
"$PYTHON_BIN" -c "from mlir.dialects import pto; print('pto dialect imported successfully')"

echo "Testing ptodsl import..."
"$PYTHON_BIN" -c "import ptodsl; print(f'ptodsl imported successfully from {ptodsl.__file__}')"

echo "Testing ptodsl public imports..."
"$PYTHON_BIN" -c "from ptodsl import pto, scalar; print('ptodsl public imports imported successfully')"

echo "Testing PTODSL compile-only probe..."
"$PYTHON_BIN" - <<'PY'
from ptodsl import pto, scalar


@pto.jit(target="a5")
def wheel_compile_probe(
    A_ptr: pto.ptr(pto.f32, "gm"),
    O_ptr: pto.ptr(pto.f32, "gm"),
    rows: pto.i32,
    cols: pto.i32,
    *,
    BLOCK: pto.const_expr = 128,
):
    a_view = pto.make_tensor_view(A_ptr, shape=[rows, cols], strides=[cols, 1])
    o_view = pto.make_tensor_view(O_ptr, shape=[rows, cols], strides=[cols, 1])
    tile = pto.alloc_tile(shape=[1, BLOCK], dtype=pto.f32)
    src = pto.partition_view(a_view, offsets=[0, 0], sizes=[rows, cols])
    dst = pto.partition_view(o_view, offsets=[0, 0], sizes=[rows, cols])
    pto.tile.load(src, tile)
    pto.tile.store(tile, dst)


mlir_text = wheel_compile_probe.compile().mlir_text()
if "func.func @wheel_compile_probe" not in mlir_text:
    raise SystemExit("PTODSL compile probe did not preserve the kernel symbol")
if "pto.tload" not in mlir_text or "pto.tstore" not in mlir_text:
    raise SystemExit("PTODSL compile probe did not emit the expected tile ops")
print("PTODSL compile probe succeeded")
PY

echo "All wheel import tests passed!"
