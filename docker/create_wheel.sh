#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

set -euo pipefail

for var in PTO_SOURCE_DIR PTO_INSTALL_DIR LLVM_BUILD_DIR; do
  if [[ -z "${!var:-}" ]]; then
    echo "Error: $var environment variable is not set" >&2
    exit 1
  fi
done

MLIR_PYTHON_PACKAGE_DIR="${LLVM_BUILD_DIR}/tools/mlir/python_packages/mlir_core"
WHEEL_STAGING_DIR="${PTO_WHEEL_STAGING_DIR:-${PTO_SOURCE_DIR}/build/wheel-staging}"
WHEEL_DIST_DIR="${PTO_WHEEL_DIST_DIR:-${PTO_SOURCE_DIR}/build/wheel-dist}"
PYTHON_BIN="${PYTHON:-python3}"
PTOAS_PYTHON_PACKAGE_VERSION="${PTOAS_PYTHON_PACKAGE_VERSION:-${PTOAS_VERSION:-}}"
PTODSL_INSTALL_DIR="${PTO_INSTALL_DIR}/ptodsl"

if [[ -z "${PTOAS_PYTHON_PACKAGE_VERSION}" ]]; then
  PTOAS_PYTHON_PACKAGE_VERSION="$("${PYTHON_BIN}" "${PTO_SOURCE_DIR}/.github/scripts/compute_ptoas_version.py" \
    --cmake-file "${PTO_SOURCE_DIR}/CMakeLists.txt" --mode dev)"
fi
export PTOAS_PYTHON_PACKAGE_VERSION

echo "Creating Python wheel..."
echo "Wheel package version: ${PTOAS_PYTHON_PACKAGE_VERSION}"

rm -rf "${WHEEL_STAGING_DIR}" "${WHEEL_DIST_DIR}"
mkdir -p "${WHEEL_STAGING_DIR}" "${WHEEL_DIST_DIR}"

echo "Copying MLIR Python package into wheel staging..."
cp -a "${MLIR_PYTHON_PACKAGE_DIR}/." "${WHEEL_STAGING_DIR}/"

echo "Overlaying PTO dialect files..."
mkdir -p "${WHEEL_STAGING_DIR}/mlir/dialects"
find "${PTO_INSTALL_DIR}/mlir/dialects" -maxdepth 1 -type f -name '*.py' -exec cp {} "${WHEEL_STAGING_DIR}/mlir/dialects/" \;

echo "Overlaying PTO native extension..."
mkdir -p "${WHEEL_STAGING_DIR}/mlir/_mlir_libs"
find "${PTO_INSTALL_DIR}/mlir/_mlir_libs" -maxdepth 1 -type f -name '_pto*' -exec cp {} "${WHEEL_STAGING_DIR}/mlir/_mlir_libs/" \;

echo "Copying TileLang resources..."
rm -rf "${WHEEL_STAGING_DIR}/tilelang_dsl" "${WHEEL_STAGING_DIR}/TileOps"
cp -R "${PTO_INSTALL_DIR}/tilelang_dsl" "${WHEEL_STAGING_DIR}/tilelang_dsl"
cp -R "${PTO_INSTALL_DIR}/share/ptoas/TileOps" "${WHEEL_STAGING_DIR}/TileOps"

echo "Copying ptodsl package..."
if [[ ! -d "${PTODSL_INSTALL_DIR}" ]]; then
  echo "Error: ptodsl package directory not found at ${PTODSL_INSTALL_DIR}" >&2
  exit 1
fi
if [[ ! -f "${PTODSL_INSTALL_DIR}/__init__.py" ]]; then
  echo "Error: ptodsl package is missing ${PTODSL_INSTALL_DIR}/__init__.py" >&2
  exit 1
fi
rm -rf "${WHEEL_STAGING_DIR}/ptodsl"
cp -R "${PTODSL_INSTALL_DIR}" "${WHEEL_STAGING_DIR}/ptodsl"

echo "Removing packaging residue..."
find "${WHEEL_STAGING_DIR}" \( -name '*.egg-info' -o -name '*.dist-info' \) -prune -exec rm -rf {} +
find "${WHEEL_STAGING_DIR}" -name '__pycache__' -prune -exec rm -rf {} +

export PTO_WHEEL_STAGING_DIR="${WHEEL_STAGING_DIR}"
export PTO_WHEEL_DIST_DIR="${WHEEL_DIST_DIR}"
export PTOAS_PYTHON_PACKAGE_VERSION

echo "Building wheel archive directly..."
"${PYTHON_BIN}" - <<'PY'
import base64
import csv
import hashlib
import os
import platform
import sys
import zipfile
from pathlib import Path

staging = Path(os.environ["PTO_WHEEL_STAGING_DIR"])
dist = Path(os.environ["PTO_WHEEL_DIST_DIR"])
version = os.environ["PTOAS_PYTHON_PACKAGE_VERSION"]

py_tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
if sys.platform == "darwin":
    platform_tag = os.environ.get("WHEEL_PLAT_NAME") or "macosx_11_0_arm64"
else:
    arch = platform.machine().lower().replace("-", "_")
    platform_tag = os.environ.get("WHEEL_PLAT_NAME") or f"linux_{arch}"

wheel_name = f"ptoas-{version}-{py_tag}-{py_tag}-{platform_tag}.whl"
wheel_path = dist / wheel_name
dist_info = f"ptoas-{version}.dist-info"

metadata = "\n".join([
    "Metadata-Version: 2.1",
    "Name: ptoas",
    f"Version: {version}",
    "Summary: PTO Assembler & Optimizer",
    "Requires-Python: >=3.9",
    "License: Apache-2.0",
    "Requires-Dist: numpy",
    "",
]).encode("utf-8")

wheel = "\n".join([
    "Wheel-Version: 1.0",
    "Generator: create_wheel.sh",
    "Root-Is-Purelib: false",
    f"Tag: {py_tag}-{py_tag}-{platform_tag}",
    "",
]).encode("utf-8")

record_rows = []
has_ptodsl_init = False

def hash_bytes(data: bytes) -> str:
    digest = hashlib.sha256(data).digest()
    return "sha256=" + base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")

with zipfile.ZipFile(wheel_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
    for path in sorted(staging.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(staging).as_posix()
        data = path.read_bytes()
        zf.writestr(rel, data)
        record_rows.append((rel, hash_bytes(data), str(len(data))))
        if rel == "ptodsl/__init__.py":
            has_ptodsl_init = True

    for rel, data in [
        (f"{dist_info}/METADATA", metadata),
        (f"{dist_info}/WHEEL", wheel),
    ]:
        zf.writestr(rel, data)
        record_rows.append((rel, hash_bytes(data), str(len(data))))

    record_rel = f"{dist_info}/RECORD"
    from io import StringIO
    buf = StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    for row in record_rows:
        writer.writerow(row)
    writer.writerow((record_rel, "", ""))
    record_bytes = buf.getvalue().encode("utf-8")
    zf.writestr(record_rel, record_bytes)

if not has_ptodsl_init:
    raise SystemExit("Wheel staging payload is missing ptodsl/__init__.py")

with zipfile.ZipFile(wheel_path) as zf:
    if "ptodsl/__init__.py" not in zf.namelist():
        raise SystemExit("Built wheel is missing ptodsl/__init__.py")

print(f"Wheel created at {wheel_path}")
PY

echo "Wheel created at ${WHEEL_DIST_DIR}/"
ls -la "${WHEEL_DIST_DIR}/"*.whl

EXPECTED_WHEEL_GLOB="${WHEEL_DIST_DIR}/ptoas-${PTOAS_PYTHON_PACKAGE_VERSION}-"*.whl
if ! compgen -G "${EXPECTED_WHEEL_GLOB}" >/dev/null 2>&1; then
  echo "Error: expected wheel matching ${EXPECTED_WHEEL_GLOB}" >&2
  exit 1
fi
