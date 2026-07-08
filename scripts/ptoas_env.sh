#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# PTOAS runtime environment bootstrap.
# Usage:
#   source scripts/ptoas_env.sh
#
# Optional overrides before sourcing:
#   export WORKSPACE_DIR=/path/to/workspace
#   export LLVM_BUILD_DIR=/path/to/llvm-project/build-shared
#   export PTO_SOURCE_DIR=/path/to/PTOAS
#   export PTO_INSTALL_DIR=/path/to/PTOAS/install
#   export PTO_PYTHON_BIN=/path/to/python3
#   export PTOAS_ENV_SKIP_SMOKE_TEST=1  # skip legacy MatMul/Abs sample checks
# CI shells skip the legacy smoke test by default.

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
	echo "This script must be sourced: source scripts/ptoas_env.sh"
	exit 1
fi

_PTOAS_ENV_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
_PTOAS_REPO_DIR="$(cd -- "${_PTOAS_ENV_SCRIPT_DIR}/.." && pwd)"

# Default layout:
#   <workspace>/
#     ├── PTOAS/
#     └── llvm-project/
export PTO_SOURCE_DIR="${PTO_SOURCE_DIR:-${_PTOAS_REPO_DIR}}"
export WORKSPACE_DIR="${WORKSPACE_DIR:-$(cd -- "${PTO_SOURCE_DIR}/.." && pwd)}"
export LLVM_SOURCE_DIR="${LLVM_SOURCE_DIR:-${WORKSPACE_DIR}/llvm-project}"
export LLVM_BUILD_DIR="${LLVM_BUILD_DIR:-${LLVM_SOURCE_DIR}/build-shared}"
export PTO_INSTALL_DIR="${PTO_INSTALL_DIR:-${PTO_SOURCE_DIR}/install}"
export PTO_ISA_PATH="${PTO_ISA_PATH:-${WORKSPACE_DIR}/pto-isa}"
export ASCEND_HOME_PATH="${ASCEND_HOME_PATH:-${HOME}/cann}"

export MLIR_PYTHON_ROOT="${MLIR_PYTHON_ROOT:-${LLVM_BUILD_DIR}/tools/mlir/python_packages/mlir_core}"
if [[ -z "${PTOAS_PYTHON_SITE:-}" ]]; then
	PTOAS_PYTHON_SITE="$(
		PTO_INSTALL_DIR="${PTO_INSTALL_DIR}" python3 - <<'PY' 2>/dev/null || true
import os
import sysconfig

prefix = os.environ["PTO_INSTALL_DIR"]
print(sysconfig.get_path("purelib", vars={"base": prefix, "platbase": prefix}))
PY
	)"
fi
export PTOAS_PYTHON_SITE
export PTO_PYTHON_ROOT="${PTO_PYTHON_ROOT:-${PTO_INSTALL_DIR}}"
export PTO_PYTHON_BUILD_ROOT="${PTO_PYTHON_BUILD_ROOT:-${PTO_SOURCE_DIR}/build/python}"
export PYBIND11_CMAKE_DIR=$(python3 -m pybind11 --cmakedir)
export PTOAS_FLAGS="${PTOAS_FLAGS:-}"
export PTOAS_OUT_DIR="${PTOAS_OUT_DIR:-${PTO_SOURCE_DIR}/build/output}"

_ptoas_prepend_path() {
	local var_name="$1"
	local value="$2"
	local current="${!var_name:-}"
	if [[ -z "${value}" ]]; then
		return 0
	fi
	if [[ ! -e "${value}" ]]; then
		return 0
	fi
	if [[ ":${current}:" == *":${value}:"* ]]; then
		return 0
	fi
	if [[ -z "${current}" ]]; then
		printf -v "${var_name}" '%s' "${value}"
	else
		printf -v "${var_name}" '%s:%s' "${value}" "${current}"
	fi
	export "${var_name}"
}

_ptoas_run_legacy_smoke_test() {
	local python_bin="${PTO_PYTHON_BIN:-${PYTHON_BIN:-$(command -v python3 || command -v python || true)}}"
	if [[ -z "${python_bin}" ]]; then
		echo "error: unable to locate python3 or python for scripts/ptoas_env.sh" >&2
		return 1
	fi

	if ! command -v ptoas >/dev/null 2>&1; then
		echo "error: ptoas is not on PATH after sourcing scripts/ptoas_env.sh" >&2
		return 1
	fi

	local tmp_root="${PTOAS_ENV_TMP:-${PTO_SOURCE_DIR}/tmp/set_ptoas_env}"
	mkdir -p "${tmp_root}/MatMul" "${tmp_root}/Abs"
	(
		cd "${PTO_SOURCE_DIR}/test/samples/MatMul" &&
			"${python_bin}" ./tmatmulk.py > "${tmp_root}/MatMul/tmatmulk.pto" &&
			ptoas "${tmp_root}/MatMul/tmatmulk.pto" -o "${tmp_root}/MatMul/tmatmulk.cpp"
	)
	(
		cd "${PTO_SOURCE_DIR}/test/samples/Abs" &&
			"${python_bin}" ./abs.py > "${tmp_root}/Abs/abs.pto" &&
			ptoas --enable-insert-sync "${tmp_root}/Abs/abs.pto" -o "${tmp_root}/Abs/abs.cpp"
	)

	echo "test set_env: OK"
}

# Prefer the in-tree PTO Python overlay plus LLVM's full MLIR package first.
# The install prefix may only contain the PTO overlay fragments, and when it is
# placed ahead of mlir_core it can shadow the real MLIR Python bindings.
_ptoas_prepend_path PYTHONPATH "${PTO_INSTALL_DIR}"
_ptoas_prepend_path PYTHONPATH "${PTO_PYTHON_ROOT}"
_ptoas_prepend_path PYTHONPATH "${PTOAS_PYTHON_SITE}"
_ptoas_prepend_path PYTHONPATH "${MLIR_PYTHON_ROOT}"
_ptoas_prepend_path PYTHONPATH "${PTO_PYTHON_BUILD_ROOT}"

_ptoas_prepend_path LD_LIBRARY_PATH "${LLVM_BUILD_DIR}/lib"
_ptoas_prepend_path LD_LIBRARY_PATH "${PTO_INSTALL_DIR}/lib"
_ptoas_prepend_path LD_LIBRARY_PATH "${PTO_SOURCE_DIR}/build/lib"

_ptoas_prepend_path PATH "${PTO_SOURCE_DIR}/build/tools/ptoas"

if [[ -n "${PTO_PYTHON_BIN:-}" && -x "${PTO_PYTHON_BIN}" ]]; then
	alias ptoas-python="${PTO_PYTHON_BIN}"
fi

echo "[ptoas_env] PTO_SOURCE_DIR=${PTO_SOURCE_DIR}"
echo "[ptoas_env] LLVM_BUILD_DIR=${LLVM_BUILD_DIR}"
echo "[ptoas_env] PTO_INSTALL_DIR=${PTO_INSTALL_DIR}"
echo "[ptoas_env] PTO_ISA_PATH=${PTO_ISA_PATH}"
echo "[ptoas_env] ASCEND_HOME_PATH=${ASCEND_HOME_PATH}"
echo "[ptoas_env] PATH/PYTHONPATH/LD_LIBRARY_PATH updated"

_ptoas_env_skip_smoke="${PTOAS_ENV_SKIP_SMOKE_TEST:-0}"
if [[ -z "${PTOAS_ENV_SKIP_SMOKE_TEST:-}" &&
	("${CI:-}" == "1" || "${CI:-}" == "true") ]]; then
	_ptoas_env_skip_smoke=1
fi

if [[ "${_ptoas_env_skip_smoke}" != "1" ]]; then
	_ptoas_run_legacy_smoke_test
fi

unset -f _ptoas_prepend_path
unset -f _ptoas_run_legacy_smoke_test
unset _PTOAS_ENV_SCRIPT_DIR
unset _PTOAS_REPO_DIR
unset _ptoas_env_skip_smoke
