#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# ---------------------------------------------------------------------------
# VPTO print type-coverage validation script
#
# Validates that pto.print correctly lowers f16, f64, i32, i64 scalar
# types through the DebugTunnel protocol.  The kernel uses constants
# (no host parameters), so the host stub is trivial.
#
# Usage:
#   CASE_DIR=/path/to/print-scalar-types \
#   WORK_SPACE=/tmp/vpto-print-types \
#   ASCEND_HOME_PATH=/usr/local/Ascend/cann \
#   bash run_vpto_print_types_validation.sh
# ---------------------------------------------------------------------------
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

# ---- user-overridable settings ----
WORK_SPACE="${WORK_SPACE:-}"
ASCEND_HOME_PATH="${ASCEND_HOME_PATH:-}"
PTOAS_BIN="${PTOAS_BIN:-${ROOT_DIR}/build-llvm21/tools/ptoas/ptoas}"
CASE_DIR="${CASE_DIR:-}"
KERNEL_NAME="${KERNEL_NAME:-}"
SOC_VERSION="${SOC_VERSION:-Ascend950PR_950x}"
AICORE_ARCH="${AICORE_ARCH:-dav-c310-vec}"
COMPILE_ONLY="${COMPILE_ONLY:-0}"
# --------------------------------------------------------------------

log() { echo "[$(date +'%F %T')] $*"; }
die() { echo "ERROR: $*" >&2; exit 1; }

require_env() {
  local name="$1" value="$2"
  [[ -n "${value}" ]] || die "${name} is required"
}

require_env "WORK_SPACE"       "${WORK_SPACE}"
require_env "ASCEND_HOME_PATH" "${ASCEND_HOME_PATH}"
require_env "CASE_DIR"         "${CASE_DIR}"
[[ -x "${PTOAS_BIN}" ]] || die "PTOAS_BIN is not executable: ${PTOAS_BIN}"
[[ -f "${CASE_DIR}/kernel.pto" ]] || die "missing ${CASE_DIR}/kernel.pto"

# source CANN environment
if [[ -f "${ASCEND_HOME_PATH}/set_env.sh" ]]; then
  set +u; source "${ASCEND_HOME_PATH}/set_env.sh" >/dev/null 2>&1; set -u
fi

BISHENG_BIN="${BISHENG_BIN:-${ASCEND_HOME_PATH}/bin/bisheng}"
command -v "${BISHENG_BIN}" >/dev/null 2>&1 || die "bisheng not found"

# resolve simulator library directory
resolve_sim_lib_dir() {
  local candidates=()
  readarray -t candidates < <(
    find "${ASCEND_HOME_PATH}" -type d -path '*/simulator/dav_3510/lib' 2>/dev/null | sort
  )
  if [[ "${#candidates[@]}" -ge 1 ]]; then
    SIM_LIB_DIR="${candidates[0]}"
    log "auto-detected SIM_LIB_DIR: ${SIM_LIB_DIR}"
    return 0
  fi
  die "cannot find dav_3510 simulator lib dir under ${ASCEND_HOME_PATH}"
}

# auto-detect kernel name from kernel.pto
if [[ -z "${KERNEL_NAME}" ]]; then
  KERNEL_NAME="$(grep -oP 'func\.func @\K\w+' "${CASE_DIR}/kernel.pto" | head -1)_mix_aiv"
  log "auto-detected KERNEL_NAME: ${KERNEL_NAME}"
fi

# ---- build steps ----
mkdir -p "${WORK_SPACE}"
WORK_SPACE="$(cd "${WORK_SPACE}" && pwd)"

CASE_TOKEN="$(basename "${CASE_DIR}")"
OUT_DIR="${WORK_SPACE}/${CASE_TOKEN}"
rm -rf "${OUT_DIR}"
mkdir -p "${OUT_DIR}"

# ------------------------------------------------------------------
# step 1: PTOAS → LLVM IR
# ------------------------------------------------------------------
log "[${CASE_TOKEN}] step 1/5: ptoas → LLVM IR"
"${PTOAS_BIN}" --pto-arch a5 --pto-backend=vpto --emit-vpto-llvm-ir \
  "${CASE_DIR}/kernel.pto" -o "${OUT_DIR}/kernel.ll"

# Strip `nuw` from constant GEPs (bisheng's LLVM 15 does not support it).
sed -i 's/inbounds nuw/inbounds/g' "${OUT_DIR}/kernel.ll"

# ------------------------------------------------------------------
# step 2: compile LLVM IR → device.o
# ------------------------------------------------------------------
log "[${CASE_TOKEN}] step 2/5: LLVM IR → device.o"
"${BISHENG_BIN}" --cce-aicore-arch="${AICORE_ARCH}" --cce-aicore-only -O2 \
  --cce-generic-addrspace=off -cce-bitcode-is-aicore \
  -Wno-override-module -dc -c -x ir \
  "${OUT_DIR}/kernel.ll" -o "${OUT_DIR}/kernel_device.o"

# ------------------------------------------------------------------
# step 3: host stub + fatobj
#   No-arg kernel → trivial host stub with no parameters.
# ------------------------------------------------------------------
log "[${CASE_TOKEN}] step 3/5: host stub → fatobj"
HOST_STUB="${OUT_DIR}/host_stub.cpp"
cat > "${HOST_STUB}" << HOSTEOF
#ifndef AICORE
#define AICORE [aicore]
#endif
extern "C" __global__ AICORE void KERNEL_NAME_PLACEHOLDER() {}
extern "C" void LAUNCH_FN_NAME(void *stream) {
    KERNEL_NAME_PLACEHOLDER<<<1, nullptr, stream>>>();
}
HOSTEOF

# Detect the launch function name from the case's launch.cpp
LAUNCH_FN="$(grep -oP 'void \KLaunch\w+' "${CASE_DIR}/launch.cpp" | head -1 || echo "LaunchPrintScalarTypesKernelMixAiv")"
sed -i "s/KERNEL_NAME_PLACEHOLDER/${KERNEL_NAME}/g" "${HOST_STUB}"
sed -i "s/LAUNCH_FN_NAME/${LAUNCH_FN}/g" "${HOST_STUB}"

MODULE_ID="vpto_print_types_$(date +%s)"
"${BISHENG_BIN}" -xcce --cce-enable-print -cce-enable-mix \
  -cce-launch-with-flagv2-impl \
  --cce-aicore-arch="${AICORE_ARCH}" -DREGISTER_BASE -std=c++17 -fPIC \
  -Xclang -fcce-include-aibinary -Xclang "${OUT_DIR}/kernel_device.o" \
  -Xclang -fcce-device-module-id -Xclang "${MODULE_ID}" \
  -c "${HOST_STUB}" -o "${OUT_DIR}/kernel.fatobj.o"

# ------------------------------------------------------------------
# step 4: link kernel shared library
# ------------------------------------------------------------------
log "[${CASE_TOKEN}] step 4/5: link kernel shared library"
SIM_LIB_DIR="${SIM_LIB_DIR:-}"
if [[ -z "${SIM_LIB_DIR}" ]]; then
  resolve_sim_lib_dir
fi

KERNEL_SO="${OUT_DIR}/lib${CASE_TOKEN}_kernel.so"
"${BISHENG_BIN}" \
  -fPIC -s -Wl,-z,relro -Wl,-z,now --cce-fatobj-link \
  --cce-aicore-arch="${AICORE_ARCH}" \
  -shared \
  -L "${ASCEND_HOME_PATH}/lib64" \
  -L "${SIM_LIB_DIR}" -Wl,-rpath,"${SIM_LIB_DIR}" \
  -Wl,-rpath,"${ASCEND_HOME_PATH}/lib64" \
  -o "${KERNEL_SO}" \
  "${OUT_DIR}/kernel.fatobj.o" \
  -Wl,--no-as-needed -lruntime_camodel

# ------------------------------------------------------------------
# step 5: build host runner + run on simulator
# ------------------------------------------------------------------
if [[ "${COMPILE_ONLY}" == "1" ]]; then
  log "[${CASE_TOKEN}] COMPILE_ONLY=1, stopping after kernel .so"
  log "[${CASE_TOKEN}] output: ${KERNEL_SO}"
  exit 0
fi

log "[${CASE_TOKEN}] step 5/5: build host runner + run simulator"

# Use the case's own main.cpp if available, otherwise generate one.
if [[ -f "${CASE_DIR}/main.cpp" ]]; then
  HOST_RUNNER="${CASE_DIR}/main.cpp"
else
  HOST_RUNNER="${OUT_DIR}/print_runner.cpp"
  cat > "${HOST_RUNNER}" << 'RUNNEREOF'
#include <cstdio>
#include "acl/acl.h"

extern "C" void LAUNCH_FN_NAME(void *stream);

int main() {
  aclError ret = aclInit(nullptr);
  if (ret != ACL_SUCCESS) { std::fprintf(stderr, "aclInit failed: %d\n", ret); return 1; }
  ret = aclrtSetDevice(0);
  if (ret != ACL_SUCCESS) { std::fprintf(stderr, "aclrtSetDevice failed: %d\n", ret); aclFinalize(); return 1; }
  aclrtStream stream = nullptr;
  ret = aclrtCreateStream(&stream);
  if (ret != ACL_SUCCESS) { std::fprintf(stderr, "aclrtCreateStream failed: %d\n", ret); aclrtResetDevice(0); aclFinalize(); return 1; }

  std::printf("[Host] Launching kernel\n");
  LAUNCH_FN_NAME(stream);
  aclrtSynchronizeStream(stream);
  std::printf("[Host] Done.\n");

  aclrtDestroyStream(stream);
  aclrtResetDevice(0);
  aclFinalize();
  return 0;
}
RUNNEREOF
  sed -i "s/LAUNCH_FN_NAME/${LAUNCH_FN}/g" "${HOST_RUNNER}"
fi

# Compile host runner with g++ for ABI compatibility with simulator libs
HOST_BIN="${OUT_DIR}/${CASE_TOKEN}_runner"
g++ -std=c++17 -O2 \
  "${HOST_RUNNER}" \
  -I "${ASCEND_HOME_PATH}/include" \
  -L "${OUT_DIR}" \
  -L "${ASCEND_HOME_PATH}/lib64" \
  -L "${SIM_LIB_DIR}" \
  -Wl,-rpath,"${OUT_DIR}" \
  -Wl,-rpath,"${SIM_LIB_DIR}" \
  -Wl,-rpath,"${ASCEND_HOME_PATH}/lib64" \
  -Wl,--allow-shlib-undefined \
  -o "${HOST_BIN}" \
  -l"${CASE_TOKEN}_kernel" \
  -lruntime_camodel -lascendcl -lstdc++ -lm -lpthread -ldl

# Run on simulator.
mkdir -p "${OUT_DIR}" && chmod 700 "${OUT_DIR}"
SIM_OUT_DIR="${OUT_DIR}/sim_output"
mkdir -p "${SIM_OUT_DIR}" && chmod 700 "${SIM_OUT_DIR}"

CANN_LD="${LD_LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="${OUT_DIR}:${SIM_LIB_DIR}:${ASCEND_HOME_PATH}/lib64:${CANN_LD}"

set +e
SIM_LOG="${OUT_DIR}/simulator.log"
cd "${OUT_DIR}"
msprof op simulator \
  "${HOST_BIN}" \
  --kernel-name="${KERNEL_NAME}" \
  --launch-count=1 \
  --soc-version="${SOC_VERSION}" \
  --timeout=120 \
  --output="${SIM_OUT_DIR}" 2>&1 | tee "${SIM_LOG}"
SIM_RC=${PIPESTATUS[0]}
set -e

# ------------------------------------------------------------------
# verify output — check that expected type markers appear
# ------------------------------------------------------------------
log "[${CASE_TOKEN}] checking simulator output..."

if [[ "${SIM_RC}" -ne 0 ]]; then
  cat "${SIM_LOG}"
  die "simulator exited with code ${SIM_RC}"
fi

# Build expected-value table from kernel.pto:
#   Format "f16=%f\n" + constant 1.5        → expect "f16=1.500000"
#   Format "f64=%f\n" + constant 2.718281828 → expect "f64=2.718282"
#   Format "i32=%d\n" + constant -42        → expect "i32=-42"
#   Format "i64=%d\n" + constant 1234567890123 → expect "i64=1234567890123"
#
# Float values are checked with a tolerant regex to accommodate
# minor CCE printf formatting differences.

declare -A EXPECTED_PATTERNS
EXPECTED_PATTERNS=(
  ["f16"]="f16=1\.500000"
  ["f64"]="f64=2\.718282"
  ["i32"]="i32=-42"
  ["i64"]="i64=1234567890123"
)

PASSED=true
echo ""
echo "--- HiIPU Print (from simulator log) ---"
sed -n '/---HiIPU Print---/,/^$/p' "${SIM_LOG}" | head -20
echo ""

for key in "${!EXPECTED_PATTERNS[@]}"; do
  pattern="${EXPECTED_PATTERNS[${key}]}"
  if grep -qPE "${pattern}" "${SIM_LOG}"; then
    log "  ✅ found: ${key} (matched '${pattern}')"
  else
    log "  ❌ missing: ${key} (pattern '${pattern}' not found)"
    PASSED=false
  fi
done

if [[ "${PASSED}" == "true" ]]; then
  echo "========================================"
  log "[${CASE_TOKEN}] ✅ PASS"
  echo "========================================"
else
  echo "========================================"
  log "[${CASE_TOKEN}] ❌ FAIL"
  echo "========================================"
  exit 1
fi

log "[${CASE_TOKEN}] output dir: ${OUT_DIR}"
