#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# ---------------------------------------------------------------------------
# VPTO TPRINT validation script
#
# Like run_vpto_print_validation.sh but for pto.tprint (tile data dumping).
# The script:
#   1. Runs ptoas --pto-backend=vpto → fatobj
#   2. Links fatobj into shared library
#   3. Builds host runner + launches on simulator
#   4. Checks that TPRINT banner and tile shape appear in the HiIPU Print log.
# ---------------------------------------------------------------------------
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

# ---- user-overridable settings ----
WORK_SPACE="${WORK_SPACE:-}"
ASCEND_HOME_PATH="${ASCEND_HOME_PATH:-}"
PTOAS_BIN="${PTOAS_BIN:-${ROOT_DIR}/build-llvm21/tools/ptoas/ptoas}"
PTOAS_FLAGS="${PTOAS_FLAGS:---pto-arch a5 --pto-backend=vpto}"
CASE_DIR="${CASE_DIR:-}"
KERNEL_NAME="${KERNEL_NAME:-print_tile_kernel_mix_aiv}"
EXPECTED_OUTPUT="${EXPECTED_OUTPUT:-TPRINT Tile}"
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
"${PTOAS_BIN}" --cann-output-version=9.0.0 --pto-arch a5 --pto-backend=vpto \
  --emit-vpto-llvm-ir \
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
# step 3: host stub + fatobj (driver mode: auto-handles include paths)
# ------------------------------------------------------------------
log "[${CASE_TOKEN}] step 3/5: host stub → fatobj"
HOST_STUB="${OUT_DIR}/host_stub.cpp"
# Auto-detect the launch function name from the case's launch.cpp
LAUNCH_FN="$(grep -oP 'void \KLaunch\w+' "${CASE_DIR}/launch.cpp" | head -1 || echo "LaunchPrintTileKernelMixAiv")"

cat > "${HOST_STUB}" << HOSTEOF
#ifndef AICORE
#define AICORE [aicore]
#endif
extern "C" __global__ AICORE void ${KERNEL_NAME}(float dummy) {}
extern "C" void ${LAUNCH_FN}(void *stream) {
    ${KERNEL_NAME}<<<1, nullptr, stream>>>(0.0f);
}
HOSTEOF

MODULE_ID="vpto_tprint_$(date +%s)"
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

if [[ "${COMPILE_ONLY}" == "1" ]]; then
  log "[${CASE_TOKEN}] COMPILE_ONLY=1, stopping after kernel .so"
  log "[${CASE_TOKEN}] output: ${KERNEL_SO}"
  exit 0
fi

# ------------------------------------------------------------------
# step 5: build host runner + run simulator
# ------------------------------------------------------------------
log "[${CASE_TOKEN}] step 5/5: build host runner + run simulator"

# Generate host runner with ACL init (required by simulator for device
# setup and DebugTunnel print buffer allocation).
HOST_RUNNER="${OUT_DIR}/print_runner.cpp"
cat > "${HOST_RUNNER}" << RUNNEREOF
#include <cstdio>
#include "acl/acl.h"

extern "C" void ${LAUNCH_FN}(void *stream);

int main() {
  aclError ret = aclInit(nullptr);
  if (ret != ACL_SUCCESS) { std::fprintf(stderr, "aclInit failed: %d\n", ret); return 1; }
  ret = aclrtSetDevice(0);
  if (ret != ACL_SUCCESS) { std::fprintf(stderr, "aclrtSetDevice failed: %d\n", ret); aclFinalize(); return 1; }
  aclrtStream stream = nullptr;
  ret = aclrtCreateStream(&stream);
  if (ret != ACL_SUCCESS) { std::fprintf(stderr, "aclrtCreateStream failed: %d\n", ret); aclrtResetDevice(0); aclFinalize(); return 1; }

  std::printf("[Host] Launching --kernel-name=%s\n", "${KERNEL_NAME}");
  ${LAUNCH_FN}(stream);
  aclrtSynchronizeStream(stream);
  std::printf("[Host] Done.\n");

  aclrtDestroyStream(stream);
  aclrtResetDevice(0);
  aclFinalize();
  return 0;
}
RUNNEREOF

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
	  -o "${OUT_DIR}/${CASE_TOKEN}_runner" \
	  -l"${CASE_TOKEN}_kernel" \
	  -lruntime_camodel -lascendcl -lstdc++ -lm -lpthread -ldl
# Run on simulator
mkdir -p "${OUT_DIR}" && chmod 700 "${OUT_DIR}"
SIM_OUT_DIR="${OUT_DIR}/sim_output"
mkdir -p "${SIM_OUT_DIR}" && chmod 700 "${SIM_OUT_DIR}"

CANN_LD="${LD_LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="${OUT_DIR}:${SIM_LIB_DIR}:${ASCEND_HOME_PATH}/lib64:${CANN_LD}"

set +e
SIM_LOG="${OUT_DIR}/simulator.log"
cd "${OUT_DIR}"
msprof op simulator \
  "${OUT_DIR}/${CASE_TOKEN}_runner" \
  --kernel-name="${KERNEL_NAME}" \
  --launch-count=1 \
  --soc-version="${SOC_VERSION}" \
  --timeout=120 \
  --output="${SIM_OUT_DIR}" 2>&1 | tee "${SIM_LOG}"
SIM_RC=${PIPESTATUS[0]}
set -e

# ------------------------------------------------------------------
# verify output
# ------------------------------------------------------------------
log "[${CASE_TOKEN}] checking simulator output..."

if [[ "${SIM_RC}" -ne 0 ]]; then
  cat "${SIM_LOG}"
  die "simulator exited with code ${SIM_RC}"
fi

# Extract the HiIPU Print section for inspection
echo ""
echo "--- HiIPU Print (from simulator log) ---"
sed -n '/---HiIPU Print---/,/^$/p' "${SIM_LOG}" | head -20 || true
echo ""

# Check for TPRINT banner
if echo "${EXPECTED_OUTPUT}" | grep -q ' '; then
  # Multi-word: check each word
  PASS=true
  for word in ${EXPECTED_OUTPUT}; do
    if ! grep -qF "${word}" "${SIM_LOG}"; then
      PASS=false
      log "MISSING: ${word}"
    fi
  done
  if ${PASS}; then
    log "[${CASE_TOKEN}] ✅ PASS — TPRINT banner found"
    echo "   expected keywords: ${EXPECTED_OUTPUT}"
  else
    log "[${CASE_TOKEN}] ❌ FAIL — some keywords not found"
    exit 1
  fi
else
  if grep -qF "${EXPECTED_OUTPUT}" "${SIM_LOG}"; then
    log "[${CASE_TOKEN}] ✅ PASS"
    echo "   expected: ${EXPECTED_OUTPUT}"
  else
    log "[${CASE_TOKEN}] ❌ FAIL"
    echo "   expected: ${EXPECTED_OUTPUT}"
    exit 1
  fi
fi

log "[${CASE_TOKEN}] output dir: ${OUT_DIR}"
