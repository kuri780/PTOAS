#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# ---------------------------------------------------------------------------
# VPTO print validation script
#
# Unlike compute-kernel tests (which use device-memory I/O + golden/compare),
# pto.print / pto.tprint emit output through the DebugTunnel host-side printf.
# This script:
#   1. Runs ptoas --pto-backend=vpto → fatobj (PTOAS emits LLVM IR, strips
#      nuw, compiles device .o, generates host stub, compiles host fatobj
#      with --cce-enable-print).
#   2. Links the fatobj into a shared library.
#   3. Compiles a minimal host runner that launches the kernel.
#   4. Runs on the simulator and checks that the expected string appears
#      in the HiIPU Print console output.
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
KERNEL_NAME="${KERNEL_NAME:-}"       # e.g. print_scalar_kernel_mix_aiv
EXPECTED_OUTPUT="${EXPECTED_OUTPUT:-}" # e.g. "scalar = +003.250"
LAUNCH_VALUE="${LAUNCH_VALUE:-3.25}"   # float value to pass to the kernel
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

# auto-detect kernel name and expected output from kernel.pto if not set
auto_detect() {
  local pto_file="${CASE_DIR}/kernel.pto"
  if [[ -z "${KERNEL_NAME}" ]]; then
    KERNEL_NAME="$(grep -oP 'func\.func @\K\w+' "${pto_file}" | head -1)_mix_aiv"
    log "auto-detected KERNEL_NAME: ${KERNEL_NAME}"
  fi
  if [[ -z "${EXPECTED_OUTPUT}" ]]; then
    local fmt
    fmt="$(grep -oP 'pto\.print ins\("\K[^"]+' "${pto_file}" | head -1 || true)"
    if [[ -n "${fmt}" ]]; then
      # build expected output by substituting the launch value into the format
      EXPECTED_OUTPUT="$(python3 -c "
import struct, sys
val = float('${LAUNCH_VALUE}')
# crude: replace %f / %+08.3f etc with the formatted value
fmt = '${fmt}'
# just check that the value appears somewhere in the output
print(f'scalar = +{val:08.3f}' if 'scalar' in fmt else f'{val}')
" 2>/dev/null || echo "${LAUNCH_VALUE}")"
      log "auto-detected EXPECTED_OUTPUT: ${EXPECTED_OUTPUT}"
    fi
  fi
}

# ---- build steps ----
mkdir -p "${WORK_SPACE}"
WORK_SPACE="$(cd "${WORK_SPACE}" && pwd)"

CASE_TOKEN="$(basename "${CASE_DIR}")"
OUT_DIR="${WORK_SPACE}/${CASE_TOKEN}"
rm -rf "${OUT_DIR}"
mkdir -p "${OUT_DIR}"

auto_detect

# ------------------------------------------------------------------
# step 1: PTOAS → LLVM IR
#   Emit LLVM IR from PTOAS.  We bypass the CC1 host-stub compilation
#   (which cannot handle C++ stdlib headers needed by DebugTunnel) and
#   instead compile device + host sides separately using the bisheng
#   driver mode that auto-manages all include paths.
# ------------------------------------------------------------------
log "[${CASE_TOKEN}] step 1/4: ptoas → LLVM IR"
"${PTOAS_BIN}" --pto-arch a5 --pto-backend=vpto --emit-vpto-llvm-ir \
  "${CASE_DIR}/kernel.pto" -o "${OUT_DIR}/kernel.ll"

# Strip `nuw` from constant GEPs (bisheng's LLVM 15 does not support it).
sed -i 's/inbounds nuw/inbounds/g' "${OUT_DIR}/kernel.ll"

# ------------------------------------------------------------------
# step 2: compile LLVM IR → device.o
# ------------------------------------------------------------------
log "[${CASE_TOKEN}] step 2/4: LLVM IR → device.o"
"${BISHENG_BIN}" --cce-aicore-arch="${AICORE_ARCH}" --cce-aicore-only -O2 \
  --cce-generic-addrspace=off -cce-bitcode-is-aicore \
  -Wno-override-module -dc -c -x ir \
  "${OUT_DIR}/kernel.ll" -o "${OUT_DIR}/kernel_device.o"

# ------------------------------------------------------------------
# step 3: host stub + fatobj  (driver mode: auto-handles include paths)
# ------------------------------------------------------------------
log "[${CASE_TOKEN}] step 3/4: host stub → fatobj"
HOST_STUB="${OUT_DIR}/host_stub.cpp"
cat > "${HOST_STUB}" << 'HOSTEOF'
#ifndef AICORE
#define AICORE [aicore]
#endif
extern "C" __global__ AICORE void KERNEL_NAME_PLACEHOLDER(float arg0) {}
extern "C" void LaunchPrintScalarKernelMixAiv(float arg0, void *stream) {
    KERNEL_NAME_PLACEHOLDER<<<1, nullptr, stream>>>(arg0);
}
HOSTEOF
sed -i "s/KERNEL_NAME_PLACEHOLDER/${KERNEL_NAME}/g" "${HOST_STUB}"

MODULE_ID="vpto_print_$(date +%s)"
"${BISHENG_BIN}" -xcce --cce-enable-print -cce-enable-mix \
  -cce-launch-with-flagv2-impl \
  --cce-aicore-arch="${AICORE_ARCH}" -DREGISTER_BASE -std=c++17 -fPIC \
  -Xclang -fcce-include-aibinary -Xclang "${OUT_DIR}/kernel_device.o" \
  -Xclang -fcce-device-module-id -Xclang "${MODULE_ID}" \
  -c "${HOST_STUB}" -o "${OUT_DIR}/kernel.fatobj.o"

# ------------------------------------------------------------------
# step 4: link + run  (same as before)
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
# step 3: build minimal host runner + run on simulator
# ------------------------------------------------------------------
if [[ "${COMPILE_ONLY}" == "1" ]]; then
  log "[${CASE_TOKEN}] COMPILE_ONLY=1, stopping after kernel .so"
  log "[${CASE_TOKEN}] output: ${KERNEL_SO}"
  exit 0
fi

log "[${CASE_TOKEN}] step 5/5: build host runner + run simulator"

# Host runner with ACL init (required by simulator for device setup).
# Values are hardcoded — argv is consumed by msprof flags, not passed to us.
HOST_RUNNER="${OUT_DIR}/print_runner.cpp"
cat > "${HOST_RUNNER}" << 'RUNNEREOF'
#include <cstdio>
#include "acl/acl.h"

extern "C" void LaunchPrintScalarKernelMixAiv(float arg0, void *stream);

int main() {
  float value = LAUNCH_VALUE_PLACEHOLDER;

  aclError ret = aclInit(nullptr);
  if (ret != ACL_SUCCESS) { std::fprintf(stderr, "aclInit failed: %d\n", ret); return 1; }
  ret = aclrtSetDevice(0);
  if (ret != ACL_SUCCESS) { std::fprintf(stderr, "aclrtSetDevice failed: %d\n", ret); aclFinalize(); return 1; }
  aclrtStream stream = nullptr;
  ret = aclrtCreateStream(&stream);
  if (ret != ACL_SUCCESS) { std::fprintf(stderr, "aclrtCreateStream failed: %d\n", ret); aclrtResetDevice(0); aclFinalize(); return 1; }

  std::printf("[Host] Launching with value=%f\n", value);
  LaunchPrintScalarKernelMixAiv(value, stream);
  aclrtSynchronizeStream(stream);
  std::printf("[Host] Done.\n");

  aclrtDestroyStream(stream);
  aclrtResetDevice(0);
  aclFinalize();
  return 0;
}
RUNNEREOF
sed -i "s/LAUNCH_VALUE_PLACEHOLDER/${LAUNCH_VALUE}/g" "${HOST_RUNNER}"

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

# Run on simulator.  msprof requires the working directory to be private
# (not writable by group/other), so cd into the already-chmod-700 output
# directory.  Keep CANN's LD_LIBRARY_PATH so msprof itself can load its
# dependencies (libascend_hal.so etc.).
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
# verify output
# ------------------------------------------------------------------
log "[${CASE_TOKEN}] checking simulator output..."

if [[ "${SIM_RC}" -ne 0 ]]; then
  cat "${SIM_LOG}"
  die "simulator exited with code ${SIM_RC}"
fi

if [[ -z "${EXPECTED_OUTPUT}" ]]; then
  log "[${CASE_TOKEN}] no EXPECTED_OUTPUT set — showing HiIPU Print section:"
  sed -n '/---HiIPU Print---/,/^$/p' "${SIM_LOG}"
  log "[${CASE_TOKEN}] PASS (no output check requested)"
else
  # Extract the HiIPU Print section for a concise summary
  echo ""
  echo "--- HiIPU Print (from simulator log) ---"
  sed -n '/---HiIPU Print---/,/^$/p' "${SIM_LOG}" | head -10
  echo ""

  if grep -qF "${EXPECTED_OUTPUT}" "${SIM_LOG}"; then
    echo "========================================"
    log "[${CASE_TOKEN}] ✅ PASS"
    echo "   expected: ${EXPECTED_OUTPUT}"
    echo "========================================"
  else
    echo "========================================"
    log "[${CASE_TOKEN}] ❌ FAIL"
    echo "   expected: ${EXPECTED_OUTPUT}"
    echo "========================================"
    exit 1
  fi
fi

log "[${CASE_TOKEN}] output dir: ${OUT_DIR}"
