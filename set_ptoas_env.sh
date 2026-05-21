#!/usr/bin/env bash
# after `quick_install.sh`, run `source set_ptoas_env.sh` in a new shell to find the lib
export PTO_SOURCE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PTO_INSTALL_DIR="${PTO_INSTALL_DIR:-${PTO_SOURCE_DIR}/install}"
export PATH="${PTO_SOURCE_DIR}/build/tools/ptoas:${PATH}"
export LD_LIBRARY_PATH="${LLVM_BUILD_DIR}/lib:${PTO_INSTALL_DIR}/lib:${LD_LIBRARY_PATH:-}"

PTOAS_ENV_TMP="${PTO_SOURCE_DIR}/tmp/set_ptoas_env"
mkdir -p "${PTOAS_ENV_TMP}/MatMul" "${PTOAS_ENV_TMP}/Abs"
(cd "${PTO_SOURCE_DIR}/test/samples/MatMul" && python ./tmatmulk.py > "${PTOAS_ENV_TMP}/MatMul/tmatmulk.pto" && ptoas "${PTOAS_ENV_TMP}/MatMul/tmatmulk.pto" -o "${PTOAS_ENV_TMP}/MatMul/tmatmulk.cpp")
(cd "${PTO_SOURCE_DIR}/test/samples/Abs" && python ./abs.py > "${PTOAS_ENV_TMP}/Abs/abs.pto" && ptoas --enable-insert-sync "${PTOAS_ENV_TMP}/Abs/abs.pto" -o "${PTOAS_ENV_TMP}/Abs/abs.cpp")

echo "test set_env: OK"
