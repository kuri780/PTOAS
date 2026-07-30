// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// -----------------------------------------------------------------------------
// case: kernels/print-all-features
// family: kernels
// target_ops: pto.print, arith.constant
// scenarios: scalar-print-pto-coexistence
//
// VPTO scalar print integration smoke test.
// -----------------------------------------------------------------------------
#include <cstdio>
#include <cstdlib>
#include "acl/acl.h"

void LaunchPrint_all_features_kernel_mix_aiv(float arg0, int32_t arg1,
                                                          void *stream);

#define ACL_CHECK(expr) do { \
  const aclError ret = (expr); \
  if (ret != ACL_SUCCESS) { \
    std::fprintf(stderr, "[ERROR] %s failed: %d\n", #expr, (int)ret); \
    rc = 1; goto cleanup; \
  } \
} while (0)

int main(int argc, char **argv) {
  const char *kernelName = "print_all_features_kernel_mix_aiv";
  float       value      = 3.25f;
  int32_t     nElems     = 128;
  int rc = 0;
  bool aclInited = false, deviceSet = false;
  int deviceId = 0;
  aclrtStream stream = nullptr;

  if (argc > 1) kernelName = argv[1];
  if (argc > 2) value = static_cast<float>(std::atof(argv[2]));
  if (argc > 3) nElems = static_cast<int32_t>(std::atoi(argv[3]));

  ACL_CHECK(aclInit(nullptr));
  aclInited = true;
  if (const char *env = std::getenv("ACL_DEVICE_ID"))
    deviceId = std::atoi(env);
  ACL_CHECK(aclrtSetDevice(deviceId));
  deviceSet = true;
  ACL_CHECK(aclrtCreateStream(&stream));

  std::printf("[Host] Launching %s with value=%f nElems=%d\n",
              kernelName, (double)value, (int)nElems);
  LaunchPrint_all_features_kernel_mix_aiv(value, nElems, stream);
  ACL_CHECK(aclrtSynchronizeStream(stream));
  std::printf("[Host] Kernel completed.\n");
cleanup:
  if (stream) aclrtDestroyStream(stream);
  if (deviceSet) aclrtResetDevice(deviceId);
  if (aclInited) aclFinalize();
  return rc;
}
