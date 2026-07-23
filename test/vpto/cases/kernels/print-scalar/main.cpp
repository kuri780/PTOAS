// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// -----------------------------------------------------------------------------
// case: kernels/print-scalar
// family: kernels
// target_ops: pto.print
// scenarios: debug-tunnel, scalar-float-print
//
// Print validation uses the HiIPU Print console log, not device memory.
// Expected output: "scalar = +003.250" (when launched with value 3.25f)
// -----------------------------------------------------------------------------
#include <cstdio>
#include <cstdlib>

extern "C" void LaunchPrintScalarKernelMixAiv(float arg0, void *stream);

int main(int argc, char **argv) {
  const char *kernelName = "print_scalar_kernel_mix_aiv";
  float       value      = 3.25f;

  if (argc > 1) kernelName = argv[1];
  if (argc > 2) value = static_cast<float>(std::atof(argv[2]));

  std::printf("[Host] Launching %s with value=%f\n", kernelName, value);
  LaunchPrintScalarKernelMixAiv(value, nullptr);
  std::printf("[Host] Kernel completed.\n");
  return 0;
}
