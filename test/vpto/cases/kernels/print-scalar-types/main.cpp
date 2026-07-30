// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// -----------------------------------------------------------------------------
// case: kernels/print-scalar-types
// family: kernels
// target_ops: pto.print (f16, f64, i32)
// scenarios: debug-tunnel, scalar-multi-type-print
//
// Print validation uses the HiIPU Print console log, not device memory.
// Expected output: "f16=", "f64=", "i32=" lines from the kernel.
// -----------------------------------------------------------------------------
#include <cstdio>

extern "C" void LaunchPrintScalarTypesKernelMixAiv(void *stream);

int main(int argc, char **argv) {
  const char *kernelName = "print_scalar_types_kernel_mix_aiv";

  if (argc > 1) kernelName = argv[1];
  (void)kernelName; // used for msprof --kernel-name in scripts

  std::printf("[Host] Launching %s\n", kernelName);
  LaunchPrintScalarTypesKernelMixAiv(nullptr);
  std::printf("[Host] Kernel completed.\n");
  return 0;
}
