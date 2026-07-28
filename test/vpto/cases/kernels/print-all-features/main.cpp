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
// target_ops: pto.print, pto.tprint, pto.alloc_tile, scf.for, scf.if
// scenarios: all-print-features-combined
//
// VPTO print/tprint integration smoke test.
// Expected HiIPU output per block:
//   - [block N] value = +3.250
//   - [block N] total_elems = 128
//   - [block N] inside scf.for loop
//   - [block N] this is block 0 or 1 (conditional)
//   - === [TPRINT Tile] ... Shape: [8, 8] ...
//   - [block N] finished (2 blocks)
// -----------------------------------------------------------------------------
#include <cstdio>
#include <cstdlib>

extern "C" void LaunchPrint_all_features_kernel_mix_aiv(float arg0, int32_t arg1,
                                                          void *stream);

int main(int argc, char **argv) {
  const char *kernelName = "print_all_features_kernel_mix_aiv";
  float       value      = 3.25f;
  int32_t     nElems     = 128;

  if (argc > 1) kernelName = argv[1];
  if (argc > 2) value = static_cast<float>(std::atof(argv[2]));
  if (argc > 3) nElems = static_cast<int32_t>(std::atoi(argv[3]));

  std::printf("[Host] Launching %s with value=%f nElems=%d (2 blocks)\n",
              kernelName, (double)value, (int)nElems);
  LaunchPrint_all_features_kernel_mix_aiv(value, nElems, nullptr);
  std::printf("[Host] Kernel completed.\n");
  return 0;
}
