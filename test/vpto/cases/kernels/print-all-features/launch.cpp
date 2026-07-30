// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// -----------------------------------------------------------------------------
// case: kernels/print-all-features — VPTO scalar print integration
// -----------------------------------------------------------------------------
#ifndef AICORE
#define AICORE [aicore]
#endif

extern "C" __global__ AICORE void print_all_features_kernel(float arg0,
                                                             int32_t arg1);

void LaunchPrint_all_features_kernel_mix_aiv(float arg0, int32_t arg1,
                                               void *stream) {
  print_all_features_kernel<<<2, nullptr, stream>>>(arg0, arg1);
}
