// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// -----------------------------------------------------------------------------
// case: kernels/print-scalar-types
// target_ops: pto.print (f16, f64, i32, i64)
// Dummy float arg required for CCE runtime to inject DTData for print support.
// All print values are kernel-side constants.
// -----------------------------------------------------------------------------
#ifndef AICORE
#define AICORE [aicore]
#endif

extern "C" __global__ AICORE void print_scalar_types_kernel_mix_aiv(float dummy);

void LaunchPrintScalarTypesKernelMixAiv(void *stream) {
  print_scalar_types_kernel_mix_aiv<<<1, nullptr, stream>>>(0.0f);
}
