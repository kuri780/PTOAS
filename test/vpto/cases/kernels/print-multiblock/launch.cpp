// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// SPDX-License-Identifier: CANN-OSL-2.0
// case: kernels/print-multiblock — multi-block kernel with pto.print
#ifndef __VEC_SCOPE__
#define __VEC_SCOPE__
#endif
#if defined(__CCE_AICORE__) && defined(__NPU_ARCH__) && (__NPU_ARCH__ == 2201)
typedef struct { unsigned char v; } hifloat8_t;
typedef struct { unsigned char v; } float8_e4m3_t, float8_e5m2_t, float8_e8m0_t;
typedef struct { unsigned char v; } float4_e1m2x2_t, float4_e2m1x2_t;
#endif
#include <stdint.h>
#if defined(__CCE_AICORE__) && defined(PTOAS_ENABLE_CCE_PRINT)
#include <ccelib/print/print.h>
#endif
#ifndef __CPU_SIM
#include "acl/acl.h"
#endif

extern "C" __global__ [aicore] void print_multiblock_kernel(
    __gm__ float *v1, __gm__ float *v2, int32_t v3);

void LaunchPrint_multiblock_kernel_mix_aiv(float *v1, float *v2,
                                             int32_t v3, void *stream) {
  const int32_t kBlockElems = 32;
  const int32_t blocks = (v3 + kBlockElems - 1) / kBlockElems;
  print_multiblock_kernel<<<blocks, nullptr, stream>>>(
      (__gm__ float *)v1, (__gm__ float *)v2, v3);
}
