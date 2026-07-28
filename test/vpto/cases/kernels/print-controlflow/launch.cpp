// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// SPDX-License-Identifier: CANN-OSL-2.0
// case: kernels/print-controlflow
#ifndef __VEC_SCOPE__
#define __VEC_SCOPE__
#endif
#if defined(__CCE_AICORE__) && defined(__NPU_ARCH__) && (__NPU_ARCH__ == 2201)
typedef struct { unsigned char v; } hifloat8_t;
typedef struct { unsigned char v; } float8_e4m3_t;
typedef struct { unsigned char v; } float8_e5m2_t;
typedef struct { unsigned char v; } float8_e8m0_t;
typedef struct { unsigned char v; } float4_e1m2x2_t;
typedef struct { unsigned char v; } float4_e2m1x2_t;
#endif
#include <stdint.h>
#if defined(__CCE_AICORE__) && defined(PTOAS_ENABLE_CCE_PRINT)
#include <ccelib/print/print.h>
#endif
#ifndef __CPU_SIM
#include "acl/acl.h"
#endif

extern "C" __global__ [aicore] void print_controlflow_kernel(
    __gm__ float *v1, __gm__ float *v2, int32_t v3, int32_t v4);

void LaunchPrint_controlflow_kernel_mix_aiv(float *v1, float *v2,
                                              int32_t v3, int32_t v4,
                                              void *stream) {
  print_controlflow_kernel<<<1, nullptr, stream>>>(
      (__gm__ float *)v1, (__gm__ float *)v2, v3, v4);
}
