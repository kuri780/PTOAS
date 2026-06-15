// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

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

#include <cstdint>

#if defined(__CCE_AICORE__) && defined(PTOAS_ENABLE_CCE_PRINT)
#include <ccelib/print/print.h>
#endif

#if !defined(__CCE_AICORE__) && !defined(TMRGSORT_HPP)
struct MrgSortExecutedNumList {
  uint16_t mrgSortList0;
  uint16_t mrgSortList1;
  uint16_t mrgSortList2;
  uint16_t mrgSortList3;
};
#endif

#ifndef __CPU_SIM
#include "acl/acl.h"
#endif

using bf16_storage_t = uint16_t;

// Kernel args:
//   %a_gm         – BF16[16,64]
//   %b_gm         – BF16[64,256]
//   %residual_gm  – BF16[16,256]
//   %out_gm       – FP32[16,256]
extern "C" __global__ [aicore] void gemm_residual_tile_c2v_split_m_kernel(
    __gm__ bf16_storage_t *a,
    __gm__ bf16_storage_t *b,
    __gm__ bf16_storage_t *residual,
    __gm__ float *out);

void LaunchGemm_residual_tile_c2v_split_m_kernel(
    bf16_storage_t *a, bf16_storage_t *b,
    bf16_storage_t *residual,
    float *out,
    void *stream) {
  gemm_residual_tile_c2v_split_m_kernel<<<1, nullptr, stream>>>(
      (__gm__ bf16_storage_t *)a,
      (__gm__ bf16_storage_t *)b,
      (__gm__ bf16_storage_t *)residual,
      (__gm__ float *)out);
}
