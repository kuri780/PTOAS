// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// Licensed under CANN Open Software License Agreement Version 2.0.

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
//   %a_gm           – BF16[16,64]
//   %b_gm           – BF16[64,256]
//   %residual_gm_0  – BF16[8,256]
//   %residual_gm_1  – BF16[8,256]
//   %gamma_gm       – FP32[256]
//   %out_gm_0       – FP32[8,256]
//   %out_gm_1       – FP32[8,256]
extern "C" __global__ [aicore] void gemm_residual_rmsnorm_c2v_split_m_kernel(
    __gm__ bf16_storage_t *a,
    __gm__ bf16_storage_t *b,
    __gm__ bf16_storage_t *residual0,
    __gm__ bf16_storage_t *residual1,
    __gm__ float *gamma,
    __gm__ float *out0,
    __gm__ float *out1);

void LaunchGemm_residual_rmsnorm_c2v_split_m_kernel(
    bf16_storage_t *a, bf16_storage_t *b,
    bf16_storage_t *residual0, bf16_storage_t *residual1,
    float *gamma,
    float *out0, float *out1,
    void *stream) {
  gemm_residual_rmsnorm_c2v_split_m_kernel<<<1, nullptr, stream>>>(
      (__gm__ bf16_storage_t *)a,
      (__gm__ bf16_storage_t *)b,
      (__gm__ bf16_storage_t *)residual0,
      (__gm__ bf16_storage_t *)residual1,
      (__gm__ float *)gamma,
      (__gm__ float *)out0,
      (__gm__ float *)out1);
}
