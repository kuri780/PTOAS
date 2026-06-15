// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// Licensed under CANN Open Software License Agreement Version 2.0.

#include "test_common.h"
#include "acl/acl.h"
#include <cstdio>
#include <cstdlib>
#include <cstdint>

using namespace PtoTestCommon;

#ifndef TMRGSORT_HPP
struct MrgSortExecutedNumList {
  uint16_t mrgSortList0;
  uint16_t mrgSortList1;
  uint16_t mrgSortList2;
  uint16_t mrgSortList3;
};
#endif

#define ACL_CHECK(expr)                                                          \
  do {                                                                           \
    const aclError _ret = (expr);                                                \
    if (_ret != ACL_SUCCESS) {                                                   \
      std::fprintf(stderr, "[ERROR] %s failed: %d (%s:%d)\n", #expr,             \
                   (int)_ret, __FILE__, __LINE__);                               \
      const char *_recent = aclGetRecentErrMsg();                                \
      if (_recent != nullptr && _recent[0] != '\0')                              \
        std::fprintf(stderr, "[ERROR] RecentErrMsg: %s\n", _recent);             \
      rc = 1;                                                                    \
      goto cleanup;                                                              \
    }                                                                            \
  } while (0)

#define FILE_CHECK(expr, path)                                                   \
  do {                                                                           \
    if (!(expr)) {                                                               \
      std::fprintf(stderr, "[ERROR] file operation failed: %s (%s:%d)\n",       \
                   path, __FILE__, __LINE__);                                    \
      rc = 1;                                                                    \
      goto cleanup;                                                              \
    }                                                                            \
  } while (0)

using bf16_storage_t = uint16_t;

void LaunchGemm_residual_rmsnorm_c2v_split_m_kernel(
    bf16_storage_t *a, bf16_storage_t *b,
    bf16_storage_t *residual0, bf16_storage_t *residual1,
    float *gamma,
    float *out0, float *out1,
    void *stream);

int main() {
  constexpr size_t kM = 16;
  constexpr size_t kN = 256;
  constexpr size_t kK = 64;
  constexpr size_t kRowsPerSb = kM / 2;

  constexpr size_t aElem = kM * kK;
  constexpr size_t bElem = kK * kN;
  constexpr size_t residualElem = kM * kN;
  constexpr size_t gammaElem = kN;
  constexpr size_t outElem = kM * kN;

  constexpr size_t aSize = aElem * sizeof(bf16_storage_t);
  constexpr size_t bSize = bElem * sizeof(bf16_storage_t);
  constexpr size_t residualSize = residualElem * sizeof(bf16_storage_t);
  constexpr size_t gammaSize = gammaElem * sizeof(float);
  constexpr size_t outSize = outElem * sizeof(float);

  bf16_storage_t *aHost = nullptr;
  bf16_storage_t *bHost = nullptr;
  bf16_storage_t *residualHost = nullptr;
  float *gammaHost = nullptr;
  float *outHost = nullptr;
  bf16_storage_t *aDevice = nullptr;
  bf16_storage_t *bDevice = nullptr;
  bf16_storage_t *residualDevice = nullptr;
  float *gammaDevice = nullptr;
  float *outDevice = nullptr;

  int rc = 0;
  bool aclInited = false;
  bool deviceSet = false;
  int deviceId = 0;
  aclrtStream stream = nullptr;
  size_t inputSize = 0;

  // Declare before ACL_CHECK to avoid goto-jump-bypass warning
  bf16_storage_t *residualDev0 = nullptr;
  bf16_storage_t *residualDev1 = nullptr;
  float *outDev0 = nullptr;
  float *outDev1 = nullptr;

  ACL_CHECK(aclInit(nullptr));
  aclInited = true;
  if (const char *envDevice = std::getenv("ACL_DEVICE_ID"))
    deviceId = std::atoi(envDevice);
  ACL_CHECK(aclrtSetDevice(deviceId));
  deviceSet = true;
  ACL_CHECK(aclrtCreateStream(&stream));

  ACL_CHECK(aclrtMallocHost((void **)(&aHost), aSize));
  ACL_CHECK(aclrtMallocHost((void **)(&bHost), bSize));
  ACL_CHECK(aclrtMallocHost((void **)(&residualHost), residualSize));
  ACL_CHECK(aclrtMallocHost((void **)(&gammaHost), gammaSize));
  ACL_CHECK(aclrtMallocHost((void **)(&outHost), outSize));
  ACL_CHECK(aclrtMalloc((void **)&aDevice, aSize, ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK(aclrtMalloc((void **)&bDevice, bSize, ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK(aclrtMalloc((void **)&residualDevice, residualSize, ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK(aclrtMalloc((void **)&gammaDevice, gammaSize, ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK(aclrtMalloc((void **)&outDevice, outSize, ACL_MEM_MALLOC_HUGE_FIRST));

  inputSize = aSize;
  FILE_CHECK(ReadFile("./v1.bin", inputSize, aHost, aSize) && inputSize == aSize, "./v1.bin");
  inputSize = bSize;
  FILE_CHECK(ReadFile("./v2.bin", inputSize, bHost, bSize) && inputSize == bSize, "./v2.bin");
  inputSize = residualSize;
  FILE_CHECK(ReadFile("./v3.bin", inputSize, residualHost, residualSize) && inputSize == residualSize, "./v3.bin");
  inputSize = gammaSize;
  FILE_CHECK(ReadFile("./v4.bin", inputSize, gammaHost, gammaSize) && inputSize == gammaSize, "./v4.bin");
  inputSize = outSize;
  FILE_CHECK(ReadFile("./v5.bin", inputSize, outHost, outSize) && inputSize == outSize, "./v5.bin");

  ACL_CHECK(aclrtMemcpy(aDevice, aSize, aHost, aSize, ACL_MEMCPY_HOST_TO_DEVICE));
  ACL_CHECK(aclrtMemcpy(bDevice, bSize, bHost, bSize, ACL_MEMCPY_HOST_TO_DEVICE));
  ACL_CHECK(aclrtMemcpy(residualDevice, residualSize, residualHost, residualSize, ACL_MEMCPY_HOST_TO_DEVICE));
  ACL_CHECK(aclrtMemcpy(gammaDevice, gammaSize, gammaHost, gammaSize, ACL_MEMCPY_HOST_TO_DEVICE));
  ACL_CHECK(aclrtMemcpy(outDevice, outSize, outHost, outSize, ACL_MEMCPY_HOST_TO_DEVICE));

  // Subblock-specific pointers into contiguous buffers
  residualDev0 = residualDevice;
  residualDev1 = residualDevice + kRowsPerSb * kN;
  outDev0 = outDevice;
  outDev1 = outDevice + kRowsPerSb * kN;

  LaunchGemm_residual_rmsnorm_c2v_split_m_kernel(
      aDevice, bDevice, residualDev0, residualDev1, gammaDevice,
      outDev0, outDev1, stream);
  ACL_CHECK(aclrtSynchronizeStream(stream));

  ACL_CHECK(aclrtMemcpy(outHost, outSize, outDevice, outSize, ACL_MEMCPY_DEVICE_TO_HOST));
  FILE_CHECK(WriteFile("./v5.bin", outHost, outSize), "./v5.bin");

cleanup:
  aclrtFree(aDevice);
  aclrtFree(bDevice);
  aclrtFree(residualDevice);
  aclrtFree(gammaDevice);
  aclrtFree(outDevice);
  aclrtFreeHost(aHost);
  aclrtFreeHost(bHost);
  aclrtFreeHost(residualHost);
  aclrtFreeHost(gammaHost);
  aclrtFreeHost(outHost);
  if (stream != nullptr)
    aclrtDestroyStream(stream);
  if (deviceSet)
    aclrtResetDevice(deviceId);
  if (aclInited)
    aclFinalize();
  return rc;
}
