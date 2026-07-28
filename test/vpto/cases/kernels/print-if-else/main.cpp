// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// SPDX-License-Identifier: CANN-OSL-2.0
// case: kernels/print-controlflow
// Verifies pto.print works inside scf.if branches.
// The kernel does GM→UB→GM passthrough; print output confirms branch taken.
#include "test_common.h"
#include "acl/acl.h"
#include <cstdio>
#include <cstdlib>
using namespace PtoTestCommon;

#define ACL_CHECK(expr) do { \
    const aclError _ret = (expr); \
    if (_ret != ACL_SUCCESS) { \
        std::fprintf(stderr, "[ERROR] %s failed: %d (%s:%d)\n", #expr, (int)_ret, __FILE__, __LINE__); \
        const char *_recent = aclGetRecentErrMsg(); \
        if (_recent != nullptr && _recent[0] != '\0') std::fprintf(stderr, "[ERROR] RecentErrMsg: %s\n", _recent); \
        rc = 1; goto cleanup; \
    } } while (0)

void LaunchPrint_if_else_kernel_mix_aiv(float *v1, float *v2, int32_t v3, int32_t v4, void *stream);

int main() {
    constexpr size_t kElemCount = 64;
    constexpr size_t kFileSize = kElemCount * sizeof(float);
    int32_t modeHost = 1;  // positive → triggers "positive" branch
    int32_t nHost = static_cast<int32_t>(kElemCount);
    float *v1Host = nullptr, *v2Host = nullptr;
    float *v1Device = nullptr, *v2Device = nullptr;
    int rc = 0; bool aclInited = false, deviceSet = false;
    int deviceId = 0; aclrtStream stream = nullptr;
    size_t fileSize = 0;

    ACL_CHECK(aclInit(nullptr)); aclInited = true;
    if (const char *e = std::getenv("ACL_DEVICE_ID")) deviceId = std::atoi(e);
    ACL_CHECK(aclrtSetDevice(deviceId)); deviceSet = true;
    ACL_CHECK(aclrtCreateStream(&stream));
    ACL_CHECK(aclrtMallocHost((void **)&v1Host, kFileSize));
    ACL_CHECK(aclrtMallocHost((void **)&v2Host, kFileSize));
    ACL_CHECK(aclrtMalloc((void **)&v1Device, kFileSize, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc((void **)&v2Device, kFileSize, ACL_MEM_MALLOC_HUGE_FIRST));
    if (!ReadFile("./v1.bin", fileSize, v1Host, kFileSize) || fileSize != kFileSize) {
        std::fprintf(stderr, "[ERROR] v1.bin read fail\n"); rc = 1; goto cleanup; }
    ACL_CHECK(aclrtMemcpy(v1Device, kFileSize, v1Host, kFileSize, ACL_MEMCPY_HOST_TO_DEVICE));
    LaunchPrint_if_else_kernel_mix_aiv(v1Device, v2Device, modeHost, nHost, stream);
    ACL_CHECK(aclrtSynchronizeStream(stream));
    ACL_CHECK(aclrtMemcpy(v2Host, kFileSize, v2Device, kFileSize, ACL_MEMCPY_DEVICE_TO_HOST));
    WriteFile("./v2.bin", v2Host, kFileSize);
cleanup:
    aclrtFree(v1Device); aclrtFree(v2Device);
    aclrtFreeHost(v1Host); aclrtFreeHost(v2Host);
    if (stream) aclrtDestroyStream(stream);
    if (deviceSet) aclrtResetDevice(deviceId);
    if (aclInited) aclFinalize();
    return rc;
}
