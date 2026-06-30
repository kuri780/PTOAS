#include "test_common.h"
#include "acl/acl.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>

using namespace PtoTestCommon;

#define ACL_CHECK(expr) do { const aclError _ret = (expr); if (_ret != ACL_SUCCESS) { std::fprintf(stderr, "[ERROR] %s failed: %d (%s:%d)\n", #expr, (int)_ret, __FILE__, __LINE__); rc = 1; goto cleanup; } } while (0)

void Launch_kernel(float *out, void *stream);

int main() {
  size_t elemCount = 128;
  size_t fileSize = elemCount * sizeof(float);
  float *outHost = nullptr;
  float *outDevice = nullptr;
  int rc = 0;
  bool aclInited = false, deviceSet = false;
  int deviceId = 0;
  aclrtStream stream = nullptr;

  ACL_CHECK(aclInit(nullptr)); aclInited = true;
  if (const char *e = std::getenv("ACL_DEVICE_ID")) deviceId = std::atoi(e);
  ACL_CHECK(aclrtSetDevice(deviceId)); deviceSet = true;
  ACL_CHECK(aclrtCreateStream(&stream));
  ACL_CHECK(aclrtMallocHost((void **)(&outHost), fileSize));
  ACL_CHECK(aclrtMalloc((void **)&outDevice, fileSize, ACL_MEM_MALLOC_HUGE_FIRST));
  // Input: zero-initialize output buffer (kernel writes results)
  std::memset(outHost, 0, fileSize);
  ACL_CHECK(aclrtMemcpy(outDevice, fileSize, outHost, fileSize, ACL_MEMCPY_HOST_TO_DEVICE));
  Launch_kernel(outDevice, stream);
  ACL_CHECK(aclrtSynchronizeStream(stream));
  ACL_CHECK(aclrtMemcpy(outHost, fileSize, outDevice, fileSize, ACL_MEMCPY_DEVICE_TO_HOST));
  WriteFile("./out.bin", outHost, fileSize);

cleanup:
  aclrtFree(outDevice); aclrtFreeHost(outHost);
  if (stream) aclrtDestroyStream(stream);
  if (deviceSet) aclrtResetDevice(deviceId);
  if (aclInited) aclFinalize();
  return rc;
}
