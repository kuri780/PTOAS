#ifndef __VEC_SCOPE__
#define __VEC_SCOPE__
#endif
#include <stdint.h>
#ifndef __CPU_SIM
#include "acl/acl.h"
#endif
extern "C" __global__ [aicore] void _kernel(__gm__ float *out);
void Launch_kernel(float *out, void *stream) {
  _kernel<<<1, nullptr, stream>>>((__gm__ float *)out);
}
