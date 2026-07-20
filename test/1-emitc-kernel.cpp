// ============================================================================
// 文件 1：EmitC 后端输出 — kernel.cpp
//
// 命令：ptoas --pto-arch=a5 --pto-backend=emitc kernel.pto
// 输入 PTO IR: pto.print ins("cst=%d\n", %cst : i8)
// 输出：C++ 源码，包含 cce::printf("cst=%d\n", v2)
// ============================================================================

#include "pto/pto-inst.hpp"
using namespace pto;

enum class PTOAutoSyncTailMode : int {
  kBarrierAll = 0,
  kSetWaitMte3ToSEvent0 = 1,
};

static AICORE inline void ptoas_auto_sync_tail(
    PTOAutoSyncTailMode mode = PTOAutoSyncTailMode::kBarrierAll) {
  switch (mode) {
  case PTOAutoSyncTailMode::kSetWaitMte3ToSEvent0:
    set_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
    break;
  case PTOAutoSyncTailMode::kBarrierAll:
  default:
    pipe_barrier(PIPE_ALL);
    break;
  }
}

template <typename Ptr>
static AICORE inline void PTOAS__DCCI_SINGLE_CACHE_LINE(Ptr ptr) {
  dcci((__gm__ void*)ptr, cache_line_t::SINGLE_CACHE_LINE);
}

extern "C" __global__ AICORE void vbr_i8_kernel_2d(__gm__ int8_t* v1) {
  const int8_t v2 = -7;
  using T = float;
  cce::printf("cst=%d\n", v2);    // <-- pto.print 降级结果
  return;
}
