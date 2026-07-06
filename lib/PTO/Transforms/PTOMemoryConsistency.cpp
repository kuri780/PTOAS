// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "PTO/IR/PTO.h"
#include "PTO/Transforms/InsertSync/SyncMacroModel.h"
#include "PTO/Transforms/MemoryConsistencyAttrs.h"
#include "PTO/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseSet.h"

namespace mlir {
namespace pto {
#define GEN_PASS_DEF_PTOMEMORYCONSISTENCY
#include "PTO/Transforms/Passes.h.inc"
} // namespace pto
} // namespace mlir

using namespace mlir;
using namespace mlir::pto;

namespace {

static bool isGmAddressSpace(pto::AddressSpace space) {
  return space == pto::AddressSpace::GM || space == pto::AddressSpace::Zero;
}

struct TNotifyReleaseState {
  bool drainMte2 = false;
  bool drainMte3 = false;
  bool drainFix = false;
  bool needsDsbDdr = false;
  bool needsGmCacheCmo = false;

  void merge(const TNotifyReleaseState &other) {
    drainMte2 |= other.drainMte2;
    drainMte3 |= other.drainMte3;
    drainFix |= other.drainFix;
    needsDsbDdr |= other.needsDsbDdr;
    needsGmCacheCmo |= other.needsGmCacheCmo;
  }

  void clear() {
    drainMte2 = false;
    drainMte3 = false;
    drainFix = false;
    needsDsbDdr = false;
    needsGmCacheCmo = false;
  }

  void applyBarrier(pto::PIPE pipe) {
    switch (pipe) {
    case pto::PIPE::PIPE_MTE2:
      drainMte2 = false;
      break;
    case pto::PIPE::PIPE_MTE3:
      drainMte3 = false;
      break;
    case pto::PIPE::PIPE_FIX:
      drainFix = false;
      break;
    case pto::PIPE::PIPE_ALL:
      drainMte2 = false;
      drainMte3 = false;
      drainFix = false;
      break;
    default:
      break;
    }
  }

  void applyFenceBarrierAll(pto::FenceScope scope) {
    if (scope != pto::FenceScope::GM && scope != pto::FenceScope::All)
      return;
    if (drainMte3 || drainFix || needsGmCacheCmo)
      return;
    needsDsbDdr = false;
  }

  void applyCmoCacheInvalid(pto::AddressSpace space) {
    if (!isGmAddressSpace(space))
      return;
    needsGmCacheCmo = false;
  }
};

struct SignalAcquireState {
  bool pendingInvalidateGmCache = false;
  bool dirtyGmCache = false;

  void merge(const SignalAcquireState &other) {
    pendingInvalidateGmCache |= other.pendingInvalidateGmCache;
    dirtyGmCache |= other.dirtyGmCache;
  }

  void consumeAcquire() {
    pendingInvalidateGmCache = false;
    dirtyGmCache = false;
  }

  void applyCmoCacheInvalid(pto::AddressSpace space) {
    if (!isGmAddressSpace(space))
      return;
    dirtyGmCache = false;
    pendingInvalidateGmCache = false;
  }
};

static bool isGmScalarMemory(Type type) {
  if (auto ptrTy = dyn_cast<pto::PtrType>(type)) {
    pto::AddressSpace space = ptrTy.getMemorySpace().getAddressSpace();
    return isGmAddressSpace(space);
  }

  if (auto memTy = dyn_cast<MemRefType>(type)) {
    auto spaceAttr = dyn_cast_or_null<pto::AddressSpaceAttr>(memTy.getMemorySpace());
    return !spaceAttr || isGmAddressSpace(spaceAttr.getAddressSpace());
  }

  return false;
}

static TNotifyReleaseState getMte2PayloadReadReleaseState() {
  TNotifyReleaseState state;
  state.drainMte2 = true;
  return state;
}

static TNotifyReleaseState getMte3GmWriteReleaseState() {
  TNotifyReleaseState state;
  state.drainMte3 = true;
  state.needsDsbDdr = true;
  return state;
}

static TNotifyReleaseState getFixGmWriteReleaseState() {
  TNotifyReleaseState state;
  state.drainFix = true;
  state.needsDsbDdr = true;
  return state;
}

static TNotifyReleaseState getCacheableGmStoreReleaseState() {
  TNotifyReleaseState state;
  state.needsGmCacheCmo = true;
  state.needsDsbDdr = true;
  return state;
}

static TNotifyReleaseState getReleaseStateForMacroModel(Operation *op) {
  TNotifyReleaseState state;
  auto model = getSyncMacroModel(op);
  if (!model)
    return state;

  for (const SyncMacroPhase &phase : model->phases) {
    // Macro MTE3 phases write GM payloads internally. A following TNotify must
    // publish its signal only after those stores are drained and DDR-visible.
    if (phase.pipe == PipelineType::PIPE_MTE3) {
      state.drainMte3 = true;
      state.needsDsbDdr = true;
    }
  }
  return state;
}

static TNotifyReleaseState getDirectTNotifyReleaseState(Operation *op) {
  if (isa<pto::BarrierOp, pto::CmoCacheInvalidOp, pto::FenceBarrierAllOp>(op))
    return {};

  if (auto store = dyn_cast<pto::StoreScalarOp>(op)) {
    if (isGmScalarMemory(store.getPtr().getType()))
      return getCacheableGmStoreReleaseState();
  }

  if (isa<pto::TLoadOp, pto::TPrefetchOp>(op))
    return getMte2PayloadReadReleaseState();

  if (auto tstore = dyn_cast<pto::TStoreOp>(op)) {
    if (tstore.getPipe() == pto::PIPE::PIPE_MTE3)
      return getMte3GmWriteReleaseState();
    if (tstore.getPipe() == pto::PIPE::PIPE_FIX)
      return getFixGmWriteReleaseState();
    return {};
  }

  if (isa<pto::TStoreFPOp>(op))
    return getFixGmWriteReleaseState();

  TNotifyReleaseState macroState = getReleaseStateForMacroModel(op);
  if (macroState.drainMte3 || macroState.drainFix ||
      macroState.needsDsbDdr || macroState.needsGmCacheCmo)
    return macroState;

  return {};
}

static TNotifyReleaseState collectTNotifyReleaseState(Operation *op) {
  TNotifyReleaseState state = getDirectTNotifyReleaseState(op);
  for (Region &region : op->getRegions())
    for (Block &block : region)
      for (Operation &nested : block)
        state.merge(collectTNotifyReleaseState(&nested));
  return state;
}

static TNotifyReleaseState collectTNotifyReleaseState(Region &region) {
  TNotifyReleaseState state;
  for (Block &block : region)
    for (Operation &nested : block)
      state.merge(collectTNotifyReleaseState(&nested));
  return state;
}

static void applyFenceBarrierAllForSummary(pto::FenceBarrierAllOp fence,
                                           TNotifyReleaseState &state) {
  if (fence.getScope().getScope() != pto::FenceScope::GM &&
      fence.getScope().getScope() != pto::FenceScope::All)
    return;

  // The real annotation pass inserts the pending GM-write pipe drain before a
  // barrier_all.  Loop summaries must model that transfer without mutating IR,
  // otherwise already-released loop-carried writes are reported again at the
  // next iteration's TNotify.
  state.drainMte3 = false;
  state.drainFix = false;
  state.applyFenceBarrierAll(fence.getScope().getScope());
}

static TNotifyReleaseState getTNotifyReleaseExitStateForBlock(
    Block &block, TNotifyReleaseState pendingState);

static TNotifyReleaseState
getTNotifyReleaseExitState(Operation *op,
                           TNotifyReleaseState pendingState = {}) {
  if (isa<pto::TNotifyOp>(op))
    pendingState.clear();

  pendingState.merge(getDirectTNotifyReleaseState(op));

  TNotifyReleaseState regionEntryState = pendingState;
  TNotifyReleaseState combinedRegionExitState;
  for (Region &region : op->getRegions()) {
    if (region.hasOneBlock()) {
      combinedRegionExitState.merge(
          getTNotifyReleaseExitStateForBlock(region.front(), regionEntryState));
      continue;
    }

    TNotifyReleaseState regionExitState = regionEntryState;
    regionExitState.merge(collectTNotifyReleaseState(region));
    combinedRegionExitState.merge(regionExitState);
  }
  pendingState.merge(combinedRegionExitState);

  if (auto barrier = dyn_cast<pto::BarrierOp>(op))
    pendingState.applyBarrier(barrier.getPipe().getPipe());
  if (auto cmo = dyn_cast<pto::CmoCacheInvalidOp>(op))
    pendingState.applyCmoCacheInvalid(cmo.getSpace().getAddressSpace());
  if (auto fence = dyn_cast<pto::FenceBarrierAllOp>(op))
    applyFenceBarrierAllForSummary(fence, pendingState);
  return pendingState;
}

static TNotifyReleaseState getTNotifyReleaseExitStateForBlock(
    Block &block, TNotifyReleaseState pendingState) {
  for (Operation &op : block)
    pendingState = getTNotifyReleaseExitState(&op, pendingState);
  return pendingState;
}

static bool isLoopLikeOp(Operation *op) {
  return isa<scf::ForOp, scf::WhileOp, scf::ParallelOp, scf::ForallOp>(op);
}

static func::FuncOp lookupCallee(func::CallOp call) {
  return SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(
      call.getOperation(), call.getCalleeAttr());
}

static bool isMemoryConsistencyRelevantDirectOp(Operation *op) {
  if (isa<pto::BarrierOp, pto::CmoCacheInvalidOp, pto::FenceBarrierAllOp, pto::TNotifyOp,
          pto::TWaitOp, pto::TTestOp, pto::TLoadOp, pto::TPrefetchOp,
          pto::TStoreOp, pto::TStoreFPOp>(op))
    return true;

  if (auto load = dyn_cast<pto::LoadScalarOp>(op))
    return isGmScalarMemory(load.getPtr().getType());
  if (auto store = dyn_cast<pto::StoreScalarOp>(op))
    return isGmScalarMemory(store.getPtr().getType());

  TNotifyReleaseState macroState = getReleaseStateForMacroModel(op);
  return macroState.drainMte2 || macroState.drainMte3 ||
         macroState.drainFix || macroState.needsDsbDdr ||
         macroState.needsGmCacheCmo;
}

static bool calleeContainsMemoryConsistencyRelevantOps(
    func::FuncOp callee, llvm::DenseSet<Operation *> &activeCallees) {
  if (!callee || callee.isExternal())
    return false;
  if (!activeCallees.insert(callee.getOperation()).second)
    return false;

  WalkResult result = callee.walk([&](Operation *op) -> WalkResult {
    if (op == callee.getOperation())
      return WalkResult::advance();

    if (auto nestedCall = dyn_cast<func::CallOp>(op)) {
      func::FuncOp nestedCallee = lookupCallee(nestedCall);
      if (calleeContainsMemoryConsistencyRelevantOps(nestedCallee,
                                                     activeCallees))
        return WalkResult::interrupt();
      return WalkResult::advance();
    }

    if (isMemoryConsistencyRelevantDirectOp(op))
      return WalkResult::interrupt();
    return WalkResult::advance();
  });

  activeCallees.erase(callee.getOperation());
  return result.wasInterrupted();
}

static bool diagnoseNonInlinedMemoryConsistencyCalls(ModuleOp module) {
  bool hasFailure = false;
  for (auto func : module.getOps<func::FuncOp>()) {
    if (func.isExternal())
      continue;
    // Entry wrappers are launch/orchestration functions in EmitC tests.  They
    // can call several kernel functions that are analyzed independently by this
    // module pass, so rejecting those calls would incorrectly forbid the normal
    // multi-kernel entry shape.  The unsafe case is a non-inlined call from
    // inside an actual kernel body, where the caller-side signal/fence cannot
    // see payload actions hidden in the callee.
    if (func->hasAttr("pto.entry"))
      continue;

    func.walk([&](func::CallOp call) {
      func::FuncOp callee = lookupCallee(call);
      if (!callee || callee.isExternal())
        return;

      llvm::DenseSet<Operation *> activeCallees;
      if (!calleeContainsMemoryConsistencyRelevantOps(callee, activeCallees))
        return;

      call.emitOpError()
          << "calls @" << callee.getSymName()
          << ", which contains PTO memory consistency relevant operations; "
             "inline the callee before `pto-memory-consistency` or keep "
             "payload, CMO, fence, and signal operations in the caller";
      hasFailure = true;
    });
  }
  return hasFailure;
}

static void setTNotifyReleaseAttrs(pto::TNotifyOp op,
                                   const TNotifyReleaseState &state) {
  op->removeAttr(kTNotifyDrainMte2AttrName);
  op->removeAttr(kTNotifyDrainMte3AttrName);
  if (state.drainMte2)
    op->setAttr(kTNotifyDrainMte2AttrName, UnitAttr::get(op.getContext()));
  if (state.drainMte3)
    op->setAttr(kTNotifyDrainMte3AttrName, UnitAttr::get(op.getContext()));
}

static void setTNotifyPipeDrainAttrs(pto::TNotifyOp op,
                                     const TNotifyReleaseState &state) {
  TNotifyReleaseState emitState;
  emitState.drainMte2 = state.drainMte2;
  setTNotifyReleaseAttrs(op, emitState);
}

static void diagnoseTNotifyRelease(pto::TNotifyOp op,
                                   const TNotifyReleaseState &state,
                                   bool &hasFailure) {
  if (state.needsGmCacheCmo) {
    op.emitOpError()
        << "requires explicit "
           "`pto.cmo.cacheinvalid all #pto.address_space<gm>` before "
           "`pto.fence.barrier_all #pto.fence_scope<gm>` when publishing a "
           "signal after cacheable GM scalar stores";
    hasFailure = true;
    return;
  }
  if (state.needsDsbDdr) {
    op.emitOpError()
        << "requires explicit `pto.fence.barrier_all #pto.fence_scope<gm>` "
           "before publishing a signal after GM writes; "
           "PTOAS inserts the required MTE3/FIX pipe drain before "
           "`pto.fence.barrier_all` when needed";
    hasFailure = true;
  }
}

static void insertDrainsBeforeBarrierAll(pto::FenceBarrierAllOp fence,
                                         TNotifyReleaseState &state) {
  if (fence.getScope().getScope() != pto::FenceScope::GM &&
      fence.getScope().getScope() != pto::FenceScope::All)
    return;
  OpBuilder builder(fence);
  auto insertBarrier = [&](pto::PIPE pipe) {
    builder.create<pto::BarrierOp>(
        fence.getLoc(), pto::PipeAttr::get(fence.getContext(), pipe));
  };
  if (state.drainMte3) {
    insertBarrier(pto::PIPE::PIPE_MTE3);
    state.drainMte3 = false;
  }
  if (state.drainFix) {
    insertBarrier(pto::PIPE::PIPE_FIX);
    state.drainFix = false;
  }
}

static void markNestedTNotifyWithState(Operation *op,
                                       const TNotifyReleaseState &state,
                                       bool &hasFailure) {
  op->walk([&](pto::TNotifyOp notify) {
    diagnoseTNotifyRelease(notify, state, hasFailure);
    setTNotifyPipeDrainAttrs(notify, state);
  });
}

static void markNestedTNotifyWithState(Region &region,
                                       const TNotifyReleaseState &state,
                                       bool &hasFailure) {
  for (Block &block : region) {
    for (Operation &nested : block)
      markNestedTNotifyWithState(&nested, state, hasFailure);
  }
}

static TNotifyReleaseState
annotateTNotifyReleaseForBlock(Block &block,
                               TNotifyReleaseState entryPendingState,
                               TNotifyReleaseState loopCarriedState,
                               bool &hasFailure) {
  TNotifyReleaseState pendingState = entryPendingState;
  for (Operation &op : block) {
    if (auto notify = dyn_cast<pto::TNotifyOp>(op)) {
      TNotifyReleaseState notifyState = pendingState;
      notifyState.merge(loopCarriedState);
      diagnoseTNotifyRelease(notify, notifyState, hasFailure);
      setTNotifyPipeDrainAttrs(notify, notifyState);
      pendingState.clear();
    }

    pendingState.merge(getDirectTNotifyReleaseState(&op));

    TNotifyReleaseState regionEntryState = pendingState;
    TNotifyReleaseState combinedRegionExitState;
    for (Region &region : op.getRegions()) {
      TNotifyReleaseState nestedLoopCarriedState = loopCarriedState;
      if (isLoopLikeOp(&op))
        nestedLoopCarriedState.merge(getTNotifyReleaseExitState(&op));

      if (region.hasOneBlock()) {
        combinedRegionExitState.merge(annotateTNotifyReleaseForBlock(
            region.front(), regionEntryState, nestedLoopCarriedState,
            hasFailure));
      } else {
        TNotifyReleaseState regionState = collectTNotifyReleaseState(region);
        TNotifyReleaseState nestedNotifyState = regionEntryState;
        nestedNotifyState.merge(nestedLoopCarriedState);
        nestedNotifyState.merge(regionState);
        markNestedTNotifyWithState(region, nestedNotifyState, hasFailure);

        TNotifyReleaseState regionExitState = regionEntryState;
        regionExitState.merge(regionState);
        combinedRegionExitState.merge(regionExitState);
      }
    }
    pendingState.merge(combinedRegionExitState);

    if (auto barrier = dyn_cast<pto::BarrierOp>(op))
      pendingState.applyBarrier(barrier.getPipe().getPipe());
    if (auto cmo = dyn_cast<pto::CmoCacheInvalidOp>(op))
      pendingState.applyCmoCacheInvalid(cmo.getSpace().getAddressSpace());
    if (auto fence = dyn_cast<pto::FenceBarrierAllOp>(op)) {
      insertDrainsBeforeBarrierAll(fence, pendingState);
      pendingState.applyFenceBarrierAll(fence.getScope().getScope());
    }
  }
  return pendingState;
}

static bool annotateTNotifyRelease(ModuleOp module) {
  bool hasFailure = false;
  for (auto func : module.getOps<func::FuncOp>()) {
    if (func.isExternal())
      continue;

    if (func.getBody().hasOneBlock()) {
      (void)annotateTNotifyReleaseForBlock(func.getBody().front(),
                                           TNotifyReleaseState{},
                                           TNotifyReleaseState{},
                                           hasFailure);
      continue;
    }

    // Be conservative for pre-existing CFG: without a path-sensitive CFG data
    // flow here, every TNotify may observe any release-relevant work in the
    // function.
    TNotifyReleaseState funcState = collectTNotifyReleaseState(func.getBody());
    markNestedTNotifyWithState(func.getBody(), funcState, hasFailure);
  }
  return hasFailure;
}

static void clearAcquireAttrs(pto::LoadScalarOp op) {
  op->removeAttr(kAcquireInvalidateGmCacheAttrName);
}

static void diagnoseAcquireLoad(pto::LoadScalarOp op,
                                const SignalAcquireState &state,
                                bool &hasFailure) {
  if (!state.pendingInvalidateGmCache ||
      !isGmScalarMemory(op.getPtr().getType()))
    return;
  if (state.dirtyGmCache) {
    op.emitOpError()
        << "cannot perform a cacheable GM load after signal acquire while "
           "dirty GM cache may exist; insert explicit "
           "`pto.cmo.cacheinvalid all #pto.address_space<gm>` before the load";
    hasFailure = true;
    return;
  }
  op.emitOpError()
      << "requires explicit `pto.cmo.cacheinvalid all #pto.address_space<gm>` "
         "before a cacheable GM load after `pto.comm.twait` or successful "
         "`pto.comm.ttest`";
  hasFailure = true;
}

static void consumeAcquireAfterDiagnostic(SignalAcquireState &state) {
  if (state.pendingInvalidateGmCache)
    state.consumeAcquire();
}

static SignalAcquireState collectSignalAcquireState(Operation *op) {
  SignalAcquireState state;
  if (isa<pto::TWaitOp, pto::TTestOp>(op))
    state.pendingInvalidateGmCache = true;
  if (auto store = dyn_cast<pto::StoreScalarOp>(op);
      store && isGmScalarMemory(store.getPtr().getType()))
    state.dirtyGmCache = true;
  if (auto cmo = dyn_cast<pto::CmoCacheInvalidOp>(op))
    state.applyCmoCacheInvalid(cmo.getSpace().getAddressSpace());

  for (Region &region : op->getRegions())
    for (Block &block : region)
      for (Operation &nested : block)
        state.merge(collectSignalAcquireState(&nested));
  return state;
}

static SignalAcquireState collectSignalAcquireState(Region &region) {
  SignalAcquireState state;
  for (Block &block : region)
    for (Operation &nested : block)
      state.merge(collectSignalAcquireState(&nested));
  return state;
}

static void markNestedAcquireLoadsWithState(Operation *op,
                                            SignalAcquireState state,
                                            bool &hasFailure) {
  op->walk([&](pto::LoadScalarOp load) {
    clearAcquireAttrs(load);
    diagnoseAcquireLoad(load, state, hasFailure);
    consumeAcquireAfterDiagnostic(state);
  });
}

static void markNestedAcquireLoadsWithState(Region &region,
                                            SignalAcquireState state,
                                            bool &hasFailure) {
  for (Block &block : region) {
    for (Operation &nested : block)
      markNestedAcquireLoadsWithState(&nested, state, hasFailure);
  }
}

static SignalAcquireState
annotateSignalAcquireForBlock(Block &block, SignalAcquireState entryState,
                              bool &hasFailure) {
  SignalAcquireState state = entryState;
  for (Operation &op : block) {
    if (auto load = dyn_cast<pto::LoadScalarOp>(op)) {
      clearAcquireAttrs(load);
      diagnoseAcquireLoad(load, state, hasFailure);
      consumeAcquireAfterDiagnostic(state);
    }

    if (auto store = dyn_cast<pto::StoreScalarOp>(op);
        store && isGmScalarMemory(store.getPtr().getType()))
      state.dirtyGmCache = true;

    if (isa<pto::TWaitOp, pto::TTestOp>(op))
      state.pendingInvalidateGmCache = true;

    if (auto cmo = dyn_cast<pto::CmoCacheInvalidOp>(op))
      state.applyCmoCacheInvalid(cmo.getSpace().getAddressSpace());

    SignalAcquireState combinedRegionExitState;
    for (Region &region : op.getRegions()) {
      if (region.hasOneBlock()) {
        combinedRegionExitState.merge(
            annotateSignalAcquireForBlock(region.front(), state, hasFailure));
      } else {
        markNestedAcquireLoadsWithState(region, state, hasFailure);
        SignalAcquireState regionState = collectSignalAcquireState(region);
        SignalAcquireState regionExitState = state;
        regionExitState.merge(regionState);
        combinedRegionExitState.merge(regionExitState);
      }
    }

    if (isLoopLikeOp(&op))
      combinedRegionExitState.merge(state);
    state.merge(combinedRegionExitState);
  }
  return state;
}

static bool annotateSignalAcquire(ModuleOp module) {
  bool hasFailure = false;
  for (auto func : module.getOps<func::FuncOp>()) {
    if (func.isExternal())
      continue;

    if (func.getBody().hasOneBlock()) {
      (void)annotateSignalAcquireForBlock(func.getBody().front(),
                                          SignalAcquireState{}, hasFailure);
      continue;
    }

    SignalAcquireState funcState = collectSignalAcquireState(func.getBody());
    markNestedAcquireLoadsWithState(func.getBody(), funcState, hasFailure);
  }
  return hasFailure;
}

struct PTOMemoryConsistencyPass
    : public mlir::pto::impl::PTOMemoryConsistencyBase<
          PTOMemoryConsistencyPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    bool callFailed = diagnoseNonInlinedMemoryConsistencyCalls(module);
    bool releaseFailed = annotateTNotifyRelease(module);
    bool acquireFailed = annotateSignalAcquire(module);
    if (callFailed || releaseFailed || acquireFailed)
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> mlir::pto::createPTOMemoryConsistencyPass() {
  return std::make_unique<PTOMemoryConsistencyPass>();
}
