// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

//===- PTOLowerPipeFamilyOps.cpp - TPipe family bridge lowering ----------===//
//===----------------------------------------------------------------------===//
//
// TPipe family pass of the VPTO C++ interface bridge. It understands the
// semantics of the internal pipe ops (initialize_l2l_pipe / tpush / tpop /
// tfree) plus the tile handles they consume (alloc_tile / declare_tile /
// tile_buf_addr) and rewrites them into generic pto.bridge_call /
// pto.bridge_inttoptr ops that carry only wrapper callee names and ABI
// values. All family semantics (config validation, storage handle flow, and
// the runtime rebinding of a declared tile to the FIFO slot returned by
// TPOP) are resolved here; the generic bridge lowering pass only sees the
// resulting bridge ops.
//
// Routing is whitelist driven: the wrapper callee of every converted op is
// looked up in the bridge whitelist by IR op name, so this pass holds no
// hardcoded wrapper entry names. Functions without pipe family ops are
// left untouched entirely (their tile handles keep flowing through the
// regular FoldTileBufIntrinsics path).
//
//===----------------------------------------------------------------------===//

#include "PTO/IR/PTO.h"
#include "PTO/Transforms/VPTOBridgeRegistry.h"
#include "PTO/Transforms/VPTOBridgeTokens.h"
#include "PTO/Transforms/VPTOBridgeWhitelist.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
namespace pto {

#define GEN_PASS_DECL_PTOLOWERPIPEFAMILYOPS
#define GEN_PASS_DEF_PTOLOWERPIPEFAMILYOPS
#include "PTO/Transforms/Passes.h.inc"

namespace {

/// Emits a bridge call with no results and no synthesized storage.
static BridgeCallOp emitVoidBridgeCall(OpBuilder &builder, Location loc,
                                       BridgeEntryId entryId,
                                       DictionaryAttr specialization,
                                       ValueRange args) {
  const BridgeFunctionDesc *entry = lookupBridgeEntry(entryId);
  return builder.create<BridgeCallOp>(
      loc, TypeRange{}, entry->symbol,
      builder.getStringAttr(getBridgeEntryName(entryId)), nullptr,
      specialization, args);
}

/// Returns the address value a tile_buf_addr operand resolves to, or nullptr
/// when the source cannot be resolved (the caller emits the diagnostic).
/// alloc_tile carries the planned address as an i64 operand; a declare_tile
/// rebound by TPOP resolves to the FIFO slot address returned by the pop.
static Value resolveTileAddress(Value tile, OpBuilder &builder,
                                llvm::DenseMap<Value, Value> &popAddresses) {
  if (auto alloc = tile.getDefiningOp<AllocTileOp>()) {
    return alloc.getAddr();
  }
  if (isa_and_nonnull<DeclareTileOp>(tile.getDefiningOp())) {
    auto it = popAddresses.find(tile);
    if (it == popAddresses.end()) {
      return nullptr;
    }
    return it->second;
  }
  return nullptr;
}

struct PTOLowerPipeFamilyOpsPass final
    : public impl::PTOLowerPipeFamilyOpsBase<PTOLowerPipeFamilyOpsPass> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PTOLowerPipeFamilyOpsPass)

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    OpBuilder builder(func);
    bool hadError = false;

    // Collect first; rewriting during the walk would invalidate the walker.
    SmallVector<InitializeL2LPipeOp> inits;
    SmallVector<TPushOp> pushes;
    SmallVector<TPopOp> pops;
    SmallVector<TFreeOp> frees;
    SmallVector<TileBufAddrOp> addrs;
    SmallVector<AllocTileOp> allocs;
    SmallVector<DeclareTileOp> decls;
    func.walk([&](Operation *op) {
      if (auto init = dyn_cast<InitializeL2LPipeOp>(op)) {
        inits.push_back(init);
      } else if (auto push = dyn_cast<TPushOp>(op)) {
        pushes.push_back(push);
      } else if (auto pop = dyn_cast<TPopOp>(op)) {
        pops.push_back(pop);
      } else if (auto free = dyn_cast<TFreeOp>(op)) {
        frees.push_back(free);
      } else if (auto addr = dyn_cast<TileBufAddrOp>(op)) {
        addrs.push_back(addr);
      } else if (auto alloc = dyn_cast<AllocTileOp>(op)) {
        allocs.push_back(alloc);
      } else if (auto decl = dyn_cast<DeclareTileOp>(op)) {
        decls.push_back(decl);
      }
    });

    // Whitelist-driven routing: the pass only acts on functions that carry
    // pipe family ops. Tile handles of pipe-less functions keep flowing
    // through the regular lowering (FoldTileBufIntrinsics), matching the
    // pre-bridge behavior.
    if (inits.empty() && pushes.empty() && pops.empty() && frees.empty()) {
      return;
    }

    std::string policyName;
    FailureOr<BridgeRoutePolicy> policy =
        loadBridgeRoutePolicy(whitelistPath, llvm::errs(), &policyName);
    if (failed(policy)) {
      signalPassFailure();
      return;
    }
    if (!policy->families.pipe.enabled) {
      return;
    }
    auto requireEntry = [&](Operation *op) -> const BridgeFunctionDesc * {
      const BridgeFunctionDesc *entry =
          lookupBridgeEntryForOp(op->getName().getStringRef());
      if (!entry || entry->family != BridgeFamily::Pipe) {
        op->emitError()
            << "VPTO Pipe bridge has no registered handler; policy: '"
            << policyName << "'";
        hadError = true;
      }
      return entry;
    };

    DictionaryAttr pipeConfig;
    DictionaryAttr producerSpec;
    DictionaryAttr consumerSpec;
    IntegerAttr splitSpec;
    auto buildCallSpec = [&](OpBuilder &at) -> DictionaryAttr {
      SmallVector<NamedAttribute> fields;
      if (pipeConfig) {
        fields.push_back(at.getNamedAttr("pipe", pipeConfig));
      }
      if (producerSpec) {
        fields.push_back(at.getNamedAttr("producer", producerSpec));
      }
      if (consumerSpec) {
        fields.push_back(at.getNamedAttr("consumer", consumerSpec));
      }
      if (splitSpec) {
        fields.push_back(at.getNamedAttr("split", splitSpec));
      }
      return DictionaryAttr::get(at.getContext(), fields);
    };

    // Phase 1: initialize_l2l_pipe -> storage-producing bridge init call.
    // The SSA pipe value becomes the bridge call result (the storage handle);
    // push/pop/free below consume that same value.
    for (InitializeL2LPipeOp init : inits) {
      const BridgeFunctionDesc *entry = requireEntry(init);
      if (!entry) {
        continue;
      }
      if (!isSupportedPipeCapability(init)) {
        init.emitError(
            "VPTO pipe bridge supports only a local pipe with dir_mask 1 "
            "(C2V) or 2 (V2C), no acc_push_epilogue, and an i32 local buffer "
            "address");
        hadError = true;
        continue;
      }
      auto pipeTokOr = buildBridgePipeToken(init);
      if (failed(pipeTokOr)) {
        init.emitError("VPTO pipe bridge failed to build the TPipe template "
                       "token from the init attributes (flag_base is "
                       "required, dir_mask must be 1, 2 or 3)");
        hadError = true;
        continue;
      }
      pipeConfig = DictionaryAttr::get(
          builder.getContext(),
          {builder.getNamedAttr("flag_base", init.getFlagBaseAttr()),
           builder.getNamedAttr("direction", init.getDirMaskAttr()),
           builder.getNamedAttr("slot_size", init.getSlotSizeAttr()),
           builder.getNamedAttr("slot_num", init.getSlotNumAttr()),
           builder.getNamedAttr("local_slot_num", builder.getI32IntegerAttr(2)),
           builder.getNamedAttr("nosplit",
                                init.getNosplitAttr()
                                    ? Attribute(init.getNosplitAttr())
                                    : Attribute(builder.getBoolAttr(false)))});
      builder.setInsertionPoint(init);
      BridgeObjectCreateOp call = builder.create<BridgeObjectCreateOp>(
          init.getLoc(), init.getPipe().getType(),
          getBridgeEntryName(BridgeEntryId::PipeInit), nullptr,
          buildCallSpec(builder), ValueRange{init.getLocalAddr()});
      // The bridge call result becomes the storage handle: push/pop/free
      // consume the same SSA value instead of the erased pipe op.
      init.getPipe().replaceAllUsesWith(call.getObject());
      init.erase();
    }

    // Phase 2: tpop -> bridge pop call; record the returned FIFO slot
    // address for the declared tile it rebinds.
    llvm::DenseMap<Value, Value> popAddresses;
    // The wrapper renders one shared TileSplitAxis template argument for the
    // push/pop/free entries, so every bridged op of the function must agree
    // on the split value. The first bridged op fixes it; later ops check
    // against it. Cross-function mismatches surface as a spec merge conflict
    // in the wrapper generation pass.
    std::optional<int64_t> bridgedSplit;
    auto checkSplitConsistency = [&](Operation *op, int64_t split,
                                     llvm::StringRef opName) {
      if (bridgedSplit && *bridgedSplit != split) {
        op->emitError() << "VPTO pipe bridge " << opName << " split " << split
                        << " does not match the split " << *bridgedSplit
                        << " already bridged in this function; the wrapper "
                           "renders one shared TileSplitAxis";
        hadError = true;
        return false;
      }
      if (!bridgedSplit) {
        auto splitTokOr = buildBridgeTileSplitToken(split);
        if (failed(splitTokOr)) {
          op->emitError() << "VPTO pipe bridge " << opName
                          << " carries an unsupported split value " << split;
          hadError = true;
          return false;
        }
        splitSpec = builder.getI64IntegerAttr(split);
        bridgedSplit = split;
      }
      return true;
    };
    for (TPopOp pop : pops) {
      const BridgeFunctionDesc *entry = requireEntry(pop);
      if (!entry) {
        continue;
      }
      if (popAddresses.count(pop.getTile())) {
        pop.emitError(
            "VPTO pipe bridge supports at most one TPOP per declared tile; "
            "sequential rebind consumption is not supported yet");
        hadError = true;
        continue;
      }
      auto consumerTileTy = dyn_cast<TileBufType>(pop.getTile().getType());
      if (!consumerTileTy) {
        pop.emitError("VPTO pipe bridge TPOP tile must be a tile_buf");
        hadError = true;
        continue;
      }
      auto consumerTokOr = buildBridgeTileToken(consumerTileTy);
      if (failed(consumerTokOr)) {
        pop.emitError("VPTO pipe bridge failed to build the consumer tile "
                      "template token for TPOP");
        hadError = true;
        continue;
      }
      if (!checkSplitConsistency(pop, pop.getSplit(), "TPOP")) {
        continue;
      }
      consumerSpec = DictionaryAttr::get(
          builder.getContext(),
          {builder.getNamedAttr("tile", TypeAttr::get(consumerTileTy))});
      builder.setInsertionPoint(pop);
      BridgeCallOp call = builder.create<BridgeCallOp>(
          pop.getLoc(), /*results=*/TypeRange{builder.getI64Type()},
          /*callee=*/entry->symbol,
          /*entry_id=*/builder.getStringAttr("pipe.pop"),
          /*instance_id=*/nullptr,
          /*specialization=*/buildCallSpec(builder),
          /*args=*/ValueRange{pop.getPipeHandle()});
      popAddresses[pop.getTile()] = call.getResults().front();
      pop.erase();
    }

    // Phase 3: tile_buf_addr -> bridge_inttoptr on the resolved address.
    for (TileBufAddrOp addr : addrs) {
      Value address = resolveTileAddress(addr.getSrc(), builder, popAddresses);
      if (!address) {
        addr.emitError(
            "VPTO pipe bridge requires tile_buf_addr sources to be a planned "
            "alloc_tile or a declare_tile rebound by tpop");
        hadError = true;
        continue;
      }
      builder.setInsertionPoint(addr);
      BridgeIntToPtrOp pointer = builder.create<BridgeIntToPtrOp>(
          addr.getLoc(), addr.getDst().getType(), address);
      addr.getDst().replaceAllUsesWith(pointer.getResult());
      addr.erase();
    }

    // Phase 4: tpush -> bridge push call on the planned alloc_tile address.
    for (TPushOp push : pushes) {
      const BridgeFunctionDesc *entry = requireEntry(push);
      if (!entry) {
        continue;
      }
      auto alloc = push.getTile().getDefiningOp<AllocTileOp>();
      if (!alloc || !alloc.getAddr()) {
        push.emitError("VPTO pipe bridge TPUSH requires a tile from an "
                       "alloc_tile with a planned address");
        hadError = true;
        continue;
      }
      auto producerTileTy = dyn_cast<TileBufType>(push.getTile().getType());
      if (!producerTileTy) {
        push.emitError("VPTO pipe bridge TPUSH tile must be a tile_buf");
        hadError = true;
        continue;
      }
      auto producerTokOr = buildBridgeTileToken(producerTileTy);
      if (failed(producerTokOr)) {
        push.emitError("VPTO pipe bridge failed to build the producer tile "
                       "template token for TPUSH");
        hadError = true;
        continue;
      }
      if (!checkSplitConsistency(push, push.getSplit(), "TPUSH")) {
        continue;
      }
      producerSpec = DictionaryAttr::get(
          builder.getContext(),
          {builder.getNamedAttr("tile", TypeAttr::get(producerTileTy))});
      builder.setInsertionPoint(push);
      emitVoidBridgeCall(builder, push.getLoc(), BridgeEntryId::PipePush,
                         buildCallSpec(builder),
                         ValueRange{push.getPipeHandle(), alloc.getAddr()});
      push.erase();
    }

    // Phase 5: tfree -> bridge free call.
    for (TFreeOp free : frees) {
      const BridgeFunctionDesc *entry = requireEntry(free);
      if (!entry) {
        continue;
      }
      if (free.getEntry()) {
        free.emitError("VPTO pipe bridge TFREE supports the pipe-entry form "
                       "without a tile operand");
        hadError = true;
        continue;
      }
      if (!checkSplitConsistency(free, free.getSplit(), "TFREE")) {
        continue;
      }
      builder.setInsertionPoint(free);
      emitVoidBridgeCall(builder, free.getLoc(), BridgeEntryId::PipeFree,
                         buildCallSpec(builder),
                         ValueRange{free.getPipeHandle()});
      free.erase();
    }

    // Phase 6: erase tile handles whose consumers are all bridged now.
    for (AllocTileOp alloc : allocs) {
      if (!alloc.use_empty()) {
        alloc.emitError("VPTO pipe bridge: alloc_tile still has users after "
                        "pipe family lowering");
        hadError = true;
        continue;
      }
      alloc.erase();
    }
    for (DeclareTileOp decl : decls) {
      if (!decl.use_empty()) {
        decl.emitError("VPTO pipe bridge: declare_tile still has users after "
                       "pipe family lowering");
        hadError = true;
        continue;
      }
      decl.erase();
    }

    if (hadError) {
      signalPassFailure();
    }
  }

private:
  /// Capability check for the pipe bridge. The concrete configuration
  /// (slot_size/slot_num/flag_base/nosplit) is read from the op attributes
  /// and flows into the generated wrapper; only genuinely unsupported forms
  /// are rejected here.
  static bool isSupportedPipeCapability(InitializeL2LPipeOp init) {
    int8_t dirMask = init.getDirMask();
    if (dirMask != 1 && dirMask != 2)
      return false;
    if (init.getAccPushEpilogueAttr())
      return false;
    auto localAddrTy = dyn_cast<IntegerType>(init.getLocalAddr().getType());
    if (!localAddrTy || localAddrTy.getWidth() != 32)
      return false;
    return true;
  }
};

} // namespace

std::unique_ptr<Pass> createPTOLowerPipeFamilyOpsPass() {
  return std::make_unique<PTOLowerPipeFamilyOpsPass>();
}

} // namespace pto
} // namespace mlir
