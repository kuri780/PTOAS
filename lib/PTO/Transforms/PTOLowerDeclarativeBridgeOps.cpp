// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

//===- PTOLowerDeclarativeBridgeOps.cpp - typed Cube bridge lowering -----===//
//===----------------------------------------------------------------------===//
//
// One pass owns typed direct/custom Cube bridge patterns. Route policy only
// selects the ops; adapters own operand access and specialization semantics.
//
//===----------------------------------------------------------------------===//

#include "PTO/IR/PTO.h"
#include "PTO/Transforms/VPTOBridgeRegistry.h"
#include "PTO/Transforms/VPTOBridgeWhitelist.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::pto {

#define GEN_PASS_DECL_PTOLOWERDECLARATIVEBRIDGEOPS
#define GEN_PASS_DEF_PTOLOWERDECLARATIVEBRIDGEOPS
#include "PTO/Transforms/Passes.h.inc"

namespace {

static FailureOr<Value> getPlannedAddress(Operation *anchor, Value tile,
                                          llvm::StringRef role) {
  auto tileType = dyn_cast<TileBufType>(tile.getType());
  auto alloc = tile.getDefiningOp<AllocTileOp>();
  if (!tileType || !alloc || !alloc.getAddr()) {
    anchor->emitError() << "Cube bridge " << role
                        << " must be a tile_buf from a planned alloc_tile";
    return failure();
  }
  return alloc.getAddr();
}

struct TMatmulMxBridgeAdapter {
  static BridgeEntryId getEntry() { return BridgeEntryId::CubeTMatmulMx; }
  static LogicalResult collectOperands(TMatmulMxOp op, SmallVectorImpl<Value> &args) {
    return collectTMatmulMxBridgeOperands(op, args);
  }
  static FailureOr<DictionaryAttr> collectSpec(TMatmulMxOp op, PatternRewriter &rewriter) {
    SmallVector<NamedAttribute> fields;
    Value tiles[] = {op.getDst(), op.getA(), op.getAScale(), op.getB(), op.getBScale()};
    StringRef names[] = {"dst", "a", "a_scale", "b", "b_scale"};
    for (auto [name, tile] : llvm::zip_equal(names, tiles))
      fields.push_back(rewriter.getNamedAttr(name, TypeAttr::get(cast<TileBufType>(tile.getType()))));
    return DictionaryAttr::get(rewriter.getContext(), fields);
  }
};

struct TMatmulBridgeAdapter {
  static BridgeEntryId getEntry() { return BridgeEntryId::CubeTMatmul; }

  static LogicalResult collectOperands(TMatmulOp op,
                                       SmallVectorImpl<Value> &arguments) {
    Value tiles[] = {op.getDst(), op.getLhs(), op.getRhs()};
    StringRef roles[] = {"dst", "lhs", "rhs"};
    for (auto [tile, role] : llvm::zip_equal(tiles, roles)) {
      FailureOr<Value> address = getPlannedAddress(op, tile, role);
      if (failed(address)) {
        return failure();
      }
      arguments.push_back(*address);
    }
    return success();
  }

  static FailureOr<DictionaryAttr> collectSpec(TMatmulOp op,
                                               PatternRewriter &rewriter) {
    SmallVector<NamedAttribute> fields;
    fields.push_back(rewriter.getNamedAttr(
        "dst", TypeAttr::get(cast<TileBufType>(op.getDst().getType()))));
    fields.push_back(rewriter.getNamedAttr(
        "lhs", TypeAttr::get(cast<TileBufType>(op.getLhs().getType()))));
    fields.push_back(rewriter.getNamedAttr(
        "rhs", TypeAttr::get(cast<TileBufType>(op.getRhs().getType()))));
    if (Attribute phase = op->getAttr("acc_phase")) {
      fields.push_back(rewriter.getNamedAttr("phase", phase));
    }
    return DictionaryAttr::get(rewriter.getContext(), fields);
  }
};

template <typename OpTy, typename Adapter>
class LowerDirectBridgeOp final : public OpRewritePattern<OpTy> {
public:
  LowerDirectBridgeOp(MLIRContext *context,
                      llvm::DenseSet<Operation *> &lowered)
      : OpRewritePattern<OpTy>(context), lowered(lowered) {}

  LogicalResult matchAndRewrite(OpTy op,
                                PatternRewriter &rewriter) const override {
    if (op->getNumResults() != 0) {
      return rewriter.notifyMatchFailure(op, "tensor-result form is custom");
    }
    SmallVector<Value> arguments;
    if (failed(Adapter::collectOperands(op, arguments))) {
      return failure();
    }
    FailureOr<DictionaryAttr> spec = Adapter::collectSpec(op, rewriter);
    if (failed(spec)) {
      return failure();
    }
    const BridgeFunctionDesc *entry = lookupBridgeEntry(Adapter::getEntry());
    if (!entry) {
      return op.emitError("typed Cube adapter references no registry entry");
    }
    SmallVector<Operation *> allocations;
    for (Value tile : {op.getDst(), op.getLhs(), op.getRhs()}) {
      if (auto alloc = tile.getDefiningOp<AllocTileOp>()) {
        allocations.push_back(alloc);
      }
    }
    rewriter.create<BridgeCallOp>(
        op.getLoc(), TypeRange{}, entry->symbol,
        rewriter.getStringAttr(getBridgeEntryName(entry->id)), nullptr, *spec,
        arguments);
    rewriter.eraseOp(op);
    lowered.insert(allocations.begin(), allocations.end());
    return success();
  }

private:
  llvm::DenseSet<Operation *> &lowered;
};

class LowerTMatmulMxBridgeOp final : public OpRewritePattern<TMatmulMxOp> {
public:
  using OpRewritePattern<TMatmulMxOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(TMatmulMxOp op, PatternRewriter &rewriter) const override {
    if (op->getNumResults() != 0) return rewriter.notifyMatchFailure(op, "tensor-result form is custom");
    SmallVector<Value> args;
    if (failed(TMatmulMxBridgeAdapter::collectOperands(op, args))) return failure();
    auto spec = TMatmulMxBridgeAdapter::collectSpec(op, rewriter);
    if (failed(spec)) return failure();
    const BridgeFunctionDesc *entry = lookupBridgeEntry(BridgeEntryId::CubeTMatmulMx);
    rewriter.create<BridgeCallOp>(op.getLoc(), TypeRange{}, entry->symbol,
      rewriter.getStringAttr(getBridgeEntryName(entry->id)), nullptr, *spec, args);
    op.erase();
    return success();
  }
};

struct PTOLowerDeclarativeBridgeOpsPass final
    : impl::PTOLowerDeclarativeBridgeOpsBase<PTOLowerDeclarativeBridgeOpsPass> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PTOLowerDeclarativeBridgeOpsPass)

  void runOnOperation() override {
    func::FuncOp function = getOperation();
    FailureOr<BridgeRoutePolicy> policy =
        loadBridgeRoutePolicy(whitelistPath, llvm::errs());
    if (failed(policy)) {
      signalPassFailure();
      return;
    }
    bool enableMatmul = llvm::is_contained(policy->families.cube.enabledOps, StringRef("pto.tmatmul"));
    bool enableMatmulMx = llvm::is_contained(policy->families.cube.enabledOps, StringRef("pto.tmatmul.mx"));
    if (!enableMatmul && !enableMatmulMx) return;

    llvm::DenseSet<Operation *> loweredAllocs;
    RewritePatternSet patterns(&getContext());
    if (enableMatmul)
      patterns.add<LowerDirectBridgeOp<TMatmulOp, TMatmulBridgeAdapter>>(&getContext(), loweredAllocs);
    if (enableMatmulMx)
      patterns.add<LowerTMatmulMxBridgeOp>(&getContext());
    if (failed(applyPatternsAndFoldGreedily(function, std::move(patterns)))) {
      signalPassFailure();
      return;
    }
    for (Operation *operation : loweredAllocs) {
      if (operation->use_empty()) {
        operation->erase();
      }
    }
  }
};

} // namespace

std::unique_ptr<Pass> createPTOLowerDeclarativeBridgeOpsPass() {
  return std::make_unique<PTOLowerDeclarativeBridgeOpsPass>();
}

} // namespace mlir::pto
