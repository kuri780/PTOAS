// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "PTO/IR/PTO.h"
#include "PTO/Transforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

#include <utility>

namespace mlir {
namespace pto {
#define GEN_PASS_DEF_PTOCANONICALIZEIR
#include "PTO/Transforms/Passes.h.inc"
} // namespace pto
} // namespace mlir

using namespace mlir;
using namespace mlir::pto;

namespace {

// ---------------------------------------------------------------------------
// Design note: which ops need structural rewriting vs. type-only walk
// ---------------------------------------------------------------------------
//
// This pass canonicalizes rank-2 TensorViewType / PartitionTensorViewType
// into the right-aligned rank-5 form [1, 1, 1, R, C] used by all backends
// (A3, A5, VPTO EmitC codegen and the 5D memref rank in PTOViewToMemref).
//
// Ops that carry **rank-dependent operands** must be structurally rewritten
// (their operand count or operand values change when rank changes):
//   - MakeTensorViewOp  : shape/strides expanded from 2 → 5
//   - PartitionViewOp   : offsets/sizes expanded from 2 → 5
//   - GetTensorViewDimOp / GetTensorViewStrideOp : dim index offset by +3
//
// Ops that only **carry view-typed operands/results** (no rank-dependent
// operand structure) are handled by the type walk (canonicalizeValueTypes)
// which in-place mutates TensorViewType and PartitionTensorViewType from
// rank-2 to rank-5:
//   - TAllocToAivOp, TAllocToAicOp, DeclareGlobalOp (producers)
//   - TAllocOp, TPushOp, TPopOp, TFreeOp, AicInitializePipeOp,
//     AivInitializePipeOp, TensorViewAddrOp (consumers)
//   - All PTODpsType consumers (TLoadOp, TStoreOp, TMatmulOp, etc.)
//   - All PTOPipeEntryType consumers (TPushToAivOp, TPopFromAicOp, etc.)
//
// A post-canonicalization verification (verifyNoRank2ViewSurvivors) detects
// any surviving rank-2 view types to prevent silent failures when new
// view-consuming ops with rank-dependent operands are added.
//
// NZ layout cannot appear on rank-2 views (it requires rank >= 5 with
// shape[2] == 16), so only ND and DN strides need expansion logic.
// ---------------------------------------------------------------------------

constexpr unsigned kLogicalRank2 = 2;
constexpr unsigned kCanonicalRank5 = 5;
constexpr int64_t kUnitExtent = 1;
constexpr unsigned kRank2RowDim = 0; // row dimension index in rank-2 view
constexpr unsigned kRank2ColDim = 1; // column dimension index in rank-2 view
constexpr int64_t kRank2ToRank5DimOffset = 3;

static SmallVector<int64_t, kCanonicalRank5>
rightAlignRank2Shape(ArrayRef<int64_t> shape) {
  return {kUnitExtent, kUnitExtent, kUnitExtent, shape[kRank2RowDim],
          shape[kRank2ColDim]};
}

static Value getOrCreateIndexConstant(OpBuilder &builder, Location loc,
                                      int64_t value) {
  return builder.create<arith::ConstantIndexOp>(loc, value);
}

static SmallVector<Value, kCanonicalRank5>
prependThreeValues(ValueRange values, Value fill) {
  return {fill, fill, fill, values[kRank2RowDim], values[kRank2ColDim]};
}

// ---------------------------------------------------------------------------
// Stride expansion: uses the same cumulative-product rule as
// rightAlignTo5D (InferPTOLayout.cpp) and buildGlobalTensorShapeAndStride
// (PTOToEmitC.cpp): stride[i] = shape[i+1] * stride[i+1].
//
// For a rank-2 view [R, C] right-aligned into [1, 1, 1, R, C]:
//   - ND (row-major): original strides = [C, 1]
//     padded strides: stride[2] = shape[3]*stride[3] = R*C,
//                    stride[1] = shape[2]*stride[2] = 1*R*C = R*C,
//                    stride[0] = shape[1]*stride[1] = 1*R*C = R*C
//     → [R*C, R*C, R*C, C, 1]
//
//   - DN (col-major): original strides = [1, R]
//     padded strides: stride[2] = shape[3]*stride[3] = R*1 = R,
//                    stride[1] = shape[2]*stride[2] = 1*R = R,
//                    stride[0] = shape[1]*stride[1] = 1*R = R
//     → [R, R, R, 1, R]
//
// Note: the ND branch was previously incorrectly using rowStride (=C) for
// all three leading dims, producing [C, C, C, C, 1] instead of the correct
// cumulative product [R*C, R*C, R*C, C, 1]. The DN branch was correct by
// coincidence because colStride == R and the cumulative product of unit-extent
// leading dims also collapses to R.
// ---------------------------------------------------------------------------
static SmallVector<Value, kCanonicalRank5>
buildCanonicalRank2Strides(MakeTensorViewOp op, IRRewriter &rewriter) {
  Value rowStride = op.getStrides()[kRank2RowDim];
  Value colStride = op.getStrides()[kRank2ColDim];

  rewriter.setInsertionPoint(op);
  auto loc = op.getLoc();

  auto layout = op.getLayoutAttr();

  // For ND (row-major): original strides = [rowStride, colStride]
  // where rowStride = C (shape[1]) and colStride = 1.
  // Cumulative product rule for leading dims:
  //   stride[2] = shape[3] * stride[3] = rowStride_vals * rowStride
  //   But shape[3] and stride[3] are SSA values, not constants.
  //   We compute: shape[kRank2RowDim] * rowStride for stride[2],
  //               1 * (shape[kRank2RowDim] * rowStride) for strides [0..1].
  //
  // Simplification: since shape[0..2] are all 1 (unit-extent padding),
  // the cumulative product collapses: stride[i] = stride[shift] for all
  // i < shift, where shift = kRank2ToRank5DimOffset = 3.
  //
  // For ND: stride[3] = rowStride, so stride[0..2] = rowStride.
  //         BUT wait — the cumulative rule is stride[i] = shape[i+1]*stride[i+1].
  //         stride[2] = shape[3] * stride[3] = R * rowStride.
  //         stride[1] = shape[2] * stride[2] = 1 * (R*rowStride) = R*rowStride.
  //         stride[0] = shape[1] * stride[1] = 1 * (R*rowStride) = R*rowStride.
  //         So the leading strides are NOT rowStride; they are R*rowStride.
  //
  // We must compute the product: shape[kRank2RowDim] * rowStride.
  if (layout && layout.getLayout() == Layout::DN) {
    // DN (col-major): strides = [1, R]
    // Cumulative product: stride[2] = shape[3]*stride[3] = R*1 = R,
    //                     stride[1] = 1*R = R, stride[0] = 1*R = R.
    // Since colStride = R for DN, this collapses to colStride for all
    // three leading dims. This is the same as the old DN branch.
    return {colStride, colStride, colStride, rowStride, colStride};
  }

  // ND (row-major) or no explicit layout attr (default = ND):
  // strides = [rowStride, colStride] where rowStride = C, colStride = 1.
  // Cumulative product: stride[2] = shape[kRank2RowDim] * stride[3],
  //                     stride[1] = 1 * stride[2],
  //                     stride[0] = 1 * stride[2].
  // = shape[kRank2RowDim] * rowStride for all three leading dims.
  Value rowsValue = op.getShape()[kRank2RowDim];
  Value leadingStride = rewriter.create<arith::MulIOp>(loc, rowsValue, rowStride);
  return {leadingStride, leadingStride, leadingStride, rowStride, colStride};
}

static bool isRank2ViewLike(Type type) {
  if (auto viewType = dyn_cast<TensorViewType>(type))
    return viewType.getRank() == kLogicalRank2;
  if (auto viewType = dyn_cast<PartitionTensorViewType>(type))
    return viewType.getRank() == kLogicalRank2;
  return false;
}

static Type canonicalViewType(Type type) {
  if (auto viewType = dyn_cast<TensorViewType>(type)) {
    if (viewType.getRank() == kLogicalRank2)
      return TensorViewType::get(type.getContext(),
                                 rightAlignRank2Shape(viewType.getShape()),
                                 viewType.getElementType());
    return type;
  }
  if (auto viewType = dyn_cast<PartitionTensorViewType>(type)) {
    if (viewType.getRank() == kLogicalRank2)
      return PartitionTensorViewType::get(
          type.getContext(), rightAlignRank2Shape(viewType.getShape()),
          viewType.getElementType());
    return type;
  }
  return type;
}

static bool canonicalizeValueType(Value value) {
  Type oldType = value.getType();
  Type newType = canonicalViewType(oldType);
  if (newType == oldType)
    return false;
  value.setType(newType);
  return true;
}

static LogicalResult rewriteMakeTensorView(MakeTensorViewOp op,
                                           IRRewriter &rewriter) {
  auto oldType = dyn_cast<TensorViewType>(op.getResult().getType());
  if (!oldType || oldType.getRank() != kLogicalRank2)
    return success();

  if (op.getShape().size() != kLogicalRank2 ||
      op.getStrides().size() != kLogicalRank2)
    return op.emitOpError(
        "rank-2 tensor_view must have exactly 2 shape and stride operands");

  rewriter.setInsertionPoint(op);
  Value one = getOrCreateIndexConstant(rewriter, op.getLoc(), kUnitExtent);
  SmallVector<Value, kCanonicalRank5> newShape =
      prependThreeValues(op.getShape(), one);
  SmallVector<Value, kCanonicalRank5> newStrides =
      buildCanonicalRank2Strides(op, rewriter);
  auto newType = cast<TensorViewType>(canonicalViewType(oldType));

  auto newOp = rewriter.create<MakeTensorViewOp>(
      op.getLoc(), newType, op.getPtr(), newShape, newStrides,
      op.getLayoutAttr());
  rewriter.replaceOp(op, newOp.getResult());
  return success();
}

static LogicalResult rewritePartitionView(PartitionViewOp op,
                                          IRRewriter &rewriter) {
  auto sourceType = dyn_cast<TensorViewType>(op.getSource().getType());
  auto resultType = dyn_cast<PartitionTensorViewType>(op.getResult().getType());
  if (!sourceType || !resultType)
    return success();

  if (op.getOffsets().size() != kLogicalRank2 ||
      op.getSizes().size() != kLogicalRank2)
    return success();

  if (sourceType.getRank() != kCanonicalRank5)
    return op.emitOpError(
        "rank-2 partition_tensor_view normalization expects canonical rank-5 "
        "source tensor_view");

  rewriter.setInsertionPoint(op);
  Value zero = getOrCreateIndexConstant(rewriter, op.getLoc(), 0);
  Value one = getOrCreateIndexConstant(rewriter, op.getLoc(), kUnitExtent);
  SmallVector<Value, kCanonicalRank5> newOffsets =
      prependThreeValues(op.getOffsets(), zero);
  SmallVector<Value, kCanonicalRank5> newSizes =
      prependThreeValues(op.getSizes(), one);
  auto newType = cast<PartitionTensorViewType>(canonicalViewType(resultType));

  auto newOp = rewriter.create<PartitionViewOp>(
      op.getLoc(), newType, op.getSource(), newOffsets, newSizes);
  rewriter.replaceOp(op, newOp.getResult());
  return success();
}

static Value buildCanonicalDimIndex(Value dimIndex, IRRewriter &rewriter,
                                    Location loc) {
  rewriter.setInsertionPointAfterValue(dimIndex);
  Value offset =
      getOrCreateIndexConstant(rewriter, loc, kRank2ToRank5DimOffset);
  return rewriter.create<arith::AddIOp>(loc, dimIndex, offset);
}

static void rewriteTensorViewDimOperand(Operation *op, Value dimIndex,
                                        IRRewriter &rewriter) {
  Value newDim = buildCanonicalDimIndex(dimIndex, rewriter, op->getLoc());
  op->setOperand(1, newDim);
}

static void canonicalizeFunctionType(func::FuncOp func) {
  auto oldType = func.getFunctionType();
  SmallVector<Type> inputs;
  SmallVector<Type> results;
  bool changed = false;

  inputs.reserve(oldType.getNumInputs());
  for (Type type : oldType.getInputs()) {
    Type newType = canonicalViewType(type);
    changed |= newType != type;
    inputs.push_back(newType);
  }

  results.reserve(oldType.getNumResults());
  for (Type type : oldType.getResults()) {
    Type newType = canonicalViewType(type);
    changed |= newType != type;
    results.push_back(newType);
  }

  if (changed)
    func.setFunctionType(FunctionType::get(func.getContext(), inputs, results));
}

static void canonicalizeValueTypes(func::FuncOp func) {
  canonicalizeFunctionType(func);

  func->walk([](Operation *op) {
    for (Region &region : op->getRegions()) {
      for (Block &block : region) {
        for (BlockArgument arg : block.getArguments())
          canonicalizeValueType(arg);
      }
    }

    for (OpResult result : op->getResults())
      canonicalizeValueType(result);
  });
}

/// Verify that no rank-2 view types survived canonicalization.
/// This catches cases where a new op with rank-dependent operands
/// was added but not given a structural rewrite in this pass.
static LogicalResult verifyNoRank2ViewSurvivors(func::FuncOp func) {
  bool anyFailed = false;
  func.walk([&](Operation *op) {
    for (Region &region : op->getRegions()) {
      for (Block &block : region) {
        for (BlockArgument arg : block.getArguments()) {
          if (isRank2ViewLike(arg.getType())) {
            emitError(arg.getLoc())
                << "rank-2 view type survived canonicalization: "
                << arg.getType() << " as block argument";
            anyFailed = true;
          }
        }
      }
    }
    for (OpResult result : op->getResults()) {
      if (isRank2ViewLike(result.getType())) {
        emitError(op->getLoc())
            << "rank-2 view type survived canonicalization: "
            << result.getType() << " in op " << op->getName();
        anyFailed = true;
      }
    }
  });
  return anyFailed ? failure() : success();
}

struct PTOCanonicalizeIRPass
    : public mlir::pto::impl::PTOCanonicalizeIRBase<PTOCanonicalizeIRPass> {
  void runOnOperation() override {
    func::FuncOp func = getOperation();
    SmallVector<MakeTensorViewOp> makeViews;
    SmallVector<PartitionViewOp> partitionViews;
    SmallVector<std::pair<Operation *, Value>> dimIndexOps;

    func.walk([&](MakeTensorViewOp op) {
      if (isRank2ViewLike(op.getResult().getType()))
        makeViews.push_back(op);
    });
    func.walk([&](PartitionViewOp op) {
      if (op.getOffsets().size() == kLogicalRank2 &&
          op.getSizes().size() == kLogicalRank2)
        partitionViews.push_back(op);
    });
    func.walk([&](GetTensorViewDimOp op) {
      if (isRank2ViewLike(op.getTensorView().getType()))
        dimIndexOps.emplace_back(op.getOperation(), op.getDimIndex());
    });
    func.walk([&](GetTensorViewStrideOp op) {
      if (isRank2ViewLike(op.getTensorView().getType()))
        dimIndexOps.emplace_back(op.getOperation(), op.getDimIndex());
    });

    IRRewriter rewriter(func.getContext());
    for (MakeTensorViewOp op : makeViews) {
      if (failed(rewriteMakeTensorView(op, rewriter))) {
        signalPassFailure();
        return;
      }
    }
    for (auto [op, dimIndex] : dimIndexOps)
      rewriteTensorViewDimOperand(op, dimIndex, rewriter);
    canonicalizeValueTypes(func);
    for (PartitionViewOp op : partitionViews) {
      if (failed(rewritePartitionView(op, rewriter))) {
        signalPassFailure();
        return;
      }
    }

    // Post-canonicalization verification: ensure no rank-2 view types
    // survived. If any do, it means an op with rank-dependent operands
    // was not given a structural rewrite.
    if (failed(verifyNoRank2ViewSurvivors(func))) {
      signalPassFailure();
      return;
    }
  }
};

} // namespace

std::unique_ptr<Pass> mlir::pto::createPTOCanonicalizeIRPass() {
  return std::make_unique<PTOCanonicalizeIRPass>();
}
