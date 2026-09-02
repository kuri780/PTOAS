// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

//===- VPTOBridgeLowering.cpp - generic C++ interface bridge lowering ----===//
//===----------------------------------------------------------------------===//
//
// Generic bridge lowering pass of the VPTO C++ interface bridge. It knows
// nothing about individual PTO-ISA interface families: it validates each
// pto.bridge_call against the bridge whitelist and mechanically lowers it
// into a call to the wrapper entry, materializing the wrapper declaration
// at module level. Stateful object creation is represented explicitly by bridge_object_create and
// resolved through the registry size-entry relation.
//
// The whitelist is also the routing check of last resort: any op still
// present in the IR that the whitelist routes to a wrapper entry was
// missed by its family pass, and is rejected with a diagnostic instead of
// silently flowing into the regular LLVM emission path.
//
//===----------------------------------------------------------------------===//

#include "PTO/IR/PTO.h"
#include "PTO/Transforms/VPTOBridgeRegistry.h"
#include "PTO/Transforms/VPTOBridgeWhitelist.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"

namespace mlir {
namespace pto {

#define GEN_PASS_DECL_VPTOBRIDGELOWERING
#define GEN_PASS_DEF_VPTOBRIDGELOWERING
#include "PTO/Transforms/Passes.h.inc"

namespace {

/// Converts the carrier types a bridge op may hold. These rules mirror the
/// PipeType/PtrType entries of the VPTO type converter
/// (VPTOCANN900LLVMEmitter.cpp convertVPTOType); the bridge lowering runs
/// before that converter and must agree with it so values flow into the
/// remaining PTO ops without extra casts.
class BridgeTypeConverter final : public TypeConverter {
public:
  explicit BridgeTypeConverter(MLIRContext *context) {
    addConversion([](Type type) -> Type {
      if (isa<pto::PipeType>(type)) {
        return LLVM::LLVMPointerType::get(type.getContext());
      }
      if (auto ptrType = dyn_cast<pto::PtrType>(type)) {
        return LLVM::LLVMPointerType::get(
            type.getContext(),
            static_cast<unsigned>(ptrType.getMemorySpace().getAddressSpace()));
      }
      return type;
    });
    addSourceMaterialization(materializeBridgeCast);
    addTargetMaterialization(materializeBridgeCast);
  }

private:
  static std::optional<Value> materializeBridgeCast(OpBuilder &builder,
                                                    Type resultType,
                                                    ValueRange inputs,
                                                    Location loc) {
    if (inputs.size() != 1) {
      return std::nullopt;
    }
    return builder
        .create<UnrealizedConversionCastOp>(loc, TypeRange{resultType}, inputs)
        .getResult(0);
  }
};

struct BridgeLoweringState {
  llvm::StringSet<> declaredEntries;
};

/// Creates the module-level private declaration of a wrapper entry the
/// first time it is called.
static void ensureWrapperDecl(ModuleOp module, BridgeLoweringState &state,
                              PatternRewriter &rewriter, StringRef callee,
                              TypeRange argTypes, TypeRange resultTypes) {
  if (state.declaredEntries.contains(callee)) {
    return;
  }
  state.declaredEntries.insert(callee);
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(&module.getBodyRegion().front());
  auto decl = rewriter.create<func::FuncOp>(
      module.getLoc(), callee,
      FunctionType::get(module.getContext(), argTypes, resultTypes));
  decl.setPrivate();
}

static LogicalResult validateRegistryAbi(Operation *op,
                                         const BridgeFunctionDesc &desc,
                                         ValueRange callArgs) {
  if (callArgs.size() != desc.arguments.size()) {
    op->emitError() << "VPTO bridge registry entry '" << desc.symbol
                    << "' expects " << desc.arguments.size()
                    << " argument(s), got " << callArgs.size();
    return failure();
  }
  for (auto [index, arg] : llvm::enumerate(callArgs)) {
    if (index >= desc.arguments.size()) {
      return failure();
    }
    BridgeValueKind kind = desc.arguments[index];
    if (!bridgeValueKindMatches(kind, arg.getType())) {
      op->emitError() << "VPTO bridge registry entry '" << desc.symbol
                      << "' argument #" << index << " has type "
                      << arg.getType();
      return failure();
    }
  }
  for (auto [index, resultType] : llvm::enumerate(desc.results)) {
    if (index >= op->getNumResults() ||
        !bridgeValueKindMatches(resultType, op->getResult(index).getType())) {
      op->emitError() << "VPTO bridge registry entry '" << desc.symbol
                      << "' has an incompatible result #" << index;
      return failure();
    }
  }
  return success();
}

class LowerBridgeCallPattern final : public OpConversionPattern<BridgeCallOp> {
public:
  LowerBridgeCallPattern(TypeConverter &converter, MLIRContext *context,
                         BridgeLoweringState &state)
      : OpConversionPattern<BridgeCallOp>(converter, context), state(state) {}

  LogicalResult
  matchAndRewrite(BridgeCallOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    StringRef callee = op.getCalleeAttr().getValue();
    auto entryId = op.getEntryId();
    if (!entryId) {
      return op.emitError("VPTO bridge call requires a logical entry_id");
    }
    const BridgeFunctionDesc *registryEntry =
        lookupBridgeEntryForName(*entryId);
    if (!registryEntry) {
      return op.emitError()
             << "VPTO bridge call uses unknown entry_id '" << *entryId << "'";
    }
    if (callee.empty()) {
      return op.emitError(
          "VPTO bridge call must be resolved to a concrete instance symbol");
    }
    ModuleOp module = op->getParentOfType<ModuleOp>();
    SmallVector<Value> callArgs(adaptor.getArgs().begin(),
                                adaptor.getArgs().end());

    if (failed(validateRegistryAbi(op, *registryEntry, callArgs))) {
      return failure();
    }

    SmallVector<Type> resultTypes;
    for (Type resultType : op.getResultTypes()) {
      Type converted = getTypeConverter()->convertType(resultType);
      if (!converted) {
        return op.emitError() << "VPTO bridge call result type " << resultType
                              << " has no bridge conversion";
      }
      resultTypes.push_back(converted);
    }

    func::CallOp call = rewriter.create<func::CallOp>(
        loc, callee, TypeRange(resultTypes), ValueRange(callArgs));
    ensureWrapperDecl(module, state, rewriter, callee,
                      llvm::map_to_vector<4>(
                          callArgs, [](Value arg) { return arg.getType(); }),
                      TypeRange(resultTypes));

    if (call.getNumResults() == 0) {
      rewriter.eraseOp(op);
      return success();
    }
    rewriter.replaceOp(op, call.getResults());
    return success();
  }

private:
  BridgeLoweringState &state;
};

class LowerBridgeObjectCreatePattern final
    : public OpConversionPattern<BridgeObjectCreateOp> {
public:
  LowerBridgeObjectCreatePattern(TypeConverter &converter, MLIRContext *context,
                                 BridgeLoweringState &state)
      : OpConversionPattern<BridgeObjectCreateOp>(converter, context),
        state(state) {}

  LogicalResult
  matchAndRewrite(BridgeObjectCreateOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    const BridgeFunctionDesc *desc =
        lookupBridgeEntryForName(op.getEntryAttr().getValue());
    if (!desc || !desc->stateful || !desc->storageSizeEntry) {
      return op.emitError(
          "bridge object create requires a registered stateful entry");
    }
    const BridgeFunctionDesc *sizeDesc =
        lookupBridgeEntry(*desc->storageSizeEntry);
    if (!op.getInstanceId()) {
      return op.emitError("bridge object create must be instance-resolved");
    }
    int64_t instanceId = *op.getInstanceId();
    std::string initSymbol = getBridgeInstanceSymbol(desc->id, instanceId);
    std::string sizeSymbol = getBridgeInstanceSymbol(sizeDesc->id, instanceId);
    ModuleOp module = op->getParentOfType<ModuleOp>();
    Value size = rewriter
                     .create<func::CallOp>(op.getLoc(), sizeSymbol,
                                           rewriter.getI64Type(), ValueRange{})
                     .getResult(0);
    ensureWrapperDecl(module, state, rewriter, sizeSymbol, {},
                      {rewriter.getI64Type()});
    Value storage = rewriter.create<LLVM::AllocaOp>(
        op.getLoc(), LLVM::LLVMPointerType::get(rewriter.getContext()),
        rewriter.getI8Type(), size, 8);
    SmallVector<Value> args{storage};
    args.append(adaptor.getArgs().begin(), adaptor.getArgs().end());
    rewriter.create<func::CallOp>(op.getLoc(), initSymbol, TypeRange{}, args);
    ensureWrapperDecl(module, state, rewriter, initSymbol,
                      llvm::map_to_vector<4>(
                          args, [](Value value) { return value.getType(); }),
                      {});
    rewriter.replaceOp(op, storage);
    return success();
  }

private:
  BridgeLoweringState &state;
};

class LowerBridgeIntToPtrPattern final
    : public OpConversionPattern<BridgeIntToPtrOp> {
public:
  LowerBridgeIntToPtrPattern(TypeConverter &converter, MLIRContext *context)
      : OpConversionPattern<BridgeIntToPtrOp>(converter, context) {}

  LogicalResult
  matchAndRewrite(BridgeIntToPtrOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type convertedResult =
        getTypeConverter()->convertType(op.getResult().getType());
    if (!convertedResult || !isa<LLVM::LLVMPointerType>(convertedResult)) {
      return op.emitError()
             << "VPTO bridge inttoptr requires a result type that converts "
                "to an LLVM pointer, got "
             << op.getResult().getType();
    }
    rewriter.replaceOpWithNewOp<LLVM::IntToPtrOp>(op, convertedResult,
                                                  adaptor.getAddr());
    return success();
  }
};

struct VPTOBridgeLoweringPass final
    : public impl::VPTOBridgeLoweringBase<VPTOBridgeLoweringPass> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(VPTOBridgeLoweringPass)

  void runOnOperation() override {
    ModuleOp module = getOperation();
    bool hasBridgeOps = false;
    module.walk([&](Operation *op) {
      if (isa<BridgeCallOp, BridgeObjectCreateOp, BridgeIntToPtrOp>(op)) {
        hasBridgeOps = true;
      }
    });

    // The whitelist always resolves through the formal chain (pass option,
    // PTOAS_VPTO_BRIDGE_WHITELIST, built-in default), so this pass always
    // validates; `whitelistName` is only for diagnostics.
    std::string whitelistName;
    FailureOr<BridgeRoutePolicy> policyOr =
        loadBridgeRoutePolicy(whitelistPath, llvm::errs(), &whitelistName);
    if (failed(policyOr)) {
      signalPassFailure();
      return;
    }
    bool leftoversFound = false;
    module.walk([&](Operation *op) {
      StringRef opName = op->getName().getStringRef();
      const BridgeFunctionDesc *entry = lookupBridgeEntryForOp(opName);
      bool routed =
          entry && (entry->family == BridgeFamily::Pipe
                        ? policyOr->families.pipe.enabled
                        : llvm::is_contained(policyOr->families.cube.enabledOps,
                                             opName));
      if (!routed) {
        return;
      }
      op->emitError() << "VPTO bridge: routed operation '" << opName
                      << "' was not lowered by its typed family pattern; "
                         "policy source: '"
                      << whitelistName << "'";
      leftoversFound = true;
    });
    if (leftoversFound) {
      signalPassFailure();
      return;
    }

    if (!hasBridgeOps) {
      return;
    }

    BridgeTypeConverter converter(&getContext());
    ConversionTarget target(getContext());
    target.addIllegalOp<BridgeCallOp, BridgeObjectCreateOp, BridgeIntToPtrOp>();
    // Everything the patterns create (func.call, llvm.alloca, private
    // declarations) must be legal on the target, otherwise the conversion
    // driver rejects the generated operations and rolls the pattern back.
    target.markUnknownOpDynamicallyLegal([](Operation *op) { return true; });

    RewritePatternSet patterns(&getContext());
    BridgeLoweringState state;
    patterns.add<LowerBridgeCallPattern>(converter, &getContext(), state);
    patterns.add<LowerBridgeObjectCreatePattern>(converter, &getContext(),
                                                 state);
    patterns.add<LowerBridgeIntToPtrPattern>(converter, &getContext());
    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<Pass> createVPTOBridgeLoweringPass() {
  return std::make_unique<VPTOBridgeLoweringPass>();
}

} // namespace pto
} // namespace mlir
