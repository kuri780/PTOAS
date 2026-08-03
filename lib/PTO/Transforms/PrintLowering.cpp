// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Runtime-ABI layer of the VPTO print lowering: pto.debug.* ops -> inline
// DebugTunnel protocol writes.
//
// pto.print / pto.tprint are first rewritten to internal pto.debug.* ops by
// LowerPTOPrintToDebugRuntime.cpp (semantic layer); the patterns here are
// the only place that knows the DebugTunnel ABI: DTData addressing, null
// guards, the overflow check, byte-level node encoding and the DCCI flush.
// This component is shared by the beta1 (A2/A3) and CANN 9.0 (A5) emitters;
// target differences are expressed through DebugTargetInfo.
//
// The lowering pipeline (driven by the emitters' lowerVPTOOps):
//   1. lowerPrintToDebugRuntime        — semantic layer: pto.print/tprint
//      become pto.debug.* chains + string globals (BEFORE conversion).
//   2. addDTDataParamToEntryFunctions  — hidden ptr addrspace(1) DTData
//      parameter + CCE intrinsic declarations + kernel-ABI role recording.
//   3. applyPartialConversion          — the pto.debug.* patterns below.
//   4. injectPrintPrologue             — fix-stack init +
//      kernelWriteType = AiV at each entry function.
//   5. injectPrintEpilogue             — kernel-finish DCCI flush before each
//      entry-function return (ccelib's OnKernelFinish equivalent).

#include "PTO/Transforms/PrintLowering.h"

#include "PTO/IR/PTO.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Transforms/DialectConversion.h"

#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"

namespace mlir {
namespace pto {

// ---------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------

static Value getI64Constant(OpBuilder &builder, Location loc, uint64_t value) {
  return builder.create<arith::ConstantOp>(loc, builder.getI64IntegerAttr(value))
      .getResult();
}

// ---------------------------------------------------------------------------
// Kernel ABI: DTData parameter injection
// ---------------------------------------------------------------------------

// Add a hidden ptr addrspace(1) (DTData) parameter to every pto.entry function
// and declare the CCE intrinsics required by print lowering.  Must run before
// dialect conversion so the type converter can handle the new function
// signature.  The injected position is recorded in state.kernelABI so the
// lowering looks it up by role instead of assuming "the last argument".
LogicalResult addDTDataParamToEntryFunctions(ModuleOp module,
                                             PrintLoweringState &state) {
  if (!state.usesPrint)
    return success();

  MLIRContext *ctx = module.getContext();
  auto llvmPtr1Type = LLVM::LLVMPointerType::get(ctx, 1);
  auto llvmPtr0Type = LLVM::LLVMPointerType::get(ctx, 0);
  auto i64Type = IntegerType::get(ctx, 64);

  auto declareFunc = [&](StringRef name, LLVM::LLVMFunctionType fty) {
    if (module.lookupSymbol<LLVM::LLVMFuncOp>(name))
      return;
    OpBuilder b(module.getBodyRegion());
    b.setInsertionPointToStart(&module.getBodyRegion().front());
    auto func = b.create<LLVM::LLVMFuncOp>(module.getLoc(), name, fty);
    func.setPrivate();
  };

  // @llvm.hivm.get.sycl.fix.stack.object() -> ptr addrspace(0)
  if (!state.target.fixStackIntrinsic.empty()) {
    declareFunc(state.target.fixStackIntrinsic,
                LLVM::LLVMFunctionType::get(llvmPtr0Type, {}, false));
    state.fixStackFuncName = state.target.fixStackIntrinsic;
  }
  // @llvm.hivm.DCCI(ptr addrspace(1), i64) -> ()
  if (!state.target.dcciIntrinsic.empty()) {
    declareFunc(state.target.dcciIntrinsic,
                LLVM::LLVMFunctionType::get(LLVM::LLVMVoidType::get(ctx),
                                            {llvmPtr1Type, i64Type}, false));
    state.dcciFuncName = state.target.dcciIntrinsic;
  }
  // @llvm.hivm.GET.BLOCK.IDX() -> i64
  if (!state.target.blockIdxIntrinsic.empty()) {
    declareFunc(state.target.blockIdxIntrinsic,
                LLVM::LLVMFunctionType::get(i64Type, {}, false));
    state.blockIdxFuncName = state.target.blockIdxIntrinsic;
  }
  // @llvm.hivm.GET.SYS.VA.BASE() -> i64 (UB virtual-address base for tprint)
  if (!state.target.sysVaBaseIntrinsic.empty()) {
    declareFunc(state.target.sysVaBaseIntrinsic,
                LLVM::LLVMFunctionType::get(i64Type, {}, false));
    state.sysVaBaseFuncName = state.target.sysVaBaseIntrinsic;
  }

  // Add DTData parameter to every pto.entry function and record its position
  // in the kernel ABI table.
  SmallVector<func::FuncOp> entryFuncs;
  module.walk([&](func::FuncOp func) {
    if (pto::isPTOEntryFunction(func))
      entryFuncs.push_back(func);
  });

  for (func::FuncOp func : entryFuncs) {
    unsigned idx = func.getNumArguments();
    (void)func.insertArgument(idx, llvmPtr1Type, {}, func.getLoc());
    SmallVector<Type> newArgTypes(func.getArgumentTypes());
    auto newFuncType =
        FunctionType::get(ctx, newArgTypes, func.getResultTypes());
    (void)func.setFunctionType(newFuncType);
    state.kernelABI.recordDebugContextArg(func, idx);
  }

  return success();
}

// After dialect conversion, inject the prologue (fix-stack init +
// kernelWriteType = AiV) at the beginning of every entry function.
LogicalResult injectPrintPrologue(ModuleOp module, PrintLoweringState &state) {
  if (!state.usesPrint)
    return success();

  const DebugTunnelABIDescriptor &abi = getDebugTunnelABI();
  MLIRContext *ctx = module.getContext();
  auto ptr0Type = LLVM::LLVMPointerType::get(ctx, 0);
  auto ptr1Type = LLVM::LLVMPointerType::get(ctx, 1);
  auto i64Type = IntegerType::get(ctx, 64);
  auto i8Type = IntegerType::get(ctx, 8);
  auto i32Type = IntegerType::get(ctx, 32);
  auto i1Type = IntegerType::get(ctx, 1);

  SmallVector<func::FuncOp> entryFuncs;
  module.walk([&](func::FuncOp func) {
    if (pto::isPTOEntryFunction(func))
      entryFuncs.push_back(func);
  });

  for (func::FuncOp func : entryFuncs) {
    Value dtDataArg = state.kernelABI.getDebugContextArgument(func);
    if (!dtDataArg || dtDataArg.getType() != ptr1Type)
      continue;

    Region &body = func.getBody();
    if (body.empty())
      continue;
    Block &origEntry = body.front();
    Location loc = func.getLoc();

    Block *bodyBlock = origEntry.splitBlock(origEntry.begin());
    OpBuilder builder(&origEntry, origEntry.begin());

    auto fixStackFunc = module.lookupSymbol<LLVM::LLVMFuncOp>(
        state.fixStackFuncName);
    if (!fixStackFunc)
      return failure();
    auto fixCall = builder.create<LLVM::CallOp>(
        loc, ptr0Type, fixStackFunc.getSymName(), ValueRange{});
    Value fixStackPtr = fixCall.getResult();

    auto nullPtr = builder.create<LLVM::ZeroOp>(loc, ptr1Type);
    auto nullCheck = builder.create<LLVM::ICmpOp>(
        loc, i1Type, LLVM::ICmpPredicate::eq, dtDataArg, nullPtr);

    Block *initBlock = builder.createBlock(bodyBlock);
    Block *nullBlock = builder.createBlock(bodyBlock);
    builder.setInsertionPointToEnd(&origEntry);
    builder.create<LLVM::CondBrOp>(loc, nullCheck.getResult(), nullBlock,
                                   initBlock);

    // nullBlock: store 0 to fix stack.
    builder.setInsertionPointToStart(nullBlock);
    auto zeroI64 = builder.create<LLVM::ConstantOp>(loc, i64Type,
                                                    builder.getI64IntegerAttr(0));
    builder.create<LLVM::StoreOp>(loc, zeroI64.getResult(), fixStackPtr);
    builder.create<LLVM::BrOp>(loc, ValueRange{}, bodyBlock);

    // initBlock: store DTData to fix stack, set kernelWriteType = AiV.
    builder.setInsertionPointToStart(initBlock);
    auto dtI64 = builder.create<LLVM::PtrToIntOp>(loc, i64Type, dtDataArg);
    builder.create<LLVM::StoreOp>(loc, dtI64.getResult(), fixStackPtr);
    auto kwoff = builder.create<LLVM::ConstantOp>(
        loc, i64Type, builder.getI64IntegerAttr(abi.kernelWriteTypeOffset));
    auto kwPtr = builder.create<LLVM::GEPOp>(loc, ptr1Type, i8Type, dtDataArg,
                                             ValueRange{kwoff.getResult()});
    auto twoVal = builder.create<LLVM::ConstantOp>(
        loc, i32Type,
        builder.getI32IntegerAttr(abi.kernelWriteTypeAiV));
    builder.create<LLVM::StoreOp>(loc, twoVal.getResult(), kwPtr.getResult());
    builder.create<LLVM::BrOp>(loc, ValueRange{}, bodyBlock);
  }

  return success();
}

// After dialect conversion, inject the kernel-finish hook (the equivalent of
// ccelib's OnKernelFinish) before every return of every entry function:
// when LogWholeRegion != null, emit a final DCCI flush.  The authoritative
// cce::printf path flushes at kernel exit; without it, prints that happen
// early in the kernel may not be visible to the Host Close (observed: a
// single early print in a merged kernel produced no output on SIM).
LogicalResult injectPrintEpilogue(ModuleOp module, PrintLoweringState &state) {
  if (!state.usesPrint)
    return success();

  const DebugTunnelABIDescriptor &abi = getDebugTunnelABI();
  MLIRContext *ctx = module.getContext();
  auto ptr1Type = LLVM::LLVMPointerType::get(ctx, 1);
  auto i64Type = IntegerType::get(ctx, 64);
  auto i8Type = IntegerType::get(ctx, 8);
  auto i1Type = IntegerType::get(ctx, 1);

  auto dcciFunc = module.lookupSymbol<LLVM::LLVMFuncOp>(state.dcciFuncName);
  if (!dcciFunc)
    return success();

  SmallVector<func::FuncOp> entryFuncs;
  module.walk([&](func::FuncOp func) {
    if (pto::isPTOEntryFunction(func))
      entryFuncs.push_back(func);
  });

  for (func::FuncOp func : entryFuncs) {
    Value dtDataArg = state.kernelABI.getDebugContextArgument(func);
    if (!dtDataArg || dtDataArg.getType() != ptr1Type)
      continue;
    Region &body = func.getBody();
    if (body.empty())
      continue;

    SmallVector<func::ReturnOp> returns;
    body.walk([&](func::ReturnOp ret) { returns.push_back(ret); });
    for (func::ReturnOp ret : returns) {
      Block *retBlock = ret->getBlock();
      // Move the return (and any following ops) into a fresh block; the
      // guard is inserted at the end of the original block.
      Block *returnBlock = retBlock->splitBlock(Block::iterator(ret));
      Location loc = ret.getLoc();
      OpBuilder builder(retBlock, retBlock->end());

      // Guard: LogWholeRegion != null (matches PrintPayload::OnKernelFinish).
      auto lrAddr = builder.create<LLVM::GEPOp>(
          loc, ptr1Type, i8Type, dtDataArg,
          ValueRange{getI64Constant(builder, loc, abi.logWholeRegionOffset)});
      auto logRegion = builder.create<LLVM::LoadOp>(loc, ptr1Type, lrAddr);
      auto nullPtr = builder.create<LLVM::ZeroOp>(loc, ptr1Type);
      auto notNull = builder.create<LLVM::ICmpOp>(
          loc, i1Type, LLVM::ICmpPredicate::ne, logRegion.getResult(),
          nullPtr);

      Block *flushBlock = builder.createBlock(returnBlock);
      builder.setInsertionPointToEnd(retBlock);
      builder.create<LLVM::CondBrOp>(loc, notNull.getResult(), flushBlock,
                                     returnBlock);
      builder.setInsertionPointToStart(flushBlock);
      auto flushNull = builder.create<LLVM::ZeroOp>(loc, ptr1Type);
      auto flushOne = builder.create<LLVM::ConstantOp>(
          loc, i64Type, builder.getI64IntegerAttr(1));
      builder.create<LLVM::CallOp>(loc, TypeRange{}, dcciFunc.getSymName(),
                                   ValueRange{flushNull.getResult(),
                                              flushOne.getResult()});
      builder.create<LLVM::BrOp>(loc, ValueRange{}, returnBlock);
    }
  }

  return success();
}

// ---------------------------------------------------------------------------
// Lower pto.alloc_tile -> ZeroOp null pointer (tiles are UB-placeholders in
// VPTO).
// ---------------------------------------------------------------------------
class LowerAllocTileOpPattern final : public OpConversionPattern<pto::AllocTileOp> {
public:
  LowerAllocTileOpPattern(TypeConverter &typeConverter, MLIRContext *context,
                         PrintLoweringState &state)
      : OpConversionPattern(typeConverter, context) { (void)state; }

  LogicalResult
  matchAndRewrite(pto::AllocTileOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto ptr6Type = LLVM::LLVMPointerType::get(rewriter.getContext(), 6);
    auto nullPtr = rewriter.create<LLVM::ZeroOp>(op.getLoc(), ptr6Type);
    rewriter.replaceOp(op, nullPtr.getResult());
    return success();
  }
};

// ---------------------------------------------------------------------------
// pto.debug.reserve -> DTData guards + per-block addressing + overflow check.
//
// Yields the payload write pointer (LogBuffer + header + pLogSize), or a
// null pointer to no-op the whole chain: DTData == null, LogWholeRegion ==
// null, or the record would overflow the per-block buffer.  pLogSize is
// updated in both the overflow and the write path (the Host detects a
// truncated log by pLogSize exceeding the valid record bytes).
// ---------------------------------------------------------------------------
class LowerDebugReserveOpPattern final
    : public OpConversionPattern<pto::DebugReserveOp> {
public:
  LowerDebugReserveOpPattern(TypeConverter &typeConverter, MLIRContext *context,
                             PrintLoweringState &state)
      : OpConversionPattern<pto::DebugReserveOp>(typeConverter, context),
        state(state) {}

  LogicalResult
  matchAndRewrite(pto::DebugReserveOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    MLIRContext *ctx = rewriter.getContext();
    auto i64Type = rewriter.getI64Type();
    auto i8Type = rewriter.getI8Type();
    auto i1Type = rewriter.getI1Type();
    const DebugTunnelABIDescriptor &abi = getDebugTunnelABI();
    auto ptr1Type = LLVM::LLVMPointerType::get(ctx, 1);

    // DTData argument by role (PTOKernelDebugABI), not by position.
    auto func = op->getParentOfType<func::FuncOp>();
    if (!func)
      return op.emitError("internal: pto.debug.reserve outside a function");
    Value dtDataArg = state.kernelABI.getDebugContextArgument(func);
    if (!dtDataArg)
      return op.emitError("internal: DTData argument not recorded for entry "
                          "function '")
             << func.getSymName() << "'";

    ModuleOp moduleOp = op->getParentOfType<ModuleOp>();
    auto blockIdxFunc =
        moduleOp.lookupSymbol<LLVM::LLVMFuncOp>(state.blockIdxFuncName);
    if (!blockIdxFunc) {
      if (!state.target.skipPrintIfUnavailable)
        return op.emitError(
            "internal: block-idx intrinsic unavailable for print lowering");
      // Print is unavailable on this target: no-op the whole chain by
      // yielding a null write pointer.
      auto nullPtr = rewriter.create<LLVM::ZeroOp>(loc, ptr1Type);
      rewriter.replaceOp(op, nullPtr.getResult());
      return success();
    }

    int64_t recordSize = op.getRecordSize();
    auto nullPtr = rewriter.create<LLVM::ZeroOp>(loc, ptr1Type);

    // ---- Outer scf.if: DTData != null guard ----
    auto dtNotNull = rewriter.create<LLVM::ICmpOp>(
        loc, i1Type, LLVM::ICmpPredicate::ne, dtDataArg, nullPtr);
    auto outerIf = rewriter.create<scf::IfOp>(
        loc, TypeRange{ptr1Type}, dtNotNull.getResult(),
        /*withElseRegion=*/true);
    {
      // ---- Inner scf.if: logRegion != null guard (safe to deref now) ----
      rewriter.setInsertionPointToStart(&outerIf.getThenRegion().front());
      auto lrAddr = rewriter.create<LLVM::GEPOp>(
          loc, ptr1Type, i8Type, dtDataArg,
          ValueRange{getI64Constant(rewriter, loc, abi.logWholeRegionOffset)});
      auto logRegion = rewriter.create<LLVM::LoadOp>(loc, ptr1Type, lrAddr);
      auto lrNull = rewriter.create<LLVM::ZeroOp>(loc, ptr1Type);
      auto lrNotNull = rewriter.create<LLVM::ICmpOp>(
          loc, i1Type, LLVM::ICmpPredicate::ne, logRegion.getResult(), lrNull);

      auto innerIf = rewriter.create<scf::IfOp>(
          loc, TypeRange{ptr1Type}, lrNotNull.getResult(),
          /*withElseRegion=*/true);
      {
        // --- DTData and LogRegion are valid ---
        rewriter.setInsertionPointToStart(&innerIf.getThenRegion().front());

        // Load LogBufferSize, compute per-block stride and buffer base.
        auto lbsAddr = rewriter.create<LLVM::GEPOp>(
            loc, ptr1Type, i8Type, dtDataArg,
            ValueRange{
                getI64Constant(rewriter, loc, abi.logBufferSizeOffset)});
        auto logBufSize = rewriter.create<LLVM::LoadOp>(loc, i64Type, lbsAddr);

        auto stride64 = rewriter.create<LLVM::ConstantOp>(
            loc, i64Type,
            rewriter.getI64IntegerAttr(abi.logBufferHeaderBytes));
        auto stride = rewriter.create<LLVM::AddOp>(loc, i64Type,
                                                    logBufSize.getResult(),
                                                    stride64.getResult());
        auto blockIdxCall = rewriter.create<LLVM::CallOp>(
            loc, i64Type, blockIdxFunc.getSymName(), ValueRange{});
        auto blockOff = rewriter.create<LLVM::MulOp>(loc, i64Type,
                                                      blockIdxCall.getResult(),
                                                      stride.getResult());
        auto logBufBase = rewriter.create<LLVM::GEPOp>(
            loc, ptr1Type, i8Type, logRegion.getResult(),
            ValueRange{blockOff.getResult()});

        auto pLogSize =
            rewriter.create<LLVM::LoadOp>(loc, i64Type, logBufBase);

        // Overflow check; pLogSize is updated in both paths so the Host can
        // detect a truncated log.
        auto recSizeVal = rewriter.create<LLVM::ConstantOp>(
            loc, i64Type, rewriter.getI64IntegerAttr(recordSize));
        auto newPls = rewriter.create<LLVM::AddOp>(loc, i64Type,
                                                    pLogSize.getResult(),
                                                    recSizeVal.getResult());
        rewriter.create<LLVM::StoreOp>(loc, newPls.getResult(), logBufBase);
        auto overflow = rewriter.create<LLVM::ICmpOp>(
            loc, i1Type, LLVM::ICmpPredicate::ugt, newPls.getResult(),
            logBufSize.getResult());

        auto overflowIf = rewriter.create<scf::IfOp>(
            loc, TypeRange{ptr1Type}, overflow.getResult(),
            /*withElseRegion=*/true);
        {
          // Overflow: no payload written, yield null.
          rewriter.setInsertionPointToStart(&overflowIf.getThenRegion().front());
          rewriter.create<scf::YieldOp>(loc, ValueRange{nullPtr.getResult()});
        }
        {
          // No overflow: yield the payload write pointer.
          rewriter.setInsertionPointToStart(&overflowIf.getElseRegion().front());
          auto headerOff = rewriter.create<LLVM::ConstantOp>(
              loc, i64Type,
              rewriter.getI64IntegerAttr(abi.logBufferHeaderBytes));
          auto writeOff = rewriter.create<LLVM::AddOp>(
              loc, i64Type, headerOff.getResult(), pLogSize.getResult());
          auto writePtr = rewriter.create<LLVM::GEPOp>(
              loc, ptr1Type, i8Type, logBufBase.getResult(),
              ValueRange{writeOff.getResult()});
          rewriter.create<scf::YieldOp>(loc,
                                        ValueRange{writePtr.getResult()});
        }
        rewriter.setInsertionPointToEnd(&innerIf.getThenRegion().front());
        rewriter.create<scf::YieldOp>(loc, ValueRange{overflowIf.getResult(0)});
      }
      {
        rewriter.setInsertionPointToEnd(&innerIf.getElseRegion().front());
        rewriter.create<scf::YieldOp>(loc, ValueRange{nullPtr.getResult()});
      }
      rewriter.setInsertionPointToEnd(&outerIf.getThenRegion().front());
      rewriter.create<scf::YieldOp>(loc, ValueRange{innerIf.getResult(0)});
    }
    {
      rewriter.setInsertionPointToEnd(&outerIf.getElseRegion().front());
      rewriter.create<scf::YieldOp>(loc, ValueRange{nullPtr.getResult()});
    }

    rewriter.replaceOp(op, outerIf.getResult(0));
    return success();
  }

private:
  PrintLoweringState &state;
};

// ---------------------------------------------------------------------------
// pto.debug.write_text -> literal text node.
//
// Writes [marker=1][len16][text bytes][NUL] at the write pointer and yields
// the next write pointer; a null input pointer propagates unchanged.
// ---------------------------------------------------------------------------
class LowerDebugWriteTextOpPattern final
    : public OpConversionPattern<pto::DebugWriteTextOp> {
public:
  LowerDebugWriteTextOpPattern(TypeConverter &typeConverter,
                               MLIRContext *context, PrintLoweringState &state)
      : OpConversionPattern<pto::DebugWriteTextOp>(typeConverter, context),
        state(state) {}

  LogicalResult
  matchAndRewrite(pto::DebugWriteTextOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    MLIRContext *ctx = rewriter.getContext();
    auto ptr0Type = LLVM::LLVMPointerType::get(ctx, 0);
    auto ptr1Type = LLVM::LLVMPointerType::get(ctx, 1);
    auto i64Type = rewriter.getI64Type();
    auto i8Type = rewriter.getI8Type();
    auto i1Type = rewriter.getI1Type();

    Value ptr = adaptor.getPtr();
    int64_t bytes = op.getBytes();
    int64_t recordSize = debugtunnel::textRecordSize(bytes);

    auto nullPtr = rewriter.create<LLVM::ZeroOp>(loc, ptr1Type);
    auto notNull = rewriter.create<LLVM::ICmpOp>(
        loc, i1Type, LLVM::ICmpPredicate::ne, ptr, nullPtr);
    auto ifOp = rewriter.create<scf::IfOp>(loc, TypeRange{ptr1Type},
                                           notNull.getResult(),
                                           /*withElseRegion=*/true);
    {
      rewriter.setInsertionPointToStart(&ifOp.getThenRegion().front());
      auto fmtGlobal =
          rewriter.create<LLVM::AddressOfOp>(loc, ptr0Type, op.getGlobal());
      auto storeI8 = [&](int64_t offset, Value val) {
        auto off = rewriter.create<LLVM::ConstantOp>(
            loc, i64Type, rewriter.getI64IntegerAttr(offset));
        auto dst = rewriter.create<LLVM::GEPOp>(loc, ptr1Type, i8Type, ptr,
                                                ValueRange{off.getResult()});
        rewriter.create<LLVM::StoreOp>(loc, val, dst);
      };
      auto marker = rewriter.create<LLVM::ConstantOp>(
          loc, i8Type, rewriter.getI8IntegerAttr(debugtunnel::kTextMarker));
      storeI8(0, marker);
      auto lenLow = rewriter.create<LLVM::ConstantOp>(
          loc, i8Type, rewriter.getI8IntegerAttr(bytes & 0xff));
      auto lenHigh = rewriter.create<LLVM::ConstantOp>(
          loc, i8Type, rewriter.getI8IntegerAttr(bytes >> 8));
      storeI8(1, lenLow);
      storeI8(2, lenHigh);
      for (int64_t i = 0; i + 1 < bytes; ++i) {
        auto source = rewriter.create<LLVM::ConstantOp>(
            loc, i64Type, rewriter.getI64IntegerAttr(i));
        auto charPtr = rewriter.create<LLVM::GEPOp>(
            loc, ptr0Type, i8Type, fmtGlobal.getResult(),
            ValueRange{source.getResult()});
        auto ch = rewriter.create<LLVM::LoadOp>(loc, i8Type, charPtr);
        storeI8(3 + i, ch);
      }
      auto nul = rewriter.create<LLVM::ConstantOp>(loc, i8Type,
                                                   rewriter.getI8IntegerAttr(0));
      storeI8(2 + bytes, nul);
      auto nextOff = rewriter.create<LLVM::ConstantOp>(
          loc, i64Type, rewriter.getI64IntegerAttr(recordSize));
      auto nextPtr = rewriter.create<LLVM::GEPOp>(
          loc, ptr1Type, i8Type, ptr, ValueRange{nextOff.getResult()});
      rewriter.create<scf::YieldOp>(loc, ValueRange{nextPtr.getResult()});
    }
    {
      rewriter.setInsertionPointToStart(&ifOp.getElseRegion().front());
      rewriter.create<scf::YieldOp>(loc, ValueRange{ptr});
    }
    rewriter.replaceOp(op, ifOp.getResult(0));
    return success();
  }

private:
  PrintLoweringState &state;
};

// ---------------------------------------------------------------------------
// pto.debug.write_scalar -> value node.
//
// Writes [marker][raw value LE][len16][spec][NUL] at the write pointer and
// yields the next write pointer; a null input pointer propagates unchanged.
// `kind` selects the protocol marker and the sign/zero extension of narrow
// integers.
// ---------------------------------------------------------------------------
class LowerDebugWriteScalarOpPattern final
    : public OpConversionPattern<pto::DebugWriteScalarOp> {
public:
  LowerDebugWriteScalarOpPattern(TypeConverter &typeConverter,
                                 MLIRContext *context, PrintLoweringState &state)
      : OpConversionPattern<pto::DebugWriteScalarOp>(typeConverter, context),
        state(state) {}

  LogicalResult
  matchAndRewrite(pto::DebugWriteScalarOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    MLIRContext *ctx = rewriter.getContext();
    auto ptr0Type = LLVM::LLVMPointerType::get(ctx, 0);
    auto ptr1Type = LLVM::LLVMPointerType::get(ctx, 1);
    auto i64Type = rewriter.getI64Type();
    auto i32Type = rewriter.getI32Type();
    auto i8Type = rewriter.getI8Type();
    auto i1Type = rewriter.getI1Type();

    Value ptr = adaptor.getPtr();
    Value scalar = adaptor.getValue();
    int64_t bytes = op.getBytes(); // spec chars + NUL
    PrintConversionKind kind = PrintConversionKind::SignedInt;
    if (op.getKind() == "float")
      kind = PrintConversionKind::Float;
    else if (op.getKind() == "unsigned")
      kind = PrintConversionKind::UnsignedInt;

    auto enc = encodePrintScalar(rewriter, loc, scalar.getType(), scalar, kind);
    if (failed(enc))
      return failure();
    int64_t recordSize = debugtunnel::scalarRecordSize(enc->byteWidth, bytes);

    auto nullPtr = rewriter.create<LLVM::ZeroOp>(loc, ptr1Type);
    auto notNull = rewriter.create<LLVM::ICmpOp>(
        loc, i1Type, LLVM::ICmpPredicate::ne, ptr, nullPtr);
    auto ifOp = rewriter.create<scf::IfOp>(loc, TypeRange{ptr1Type},
                                           notNull.getResult(),
                                           /*withElseRegion=*/true);
    {
      rewriter.setInsertionPointToStart(&ifOp.getThenRegion().front());
      auto fmtGlobal =
          rewriter.create<LLVM::AddressOfOp>(loc, ptr0Type, op.getGlobal());
      auto storeI8 = [&](int64_t offset, Value val) {
        auto off = rewriter.create<LLVM::ConstantOp>(
            loc, i64Type, rewriter.getI64IntegerAttr(offset));
        auto dst = rewriter.create<LLVM::GEPOp>(loc, ptr1Type, i8Type, ptr,
                                                ValueRange{off.getResult()});
        rewriter.create<LLVM::StoreOp>(loc, val, dst);
      };
      auto marker = rewriter.create<LLVM::ConstantOp>(
          loc, i8Type, rewriter.getI8IntegerAttr(enc->marker));
      storeI8(0, marker);
      Type shiftType = (enc->byteWidth == 4) ? i32Type : i64Type;
      for (unsigned i = 0; i < enc->byteWidth; ++i) {
        auto shift = rewriter.create<LLVM::ConstantOp>(
            loc, shiftType, rewriter.getIntegerAttr(shiftType, i * 8));
        auto shifted = rewriter.create<LLVM::LShrOp>(loc, shiftType,
                                                      enc->bits, shift);
        auto byte = rewriter.create<LLVM::TruncOp>(loc, i8Type, shifted);
        storeI8(1 + i, byte);
      }
      auto lenLow = rewriter.create<LLVM::ConstantOp>(
          loc, i8Type, rewriter.getI8IntegerAttr(bytes & 0xff));
      auto lenHigh = rewriter.create<LLVM::ConstantOp>(
          loc, i8Type, rewriter.getI8IntegerAttr(bytes >> 8));
      storeI8(1 + enc->byteWidth, lenLow);
      storeI8(2 + enc->byteWidth, lenHigh);
      int64_t specOff = 3 + enc->byteWidth;
      for (int64_t i = 0; i + 1 < bytes; ++i) {
        auto source = rewriter.create<LLVM::ConstantOp>(
            loc, i64Type, rewriter.getI64IntegerAttr(i));
        auto charPtr = rewriter.create<LLVM::GEPOp>(
            loc, ptr0Type, i8Type, fmtGlobal.getResult(),
            ValueRange{source.getResult()});
        auto ch = rewriter.create<LLVM::LoadOp>(loc, i8Type, charPtr);
        storeI8(specOff + i, ch);
      }
      auto nul = rewriter.create<LLVM::ConstantOp>(loc, i8Type,
                                                   rewriter.getI8IntegerAttr(0));
      storeI8(specOff + bytes - 1, nul);
      auto nextOff = rewriter.create<LLVM::ConstantOp>(
          loc, i64Type, rewriter.getI64IntegerAttr(recordSize));
      auto nextPtr = rewriter.create<LLVM::GEPOp>(
          loc, ptr1Type, i8Type, ptr, ValueRange{nextOff.getResult()});
      rewriter.create<scf::YieldOp>(loc, ValueRange{nextPtr.getResult()});
    }
    {
      rewriter.setInsertionPointToStart(&ifOp.getElseRegion().front());
      rewriter.create<scf::YieldOp>(loc, ValueRange{ptr});
    }
    rewriter.replaceOp(op, ifOp.getResult(0));
    return success();
  }

private:
  PrintLoweringState &state;
};

// ---------------------------------------------------------------------------
// pto.debug.commit -> END marker + DCCI flush (no-op on null pointer).
// ---------------------------------------------------------------------------
class LowerDebugCommitOpPattern final
    : public OpConversionPattern<pto::DebugCommitOp> {
public:
  LowerDebugCommitOpPattern(TypeConverter &typeConverter, MLIRContext *context,
                            PrintLoweringState &state)
      : OpConversionPattern<pto::DebugCommitOp>(typeConverter, context),
        state(state) {}

  LogicalResult
  matchAndRewrite(pto::DebugCommitOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    MLIRContext *ctx = rewriter.getContext();
    auto ptr1Type = LLVM::LLVMPointerType::get(ctx, 1);
    auto i64Type = rewriter.getI64Type();
    auto i8Type = rewriter.getI8Type();
    auto i1Type = rewriter.getI1Type();

    Value ptr = adaptor.getPtr();
    auto nullPtr = rewriter.create<LLVM::ZeroOp>(loc, ptr1Type);
    auto notNull = rewriter.create<LLVM::ICmpOp>(
        loc, i1Type, LLVM::ICmpPredicate::ne, ptr, nullPtr);
    auto ifOp = rewriter.create<scf::IfOp>(loc, notNull.getResult(),
                                           /*withElseRegion=*/false);
    rewriter.setInsertionPointToStart(&ifOp.getThenRegion().front());

    auto endMarker = rewriter.create<LLVM::ConstantOp>(
        loc, i8Type, rewriter.getI8IntegerAttr(debugtunnel::kEndMarker));
    auto off = rewriter.create<LLVM::ConstantOp>(loc, i64Type,
                                                 rewriter.getI64IntegerAttr(0));
    auto dst = rewriter.create<LLVM::GEPOp>(loc, ptr1Type, i8Type, ptr,
                                            ValueRange{off.getResult()});
    rewriter.create<LLVM::StoreOp>(loc, endMarker.getResult(), dst);

    // DCCI flush: makes the log writes visible to the Host.
    ModuleOp moduleOp = op->getParentOfType<ModuleOp>();
    auto dcciFunc =
        moduleOp.lookupSymbol<LLVM::LLVMFuncOp>(state.dcciFuncName);
    if (dcciFunc) {
      auto flushNull = rewriter.create<LLVM::ZeroOp>(loc, ptr1Type);
      auto flushOne = rewriter.create<LLVM::ConstantOp>(
          loc, i64Type, rewriter.getI64IntegerAttr(1));
      rewriter.create<LLVM::CallOp>(loc, TypeRange{}, dcciFunc.getSymName(),
                                    ValueRange{flushNull.getResult(),
                                               flushOne.getResult()});
    }
    // yield is already created by the builder
    rewriter.eraseOp(op);
    return success();
  }

private:
  PrintLoweringState &state;
};

// ---------------------------------------------------------------------------
// pto.debug.get_tile_base -> UB virtual-address base as an element offset.
// ---------------------------------------------------------------------------
class LowerDebugGetTileBaseOpPattern final
    : public OpConversionPattern<pto::DebugGetTileBaseOp> {
public:
  LowerDebugGetTileBaseOpPattern(TypeConverter &typeConverter,
                                 MLIRContext *context, PrintLoweringState &state)
      : OpConversionPattern<pto::DebugGetTileBaseOp>(typeConverter, context),
        state(state) {}

  LogicalResult
  matchAndRewrite(pto::DebugGetTileBaseOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto i64Type = rewriter.getI64Type();

    ModuleOp moduleOp = op->getParentOfType<ModuleOp>();
    auto sysVaBaseFunc =
        moduleOp.lookupSymbol<LLVM::LLVMFuncOp>(state.sysVaBaseFuncName);
    Value result;
    if (sysVaBaseFunc) {
      auto sysva = rewriter.create<LLVM::CallOp>(
          loc, i64Type, sysVaBaseFunc.getSymName(), ValueRange{});
      auto ubOff = rewriter.create<LLVM::ConstantOp>(
          loc, i64Type, rewriter.getI64IntegerAttr(0x80000));
      auto baseAddr =
          rewriter.create<LLVM::AddOp>(loc, i64Type, sysva.getResult(), ubOff);
      auto elemSizeVal = rewriter.create<LLVM::ConstantOp>(
          loc, i64Type, rewriter.getI64IntegerAttr(op.getElemBytes()));
      result = rewriter.create<LLVM::UDivOp>(loc, i64Type, baseAddr.getResult(),
                                             elemSizeVal.getResult());
    } else {
      result = getI64Constant(rewriter, loc, 0);
    }
    rewriter.replaceOp(op, result);
    return success();
  }

private:
  PrintLoweringState &state;
};

// ---------------------------------------------------------------------------
// Pattern registration
// ---------------------------------------------------------------------------

void populatePrintOpLoweringPatterns(TypeConverter &typeConverter,
                                     RewritePatternSet &patterns,
                                     PrintLoweringState &state) {
  patterns.add<LowerAllocTileOpPattern, LowerDebugReserveOpPattern,
               LowerDebugWriteTextOpPattern, LowerDebugWriteScalarOpPattern,
               LowerDebugCommitOpPattern, LowerDebugGetTileBaseOpPattern>(
      typeConverter, patterns.getContext(), state);
}

} // namespace pto
} // namespace mlir
