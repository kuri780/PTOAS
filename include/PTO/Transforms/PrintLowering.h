// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#ifndef MLIR_DIALECT_PTO_TRANSFORMS_PRINTLOWERING_H
#define MLIR_DIALECT_PTO_TRANSFORMS_PRINTLOWERING_H

#include "PTO/Transforms/PrintEncoding.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

#include "llvm/ADT/StringMap.h"

#include <string>

namespace mlir {
namespace pto {

// ---------------------------------------------------------------------------
// Debug target capability hook.
//
// The VPTO print lowering is shared between the beta1 (A2/A3) and CANN 9.0
// (A5) emitters; architecture differences are expressed through this small
// descriptor instead of duplicated patterns.  Each field names a CCE
// intrinsic the target must provide; an empty name means the feature is
// unavailable on that target.
// ---------------------------------------------------------------------------
struct DebugTargetInfo {
  // Address of the fix stack, where the prologue stores DTData so that
  // ccelib-style code can recover it without extra parameters.
  std::string fixStackIntrinsic = "llvm.hivm.get.sycl.fix.stack.object";
  // Cache commit: makes log writes visible to the Host.
  std::string dcciIntrinsic = "llvm.hivm.DCCI";
  // Per-block index used to select this block's sub-region of LogWholeRegion.
  std::string blockIdxIntrinsic = "llvm.hivm.GET.BLOCK.IDX";
  // Virtual-address base of the UB (addrspace(6)) tile space, used by tprint
  // to translate a tile's element index into a UB load address.
  std::string sysVaBaseIntrinsic = "llvm.hivm.GET.SYS.VA.BASE";
  // When a required intrinsic is missing, no-op the print chain instead of
  // failing the whole lowering (keeps kernels usable on targets without
  // print).
  bool skipPrintIfUnavailable = true;
};

// ---------------------------------------------------------------------------
// Kernel ABI: records where each pto.entry function's hidden debug-context
// (DTData) argument lives, so the lowering looks it up by role instead of
// assuming "the last argument".  Future hidden arguments (workspace,
// profiling context, ...) extend this table rather than shifting positions.
// ---------------------------------------------------------------------------
class PTOKernelDebugABI {
public:
  void recordDebugContextArg(func::FuncOp func, unsigned index) {
    table[func.getSymName()] = index;
  }

  /// Returns the DTData argument of `func`, or null when this function has
  /// no debug context (i.e. it is not a pto.entry function).
  Value getDebugContextArgument(func::FuncOp func) const {
    auto it = table.find(func.getSymName());
    if (it == table.end())
      return {};
    if (it->second >= func.getNumArguments())
      return {};
    return func.getArgument(it->second);
  }

private:
  llvm::StringMap<unsigned> table;
};

// ---------------------------------------------------------------------------
// Shared state for the print/tprint lowering pipeline.
//
// Populated by lowerPrintToDebugRuntime (semantic layer, before dialect
// conversion) and addDTDataParamToEntryFunctions; consumed by the debug-op
// lowering patterns and injectPrintPrologue.  Owned by the emitter's
// LoweringState.
// ---------------------------------------------------------------------------
struct PrintLoweringState {
  // Whether the module uses pto.print / pto.tprint (set by the semantic
  // layer pre-scan).
  bool usesPrint = false;
  // Target capability/intrinsic name hook.
  DebugTargetInfo target;
  // Name of the fix-stack intrinsic declaration (created lazily).
  std::string fixStackFuncName;
  // Name of the DCCI intrinsic declaration (created lazily).
  std::string dcciFuncName;
  // Name of the get-block-idx intrinsic declaration (created lazily).
  std::string blockIdxFuncName;
  std::string sysVaBaseFuncName;
  // DTData argument positions per pto.entry function (role-based lookup).
  PTOKernelDebugABI kernelABI;
};

// ---------------------------------------------------------------------------
// Semantic layer: pto.print / pto.tprint -> pto.debug.* ops.
//
// Runs before dialect conversion.  Analyzes format strings (compile time),
// creates the LLVM globals holding the literal text / conversion specifier
// bytes, and emits the pto.debug.reserve / write_text / write_scalar /
// commit chain (plus the tile traversal loops for pto.tprint).  Also sets
// state.usesPrint.  The runtime-ABI concerns (DTData addressing, guards,
// overflow, DCCI) are deliberately absent here.
// ---------------------------------------------------------------------------
LogicalResult lowerPrintToDebugRuntime(ModuleOp module,
                                       PrintLoweringState &state);

// Add a hidden ptr addrspace(1) (DTData) parameter to every pto.entry
// function and declare the CCE intrinsics required by print lowering.
// Must run before dialect conversion so the type converter can handle the
// new function signature.  Returns failure() only on internal errors; if the
// module has no print ops this is a no-op.
LogicalResult addDTDataParamToEntryFunctions(ModuleOp module,
                                             PrintLoweringState &state);

// After dialect conversion, inject the prologue (fix-stack init +
// kernelWriteType = AiV) at the beginning of every entry function.
LogicalResult injectPrintPrologue(ModuleOp module, PrintLoweringState &state);

// After dialect conversion, inject the kernel-finish hook (the equivalent of
// ccelib's OnKernelFinish) before every return of every entry function:
// when LogWholeRegion != null, emit a final DCCI flush.  The authoritative
// cce::printf path flushes at kernel exit; without it, prints that happen
// early in the kernel may not be visible to the Host Close.
LogicalResult injectPrintEpilogue(ModuleOp module, PrintLoweringState &state);

// Register the pto.debug.* lowering patterns (runtime-ABI layer) plus the
// pto.alloc_tile placeholder pattern.  Shared by the beta1 (A2/A3) and
// CANN 9.0 (A5) emitters.
void populatePrintOpLoweringPatterns(TypeConverter &typeConverter,
                                     RewritePatternSet &patterns,
                                     PrintLoweringState &state);

} // namespace pto
} // namespace mlir

#endif // MLIR_DIALECT_PTO_TRANSFORMS_PRINTLOWERING_H
