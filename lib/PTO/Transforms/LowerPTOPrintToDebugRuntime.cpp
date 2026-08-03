// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Semantic layer of the VPTO print lowering: pto.print / pto.tprint ->
// internal pto.debug.* ops.
//
// This step is deliberately free of runtime-ABI concerns (DTData addressing,
// null guards, overflow, DCCI).  It:
//   * analyzes the format string at compile time (analyzePrintFormat),
//   * creates one LLVM global per literal text / conversion specifier,
//   * emits the pto.debug.reserve / write_text / write_scalar / commit
//     write-pointer chain, and
//   * expands pto.tprint into literal header/shape records plus row/col
//     traversal loops over the tile.
// The pto.debug.* ops are then lowered to inline DebugTunnel protocol writes
// by the runtime-ABI patterns in PTO/Transforms/PrintLowering.cpp.

#include "PTO/Transforms/PrintLowering.h"

#include "PTO/IR/PTO.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"

#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"

namespace mlir {
namespace pto {

namespace {

// Deduplicated LLVM global (i8 array + NUL) per literal text.  Created
// before dialect conversion so the runtime-ABI patterns can emit
// LLVM::AddressOfOp references against existing symbols.
class StringGlobalTable {
public:
  explicit StringGlobalTable(ModuleOp module)
      : module(module), builder(module.getBodyRegion()) {
    builder.setInsertionPointToStart(&module.getBodyRegion().front());
  }

  /// Returns the global symbol name for `text`, creating the global on first
  /// use.
  StringRef getOrCreate(StringRef text) {
    auto it = seen.find(text);
    if (it != seen.end())
      return it->second;
    std::string name = "_ptoas_printf_fmt_" + std::to_string(seen.size());
    auto [it2, inserted] = seen.try_emplace(text, name);
    (void)inserted;

    auto i8Type = IntegerType::get(module.getContext(), 8);
    SmallVector<Attribute> elements;
    for (char c : text)
      elements.push_back(IntegerAttr::get(i8Type, c));
    elements.push_back(IntegerAttr::get(i8Type, 0)); // null terminator
    auto arrayType = LLVM::LLVMArrayType::get(i8Type, elements.size());
    builder.create<LLVM::GlobalOp>(
        module.getLoc(), arrayType,
        /*isConstant=*/true, LLVM::Linkage::Private, name,
        ArrayAttr::get(module.getContext(), elements));
    return it2->second;
  }

private:
  ModuleOp module;
  llvm::StringMap<std::string> seen;
  OpBuilder builder;
};

// "float" / "signed" / "unsigned" — carried on pto.debug.write_scalar and
// interpreted by the runtime-ABI layer.
const char *kindToString(PrintConversionKind kind) {
  switch (kind) {
  case PrintConversionKind::Float:
    return "float";
  case PrintConversionKind::UnsignedInt:
    return "unsigned";
  case PrintConversionKind::SignedInt:
    return "signed";
  }
  llvm_unreachable("unknown print conversion kind");
}

// ---------------------------------------------------------------------------
// pto.print -> debug chain
// ---------------------------------------------------------------------------
static void rewritePrintOp(pto::PrintOp op, StringGlobalTable &globals) {
  Location loc = op.getLoc();
  OpBuilder builder(op);
  MLIRContext *ctx = op.getContext();
  auto ptr1Type = LLVM::LLVMPointerType::get(ctx, 1);

  StringRef fmt = op.getFormat();
  if (fmt.empty())
    fmt = "%f";
  auto info = analyzePrintFormat(fmt);
  if (failed(info)) {
    // The verifier already rejected invalid formats; keep a diagnostic for
    // safety instead of crashing.
    op.emitError("internal: failed to parse print format string '") << fmt << "'";
    return;
  }

  // f64 is rejected here rather than silently truncated: DebugTunnel FLOAT
  // records carry an f32 bit pattern, so an f64 operand would print at
  // reduced precision without any visible warning.
  if (auto ft = dyn_cast<FloatType>(op.getScalar().getType()); ft && ft.isF64()) {
    op.emitError() << "f64 print is not supported by the VPTO DebugTunnel "
                      "lowering: FLOAT records carry f32, so the value would "
                      "be silently truncated; cast to f32 (or f16/bf16) "
                      "before pto.print";
    return;
  }
  int64_t recordSize = static_cast<int64_t>(info->getRecordSize());

  // Create the write-pointer chain: reserve -> [prefix text] -> scalar ->
  // [suffix text] -> commit.  A null pointer propagates through the chain as
  // the "no-op" signal.
  Value wp = builder.create<pto::DebugReserveOp>(
      loc, ptr1Type, builder.getI64IntegerAttr(recordSize));
  if (info->prefixBytes) {
    StringRef prefix = fmt.substr(0, info->conversionOffset);
    auto wt = builder.create<pto::DebugWriteTextOp>(
        loc, ptr1Type, wp, builder.getStringAttr(globals.getOrCreate(prefix)),
        builder.getI64IntegerAttr(info->prefixBytes));
    wp = wt.getNextPtr();
  }
  StringRef spec = fmt.substr(info->conversionOffset, info->conversionBytes - 1);
  auto ws = builder.create<pto::DebugWriteScalarOp>(
      loc, ptr1Type, wp, op.getScalar(),
      builder.getStringAttr(globals.getOrCreate(spec)),
      builder.getI64IntegerAttr(info->conversionBytes),
      builder.getStringAttr(kindToString(info->conversion)));
  wp = ws.getNextPtr();
  if (info->suffixBytes) {
    StringRef suffix = fmt.substr(info->suffixOffset, info->suffixBytes - 1);
    auto wt = builder.create<pto::DebugWriteTextOp>(
        loc, ptr1Type, wp, builder.getStringAttr(globals.getOrCreate(suffix)),
        builder.getI64IntegerAttr(info->suffixBytes));
    wp = wt.getNextPtr();
  }
  builder.create<pto::DebugCommitOp>(loc, wp);
  op.erase();
}

// ---------------------------------------------------------------------------
// pto.tprint -> debug chain + tile traversal loops
// ---------------------------------------------------------------------------
static void rewriteTPrintOp(pto::TPrintOp op, StringGlobalTable &globals) {
  Location loc = op.getLoc();
  OpBuilder builder(op);
  MLIRContext *ctx = op.getContext();
  auto ptr1Type = LLVM::LLVMPointerType::get(ctx, 1);
  auto ptr6Type = LLVM::LLVMPointerType::get(ctx, 6);
  auto i64Type = builder.getI64Type();

  auto srcType = dyn_cast<pto::TileBufType>(op.getSrc().getType());
  if (!srcType) {
    op.emitError("pto.tprint: source must be a tile_buf");
    return;
  }
  ArrayRef<int64_t> shape = srcType.getShape();
  if (shape.size() != 2) {
    op.emitError("pto.tprint: only 2D tiles supported");
    return;
  }
  int64_t rows = shape[0], cols = shape[1];
  Type elemType = srcType.getElementType();
  bool isFloat = isa<FloatType>(elemType);
  int64_t dataSize = isFloat ? debugtunnel::kFloatBytes : debugtunnel::kIntBytes;
  int64_t elemBytes = elemType.getIntOrFloatBitWidth() / 8;
  StringRef valFmt = isFloat ? "%6.2f" : "%6d";
  int64_t fmtPrefixLen = isFloat ? 6 : 4; // spec chars + NUL
  int64_t elemRecordSize = debugtunnel::scalarRecordSize(dataSize,
                                                         fmtPrefixLen);
  std::string dtypeName = elemType.isF16()   ? "float16"
                          : elemType.isF32() ? "float32"
                                             : "int32";
  std::string headerText = "=== [TPRINT Tile] Data Type: " + dtypeName +
                           ", Layout: ND, TileType: Vec ===\n";
  std::string shapeText = "  Shape: [" + std::to_string(rows) + ", " +
                          std::to_string(cols) + "], Valid Shape: [" +
                          std::to_string(rows) + ", " +
                          std::to_string(cols) + "]\n";
  auto literalRecordSize = [](const std::string &text) {
    return static_cast<int64_t>(debugtunnel::textRecordSize(text.size() + 1));
  };
  int64_t totalSize = literalRecordSize(headerText) +
                      literalRecordSize(shapeText) +
                      rows * cols * elemRecordSize +
                      debugtunnel::endRecordSize();

  Value wp = builder.create<pto::DebugReserveOp>(
      loc, ptr1Type, builder.getI64IntegerAttr(totalSize));
  auto header = builder.create<pto::DebugWriteTextOp>(
      loc, ptr1Type, wp,
      builder.getStringAttr(globals.getOrCreate(headerText)),
      builder.getI64IntegerAttr(headerText.size() + 1));
  auto shapeRec = builder.create<pto::DebugWriteTextOp>(
      loc, ptr1Type, header.getNextPtr(),
      builder.getStringAttr(globals.getOrCreate(shapeText)),
      builder.getI64IntegerAttr(shapeText.size() + 1));

  // Tile element addressing: the runtime-ABI layer translates the
  // virtual-address base into an element offset; element loads are plain
  // addrspace(6) accesses (tiles are UB placeholders in VPTO).
  auto baseOff = builder.create<pto::DebugGetTileBaseOp>(
      loc, i64Type, builder.getI64IntegerAttr(elemBytes));

  auto c0Idx = builder.create<arith::ConstantIndexOp>(loc, 0);
  auto c1Idx = builder.create<arith::ConstantIndexOp>(loc, 1);
  auto rowsIdx = builder.create<arith::ConstantIndexOp>(loc, rows);
  auto colsIdx = builder.create<arith::ConstantIndexOp>(loc, cols);
  auto cols64 = builder.create<LLVM::ConstantOp>(
      loc, i64Type, builder.getI64IntegerAttr(cols));
  auto tileDataBase = builder.create<LLVM::ZeroOp>(loc, ptr6Type);

  auto rowLoop = builder.create<scf::ForOp>(loc, c0Idx, rowsIdx, c1Idx,
                                            ValueRange{shapeRec.getNextPtr()});
  {
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(rowLoop.getBody());
    Value row = rowLoop.getInductionVar();
    Value wpRow = rowLoop.getRegionIterArgs()[0];
    auto colLoop =
        builder.create<scf::ForOp>(loc, c0Idx, colsIdx, c1Idx, ValueRange{wpRow});
    {
      OpBuilder::InsertionGuard guard2(builder);
      builder.setInsertionPointToStart(colLoop.getBody());
      Value col = colLoop.getInductionVar();
      Value wpCol = colLoop.getRegionIterArgs()[0];
      auto rowI64 = builder.create<arith::IndexCastOp>(loc, i64Type, row);
      auto colI64 = builder.create<arith::IndexCastOp>(loc, i64Type, col);
      auto rowOff = builder.create<LLVM::MulOp>(loc, i64Type, rowI64, cols64);
      auto elemIdx = builder.create<LLVM::AddOp>(loc, i64Type, rowOff, colI64);
      auto virtElemOff =
          builder.create<LLVM::AddOp>(loc, i64Type, baseOff, elemIdx);
      auto elemPtr = builder.create<LLVM::GEPOp>(
          loc, ptr6Type, elemType, tileDataBase,
          ValueRange{virtElemOff.getResult()});
      Value elemVal = builder.create<LLVM::LoadOp>(loc, elemType, elemPtr);
      auto ws = builder.create<pto::DebugWriteScalarOp>(
          loc, ptr1Type, wpCol, elemVal,
          builder.getStringAttr(globals.getOrCreate(valFmt)),
          builder.getI64IntegerAttr(fmtPrefixLen),
          builder.getStringAttr(isFloat ? "float" : "signed"));
      builder.create<scf::YieldOp>(loc, ValueRange{ws.getNextPtr()});
    }
    builder.create<scf::YieldOp>(loc, colLoop.getResults());
  }
  builder.create<pto::DebugCommitOp>(loc, rowLoop.getResults()[0]);
  op.erase();
}

} // namespace

LogicalResult lowerPrintToDebugRuntime(ModuleOp module,
                                       PrintLoweringState &state) {
  // Collect before rewriting (walk + erase is unsafe in-place).
  SmallVector<pto::PrintOp> prints;
  module.walk([&](pto::PrintOp op) { prints.push_back(op); });
  SmallVector<pto::TPrintOp> tprints;
  module.walk([&](pto::TPrintOp op) { tprints.push_back(op); });
  if (prints.empty() && tprints.empty())
    return success();

  state.usesPrint = true;
  StringGlobalTable globals(module);
  for (pto::PrintOp op : prints)
    rewritePrintOp(op, globals);
  for (pto::TPrintOp op : tprints)
    rewriteTPrintOp(op, globals);
  return success();
}

} // namespace pto
} // namespace mlir
