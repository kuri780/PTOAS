// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#ifndef MLIR_DIALECT_PTO_TRANSFORMS_PRINTENCODING_H
#define MLIR_DIALECT_PTO_TRANSFORMS_PRINTENCODING_H

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace pto {

// ---------------------------------------------------------------------------
// Encoded scalar value ready for DebugTunnel protocol byte writing.
// ---------------------------------------------------------------------------
struct PrintScalarEncoding {
  Value bits;         // i32 (float) or i64 (int) — ready for byte extraction
  unsigned byteWidth; // protocol data bytes: 4 (DT_FLOAT) or 8 (DT_INT)
  uint8_t marker;     // DTType marker: 2 (FLOAT) or 3 (INT)
};

// ---------------------------------------------------------------------------
// Convert a scalar value to its DebugTunnel protocol bit pattern.
//
// Float:  f16/bf16 → fpext  → f32 → bitcast → i32  (4 bytes, marker=2)
//         f32      →          bitcast → i32          (4 bytes, marker=2)
//         f64/other→ fptrunc → f32 → bitcast → i32  (4 bytes, marker=2)
//
// Int:    i8..i32  → sext/zext → i64                (8 bytes, marker=3)
//         i64      →            (no-op)              (8 bytes, marker=3)
// ---------------------------------------------------------------------------
inline FailureOr<PrintScalarEncoding>
encodePrintScalar(ConversionPatternRewriter &rewriter, Location loc,
                  Type scalarType, Value scalar) {
  PrintScalarEncoding enc{};
  auto i32Type = rewriter.getI32Type();
  auto i64Type = rewriter.getI64Type();

  if (auto ft = dyn_cast<FloatType>(scalarType)) {
    enc.marker = 2; // DT_FLOAT
    enc.byteWidth = 4;
    auto f32Type = rewriter.getF32Type();

    if (ft.isF16() || ft.isBF16()) {
      auto f32val = rewriter.create<LLVM::FPExtOp>(loc, f32Type, scalar);
      enc.bits = rewriter.create<LLVM::BitcastOp>(loc, i32Type, f32val);
    } else if (ft.isF32()) {
      enc.bits = rewriter.create<LLVM::BitcastOp>(loc, i32Type, scalar);
    } else {
      // f64 or other: truncate to f32 for protocol compatibility
      auto f32val = rewriter.create<LLVM::FPTruncOp>(loc, f32Type, scalar);
      enc.bits = rewriter.create<LLVM::BitcastOp>(loc, i32Type, f32val);
    }
  } else if (auto it = dyn_cast<IntegerType>(scalarType)) {
    enc.marker = 3; // DT_INT
    enc.byteWidth = 8;
    unsigned w = it.getWidth();
    if (w < 64) {
      if (it.isUnsigned())
        enc.bits = rewriter.create<LLVM::ZExtOp>(loc, i64Type, scalar);
      else
        enc.bits = rewriter.create<LLVM::SExtOp>(loc, i64Type, scalar);
    } else {
      enc.bits = scalar;
    }
  } else {
    return rewriter.notifyMatchFailure(
        loc, "print encoding: expected numeric scalar type");
  }
  return enc;
}

} // namespace pto
} // namespace mlir

#endif // MLIR_DIALECT_PTO_TRANSFORMS_PRINTENCODING_H
