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
#include "llvm/ADT/StringRef.h"

namespace mlir {
namespace pto {

// ---------------------------------------------------------------------------
// Print conversion kind — drives sext/zext selection for integer encoding.
// ---------------------------------------------------------------------------
enum class PrintConversionKind : uint8_t {
  Float,       // %f, %F, %e, %E, %g, %G, %a, %A
  SignedInt,   // %d, %i
  UnsignedInt, // %u, %x, %X, %o
};

// ---------------------------------------------------------------------------
// Result of a single-pass format-string analysis.
// ---------------------------------------------------------------------------
struct PrintFormatInfo {
  PrintConversionKind conversion;
  /// Number of bytes to emit for the prefix (the part up to and including
  /// the conversion specifier), counting the null terminator.
  uint16_t prefixBytes;
  /// Number of bytes to emit for the suffix (the part after the conversion
  /// specifier), counting the null terminator.
  uint16_t suffixBytes;

  /// Protocol data size in bytes: 4 for float, 8 for integer.
  unsigned getDataSize() const {
    return (conversion == PrintConversionKind::Float) ? 4 : 8;
  }

  /// Total DebugTunnel protocol record size for this print op.
  /// Layout: [type:1][data:4/8][fmt_len:2][fmt_prefix:N][NORMAL=1][rem_len:2][fmt_suffix:M][END=0]
  uint32_t getRecordSize() const {
    unsigned ds = getDataSize();
    return 1 + ds + 2 + prefixBytes + 1 + 2 + suffixBytes + 1;
  }
};

// ---------------------------------------------------------------------------
// Parse a PrintOp format string and produce a PrintFormatInfo.
//
// The format string must contain exactly one conversion specifier (%% is
// treated as a literal '%' and does not count).  Returns failure() with a
// diagnostic message on parse errors.
// ---------------------------------------------------------------------------
inline FailureOr<PrintFormatInfo>
analyzePrintFormat(llvm::StringRef format) {
  if (format.empty())
    return failure();

  PrintFormatInfo info{};
  bool hasConversion = false;
  size_t pos = 0;

  while (pos < format.size()) {
    if (format[pos] != '%') { ++pos; continue; }

    ++pos;
    if (pos >= format.size())
      return failure(); // trailing '%'

    if (format[pos] == '%') { ++pos; continue; } // skip %%

    if (hasConversion)
      return failure(); // more than one conversion

    // Skip flags.
    while (pos < format.size() &&
           (format[pos] == '+' || format[pos] == '-' ||
            format[pos] == '0' || format[pos] == '#' ||
            format[pos] == ' '))
      ++pos;
    // Skip width.
    while (pos < format.size() && format[pos] >= '0' && format[pos] <= '9')
      ++pos;
    // Skip precision.
    if (pos < format.size() && format[pos] == '.') {
      ++pos;
      while (pos < format.size() && format[pos] >= '0' && format[pos] <= '9')
        ++pos;
    }
    // Skip length modifier.
    if (pos < format.size() &&
        (format[pos] == 'l' || format[pos] == 'h' || format[pos] == 'z'))
      ++pos;
    if (pos >= format.size())
      return failure(); // incomplete specifier

    char conv = format[pos++];
    hasConversion = true;

    // Classify conversion kind.
    switch (conv) {
    case 'f': case 'F': case 'e': case 'E':
    case 'g': case 'G': case 'a': case 'A':
      info.conversion = PrintConversionKind::Float;
      break;
    case 'd': case 'i':
      info.conversion = PrintConversionKind::SignedInt;
      break;
    case 'u': case 'x': case 'X': case 'o':
      info.conversion = PrintConversionKind::UnsignedInt;
      break;
    default:
      return failure(); // unsupported specifier
    }

    // prefixEnd = position right after the conversion character.
    // prefixBytes = everything up to-and-including conv + null terminator.
    size_t prefixEnd = pos;
    info.prefixBytes = static_cast<uint16_t>(prefixEnd + 1);
    info.suffixBytes = static_cast<uint16_t>(format.size() - prefixEnd + 1);

    // Keep scanning to detect extra conversions (which we'll reject).
  }

  if (!hasConversion)
    return failure(); // no conversion specifier

  return info;
}

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
//
// When 'kind' is UnsignedInt, narrow integers are zero-extended; otherwise
// they are sign-extended (the default for %d/%i).
// ---------------------------------------------------------------------------
inline FailureOr<PrintScalarEncoding>
encodePrintScalar(ConversionPatternRewriter &rewriter, Location loc,
                  Type scalarType, Value scalar,
                  PrintConversionKind kind = PrintConversionKind::SignedInt) {
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
      if (kind == PrintConversionKind::UnsignedInt)
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
