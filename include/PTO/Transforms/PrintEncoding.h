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

#include <cstdint>

namespace mlir {
namespace pto {

// ---------------------------------------------------------------------------
// DebugTunnel protocol byte-level constants (CANN 9.0.x).
//
// These are the node markers and payload widths of the shared device/host
// print protocol: the device writes marker + raw value + format spec, and
// Host DebugTunnel::Close decodes and formats.  The structural DTData /
// LogBuffer layout (offsets, header size) is captured in
// DebugTunnelABIDescriptor below; both must be bumped together when the
// runtime ABI changes.
// ---------------------------------------------------------------------------
namespace debugtunnel {
inline constexpr uint8_t kEndMarker = 0;    // DT_END: terminates the record list
inline constexpr uint8_t kTextMarker = 1;   // DT_NORMAL: literal text node
inline constexpr uint8_t kFloatMarker = 2;  // DT_FLOAT: 4-byte float payload
inline constexpr uint8_t kIntMarker = 3;    // DT_INT: 8-byte integer payload
inline constexpr unsigned kFloatBytes = 4;  // FLOAT payload width
inline constexpr unsigned kIntBytes = 8;    // INT payload width

// ---------------------------------------------------------------------------
// Record-size helpers — the single source of truth for the DebugTunnel log
// record layout.  Both the semantic layer (reserve accounting in
// LowerPTOPrintToDebugRuntime.cpp) and the runtime-ABI patterns (write
// pointer advancement in PrintLowering.cpp) derive their byte counts from
// these, so the two layers can never disagree again.
// ---------------------------------------------------------------------------
// [marker(1)][len16(2)][payload] — payload = text bytes + NUL.
inline constexpr uint64_t textRecordSize(uint64_t payloadBytes) {
  return 1 + 2 + payloadBytes;
}
// [marker(1)][value(dataBytes)][len16(2)][spec][NUL] — specBytes = spec + NUL.
inline constexpr uint64_t scalarRecordSize(uint64_t dataBytes,
                                           uint64_t specBytes) {
  return 1 + dataBytes + 2 + specBytes;
}
// [marker(1)] — the DT_END terminator written by the commit.
inline constexpr uint64_t endRecordSize() { return 1; }
} // namespace debugtunnel

// ---------------------------------------------------------------------------
// Versioned DebugTunnel ABI descriptor.
//
// All magic offsets and header sizes used by the print lowering come from
// this single descriptor instead of being scattered across patterns.  If a
// future CANN release changes the DTData/LogBuffer layout, bump `version`
// and update the descriptor — nothing else in the lowering should change.
// ---------------------------------------------------------------------------
struct DebugTunnelABIDescriptor {
  unsigned version;                 // ABI version of the layout below.
  uint64_t logWholeRegionOffset;    // DTData[+0]  : LogWholeRegion pointer
  uint64_t logBufferSizeOffset;     // DTData[+16] : per-block payload capacity
  uint64_t kernelWriteTypeOffset;   // DTData[+24] : AiC / AiV / Mix tag
  uint64_t logBufferHeaderBytes;    // per-block LogBuffer prefix before payload
  uint8_t kernelWriteTypeAiV;       // kernelWriteType value for an AiV kernel
};

/// DebugTunnel ABI for CANN 9.0.x (verified against the AiV print path).
inline const DebugTunnelABIDescriptor &getDebugTunnelABI() {
  static const DebugTunnelABIDescriptor kABI = {
      /*version=*/1,
      /*logWholeRegionOffset=*/0,
      /*logBufferSizeOffset=*/16,
      /*kernelWriteTypeOffset=*/24,
      /*logBufferHeaderBytes=*/64,
      /*kernelWriteTypeAiV=*/2,
  };
  return kABI;
}

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
  uint16_t prefixOffset;
  uint16_t prefixBytes;
  uint16_t conversionOffset;
  uint16_t conversionBytes;
  uint16_t suffixOffset;
  uint16_t suffixBytes;

  /// Protocol data size in bytes: 4 for float, 8 for integer.
  unsigned getDataSize() const {
    return (conversion == PrintConversionKind::Float) ? 4 : 8;
  }

  uint32_t getRecordSize() const {
    unsigned size = debugtunnel::scalarRecordSize(getDataSize(),
                                                  conversionBytes);
    if (prefixBytes) size += debugtunnel::textRecordSize(prefixBytes);
    if (suffixBytes) size += debugtunnel::textRecordSize(suffixBytes);
    return size + debugtunnel::endRecordSize();
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

    size_t conversionEnd = pos;
    size_t conversionStart = format.rfind('%', conversionEnd - 1);
    info.prefixOffset = 0;
    info.prefixBytes = static_cast<uint16_t>(conversionStart ? conversionStart + 1 : 0);
    info.conversionOffset = static_cast<uint16_t>(conversionStart);
    info.conversionBytes = static_cast<uint16_t>(conversionEnd - conversionStart + 1);
    info.suffixOffset = static_cast<uint16_t>(conversionEnd);
    info.suffixBytes = static_cast<uint16_t>(conversionEnd < format.size()
                                                 ? format.size() - conversionEnd + 1
                                                 : 0);

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
// Float:  f16/bf16 → fpext  → f32 → bitcast → i32  (4 bytes, DT_FLOAT)
//         f32      →          bitcast → i32          (4 bytes, DT_FLOAT)
//         f64      → rejected: FLOAT records carry f32, so the value would
//                    be silently truncated.  Callers must reject f64 before
//                    reaching this point (see LowerPrintOpPattern).
//
// Int:    i8..i32  → sext/zext → i64                (8 bytes, DT_INT)
//         i64      →            (no-op)              (8 bytes, DT_INT)
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
    enc.marker = debugtunnel::kFloatMarker;
    enc.byteWidth = debugtunnel::kFloatBytes;
    auto f32Type = rewriter.getF32Type();

    if (ft.isF16() || ft.isBF16()) {
      auto f32val = rewriter.create<LLVM::FPExtOp>(loc, f32Type, scalar);
      enc.bits = rewriter.create<LLVM::BitcastOp>(loc, i32Type, f32val);
    } else if (ft.isF32()) {
      enc.bits = rewriter.create<LLVM::BitcastOp>(loc, i32Type, scalar);
    } else {
      // f64 (or any wider float): cannot be transmitted without precision
      // loss — the FLOAT record carries an f32 bit pattern.
      return rewriter.notifyMatchFailure(
          loc, "print encoding: float types wider than f32 are not supported "
               "by the DebugTunnel protocol");
    }
  } else if (auto it = dyn_cast<IntegerType>(scalarType)) {
    enc.marker = debugtunnel::kIntMarker;
    enc.byteWidth = debugtunnel::kIntBytes;
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
