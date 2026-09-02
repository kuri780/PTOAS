// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- VPTOBridgeRegistry.h - typed bridge ABI registry --------*- C++ -*-===//
//===----------------------------------------------------------------------===//
// Compiler-owned bridge entry descriptions shared by lowering and rendering.
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_PTO_TRANSFORMS_VPTOBRIDGEREGISTRY_H
#define MLIR_DIALECT_PTO_TRANSFORMS_VPTOBRIDGEREGISTRY_H

#include "mlir/Support/LLVM.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include <cstdint>
#include <optional>
#include <string>

namespace mlir {
namespace pto {
class TMatmulOp;
class TMatmulMxOp;
enum class BridgeFamily : uint8_t { Pipe, Cube };
enum class BridgeCore : uint8_t { Vec, Cube, Both };
enum class BridgeValueKind : uint8_t { Pointer, I32, I64 };
enum class BridgeEntryId : uint8_t {
  PipeInit, PipeSize, PipePush, PipePop, PipeFree, CubeTMatmul, CubeTMatmulMx
};
struct BridgeFunctionDesc {
  BridgeEntryId id;
  BridgeFamily family;
  BridgeCore core;
  llvm::StringLiteral symbol;
  llvm::ArrayRef<BridgeValueKind> arguments;
  llvm::ArrayRef<BridgeValueKind> results;
  bool stateful;
  std::optional<BridgeEntryId> storageSizeEntry;
};
const BridgeFunctionDesc *lookupBridgeEntry(BridgeEntryId id);
const BridgeFunctionDesc *lookupBridgeEntryForOp(llvm::StringRef opName);
const BridgeFunctionDesc *lookupBridgeEntryForSymbol(llvm::StringRef symbol);
const BridgeFunctionDesc *lookupBridgeEntryForName(llvm::StringRef name);
llvm::StringRef getBridgeEntryName(BridgeEntryId id);
std::string getBridgeInstanceSymbol(BridgeEntryId id, int64_t instanceId);
bool bridgeValueKindMatches(BridgeValueKind kind, Type type);

struct BridgeInstanceKey {
  BridgeEntryId entry;
  std::string specialization;
  BridgeCore core;

  bool operator<(const BridgeInstanceKey &other) const {
    if (entry != other.entry)
      return static_cast<uint8_t>(entry) < static_cast<uint8_t>(other.entry);
    if (specialization != other.specialization)
      return specialization < other.specialization;
    return static_cast<uint8_t>(core) < static_cast<uint8_t>(other.core);
  }
};

/// Typed operand adapter for the stateless Cube tmatmul entry. The adapter
/// intentionally exposes no YAML-configurable operand positions.
LogicalResult collectTMatmulBridgeOperands(
    TMatmulOp op, llvm::SmallVectorImpl<mlir::Value> &addresses);
LogicalResult collectTMatmulMxBridgeOperands(
    TMatmulMxOp op, llvm::SmallVectorImpl<mlir::Value> &addresses);
} // namespace pto
} // namespace mlir

#endif // MLIR_DIALECT_PTO_TRANSFORMS_VPTOBRIDGEREGISTRY_H
