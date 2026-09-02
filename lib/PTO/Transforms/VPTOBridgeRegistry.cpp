// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

#include "PTO/Transforms/VPTOBridgeRegistry.h"
#include "PTO/IR/PTO.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include <string>

using namespace mlir;
using namespace mlir::pto;
namespace {
constexpr llvm::ArrayRef<BridgeValueKind> kNoValues;
constexpr BridgeValueKind kInitArgs[] = {BridgeValueKind::Pointer,
                                         BridgeValueKind::I32};
constexpr BridgeValueKind kSizeResults[] = {BridgeValueKind::I64};
constexpr BridgeValueKind kPushArgs[] = {BridgeValueKind::Pointer,
                                         BridgeValueKind::I64};
constexpr BridgeValueKind kPopArgs[] = {BridgeValueKind::Pointer};
constexpr BridgeValueKind kPopResults[] = {BridgeValueKind::I64};
constexpr BridgeValueKind kFreeArgs[] = {BridgeValueKind::Pointer};
constexpr BridgeValueKind kMatmulArgs[] = {
    BridgeValueKind::I64, BridgeValueKind::I64, BridgeValueKind::I64};
constexpr BridgeValueKind kMatmulMxArgs[] = {BridgeValueKind::I64, BridgeValueKind::I64, BridgeValueKind::I64, BridgeValueKind::I64, BridgeValueKind::I64};
constexpr BridgeFunctionDesc kEntries[] = {
    {BridgeEntryId::PipeInit, BridgeFamily::Pipe, BridgeCore::Both,
     "pto_vpto_pipe_init", kInitArgs, kNoValues, true, BridgeEntryId::PipeSize},
    {BridgeEntryId::PipeSize, BridgeFamily::Pipe, BridgeCore::Both,
     "pto_vpto_pipe_size", kNoValues, kSizeResults, false, std::nullopt},
    {BridgeEntryId::PipePush, BridgeFamily::Pipe, BridgeCore::Both,
     "pto_vpto_pipe_push", kPushArgs, kNoValues, false, std::nullopt},
    {BridgeEntryId::PipePop, BridgeFamily::Pipe, BridgeCore::Both,
     "pto_vpto_pipe_pop", kPopArgs, kPopResults, false, std::nullopt},
    {BridgeEntryId::PipeFree, BridgeFamily::Pipe, BridgeCore::Both,
     "pto_vpto_pipe_free", kFreeArgs, kNoValues, false, std::nullopt},
    {BridgeEntryId::CubeTMatmul, BridgeFamily::Cube, BridgeCore::Cube,
     "pto_vpto_tmatmul", kMatmulArgs, kNoValues, false, std::nullopt},
    {BridgeEntryId::CubeTMatmulMx, BridgeFamily::Cube, BridgeCore::Cube,
     "pto_vpto_tmatmul_mx", kMatmulMxArgs, kNoValues, false, std::nullopt}};
} // namespace

const BridgeFunctionDesc *pto::lookupBridgeEntry(BridgeEntryId id) {
  for (const auto &entry : kEntries) {
    if (entry.id == id) {
      return &entry;
    }
  }
  return nullptr;
}

const BridgeFunctionDesc *pto::lookupBridgeEntryForOp(llvm::StringRef op) {
  if (op == "pto.initialize_l2l_pipe") {
    return lookupBridgeEntry(BridgeEntryId::PipeInit);
  }
  if (op == "pto.tpush") {
    return lookupBridgeEntry(BridgeEntryId::PipePush);
  }
  if (op == "pto.tpop") {
    return lookupBridgeEntry(BridgeEntryId::PipePop);
  }
  if (op == "pto.tfree") {
    return lookupBridgeEntry(BridgeEntryId::PipeFree);
  }
  if (op == "pto.tmatmul") {
    return lookupBridgeEntry(BridgeEntryId::CubeTMatmul);
  }
  if (op == "pto.tmatmul.mx") {
    return lookupBridgeEntry(BridgeEntryId::CubeTMatmulMx);
  }
  return nullptr;
}

const BridgeFunctionDesc *
pto::lookupBridgeEntryForSymbol(llvm::StringRef symbol) {
  for (const auto &entry : kEntries) {
    if (entry.symbol == symbol) {
      return &entry;
    }
  }
  return nullptr;
}

llvm::StringRef pto::getBridgeEntryName(BridgeEntryId id) {
  switch (id) {
  case BridgeEntryId::PipeInit:
    return "pipe.init";
  case BridgeEntryId::PipeSize:
    return "pipe.size";
  case BridgeEntryId::PipePush:
    return "pipe.push";
  case BridgeEntryId::PipePop:
    return "pipe.pop";
  case BridgeEntryId::PipeFree:
    return "pipe.free";
  case BridgeEntryId::CubeTMatmul:
    return "cube.tmatmul";
  case BridgeEntryId::CubeTMatmulMx:
    return "cube.tmatmul.mx";
  }
  return {};
}

std::string pto::getBridgeInstanceSymbol(BridgeEntryId id, int64_t instanceId) {
  const BridgeFunctionDesc *entry = lookupBridgeEntry(id);
  if (!entry || instanceId < 0) {
    return {};
  }
  return (entry->symbol + "__" + std::to_string(instanceId)).str();
}

const BridgeFunctionDesc *pto::lookupBridgeEntryForName(llvm::StringRef name) {
  for (const auto &entry : kEntries) {
    if (getBridgeEntryName(entry.id) == name) {
      return &entry;
    }
  }
  return nullptr;
}

bool pto::bridgeValueKindMatches(BridgeValueKind kind, Type type) {
  switch (kind) {
  case BridgeValueKind::Pointer:
    return isa<LLVM::LLVMPointerType>(type);
  case BridgeValueKind::I32:
    return type.isInteger(32);
  case BridgeValueKind::I64:
    return type.isInteger(64);
  }
  return false;
}

LogicalResult pto::collectTMatmulBridgeOperands(
    TMatmulOp op, llvm::SmallVectorImpl<mlir::Value> &addresses) {
  Value operands[] = {op.getDst(), op.getLhs(), op.getRhs()};
  for (Value tile : operands) {
    auto alloc = tile.getDefiningOp<AllocTileOp>();
    if (!alloc || !alloc.getAddr()) {
      return op.emitError(
          "Cube tmatmul bridge requires planned alloc_tile operands");
    }
    addresses.push_back(alloc.getAddr());
  }
  return success();
}

LogicalResult pto::collectTMatmulMxBridgeOperands(TMatmulMxOp op, llvm::SmallVectorImpl<mlir::Value> &addresses) {
  Value operands[] = {op.getDst(), op.getA(), op.getAScale(), op.getB(), op.getBScale()};
  for (Value tile : operands) {
    auto alloc = tile.getDefiningOp<AllocTileOp>();
    if (!alloc || !alloc.getAddr()) return op.emitError("Cube tmatmul.mx bridge requires planned alloc_tile operands");
    addresses.push_back(alloc.getAddr());
  }
  return success();
}
