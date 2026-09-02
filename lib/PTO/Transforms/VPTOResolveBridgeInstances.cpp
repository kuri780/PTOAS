// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE for the specific language governing
// permissions and limitations under the License.

#include "PTO/IR/PTO.h"
#include "PTO/Transforms/VPTOBridgeRegistry.h"
#include "PTO/Transforms/VPTOBridgeTokens.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include <map>

namespace mlir::pto {
#define GEN_PASS_DECL_VPTORESOLVEBRIDGEINSTANCES
#define GEN_PASS_DEF_VPTORESOLVEBRIDGEINSTANCES
#include "PTO/Transforms/Passes.h.inc"

namespace {
struct VPTOResolveBridgeInstancesPass final
    : impl::VPTOResolveBridgeInstancesBase<VPTOResolveBridgeInstancesPass> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(VPTOResolveBridgeInstancesPass)
  void runOnOperation() override {
    ModuleOp module = getOperation();
    std::map<BridgeInstanceKey, int64_t> instances;
    int64_t nextId = 0;
    module.walk([&](BridgeCallOp op) {
      auto entryId = op.getEntryId();
      if (!entryId) {
        return;
      }
      const BridgeFunctionDesc *entry = lookupBridgeEntryForName(*entryId);
      if (!entry) {
        op.emitError() << "cannot resolve bridge instance for unknown entry '"
                       << *entryId << "'";
        signalPassFailure();
        return;
      }
      std::string spec;
      if (DictionaryAttr attr = op.getSpecializationAttr()) {
        llvm::raw_string_ostream os(spec);
        attr.print(os);
      } else if (auto attr = module->getAttrOfType<DictionaryAttr>(
                     kBridgeSpecAttrName)) {
        llvm::raw_string_ostream os(spec);
        attr.print(os);
      }
      BridgeInstanceKey key{entry->id, std::move(spec), entry->core};
      auto [it, inserted] = instances.emplace(std::move(key), nextId);
      if (inserted) {
        ++nextId;
      }
      op.setInstanceIdAttr(IntegerAttr::get(
          IntegerType::get(module.getContext(), 64), it->second));
      op.setCallee(getBridgeInstanceSymbol(entry->id, it->second));
    });
    module.walk([&](BridgeObjectCreateOp op) {
      const BridgeFunctionDesc *entry = lookupBridgeEntryForName(op.getEntry());
      if (!entry) {
        op.emitError()
            << "cannot resolve bridge object instance for unknown entry '"
            << op.getEntry() << "'";
        signalPassFailure();
        return;
      }
      std::string spec;
      if (DictionaryAttr attr = op.getSpecializationAttr()) {
        llvm::raw_string_ostream os(spec);
        attr.print(os);
      }
      BridgeInstanceKey key{entry->id, std::move(spec), entry->core};
      auto [it, inserted] = instances.emplace(std::move(key), nextId);
      if (inserted) {
        ++nextId;
      }
      op.setInstanceIdAttr(IntegerAttr::get(
          IntegerType::get(module.getContext(), 64), it->second));
    });
  }
};
} // namespace

std::unique_ptr<Pass> createVPTOResolveBridgeInstancesPass() {
  return std::make_unique<VPTOResolveBridgeInstancesPass>();
}
} // namespace mlir::pto
