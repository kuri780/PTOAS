// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#ifndef MLIR_DIALECT_PTO_TRANSFORMS_MEMORYCONSISTENCYATTRS_H
#define MLIR_DIALECT_PTO_TRANSFORMS_MEMORYCONSISTENCYATTRS_H

#include "llvm/ADT/StringRef.h"

namespace mlir {
namespace pto {

inline constexpr llvm::StringLiteral kTNotifyDrainMte2AttrName =
    "__pto.emitc.tnotify_drain_mte2";
inline constexpr llvm::StringLiteral kTNotifyDrainMte3AttrName =
    "__pto.emitc.tnotify_drain_mte3";
inline constexpr llvm::StringLiteral kAcquireInvalidateGmCacheAttrName =
    "__pto.emitc.acquire_invalidate_gm_cache";
inline constexpr llvm::StringLiteral kCmoCacheInvalidSkipLoweringAttrName =
    "__pto.emitc.cmo_cacheinvalid_skip_lowering";

} // namespace pto
} // namespace mlir

#endif // MLIR_DIALECT_PTO_TRANSFORMS_MEMORYCONSISTENCYATTRS_H
