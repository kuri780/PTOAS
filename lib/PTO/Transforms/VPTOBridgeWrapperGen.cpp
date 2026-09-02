// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

//===- VPTOBridgeWrapperGen.cpp - bridge wrapper source generation -------===//
//===----------------------------------------------------------------------===//
//
// Wrapper generation pass of the VPTO C++ interface bridge. The family
// passes collect the wrapper specialization (C++ template tokens built from
// the op attributes/operand types plus the whitelist wrapper entry names)
// into per-function attributes; this pass merges them into the module spec
// and renders it into the complete bridge wrapper C++ source, stored in the
// `pto.vpto.bridge.wrapper_source` module attribute. Object emission then
// compiles the source with Bisheng (once per core kind, selected by the
// __DAV_CUBE__/__DAV_VEC__ guards) and links the bitcode into the device
// modules, replacing the former hand-written wrapper translation unit.
//
// Rendering dispatches on the wrapper the module used. A wrapper whose
// entries carry `lowering: custom` owns a dedicated renderer that knows
// its family semantics (today only pipe: Pipe/Tile typedefs, a
// placement-new init entry, a sizeof size entry, and the producer/consumer
// entries placed on the cores implied by the pipe direction). Every other
// wrapper renders through the generic declarative renderer, driven
// entirely by the whitelist: the wrappers section supplies the includes
// and the core guard, and each used entry its call spelling, template
// arguments and abi-role tile typedefs. Adding a mechanically mapped
// interface therefore needs only a whitelist registration.
//
//===----------------------------------------------------------------------===//

#include "PTO/IR/PTO.h"
#include "PTO/Transforms/VPTOBridgeRegistry.h"
#include "PTO/Transforms/VPTOBridgeTokens.h"
#include "PTO/Transforms/VPTOBridgeWhitelist.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/raw_ostream.h"
#include <map>
#include <string>

namespace mlir {
namespace pto {

#define GEN_PASS_DECL_VPTOBRIDGEWRAPPERGEN
#define GEN_PASS_DEF_VPTOBRIDGEWRAPPERGEN
#include "PTO/Transforms/Passes.h.inc"

namespace {

/// `wrapper` whitelist values owning a dedicated renderer: entries of such
/// a wrapper carry `lowering: custom` and their family semantics (storage
/// lifecycle, address rebinding) are rendered by hand-written code below
/// instead of the generic declarative renderer. Every other wrapper is
/// whitelist declared and renders generically.
constexpr llvm::StringLiteral kPipeWrapper = "pipe";

/// The pipe bridge specialization fields read from the spec DictionaryAttr.
/// The tile typedefs are not read here: they come from the pipe entries'
/// tmpl_map declarations.
struct BridgePipeSpec {
  std::string pipe;
  std::string split;
  std::string entryInit;
  std::string entrySize;
  std::string entryPush;
  std::string entryPop;
  std::string entryFree;
};

/// One `using <target> = <token>;` typedef rendered into the wrapper.
struct BridgeTypedefDecl {
  std::string target;
  std::string token;
};

/// A {spec key -> spec struct field} row used to fill the family spec
/// structs from the spec attribute.
struct BridgeSpecField {
  llvm::StringLiteral key;
  std::string *field;
};

/// Renders the complete bridge wrapper source for the pipe specialization.
/// For a C2V pipe the cube core produces (push) and the vector core consumes
/// (pop/free); a V2C pipe swaps the roles. Each core section is guarded so
/// that compiling the same source per core kind yields exactly the entries
/// of that core. The typedef section is rendered from the tmpl_map-driven
/// declarations; the body references the fixed Pipe/ProducerTile/
/// ConsumerTile names.
FailureOr<std::string>
renderPipeBridgeSource(const BridgePipeSpec &spec,
                       ArrayRef<BridgeTypedefDecl> typedefs) {
  bool cubeProduces;
  if (llvm::StringRef(spec.pipe).contains("pto::Direction::DIR_C2V")) {
    cubeProduces = true;
  } else if (llvm::StringRef(spec.pipe).contains("pto::Direction::DIR_V2C")) {
    cubeProduces = false;
  } else {
    return failure();
  }

  std::string source;
  llvm::raw_string_ostream os(source);

  os << "// Generated by ptoas (pto-emit-vpto-bridge-wrapper). Do not edit.\n"
     << "// VPTO pipe bridge wrapper for " << spec.pipe << ".\n"
     << "#include <pto/pto-inst.hpp>\n"
     << "#include <pto/npu/a5/TFree.hpp>\n"
     << "#include <pto/npu/a5/TPop.hpp>\n"
     << "#include <pto/npu/a5/TPush.hpp>\n"
     << "#include <stddef.h>\n"
     << "#include <stdint.h>\n"
     << "\n"
     << "[aicore] inline void *operator new(size_t, void *ptr) noexcept { "
        "return ptr; }\n"
     << "\n";
  for (const BridgeTypedefDecl &decl : typedefs) {
    os << "using " << decl.target << " = " << decl.token << ";\n";
  }
  os << "\n"
     << "extern \"C\" [aicore] void " << spec.entryInit
     << "(void *storage, uint32_t localBuffer) {\n"
     << "  new (storage) Pipe(nullptr, localBuffer, 0);\n"
     << "}\n"
     << "\n"
     << "extern \"C\" [aicore] size_t " << spec.entrySize
     << "() { return sizeof(Pipe); }\n"
     << "\n";

  auto renderPush = [&]() {
    os << "extern \"C\" [aicore] void " << spec.entryPush
       << "(void *storage, uint64_t producerAddress) {\n"
       << "  auto &pipe = *reinterpret_cast<Pipe *>(storage);\n"
       << "  ProducerTile tile;\n"
       << "  pto::TASSIGN_IMPL(tile, producerAddress);\n"
       << "  pto::TPUSH<Pipe, ProducerTile, " << spec.split
       << ">(pipe, tile);\n"
       << "}\n";
  };
  auto renderPop = [&]() {
    os << "extern \"C\" [aicore] uint64_t " << spec.entryPop
       << "(void *storage) {\n"
       << "  auto &pipe = *reinterpret_cast<Pipe *>(storage);\n"
       << "  ConsumerTile tile;\n"
       << "  pto::TPOP<Pipe, ConsumerTile, " << spec.split << ">(pipe, tile);\n"
       << "  pipe_barrier(PIPE_ALL);\n"
       << "  return reinterpret_cast<uint64_t>(tile.data());\n"
       << "}\n";
  };
  auto renderFree = [&]() {
    os << "extern \"C\" [aicore] void " << spec.entryFree
       << "(void *storage) {\n"
       << "  auto &pipe = *reinterpret_cast<Pipe *>(storage);\n"
       << "  pto::TFREE<Pipe, " << spec.split << ">(pipe);\n"
       << "}\n";
  };

  os << "#ifdef __DAV_CUBE__\n";
  if (cubeProduces) {
    renderPush();
  } else {
    renderPop();
    os << "\n";
    renderFree();
  }
  os << "#endif\n"
     << "\n"
     << "#ifdef __DAV_VEC__\n";
  if (cubeProduces) {
    renderPop();
    os << "\n";
    renderFree();
  } else {
    renderPush();
  }
  os << "#endif\n";

  os.flush();
  return source;
}

/// Returns the C++ parameter spelling of an abi carrier type.
static llvm::StringLiteral bridgeAbiParamType(llvm::StringRef type) {
  if (type == "i64")
    return "uint64_t";
  if (type == "i32")
    return "uint32_t";
  return "void *";
}

/// Renders all concrete Cube tmatmul instances. Each call owns a structured
/// specialization whose TileBufType attributes remain typed until this final
/// renderer converts them to PTO-ISA C++ spellings.
static FailureOr<std::string> renderTMatmulBridgeSource(ModuleOp module) {
  std::string source;
  llvm::raw_string_ostream os(source);
  os << "// Generated by ptoas (pto-emit-vpto-bridge-wrapper). Do not edit.\n"
     << "#include <pto/pto-inst.hpp>\n"
     << "#include <pto/npu/a5/TMatmul.hpp>\n"
     << "#include <stdint.h>\n\n"
     << "#ifdef __DAV_CUBE__\n";
  bool found = false;
  LogicalResult status = success();
  module.walk([&](BridgeCallOp call) {
    if (failed(status) ||
        call.getEntryId() != getBridgeEntryName(BridgeEntryId::CubeTMatmul)) {
      return;
    }
    DictionaryAttr specialization = call.getSpecializationAttr();
    if (!specialization) {
      call.emitError(
          "Cube tmatmul bridge call has no structured specialization");
      status = failure();
      return;
    }
    auto renderTile = [&](StringRef role) -> FailureOr<std::string> {
      auto typeAttr = specialization.getAs<TypeAttr>(role);
      auto tile =
          typeAttr ? dyn_cast<TileBufType>(typeAttr.getValue()) : TileBufType();
      if (!tile) {
        call.emitError() << "Cube tmatmul specialization field '" << role
                         << "' must be a tile_buf TypeAttr";
        return failure();
      }
      return buildBridgeTileToken(tile);
    };
    FailureOr<std::string> dst = renderTile("dst");
    FailureOr<std::string> lhs = renderTile("lhs");
    FailureOr<std::string> rhs = renderTile("rhs");
    if (failed(dst) || failed(lhs) || failed(rhs)) {
      status = failure();
      return;
    }
    found = true;
    os << "extern \"C\" [aicore] void " << call.getCallee()
       << "(uint64_t dstAddress, uint64_t lhsAddress, uint64_t rhsAddress) {\n"
       << "  using Dst = " << *dst << ";\n"
       << "  using Lhs = " << *lhs << ";\n"
       << "  using Rhs = " << *rhs << ";\n"
       << "  Dst dst;\n  Lhs lhs;\n  Rhs rhs;\n"
       << "  pto::TASSIGN_IMPL(dst, dstAddress);\n"
       << "  pto::TASSIGN_IMPL(lhs, lhsAddress);\n"
       << "  pto::TASSIGN_IMPL(rhs, rhsAddress);\n"
       << "  pto::TMATMUL(dst, lhs, rhs);\n}\n\n";
  });
  if (failed(status) || !found) {
    return failure();
  }
  os << "#endif\n";
  os.flush();
  return source;
}

static FailureOr<std::string> renderTMatmulMxBridgeSource(ModuleOp module) {
  std::string source;
  llvm::raw_string_ostream os(source);
  os << "// Generated by ptoas.\n#include <pto/pto-inst.hpp>\n#include <pto/npu/a5/TMatmul.hpp>\n#include <stdint.h>\n#ifdef __DAV_CUBE__\n";
  bool found = false;
  LogicalResult status = success();
  module.walk([&](BridgeCallOp call) {
    if (failed(status) ||
        call.getEntryId() != getBridgeEntryName(BridgeEntryId::CubeTMatmulMx))
      return;
    auto spec = call.getSpecializationAttr();
    if (!spec) {
      status = failure();
      return;
    }
    StringRef names[] = {"dst", "a", "a_scale", "b", "b_scale"}; std::string toks[5];
    for (unsigned i=0;i<5;++i) { auto ta=spec.getAs<TypeAttr>(names[i]); auto tile=ta?dyn_cast<TileBufType>(ta.getValue()):TileBufType(); if(!tile){status=failure(); return;} auto tok=buildBridgeTileToken(tile); if(failed(tok)){status=failure(); return;} toks[i]=*tok; }
    found=true; os << "extern \"C\" [aicore] void " << call.getCallee() << "(uint64_t dstAddress, uint64_t aAddress, uint64_t aScaleAddress, uint64_t bAddress, uint64_t bScaleAddress) {\n";
    os << "  using Dst="<<toks[0]<<"; using A="<<toks[1]<<"; using AScale="<<toks[2]<<"; using B="<<toks[3]<<"; using BScale="<<toks[4]<<"; Dst dst; A a; AScale as; B b; BScale bs; pto::TASSIGN_IMPL(dst,dstAddress); pto::TASSIGN_IMPL(a,aAddress); pto::TASSIGN_IMPL(as,aScaleAddress); pto::TASSIGN_IMPL(b,bAddress); pto::TASSIGN_IMPL(bs,bScaleAddress); pto::TMATMUL_MX(dst,a,as,b,bs); }\n";
  });
  if (failed(status)||!found) return failure(); os << "#endif\n"; os.flush(); return source;
}

/// Renders the complete bridge wrapper source of a wrapper declared in the
/// whitelist wrappers section: the declaration supplies the includes and
/// the core guard, and every used declarative entry its abi-bound tile
/// typedefs (named by the CamelCase of the role), the TASSIGN bindings and
/// the declared call. Tile typedef targets sort by name so the rendered
/// source is stable under whitelist reordering; the merged spec guarantees
/// one token per role. A tmpl_args item either is a qualified literal or
/// names an attr spec key, and the whole template argument list is omitted
/// when the spec carries no token for one (e.g. an Unspecified phase).
FailureOr<std::string>
renderDeclarativeBridgeSource(ModuleOp module, const BridgeWhitelist &whitelist,
                              const BridgeWrapperDecl &decl,
                              DictionaryAttr specAttr,
                              const llvm::StringSet<> &usedEntries) {
  // Tile typedefs: one per abi role the used entries bind. A role token
  // missing from the spec means the declarative lowering never routed the
  // op; say so instead of rendering an undefined typedef.
  std::map<std::string, std::string> typedefTokens;
  for (const BridgeWhitelistEntry &entry : whitelist.bridgeOps) {
    if (entry.wrapper != decl.name || !entry.isDeclarative() ||
        !usedEntries.count(entry.entry)) {
      continue;
    }
    for (const BridgeAbiArg &arg : entry.abi) {
      std::string target = bridgeRoleTypedefTarget(arg.role);
      if (typedefTokens.count(target))
        continue;
      auto value = specAttr.getAs<StringAttr>(arg.role);
      if (!value || value.getValue().empty()) {
        module.emitError()
            << "VPTO bridge: whitelist entry '" << entry.entry
            << "' binds the role '" << arg.role
            << "', but no tile token was collected for it; the declarative "
               "bridge lowering must run before wrapper generation";
        return failure();
      }
      typedefTokens.emplace(target, value.getValue().str());
    }
  }
  // Attribute tmpl_map rows never render a typedef: their spec tokens are
  // constant values (e.g. an AccPhase enumerator), not types; they feed the
  // entry's tmpl_args instead. A missing spec token omits the whole
  // template argument list at call render time below.

  std::string source;
  llvm::raw_string_ostream os(source);

  os << "// Generated by ptoas (pto-emit-vpto-bridge-wrapper). Do not edit.\n"
     << "// VPTO bridge wrapper for '" << decl.name << "'.\n"
     << "#include <pto/pto-inst.hpp>\n";
  for (const std::string &include : decl.includes) {
    os << "#include <" << include << ">\n";
  }
  os << "#include <stdint.h>\n"
     << "\n";
  for (const auto &typedefToken : typedefTokens) {
    os << "using " << typedefToken.first << " = " << typedefToken.second
       << ";\n";
  }
  os << "\n";

  // Core guard: declared in the wrappers section, or derived from the tile
  // kinds the declarative lowering collected under the reserved
  // `core.<wrapper>` spec key. A wrapper whose used entries collected no
  // tile has nothing to derive from and must declare `core`.
  llvm::StringRef core = decl.core;
  if (core.empty()) {
    auto coreToken = specAttr.getAs<StringAttr>("core." + decl.name);
    if (!coreToken || coreToken.getValue().empty()) {
      module.emitError()
          << "VPTO bridge: wrapper '" << decl.name
          << "' declares no core and the declarative lowering collected "
             "no tile kind for it; declare 'core: cube|vec|both' in the "
             "wrappers section";
      return failure();
    }
    core = coreToken.getValue();
  }
  const bool guardCube = core == kBridgeWrapperCoreCube;
  const bool guardVec = core == kBridgeWrapperCoreVec;
  if (guardCube)
    os << "#ifdef __DAV_CUBE__\n";
  else if (guardVec)
    os << "#ifdef __DAV_VEC__\n";

  bool firstEntry = true;
  for (const BridgeWhitelistEntry &entry : whitelist.bridgeOps) {
    if (entry.wrapper != decl.name || !entry.isDeclarative() ||
        !usedEntries.count(entry.entry))
      continue;
    SmallVector<std::string> symbols;
    module.walk([&](BridgeCallOp call) {
      auto id = call.getEntryId();
      if (!id) {
        return;
      }
      const BridgeFunctionDesc *desc = lookupBridgeEntryForName(*id);
      if (!desc || desc->symbol != entry.entry) {
        return;
      }
      StringRef symbol = call.getCallee();
      if (symbol.empty()) {
        symbol = entry.entry;
      }
      if (!llvm::is_contained(symbols, symbol.str())) {
        symbols.push_back(symbol.str());
      }
    });
    if (symbols.empty()) {
      symbols.push_back(entry.entry);
    }
    for (const std::string &symbol : symbols) {
      if (!firstEntry) {
        os << "\n";
      }
      firstEntry = false;
      os << "extern \"C\" [aicore] void " << symbol << "(";
      llvm::interleaveComma(entry.abi, os, [&](const BridgeAbiArg &arg) {
        os << bridgeAbiParamType(arg.type) << " " << arg.arg << "Address";
      });
      os << ") {\n";
      for (const BridgeAbiArg &arg : entry.abi) {
        os << "  " << bridgeRoleTypedefTarget(arg.role) << " " << arg.arg
           << ";\n";
      }
      for (const BridgeAbiArg &arg : entry.abi) {
        os << "  pto::TASSIGN_IMPL(" << arg.arg << ", " << arg.arg
           << "Address);\n";
      }
      // Template arguments: literals render as declared; spec-backed items
      // drop the whole list when their token was omitted.
      llvm::SmallVector<llvm::StringRef, 2> tmplTokens;
      bool renderTmplArgs = true;
      for (const std::string &item : entry.tmplArgs) {
        if (llvm::StringRef(item).contains("::")) {
          tmplTokens.push_back(item);
          continue;
        }
        auto value = specAttr.getAs<StringAttr>(item);
        if (!value || value.getValue().empty()) {
          renderTmplArgs = false;
          break;
        }
        tmplTokens.push_back(value.getValue());
      }
      os << "  " << entry.call;
      if (renderTmplArgs && !tmplTokens.empty()) {
        os << "<";
        llvm::interleaveComma(tmplTokens, os);
        os << ">";
      }
      os << "(";
      llvm::interleaveComma(entry.abi, os,
                            [&](const BridgeAbiArg &arg) { os << arg.arg; });
      os << ");\n"
         << "}\n";
    }
  }
  if (guardCube || guardVec)
    os << "#endif\n";

  os.flush();
  return source;
}

/// Merges the per-function bridge specs collected by the family pass into the
/// module-level spec attribute. Identical fields deduplicate; a field with
/// two different values (e.g. two pipe configurations) is a conflict. The
/// functions are visited in module order, so the merge is deterministic.
static LogicalResult mergeFuncSpecsIntoModule(ModuleOp module) {
  SmallVector<func::FuncOp> specFuncs;
  for (auto func : module.getOps<func::FuncOp>()) {
    if (func->getAttrOfType<DictionaryAttr>(kBridgeFuncSpecAttrName))
      specFuncs.push_back(func);
  }
  if (specFuncs.empty())
    return success();

  SmallVector<NamedAttribute> merged;
  for (func::FuncOp func : specFuncs) {
    auto funcSpec =
        func->getAttrOfType<DictionaryAttr>(kBridgeFuncSpecAttrName);
    for (NamedAttribute field : funcSpec) {
      bool found = false;
      for (NamedAttribute existing : merged) {
        if (existing.getName() != field.getName())
          continue;
        if (existing.getValue() != field.getValue()) {
          // Specializations are instance-scoped.  Keep the first module-level
          // value for legacy declarative consumers; built-in Pipe/Cube
          // renderers derive their complete instances from structured bridge
          // operations below and therefore must not reject a module merely
          // because another function uses a different specialization.
          found = true;
          break;
        }
        found = true;
        break;
      }
      if (!found)
        merged.push_back(field);
    }
    func->removeAttr(kBridgeFuncSpecAttrName);
  }

  module->setAttr(kBridgeSpecAttrName,
                  DictionaryAttr::get(module.getContext(), merged));
  return success();
}

/// Collects the wrapper entry names the module spec actually uses (the
/// entry.* field values), so whitelist processing only touches entries the
/// module bridged. New families only need to collect fields under the
/// entry.* prefix; no per-family key list is maintained here.
static llvm::StringSet<> collectUsedEntries(DictionaryAttr specAttr) {
  llvm::StringSet<> usedEntries;
  for (NamedAttribute attr : specAttr) {
    if (!attr.getName().getValue().starts_with("entry."))
      continue;
    if (auto value = dyn_cast<StringAttr>(attr.getValue()))
      usedEntries.insert(value.getValue());
  }
  return usedEntries;
}

/// Builds the typedef declarations of a custom-channel wrapper from the
/// tmpl_map rows of the entries the module uses, in whitelist order, and
/// validates that every declared template slot is covered by a token the
/// family pass collected into the spec. A target may be declared by several
/// entries; the first declaration wins and the merged spec guarantees the
/// tokens are identical.
static FailureOr<SmallVector<BridgeTypedefDecl>>
buildCustomTypedefDecls(ModuleOp module, const BridgeWhitelist &whitelist,
                        llvm::StringRef wrapper, DictionaryAttr specAttr,
                        const llvm::StringSet<> &usedEntries) {
  SmallVector<BridgeTypedefDecl> decls;
  (void)whitelist;
  (void)wrapper;
  (void)usedEntries;
  // Pipe wrapper typedefs are compiler-owned and come from structured
  // specialization fields, never from YAML tmpl_map declarations.
  for (auto item : {std::pair<StringRef, StringRef>(kBridgeSpecPipeKey, "Pipe"),
                    {kBridgeSpecProducerTileKey, "ProducerTile"},
                    {kBridgeSpecConsumerTileKey, "ConsumerTile"}}) {
    auto value = specAttr.getAs<StringAttr>(item.first);
    if (!value || value.getValue().empty()) {
      module.emitError() << "VPTO pipe bridge spec is missing structured field '"
                         << item.first << "'";
      return failure();
    }
    decls.push_back({item.second.str(), value.getValue().str()});
  }
  return decls;
}

struct VPTOBridgeWrapperGenPass final
    : public impl::VPTOBridgeWrapperGenBase<VPTOBridgeWrapperGenPass> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(VPTOBridgeWrapperGenPass)

  void runOnOperation() override {
    ModuleOp module = getOperation();

    // Merge the per-function specs into the module spec first: the family
    // pass instances may have run concurrently and only write their own
    // function attribute, so this single-threaded pass owns the module
    // attribute.
    if (failed(mergeFuncSpecsIntoModule(module))) {
      signalPassFailure();
      return;
    }

    auto specAttr = module->getAttrOfType<DictionaryAttr>(kBridgeSpecAttrName);
    if (!specAttr) {
      bool hasTypedCubeCall = false;
      BridgeObjectCreateOp object;
      module.walk([&](BridgeCallOp call) {
        if (auto id = call.getEntryId()) {
          hasTypedCubeCall =
              hasTypedCubeCall ||
              *id == getBridgeEntryName(BridgeEntryId::CubeTMatmul) ||
              *id == getBridgeEntryName(BridgeEntryId::CubeTMatmulMx);
        }
      });
      module.walk([&](BridgeObjectCreateOp candidate) {
        if (!object && candidate.getEntry() ==
                           getBridgeEntryName(BridgeEntryId::PipeInit)) {
          object = candidate;
        }
      });
      if (hasTypedCubeCall) {
        bool hasMx = false;
        module.walk([&](BridgeCallOp call) {
          hasMx = hasMx || call.getEntryId() ==
                              getBridgeEntryName(BridgeEntryId::CubeTMatmulMx);
        });
        FailureOr<std::string> cubeSource = renderTMatmulBridgeSource(module);
        if (failed(cubeSource) && !hasMx) {
          signalPassFailure();
          return;
        }
        std::string combinedCubeSource = succeeded(cubeSource) ? *cubeSource : "";
        if (hasMx) {
          FailureOr<std::string> mxSource = renderTMatmulMxBridgeSource(module);
          if (failed(mxSource)) {
            signalPassFailure();
            return;
          }
          if (!combinedCubeSource.empty()) combinedCubeSource += "\n";
          combinedCubeSource += *mxSource;
        }
        OpBuilder builder(module);
        module->setAttr(kBridgeWrapperSourceAttrName,
                        builder.getStringAttr(combinedCubeSource));
      }
      SmallVector<BridgeObjectCreateOp> objects;
      module.walk([&](BridgeObjectCreateOp candidate) {
        if (candidate.getEntry() == getBridgeEntryName(BridgeEntryId::PipeInit))
          objects.push_back(candidate);
      });
      if (!objects.empty()) {
        std::string combined;
        for (BridgeObjectCreateOp object : objects) {
          auto objectSpec = object.getSpecializationAttr();
          auto pipe = objectSpec ? objectSpec.getAs<DictionaryAttr>("pipe")
                                 : DictionaryAttr();
          auto flag = pipe ? pipe.getAs<IntegerAttr>("flag_base") : IntegerAttr();
          auto dir = pipe ? pipe.getAs<IntegerAttr>("direction") : IntegerAttr();
          auto slotSize = pipe ? pipe.getAs<IntegerAttr>("slot_size") : IntegerAttr();
          auto slotNum = pipe ? pipe.getAs<IntegerAttr>("slot_num") : IntegerAttr();
          auto localSlot = pipe ? pipe.getAs<IntegerAttr>("local_slot_num") : IntegerAttr();
          auto nosplit = pipe ? pipe.getAs<BoolAttr>("nosplit") : BoolAttr();
          if (!flag || !dir || !slotSize || !slotNum || !localSlot || !nosplit) {
            module.emitError("structured Pipe spec is incomplete");
            signalPassFailure();
            return;
          }
          std::string pipeToken = ("pto::TPipe<" + std::to_string(flag.getInt()) +
              (dir.getInt() == 1 ? ", pto::Direction::DIR_C2V, " : ", pto::Direction::DIR_V2C, ") +
              std::to_string(slotSize.getInt()) + ", " + std::to_string(slotNum.getInt()) + ", " +
              std::to_string(localSlot.getInt()) + ", " + (nosplit.getValue() ? "true>" : "false>"));
          BridgePipeSpec pipeSpec;
          pipeSpec.pipe = pipeToken;
          int64_t instance = object.getInstanceId().value_or(0);
          pipeSpec.entryInit = getBridgeInstanceSymbol(BridgeEntryId::PipeInit, instance);
          pipeSpec.entrySize = getBridgeInstanceSymbol(BridgeEntryId::PipeSize, instance);
          pipeSpec.entryPush = getBridgeInstanceSymbol(BridgeEntryId::PipePush, instance);
          pipeSpec.entryPop = getBridgeInstanceSymbol(BridgeEntryId::PipePop, instance);
          pipeSpec.entryFree = getBridgeInstanceSymbol(BridgeEntryId::PipeFree, instance);
          SmallVector<BridgeTypedefDecl> typedefs{{"Pipe", pipeToken}};
          llvm::StringSet<> names;
          names.insert("Pipe");
          module.walk([&](BridgeCallOp call) {
            auto cs = call.getSpecializationAttr();
            if (!cs || (pipe && cs.getAs<DictionaryAttr>("pipe") != pipe)) return;
            if (auto id = call.getEntryId()) {
              if (*id == getBridgeEntryName(BridgeEntryId::PipePush)) pipeSpec.entryPush = call.getCallee().str();
              if (*id == getBridgeEntryName(BridgeEntryId::PipePop)) pipeSpec.entryPop = call.getCallee().str();
              if (*id == getBridgeEntryName(BridgeEntryId::PipeFree)) pipeSpec.entryFree = call.getCallee().str();
            }
            for (auto item : {std::pair<StringRef, StringRef>("producer", "ProducerTile"), {"consumer", "ConsumerTile"}}) {
              auto group = cs.getAs<DictionaryAttr>(item.first);
              auto ta = group ? group.getAs<TypeAttr>("tile") : TypeAttr();
              auto tile = ta ? dyn_cast<TileBufType>(ta.getValue()) : TileBufType();
              if (tile) { auto tok = buildBridgeTileToken(tile); if (succeeded(tok) && names.insert(item.second).second) typedefs.push_back({item.second.str(), *tok}); }
            }
            if (auto split = cs.getAs<IntegerAttr>("split")) {
              auto tok = buildBridgeTileSplitToken(split.getInt()); if (succeeded(tok)) pipeSpec.split = *tok;
            }
          });
          auto source = renderPipeBridgeSource(pipeSpec, typedefs);
          if (failed(source)) { signalPassFailure(); return; }
          if (!combined.empty()) combined += "\n";
          combined += *source;
        }
        OpBuilder builder(module);
        if (auto existing = module->getAttrOfType<StringAttr>(kBridgeWrapperSourceAttrName))
          combined = existing.getValue().str() + "\n" + combined;
        module->setAttr(kBridgeWrapperSourceAttrName, builder.getStringAttr(combined));
      }
      return;
    }

    // Consume the whitelist: the tmpl_map declarations of the entries this
    // module uses must be covered by the collected specialization, and they
    // drive the wrapper typedef sections.
    FailureOr<BridgeWhitelist> whitelistOr =
        loadBridgeWhitelist(whitelistPath, llvm::errs());
    if (failed(whitelistOr)) {
      signalPassFailure();
      return;
    }
    llvm::StringSet<> usedEntries = collectUsedEntries(specAttr);

    // Which wrapper source to render is a whitelist fact, not something to
    // infer from which spec keys happen to be present: every entry the
    // module used names its wrapper, and entries sharing a wrapper render
    // into one translation unit. Deriving the set here means a newly
    // bridged declarative interface needs no edit to this pass at all, and
    // the diagnostics below name whatever wrappers the whitelist actually
    // declares.
    SmallVector<StringRef> usedWrappers;
    for (const BridgeWhitelistEntry &entry : whitelistOr->bridgeOps) {
      if (!usedEntries.count(entry.entry))
        continue;
      if (!llvm::is_contained(usedWrappers, StringRef(entry.wrapper)))
        usedWrappers.push_back(entry.wrapper);
    }
    if (usedWrappers.empty()) {
      module.emitError()
          << "VPTO bridge: the collected specialization names no wrapper "
             "entry declared in the bridge policy";
      signalPassFailure();
      return;
    }

    std::string combinedSource;
    for (StringRef usedWrapper : usedWrappers) {
      FailureOr<std::string> source = failure();
      if (whitelistOr->wrapperHasCustomEntry(usedWrapper)) {
        if (usedWrapper == kPipeWrapper) {
          FailureOr<SmallVector<BridgeTypedefDecl>> typedefsOr =
              buildCustomTypedefDecls(module, *whitelistOr, usedWrapper,
                                      specAttr, usedEntries);
          if (failed(typedefsOr)) {
            signalPassFailure();
            return;
          }
          SmallVector<BridgeTypedefDecl> typedefs = std::move(*typedefsOr);
          // The pipe entry bodies reference fixed typedef names, so every
          // name they use must be rendered by a tmpl_map declaration.
          llvm::StringSet<> declTargets;
          for (const BridgeTypedefDecl &decl : typedefs) {
            declTargets.insert(decl.target);
          }
          auto requireTypedefTargets = [&](ArrayRef<StringRef> targets) {
            bool ok = true;
            for (StringRef target : targets) {
              if (declTargets.count(target)) {
                continue;
              }
              module.emitError()
                  << "VPTO bridge: no tmpl_map row renders the '" << target
                  << "' typedef the wrapper entry bodies need; declare it in "
                     "the whitelist entry";
              ok = false;
            }
            return ok;
          };
          // All pipe spec fields are mandatory: a pipe bridge kernel always
          // carries the full init/size/push/pop/free entry set with a single
          // pipe and tile pair configuration.
          BridgePipeSpec spec;
          BridgeSpecField fields[] = {
              {kBridgeSpecPipeKey, &spec.pipe},
              {kBridgeSpecSplitKey, &spec.split},
              {kBridgeSpecEntryInitKey, &spec.entryInit},
              {kBridgeSpecEntrySizeKey, &spec.entrySize},
              {kBridgeSpecEntryPushKey, &spec.entryPush},
              {kBridgeSpecEntryPopKey, &spec.entryPop},
              {kBridgeSpecEntryFreeKey, &spec.entryFree},
          };
          bool ok = true;
          // The tile typedefs are rendered from the tmpl_map declarations;
          // the renderer only needs to know the tokens were collected.
          for (llvm::StringLiteral key :
               {kBridgeSpecProducerTileKey, kBridgeSpecConsumerTileKey}) {
            auto value = specAttr.getAs<StringAttr>(key);
            if (!value || value.getValue().empty()) {
              module.emitError()
                  << "VPTO pipe bridge spec is missing the '" << key
                  << "' field; the pipe family pass must collect it before "
                     "wrapper generation";
              ok = false;
              break;
            }
          }
          if (ok) {
            for (const auto &field : fields) {
              auto value = specAttr.getAs<StringAttr>(field.key);
              if (!value || value.getValue().empty()) {
                module.emitError()
                    << "VPTO pipe bridge spec is missing the '" << field.key
                    << "' field; the pipe family pass must collect it before "
                       "wrapper generation";
                ok = false;
                break;
              }
              *field.field = value.getValue();
            }
          }
          if (ok) {
            ok =
                requireTypedefTargets({"Pipe", "ProducerTile", "ConsumerTile"});
          }
          if (ok) {
            source = renderPipeBridgeSource(spec, typedefs);
            if (succeeded(source)) {
              SmallVector<std::pair<std::string, std::string>> concrete;
              module.walk([&](BridgeCallOp call) {
                auto id = call.getEntryId();
                const BridgeFunctionDesc *desc =
                    id ? lookupBridgeEntryForName(*id) : nullptr;
                if (desc && desc->family == BridgeFamily::Pipe &&
                    !call.getCallee().empty()) {
                  concrete.push_back(
                      {desc->symbol.str(), call.getCallee().str()});
                }
              });
              module.walk([&](BridgeObjectCreateOp object) {
                auto instanceId = object.getInstanceId();
                const BridgeFunctionDesc *init =
                    lookupBridgeEntryForName(object.getEntry());
                if (!instanceId || !init || !init->storageSizeEntry) {
                  return;
                }
                const BridgeFunctionDesc *size =
                    lookupBridgeEntry(*init->storageSizeEntry);
                concrete.push_back(
                    {init->symbol.str(),
                     getBridgeInstanceSymbol(init->id, *instanceId)});
                concrete.push_back(
                    {size->symbol.str(),
                     getBridgeInstanceSymbol(size->id, *instanceId)});
              });
              llvm::sort(concrete, [](const auto &lhs, const auto &rhs) {
                return lhs.first.size() > rhs.first.size();
              });
              for (const auto &replacement : concrete) {
                size_t position = 0;
                while ((position = source->find(replacement.first, position)) !=
                       std::string::npos) {
                  size_t suffix = position + replacement.first.size();
                  if (suffix < source->size() && (*source)[suffix] == '_') {
                    position = suffix;
                    continue;
                  }
                  source->replace(position, replacement.first.size(),
                                  replacement.second);
                  position += replacement.second.size();
                }
              }
            }
            if (failed(source)) {
              module.emitError()
                  << "VPTO pipe bridge: cannot render the wrapper source; the "
                     "pipe token must carry a C2V or V2C direction";
            }
          }
        } else {
          // The whitelist routed these ops into a custom wrapper this pass
          // has no dedicated renderer for. Say so instead of falling through
          // to the generic renderer, which cannot express family semantics.
          module.emitError()
              << "VPTO bridge: whitelist entries name the custom wrapper '"
              << usedWrapper
              << "', which has no dedicated renderer in the bridge wrapper "
                 "generator (available: "
              << kPipeWrapper << ")";
        }
      } else {
        if (usedWrapper == "cube") {
          source = renderTMatmulBridgeSource(module);
          bool hasMx = false;
          module.walk([&](BridgeCallOp call) {
            hasMx = hasMx || call.getEntryId() ==
                getBridgeEntryName(BridgeEntryId::CubeTMatmulMx);
          });
          if (hasMx) {
            FailureOr<std::string> mx = renderTMatmulMxBridgeSource(module);
            if (failed(mx)) source = failure();
            else {
              if (failed(source)) source = std::string();
              if (!source->empty()) *source += "\n";
              *source += *mx;
            }
          }
          if (failed(source)) {
            module.emitError() << "VPTO Cube bridge wrapper rendering failed";
            signalPassFailure();
            return;
          }
          combinedSource.append(*source);
          combinedSource.push_back('\n');
          continue;
        }
        const BridgeWrapperDecl *decl = whitelistOr->findWrapper(usedWrapper);
        if (!decl) {
          module.emitError()
              << "VPTO bridge: wrapper '" << usedWrapper
              << "' routes declarative entries but has no declaration in the "
                 "whitelist wrappers section (declare its name, includes and "
                 "core there)";
        } else {
          source = renderDeclarativeBridgeSource(module, *whitelistOr, *decl,
                                                 specAttr, usedEntries);
        }
      }
      if (failed(source)) {
        signalPassFailure();
        return;
      }
      combinedSource.append(*source);
      combinedSource.push_back('\n');
    }

    OpBuilder builder(module);
    module->setAttr(kBridgeWrapperSourceAttrName,
                    builder.getStringAttr(combinedSource));
    module->removeAttr(kBridgeSpecAttrName);
  }
};

} // namespace

std::unique_ptr<Pass> createVPTOBridgeWrapperGenPass() {
  return std::make_unique<VPTOBridgeWrapperGenPass>();
}

} // namespace pto
} // namespace mlir
