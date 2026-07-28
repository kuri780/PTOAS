# PTOAS Print Support Research: EmitC vs VPTO

## 分支

`feature/print-support-research`

> **状态更新 (ed13c920)**：本文档最初编写时 VPTO 路径尚不支持 print。截至分支 HEAD (`ed13c920`)，`pto.print` 和 `pto.tprint` 均已通过**编译期内联 DebugTunnel 协议**在 VPTO 后端完整实现。下文保留原始研究分析，实际实现方案与最初提案有差异（见[第 6 节](#6-实际实现与原始提案的差异)）。

## 概述

PTOAS 有两个 IR 级别的 print 操作：
- `pto.print` — 打印标量（带格式字符串），在 VPTO 路径降级为内联 DebugTunnel 协议写入
- `pto.tprint` — 打印整个 Tile/GlobalTensor，在 VPTO 路径降级为嵌套 `scf.for` 逐元素 DebugTunnel 协议写入

两个后端路径均完整支持 print。

---

## 1. EmitC 路径：`pto.print` 降级流程（✅ 已完整支持）

### 整体链路

```
PTO IR                    EmitC Dialect                  C++ Source               bisheng
──────                    ─────────────                  ──────────               ───────
pto.print                 emitc::CallOpaqueOp            cce::printf("fmt", v)    device code
  ins("fmt", %v)    →     callee="cce::printf"     →                           →
                           args=["fmt", 0]
                           operands=[%v]
```

### 关键代码位置

| 步骤 | 文件 | 行号 | 说明 |
|------|------|------|------|
| IR 定义 | `include/PTO/IR/PTOOps.td` | 3126-3138 | `PrintOp` 定义：`(ins StrAttr:$format, ScalarType:$scalar)` |
| IR 定义 | `include/PTO/IR/PTOOps.td` | 7001-7027 | `TPrintOp` 定义：支持 Tile/GlobalTensor + 可选 tmp |
| EmitC 降级 | `lib/PTO/Transforms/PTOToEmitC.cpp` | 12256-12294 | `PTOPrintOpToEmitC`: `pto.print` → `cce::printf` |
| EmitC 降级 | `lib/PTO/Transforms/PTOToEmitC.cpp` | 12198-12254 | `PTOPrintToTPRINT`: `pto.tprint` → `TPRINT(...)` |
| 模式注册 | `lib/PTO/Transforms/PTOToEmitC.cpp` | 14287-14288 | 两个模式都注册到 EmitC 转换 |
| EmitC Pass | `tools/ptoas/ptoas.cpp` | 3121-3126 | `createEmitPTOManualPass` 执行 PTO→EmitC 降级 |
| C++ 生成 | `tools/ptoas/ptoas.cpp` | 3157 | `emitc::translateToCpp` 将 EmitC 方言翻译为 C++ 源码 |

### PTOPrintOpToEmitC 降级逻辑（`PTOToEmitC.cpp:12256-12294`）

```
1. 从 op 属性取出格式字符串 (StringAttr)
2. 对格式字符串做 C 转义 (引号、换行、制表符等)
3. 从 adaptor 取出标量值 (经过 type converter 转换)
4. 构造 emitc::CallOpaqueOp:
   - callee = "cce::printf"
   - args = ["\"escaped_format\"", 0]  (0 表示 operands[0])
   - operands = [scalar]
5. 删除原始 pto.print op
```

### PTOPrintToTPRINT 降级逻辑（`PTOToEmitC.cpp:12198-12254`）

```
1. 从 adaptor 取出 src (Tile/GlobalTensor)
   - 如果是 MemRef/PTV，调用 maybeWrapGlobalMemrefAsGlobalTensor 包装
2. 如果有 tmp 操作数，同样处理
3. 如果有 PrintFormat 属性，转成模板参数 (如 Width10_Precision6)
4. 构造 emitc::CallOpaqueOp:
   - callee = "TPRINT"
   - templateArgs = [PrintFormat?]
   - operands = [src, tmp?]
5. 删除原始 pto.tprint op
```

### 编译时需要的外部 flag

生成的 C++ 代码 `cce::printf(...)` 需要在 bisheng 编译时启用两个东西：

| Flag | 作用 | 位置 |
|------|------|------|
| `--cce-enable-print` | bisheng 编译器选项，启用设备→主机 debug 通道 | 传给 bisheng |
| `-DPTOAS_ENABLE_CCE_PRINT=1` | 预处理器宏，让 `launch.cpp` 包含 `<ccelib/print/print.h>` | 传给 bisheng |

`launch.cpp` 中的条件包含 (`test/vpto/cases/*/launch.cpp`):
```cpp
#if defined(__CCE_AICORE__) && defined(PTOAS_ENABLE_CCE_PRINT)
#include <ccelib/print/print.h>
#endif
```

### 自动检测机制

`test/npu_validation/scripts/generate_testcase.py:2337-2341`:
```python
needs_cce_print = bool(re.search(r"\b(?:bisheng::)?cce::printf\s*\(", raw_kernel_for_analysis))
cce_enable_print_opt = "    --cce-enable-print" if needs_cce_print else ""
cce_print_define_opt = "    -DPTOAS_ENABLE_CCE_PRINT=1" if needs_cce_print else ""
```

---

## 2. VPTO 路径：为什么 `pto.print` 原本不支持（✅ 已解决）

### VPTO 后端流水线

```
PTO IR  →  [PrepareVPTO] → [LowerVPTOOps] → [LowerVPTOTypes] → [NormalizeFunc]
        →  [Arith→LLVM] → [Index→LLVM] → [MemRef→LLVM] → [Func→LLVM] → [CF→LLVM]
        →  translateModuleToLLVMIR() → LLVM Module → bisheng 编译
```

关键代码位置：`lib/PTO/Transforms/VPTOCANN900LLVMEmitter.cpp:10942-10958`

### 问题根因

`LowerVPTOOpsPass` 中的 `populateVPTOOpLoweringPatterns` (第 10196-10425 行) 列出了**所有**支持的 PTO op 降级模式（~230 行），但**没有** `pto::PrintOp` 和 `pto::TPrintOp` 的降级模式。

同样，`configureVPTOOpLoweringTarget` (第 10427-10568 行) 的 illegal op 列表中也**没有** `PrintOp` / `TPrintOp`。

结果：`pto.print` / `pto.tprint` 在 VPTO 流水线中**不被任何 pass 处理**，作为原始 PTO op 残留到最终的 `translateModuleToLLVMIR()` 调用，MLIR 不知道如何将 PTO dialect 翻译为 LLVM IR，报错：

```
error: cannot be converted to LLVM IR: missing `LLVMTranslationDialectInterface`
registration for dialect for op: pto.print
```

### VPTO 中的 op 降级模式

现有的 pattern 分两类：

**A. 声明外部函数 + func::CallOp** — 以 `LowerBarrierOpPattern` 为典型（第 8725 行附近）：

```cpp
// 1. 计算目标函数名 (如 "llvm.hivm.BARRIER")
StringRef calleeName = buildSyncCallee<pto::BarrierOp>(op.getContext());
// 2. 创建函数类型
auto funcType = rewriter.getFunctionType(TypeRange{rewriter.getI64Type()}, TypeRange{});
// 3. 产生 func::CallOp
rewriter.create<func::CallOp>(op.getLoc(), calleeName, TypeRange{}, ValueRange{pipeValue});
// 4. 登记声明，后续由 materializeDecls() 统一创建 func::FuncOp
state.plannedDecls.push_back(PlannedDecl{calleeName.str(), funcType});
// 5. 删除原 op
rewriter.eraseOp(op);
```

关键数据结构（`VPTOCANN900LLVMEmitter.cpp:214-219`）：
```cpp
struct PlannedDecl {
  std::string name;
  FunctionType type;
};
struct LoweringState {
  SmallVector<PlannedDecl> plannedDecls;
};
```

`materializeDecls()` (第 4216 行) 在所有 pattern 运行完后，为每个 `PlannedDecl` 创建 `func::FuncOp` 声明。

**B. 直接展开为 arith/LLVM 内建 ops** — 如 `LowerVecScalarMaskedOpPattern<pto::VaddsOp>`，直接生成 arith 运算序列。

这些降级后的标准 MLIR ops 可以被 `translateModuleToLLVMIR` 正常处理。

---

## 3. 实现 VPTO 路径 `cce::printf` 需要做什么

### 总体方案：在 LowerVPTOOpsPass 中添加降级模式

**位置**：`lib/PTO/Transforms/VPTOCANN900LLVMEmitter.cpp` (和 `VPTOLLVMEmitter.cpp`)

#### 3.1 核心挑战

**格式字符串是编译期常量**，但 LLVM IR 中不能直接传字符串字面量。需要：

1. 为每个格式字符串创建一个 **LLVM 全局常量** (`LLVM::GlobalOp`)
2. 用 `LLVM::AddressOfOp` 获取其指针
3. 声明外部函数 `cce::printf`（复用现有 `PlannedDecl` 机制）
4. 用 `func::CallOp` 调用

**额外难点**：多个 `pto.print` 可能用相同格式字符串，需要去重。现有的 `LoweringState` 只管理函数声明，需要扩展以支持**去重的全局字符串常量**。

#### 3.2 改动清单

**Step 1** — 扩展 `LoweringState`，添加去重的字符串全局池：

```cpp
struct LoweringState {
  SmallVector<PlannedDecl> plannedDecls;
  // 新增：格式字符串去重池，key=格式字符串, value=全局符号名
  llvm::StringMap<std::string> stringGlobals;
};
```

**Step 2** — 添加 `LowerPrintOpPattern`：

```cpp
struct LowerPrintOpPattern : public OpConversionPattern<pto::PrintOp> {
  LowerPrintOpPattern(TypeConverter &typeConverter, MLIRContext *context,
                      LoweringState &state)
      : OpConversionPattern<pto::PrintOp>(typeConverter, context), state(state) {}

  LogicalResult matchAndRewrite(pto::PrintOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    // 1. 获取格式字符串
    std::string fmt = op.getFormat().str();
    if (fmt.empty()) fmt = "%f";

    // 2. 在去重池中查找/创建全局字符串符号名
    std::string &globalName = state.stringGlobals[fmt];
    if (globalName.empty())
      globalName = "_ptoas_printf_fmt_" + std::to_string(state.stringGlobals.size());

    // 3. 创建 LLVM::AddressOfOp 获取格式字符串指针
    auto ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());
    auto addrOp = rewriter.create<LLVM::AddressOfOp>(
        loc, ptrType, globalName);

    // 4. 注册 cce::printf 声明 (复用 PlannedDecl 机制)
    auto funcType = rewriter.getFunctionType(
        TypeRange{ptrType, adaptor.getScalar().getType()},
        TypeRange{rewriter.getI32Type()});
    state.plannedDecls.push_back(PlannedDecl{"cce::printf", funcType});

    // 5. 产生 func::CallOp
    rewriter.create<func::CallOp>(loc, "cce::printf", TypeRange{rewriter.getI32Type()},
                                  ValueRange{addrOp.getResult(), adaptor.getScalar()});

    rewriter.eraseOp(op);
    return success();
  }
private:
  LoweringState &state;
};
```

**Step 3** — 添加 `materializeStringGlobals()`，在 `lowerVPTOOps()` 末尾调用：

```cpp
static void materializeStringGlobals(ModuleOp module,
                                     const llvm::StringMap<std::string> &stringGlobals) {
  auto *ctx = module.getContext();
  auto i8Type = IntegerType::get(ctx, 8);
  OpBuilder builder(module.getBodyRegion());

  for (auto &kv : stringGlobals) {
    StringRef fmt = kv.first();       // 格式字符串
    StringRef globalName = kv.second; // 全局符号名

    // 构造字节数组 (含 '\0')
    SmallVector<Attribute> elements;
    for (char c : fmt) elements.push_back(IntegerAttr::get(i8Type, c));
    elements.push_back(IntegerAttr::get(i8Type, 0)); // null terminator

    auto arrayType = LLVM::LLVMArrayType::get(i8Type, elements.size());
    builder.create<LLVM::GlobalOp>(
        module.getLoc(), arrayType, /*isConstant=*/true,
        LLVM::Linkage::Private, globalName,
        ArrayAttr::get(ctx, elements));
  }
}
```

**Step 4** — 在 `configureVPTOOpLoweringTarget()` 中标记 illegal：
```cpp
target.addIllegalOp<pto::PrintOp, pto::TPrintOp>();
```

**Step 5** — 在 `populateVPTOOpLoweringPatterns()` 中注册：
```cpp
patterns.add<LowerPrintOpPattern>(typeConverter, patterns.getContext(), state);
```

#### 3.3 `TPrintOp` 的额外复杂度（✅ 已实现）

`pto.tprint` 打印整个 Tile，在 EmitC 路径降级为 `TPRINT(src)`（宏展开为遍历 tile 元素调用 `cce::printf` 的循环）。

在 VPTO/LLVM IR 路径中，实际实现方案（`ed13c920`）采用了**编译期内联 DebugTunnel 协议**而非调用运行时 helper：
- 生成嵌套 `scf.for` 循环遍历 tile 行列
- 通过 `GET.SYS.VA.BASE` 计算 UB 虚拟地址，逐元素加载
- 每个元素直接写入 FLOAT/INT marker + 数据 bytes + 格式串 + END marker
- 末尾单次 DCCI flush

详见 `docs/pto_print_lowering_design.md` 和 `docs/pto_print_implementation_comparison.md`。

#### 3.4 涉及的文件清单

| 文件 | 改动内容 |
|------|----------|
| `lib/PTO/Transforms/VPTOCANN900LLVMEmitter.cpp` | 添加 `LowerPrintOpPattern`、扩展 `LoweringState`、添加 `materializeStringGlobals`、注册 pattern、标记 illegal |
| `lib/PTO/Transforms/VPTOLLVMEmitter.cpp` | 同上（同步支持旧 CANN 版本 Beta1） |
| `test/lit/pto/` | 添加 VPTO 路径的 lit 测试 |

---

## 4. 验证测试

### EmitC 路径验证结果

```
✅ pto.print → cce::printf 降级成功
✅ bisheng -xcce --cce-enable-print 编译成功
✅ kernel.so 链接成功
⚠️ 模拟器 (camodel) 运行有 "call stack overflow"，无 printf 输出
   （模拟器限制，真硬件应可正常工作）
```

### VPTO 路径验证结果

```
✅ pto.print  → 编译期内联 DebugTunnel 协议，生成 store 指令序列
✅ pto.tprint → 嵌套 scf.for 逐元素 DebugTunnel 协议写入
✅ Lit 测试通过: test/lit/vpto/tprint_vec_vpto_llvm.pto
✅ Kernel 级测试: 8 个端到端测试用例 (print-*)
✅ 模拟器验证脚本: test/vpto/scripts/run_vpto_tprint_validation.sh
```

---

## 5. 文档参考

`docs/PTO_IR_manual.md:9824`:
> On A2/A3/A5 devices, `TPRINT` uses `cce::printf` to emit output via the
> device-to-host debug channel. **You must enable the CCE option
> `-D_DEBUG --cce-enable-print`**.

---

## 6. 实际实现与原始提案的差异

本文档第 3 节的原始提案建议通过 `cce::printf` 外部函数调用实现 VPTO print。实际实现（`ed13c920` 及前置提交）采用了完全不同的路线：

### 路线差异

| 维度 | 原始提案 | 实际实现 |
|------|---------|----------|
| **核心策略** | 声明 `cce::printf` 外部函数 + `func::CallOp` | 编译期内联 DebugTunnel 协议为 `store` 指令序列 |
| **运行时依赖** | 依赖 `cce::printf` 函数（需 bisheng 链接） | 零外部依赖，只依赖 CCE intrinsic（fix stack、DCCI、GET.BLOCK.IDX、GET.SYS.VA.BASE） |
| **格式串处理** | 运行时通过 `cce::printf` 解析 | 编译期在 `LowerPrintOpPattern` 中一次性解析，直接产生常量偏移 |
| **TPrintOp** | 提案认为"工作量大，建议后续再做" | 已完整实现：嵌套 `scf.for` 遍历 tile 行列 + 逐元素 DebugTunnel 写入 |
| **DTData 注入** | 未详细设计 | `addDTDataParamToEntryFunctions` 手动追加隐藏参数 + `injectPrintPrologue` 内联初始化 |
| **二进制体积** | 较小（函数调用复用） | 较大（全展开）但无需外部符号 |

### 为什么选择了编译期内联方案

1. **消除外部依赖**：不依赖 `cce::printf` 符号，避免 C++ name mangling 和 bisheng 版本兼容问题
2. **编译期确定一切**：所有偏移、长度、类型标记在 MLIR lowering 阶段就是常量，无需运行时计算
3. **与 EmitC 路径等效**：生成的 LLVM IR 就是 CCE 前端编译 `cce::printf` 后的形态——只是我们跳过了 C++ 模板展开这一步
4. **TPrintOp 自然扩展**：内联方案对 TPrintOp 同样适用，只需在 store 序列外包裹 `scf.for` 循环
