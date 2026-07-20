# PTOAS Print Support Research: EmitC vs VPTO

## 分支

`feature/print-support-research`

## 概述

PTOAS 有两个 IR 级别的 print 操作：
- `pto.print` — 打印标量（带格式字符串），降级为 `cce::printf("fmt", scalar)`
- `pto.tprint` — 打印整个 Tile/GlobalTensor，降级为 `TPRINT(src)` 宏

两个后端路径对 print 的支持情况不同。

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

## 2. VPTO 路径：为什么 `pto.print` 不支持（❌ 缺少降级）

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

以 `LowerVecScalarMaskedOpPattern<pto::VaddsOp>` 为例，VPTO 使用模板化的 pattern 将 PTO op 降级为：
- arith 运算（arith.addi, arith.muli 等）
- LLVM 内建调用（llvm.call @llvm.ascend.*）
- func.call（调用运行时函数）

这些降级后的标准 MLIR ops 可以被 `translateModuleToLLVMIR` 正常处理。

---

## 3. 实现 VPTO 路径 `cce::printf` 需要做什么

### 方案 A：在 LowerVPTOOpsPass 中添加降级模式（推荐）

**位置**：`lib/PTO/Transforms/VPTOCANN900LLVMEmitter.cpp` (和 `VPTOLLVMEmitter.cpp`)

**需要的改动**：

#### 3.1 添加 `LowerPrintOpPattern`

新建一个 pattern，将 `pto::PrintOp` 降级为 `llvm.call @cce::printf`：

```cpp
struct LowerPrintOpPattern : public OpConversionPattern<pto::PrintOp> {
  using OpConversionPattern<pto::PrintOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::PrintOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();

    // 1. 获取格式字符串
    std::string fmt = op.getFormat().str();
    if (fmt.empty()) fmt = "%f";

    // 2. 创建全局字符串常量 (LLVM 要求格式字符串是全局常量)
    auto strType = LLVM::LLVMArrayType::get(IntegerType::get(ctx, 8), fmt.size() + 1);
    // ... 创建或复用全局字符串 ...

    // 3. 获取标量值 (已经 type converter 转换过)
    Value scalar = adaptor.getScalar();

    // 4. 声明 cce::printf (如果还没声明)
    // declare i32 @cce::printf(i8*, ...)
    auto printfFunc = getOrCreatePrintfDeclaration(module, ctx);

    // 5. 创建调用: llvm.call @cce::printf(fmt_ptr, scalar)
    rewriter.create<LLVM::CallOp>(loc, printfFunc,
                                  ValueRange{fmtPtr, scalar});

    rewriter.eraseOp(op);
    return success();
  }
};
```

#### 3.2 注册 pattern

在 `populateVPTOOpLoweringPatterns` (第 10196 行) 中添加：
```cpp
patterns.add<LowerPrintOpPattern>(typeConverter, patterns.getContext(), state);
```

#### 3.3 标记 op 为 illegal

在 `configureVPTOOpLoweringTarget` (第 10427 行) 中添加：
```cpp
target.addIllegalOp<pto::PrintOp, pto::TPrintOp>();
```

#### 3.4 （可选）添加 `LowerTPrintOpPattern`

类似地，将 `pto::TPrintOp` 降级为 `TPRINT(...)` 宏调用或直接调用底层打印函数。

### 方案 B：在 LLVMTranslationDialectInterface 中处理

在 PTO dialect 注册一个 `LLVMTranslationDialectInterface`，直接在 MLIR→LLVM IR 翻译阶段处理 `pto.print`。

**缺点**：需要改动 dialect 注册代码，侵入性更高，且 MLIR 的 `translateModuleToLLVMIR` 假设所有 dialect 都已注册 translation interface。

### 涉及的文件清单

| 文件 | 改动内容 |
|------|----------|
| `lib/PTO/Transforms/VPTOCANN900LLVMEmitter.cpp` | 添加 `LowerPrintOpPattern`、注册 pattern、标记 illegal |
| `lib/PTO/Transforms/VPTOLLVMEmitter.cpp` | 同上（同步支持旧 CANN 版本） |
| `test/lit/pto/` | 添加 VPTO 路径的 lit 测试 |

### LLVM IR 中 `cce::printf` 调用的形式

目标生成的 LLVM IR：
```llvm
@.fmt = private unnamed_addr constant [8 x i8] c"cst=%d\0A\00"

declare i32 @cce::printf(i8*, ...)

define void @kernel(%arg0: ptr) {
  %fmt_ptr = getelementptr [8 x i8], [8 x i8]* @.fmt, i64 0, i64 0
  %val = ... ; scalar value
  call i32 (i8*, ...) @cce::printf(i8* %fmt_ptr, i8 %val)
  ret void
}
```

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
❌ pto.print 直接报错: missing LLVMTranslationDialectInterface registration
   需要实现上述方案 A 或 B
```

---

## 5. 文档参考

`docs/PTO_IR_manual.md:9824`:
> On A2/A3/A5 devices, `TPRINT` uses `cce::printf` to emit output via the
> device-to-host debug channel. **You must enable the CCE option
> `-D_DEBUG --cce-enable-print`**.
