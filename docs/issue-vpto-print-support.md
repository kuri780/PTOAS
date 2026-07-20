# [Feature Request] VPTO 后端支持 `pto.print` / `pto.tprint` 设备端打印

## 背景

PTOAS 有两个后端：

| 后端 | 路径 | 用途 |
|------|------|------|
| **EmitC** | PTO IR → EmitC 方言 → C++ 源码 → bisheng 编译 | 默认后端，生成 C++ 代码 |
| **VPTO** | PTO IR → LLVM IR → 目标文件 | 直接生成 LLVM IR，绕过 C++ |

PTO IR 层已经有两个调试打印 op，但**只在 EmitC 后端可用**：

| Op | 功能 | IR 写法 |
|----|------|---------|
| `pto.print` | 打印标量（带格式字符串） | `pto.print ins("%f", %val : f32)` |
| `pto.tprint` | 打印 Tile/GlobalTensor 全部元素 | `pto.tprint ins(%tile)` |

## 问题

在 VPTO 后端使用这两个 op 会直接崩溃：

```
error: cannot be converted to LLVM IR: missing `LLVMTranslationDialectInterface`
registration for dialect for op: pto.print
```

**根因**：VPTO 后端的流水线是 `LowerVPTOOpsPass` → `LowerVPTOTypesPass` → MLIR→LLVM IR 标准转换。`LowerVPTOOpsPass` 中 `populateVPTOOpLoweringPatterns` 列出了 ~230 行 pattern 覆盖几乎所有 PTO op，但**唯独没有 `PrintOp` 和 `TPrintOp` 的降级 pattern**。未被处理的 op 残留到 `translateModuleToLLVMIR()`，MLIR 不认识 PTO 方言，报错。

## 需求

### 1. `pto.print` — 标量打印（优先级：高）

对标 EmitC 路径的 `cce::printf("fmt", val)`。

EmitC 路径生成的 C++：
```cpp
cce::printf("value = %f\n", scalar);
```

VPTO 路径需要生成等效的 LLVM IR：
```llvm
@.fmt = private constant [13 x i8] c"value = %f\0A\00"
declare i32 @"cce::printf"(ptr, ...)
%fmt = getelementptr ... @.fmt ...
%result = call i32 @"cce::printf"(ptr %fmt, float %scalar)
```

**关键挑战**：
- 格式字符串是编译期常量，LLVM IR 中需要创建 `LLVM::GlobalOp` 全局常量
- 多个 `pto.print` 可能用相同格式字符串，需要去重
- 外部函数 `cce::printf` 需要声明（可复用已有 `PlannedDecl` 机制）

### 2. `pto.tprint` — Tile 打印（优先级：中）

对标 EmitC 路径的 `TPRINT(...)` 宏（展开为逐元素 `cce::printf` 循环）。

EmitC 路径生成的 C++：
```cpp
TPRINT<pto::PrintFormat::Width8_Precision4>(tile, tmp);
```

**关键挑战**：
- `TPrintOp` 的 operand 是 `TileBufType`（如 `!pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16, ...>`）
- `ExpandTileOp` 不处理纯 print 用途的 tile（只处理被计算 op 引用的 tile）
- `VPTOTypeConverter` 无法转换 `TileBufType`
- 需要在 MLIR 层生成元素遍历循环 + 逐元素 `cce::printf` 调用，或调用运行时 helper

## 技术方案要点

### `pto.print` 实现思路

在 `LowerVPTOOpsPass` 中添加 `LowerPrintOpPattern`，参考已有 pattern（如 `LowerRuntimeQueryOpPattern`）的模式：

1. 预扫描模块，为所有唯一格式字符串创建 `LLVM::GlobalOp`
2. Pattern 中创建 `LLVM::AddressOfOp` 获取格式字符串指针
3. 通过 `PlannedDecl` 机制声明外部函数 `cce::printf`
4. 创建 `func::CallOp @cce::printf(fmt_ptr, scalar)`

### `pto.tprint` 实现思路

有几种方案可讨论：

- **方案 A**：扩展 `ExpandTileOp`，对纯 print 用途的 tile 也展开为标量 ops，后续由已有 pattern 处理
- **方案 B**：在 `LowerTPrintOpPattern` 中直接生成 `scf.for` 循环 + `cce::printf` 调用
- **方案 C**：调用运行时 helper 函数（需 pto-isa 运行时提供 `TPRINT` 的 LLVM 可调用版本，而非 C++ 宏）

## 涉及文件（预估）

| 文件 | 改动 |
|------|------|
| `lib/PTO/Transforms/VPTOCANN900LLVMEmitter.cpp` | 添加 `LowerPrintOpPattern`、`LowerTPrintOpPattern`、格式字符串预扫描、扩展 `LoweringState`、注册 pattern、标记 illegal |
| `lib/PTO/Transforms/VPTOLLVMEmitter.cpp` | 同上（旧 CANN 版本同步） |
| `lib/PTO/Transforms/ExpandTileOp.cpp` | （可选，方案 A）处理 TPrintOp 引用的 tile |

## 编译时依赖

无论 EmitC 还是 VPTO 路径，`cce::printf` 都需要 bisheng 编译时启用：

| Flag | 作用 |
|------|------|
| `--cce-enable-print` | bisheng 选项，启用设备→主机 debug 通道 |
| `-DPTOAS_ENABLE_CCE_PRINT=1` | 预处理器宏，让 `launch.cpp` 包含 `<ccelib/print/print.h>` |
