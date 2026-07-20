# [Feature] VPTO 后端支持 `pto.print` / `pto.tprint` 设备端打印

## 背景

PTOAS 有两个后端：

| 后端 | 路径 | 用途 |
|------|------|------|
| **EmitC** | PTO IR → EmitC 方言 → C++ 源码 → bisheng 编译 | 默认后端，生成 C++ 代码 |
| **VPTO** | PTO IR → LLVM IR → 目标文件 | 直接生成 LLVM IR，绕过 C++ |

PTO IR 层有两个调试打印 op：

| Op | 功能 | 示例 |
|----|------|------|
| `pto.print` | 打印标量（带格式字符串） | `pto.print ins("%f", %val : f32)` |
| `pto.tprint` | 打印 Tile/GlobalTensor 全部内容 | `pto.tprint ins(%tile : !pto.tile_buf<...>)` |

**问题**：这两个 op 此前**只在 EmitC 后端有 lowering**，VPTO 后端遇到它们直接崩溃：

```
error: cannot be converted to LLVM IR: missing `LLVMTranslationDialectInterface`
registration for dialect for op: pto.print
```

## 当前实现状态

分支：`feature/print-support-research`

| 功能 | EmitC | VPTO（改前） | VPTO（改后） |
|------|-------|-------------|-------------|
| `pto.print` | ✅ `cce::printf("fmt", val)` | ❌ crash | ✅ `cce::printf` via LLVM IR |
| `pto.tprint` | ✅ `TPRINT(...)` 宏（逐元素） | ❌ crash | ⚠️ debug marker（桩） |

### `pto.print`（已完整实现）

**实现方式**：两阶段策略

1. **预扫描**（`collectAndCreatePrintfStringGlobals`）：在 dialect conversion 之前遍历所有 `pto::PrintOp`，收集格式字符串，为每个唯一字符串创建 `LLVM::GlobalOp`
2. **LowerPrintOpPattern**：将 `pto.print` 降级为 `LLVM::AddressOfOp` + `func::CallOp @cce::printf(fmt_ptr, scalar)`

**LLVM IR 生成效果**：
```llvm
@_ptoas_printf_fmt_0 = private constant [8 x i8] c"cst=%d\0A\00"
declare i32 @"cce::printf"(ptr, i8)
%2 = call i32 @"cce::printf"(ptr @_ptoas_printf_fmt_0, i8 -7)
```

### `pto.tprint`（桩实现，待完善）

**为什么是桩**：`TPrintOp` 的 operands 是 `TileBufType`（如 `!pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16, ...>`）。这个类型经过 `VPTOTypeConverter` 无法转换，`ExpandTileOp` 也不处理纯 print 用途的 tile。要在 LLVM IR 中做逐元素打印，需要生成元素遍历循环 + N 次 `cce::printf` 调用。

当前实现降级为打印一个硬编码标记 `[tprint]\n`，**不再崩溃**，tile 数据内容被丢弃。

## 涉及的文件

| 文件 | 改动 |
|------|------|
| `lib/PTO/Transforms/VPTOCANN900LLVMEmitter.cpp` | 添加 `LowerPrintOpPattern`、`LowerTPrintOpPattern`（桩）、`collectAndCreatePrintfStringGlobals`、扩展 `LoweringState`、注册 pattern、标记 illegal |
| `lib/PTO/Transforms/VPTOLLVMEmitter.cpp` | 同上（旧 CANN 版本同步） |
| `docs/print_support_research.md` | 研究报告（含 EmitC 降级流程分析、VPTO 管线分析、实现方案设计） |

## 剩余工作

### 1. `pto.tprint` 完整逐元素打印

需要解决 tile 类型转换问题。可能的方案：

- **方案 A**：在 `ExpandTileOp` 中添加 TPrintOp 处理，把纯 print 用途的 tile 也展开为标量操作
- **方案 B**：在 `LowerTPrintOpPattern` 中直接对已转换的 tile buffer 生成逐元素打印循环（`scf.for` + `cce::printf`）
- **方案 C**：调用运行时 helper 函数（需要 pto-isa 运行时提供 `TPRINT` 的 LLVM 可调用版本）

### 2. 端到端验证

需要在真实 NPU 硬件或模拟器上验证 `cce::printf` 输出。当前模拟器 (`camodel`) 有 `call stack overflow` 问题，可能是模拟器对 `cce::printf` 支持不完善。

## 编译时需要的 flag

无论 EmitC 还是 VPTO 路径，生成的 LLVM IR / C++ 中的 `cce::printf` 调用都需要 bisheng 编译时启用：

| Flag | 作用 |
|------|------|
| `--cce-enable-print` | bisheng 选项，启用设备→主机 debug 通道 |
| `-DPTOAS_ENABLE_CCE_PRINT=1` | 预处理器宏，让 `launch.cpp` 包含 `<ccelib/print/print.h>` |

## 参考

- 研究报告：`docs/print_support_research.md`
- 分支：`feature/print-support-research`
- 文档：`docs/PTO_IR_manual.md` 第 9824 行关于 TPRINT 的说明
