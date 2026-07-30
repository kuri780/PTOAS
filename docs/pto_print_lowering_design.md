# PTO Print Lowering 设计与实现

## 1. 设计目标

在一个 kernel 里插入 `pto.print` 和 `pto.tprint` 打印调试信息，和其他 PTO 操作（向量计算、内存搬运、多 block 并行、控制流）完全共存，不破坏任何现有编译管线。

- **`pto.print`**：打印标量值（带格式字符串），降级为编译期内联 DebugTunnel 协议
- **`pto.tprint`**：打印整个 Tile 的全部元素，降级为嵌套 `scf.for` 循环 + 逐元素 DebugTunnel 协议

## 2. 核心思路：编译期内联 DebugTunnel 协议

**不调用任何运行时函数**（不依赖 `printf`、不依赖 C 标准库）。编译器在编译期就把要打印的内容展开成一段固定的 `store i8` 指令序列，直接写入硬件的 log buffer。

### 2.1 类比：快递员 vs 自己塞信箱

- **传统 `printf`**：像是打电话给快递公司说"帮我送个包裹"，快递公司有自己的内部流程、单号格式、派送规则——运行时还得解析格式串、做格式化
- **我们的做法**：像是出门前就把明信片的内容全部抄好，走到信箱前直接塞进去——没有中间环节，所有信息在编译期就定死了

### 2.2 LLVM IR 形态

```llvm
; pto.print ins("x = %+08.3f\n", 3.14)
; 编译后变成：

store i8 1,    ptr %writePtr+0        ; NORMAL marker（普通文本节点）
store i16 5,   ptr %writePtr+1        ; "x = \0" 的长度
; 逐字节复制前缀 "x = \0"
store i8 2,    ptr %writePtr+8        ; FLOAT marker = 2
; 逐字节写入 3.14 的 f32 IEEE 754 表示
store i16 8,   ptr %writePtr+13       ; "%+08.3f\0" 的长度
; 逐字节复制转换格式 "%+08.3f\0"
store i8 1,    ptr %writePtr+23       ; NORMAL marker（后缀文本节点）
store i16 2,   ptr %writePtr+24       ; "\n\0" 的长度
; 逐字节复制后缀 "\n\0"
store i8 0,    ptr %writePtr+...      ; END marker
; 更新已写字节计数
; DCCI flush — 通知 Host 端来取
```

全是 `store` 指令，没有 `call`（除了最后 flush 那条 intrinsic）。

## 3. DebugTunnel 协议格式

### 3.1 DTData 结构体

```
DebugTunnelData (40 字节):
  [0]  LogWholeRegion    ptr addrspace(1)  — log buffer 基址
  [8]  BlockNum          i32               — block 数量
  [16] LogBufferSize     i64               — 每个 block 的 log buffer 大小
  [24] kernelWriteType   i32               — 0=Unk, 1=AiC, 2=AiV, 3=Mix
  [32] FftsAddr          ptr addrspace(1)
```

### 3.2 Log Buffer 布局

```
Log Buffer (LogBufferSize + 64 字节):
  [0..7]    pLogSize     — 当前已写入字节数
  [8..63]   padding
  [64..N]   protocol 数据 — print 的字节序列从这里开始写
```

多 block 场景下每个 block 有自己独立的 log buffer 区域：
```
stride = LogBufferSize + 64
block N 的 log buffer = LogWholeRegion + N * stride
```

### 3.3 Protocol 字节序列

以打印 `printf("x=%+08.3f\n", 3.25f)` 为例。普通文本和数值是独立的协议节点；字符串长度均包含结尾的 `\0`：

```
偏移   大小   内容
────────────────────────────────────────────
[0]    1B     NORMAL = 1           ← 前缀文本节点
[1]    2B     text_len = 3         ← "x=\0" 的长度 (i16 LE)
[3]    3B     "x=\0"
[6]    1B     FLOAT = 2            ← 数值节点
[7]    4B     0x00 0x00 0x50 0x40 ← 3.25f 的 little-endian 表示
[11]   2B     fmt_len = 8          ← "%+08.3f\0" 的长度 (i16 LE)
[13]   8B     "%+08.3f\0"
[21]   1B     NORMAL = 1           ← 后缀文本节点
[22]   2B     text_len = 2         ← "\n\0" 的长度 (i16 LE)
[24]   2B     "\n\0"
[26]   1B     END = 0              ← 整条 print 结束
```

如果转换符前后没有普通文本，对应的 NORMAL 节点不会生成。格式串必须恰好包含一个受支持的转换符。当前标量类型支持 f16、bf16、f32、f64 和不超过 32 位的整数；i64 暂不开放，因为现有 DebugTunnel Host 解码会截断高 32 位。

## 4. 编译管线

### 4.1 整体流程

```
PTO IR (.pto)
  │
  ▼
lowerVPTOOps():
  │
  ├── [预扫描] collectAndCreatePrintfStringGlobals
  │     扫描所有 pto.print → 去重格式串 → 创建 LLVM::GlobalOp
  │     扫描所有 pto.tprint → 收集元素值格式串 → 创建 LLVM::GlobalOp
  │
  ├── [Entry ABI] addDTDataParamToEntryFunctions
  │     声明 CCE intrinsics (GET.BLOCK.IDX, DCCI, 等)
  │     给每个 entry function 追加 DTData 隐藏参数 (ptr addrspace(1))
  │
  ├── [Pattern Matching] applyPartialConversion
  │     所有 PTO op 的 lowering pattern 在此同时运行:
  │     ├── LowerPrintOpPattern       ← pto.print → DebugTunnel 协议
  │     ├── LowerTPrintOpPattern      ← pto.tprint → 嵌套 scf.for + DebugTunnel
  │     ├── LowerAllocTileOpPattern   ← pto.alloc_tile → null ptr addrspace(6)
  │     ├── LowerRuntimeQueryOpPattern ← pto.get_block_idx → GET.BLOCK.IDX
  │     ├── LowerMTEOpPattern          ← pto.mte_gm_ub → MOV.OUT.TO.UB
  │     ├── ...30+ 其他 pattern
  │
  ├── [Prologue] injectPrintPrologue
  │     给 entry function 注入 fix stack 初始化 + kernelWriteType = AiV
  │
  └── [声明补全] materializeDecls
        处理各 pattern 推迟声明的调用目标
```

### 4.2 关键设计决策

| 决策 | 原因 | 效果 |
|------|------|------|
| 用嵌套 `scf::IfOp` 做空指针和溢出检查 | 保持结构化控制流，便于后续统一转换 | print 可直接出现在普通控制流 region 中 |
| 预扫描创建 format global | 保证 `AddressOfOp` 在 pattern 运行时 symbol 已存在 | 避免 pattern 内部创建 module-level op |
| 用 `LLVM::LLVMFuncOp` 声明 intrinsic | `injectPrintPrologue` 在 type conversion 后运行，需要 LLVM 类型 | func-to-LLVM conversion 前后类型一致 |
| `materializeDecls` 去重已有 symbol | `addDTDataParamToEntryFunctions` 已创建 LLVM func | get_block_idx + print 共存不冲突 |
| DTData 作为函数参数注入 | 不依赖全局变量 | ABI 干净，每个 block 有独立指针 |
| `TileBufType` → `ptr addrspace(6)` | Tile 在 UB 中通过虚拟地址访问，需要 addrspace(6) 指针 | TPrint 和 AllocTile 可统一使用 LLVM pointer 类型 |
| `LowerAllocTileOpPattern` 产生 null ptr | alloc_tile 的物理地址由 consumer op 计算（通过 `GET.SYS.VA.BASE`），alloc 本身不需要分配 | 简化类型转换，consumer 通过虚拟地址偏移访问 |
| TPrint 使用固定输出格式 | header 和 shape 在 lowering 时形成文本记录，元素值使用预创建的格式串 global | 输出稳定且无需运行时格式解析 |

## 5. 与各类 PTO Op 的集成

Print lowering 和其他 op 的 lowering 在**同一个 `applyPartialConversion` pass** 中运行。它对其他 op 完全透明：

- **不抢寄存器**：只做 `store` 指令，不分配 vector register
- **不占 UB**：数据写入独立的 DTData log buffer，不碰 UB data buffer
- **不影响流水线同步**：flush 在协议写入完成后才执行
- **不干涉控制流**：多 block 寻址用同一个 `GET.BLOCK.IDX`，每个 block 写自己那片区域


Print 在流水线同步点之间执行，不影响任何硬件 pipeline 状态。

## 6. 完整示例：print-all-features

当前 `print-all-features` 用例聚焦验证标量 Print 与算术值、多 block 启动和 DebugTunnel buffer 的共存；TPrint、循环和条件控制流由各自的独立用例覆盖。

```mlir
func.func @print_all_features_kernel_mix_aiv(
    %value: f32, %total_elems: i32) attributes {pto.entry} {
  %block = pto.get_block_idx
  %block_idx = arith.index_cast %block : i64 to index
  %block_idx_i32 = arith.index_cast %block_idx : index to i32

  pto.print ins("block %d\n", %block_idx_i32 : i32)
  pto.print ins("value = %+08.3f\n", %value : f32)
  pto.print ins("total_elems = %d\n", %total_elems : i32)
  return
}
```

Host runner 先创建 ACL device context 和 stream，再启动 kernel 并同步 stream。验证脚本保留 simulator 输出到 `runtime.log`，检查两个 block 的编号以及格式化后的标量值。

控制流场景不需要 outline helper：`LowerPrintOpPattern` 直接生成嵌套 `scf.if`，因此 Print 可以保留在已有的 `scf.for` 和 `scf.if` region 中。

## 8. TPrintOp：Tile 数据打印

`pto.tprint` 打印整个 Tile 的全部元素。与 `pto.print`（固定一条 protocol 记录）不同，TPrint 需要为 Tile 的**每个元素**生成一条 DebugTunnel protocol 记录。

### 8.1 核心思路

对标 EmitC 路径的 `TPRINT(src)` 宏（展开为遍历 tile 元素的循环 + 逐元素 `cce::printf` 调用），VPTO 路径在 MLIR 层生成：

1. **嵌套 `scf.for` 循环** — 外层遍历行，内层遍历列
2. **UB 虚拟地址计算** — 通过 `GET.SYS.VA.BASE` + `0x80000` 偏移得到 UB 基址，再按元素索引偏移
3. **逐元素 Protocol 写入** — 每个元素写 FLOAT/INT marker + 数据 bytes + 格式串 + END marker
4. **末尾单次 DCCI flush** — 所有元素写入完成后集中 flush

### 8.2 固定输出记录

TPrint 输出由三部分组成：

1. lowering 根据 Tile 类型和静态 shape 生成 header 文本记录，例如 `Data Type: float32, Layout: ND, TileType: Vec`。
2. lowering 生成 shape 文本记录，包含 `Shape` 和 `Valid Shape`。
3. 嵌套循环为每个元素生成数值记录；f16/f32 使用 `%6.2f`，i32 使用 `%6d`。

header 和 shape 已经是完全确定的文本，不需要运行时替换 `%s` 或 `%d`。元素格式串由预扫描创建为 LLVM global，供逐元素记录引用。buffer 容量检查同时计入两个文本记录和全部元素记录。

### 8.3 LowerAllocTileOpPattern：Tile 分配 → null ptr

`pto.alloc_tile` 在 VPTO 路径中不实际分配 UB 内存（物理地址由 consumer op 通过虚拟地址计算）。`LowerAllocTileOpPattern` 将其转换为 `null ptr addrspace(6)`：

```cpp
auto ptr6Type = LLVM::LLVMPointerType::get(rewriter.getContext(), 6);
auto nullPtr = rewriter.create<LLVM::ZeroOp>(op.getLoc(), ptr6Type);
rewriter.replaceOp(op, nullPtr.getResult());
```

配套的 `VPTOTypeConverter` 将 `TileBufType` 映射为 `LLVM::LLVMPointerType<6>`：

```cpp
if (isa<pto::TileBufType>(type))
  return LLVM::LLVMPointerType::get(context, 6);
```

### 8.4 LowerTPrintOpPattern：嵌套循环 + 逐元素写入

代码位置：`lib/PTO/Transforms/VPTOCANN900LLVMEmitter.cpp:9572-9754`

#### 结构总览

```
LowerTPrintOpPattern::matchAndRewrite
  │
  ├─ [1] 解析 Tile 类型 (dtype, rows, cols, element size)
  ├─ [2] Null check: if DTData && LogWholeRegion both non-null
  │     └─ scf::IfOp (withElseRegion = false)
  ├─ [3] 计算 log buffer 地址 (LogWholeRegion + block_idx * stride)
  ├─ [4] Overflow check
  │     ├─ Then: 只更新 pLogSize
  │     └─ Else:
  │         ├─ 计算 UB 虚拟地址 (GET.SYS.VA.BASE + 0x80000) / elemBytes
  │         ├─ 外层 scf.for (row: 0 → rows)
  │         │    └─ 内层 scf.for (col: 0 → cols)
  │         │         ├─ 计算元素偏移: virtElemOff = ubBase + row*cols + col
  │         │         ├─ Load 元素值: GEP ptr addrspace(6) + Load
  │         │         ├─ 写 marker byte (FLOAT=2 或 INT=3)
  │         │         ├─ 写数据 bytes (LE):
  │         │         │    float: bitcast → lshr → trunc → store × 4
  │         │         │    int:   sext → lshr → trunc → store × 8
  │         │         ├─ 写格式串长度 (i16 LE)
  │         │         ├─ 写格式串前缀 (逐字节从 global load → store)
  │         │         └─ 写 END marker (0)
  │         └─ 更新 pLogSize → DCCI flush
  └─ eraseOp
```

#### 关键代码：UB 虚拟地址计算

昇腾 AI Core 的 UB (Unified Buffer) 通过虚拟地址访问，而不是通过指针。每个 Tile 元素在 UB 中的虚拟地址由以下公式计算：

```
ub_base = (GET.SYS.VA.BASE + 0x80000) / elem_bytes
elem_addr = tile_data_base + ub_base + (row * cols + col)
```

对应的 LLVM IR 生成：

```cpp
auto sysva = rewriter.create<LLVM::CallOp>(loc, i64Type,
    sysVaBaseFunc.getSymName(), ValueRange{});
auto ubOff = rewriter.create<LLVM::ConstantOp>(loc, i64Type,
    rewriter.getI64IntegerAttr(0x80000));
auto baseAddr = rewriter.create<LLVM::AddOp>(loc, i64Type,
    sysva.getResult(), ubOff);
auto elemSizeVal = rewriter.create<LLVM::ConstantOp>(loc, i64Type,
    rewriter.getI64IntegerAttr(elemBytes));
ubBaseElemOffset = rewriter.create<LLVM::UDivOp>(loc, i64Type,
    baseAddr.getResult(), elemSizeVal);
```

然后用 `GEPOp ptr addrspace(6)` 通过虚拟地址加载元素：

```cpp
auto virtElemOff = rewriter.create<LLVM::AddOp>(loc, i64Type,
    ubBaseElemOffset, elemIdx);
auto elemPtr = rewriter.create<LLVM::GEPOp>(loc, ptr6Type,
    elemType, tileDataBase, ValueRange{virtElemOff.getResult()});
Value elemVal = rewriter.create<LLVM::LoadOp>(loc, elemType, elemPtr);
```

### 8.5 生成的 LLVM IR 形态

以下为 8×8 f32 tile 的 tprint 输出简化示例：

```llvm
; TPrint 格式串 global
@_ptoas_printf_fmt_0 = private constant [63 x i8]
    c"=== [TPRINT Tile] Data Type: %s, Layout: %s, TileType: %s ===\0A\00"
@_ptoas_printf_fmt_1 = private constant [41 x i8]
    c"  Shape: [%d, %d], Valid Shape: [%d, %d]\0A\00"
@_ptoas_printf_fmt_2 = private constant [6 x i8] c"%6.2f\00"
@_ptoas_printf_fmt_3 = private constant [5 x i8] c"%6d\00"
@_ptoas_printf_fmt_4 = private constant [2 x i8] c" \00"
; ... 更多格式串

; kernel 函数
define void @tprint_test_kernel_mix_aiv(ptr addrspace(1) %dtdata) {
  ; Prologue (fix stack 初始化)
  %fix = call ptr @llvm.hivm.get.sycl.fix.stack.object()
  ; ... null check + kernelWriteType = 2 ...

  ; Null check: DTData + LogWholeRegion
  ; Overflow check: totalRecSize = rows*cols * elemRecordSize

  ; UB 虚拟地址基址
  %sysva = call i64 @llvm.hivm.GET.SYS.VA.BASE()
  %base = add i64 %sysva, 524288           ; 0x80000
  %ubOff = udiv i64 %base, 4               ; f32 = 4 bytes

  ; 外层循环: row = 0 → 8
  scf.for %row = %c0 to %c8 step %c1 {
    ; 内层循环: col = 0 → 8
    scf.for %col = %c0 to %c8 step %c1 {
      ; 计算虚拟元素偏移
      %rowOff = mul i64 %row, 8
      %elemIdx = add i64 %rowOff, %col
      %vOff = add i64 %ubOff, %elemIdx

      ; 加载元素值 (通过 ptr addrspace(6))
      %elemPtr = getelementptr inbounds float, ptr addrspace(6) null, i64 %vOff
      %val = load float, ptr addrspace(6) %elemPtr

      ; 写 FLOAT marker
      store i8 2, ptr addrspace(1) %logBuf+%writeOff

      ; 写 float 值 (4 bytes LE)
      %bits = bitcast float %val to i32
      %b0 = lshr i32 %bits, 0;   store i8 (trunc %b0), %buf+1
      %b1 = lshr i32 %bits, 8;   store i8 (trunc %b1), %buf+2
      %b2 = lshr i32 %bits, 16;  store i8 (trunc %b2), %buf+3
      %b3 = lshr i32 %bits, 24;  store i8 (trunc %b3), %buf+4

      ; 写格式串 (逐字节从 global load → store)
      ; ... 5 字节 "%6.2f" ...

      ; 写 END marker
      store i8 0, ptr addrspace(1) %buf+endOff

      ; 推进 writeOff
      %nextOff = add i64 %writeOff, elemRecordSize
      scf.yield %nextOff
    }
    scf.yield %innerResult
  }

  ; 更新 pLogSize + DCCI flush
  store i64 %newPls, ptr addrspace(1) %logBuf
  call void @llvm.hivm.DCCI(ptr addrspace(1) null, i64 1)
  ret void
}
```

**关键特征**：
- 包含 `scf.for` 循环（编译时知道迭代次数，后续可被 LLVM 展开或保持为循环）
- 通过 `ptr addrspace(6)` 访问 UB（昇腾 AICORE 的 UB 地址空间）
- 每个元素写一条完整的 DebugTunnel protocol 记录
- 所有元素写完后仅一次 DCCI flush（而非每个元素都 flush）
- TPrint 总大小在编译期确定：两个文本记录，加上 `rows × cols × (1 + dataSize + 2 + fmtBytes + 1)`

### 8.6 TPrint 特有的设计决策

| 决策 | 原因 | 效果 |
|------|------|------|
| 嵌套 `scf.for` 而非展开所有元素 | 大 tile（如 16×64=1024 元素）全展开会生成数千条 store，代码膨胀严重 | 保留循环结构，LLVM 后端可根据 target 决定是否展开 |
| `alloc_tile` → null ptr | 物理地址由 consumer (`tprint`) 通过虚拟地址计算，alloc 本身只做类型占位 | 避免在 lowering 阶段处理 UB 物理分配 |
| 末尾单次 DCCI flush | 逐元素 flush 会产生 64×N 次 cache 同步，性能极差 | 一次 flush 完成所有元素数据的通知 |
| 固定 header/shape 文本记录 | Tile 元数据在编译期已知 | Host 输出包含稳定的类型、布局和 shape 信息 |

## 7. 遇到的关键问题与解决方案

### 7.1 get_block_idx + print intrinsic 冲突

**问题**：`addDTDataParamToEntryFunctions` 创建 `LLVM::LLVMFuncOp @GET.BLOCK.IDX`，后续 `LowerRuntimeQueryOpPattern` 通过 `PlannedDecl` → `materializeDecls` 又尝试创建 `func::FuncOp` 同名 symbol，触发 LLVM 符号重定义。

**修复**：
1. `materializeDecls` 跳过已存在 `LLVM::LLVMFuncOp` 的 symbol
2. `LowerRuntimeQueryOpPattern` 检测到已有 LLVM func 时，直接用 `LLVM::CallOp` 而非 `func::CallOp` + 延迟声明

### 7.2 scf.if / scf.for 中的 Print

`LowerPrintOpPattern` 使用嵌套 `scf::IfOp` 完成 DTData、LogWholeRegion 和 buffer overflow 检查，不再拆分所在 block，也不再创建 outline helper。因此 Print 可以直接位于已有的 `scf.if` 或 `scf.for` region 中。

### 7.3 CC1 模式无法编译 DebugTunnel host stub

**问题**：DebugTunnel host 端头文件依赖 C++ 标准库（`<string>` 等），bisheng CC1 模式 (`-cc1`) 无法处理 libstdc++ include path。

**修复**：检测到 kernel 使用 print 时，切换为 bisheng driver 模式 (`-xcce --cce-enable-print`)，driver 自动管理系统头文件路径。

### 7.4 vsts 的 dist 和 mask 语义

非 bug，但需要文档化：`{dist = "1PT_B32"}` 是单点写入，全向量写入用默认分布（不指定 dist）；`%one_mask` 只激活 1 个 lane，全 lane 写入用 `PAT_ALL`。

### 7.5 TPrint 的 UB 虚拟地址计算

**问题**：`pto.tprint` 需要从 UB 读取 Tile 元素，但 `alloc_tile` 在 lowering 阶段已转为 null ptr。如何通过合法的 LLVM IR 访问 UB 中的元素？

**修复**：
1. `alloc_tile` → null ptr addrspace(6)（占位符）
2. TPrint 内部通过 `GET.SYS.VA.BASE() + 0x80000` 计算 UB 基址，除以元素字节数得到元素索引单位的偏移
3. 通过 `GEPOp ptr addrspace(6)` + `LoadOp` 用虚拟地址访问元素
4. 这种模式与 CCE 编译器对 UB 访问的处理一致

### 7.6 InsertTemplateAttributes / ExpandTileOp 需跳过 Print/TPrint

**问题**：`InsertTemplateAttributes` 和 `ExpandTileOp` 默认处理所有 PTO op，但 `pto.print` 和 `pto.tprint` 是调试 op，不需要模板展开或 tile 扩展——它们由 VPTO lowering pattern 直接处理。

**修复**：在 `InsertTemplateAttributes` 和 `ExpandTileOp` 的 walker 中显式 `isa<pto::PrintOp> || isa<pto::TPrintOp>` 跳过。


## 9. 与 EmitC 路径的 LLVM IR 对比

EmitC 路径的 print（`cce::printf`）由 C++ 模板库在**运行时**完成格式解析和 protocol 构造。VPTO 路径在**编译期**完成这一切，直接展开为 `store` 指令序列。

下面以同一个 print 调用 `printf("scalar = %+08.3f\n", 3.25f)` 为例，对比两种路径生成的 LLVM IR。

对照文件：
- EmitC: `research/step2/golden/print_scalar_golden_dev.ll`（C++ `cce::printf` 经 bisheng CCE 前端编译）
- VPTO: `test/vpto/cases/kernels/print-scalar/kernel.pto` 经 PTOAS 编译

### 9.1 整体对比
  CCE 对于cce::printf的处理：

  cce::printf<float>(fmt, args)
    │
    ├── [1] GetKernelInstance()
    │      从 fix stack 读 DTData 指针
    │      如果为空 → 跳过打印（走 cleanup）
    │      设置 kernelWriteType = AiV
    │
    ├── [2] PrintState(fmt) << arg
    │      创建 PrintState 对象
    │      逐个字符解析格式串:
    │        while c != '%' && c != '\0' → 跳过普通字符
    │        ParseFlag()    → '+', '-', '0', ' ', '#'
    │        ParseWidth()   → "08"
    │        ParsePrec()    → ".3"
    │        ParseLength()  → "l", "z", "h"
    │        switch(specifier):
    │          case 'f' → Write(FLOAT marker + 4 bytes)
    │          case 'd' → Write(INT marker + 8 bytes)
    │          case 's' → Write(STR marker + strlen+2 bytes)
    │          ...
    │          case 'd' → Write(INT marker + 8 bytes)
    │          case 's' → Write(STR marker + strlen+2 bytes)
    │          ...
    │      调用 Write() 写入:
    │        - 格式串前缀（% 之前的部分 + '\0'）→ NORMAL marker
    │        - 剩余格式串后缀（% 之后的部分 + '\0'）→ END marker
    │
    └── [3] DCCI flush
            通知 Host 端数据已就绪


   VPTO lowering实现相同逻辑
### 9.2 主 kernel 函数形态

**EmitC 路径**（kernel 函数本身只有 3 行）：

```llvm
define dso_local void @print_scalar_kernel(
    float noundef %value, ptr addrspace(1) noundef %.DTData) {
entry:
  call void @_ZN3cce6printfIU3AS1cJfEEEvPKT_DpT0_(
      ptr addrspace(1) noundef @.str, float noundef %value)
  ret void
}
```

所有的格式解析、protocol 构造逻辑都在 `cce::printf` 模板展开中（占据其余 13 个函数、900+ 行）。kernel 函数只是一个薄薄的 wrapper。

**VPTO 路径**（kernel 函数展开全部逻辑）：

```llvm
define void @print_scalar_kernel_mix_aiv(float %0, ptr addrspace(1) %1) {
  ; --- Prologue (injectPrintPrologue 注入) ---
  %fix = call ptr @get.sycl.fix.stack.object()
  if %dtdata != null:
    store %dtdata to fix stack
    store kernelWriteType = 2 (AiV)

  ; --- Print protocol ---
  if dtdata == null → skip
  if LogWholeRegion == null → skip

  stride = LogBufferSize + 64
  offset = get_block_idx() * stride
  logBuf = LogWholeRegion + offset

  if overflow → update pLogSize, skip

  ; NORMAL 前缀节点: "scalar = \0"
  ; FLOAT 节点: marker + 4-byte value + "%+08.3f\0"
  ; NORMAL 后缀节点: "\n\0"
  ; END marker

  store newPls to logBuf            ; 更新已写字节数
  call void @llvm.hivm.DCCI(null, 1) ; flush
  ret void
}
```

没有函数调用（除 DCCI），没有循环，没有运行时格式解析——全是编译期确定好的 store 指令。

### 9.3 格式串处理：运行时解析 vs 编译期计算

**EmitC 路径**：`ccelib` 里有一个完整的 printf 状态机。在 `PrintState::ls<float>()` 中：

```llvm
; 解析 flag: 循环读 '+', '-', ' ', '#', '0'
while.body.i:
  %c = load i8, fmt[state.idx]
  switch %c {
    case 48('0'), 45('-'), 43('+'), 32(' '), 35('#'):
      state.idx++
  }

; 解析 width: 循环读数字
while.cond.i:
  %c = load i8, fmt[state.idx]
  if %c >= 48 && %c <= 57 → state.idx++  ; digit

; 解析 .precision
; 解析 length modifier: l, z, h

; 最终 switch 分发:
switch %specifier {
  case 'f','F':  store i8 2 (FLOAT), write 4 bytes
  case 'd','i':  store i8 3 (INT), write 8 bytes
  case 's':      store i8 5 (STR), write strlen+2 bytes
  case 'x','X':  store i8 3 (INT), write 8 bytes
  ...
}
```

这个状态机在**每次 print 调用时都在运行**——解析 `%+08.3f` 的每个字符，决定格式标志、宽度、精度、长度修饰符、最终的类型标记。

**VPTO 路径**：LowerPrintOpPattern 在编译期解析格式串，直接算出数值：

```cpp
// 编译期解析由 verifier 和 lowering 共用
auto info = analyzePrintFormat(format);

// info 记录前缀、转换格式和后缀的 offset/bytes。
// lowering 据此生成可选 NORMAL 文本节点和一个数值节点。
auto enc = encodePrintScalar(rewriter, loc, scalarType, scalar,
                             info->conversion);
emitTextNode(info->prefixOffset, info->prefixBytes);
emitValueNode(enc, info->conversionOffset, info->conversionBytes);
emitTextNode(info->suffixOffset, info->suffixBytes);
emitEndNode();
```

### 9.4 数据写入：通用循环 vs 直线 store

**EmitC 的 `Write()` 函数**（通用逐字节拷贝）：

```llvm
; Write(ptr %str, i64 %size) — 每次 print 都调这个函数拷贝数据
while.cond:
  %remaining = phi i64 [ %size, %entry ], [ %dec, %while.body ]
  %src       = phi ptr  [ %str, %entry ],  [ %src+1, %while.body ]
  %dst       = phi ptr  [ %buf, %entry ],  [ %dst+1, %while.body ]
  %tobool = icmp ne i64 %remaining, 0
  br i1 %tobool, label %while.body, label %while.end

while.body:
  %byte = load i8, ptr %src
  store i8 %byte, ptr addrspace(1) %dst
  %dec = add i64 %remaining, -1
  br label %while.cond
```

这个 `Write()` 函数被 print 内部多次调用——写 marker 调用一次、写数据值调用一次、写格式串前缀调用一次、写格式串后缀再调用一次。每次都是一个 `while` 循环逐字节拷贝。

**VPTO 的展开 store**：

```llvm
; 没有循环，没有函数调用。每个字节一条 store：
store i8 2,    ptr %base+0      ; marker
store i8 %b0,  ptr %base+1      ; value byte 0
store i8 %b1,  ptr %base+2      ; value byte 1
store i8 %b2,  ptr %base+3      ; value byte 2
store i8 %b3,  ptr %base+4      ; value byte 3
store i8 17,   ptr %base+5      ; fmt len low
store i8 0,    ptr %base+6      ; fmt len high
store i8 's',  ptr %base+7      ; fmt[0]
store i8 'c',  ptr %base+8      ; fmt[1]
store i8 'a',  ptr %base+9      ; fmt[2]
; ... 每条一个 store，全部直线排列
```

所有长度在编译期已知，编译器直接展开成 N 条 store 指令。硬件看到的是一段无分支、无循环的直线代码。

### 9.5 Prologue 对比

**EmitC 路径**：独立的 init/finish 函数框架。

```llvm
; Host stub 编译器在 kernel launch 前后插入调用:
@__DebugTunnel_Initialize(DTData)
  → DebugTunnel::OnKernelInitialize(DTData)
    → PrintPayload::OnKernelInitialize(PrintData)
      → fix stack ← PrintData

@__DebugTunnel_Finish(DTData)
  → DebugTunnel::OnKernelFinish(DTData)
    → PrintPayload::OnKernelFinish(PrintData)
      → DCCI(null, 1)  ; flush
```

`DebugTunnel_Initialize` / `DebugTunnel_Finish` 是两个顶层入口，由 host stub 编译器（`--cce-enable-print` 模式）生成的 launch wrapper 在 kernel 前后调用。print 内部通过 `GetKernelInstance()` 从 fix stack 读取已初始化的 DTData。

**VPTO 路径**：`injectPrintPrologue` 直接把初始化内联到 kernel 入口。

```llvm
; kernel 函数入口，prologue 直接内联:
%fix = call ptr @get.sycl.fix.stack.object()
if %dtdata != null:
  store %dtdata to fix stack
  store kernelWriteType = 2 (AiV) to dtdata+24
```

不需要 `DebugTunnel_Initialize` / `DebugTunnel_Finish` 函数——fix stack 初始化和 flush 都在 kernel 体内完成。Host stub 编译时只需加上 `--cce-enable-print` 让编译器知道这是一条 print kernel。

