# PTO Print 实现原理：EmitC 路径 vs VPTO 路径

本文档系统讲解 `pto.print` 的两条实现路径，面向需要理解 print 子系统如何工作的开发者。

**阅读目标**：理解 EmitC 路径中 ccelib 如何实现 print，以及 VPTO 路径如何在编译期复现相同的效果。

---

## 目录

1. [整体概览](#1-整体概览)
2. [数据流全景](#2-数据流全景)
3. [EmitC 路径详解](#3-emitc-路径详解)
   - [3.1 PTO → EmitC 降级](#31-pto--emitc-降级)
   - [3.2 ccelib 层叠结构](#32-ccelib-层叠结构)
   - [3.3 DebugTunnel：Host-Device ABI 桥梁](#33-debugtunnelhost-device-abi-桥梁)
   - [3.4 PrintState：格式串状态机](#34-printstate格式串状态机)
   - [3.5 Write()：协议写入引擎](#35-write协议写入引擎)
   - [3.6 cce::printf：总调度函数](#36-cceprintf总调度函数)
   - [3.7 完整调用图](#37-完整调用图)
   - [3.8 TPRINT：Tile 打印宏](#38-tprinttile-打印宏)
4. [VPTO 路径详解](#4-vpto-路径详解)
   - [4.1 核心思路：编译期内联](#41-核心思路编译期内联)
   - [4.2 编译管线分步解析](#42-编译管线分步解析)
   - [4.3 LowerPrintOpPattern 逐行解读](#43-lowerprintoppattern-逐行解读)
   - [4.4 injectPrintPrologue：入口初始化](#44-injectprintprologue入口初始化)
   - [4.5 生成的 LLVM IR 形态](#45-生成的-llvm-ir-形态)
   - [4.6 LowerTPrintOpPattern：Tile 数据打印](#46-lowertprintoppatterntile-数据打印)
   - [4.7 LowerAllocTileOpPattern + TileBufType 转换](#47-loweralloctileoppattern--tilebuftype-转换)
5. [两条路径的逐项对比](#5-两条路径的逐项对比)
6. [关键技术决策](#6-关键技术决策)

---

## 1. 整体概览

`pto.print` 是 PTO IR 中的调试打印操作。用 C 语言的 `printf` 来类比：

```mlir
// PTO IR 写法
pto.print ins("scalar = %+08.3f\n", %value : f32)

// 等价于 C 代码
cce::printf("scalar = %+08.3f\n", value);
```

但昇腾 AI Core **没有操作系统、没有 C 标准库**——无法直接调用 `printf`。那打印数据怎么从设备端传出来？

**答案**：通过一个叫 **DebugTunnel** 的 Host-Device 通信协议。设备端把打印内容按约定格式写入 HBM（高带宽内存）的一块 log buffer 中，Host 端在 kernel 执行完后读出并显示。

```
┌─────────────────────────────────────────────────────┐
│                    Device (AI Core)                  │
│                                                     │
│  pto.print "x=%f\n", 3.25                          │
│       │                                             │
│       ▼                                             │
│  按 DebugTunnel 协议格式                             │
│  逐字节写入 HBM 中的 LogBuffer                       │
│       │                                             │
│       ▼                                             │
│  DCCI cache flush ──────────────┐                   │
│                                  │                   │
├──────────────────────────────────┼───────────────────┤
│                    Host (x86 CPU)                    │
│                                  │                   │
│  __DebugTunnel_Close() ◄─────────┘                   │
│       │                                             │
│       ▼                                             │
│  读回 LogBuffer → 解析 Protocol → printf 输出到终端   │
│                                                     │
│  输出: "scalar = +003.250"                           │
└─────────────────────────────────────────────────────┘
```

PTOAS 对这个过程有**两条实现路径**：

| 路径 | 方式 | 何时做格式解析 | 对 ccelib 的依赖 |
|------|------|---------------|-----------------|
| **EmitC** | 生成 C++ 代码 → 调用 `cce::printf` → CCE 编译器展开 | **运行时** | 依赖完整的 ccelib 头文件 |
| **VPTO** | 直接在 PTOAS 内部展开为 LLVM IR 的 `store` 指令序列 | **编译期** | 不依赖——编译器自行复制了协议格式 |

---

## 2. 数据流全景

以下是以 `pto.print ins("scalar = %+08.3f\n", 3.25f)` 为例的端到端数据流：

```
═══════════════════════════════════════════════════════════════════════════════
                           Host 端 (运行时，kernel launch 前)
═══════════════════════════════════════════════════════════════════════════════

  __DebugTunnel_Open()
    │
    ├─ rtMalloc(DTData,  40 字节)     ← 分配 DebugTunnelData 结构体
    ├─ rtMalloc(LogBuffer, N×16448)    ← 分配打印缓冲区 (HBM 中)
    │
    ├─ DTData[0]  = LogBuffer 地址     ← LogWholeRegion: 这是写入目标
    ├─ DTData[8]  = BlockNum
    ├─ DTData[16] = LogBufferSize      ← 默认 16384
    ├─ DTData[24] = kernelWriteType    ← 0=Unk, 1=AiC, 2=AiV, 3=Mix
    ├─ DTData[32] = FftsAddr
    │
    ├─ rtMemcpy(DTData → Device)       ← 把 DTData 拷贝到设备可见的 HBM
    │
    └─ __cce_rtKernelLaunch(kernel, user_args..., DTData)
                                       ← DTData 作为隐藏的最后一个参数传给 kernel


═══════════════════════════════════════════════════════════════════════════════
                           Device 端 (kernel 执行中)
═══════════════════════════════════════════════════════════════════════════════

  kernel(float user_arg, ptr addrspace(1) DTData)  ← DTData 是隐藏参数
    │
    ├─ [Prologue] fix stack ← DTData 地址
    │             kernelWriteType = 2 (AiV)
    │
    ├─ [Print] GetKernelInstance() → 从 fix stack 读回 DTData
    │          LogWholeRegion ← DTData[0]     ← GM 地址
    │          LogBufferSize  ← DTData[16]
    │
    │          stride  = LogBufferSize + 64          ← = 16448
    │          blk_off = get_block_idx() × stride    ← 多 block 各自独立区域
    │          buffer  = LogWholeRegion + blk_off
    │
    │          pLogSize ← buffer[0..7]               ← 当前已写入字节数
    │          writePtr = buffer + 64 + pLogSize      ← 从 buffer 头跳过 64B 头部
    │
    │          写入 Protocol 字节序列:
    │            [0]  FLOAT marker = 2
    │            [1]  4B float value (LE)
    │            [5]  2B fmt_len = 17
    │            [7]  16B "scalar = %+08.3f"
    │            [23] '\0'
    │            [24] NORMAL marker = 1
    │            [25] 2B rem_len = 2
    │            [27] "\n\0"
    │            [29] END marker = 0
    │
    │          pLogSize ← pLogSize + 30               ← 更新已写计数
    │
    └─ [Flush] DCCI(null, 1)                          ← 通知 Host: 数据就绪


═══════════════════════════════════════════════════════════════════════════════
                           Host 端 (kernel 执行后)
═══════════════════════════════════════════════════════════════════════════════

  __DebugTunnel_Close()
    │
    ├─ 读回 DTData → 遍历每个 block:
    │    LogSize ← LogBuffer[0..7]
    │    解析 protocol 字节 → 还原为格式串 + 数值
    │
    └─ printf("scalar = %+08.3f\n", 3.25)
       → 终端输出: "scalar = +003.250"
```

---

## 3. EmitC 路径详解

EmitC 路径的核心理念是**把 print 委托给 CCE 编译器的 ccelib 库**——PTOAS 只负责把 `pto.print` 翻译成 `cce::printf` 调用，剩下的格式解析、protocol 构造、ABI 适配全部由 bisheng CCE 前端完成。

### 3.1 PTO → EmitC 降级

代码位置：`lib/PTO/Transforms/PTOToEmitC.cpp:12256-12294`

```cpp
// pto.print "format", %scalar -> cce::printf("format", scalar)
struct PTOPrintOpToEmitC : public OpConversionPattern<pto::PrintOp> {
  LogicalResult matchAndRewrite(pto::PrintOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    // ① 取出格式字符串，做 C 转义（引号、换行等）
    std::string fmt = op.getFormat().str();
    std::string quoted = "\"";
    for (char c : fmt) {
      if (c == '"' || c == '\\') quoted += '\\';
      else if (c == '\n')        quoted += "\\n";
      // ...
    }
    quoted += "\"";

    // ② 生成 emitc::CallOpaqueOp，直接调用 cce::printf
    Value scalar = peelUnrealized(adaptor.getScalar());
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "cce::printf",
        /*args=*/rewriter.getArrayAttr({
            emitc::OpaqueAttr::get(ctx, quoted),   // 格式串
            IntegerAttr::get(IndexType::get(ctx), 0) // scalar 的模板位置
        }),
        /*operands=*/ValueRange{scalar});

    rewriter.eraseOp(op);
    return success();
  }
};
```

最终生成的 C++ 代码只有一行：

```cpp
cce::printf("scalar = %+08.3f\n", value);
```

这一行代码看起来简单，但背后整个打印系统的工作都藏在 **ccelib 的头文件展开**中。

### 3.2 ccelib 层叠结构

CCE 运行时库的 print 子系统由三层组成：

```
┌──────────────────────────────────────────────────┐
│               cce::printf (公开 API)               │
│  ccelib/print/print.h                             │
│  用户只看到这一层，传入格式串和参数                    │
├──────────────────────────────────────────────────┤
│          DebugTunnel (Host-Device ABI)             │
│  OnKernelInitialize / OnKernelFinish               │
│  GetKernelInstance / fix stack 管理                 │
│  负责 DTData 生命周期和 flush                       │
├──────────────────────────────────────────────────┤
│     PrintPayload + PrintState (协议引擎)            │
│  PrintPayloadData 结构体                            │
│  PrintState: 格式串状态机                           │
│  Write(): 逐字节写入 LogBuffer                      │
│  负责协议字节序列的生成                              │
└──────────────────────────────────────────────────┘
```

这三层的关系：**PrintState 决定写什么，Write() 决定怎么写，DebugTunnel 决定写到哪里**。

#### PrintPayloadData 结构体 (40 字节)

这是整个打印系统的核心数据结构。Host 分配、Device 读写的那个结构体就是它：

```cpp
// 等价 C++ 定义（从 LLVM IR 反推）
struct PrintPayloadData {
  void*    LogWholeRegion;   // [0]  ptr addrspace(1): HBM 中 log buffer 基址
  uint32_t BlockNum;         // [8]  block 数量
  uint32_t _pad1;            // [12] (alignment padding)
  uint64_t LogBufferSize;    // [16] 每个 block 的 buffer 大小 (默认 16384)
  uint32_t kernelWriteType;  // [24] 0=Unk, 1=AiC, 2=AiV, 3=Mix
  uint32_t _pad2;            // [28] (alignment padding)
  void*    FftsAddr;         // [32] FFTS 地址
};
// sizeof = 40
```

对应的 LLVM IR 类型定义（golden IR 第 7 行）：

```llvm
%"struct.cce::internal::PrintPayloadData" = type <{
    ptr addrspace(1),   ; LogWholeRegion
    i32,                ; BlockNum
    [4 x i8],           ; padding
    i64,                ; LogBufferSize
    i32,                ; kernelWriteType
    [4 x i8]            ; padding
}>
```

### 3.3 DebugTunnel：Host-Device ABI 桥梁

DebugTunnel 层解决两个核心问题：**DTData 怎么传给 kernel** 和 **打印完成后怎么通知 Host**。

#### 3.3.1 OnKernelInitialize：fix stack 初始化

代码对应 golden IR 的第 60-88 行：

```llvm
define void @DebugTunnel::OnKernelInitialize(ptr addrspace(1) %DTData) {
entry:
  %fix = call ptr @llvm.hivm.get.sycl.fix.stack.object()
  if (%DTData == null) {
    store i64 0, ptr %fix        ; 没有 DTData → fix stack 存 0
  } else {
    PrintPayload::OnKernelInitialize(%DTData)  ; 验证 LogRegion 非空
    %addr = ptrtoint %DTData to i64
    store i64 %addr, ptr %fix     ; fix stack 存 DTData 地址
  }
}
```

**fix stack** 是 AI Core 上的一个特殊内存区域，每个 core 一个，用来在 kernel 的不同阶段之间传递数据。这里把 DTData 的地址存进去，后续 `GetKernelInstance()` 就能取出来。

#### 3.3.2 GetKernelInstance：获取 DTData

代码对应 golden IR 的第 230-247 行：

```llvm
define ptr addrspace(1) @GetKernelInstance() {
  %fix = call ptr @llvm.hivm.get.sycl.fix.stack.object()
  %val = load i64, ptr %fix       ; 读 fix stack 中的 DTData 地址
  if (%val == 0)
    return null                    ; 初始化失败 → 跳过打印
  %dtdata = inttoptr i64 %val to ptr addrspace(1)
  return %dtdata
}
```

这个函数被 `cce::printf` 和 `Write()` 反复调用。每次打印都需要：
1. 从 fix stack 拿 DTData 指针
2. 从 DTData 拿 LogWholeRegion（GM 地址）
3. 从 DTData 拿 LogBufferSize
4. 计算写入位置

#### 3.3.3 OnKernelFinish：DCCI flush

代码对应 golden IR 的第 39-54 行：

```llvm
define void @PrintPayload::OnKernelFinish(ptr addrspace(1) %PrintData) {
  %logRegion = load ptr addrspace(1), ptr addrspace(1) %PrintData
  if (%logRegion != null)
    call void @llvm.hivm.DCCI(ptr addrspace(1) null, i64 1)
  ret void
}
```

**DCCI** 是 Data Cache Coherency Interface 的缩写。AI Core 写入 HBM 的数据会先存在 cache 中，DCCI flush 强制刷到 HBM，Host 端才能读到正确数据。

#### 3.3.4 调用链

Host stub 编译器（bisheng driver 模式）生成的 launch wrapper 会在 kernel 前后插入：

```cpp
// Host 侧伪代码
__DebugTunnel_Open(DTData, kernel_name);   // ① 分配 buffer
launch_kernel(args, DTData);               // ② 执行 kernel
__DebugTunnel_Close(DTData);               // ③ 读回 + 显示
```

对应的 Device 侧：

```cpp
// Device 侧伪代码
__DebugTunnel_Initialize(DTData) {
    DebugTunnel::OnKernelInitialize(DTData) {
        fix_stack ← DTData
    }
}

kernel(args, DTData) {
    cce::printf("fmt", val) {
        dtdata = GetKernelInstance()  // 从 fix stack 取回
        // ... 写 protocol ...
        DCCI flush
    }
}
```

### 3.4 PrintState：格式串状态机

这是 EmitC 路径中最复杂的部分。`PrintState` 是一个 C++ 模板类，在**运行时**逐字符解析格式串。

对应 golden IR 的第 6 行类型定义 + 第 351-805 行的 `operator<<` 实现：

```llvm
%"class.cce::internal::PrintState" = type {
    i32,                     ; idx: 当前解析位置
    ptr addrspace(1)         ; fmt: 格式串指针
}
```

#### 3.4.1 构造函数

对应 golden IR 的第 906-912 行：

```llvm
define void @PrintState::PrintState(ptr %this, ptr addrspace(1) %str) {
  ; this->fmt = str
  store ptr addrspace(1) %str, ptr %fmt_field
  ; this->idx = 0
  store i32 0, ptr %this
}
```

#### 3.4.2 PrintState::ls<float>()：格式串状态机

这是最核心的函数（golden IR 第 352-805 行，454 行 LLVM IR）。以 `printf("scalar = %+08.3f\n", 3.25f)` 为例，分阶段解析：

```
格式串: "scalar = %+08.3f\n"
               ↑         ↑
         位置 9 (%)  位置 16 (f 之后)

阶段 ①: 扫描格式串前缀
─────────────────────────
while (fmt[idx] != '%' && fmt[idx] != '\0')
    idx++
结果: 跳过 "scalar = " (9 个字符)

阶段 ②: ParseFlag — 解析标志字符
─────────────────────────
while (fmt[idx] in {'+', '-', ' ', '#', '0'})
    idx++
结果: 消耗 '+' (1 个字符)

阶段 ③: ParseWidth — 解析宽度
─────────────────────────
while (fmt[idx] >= '0' && fmt[idx] <= '9')
    idx++
结果: 消耗 "08" (2 个字符)

阶段 ④: ParsePrec — 解析精度
─────────────────────────
if (fmt[idx] == '.')
    idx++; while (isdigit) idx++
结果: 消耗 ".3" (2 个字符)

阶段 ⑤: ParseLength — 解析长度修饰
─────────────────────────
if (fmt[idx] in {'l', 'h', 'z'})
    idx++
结果: 无长度修饰符

阶段 ⑥: 格式说明符分发
─────────────────────────
switch (fmt[idx]) {
    case 'f','F':  typeMarker=FLOAT(2); dataSize=4; break;
    case 'd','i':  typeMarker=INT(3);   dataSize=8; break;
    case 's':      typeMarker=STR(5);   dataSize=strlen; break;
    // ... 更多类型
}
结果: 命中 'f' → typeMarker=2, dataSize=4

阶段 ⑦: 写入 protocol
─────────────────────────
Write(typeMarker, 1)           // FLOAT=2
Write(&value, 4)               // float 的 4 字节
Write(fmt_prefix_len, 2)       // 格式串前缀长度 (LE i16)
Write(fmt_prefix, len-1)       // 格式串前缀 (不含 % 及之后, 不含 '\0')
Write('\0', 1)                 // 格式串前缀结束符
Write(NORMAL, 1)               // 开始后缀段
Write(fmt_suffix_len, 2)       // 格式串后缀长度 (LE i16)
Write(fmt_suffix, len-1)       // 格式串后缀 (含 '\0')
Write(END, 1)                  // 0x00 结束标记
DCCI flush                      // 通知 Host
```

下面是对应的 LLVM IR 关键片段：

```llvm
; ---- 阶段 ①: 扫描非 '%' 字符 ----
while.cond:
  %c = load i8, ptr addrspace(1) %fmt[idx+i]
  if (%c != 37 && %c != 0) → continue scanning  ; 37 = '%'

; ---- 阶段 ②: ParseFlag ----
while.body.i:
  %c = load i8, ptr addrspace(1) %fmt[idx]
  switch %c {
    case 48('0'), 45('-'), 43('+'), 32(' '), 35('#'):
      idx++; continue
  }

; ---- 阶段 ③+④: ParseWidth + ParsePrec ----
; 循环消耗数字字符 (48..57)
; 如果遇到 '.' → 再消耗一组数字

; ---- 阶段 ⑤+⑥: ParseLength + switch 分发 ----
switch %c {
  case 's': type=5(STR),  write 1B marker + strlen bytes
  case 'd','i','x','X','o','u': type=3(INT), write 1B + 8B
  case 'f','F','e','E','g','G','a','A': type=2(FLOAT), write 1B + 4B
  case 'p': type=6(PTR),  write 1B + 8B
  case 'c': type=4(CHAR), write 1B + 1B
}

; ---- 阶段 ⑦: 写入 protocol ----
; 每次 Write() 调用都:
;   ① GetKernelInstance() → fix stack → DTData
;   ② DTData → LogWholeRegion + block offset
;   ③ 逐字节拷贝到 LogBuffer
```

### 3.5 Write()：协议写入引擎

代码位置：golden IR 第 250-342 行（`Write<char*>`）和第 808-900 行（`Write<char addrspace(1)*>`）。

两个 `Write()` 重载的区别仅在于源数据地址空间（private vs global），核心逻辑完全一致：

```llvm
define void @Write<char*>(ptr %str, i64 %size) {
  ; ① 获取 DTData
  %dtdata = call @GetKernelInstance()
  if (%dtdata == null) return

  ; ② 获取 LogWholeRegion (GM 地址)
  %logRegion = load ptr addrspace(1), ptr addrspace(1) %dtdata
  if (%logRegion == null) return

  ; ③ 计算本 block 的 buffer 基址
  %LogBufferSize = load i64, ptr addrspace(1) %dtdata[16]
  %stride = %LogBufferSize + 64
  %blockIdx = call @llvm.hivm.GET.BLOCK.IDX()
  %blockOff = %blockIdx * %stride
  %blkBuf = %logRegion + %blockOff           ; ← 本 block 的 buffer

  ; ④ 读当前写入位置
  %pLogSize = load i64, ptr addrspace(1) %blkBuf    ; [0..7]
  %remaining = %LogBufferSize - %pLogSize           ; 剩余空间

  ; ⑤ 溢出检查
  if (%remaining <= 0) {
    ; 溢出: 只更新 pLogSize，不写数据
    %blkBuf[0] = %pLogSize + %size
    return
  }

  ; ⑥ 逐字节拷贝
  %writePtr = %blkBuf + 64 + %pLogSize
  %actualSize = min(%size, %remaining)
  while (%actualSize > 0) {
    %byte = load i8, ptr %str          ; 从源读取
    store i8 %byte, ptr addrspace(1) %writePtr  ; 写入 HBM
    %str++; %writePtr++; %actualSize--
  }

  ; ⑦ 更新 pLogSize
  %blkBuf[0] = %LogBufferSize - (%remaining - %size)
}
```

核心要点：
- **每次写都要重新 GetKernelInstance()**：因为可能有多个 block 并发执行
- **LogBuffer 布局是 [pLogSize(8B) | padding(56B) | data(N)]**
- **溢出时只更新计数，不写数据**：防止破坏 buffer 边界外的内存

### 3.6 cce::printf：总调度函数

代码位置：golden IR 第 138-227 行。这是整个流程的入口：

```llvm
define void @cce::printf<char addrspace(1), float>(ptr addrspace(1) %fmt, float %Args) {
  ; ① 获取 DTData + 验证
  %dtdata = call @GetKernelInstance()
  if (%dtdata == null || dtdata->LogWholeRegion == null)
    return                  ; 没有初始化 → 跳过打印

  ; ② 设置写入类型
  dtdata->kernelWriteType = 2  // AiV

  ; ③ 格式串为 null → 写 END marker 然后退出
  if (%fmt == null) {
    Write(END=0, 1)
    return
  }

  ; ④ 创建 PrintState 并运行状态机
  PrintState s(fmt)             ; s.idx=0, s.fmt=fmt
  s.ls<float>(Args)             ; 运行格式串状态机 → 写入 protocol

  ; ⑤ 写入尾段 NORMAL + 格式串后缀 + END
  Write(NORMAL=1, 1)
  Write(suffix_length, 2)       ; 格式串后缀长度
  Write(fmt.suffix, len)        ; 格式串后缀内容（% 之后的部分含 '\0'）

  ; ⑥ Flush
  DCCI(null, 1)                 ; 通知 Host 端
}
```

### 3.7 完整调用图

```
cce::printf<float>("scalar = %+08.3f\n", 3.25f)
│
├─ GetKernelInstance()
│    └─ fix stack → DTData 地址
│
├─ PrintState("scalar = %+08.3f\n")
│    │
│    └─ s.ls<float>(3.25f)
│         │
│         ├─ [扫描前缀] 跳过 "scalar = "
│         │
│         ├─ [ParseFlag]   消耗 '+' → idx++
│         ├─ [ParseWidth]  消耗 "08" → idx+=2
│         ├─ [ParsePrec]   消耗 ".3" → idx+=2
│         ├─ [ParseLength] 无修饰符 → 不变
│         │
│         ├─ [switch 分发] '%f'
│         │    │
│         │    ├─ Write(FLOAT=2, 1)          ← 写 marker
│         │    ├─ Write(&3.25f, 4)           ← 写 float 值 (LE)
│         │    │    └─ GetKernelInstance → LogRegion → block offset
│         │    │       → 逐字节 [0x00,0x00,0x50,0x40]
│         │    │       → 更新 pLogSize
│         │    │
│         │    ├─ Write(prefix_len, 2)       ← 格式串前缀长度
│         │    ├─ Write("scalar = %+08.3f", 16) ← 格式串前缀
│         │    ├─ Write('\0', 1)             ← 结束符
│         │    │
│         │    └─ 返回 (idx 现在指向 'f' 之后)
│         │
│         ├─ Write(NORMAL=1, 1)              ← 后缀段开始
│         ├─ Write(suffix_len, 2)            ← 后缀长度
│         ├─ Write("\n\0", 2)                ← 后缀内容
│         └─ Write(END=0, 1)                 ← 结束标记
│
└─ DCCI(null, 1)                              ← cache flush
```

**整个过程在运行时执行**。每次 `cce::printf` 调用都要走一遍：
1. fix stack → DTData
2. 格式串逐字符扫描
3. 多次 `Write()` 逐字节拷贝
4. DCCI flush

这在 AI Core 上产生了不小的开销，但对于调试来说是可以接受的（调试 kernel 本身不以性能为目标）。

### 3.8 TPRINT：Tile 打印宏

`TPRINT(src)` 是 ccelib 提供的 Tile 打印宏（展开为 C++ 模板函数），用于打印整个 Tile 的全部元素。其在 EmitC 路径的降级非常简单：

```cpp
// pto.tprint ins(%tile)  →  TPRINT(tile)
struct PTOPrintToTPRINT : public OpConversionPattern<pto::TPrintOp> {
  LogicalResult matchAndRewrite(pto::TPrintOp op, OpAdaptor adaptor, ...) {
    // ① 取出 src (Tile/GlobalTensor)，必要时包装为 GlobalTensor
    // ② 如果有 PrintFormat 属性，转成模板参数 (如 Width10_Precision6)
    // ③ 生成 emitc::CallOpaqueOp:
    //      callee = "TPRINT"
    //      templateArgs = [PrintFormat?]
    //      operands = [src, tmp?]
    rewriter.eraseOp(op);
  }
};
```

生成的 C++ 代码只有一行：

```cpp
TPRINT(tile);  // 或 TPRINT<pto::PrintFormat::Width10_Precision6>(tile, tmp);
```

在 bisheng CCE 前端编译时，`TPRINT` 宏展开为：
1. `TPrintTileImpl` 模板函数 — 遍历 tile 元素
2. 对每个元素调用 `cce::printf` — 走标量 print 的完整流程
3. 即：每个元素都要走一遍格式串状态机 + Write() 循环 + DCCI flush

因此 EmitC 路径的 `TPRINT` 本质上是对 `cce::printf` 的批量包装——运行时开销是 `rows × cols` 倍的标量 print。

---

## 4. VPTO 路径详解

VPTO 路径的核心洞察是：**既然 DebugTunnel 协议格式是确定性的，为什么不在编译期就把它全部算好？**

### 4.1 核心思路：编译期内联

用一个具体例子说明。对于以下 PTO IR：

```mlir
pto.print ins("scalar = %+08.3f\n", %value : f32)
```

EmitC 路径生成的运行时行为：

```
运行时: 扫描格式串 → 找到 %f → 确定是 FLOAT → 写 marker 2 → 写 4 字节 → ...
        每次 print 都这样做一遍
```

VPTO 路径在编译期做的事情：

```
编译期: 扫描格式串 → 找到 %f → 确定是 FLOAT=2, dataSize=4, prefixLen=17, suffixLen=2
        直接生成: store i8 2, store i32 <float bits>, store i16 17, ...
        运行时: 只需要执行这些 store 指令，零分支零循环
```

类比：EmitC 像是每次出门才查地图、规划路线；VPTO 像是提前把路线背下来，出门时直接走。

### 4.2 编译管线分步解析

整个 VPTO print lowering 分 5 步，在 `lowerVPTOOps()` 中完成：

```
Step 1: collectAndCreatePrintfStringGlobals  ← 预扫描格式串，预创建 global
Step 2: addDTDataParamToEntryFunctions      ← 声明 intrinsics + 追加 DTData 参数
Step 3: applyPartialConversion              ← 所有 pattern（包括 print）同时运行
Step 4: injectPrintPrologue                 ← 注入 fix stack 初始化 + kernelWriteType
Step 5: materializeDecls                    ← 补全延迟声明
```

代码位置：`lib/PTO/Transforms/VPTOLLVMEmitter.cpp` / `VPTOCANN900LLVMEmitter.cpp`

#### Step 1: collectAndCreatePrintfStringGlobals — 预扫描

```cpp
static void collectAndCreatePrintfStringGlobals(ModuleOp module, LoweringState &state) {
  // 1.1 扫描所有 pto.print 的格式串 → 去重创建 LLVM::GlobalOp
  module.walk([&](pto::PrintOp printOp) {
    StringRef fmt = printOp.getFormat();
    std::string globalName = "_ptoas_printf_fmt_" + std::to_string(seen.size());
    state.stringGlobals.push_back({fmt.str(), globalName});
    state.usesPrint = true;
  });

  // 1.2 扫描 pto.tprint → 预创建硬编码格式串 global
  //     （如 "%6.2f"、"%6d"、TPRINT header 横幅等）
  module.walk([&](pto::TPrintOp tprintOp) {
    state.usesPrint = true;
    hasTPrint = true;
  });
  if (hasTPrint) {
    // 预创建 TPRINT 专用格式串: header, shape, valF32, valInt, ...
  }
}
```

为什么需要预扫描？因为后续的 `applyPartialConversion` 中的 pattern 需要 `LLVM::AddressOfOp` 引用这些 global symbol——必须在 dialect conversion 之前创建好。

#### Step 2: addDTDataParamToEntryFunctions — 追加 DTData 参数

```cpp
static LogicalResult addDTDataParamToEntryFunctions(ModuleOp module,
                                                     LoweringState &state) {
  if (!state.usesPrint) return success();

  // 声明 CCE intrinsics
  declareFunc("llvm.hivm.get.sycl.fix.stack.object", ...);
  declareFunc("llvm.hivm.DCCI", ...);
  declareFunc("llvm.hivm.GET.BLOCK.IDX", ...);

  // 给每个 kernel entry function 末尾追加 ptr addrspace(1) 参数
  for (func::FuncOp func : entryFuncs) {
    func.insertArgument(idx, llvmPtr1Type, {}, func.getLoc());
    // 同时更新 FunctionType
  }
}
```

这一步模拟 CCE 前端的工作——给 kernel 函数签名加上 DTData 隐藏参数。

**之前**：`func.func @kernel(f32)`
**之后**：`func.func @kernel(f32, !llvm.ptr<1>)`  ← 末尾多了 DTData

#### Step 3: applyPartialConversion — Pattern 匹配

这是 `dialectConversion` 的核心：所有 op 的 lowering pattern 在此**同时运行**。对于 `pto.print`，匹配 `LowerPrintOpPattern`：

```cpp
// 注册 print pattern（和其他 30+ 个 pattern 一起）
patterns.add<LowerPrintOpPattern>(typeConverter, ctx, state);
// patterns.add<LowerMTEOpPattern>(...);
// patterns.add<LowerVectorOpPattern>(...);
// ...
applyPartialConversion(module, target, std::move(patterns));
```

#### Step 4: injectPrintPrologue — 入口初始化

在 dialect conversion **完成后**，在所有 kernel 函数入口注入 prologue BB：

```cpp
static LogicalResult injectPrintPrologue(ModuleOp module, LoweringState &state) {
  for (func::FuncOp func : entryFuncs) {
    Value dtDataArg = func.getArgument(func.getNumArguments() - 1);

    // 在 entry body 最前面插入:
    //   %fix = call @get.sycl.fix.stack.object()
    //   if (%dtdata == null) → store 0 to fix stack
    //   else → store %dtdata to fix stack + kernelWriteType = 2
  }
}
```

### 4.3 LowerPrintOpPattern 逐行解读

代码位置：`VPTOLLVMEmitter.cpp:9221-9492`（两个 emitter 完全一致）

#### 第一段：编译期解析格式串（共享分析函数）

VPTO 路径的格式串解析已提取为共享函数 `analyzePrintFormat`，定义在 `include/PTO/Transforms/PrintEncoding.h`，verifier、预扫描、lowering 共用同一份解析逻辑：

```cpp
// PrintEncoding.h — 共享分析
enum class PrintConversionKind : uint8_t {
  Float,        // %f, %F, %e, %E, %g, %G, %a, %A
  SignedInt,    // %d, %i
  UnsignedInt,  // %u, %x, %X, %o
};

struct PrintFormatInfo {
  PrintConversionKind conversion;     // 转换类型
  uint16_t prefixBytes;               // 前缀字节数（含 '\0'）
  uint16_t suffixBytes;               // 后缀字节数（含 '\0'）

  unsigned getDataSize() const {      // 协议数据大小: float→4, int→8
    return (conversion == PrintConversionKind::Float) ? 4 : 8;
  }

  uint32_t getRecordSize() const {    // 完整协议记录大小
    unsigned ds = getDataSize();
    return 1 + ds + 2 + prefixBytes + 1 + 2 + suffixBytes + 1;
  }
};

FailureOr<PrintFormatInfo> analyzePrintFormat(StringRef format);
```

对于格式串 `"scalar = %+08.3f\n"`，`analyzePrintFormat` 编译期算出：
- `conversion = Float`
- `prefixBytes = 18`（`"scalar = %+08.3f\0"` 共 18 字节）
- `suffixBytes = 2`（`"\n\0"` 共 2 字节）
- `getDataSize() = 4`（float→4 字节）
- `getRecordSize() = 30`（整个 protocol）

lowering pattern 中直接使用：

```cpp
auto formatInfo = analyzePrintFormat(op.getFormat());
if (failed(formatInfo))
  return op.emitError("...");
int64_t dataSize = formatInfo->getDataSize();
int64_t fmtPrefixLen = formatInfo->prefixBytes;
int64_t fmtSuffixLen = formatInfo->suffixBytes;
int64_t recordSize = formatInfo->getRecordSize();
```

#### 第二段：三层嵌套 scf::IfOp 守卫

从 `d362822d` 开始，print lowering 使用**嵌套 `scf::IfOp`** 替代早期的 flat `bothOk=dtNotNull&&lrNotNull` 结构。这避免了"在 null check 之前解引用 DTData"的安全问题：

```cpp
  Value dtDataArg = func.getArgument(func.getNumArguments() - 1);

  // === 外层: DTData != null ===
  auto dtNotNull = ICmpOp(NE, dtDataArg, nullPtr1);
  auto outerIf = scf::IfOp(dtNotNull, /*withElseRegion=*/false);
  // 在 then region 内:

  // === 中层: safe to deref — load LogWholeRegion ===
  auto lrAddr = GEPOp(DTData, 0);          // safe: DTData 已确认非 null
  auto logRegion = LoadOp(lrAddr);
  auto lrNotNull = ICmpOp(NE, logRegion, null);
  auto innerIf = scf::IfOp(lrNotNull, /*withElseRegion=*/false);
  // 在 then region 内:

  // === 内层: overflow check (with else → 溢出时只更新计数) ===
  auto newPls = AddOp(pLogSize, recordSize);
  auto overflow = ICmpOp(UGT, newPls, logBufSize);
  auto overflowIf = scf::IfOp(overflow, /*withElseRegion=*/true);

  // Overflow then: 只更新 pLogSize（CCE 安全策略）
  { StoreOp(newPls, logBufBase); }

  // Overflow else: 写入完整 protocol
  { ... }
```

嵌套 `scf::IfOp` 的优点：
1. 在 null check 之后才解引用，消除潜在的 null dereference
2. 不需要 `splitBlock` 创建多 BB（兼容 `scf.if` 内部的 print）
3. MLIR 自动处理 region 边界和 yield

#### 第三段：标量编码 + protocol 写入

数据值通过 `encodePrintScalar` 统一编码，同样定义在 `PrintEncoding.h`：

```cpp
// 类型无关的标量编码
auto enc = encodePrintScalar(rewriter, loc, scalarType, scalar,
                              formatInfo->conversion);
// enc.bits:     i32 (float) 或 i64 (int) — ready for byte extraction
// enc.byteWidth: 4 (float) 或 8 (int)
// enc.marker:    2 (FLOAT) 或 3 (INT)

// [0] Type marker
storeI8(writePtr, 0, enc->marker);

// [1..N] Data bytes（LE）
for (unsigned i = 0; i < enc->byteWidth; ++i) {
  auto shifted = LShrOp(enc->bits, i*8);
  auto byte = TruncOp(shifted, i8);
  storeI8(writePtr, 1 + i, byte);
}

// 格式串前缀长度 + 前缀字节
storeI8(writePtr, fmtOff,   fmtLenLow);
storeI8(writePtr, fmtOff+1, fmtLenHigh);
for (int i = 0; i < fmtPrefixLen; ++i) {
  auto ch = LoadOp(GEPOp(fmtGlobal, i));
  storeI8(writePtr, fmtOff+2+i, ch);
}

// NORMAL=1 + 后缀长度 + 后缀字节
storeI8(writePtr, normalOff, 1);
// ...
// END=0
storeI8(writePtr, endOff, 0);

// 更新 pLogSize
StoreOp(newPls, logBufBase);
```

`encodePrintScalar` 的编码规则：

| 源类型 | 转换 | 输出 bits | byteWidth | marker |
|--------|------|-----------|-----------|--------|
| f16/bf16 | `fpext → f32 → bitcast → i32` | i32 | 4 | 2 (FLOAT) |
| f32 | `bitcast → i32` | i32 | 4 | 2 (FLOAT) |
| f64 | `fptrunc → f32 → bitcast → i32` | i32 | 4 | 2 (FLOAT) |
| i8/i16/i32 | `sext → i64` (SignedInt) 或 `zext → i64` (UnsignedInt) | i64 | 8 | 3 (INT) |
| i64 | 无扩展 | i64 | 8 | 3 (INT) |

> **注意**：i64 当前被 verifier 禁用，因 DebugTunnel 协议仅传输低 32 位（`%d` 格式期望 4 字节 int）。等协议升级后重新启用。

### 4.4 injectPrintPrologue：入口初始化

代码位置：`VPTOLLVMEmitter.cpp:4335-4412`（两个 emitter 共享同一份实现）

Dialect conversion 后，entry function 已经是 `LLVM::LLVMFuncOp`。我们在函数体最前面注入：

```llvm
; 注入后的 entry block:
entry:
  %fix = call ptr @llvm.hivm.get.sycl.fix.stack.object()
  %is_null = icmp eq ptr addrspace(1) %dtdata, null
  br i1 %is_null, label %init_null, label %init_ok

init_null:
  store i64 0, ptr %fix
  br label %body

init_ok:
  %dt_i64 = ptrtoint ptr addrspace(1) %dtdata to i64
  store i64 %dt_i64, ptr %fix          ; DTData 地址 → fix stack
  store i32 2, ptr addrspace(1) %dtdata+24  ; kernelWriteType = AiV
  br label %body

body:                                  ; 原始函数体从这里开始
  ; ... 其他代码 + pto.print 展开的 protocol 写入 ...
```

### 4.5 生成的 LLVM IR 形态

最终 PTOAS 生成的 LLVM IR（`ptoas_gen_vpto_print.ll`）的形态：

```llvm
; 全局格式串
@_ptoas_printf_fmt_0 = private constant [18 x i8] c"scalar = %+08.3f\0A\00"

; CCE intrinsic 声明
declare i64 @llvm.hivm.get.block.idx()
declare void @llvm.hivm.DCCI(ptr addrspace(1), i64)
declare ptr @llvm.hivm.get.sycl.fix.stack.object()

define void @print_scalar_kernel_mix_aiv(float %0, ptr addrspace(1) %1) {
  ; === Prologue (编译器注入) ===
  %fix = call ptr @get.sycl.fix.stack.object()
  if %dtdata == null → store 0 to fix, jump to merge
  else → store %dtdata to fix, kernelWriteType=2

  ; === Null + Overflow 检查 ===
  if %dtdata == null || %logRegion == null → skip
  %newPls = %pLogSize + 30
  if %newPls > %logBufSize → store %newPls, skip  (只更新计数)

  ; === Protocol 写入 (编译期展开) ===
  store i8 2,   ptr %buf+0        ; FLOAT marker
  %bits = bitcast float to i32
  %b0 = lshr %bits, 0;  %b0_trunc = trunc %b0 to i8; store %b0_trunc, %buf+1
  %b1 = lshr %bits, 8;  %b1_trunc = trunc %b1 to i8; store %b1_trunc, %buf+2
  %b2 = lshr %bits, 16; %b2_trunc = trunc %b2 to i8; store %b2_trunc, %buf+3
  %b3 = lshr %bits, 24; %b3_trunc = trunc %b3 to i8; store %b3_trunc, %buf+4
  store i8 17,  ptr %buf+5        ; fmt_len = 17 (LE)
  store i8 0,   ptr %buf+6
  ; 16 条 load+store (格式串前缀)
  store i8 0,   ptr %buf+23       ; '\0' 前缀结束符
  store i8 1,   ptr %buf+24       ; NORMAL marker
  store i8 2,   ptr %buf+25       ; rem_len = 2 (LE)
  store i8 0,   ptr %buf+26
  ; 2 条 load+store (后缀 "\n\0")
  store i8 0,   ptr %buf+29       ; END marker

  ; === 更新计数 + Flush ===
  store i64 %newPls, ptr %logBuf
  call void @llvm.hivm.DCCI(null, 1)
  ret void
}
```

**关键特征**：
- 没有 `cce::printf` 调用——零未定义符号
- 没有运行时格式解析——所有长度都是编译期常量
- 没有循环——每个字节都是独立的 `load + store`
- 唯一的函数调用是 DCCI flush（必须的硬件 intrinsic）

### 4.6 LowerTPrintOpPattern：Tile 数据打印

代码位置：`VPTOLLVMEmitter.cpp:9510-9683`（两个 emitter 共享同一份实现）

对标 EmitC 的 `TPRINT` 宏，VPTO 路径在 MLIR 层直接生成嵌套循环 + 逐元素 DebugTunnel 写入。核心策略与 `LowerPrintOpPattern` 一致——**编译期内联**，但不展开循环（保留 `scf.for` 结构避免大 tile 时代码膨胀）。

#### 总体结构

```
LowerTPrintOpPattern::matchAndRewrite
  │
  ├─ [预处理] 解析 Tile 类型、dtype、shape、元素字节数
  ├─ [Null check] scf::IfOp(DTData && LogWholeRegion both non-null)
  ├─ [地址计算] LogBuffer + block_idx * stride + 64 header
  ├─ [UB 虚拟地址] GET.SYS.VA.BASE + 0x80000 → ubBaseElemOffset
  ├─ [Overflow check] scf::IfOp(totalRecSize > remaining)
  │    ├─ Then: 只更新 pLogSize (CCE 安全策略)
  │    └─ Else:
  │         ├─ 外层 scf.for row = 0 → rows
  │         │    └─ 内层 scf.for col = 0 → cols
  │         │         ├─ 计算虚拟元素偏移 + Load
  │         │         ├─ 写 marker (FLOAT=2 / INT=3)
  │         │         ├─ 写数据 bytes (LE: lshr + trunc × N)
  │         │         ├─ 写格式串长度 (i16 LE)
  │         │         ├─ 写格式串前缀 (逐字节 load global → store)
  │         │         ├─ 写 END marker (0)
  │         │         └─ scf.yield writeOff + elemRecordSize
  │         └─ 更新 pLogSize + DCCI flush
  └─ eraseOp
```

#### 类型判定与每条记录大小（编译期确定）

```cpp
auto srcType = dyn_cast<pto::TileBufType>(op.getSrc().getType());
bool isFloat = isa<FloatType>(elemType);
int64_t dataSize = isFloat ? 4 : 8;     // float→4, int→8
int64_t elemBytes = isFloat ? (elemType.isF16() ? 2 : 4)
                            : (elemType.getIntOrFloatBitWidth() / 8);
int64_t fmtPrefixLen = isFloat ? 6 : 4;  // "%6.2f"→6, "%6d"→4
int64_t elemRecordSize = 1 + dataSize + 2 + fmtPrefixLen + 1;
// = marker(1) + data + fmtLen(2) + fmtPrefix + end(1)
```

对于 f32 tile：每条记录 1 + 4 + 2 + 6 + 1 = **14 字节**
对于 i32 tile：每条记录 1 + 8 + 2 + 4 + 1 = **16 字节**

#### UB 虚拟地址计算

昇腾 AI Core 的 UB 通过虚拟地址访问，不能直接用指针。每个元素在 UB 中的虚拟地址：

```
ub_base = (GET.SYS.VA.BASE() + 0x80000) / elem_bytes
elem_virt_addr = ub_base + (row * cols + col)
```

对应的 IR 生成：

```cpp
// GET.SYS.VA.BASE() 返回系统 VA 基址
auto sysva = rewriter.create<LLVM::CallOp>(loc, i64Type,
    sysVaBaseFunc.getSymName(), ValueRange{});
// +0x80000: 用户可用的 UB 偏移
auto ubOff = rewriter.create<LLVM::ConstantOp>(loc, i64Type,
    rewriter.getI64IntegerAttr(0x80000));
auto baseAddr = rewriter.create<LLVM::AddOp>(loc, i64Type,
    sysva.getResult(), ubOff);
// 除以元素字节数 → 元素索引单位的基址
auto elemSizeVal = rewriter.create<LLVM::ConstantOp>(loc, i64Type,
    rewriter.getI64IntegerAttr(elemBytes));
ubBaseElemOffset = rewriter.create<LLVM::UDivOp>(loc, i64Type,
    baseAddr.getResult(), elemSizeVal);
// 用 GEP ptr addrspace(6) 通过虚拟偏移加载元素
auto elemPtr = rewriter.create<LLVM::GEPOp>(loc, ptr6Type,
    elemType, tileDataBase, ValueRange{virtElemOff});
Value elemVal = rewriter.create<LLVM::LoadOp>(loc, elemType, elemPtr);
```

#### 数据写入：统一编码

TPrint 复用与 PrintOp 相同的 `encodePrintScalar` 辅助函数（`PrintEncoding.h`），实现类型无关的协议编码。f16 通过 `FPExtOp → f32 → BitcastOp → i32` 写入 4 字节，i32 通过 `SExtOp → i64` 写入 8 字节：

```cpp
Value elemVal = rewriter.create<LLVM::LoadOp>(loc, elemType, elemPtr);
// 复用 print/scalar 的编码逻辑
auto enc = encodePrintScalar(rewriter, loc, elemType, elemVal);
if (failed(enc)) return failure();
// enc->marker → store i8
// enc->bits  → lshr + trunc × N 字节 LE
```

#### 与 LowerPrintOpPattern 的关键差异

| 维度 | LowerPrintOpPattern | LowerTPrintOpPattern |
|------|---------------------|----------------------|
| 格式串 | 用户自定义（从 op 属性读取） | 固定硬编码（3 种：f32/f16/int） |
| 数据源 | 函数参数（scalar） | Tile 元素（通过 UB 虚拟地址 Load） |
| 写入次数 | 每个 print 调用写 1 条记录 | 每个 print 调用写 rows×cols 条记录 |
| 控制流 | 纯直线 store（零循环） | 嵌套 `scf.for`（保留循环结构） |
| 循环展开 | N/A | 不展开——LLVM 后端可根据 target 决定 |
| Flush 策略 | 每条记录后 DCCI | 所有元素写完后单次 DCCI |
| 格式串长度 | 编译期从用户格式串计算 | 编译期常量（"%6.2f"=6, "%6d"=4） |

### 4.7 LowerAllocTileOpPattern + TileBufType 转换

`pto.alloc_tile` 在 VPTO 路径中不实际分配 UB 内存——物理地址由 consumer op（如 `tprint`、`vlds`、`vsts`）通过虚拟地址计算得出。`LowerAllocTileOpPattern` 生成一个类型正确的 null ptr 占位符：

```cpp
auto ptr6Type = LLVM::LLVMPointerType::get(rewriter.getContext(), 6);
auto nullPtr = rewriter.create<LLVM::ZeroOp>(op.getLoc(), ptr6Type);
rewriter.replaceOp(op, nullPtr.getResult());
```

配套的 `VPTOTypeConverter` 将 `TileBufType` 映射为 `LLVM::LLVMPointerType<6>`（addrspace(6) = UB 地址空间）：

```cpp
if (isa<pto::TileBufType>(type))
  return LLVM::LLVMPointerType::get(context, 6);
```

此外，`InsertTemplateAttributes` 和 `ExpandTileOp` 中显式跳过 `PrintOp` 和 `TPrintOp`——这些调试 op 不需要模板展开，直接由 VPTO lowering pattern 处理。

---

## 5. 两条路径的逐项对比

| 维度 | EmitC 路径 | VPTO 路径 |
|------|-----------|----------|
| **生成目标** | C++ 源码 (`cce::printf` / `TPRINT` 调用) | LLVM IR (`store` 指令序列 / `scf.for` + store) |
| **编译工具** | bisheng CCE 前端 (编译 C++) | PTOAS 直接生成 LLVM IR |
| **依赖** | ccelib 头文件完整工具链 | 只依赖 CCE intrinsic (fix stack, DCCI, GET.BLOCK.IDX, GET.SYS.VA.BASE) |
| **标量 print 方式** | `cce::printf("fmt", val)` 运行时解析格式串 | 编译期解析格式串 → 展开为直线 store |
| **Tile print 方式** | `TPRINT(tile)` 宏 → 逐元素 `cce::printf` → 运行时解析 | 嵌套 `scf.for` → 逐元素编译期展开 → 直线 store |
| **格式串解析时机** | 运行时 | 编译期 |
| **格式串解析方式** | PrintState 状态机逐字符扫描 | C++ 代码在 pattern::matchAndRewrite 中一次性解析 |
| **数据写入方式** | Write() while 循环逐字节拷贝 | 编译期展开为 N 条独立的 `store i8` |
| **DTData 注入方式** | CCE 前端自动追加隐藏参数 | `addDTDataParamToEntryFunctions` 手动追加 |
| **fix stack 初始化** | 独立的 `DebugTunnel_Initialize` 函数 | `injectPrintPrologue` 内联到 kernel 入口 |
| **DCCI flush 位置** | `DebugTunnel_Finish` 中统一 flush | 每次 print/tprint 后立即 flush |
| **kernel 函数大小** | 很小（thin wrapper，复杂逻辑在外部） | 较大（所有逻辑内联在一个函数里） |
| **未定义符号** | `cce::printf` / `TPRINT` (C++ mangled)，需 CCE 工具链处理 | 零未定义符号 |
| **运行时开销** | 状态机 + 多次 Write() 循环（Tile print: rows×cols 倍） | 直线 store 序列 / scf.for + 直线 store |
| **代码行数 (LLVM IR)** | ~938 行 (14 个函数) | ~160 行 (标量) / ~250 行 (tile) |

### 格式串处理对比

以 `printf("scalar = %+08.3f\n", 3.25f)` 为例：

```
EmitC (运行时):
══════════════
PrintState state("scalar = %+08.3f\n")
  ┌─ while (c != '%') idx++      ; 循环 9 次跳过 "scalar = "
  ├─ while (is_flag) idx++       ; 消耗 '+'
  ├─ while (is_digit) idx++      ; 消耗 '0','8'
  ├─ if ('.') while (isdigit)    ; 消耗 '3'
  ├─ if (is_length) idx++        ; 无长度修饰符
  └─ switch ('f') → Write(FLOAT, 4B)  ; 5 次 Write() 循环调用

VPTO (编译期):
══════════════
LowerPrintOpPattern::matchAndRewrite 中:
  size_t prefixLen = 0, pos = 0;
  while (pos < fmtStr.size()) {
    if (fmtStr[pos] == '%') {    ; 找到 '%'
      prefixLen = pos;
      pos++; while (flags)  pos++;   ; 消耗 '+'
      while (digits) pos++;          ; 消耗 '0','8'
      if ('.') while (digits) pos++; ; 消耗 '3'
      if (length) pos++;
      pos++; prefixLen = pos;
    }
  }
  ; 结果: prefixLen=17, suffixLen=2, dataSize=4, typeMarker=2
  ; 直接生成 30 条 store 指令
```

---

## 6. 关键技术决策

### 6.1 为什么用 `scf::IfOp` 而非手写 CFG？

最初版本（v1-v2）用 `splitBlock` 创建多 block CFG（null check → overflow check → write → merge）。但这有两个问题：

1. **scf.if 内的 print 无法 splitBlock**：`scf::IfOp` 的 region 是 `SizedRegion<1>`，不能容纳多个 block。解决方案是预创建 outline helper stub，if-region 内用 `CallOp` 调用。

2. **get_block_idx + print 的 intrinsic 冲突**：`addDTDataParamToEntryFunctions` 创建 `LLVM::LLVMFuncOp @llvm.hivm.GET.BLOCK.IDX`，然后 `LowerRuntimeQueryOpPattern` 又尝试创建同名 `func::FuncOp`，触发符号冲突。解决方案：`materializeDecls` 跳过已有 LLVM func 的 symbol。

最终版本用嵌套 `scf::IfOp` 替代手写 CFG（commit `fcf32e21`），简化了结构并避免了上述问题。

### 6.2 为什么 host stub 编译要切换 driver 模式？

CC1 模式 (`-cc1`) 无法处理 C++ 标准库 include path。DebugTunnel host 端头文件依赖 `<string>` 等 STL 头文件，只能用 bisheng **driver 模式** (`-xcce --cce-enable-print`)，因为 driver 会自动管理所有系统头文件路径。

代码位置：`tools/ptoas/ObjectEmission.cpp:553-569`

```cpp
static bool compileHostStubToObject(...) {
  // 当 module 使用了 pto.print / pto.tprint 时，改用 driver 模式
  if (usesPrint)
    return compileHostStubToObjectDriverMode(stubPath, outObjPath, moduleId,
                                              targetCPU, runtimeObjPath, stderrPath, diagOS);
  // ...
}
```

### 6.3 整数符号扩展与 PrintConversionKind

CCE 的 print payload 解析器期望所有整数都是 **8 字节**（i64）。对于 `i8`、`i16`、`i32` 类型的 print 参数，需要先扩展到 `i64` 再写入。**符号扩展还是零扩展取决于格式说明符**：

```cpp
// PrintEncoding.h
inline FailureOr<PrintScalarEncoding>
encodePrintScalar(ConversionPatternRewriter &rewriter, Location loc,
                  Type scalarType, Value scalar,
                  PrintConversionKind kind = PrintConversionKind::SignedInt) {
  // ...
  if (w < 64) {
    if (kind == PrintConversionKind::UnsignedInt)
      enc.bits = rewriter.create<LLVM::ZExtOp>(loc, i64Type, scalar);  // %u/%x/%o
    else
      enc.bits = rewriter.create<LLVM::SExtOp>(loc, i64Type, scalar);  // %d/%i
  }
}
```

| 格式说明符 | PrintConversionKind | 扩展方式 | i8=0xff 结果 |
|-----------|--------------------|----------|-------------|
| `%d`, `%i` | SignedInt | `sext → i64` | -1 (正确) |
| `%u`, `%x`, `%X`, `%o` | UnsignedInt | `zext → i64` | 255 (正确) |

> **注意**：MLIR 的 integer type 默认为 signless，`IntegerType::isUnsigned()` 对 signless integer 恒返回 `false`。因此不能依赖 MLIR 的类型属性来决定扩展方式，必须由格式串解析出的 `PrintConversionKind` 驱动。

当前 verifier 禁用了 i64 类型——DebugTunnel 协议中 `%d` 格式仅传输低 32 位，64 位整数会被截断。等协议升级到支持 64 位后再重新启用。

### 6.4 统一格式解析：analyzePrintFormat

VPTO 路径最初在 verifier 和 lowering 各维护一套独立的格式串 parser，存在 ~~两处重复代码~~ 和 `%%` 处理不一致的 bug。当前版本已提取为共享函数 `analyzePrintFormat`：

```cpp
// PrintEncoding.h — verifier、pre-scan、lowering 共用
FailureOr<PrintFormatInfo> analyzePrintFormat(StringRef format);
```

**统一前**：verifier（~60 行）+ VPTOLLVMEmitter（~30 行）+ VPTOCANN900LLVMEmitter（~30 行）≈ 120 行重复代码
**统一后**：`analyzePrintFormat`（~40 行），三处各 ~5 行调用

`PrintFormatInfo` 编译期确定所有 layout 参数（prefixBytes, suffixBytes, recordSize），消除了运行时格式串扫描。Host 端 `printf` 仍然负责 flags/width/precision 的语义——VPTO 只需按协议格式把格式串原样写入字节流。

### 6.5 多 block print 的独立性

### 6.5 多 block print 的独立性

每个 AI Core block 有自己独立的 log buffer 区域：

```
LogWholeRegion (HBM 中的基址)
  │
  ├─ Block 0: [pLogSize(8B) | pad(56B) | data(N)]
  ├─ Block 1: [pLogSize(8B) | pad(56B) | data(N)]
  └─ Block N: ...
```

VPTO 通过 `get_block_idx() * stride` 计算每个 block 的偏移，各 block 写各自区域，互不干扰。这个逻辑和 EmitC 的 `Write()` 完全一致。

### 6.6 使用要点与常见陷阱

#### 6.6.1 格式串约束

- **恰好一个转换说明符**：`pto.print` 只能有一个 `%d`/`%f` 等占位符（`%%` 不算）
- **类型匹配**：float 值用 `%f/%e/%g`，int 值用 `%d/%i/%u/%x/%o`
- **i64 暂不支持**：DebugTunnel 协议仅传输低 32 位，待协议升级

```mlir
// ✅ 正确
pto.print ins("value = %+08.3f\n", %val : f32)
pto.print ins("idx = %d\n", %idx : i32)

// ❌ 错误：两个转换说明符
pto.print ins("[block %d] value = %f\n", %val : f32)
// → 拆成两个 pto.print

// ❌ 错误：零转换说明符
pto.print ins("inside loop\n", %mode : i32)
// → 改为 pto.print ins("inside loop, mode=%d\n", %mode : i32)
```

#### 6.6.2 Kernel 至少需要一个用户参数

CCE runtime 只在 kernel 有 ≥1 个用户参数时才自动注入 DTData 隐藏参数。零参数 kernel 的 `pto.print`/`pto.tprint` 不会有输出：

```mlir
// ❌ 零参数 — DTData 不会被注入，print 无输出
func.func @my_kernel() attributes {pto.entry} { ... }

// ✅ 至少一个参数
func.func @my_kernel(%dummy: f32) attributes {pto.entry} { ... }
```

#### 6.6.3 主机端 ACL 初始化

使用 `run_vpto_tprint_validation.sh` 或 `run_host_vpto_validation.sh` 时，host runner 必须调用 `aclInit()` + `aclrtSetDevice(0)` + `aclrtCreateStream()` 来初始化 ACL runtime。缺少 ACL 初始化会导致 `PrintPayloadData` 和 `DebugTunnelData` 设备内存分配失败，打印输出为空。

#### 6.6.4 TileLib daemon 环境变量

使用 `pto.alloc_tile` / `pto.get_block_idx` 等 PTODSL op 的测试需要 TileLib daemon。如果 Python MLIR 绑定不在默认路径，需要设置：

```bash
export MLIR_PYTHON_ROOT="/path/to/llvm-project/build-shared/tools/mlir/python_packages/mlir_core"
export PTO_INSTALL_DIR="/path/to/PTOAS/install"
export PYTHONPATH="$MLIR_PYTHON_ROOT:$PTO_INSTALL_DIR:$PYTHONPATH"
```

#### 6.6.5 GEP `nuw` 兼容性

MLIR（基于 LLVM 21）输出的 `getelementptr inbounds nuw` 不被 bisheng（基于 LLVM 15）识别。E2E 脚本第 127 行有自动 workaround：
```bash
sed -i 's/inbounds nuw/inbounds/g' "${OUT_DIR}/kernel.ll"
```

#### 6.6.6 类型支持矩阵

| 类型 | `pto.print` | `pto.tprint` | 备注 |
|------|:---:|:---:|------|
| f16 | ✅ | ✅ | fpext→f32→4B |
| bf16 | ✅ | ❌ | fpext→f32→4B, tile 待硬件验证 |
| f32 | ✅ | ✅ | bitcast→4B |
| f64 | ✅ | ❌ | fptrunc→f32→4B |
| i8/i16/i32 | ✅ | ✅ (i32) | sext/zext→i64→8B |
| i64 | ❌ | — | 协议截断，暂禁用 |

---

## 总结

EmitC 和 VPTO 两条路径解决的是同一个问题（`pto.print` / `pto.tprint` → DebugTunnel protocol），但采用了截然不同的策略：

- **EmitC** 是"委托"策略：生成 C++ 代码，让 CCE 编译器去处理所有复杂细节。优点是复用成熟的 ccelib 实现（`cce::printf` 和 `TPRINT` 宏），缺点是依赖完整的 CCE 工具链，且 Tile print 的运行时开销是 `rows × cols` 倍的标量 print。
- **VPTO** 是"内联"策略：在编译期复现 CCE 编译后的最终效果。优点是无运行时开销、无外部依赖，缺点是需要精确理解 DebugTunnel ABI 的每一个细节（包括 UB 虚拟地址计算和 `GET.SYS.VA.BASE` 的用法）。

VPTO 的 `LowerPrintOpPattern` 和 `LowerTPrintOpPattern` 本质上做了 CCE 前端对 `cce::printf` / `TPRINT` 的模板展开工作——但不是在 C++ 编译期做，而是在 MLIR lowering 阶段做。对于标量 print，结果是一段纯粹的、无分支的、无函数调用的 LLVM IR；对于 Tile print，结果是嵌套 `scf.for` 循环 + 逐元素直线 store，末尾单次 DCCI flush。两者都可以直接被 bisheng 后端编译为 AI Core 指令。
