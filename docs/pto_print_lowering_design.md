# PTO Print Lowering：从 ccelib 到 VPTO

## 1. 这份文档解决什么问题

`pto.print` 和 `pto.tprint` 最终都要让 Host 看到 AI Core 上产生的调试文本。VPTO 后端没有直接调用设备端 `printf`，而是复现 EmitC 路径中 ccelib 的核心行为：按照 DebugTunnel 协议把“格式信息 + 原始数据”写入 GM 日志区，再由 Host 回拷、解析和格式化。

理解这套实现，关键是分清三个职责：

- **Host DebugTunnel**：分配 GM 日志区，把地址交给 kernel；kernel 结束后回拷并解析日志。
- **设备端 PrintState/Write**：EmitC 路径在运行时解析格式串，定位当前 block 的日志区并写协议记录。
- **PTOAS VPTO lowering**：在编译期完成同样的格式分析，用 MLIR/LLVM dialect 直接构造地址计算、边界检查和逐字节写入。

两条路径的外部协议相同，主要区别是逻辑发生的时间：

| 阶段 | EmitC + ccelib | VPTO + PTOAS |
|---|---|---|
| 格式串分析 | kernel 运行时由 `PrintState` 状态机完成 | 编译期由共享格式分析完成 |
| 协议构造 | ccelib 模板和 `Write()` 运行时执行 | lowering 生成固定的 MLIR/LLVM 操作 |
| GM 日志地址 | `Write()` 每次写入时计算 | `LowerPrintOpPattern` 生成地址计算 |
| Host 分配与解析 | DebugTunnel runtime | 同一套 DebugTunnel runtime |

## 2. DebugTunnel 的完整生命周期

### 2.1 Host 先分配 GM，并把地址写入 DTData

GM 日志区不是 kernel 内创建的。启用 print 的 Host launch wrapper 在 kernel 启动前调用 DebugTunnel 的打开流程：

```text
DebugTunnel::Open(blockNum)
  -> PrintPayload::OnHostInitialize
       计算每个 block 的日志区大小
       在设备 GM 中分配 LogWholeRegion
       将每个 block 的 pLogSize 初始化为 0
       设置 LogBufferSize 和 BlockNum
  -> 在设备 GM 中分配 DebugTunnelData
  -> 将包含 LogWholeRegion 地址的 DebugTunnelData 拷到设备
  -> 返回设备侧 DTData 指针
```

与 Print 直接相关的字段如下：

```text
DebugTunnelData
  PrintData.LogWholeRegion : GM 日志总区的设备地址
  PrintData.BlockNum       : 启动的 block 数量
  PrintData.LogBufferSize  : 单个 block 的 payload 容量，单位为字节
  PrintData.kernelWriteType: AiC / AiV / Mix
```

因此“GM 地址何时设置”的答案是：**kernel launch 前，Host 分配日志 GM 后，将设备地址写入 `LogWholeRegion`，再随 DTData 一起拷到设备。**

### 2.2 Kernel 初始化 DTData 访问入口

EmitC 路径的 launch 框架在 kernel 前调用 `__DebugTunnel_Initialize(DTData)`。ccelib 将 DTData 的 GM 地址写入 fix stack；后续 `cce::printf` 可通过 `DebugTunnel::GetKernelInstance()` 取回它。

VPTO 路径把 DTData 作为 entry function 的隐藏 GM 指针参数。PTOAS 在函数入口内联等价 prologue：

```text
if DTData == null:
  fix_stack = 0
else:
  fix_stack = ptr_to_int(DTData)
  DTData.PrintData.kernelWriteType = AiV
```

这样既保持了 ccelib 的 fix-stack 约定，也让 VPTO 生成的 Print 可以直接使用 entry 参数。

### 2.3 Kernel 计算当前 block 的 GM 子区并写入

日志总区按 block 切分。每个子区前 64 字节是 header，其中首 8 字节保存 `pLogSize`，payload 从偏移 64 开始：

```text
stride           = LogBufferSize + 64
LogBuffer(block) = LogWholeRegion + get_block_idx() * stride
pLogSize         = *(uint64_t *)(LogBuffer + 0)
writePtr         = LogBuffer + 64 + pLogSize
```

因此“GM 地址何时写”的答案是：**每次执行 Print 时，设备端根据 `LogWholeRegion`、`LogBufferSize` 和 block id 算出 `writePtr`，随后把协议字节直接 store 到这段 GM；最后更新同一子区的 `pLogSize`。**

如果本次记录会超过 `LogBufferSize`，实现不再写 payload，但仍把 `pLogSize` 增加到所需长度。Host 因而能报告日志被截断以及实际需要的空间。

### 2.4 Flush、回拷与 Host 格式化

设备写完后执行 DCCI，使 Host 可见日志内容。kernel 完成后，DebugTunnel 的关闭流程同步 stream，然后：

```text
DebugTunnel::Close(DTData, stream)
  -> 将设备侧 DTData 回拷到 Host
  -> PrintPayload::OnHostFinish
       将 LogWholeRegion 整块 GM 回拷到 Host
       按 block 读取 pLogSize 和 payload
       解析 DebugTunnel 节点并完成 printf 格式化
       释放日志 GM
  -> 释放设备侧 DTData
```

完整数据流可以概括为：

```text
Host 分配 GM
  -> LogWholeRegion 写入 DTData
  -> DTData 作为隐藏参数传给 kernel
  -> kernel 按 block 写协议记录到 GM
  -> DCCI + stream 同步
  -> Host 回拷 GM
  -> Host 解析节点并输出文本
```

## 3. EmitC 路径：ccelib 如何实现 Print

EmitC 生成的 kernel 调用 `cce::printf(format, args...)`。模板展开后，核心由 `PrintState` 和 `Write()` 两部分组成。

### 3.1 PrintState 是格式串状态机

`PrintState` 保存两个状态：格式串地址 `fmt` 和当前扫描位置 `curpos`。每消费一个参数，`operator<<` 从 `curpos` 开始寻找下一个 `%`，然后依次解析：

```text
普通文本 -> % -> flags -> width -> precision -> length -> conversion
```

对应状态转移为：

```text
Start
  -> 扫描到 % 或字符串结尾
  -> ParseFlag      // 0 - + 空格 #
  -> ParseWidth     // 十进制宽度
  -> ParsePrec      // .precision
  -> ParseLength    // l, ll, h, z
  -> 根据 conversion 选择节点类型
       d/i/x/X/o/u -> INT
       a/A/g/G/e/E/f/F -> FLOAT
       s -> STRING
       p -> POINTER
  -> 写入值节点和该转换对应的格式片段
  -> curpos 指向下一段，等待下一个参数
```

PrintState 不在设备端生成最终字符串。它保存格式片段和原始二进制值，让 Host 按格式片段完成最终格式化。

### 3.2 Write() 负责定位 GM 和追加字节

ccelib 的 `Write(data, size)` 每次调用都会：

1. 从 fix stack 取得 DTData。
2. 检查 DTData 和 `LogWholeRegion` 是否为空。
3. 用 block id 计算当前 block 的 `LogBuffer`。
4. 读取 `pLogSize`，得到 payload 追加位置。
5. 在剩余容量内逐字节写入 GM。
6. 更新 `pLogSize`；溢出时保留“所需总长度”。

PrintState 通过多次调用 `Write()` 依次写 marker、值、格式长度和格式内容。职责分离很清楚：**PrintState 决定写什么，Write 决定写到哪。**

### 3.3 DebugTunnel 节点格式

以 `cce::printf("x=%+08.3f\n", 3.25f)` 为例，EmitC 的逻辑记录顺序是：

```text
FLOAT   value=3.25f  fmt_len=10  "x=%+08.3f\0"
NORMAL  len=2        "\n\0"
NORMAL  len=1        "\0"
END
```

`PrintState::WriteFormatString(bufferstart)` 会把从普通文本起点到当前转换符的完整片段写入 FLOAT 节点；`cce::printf` 收尾时再写转换符之后的剩余文本和 END。字符串长度包含结尾的 `\0`。VPTO 为了让每个片段的边界在编译期明确，将同一语义拆成“可选前缀 NORMAL + 数值/转换格式节点 + 可选后缀 NORMAL + END”；两者共享 marker、长度和原始值的 DebugTunnel 基本约定，差异由 lowering 和 Host 解码侧共同配合。

## 4. VPTO 路径：PTOAS 如何用 MLIR 构造等价逻辑

VPTO 后端不链接设备端 PrintState。它观察到 PTO 格式串是编译期属性，因此将 PrintState 的工作前移到编译期，再用 MLIR 构造等价的设备代码。

### 4.1 lowerVPTOOps 的四个阶段

```text
PTO IR
  -> collectAndCreatePrintfStringGlobals
       扫描 Print/TPrint
       为格式字符串创建 LLVM global
  -> addDTDataParamToEntryFunctions
       给每个 entry function 追加 DTData GM 隐藏参数
       声明 fix-stack、block-id 和 DCCI 接口
  -> applyPartialConversion
       LowerPrintOpPattern: Print -> 地址计算 + 协议 store
       LowerTPrintOpPattern: TPrint -> 文本记录 + 循环 + 协议 store
  -> injectPrintPrologue
       在 entry 入口初始化 fix stack 和 kernelWriteType
```

预扫描必须早于 pattern lowering，因为 `LLVM::AddressOfOp` 只能引用已经存在的 module-level global。DTData 参数也必须在类型转换前加入函数签名；prologue 则在转换后生成，方便直接使用 LLVM dialect 的指针和控制流操作。

### 4.2 编译期复现 PrintState

PTOAS 使用共享的格式分析结果描述三个片段：

```text
prefix      : 转换符之前的普通文本
conversion  : 完整转换格式，如 %+08.3f
suffix      : 转换符之后的普通文本
```

分析同时给出 conversion 类别，用于选择 FLOAT、SIGNED INT 或 UNSIGNED INT 编码。lowering 随后生成：

```text
emitTextNode(prefix)       // 可选 NORMAL
encodePrintScalar(value)
emitValueNode(conversion)  // FLOAT 或 INT
emitTextNode(suffix)       // 可选 NORMAL
emit END
```

这与 PrintState 状态机的结果相同，但扫描、分类和记录长度都在编译期确定。设备运行时不再执行 flag/width/precision 状态机。

### 4.3 LowerPrintOpPattern 复现 Write()

`LowerPrintOpPattern` 用嵌套 `scf.if` 构造安全检查，并用 LLVM dialect 构造 GM 地址和 store：

```text
if DTData != null:
  LogWholeRegion = load DTData + 0
  if LogWholeRegion != null:
    LogBufferSize = load DTData + 16
    stride = LogBufferSize + 64
    blockOffset = GET.BLOCK.IDX() * stride
    LogBuffer = gep LogWholeRegion, blockOffset
    pLogSize = load LogBuffer
    newSize = pLogSize + recordSize

    if newSize > LogBufferSize:
      store newSize -> LogBuffer
    else:
      writePtr = gep LogBuffer, 64 + pLogSize
      store protocol bytes -> writePtr
      store newSize -> LogBuffer
      DCCI(null, 1)
```

其中 `LogWholeRegion` 和 `writePtr` 都是 addrspace(1) 的 GM 指针。格式串 global 位于普通常量地址空间，lowering 从 global load 每个字符，再 store 到 GM。

### 4.4 为什么生成逐字节 store

ccelib 的 `Write()` 是通用循环：同一个函数可以写任意长度的数据。PTOAS 已在编译期知道 marker、数据宽度、格式片段和记录总长度，因此可以直接展开：

```llvm
; 示例形态，省略 GEP
store i8 1, %writePtr       ; NORMAL
store i8 3, %writePtr+1     ; len low
store i8 0, %writePtr+2     ; len high
store i8 2, %writePtr+6     ; FLOAT
store i8 %valueByte0, ...
store i8 %valueByte1, ...
store i8 %valueByte2, ...
store i8 %valueByte3, ...
; conversion length and bytes
; suffix node
store i8 0, ...             ; END
```

这样生成代码没有设备端格式解析，也没有通用 memcpy 循环；只保留必要的空指针检查、容量检查、地址计算、store 和 flush。

### 4.5 EmitC 与 VPTO 的逐项对应

| ccelib 行为 | VPTO lowering |
|---|---|
| Host `Open` 分配 GM、设置 `LogWholeRegion` | 继续复用 DebugTunnel Host runtime |
| `OnKernelInitialize` 将 DTData 写入 fix stack | `injectPrintPrologue` 内联相同初始化 |
| `GetKernelInstance` 取 DTData | Print pattern 使用 entry 的 DTData 隐藏参数 |
| PrintState 运行时解析格式串 | `analyzePrintFormat` 编译期解析 |
| `ConvertTo` 规范化参数位宽 | `encodePrintScalar` 生成协议位模式 |
| `Write()` 计算 block GM 地址 | pattern 生成 block-id、stride 和 GEP |
| `Write()` 循环复制数据 | pattern 展开为固定数量的 load/store |
| `cce::printf` 写完记录后执行 DCCI，kernel finish 再保证可见性 | pattern 写完记录后调用 DCCI |
| Host `Close` 回拷并解析 | 继续复用 DebugTunnel Host runtime |

## 5. TPrint：把同一机制扩展到 Tile

`pto.tprint` 不需要新的传输协议。它仍然写 NORMAL、FLOAT/INT 和 END 节点，只是数据来源从一个 SSA 标量变成 Tile 中的全部元素。

### 5.1 VPTO lowering 的结构

```text
解析 Tile dtype、rows、cols
  -> 生成固定的 header 文本记录
  -> 生成 shape 文本记录
  -> 计算 Tile 在 UB 中的访问基址
  -> scf.for row
       -> scf.for col
            -> 从 UB load 一个元素
            -> 写 FLOAT/INT 节点
  -> 更新 pLogSize
  -> 单次 DCCI
```

header、shape 和元素记录的总大小在编译期可知，因此容量检查发生在写入前。所有元素写完后只做一次 flush，避免逐元素同步。

### 5.2 UB 地址与 GM 日志地址不要混淆

TPrint 同时涉及两个地址空间：

- **UB 地址**：数据来源。lowering 通过系统虚拟地址基址和 UB 偏移构造 addrspace(6) 指针，逐元素 load。
- **GM 地址**：日志目的地。仍由 Host 设置 `LogWholeRegion`，kernel 按 block 计算 addrspace(1) 的 `writePtr` 并 store 协议字节。

数据路径是：

```text
Tile element in UB
  -> LLVM load
  -> 转成 DebugTunnel FLOAT/INT 位模式
  -> LLVM store to GM LogWholeRegion
  -> Host 回拷并格式化
```

因此 TPrint 只是复用了标量 Print 的“GM 日志写入端”，并在前面增加 Tile 遍历和 UB load。

## 6. 阅读实现时应抓住的主线

阅读生成 IR 或调试 Print 时，可以按以下顺序定位：

1. Host 是否在 launch 前成功分配 `LogWholeRegion`，并把地址写进 DTData。
2. entry function 是否收到 DTData 隐藏参数，prologue 是否初始化 fix stack。
3. block id、stride 和 GEP 是否得到当前 block 的 GM 日志子区。
4. `pLogSize`、容量检查和 `writePtr = LogBuffer + 64 + pLogSize` 是否正确。
5. 协议节点的 marker、数据位宽、格式长度和 NUL 是否与 ccelib 一致。
6. 写入后是否更新 `pLogSize` 并执行 DCCI。
7. Host 是否在 stream 完成后回拷日志 GM 并运行协议解析。

整套实现的本质是：**保留 ccelib/DebugTunnel 的 Host 生命周期和协议，把设备端 PrintState + Write 的运行时逻辑改写成 PTOAS 在编译期生成的 MLIR/LLVM 地址计算与 store。**
