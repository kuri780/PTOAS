# 从 EmitC 到 VPTO：`pto.print` 是如何实现的

## 1. 问题开始

PTO IR 已经定义了 `pto.print`。我们希望在 kernel 中写下：

```mlir
pto.print ins("scale = %+08.3f\n", %scale : f32)
```

然后在 kernel 运行结束后，从 Host 侧看到格式化文本。

EmitC 后端已经能够处理这个操作，但 VPTO 后端当前不支持。只要 IR 中出现 `pto.print`，VPTO lowering 就无法完成。因此开发目标就是：让 `pto.print` 能够沿 VPTO 路径编译和运行，并得到与 EmitC 路径相同的可观察输出。

开始实现时，第一个问题不是“如何写 lowering pattern”，而是：**AI Core 上的 Print 到底是怎样工作的？**

设备端没有一个可以直接替换 `pto.print` 的普通 `printf` intrinsic。如果只看 PTO op 的语法，很难知道它最终应该 lower 成什么。已有的 EmitC 路径是最可靠的参考，因此第一步是从一个能够正常打印的 EmitC kernel 出发，沿调用和数据流逐层调查。

## 2. 先看 EmitC Print 的执行入口

先从一个最小的标量 Print 开始（同时开启cce-enable-print）：

```mlir
func.func @print_kernel(%value: f32) attributes {pto.entry} {
  pto.print ins("value = %+08.3f\n", %value : f32)
  return
}
```

要理解这个操作，不能只看生成的 kernel。一次 Print 实际上从 Host launch 开始：launch wrapper 先开启 DebugTunnel，准备设备日志空间，再把它交给 kernel。

### 2.1 Host 先开启 DebugTunnel

当检测到cce-enable-print选项开启时，Host就会开启DebugTunnel

EmitC launch 代码的关键执行顺序可以概括为：

```text
DTData = DebugTunnel::Open(blockNum)
  -> 在设备 GM 中准备 Print 日志空间
  -> 将日志空间地址记录到 DebugTunnelData

launch kernel(..., DTData)

DebugTunnel::Close(DTData, stream)
  -> 等待 kernel 完成
  -> 回拷并解析设备日志
```

**Print 开始于 Host；Host 在 kernel 启动前创建日志通路，并把返回的 DTData 作为隐藏参数传给 kernel。**


### 2.2 kernel 调用 `cce::printf`

EmitC 路径会将 `pto.print` 转换成类似下面的 CCE C++：

```cpp
cce::printf("value = %+08.3f\n", value);
```

查看 EmitC kernel 的入口函数，可以看到 `pto.print` 确实变成了 `cce::printf` 调用；与此同时，Host 创建的 DTData 也出现在函数签名末尾：

```llvm
define void @print_kernel(float %value, ptr addrspace(1) %.DTData) {
  call void @_ZN3cce6printf...(
      ptr addrspace(1) @.str, float %value)
  ret void
}
```

这段 IR 把 Host 与设备端的两个入口连接起来了：Host 通过 `DebugTunnel::Open` 得到 DTData，launch 时将它作为 `.DTData` 传入；kernel 内的 `cce::printf` 则负责产生日志内容。

不过，它仍然只是调用点，还不是 Print 的完整实现。IR 又给出了两个需要继续调查的线索：

1. `cce::printf` 不是一条硬件指令，而是 ccelib 提供的 C++ 模板实现。
2. `cce::printf` 的显式参数中没有 `.DTData`，因此 ccelib 必然通过另一条路径取得 Host 传入的日志上下文。

顺着 `cce::printf` 的符号继续查看 ccelib 展开到同一 LLVM 模块中的函数，可以看到更完整的调用关系。
下面只保留用于调查的关键结构：

```llvm
; cce::printf 创建状态对象，并逐个处理参数
define void @cce_printf(..., float %value) {
  ; PrintState state(format)
  call void @PrintState_ctor(...)
  call void @PrintState_operator_float(..., float %value)
  call void @PrintState_finish(...)
  ret void
}

; PrintState 解析格式后，通过 Write 追加协议节点
define void @PrintState_operator_float(...) {
  ; scan flags/width/precision/conversion
  call void @Write(...)
  ret void
}

; Write 通过 fix stack 找回 Host 传入的 DTData，再计算日志地址
define void @Write(...) {
  %fix = call ptr @llvm.hivm.get.sycl.fix.stack.object()
  %dtDataInt = load i64, ptr %fix
  %dtData = inttoptr i64 %dtDataInt to ptr addrspace(1)
  ; load LogWholeRegion, calculate block buffer, append bytes
  ret void
}
```

fix stack：可以把它理解为设备端的一小块约定存储：kernel 入口初始化时将 DTData 地址写入其中，后续 ccelib 函数再通过 `llvm.hivm.get.sycl.fix.stack.object()` 取得这块存储并读回 DTData。这样，`cce::printf`、`PrintState` 和 `Write()` 之间不必逐层传递一个额外的 DTData 参数。
因此， `%fix` 不是日志区地址，而是保存 DTData 地址的位置；从 `%fix` load 并转换为 GM 指针后得到的 `%dtData`，才指向 Host 传入的 DebugTunnelData。

这段调用链把后续需要回答的问题摆了出来：

1. `PrintState` 如何把格式串和值变成协议节点？
2. `Write()` 为什么要从 fix stack 取回 DTData？
3. Host 创建的 DTData 中保存了什么，日志空间如何组织？
4. Host 最后如何把这些节点还原成文本？

接下来沿调用链逐层展开：先看离 `cce::printf` 最近的 `PrintState`，再展开刚才略过的 DebugTunnel 日志布局，最后带着这些信息回来看 `Write()`。

## 3. 沿着 `cce::printf` 调查 ccelib

### 3.1 PrintState 不生成最终字符串

`cce::printf(format, args...)` 的核心是一个 `PrintState` 状态机。它保存格式串地址和当前扫描位置。每消费一个参数，就寻找下一个转换符，并依次解析：

```text
普通文本 -> % -> flags -> width -> precision -> length -> conversion
```

不同 conversion 会被分为不同节点类型：

```text
d/i/x/X/o/u       -> INT
a/A/g/G/e/E/f/F   -> FLOAT
s                  -> STRING
p                  -> POINTER
```

这里最关键的发现是：设备端不会把 `3.25f` 直接格式化成最终文本。`PrintState` 保存的是格式片段和参数的原始二进制值，真正的字符串格式化发生在 Host。

以 `cce::printf("scale = %+08.3f\n", 3.25f)` 为例，设备侧写出的逻辑记录类似：

```text
NORMAL               text="scale =\0"
FLOAT   value=3.25f  format="x=%+08.3f\0"
NORMAL               text="\n\0"

END
```

这解释了为什么 VPTO Print 不能简单地“把字符串逐字符写出去”。设备必须写出 Host 能识别的 DebugTunnel 节点，包括 marker、原始值、格式长度和以 NUL 结尾的格式内容。

`PrintState` 回答了“写什么”，但调用链中的 `Write()` 仍留下了“写到哪里”的问题。尤其是它没有直接使用 kernel 的 `.DTData` 参数，而是先从 fix stack 找回 DTData。要理解这段逻辑，需要暂时离开设备端，先看 DebugTunnel 如何在 Host 和 kernel 之间建立日志通路。

## 4. 展开 DebugTunnel 的日志通路

### 4.1 `Open` 如何准备日志 GM

第 2 章已经看到，日志区不是 kernel 自己创建的。现在为了理解 `Write()` 的地址计算，再展开 `DebugTunnel::Open` 内部与 Print 相关的工作：

```text
DebugTunnel::Open(blockNum)
  -> 计算每个 block 的日志区大小
  -> 在设备 GM 中分配 LogWholeRegion
  -> 将每个 block 的 pLogSize 初始化为 0
  -> 填充 LogBufferSize、BlockNum 等描述信息
  -> 在设备 GM 中分配 DebugTunnelData
  -> 将 LogWholeRegion 地址随 DebugTunnelData 拷到设备
  -> 返回设备侧 DTData 指针
```

Print 直接关心的字段可以简化为：

```text
DebugTunnelData
  PrintData.LogWholeRegion  : GM 日志总区地址
  PrintData.BlockNum        : 启动的 block 数量
  PrintData.LogBufferSize   : 每个 block 的 payload 容量，单位为字节
  PrintData.kernelWriteType : AiC / AiV / Mix
```

因此，DTData 不是日志 payload 本身，而是一份由 Host 建立的描述信息。它保存日志 GM 的地址和布局参数，再通过隐藏参数进入 kernel。

### 4.2 每个 block 写自己的日志子区

`LogWholeRegion` 按 block 切分。每个 block 的子区前 64 字节是 header，首 8 字节保存 `pLogSize`，payload 从偏移 64 开始：

```text
stride           = LogBufferSize + 64
LogBuffer(block) = LogWholeRegion + get_block_idx() * stride
pLogSize         = *(uint64_t *)(LogBuffer + 0)
writePtr         = LogBuffer + 64 + pLogSize
```

这个布局解决了两个问题：不同 block 不会覆盖彼此的日志；同一 block 的多次 Print 可以通过 `pLogSize` 依次追加。

### 4.3 `Close` 如何完成 Host 格式化

与 launch 前的 `Open` 对应，设备写完协议记录并执行 DCCI 后，Host 调用 `Close` 同步 stream、回拷日志区，再解析各个节点：

```text
Host 分配日志 GM
  -> DTData 随 kernel 参数传入设备
  -> kernel 写入格式节点和原始值
  -> DCCI + stream 同步
  -> Host 回拷 LogWholeRegion
  -> Host 解析节点并执行最终格式化
  -> 用户看到调试文本
```

到这一步，EmitC Print 的完整数据流才真正闭合。`cce::printf` 只是设备侧入口；能够看到文本，是 ccelib 与 Host DebugTunnel 共同完成的结果。

## 5. 回到设备端：Write 如何追加日志

现在已经知道 DTData 从哪里来，也知道 `LogWholeRegion` 的布局，再回看 ccelib 的 `Write(data, size)`，它的每一步都有了明确含义：

1. 从 fix stack 取得 DTData。EmitC launch prologue 已经把隐藏参数 `.DTData` 写入这里，因此 ccelib 的深层模板函数不必层层传递该参数。
2. 检查 DTData 是否为空，再从中加载 `LogWholeRegion` 并检查日志区是否有效。
3. 使用 block id 和 `LogBufferSize + 64` 计算当前 block 的 `LogBuffer`。
4. 从 `LogBuffer` 头部读取 `pLogSize`，得到 payload 的追加位置。
5. 在剩余容量内将 marker、原始值、长度和格式内容逐字节写入 GM。
6. 更新 `pLogSize`；写完后执行 DCCI，使日志对 Host 可见。

核心地址关系正是上一章介绍的布局：

```text
DTData
  -> LogWholeRegion
  -> LogBuffer(block)
  -> payload + pLogSize
  -> writePtr
```

至此，设备端的两个职责可以清楚地区分：**PrintState 决定写什么，Write 决定写到哪里。** DebugTunnel 则在它们之外负责准备和回收这条 Host-device 日志通路。

## 6. 从调研结果反推 VPTO 方案

理解 EmitC 后，可以把 VPTO 要解决的问题具体化。它不需要重新设计一套 Print runtime，而需要补齐以下能力：

1. 继续复用 Host DebugTunnel 的分配、回拷和解析流程。
2. 给 VPTO entry function 增加 Host 能传入的 DTData 隐藏参数。
3. 在编译期解析 `pto.print` 的格式串。
4. 生成与 Host decoder 兼容的 DebugTunnel 节点。
5. 生成 block 日志地址计算、容量检查、GM store、`pLogSize` 更新和 DCCI。

这里还有一个有利条件：PTO 的格式串是编译期属性。EmitC 必须在设备运行时通过 `PrintState` 扫描格式串，而 PTOAS 可以在 lowering 时完成同样的分析。设备端最终只需执行固定的地址计算和 store。

| 行为 | EmitC + ccelib | VPTO + PTOAS |
|---|---|---|
| 格式串分析 | kernel 运行时由 `PrintState` 完成 | PTOAS 编译期完成 |
| 节点构造 | ccelib 模板运行时执行 | lowering 生成固定操作 |
| 日志地址计算 | `Write()` 运行时执行 | lowering 生成等价地址计算 |
| Host 分配与解析 | DebugTunnel runtime | 复用同一 runtime |

目标不是让两条路径生成相同的 LLVM IR，也不是要求节点以完全相同的方式分段，而是保持 Host decoder 能识别的节点编码，并得到相同的可观察文本。

## 7. VPTO 侧的实现过程

### 7.1 先补上 DTData 隐藏 ABI

PTOAS 检测到模块中存在 `pto.print` 或 `tprint` 后，在类型转换前给每个 entry function 的参数列表末尾追加一个 addrspace(1) 指针：

```text
原始 entry:
  kernel(arg0, arg1)

加入 Print ABI 后:
  kernel(arg0, arg1, DTData: ptr addrspace(1))
```

Host launch stub 通过 `DebugTunnel::Open` 得到设备侧 DTData 地址，并把它作为最后一个 kernel 参数传入。这个参数不出现在用户编写的 PTO IR 中，而是 PTOAS 与 launch ABI 共同维护的隐藏参数。

类型转换后，`injectPrintPrologue` 在 entry 入口初始化 ccelib 兼容状态：

```text
if DTData == null:
  fix_stack = 0
else:
  fix_stack = ptr_to_int(DTData)
  DTData.PrintData.kernelWriteType = AiV
```

这样既保留了 fix-stack 约定，Print lowering 也可以直接从 entry 的最后一个参数读取 DTData。

### 7.2 先降到与 ABI 无关的 Debug 语义

第一层 lowering 只描述“需要预留多少空间、写什么内容、何时提交”，生成内部 Debug 语义：

```text
pto.print / pto.tprint
  -> reserve(totalBytes)
  -> write_text / write_scalar / tile traversal
  -> commit
```

这一层完成格式串分析、标量类型分类，但不关心 DTData 字段偏移、block 日志地址或 DCCI。这样，Print 的用户语义与 DebugTunnel ABI 被明确分开。

格式串仍在编译期拆成 prefix、conversion 和 suffix。标量 Print 对应可选的文本记录、一个 FLOAT/INT 记录、可选的尾部文本记录以及一个 END marker。

### 7.3 再把 Debug 语义降到 DebugTunnel ABI

第二层 lowering 负责把内部 Debug 操作转换成真正的运行时 ABI：

```text
reserve(totalBytes)
  -> 检查 DTData 和 LogWholeRegion
  -> 计算当前 block 的 LogBuffer
  -> 检查容量并返回 writePtr

write_text / write_scalar
  -> 按 DebugTunnel 字节布局写 marker、payload 和格式内容
  -> 按记录大小推进 writePtr

commit
  -> 写 END marker
  -> 更新 pLogSize
  -> DCCI flush
```


### 7.4 统一记录大小计算

实现集中定义三类记录大小：

```text
textRecordSize(payloadBytes)
  = marker(1) + length(2) + payloadBytes

scalarRecordSize(dataBytes, specBytes)
  = marker(1) + value(dataBytes) + length(2) + specBytes

endRecordSize()
  = marker(1)
```

语义层的 `reserve(totalBytes)` 与 ABI 层实际推进 write pointer 都使用这些公式。这样，预留容量、实际写入字节数和 `pLogSize` 始终保持一致；新增记录类型时也只需要维护一套布局定义。

容量不足时仍记录所需总长度而不写 payload，使 Host 能识别截断。

### 7.5 Prologue 与 Epilogue 缺一不可

prologue 在 entry 入口初始化 fix stack 和 `kernelWriteType`。除此之外，当前实现还在每个 kernel return 前注入退出 hook：当 `LogWholeRegion` 有效时执行最终 DCCI flush。

```text
kernel entry
  -> print prologue
  -> kernel body / one or more prints
  -> print epilogue: final DCCI
  -> return
```

单次 `commit` 的 DCCI 负责提交当前记录；kernel-exit DCCI 对应 ccelib 的 `OnKernelFinish` 行为，确保 kernel 任意阶段产生的日志在 Host `Close` 前都对 Host 可见。

### 7.6 当前 lowering 顺序

```text
PTO IR
  -> lowerPrintToDebugRuntime
       print/tprint -> 内部 Debug 语义，并创建字符串 global
  -> addDTDataParamToEntryFunctions
       追加隐藏 DebugTunnel 上下文参数
  -> applyPartialConversion
       普通 VPTO op 与 Debug ABI 一起转换
  -> injectPrintPrologue
  -> injectPrintEpilogue
```



## 8. 从标量 Print 扩展到 TPrint

标量 Print 打通后，`pto.tprint` 不需要另一套 Host 协议。它仍然写 NORMAL、FLOAT/INT 和 END 节点，只是值不再来自一个 SSA 标量，而是来自 Tile 中的所有元素。

当前 VPTO 实现只处理二维 `tile_buf`。lowering 生成固定的 header 和 shape 文本，然后遍历 `rows x cols`：

```text
生成 header 和 shape 节点
  -> 计算 Tile 的 UB 访问基址
  -> scf.for row
       -> scf.for col
            -> 从 addrspace(6) UB load 一个元素
            -> 编码为 FLOAT/INT 节点
            -> 写入 addrspace(1) GM 日志区
  -> 更新 pLogSize
  -> commit DCCI；kernel return 前再执行最终 DCCI
```

这有两个地址空间：UB 是待打印 Tile 的数据来源；`LogWholeRegion` 的 GM 是 DebugTunnel 日志的写入目的地。TPrint 的本质，是在标量 Print 已经建立的日志机制前增加 Tile 遍历和 UB load。



## 9.  A5 模拟器验证

验证方式：在已有的VPTO测试案例中插入pto.print . 观察是否会影响正常结果以及能否正常打印 

| 覆盖方向 | Cases | 验证重点 |
|---|---|---|
| 整数与窄整数 | `vabs`、`vcmps-i8-signed` | i32 编码、i8 符号扩展 |
| 浮点格式 | `vcmps-f32`、`vaxpy-f32`、`vexpdiff-f32` | f32 编码、`%.2f` 格式、同一 kernel 多次追加 |
| 类型转换 | `vcvt-f32-to-f16`、`vcvt-f16-to-f32` | Print 与 deep-merged 转换 pipeline 组合 |
| 数据相关标量 | `load-store-scalar-ub` | 打印 kernel 计算得到的 i16 值，而不只是常量 |
| SIMT 与同步 | `simt-store-tid` | SIMT launch 前后打印、同一日志连续追加 |
| Gather 与 predicate | `vgather2`、`pldi-norm` | Print 与 gather、predicate load/store 共存 |
| Cube 路径 | `cube-bridge-matmul` | AIC/Cube kernel 的隐藏参数和 DebugTunnel 通路 |

这些 case 覆盖常量和数据相关值、整数和浮点格式、单次和多次打印，以及 Vector、SIMT 和 Cube 等不同执行路径；所有 case 的打印结果与原有计算验证均通过。

