# PTO Print Lowering：C Wrapper 委托 cce::printf

## 1. 这份文档解决什么问题

`pto.print` 和 `pto.tprint` 最终都要让 Host 看到 AI Core 上产生的调试文本。EmitC 路径里，`pto.print` 直接编译成设备端 `cce::printf(format, args...)` 模板调用；VPTO 后端却做不到这一点——`cce::printf` 是 CCE 前端（bisheng/clang）拥有的 C++ 模板，PTOAS 的 LLVM 发射器无法实例化它。

本方案用一个 **C wrapper TU** 解决：

- `tools/ptoas/cce/pt_print.cpp` 提供一组 `extern "C" [aicore]` 的薄封装函数（`pto_print_f32/i64/u64/str`），内部直接调用 `cce::printf`。
- VPTO lowering 把 `pto.print` 降级为对这些封装函数的调用。
- 构建时用 bisheng 把 wrapper 编译成 bitcode，再用 `llvm-link` 与 kernel bitcode **在 bitcode 层面合并**，最后统一编成设备 `.o`。

这样 VPTO 路径的格式解析、协议编码、日志写入全部复用 ccelib 中经过验证的实现，PTOAS 不再自己逐字节构造 DebugTunnel 协议。

三个职责：

- **Host DebugTunnel**：分配 GM 日志区，把地址交给 kernel；kernel 结束后回拷并解析日志。这条路径两个后端完全共用，没有改动。
- **设备端 cce::printf / ccelib**：wrapper 委托的实现。负责格式串状态机、协议记录写入和 DCCI flush。
- **PTOAS VPTO lowering**：把 `pto.print` / `pto.tprint` 翻译成 wrapper 调用，并负责 entry function 的 `pto_print_init/finish` 生命周期。

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

### 2.2 Kernel 初始化 DTData 访问入口

EmitC 路径的 launch 框架在 kernel 前调用 `__DebugTunnel_Initialize(DTData)`，ccelib 将 DTData 的 GM 地址写入 fix stack；后续 `cce::printf` 通过 `DebugTunnel::GetKernelInstance()` 从 fix stack 取回它。

VPTO 路径把 DTData 作为 entry function 的隐藏 GM 指针参数。由于 CCE 的 `-cce-aicore-enable-print-init-finish` 注入 pass 在 `-x ir` 输入上会崩溃，VPTO lowering 在 entry 中直接生成两个 wrapper 调用完成等价的初始化/收尾：

```text
pto_print_init(DTData)   // = DebugTunnel::OnKernelInitialize：
                         //   将 DTData 写入 fix stack，设置 kernelWriteType
...
pto_print_finish(DTData) // = DebugTunnel::OnKernelFinish：
                         //   触发 DCCI flush，保证 Host 可见日志
```

`pto_print_init/finish` 与 `cce::printf` 同属 ccelib；`cce::printf` 内部通过 `GetKernelInstance()` 读回 init 写入的 DTData，因此 wrapper 函数之间不需要再逐层传递 DTData 参数。

### 2.3 cce::printf 写日志

`cce::printf` 是 ccelib 提供的 C++ 模板。它创建 `PrintState` 状态机，运行时扫描格式串，把“格式片段 + 原始二进制值”按 DebugTunnel 节点格式写入当前 block 的 GM 日志子区。详见第 3 节。

### 2.4 Flush、回拷与 Host 格式化

`cce::printf` 写完记录后执行 DCCI，`pto_print_finish`（`OnKernelFinish`）再保证最终可见。kernel 完成后，DebugTunnel 的关闭流程同步 stream，然后：

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

完整数据流：

```text
Host 分配 GM
  -> LogWholeRegion 写入 DTData
  -> DTData 作为隐藏参数传给 kernel
  -> pto_print_init 写入 fix stack
  -> kernel 调用 pto_print_* wrapper -> cce::printf 写协议记录
  -> pto_print_finish + DCCI + stream 同步
  -> Host 回拷 GM
  -> Host 解析节点并输出文本
```

## 3. cce::printf 的关键事实

理解 wrapper 的设计，需要知道 cce::printf 的四个事实：

### 3.1 fmt 是运行时指针

`cce::printf` 的格式串参数是运行时指针，它只是把字符串字节写进日志区，从不按地址解析格式串。因此 kernel TU 里定义的 module-level global 格式串（`LLVM::AddressOfOp` 得到的指针）可以直接传给 wrapper，不需要特殊的 section 注册。

### 3.2 节点宽度：FLOAT 4 字节、INT 8 字节

DebugTunnel 记录中：

- **FLOAT 节点**携带 4 字节值 → wrapper 只暴露 `pto_print_f32(fmt, float)`。f16/bf16/f64 由 lowering 先转成 f32 再调用。
- **INT 节点**携带 8 字节值，ccelib 通过 `ConvertTo<T, long long>` 把参数规范化为 64 位 → wrapper 暴露 `pto_print_i64(fmt, i64)` 和 `pto_print_u64(fmt, uint64_t)`。

### 3.3 Support<> 类型表没有 64 位无符号类型

`cce::internal::PrintState::Support<T>` 支持的类型列表（CCE 15.0.5 头文件）：

```text
char*, const char*（含 __gm__ 变体）
signed char / short / int / long / long long / int8_t..int64_t
unsigned char / unsigned short / unsigned int / uint8_t / uint16_t / uint32_t
half / float / char / 任意指针
```

**没有 `unsigned long` / `uint64_t` / `unsigned long long`**——直接传 `uint64_t` 会触发 `static_assert(Support<T>::value, "Unsupported datatype!")`。因此 `pto_print_u64` 把值强制转换为 `long long` 再传给 `cce::printf`：INT 节点携带的仍是同样的 8 个字节，Host 端按 `%u/%x/%o` 重新解释这些位。

### 3.4 PrintState 是格式串状态机

`PrintState` 保存格式串地址 `fmt` 和当前扫描位置 `curpos`。每消费一个参数，`operator<<` 从 `curpos` 开始寻找下一个 `%`，然后依次解析：

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

PrintState 不在设备端生成最终字符串；它保存格式片段和原始二进制值，Host 按格式片段完成最终格式化。

## 4. C Wrapper：tools/ptoas/cce/pt_print.cpp

wrapper 的全部对外符号都是 `extern "C" [aicore]`，ABI 与 VPTO lowering 生成的声明一一对应：

| wrapper | 签名 | 用途 |
|---|---|---|
| `pto_print_str` | `(const char *fmt)` | 纯文本节点：tprint 的 header / shape 记录 |
| `pto_print_f32` | `(const char *fmt, float v)` | FLOAT 值节点（4 字节） |
| `pto_print_i64` | `(const char *fmt, int64_t v)` | 有符号 INT 值节点（8 字节） |
| `pto_print_u64` | `(const char *fmt, uint64_t v)` | 无符号 INT 值节点；内部转 `long long` 后委托（见 3.3） |
| `pto_print_init` | `(__gm__ void *dt)` | `DebugTunnel::OnKernelInitialize` |
| `pto_print_finish` | `(__gm__ void *dt)` | `DebugTunnel::OnKernelFinish` |

要点：

- wrapper 只包含两个 include：`<ccelib/__ccelib.h>` 和 `<stdint.h>`，不包含 C++ 标准库头，保证编译产物干净。
- `cce::printf` 的实例化只发生在 wrapper TU 中，PTOAS 侧永远只是 `declare` + `call`。
- 该文件不属于 ptoas 的 Host 构建（`tools/ptoas/cce/` 目录刻意没有 CMakeLists.txt），而是通过 `PTOAS_DEFAULT_PRINT_WRAPPER_PATH` compile definition 暴露给 ObjectEmission 管线。

## 5. VPTO Lowering：pto.print / pto.tprint -> wrapper 调用

### 5.1 lowerVPTOOps 中 Print 相关的三个阶段

```text
PTO IR
  -> collectAndCreatePrintfStringGlobals
       扫描 Print/TPrint
       为格式字符串创建 LLVM global（@_ptoas_printf_fmt_N）
  -> addDTDataParamToEntryFunctions
       给每个 entry function 追加 DTData GM 隐藏参数
       声明 pto_print_str/f32/i64/u64/init/finish 六个 wrapper
  -> applyPartialConversion
       LowerPrintOpPattern: Print -> AddressOfOp(fmt) + wrapper call
       LowerTPrintOpPattern: TPrint -> str 记录 + 双层循环 + 元素 wrapper call
  -> injectPrintPrologue
       在 entry 入口调用 pto_print_init(DTData)
       在每个 return 前调用 pto_print_finish(DTData)
```

预扫描必须早于 pattern lowering，因为 `LLVM::AddressOfOp` 只能引用已经存在的 module-level global。DTData 参数也必须在类型转换前加入函数签名。

### 5.2 LowerPrintOpPattern：按类型选 wrapper

格式串 global 通过 `LLVM::AddressOfOp` 取得，然后按标量类型和转换类别选择 wrapper：

```text
f16 / bf16  -> fpext 到 f32 -> pto_print_f32
f64         -> fptrunc 到 f32 -> pto_print_f32
f32         -> 原样          -> pto_print_f32
i8/i16/i32  -> sext/zext 到 i64 -> pto_print_i64 / pto_print_u64
i64         -> 原样          -> pto_print_i64 / pto_print_u64
```

（`%u/%x/%o` 等无符号转换走 `pto_print_u64`，其余整数走 `pto_print_i64`。）

> **i64 必须配 64 位长度修饰符**：INT 节点携带 8 字节，但 Host 端解码器按格式片段消费字节——`%d` 只读低 32 位，`%lld`/`%lu` 才读满 8 字节。打印 i64 时格式串要用 `%lld`（实测 `%d` 会把 `1234567890123` 截断成 `1912276171`）。

生成的调用形态：

```llvm
call void @pto_print_f32(ptr @_ptoas_printf_fmt_0, float %0)
call void @pto_print_i64(ptr @_ptoas_printf_fmt_4, i64 %0)
```

### 5.3 LowerTPrintOpPattern：文本记录 + 元素循环

`pto.tprint` 不需要新的传输协议，仍然是“文本记录 + FLOAT/INT 值节点”：

```text
pto_print_str(fmt_0)  // header: "=== [TPRINT Tile] Data Type: ..., TileType: Vec ===\n"
pto_print_str(fmt_1)  // shape:  "  Shape: [8, 8], Valid Shape: [8, 8]\n"
scf.for row:
  scf.for col:
    load tile[row][col]  // UB addrspace(6)，基址由 GET.SYS.VA.BASE + UB 偏移算出
    f16 -> fpext f32 -> pto_print_f32(fmt_2, v)   // "%6.2f"
    i32 -> sext i64  -> pto_print_i64(fmt_6, v)   // "%6d"
```

header / shape 与元素转换用的格式串也走 `collectAndCreatePrintfStringGlobals` 创建的 global。

### 5.4 生成 IR 示例

`print_scalar_vpto_llvm.pto` 的 f32 kernel 生成的 IR 形态：

```llvm
@_ptoas_printf_fmt_0 = private constant [18 x i8] c"scalar = %+08.3f\0A\00"

declare void @pto_print_finish(ptr addrspace(1)) #0
declare void @pto_print_init(ptr addrspace(1)) #0
...

define void @print_scalar_kernel_mix_aiv(float %0, ptr addrspace(1) %1) #1 {
  call void @pto_print_init(ptr addrspace(1) %1)
  call void @pto_print_f32(ptr @_ptoas_printf_fmt_0, float %0)
  call void @pto_print_finish(ptr addrspace(1) %1)
  ret void
}
```

## 6. 构建管线：bitcode 级 llvm-link 合并

wrapper 的编译和合并发生在两个地方，逻辑相同：

### 6.1 ObjectEmission.cpp（生产管线，ptoas fatobj）

`VPTOFatobjArtifacts::emitCubeObject / emitVectorObject` 在 `usesPrint` 时：

```text
写 kernel.ll（applyVPTOLLVMABINames + writeLLVMModule）
  -> mergePrintWrapper:
       compilePrintWrapperToBitcode:   bisheng 编译 pt_print.cpp -> pt_print.bc
       linkLLVMBitcode:                bisheng 的 llvm-link 合并 kernel.ll + pt_print.bc
  -> compileDeviceLLVMToObject:        把合并后的 .bc 编成设备 .o
```

wrapper 必须按每个 target CPU（`dav-c310-vec` / `dav-c310-cube`）分别编译，保证 wrapper 的 `target-cpu` 与产物一致。

### 6.2 编译命令（driver 模式）

```text
bisheng -xcce --cce-aicore-only --cce-aicore-arch=<cpu>
        -D__CCE_ENABLE_PRINT_FOUND_CANN__ --cce-enable-print
        -std=c++17 -c -emit-llvm -x cce pt_print.cpp -o pt_print.bc
```

用 **driver 模式 `-emit-llvm`** 而不是 CC1：driver 自动管理 CCE/CANN include 链，且不会把符号拆成 `.vector`/`.cube` variants（不要传 `-cce-enable-mix`）。

### 6.3 必须用 bisheng 的 llvm-link

合并必须用 bisheng（LLVM 15）自带的 `llvm-link`（`${ASCEND_HOME_PATH}/bin/llvm-link` 或 `tools/bisheng_compiler/bin/llvm-link`）。PTOAS 仓库自带的 LLVM 21 llvm-link 生成的 bitcode 会被 bisheng 拒绝：

```text
Not an int attribute (Producer: 'LLVM21.1.8' Reader: 'LLVM 15.0.5')
```

`llvm-link` 同时接受 textual `.ll` 和 binary `.bc` 输入；合并后的 `.bc` 再经 `-x ir` 编译为设备 `.o` 与直接编 `.ll` 完全等价。

### 6.4 E2E 验证脚本

`test/vpto/scripts/run_vpto_print_validation.sh` / `run_vpto_tprint_validation.sh` / `run_vpto_print_types_validation.sh` 在“ptoas 生成 kernel.ll”之后、设备编译之前插入同样的两步：wrapper bitcode 编译 + `llvm-link` 合并。脚本先 `sed` 去掉 constant GEP 的 `nuw`（bisheng 的 LLVM 15 不支持），再编译合并后的 bitcode。

## 7. 阅读实现时应抓住的主线

阅读生成 IR 或调试 Print 时，可以按以下顺序定位：

1. Host 是否在 launch 前成功分配 `LogWholeRegion`，并把地址写进 DTData。
2. entry function 是否收到 DTData 隐藏参数，`pto_print_init` 是否在入口被调用。
3. `pto.print` 是否按类型降级为正确的 wrapper 调用（f16/bf16 先 fpext、f64 先 fptrunc、窄整数先 sext/zext）。
4. 每个 entry 的 return 前是否有 `pto_print_finish`（flush 依赖它）。
5. tprint 的 header / shape 是否走 `pto_print_str`，元素循环是否按 dtype 走 f32/i64 wrapper。
6. 构建时 wrapper bitcode 是否被 `llvm-link`（bisheng 版）合并进设备模块。
7. Host 是否在 stream 完成后回拷日志 GM 并运行协议解析。

整套实现的本质是：**保留 ccelib/DebugTunnel 的 Host 生命周期和协议，设备端的格式解析与日志写入全部委托给 cce::printf，PTOAS 只负责把 PTO 操作降级为对 C wrapper 的调用，并在 bitcode 层面把 wrapper 与 kernel 合并。**
