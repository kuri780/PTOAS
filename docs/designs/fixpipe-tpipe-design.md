# `tpipe` 支持 `fixpipe` 接口设计

## 1. 背景

最新 `pto-isa` 已提供 `TPUSH<Pipe, TileProd, TConfig>(pipe, tile)` 这一类
fixpipe 接口。其中 `TConfig` 是 `TPUSH` 的模板参数，通常使用
`FixpipeParams<...>` 表达，配置项至少包括：

- `LayoutMode_t`
- `QuantMode_t`
- `ReluPreMode`

并且当前 `FixpipeParams` 还继续承载：

- `STPhase`
- `SubBlockId`
- `AtomicType`
- `ClipReluMode_t`
- `IsChannelSplit`

典型写法如下：

```cpp
using MyConfig = FixpipeParams<
    LayoutMode_t::NZ2ND,
    QuantMode_t::DEQF16,
    ReluPreMode::NormalRelu>;
Pipe pipe(fifoMem, 0x0, 0x0);
AccTile accTile;
TASSIGN(accTile, 0x0);

TPUSH<Pipe, AccTile, MyConfig>(pipe, accTile);
```

这说明 `pto-isa` 的 C++ 用户接口已经允许“通过 `TPUSH` 的模板参数指定
fixpipe 行为”。

但如果 PTOAS 直接把这套形状一比一投影到前端 IR，把 fixpipe config 绑定到每一条
`tpush` 上，会引出两个抽象问题：

- `TPOP` 端没有对称的 `TConfig` 接口，无法从消费侧表达“这一条 pipe entry 是按什么
  fixpipe 语义生产出来的”
- 同一条逻辑 pipe 上如果允许多次 `tpush` 带不同 config，consumer 很难在 IR 层证明
  该 pipe 的 entry 类型、layout、quant 语义和同步契约始终一致

因此，这个设计文档讨论的不是“是否支持 fixpipe over tpipe”，而是“PTOAS 应该把
fixpipe config 挂在哪一层”。

## 2. 目标

本文目标如下：

- 在 PTOAS 中为 `tpipe` 增加对 fixpipe 型 `TPUSH` 的语义承载能力
- 在 EmitC 层保持与 `pto-isa` 现有 `TPUSH<..., TConfig>` 调用形状兼容
- 明确 PTOIR 当前只暴露“前端公开配置、且影响 pipe 公共语义”的 fixpipe 参数
- 避免把 fixpipe config 建模成“某一次 `tpush` 的私有属性”
- 让 producer / consumer 两侧都能在各自本地 `initialize_pipe` 上看到同一份
  pipe 级契约，而不要求 data op verifier 直接跨函数回看 peer kernel
- 第一版 `initialize_pipe` 只显式承载一个前端公开复合 attr：
  `acc_push_epilogue`
- `acc_push_epilogue` 内部包含三个独立配置维度：
  - `layout`
  - `quant`
  - `relu`
- 从第一版起明确覆盖 vector quant，而不是只覆盖 no-quant 或 scalar quant

## 3. 现状

### 3.1 `pto-isa` 现状

`pto-isa` 当前公开语义存在以下不对称：

- `TPUSH` 有 `TileSplitAxis` 重载
- `TPUSH` 有 `GlobalData` 重载
- `TPUSH` 有 `TConfig` 重载
- `TPOP` 只有 `TileSplitAxis` / `GlobalData` 重载
- `TPOP` 没有 `TConfig` 重载

因此，`TConfig` 更像 producer-side fixpipe 写入方式，而不是一次完整 pipe
transaction 的双端公共契约。

### 3.2 PTOAS 当前 pipe 设计现状

PTOAS 已经有前端 pipe 抽象：

- `pto.aic_initialize_pipe`
- `pto.aiv_initialize_pipe`
- `pto.tpush_to_aiv`
- `pto.tpush_to_aic`
- `pto.tpop_from_aic`
- `pto.tpop_from_aiv`

当前设计中：

- `initialize_pipe` 负责声明 pipe 级配置
- `tpush/tpop/tfree` 只带 `id + split`
- `nosplit` 已经是 pipe 级属性，而不是逐条 transfer op 属性

这和 `pto-isa` 的 `TPipe<..., IsNoSplit>` 语义是同一件事的两层表达：

- PTOAS 前端用 `nosplit = true/false` 暴露给用户
- lowering 时再映射为 `TPipe<..., IsNoSplit = true/false>`

这说明 PTOAS 当前前端 IR 的总体方向，本来就更偏向“per pipe contract”，而不是
“per transfer config”。

### 3.3 当前前端 init 查找是 function-local

当前 PTOAS 前端 verifier / lowering 还有两个直接影响本设计的既有约束：

- `tpush_to_aiv` / `tpop_from_aic` / `tfree_from_*` 等 data op，都是按 `id`
  在 **同一个 function** 内查找匹配的 frontend `initialize_pipe`
- 同一个 function 内不允许混用 `pto.aic_initialize_pipe` 和
  `pto.aiv_initialize_pipe`

因此，如果 fixpipe contract 只挂在 Cube 侧 `pto.aic_initialize_pipe` 上，那么：

- Vector 侧 `tpop_from_aic` 的本地 verifier 看不到这组 contract
- Vector function 自身的 lowering 也看不到这组 contract

这意味着第一版设计不能把“consumer 侧类型 / layout / quant 语义检查”建立在
“consumer 去跨函数读取 producer init”上，而应让同一条逻辑 pipe 的两端
frontend init 都显式携带同一份 contract。

## 4. 核心判断

### 4.1 挂载层级

把 fixpipe config 建模为 **per pipe**，而不是 **per tpush op**。

也就是说：

- PTOAS 前端 IR / PTOIR 中，fixpipe config 绑定到 `initialize_pipe`
  所定义的逻辑 pipe
- 对一条 C2V fixpipe logical pipe，这份 contract 需要同时体现在：
  - producer Cube function 内的 `pto.aic_initialize_pipe`
  - peer Vector function 内的 `pto.aiv_initialize_pipe`
- 两端 `initialize_pipe` 上的 `acc_push_epilogue` 必须逐字段一致
- lowering 到 EmitC 时，再把该 pipe 的 fixpipe config 转写成具体的
  `TPUSH<Pipe, TileProd, TConfig>(pipe, tile)` 模板调用

这里的 **per pipe** 只表示“同一条 pipe 内语义稳定”，并不要求“同一个 kernel
里所有 fixpipe pipe 共用同一组 epilogue 配置”。

同一 kernel 中可以同时存在多条开启 `acc_push_epilogue` 的 pipe，并且它们的：

- `layout` 可以不同
- `quant` 可以不同
- `relu` 可以不同

稳定性约束只在“单条 pipe 内”成立。

### 4.2 不用 per-`tpush`

使用per-tpush会产生问题：

- 同一条 pipe 的不同 producer push 可以携带不同 config，破坏 pipe entry 语义稳定性
- `tpop` 无法在 op 自身上表达对应 config，只能靠隐式推导
- verifier 很难判断 consumer tile type、layout、quant 结果类型是否与 producer 一致
- lowering 到内部 pipe handle 后，pipe 本身不再是纯同步/地址资源，还会隐含一组逐次变化的 entry 语义

## 5. 承载范围

IR 层是否暴露某个 fixpipe 参数，不取决于 `pto-isa` 后端有没有消费它，而取决于它是否
同时满足下面三个条件：

- 用户在 PTOAS 前端确实可以配置它
- 它会进入这条 pipe 的公共语义契约，而不是只影响某个 target 的实现路径
- verifier 可以围绕它建立稳定的前端约束

按这个标准，第一版 PTOAS 前端 IR / PTOIR 只显式建模一个统一的
pipe-level public contract：

- `acc_push_epilogue`

其内部再承载三个彼此独立的字段：

- `layout`
- `quant`
- `relu`

这里的“只建模三个字段”并不意味着 `FixpipeParams` 的其余模板参数只是占位。
对照当前 `pto-isa` 实现，至少有一部分已经是生效语义：

- A2/A3 的 fixpipe `TPUSH` 已消费 `AtomicType`、`STPhase`、`LayoutMode`、
  `QuantMode`、`ReluPreMode`
- A5 的 fixpipe `TPUSH` 已消费 `SubBlockId`、`STPhase`、`LayoutMode`、
  `QuantMode`、`ReluPreMode`；其 GM 路径还会消费 `AtomicType`

但这些字段当前阶段暂不作为 PTOIR 前端公开配置项，原因也很直接：

- `STPhase`、`AtomicType`、`SubBlockId` 虽然会被后端实现消费，但当前还不适合作为
  `initialize_pipe` 的前端公开配置
- `ClipReluMode_t`、`IsChannelSplit` 目前也没有形成明确的前端公开配置需求
- 在没有完整前端语义、verifier 和跨平台约束之前，它们不应被写进 IR surface，
  更不应成为 `initialize_pipe` 的公共 contract

这三个字段虽然由一个复合 attr 承载，但语义上仍必须独立建模，不能折叠成单个互斥
enum。

原因如下：

- `LayoutMode_t::NZ2ND` / `NZ2DN` / `NZ2NZ` 描述 layout
- `QuantMode_t::NoQuant` / `DEQF16` / `VDEQF16` 等描述数值转换
- `ReluPreMode::NoRelu` / `NormalRelu` 描述激活预处理

这三类语义可以自由组合，不是单选关系。

但“字段彼此独立”不等于“直接开放底层完整 enum 的所有取值”。第一版 front-end
公开值域需要显式收窄。

第一版建议的公开值域如下：

- `layout`
  - `nz2nd`
  - `nz2dn`
  - `nz2nz`
- `quant`
  - `no_convert`
  - `f32_f16`
  - `req8_scalar`
  - `req8_vec`
  - `deqf16_scalar`
  - `deqf16_vec`
  - `f32_bf16`
  - `qf322b8_pre_scalar`
  - `qf322b8_pre_vec`
  - `qf322f16_pre_scalar`
  - `qf322bf16_pre_scalar`
  - `qs322bf16_pre_scalar`（A5-only）
  - `qs322bf16_pre_vec`（A5-only）
  - `qf322hif8_pre_scalar`
  - `qf322fp8_pre_scalar`
- `relu`
  - `no_relu`
  - `normal_relu`

也就是说：

- `PTO_ReluPreModeEnum` 虽然还定义了 `scalar_relu` / `vector_relu` / `pwl`
  等值，但它们暂不属于 v1 `acc_push_epilogue.relu` 的前端公开配置
- `PTO_AccStoreQuantPreModeEnum` 虽然还定义了 `*_hybrid_*`、`qf322f32_pre_*`、
  `*_s4_*`、`*_s16_*`，以及未纳入本文的其它 `*_vec` / `*_scalar` 变体，但它们暂不
  属于 v1 `acc_push_epilogue.quant` 的前端公开配置
- 在 `*_hif8_*` / `*_fp8_*` family 中，v1 当前只公开
  `qf322hif8_pre_scalar` / `qf322fp8_pre_scalar`；其它同 family 变体仍保持关闭
- 在 `qs322bf16_*` family 中，当前 `pto-isa` 源码证据只在 A5 路径坐实；
  因此 v1 虽然公开 `qs322bf16_pre_scalar` / `qs322bf16_pre_vec`，但必须按 A5-only
  做 target/profile gating，而不应被表述成无条件跨平台能力

这并不是说这些模式在 `pto-isa` 后端没有实现价值，而是：

- 本文只把当前已经补齐完整、稳定、可验证前端语义的最小子集开放到
  `acc_push_epilogue.quant`
- 特别是 8-bit family 需要区分 scalar 与 vector 两类：
  - 对 `req8_scalar` / `qf322b8_pre_scalar`，不能只靠 `quant` enum 自身恢复
    destination signedness；更稳妥的做法是由 peer consumer `tpop` 的结果 dtype
    提供这部分信息，并要求同一条 logical pipe 上的所有 `tpop` 结果类型保持一致。
    对这类 scalar 8-bit family，consumer result element type 必须显式写成 `si8`
    或 `ui8`，不得使用 signless `i8`
  - 对 `req8_vec` / `qf322b8_pre_vec`，当前 `pto-isa` fixpipe vector quant 路径
    没有独立 unsigned consumer contract / signedness 通道，v1 前端应收紧为只允许
    显式 `si8`，不得使用 `ui8` 或 signless `i8`
- 公共手册 `docs/PTO_IR_manual.md` 与相关 type-system 说明需要与这里的 fixpipe
  8-bit contract 保持一致：明确这里依赖 `si8`，以及在 scalar 8-bit 场景下依赖
  `ui8`，作为 peer consumer `tpop` 结果类型上的公开语义。本 PR 已同步修订
  `docs/PTO_IR_manual.md` 中相关历史表述，避免继续把 `ui8` 解释成仅由 signless
  `i8` 近似代指
- 后续若要继续扩展更多 quant 模式，也应优先沿现有 contract 扩展：scalar 8-bit
  继续由 peer `tpop` dtype 提供 signedness；若要开放 vector 8-bit 的 `ui8` 语义，
  则应先补齐底层 vector quant 的 unsigned contract，而不是在 PTOIR 侧先行承诺

第一版暂不把其余模板参数纳入 PTOIR，原因是：

- `STPhase`、`AtomicType`、`SubBlockId` 更接近具体 producer 执行路径、target
  特化或 lowering 选择；现阶段还不能证明它们都应统一建模为跨平台的 pipe-level
  public contract
- `ClipReluMode_t`、`IsChannelSplit` 目前还缺少明确的 PTOAS 前端需求、类型推导
  规则和 verifier 约束
- 在这些字段正式成为 PTOIR 前端公开配置项之前，PTOAS 不应把它们“半暴露半忽略”；
  后续若确有需求，应单独补齐语义设计，再决定它们属于 `initialize_pipe`
  attrs、producer-side config op，还是 target-specific lowering 选择

应明确：

- `vector quant` 必须支持
- `scalar quant` 必须支持
- `NoQuant` 只是其中一种合法值

其中 v1 的 “vector quant / scalar quant” 只覆盖当前已纳入公开值域的模式：

- scalar quant：`deqf16_scalar`、`req8_scalar`、`qf322b8_pre_scalar`、
  `qf322f16_pre_scalar`、`qf322bf16_pre_scalar`、`qs322bf16_pre_scalar`（A5-only）、
  `qf322hif8_pre_scalar`、`qf322fp8_pre_scalar`
- vector quant：`deqf16_vec`、`req8_vec`、`qf322b8_pre_vec`、`qs322bf16_pre_vec`（A5-only）

其中：

- `qs322bf16_pre_scalar` / `qs322bf16_pre_vec` 当前应按 A5-only 做额外 gating；
  若目标平台没有坐实这两条 fixpipe quant 路径，则 verifier 应直接拒绝。当前
  PTOAS 实现中，这里的 A5 识别至少应覆盖 `--pto-arch=a5`、`pto.target_arch = "a5"`
  与 A5 `pto.device-spec`（如 `Ascend950*` / `Ascend910_95*`）
- `qf322hif8_pre_scalar` / `qf322fp8_pre_scalar` 属于低精度目标元素类型 family，
  第一版应按 target/profile 能力做额外 gating；若目标平台不支持 `!pto.hif8` 或
  `f8E4M3FN` 作为 consumer dtype，则 verifier 应直接拒绝。当前 PTOAS 实现中，
  这类 A5-only gating 的 target 识别也至少应覆盖 `--pto-arch=a5`、
  `pto.target_arch = "a5"` 与 A5 `pto.device-spec`

同时需要区分两层语义：

- `acc_push_epilogue.quant` 只描述这条 pipe entry 采用哪一种量化/反量化模式
- 真正的运行时量化参数仍来自 producer 侧额外量化配置

对 `deqf16_*` / `req8_*` / `qf322b8_pre_*` / `qf322f16_pre_scalar` /
`qf322bf16_pre_scalar` / `qs322bf16_pre_*` / `qf322hif8_pre_scalar` /
`qf322fp8_pre_scalar` 这类模式，仅有
`acc_push_epilogue.quant` 还不够；还需要与之匹配的 producer-side quant
state。该状态在 `pto-isa` 中通过：

- `SET_QUANT_SCALAR<OutType>(scalar)`
- `SET_QUANT_VECTOR(fpTile)`

显式建立。

这类运行时量化参数不应折叠进 pipe attrs。它们可以在同一条 pipe 的不同
`tpush` 之间变化，而不会改变该 pipe 的 entry layout / element type contract。

还要注意：PTOAS 当前已有的 `preQuantScalar : i64` 形式只服务于
`tmov/tstore/textract/tinsert` 等直接数据搬运接口，对应的是 `pto-isa` 里
`TMOV/TSTORE/TEXTRACT/TINSERT(..., uint64_t preQuantScalar)` 这一类接口，不等价于
fixpipe `TPUSH` 所依赖的 `SET_QUANT_SCALAR(float)` / `SET_QUANT_VECTOR(...)`
状态，因此不建议直接复用。

## 6. 语义约束

### 6.1 适用方向

fixpipe-over-tpipe 仅支持：

- producer 为 Cube
- source entry 为 `acc` tile
- 方向为 C2V

前端语义目标需要覆盖 `pto-isa` 当前在 A2/A3 与 A5 上都已存在的
`TPUSH<Pipe, AccTile, FixpipeParams>` 能力，但允许后端按平台选择不同的数据路径。

只覆盖：

- `pto.tpush_to_aiv`

不覆盖：

- `pto.tpush_to_aic`
- `global entry` 路径
- V2C producer push

也就是说，第一版统一的是前端 pipe contract：

- Cube producer 生产 `acc` entry
- Vector consumer 消费 fixpipe 语义下的 C2V entry
- front-end 不把“该平台最终走 UB FIFO 还是 GM FIFO”暴露成额外用户接口差异
- front-end 也不在 `dir_mask = 3` 的 DIR_BOTH pipe 上只对单侧方向局部挂载
  fixpipe config

`pto-isa` 当前 `TConfig` 语义本质上描述的是 Acc producer 的 fixpipe 行为。

同时，由于当前 frontend init 查找是 function-local，第一版还应规定：

- 开启 fixpipe contract 的 C2V logical pipe，在 Cube producer 侧必须有
  `pto.aic_initialize_pipe`
- 其 peer Vector consumer 侧必须有同一条 logical pipe 的
  `pto.aiv_initialize_pipe`
- 两端 `acc_push_epilogue` 必须完全一致

这样 producer / consumer 两侧都可以只依赖本地 init 完成各自的 verifier /
lowering，而不需要在 data op 上直接跨函数读取 peer kernel 的 frontend init。

这里还需要把“同一条 peer logical pipe 如何配对”说清楚。第一版不建议仅靠
`initialize_pipe.id` 做跨函数配对，因为在同一个 module 中完全可能存在多组
cube/vector function，各自都含有 `id = 0` 的 pipe。

更合适的做法是复用当前 frontend pipe 已有的 peer 建模关系：C2V / V2C pipe
本来就是通过 `reserve_buffer` / `import_reserved_buffer` 以及
`import_reserved_buffer.peer_func` 建立跨函数配对。

因此第一版文档建议把 peer logical pipe identity 明确为：

- 由 `initialize_pipe` 绑定的 C2V / V2C local buffer 来源决定
- 对 local `reserve_buffer`，使用“当前 function symbol + reserve_buffer name”
- 对 `import_reserved_buffer`，使用“peer_func symbol + import name”
- 再结合实际方向 `dir_mask` 区分 C2V / V2C

也就是说，peer contract verify 不应把裸 `pipe id` 视为 module 级全局 key；更稳妥的
做法是复用当前 frontend pipe / reserved-buffer 体系已经存在的 peer key。

第一版还应进一步收紧：

- 对开启 `acc_push_epilogue` 的 C2V fixpipe pipe，`c2v_consumer_buf` 必须可追溯到
  `reserve_buffer` 或 `import_reserved_buffer`
- 如果对应 operand 只是某个无法继续追溯 peer identity 的普通 `i32` 值，则
  peer contract check 应直接报错，而不是尝试猜测对端 init

否则即使“理论上想复用 peer key”，实现上也无法稳定完成 producer / consumer 的配对。

### 6.2 与 split 的关系

设计应规定：

- fixpipe pipe 与 split pipe 互斥
- 一条开启 fixpipe config 的 pipe，只允许 `split = 0`
- 也即 fixpipe pipe 必须等价于 `nosplit = true`
- 若用户同时需要 `C2V fixpipe` 与 `V2C normal pipe`，v1 必须拆成两条
  logical pipe，而不是继续复用一条 `dir_mask = 3` 的 bidirectional pipe

对应到 `pto-isa` 后端时，这会落到 `TPipe<..., IsNoSplit = true>`；反过来，
PTOAS 中 `nosplit = false` 则对应 `TPipe<..., IsNoSplit = false>`。

原因不是“硬件绝对做不到所有组合”，而是：

- `pto-isa` 当前 public API 没有 `TPUSH<Pipe, TileProd, Split, TConfig>` 这种统一形状
- `TPOP` 端仍只按 split 消费
- 当前 PTOAS 若同时把 split 和 fixpipe 混进同一条 pipe，前端契约会变得不清晰

### 6.3 一条 pipe 的 config 稳定性

同一逻辑 pipe 上：

- 只能存在一组 fixpipe config
- producer 侧 `aic_initialize_pipe` 与 consumer 侧 peer `aiv_initialize_pipe`
  必须共享同一个 `acc_push_epilogue`
- 所有绑定到该 pipe 的 producer `tpush` 都必须共享同一个
  `acc_push_epilogue`
- consumer 侧若出现多次 `tpop`，这些 `tpop` 的结果类型必须一致
- consumer `tpop` 结果类型必须与这组 pipe config 推导出的 entry type 一致

但这条约束不应被误解为“整个 kernel 里只能有一组 fixpipe config”。

不同逻辑 pipe 之间可以各自选择不同的 `acc_push_epilogue`。例如：

- `pipe0` 使用 `layout = nz2nd, quant = deqf16_scalar, relu = normal_relu`
- `pipe1` 使用 `layout = nz2dn, quant = f32_bf16, relu = no_relu`

这是合法的；要求只是在 `pipe0` 内始终保持 `pipe0` 的配置稳定，在 `pipe1`
内始终保持 `pipe1` 的配置稳定。

不允许：

- 同一 pipe 上一次 `DEQF16`，下一次 `VDEQF16`
- 同一 pipe 上一次 `NZ2ND`，下一次 `NZ2DN`
- 同一 pipe 上混合 `NoRelu` 与 `NormalRelu`

### 6.4 运行时量化参数

对 fixpipe pipe，需要把“静态 mode”和“动态 quant 参数”分开处理：

- `acc_push_epilogue` 是 pipe-level contract
- scalar/vector quant 参数是 producer-side runtime state

因此：

- `NoQuant` 不要求额外 quant config op
- `deqf16_scalar`、`req8_scalar`、`qf322b8_pre_scalar`、
  `qf322f16_pre_scalar`、`qf322bf16_pre_scalar`、`qs322bf16_pre_scalar`（A5-only）、
  `qf322hif8_pre_scalar`、`qf322fp8_pre_scalar`
  这类 scalar quant 模式要求在对应 `tpush` 之前存在匹配的 producer-side
  scalar quant config
- `deqf16_vec`、`req8_vec`、`qf322b8_pre_vec`、`qs322bf16_pre_vec`（A5-only）
  这类 vector quant 模式要求在对应 `tpush` 之前存在匹配的 producer-side
  vector quant config
- `f32_f16` / `f32_bf16` 虽然最终也会把结果写成 `f16` / `bf16`，但它们仍属于
  “静态 mode 直接决定结果类型”的 family，不消费额外 `pto.set_quant_*` runtime state

这类 runtime quant state 的前端语义还应进一步明确为：

- 它是 **per producer context + logical pipe id** 的 producer-side config op，而不是
  per pipe attr
- 底层 `pto-isa` 接口更接近 producer-side machine state；但为了消除多 pipe
  场景下“这一份 quant state 到底属于哪条 pipe”的歧义，PTOIR 第一版应主动把前端
  contract 收紧为“按 logical pipe `id` 显式绑定、直到同 id 新配置覆盖”为止的语义
- `pto.set_quant_scalar {id = k}` 表示更新同一 producer context 中、logical pipe
  `id = k` 当前生效的 scalar quant state
- `pto.set_quant_vector {id = k}` 表示更新同一 producer context 中、logical pipe
  `id = k` 当前生效的 vector quant state
- 在同一基本块内，后续每一条 `id = k` 且 family 匹配的 fixpipe `TPUSH`，都应读取
  当前最近一次、程序顺序上支配它的同类 `pto.set_quant_* {id = k}` 绑定
- 这份按 `id` 建立的 quant binding 会持续生效，直到被下一条同类、同 id 的
  `pto.set_quant_*` 显式覆盖；因此当 payload 不变时，IR 层不需要在每次 `TPUSH`
  前都重复写一条同样的 `pto.set_quant_*`
- 但 lowering / EmitC 为了匹配底层 producer-side machine state，必须在匹配的
  `TPUSH` 前按需重新 materialize `SET_QUANT_SCALAR` / `SET_QUANT_VECTOR`；
  保守实现可以在每次命中的 quant `TPUSH` 前都重发一次，至少在从别的 pipe
  切回 `id = k` 时必须重新发射
- 如果某条需要 scalar/vector quant 的 fixpipe `TPUSH` 在当前基本块内找不到一条
  程序顺序上先于它、同类且同 id 的 `pto.set_quant_*` 绑定，则该 IR 非法
- 当 `acc_push_epilogue.quant = no_convert` 时，该次 `TPUSH` 不消费 quant state
- 因而，同一 kernel 内即使存在多条带不同 `acc_push_epilogue` 的 fixpipe pipe，
  quant state 的归属也始终由“显式 `id` + 程序顺序上最近一次生效绑定 + 实际命中的
  同 id `TPUSH`”决定，而不是由 kernel 级全局唯一配置决定

这里还需要区分“公共 IR 形状”和“payload 的可验证细节”：

- v1 必须公开
  `pto.set_quant_vector(%fp : !pto.tile_buf<loc=scaling, ...>) {id = ...}`
  这一前端 IR 形状
- 当前实现已经把 `vector quant payload` 的一部分公共下限约束前移到了
  `pto.set_quant_vector` 公共 verifier：
  - 输入必须是 `loc=scaling`
  - element type 必须属于 `f16` / `bf16` / `f32` family
- 但 `vector quant payload` 的精确 shape 与更细的 layout / 打包约束，仍不建议在
  当前阶段一次性写死成所有 target 共享的唯一公共 contract
- 更稳妥的做法是：公共 verifier 负责上述公共下限约束，其余 payload shape / layout
  由 arch-specific verifier 根据目标平台现有 fixpipe / quant-vector 约束继续细化

这些 quant config 不应挂在 `initialize_pipe.acc_push_epilogue` 上，也不应并入
`FixpipeParams` 的 pipe attrs。

第一版已经补齐了与 `SET_QUANT_SCALAR` / `SET_QUANT_VECTOR` 一一对应的
前端 pipe IR op 与 lowering 处理。也就是说，除了 pipe-level
`acc_push_epilogue` 外，PTOAS 现在已经显式承载了一套 producer-side
quant-config 表达与 lowering。

第一版显式补了两类前端 op：

- `pto.set_quant_scalar`
- `pto.set_quant_vector`

它们的职责不是修改 pipe attrs，而是在 producer 侧为某个 logical pipe `id`
建立“当前生效、可被后续同 id fixpipe `TPUSH` 复用的 runtime quant state binding”。

同时，这两类 op 在 IR 语义上必须与“消费它们的 fixpipe `TPUSH`”一起，被视为
**有序的 producer-side quant-state 依赖链**：

- 它们不能建模成 `Pure` op
- 它们不应被 DCE / CSE 当成“无结果、可删除”的普通 op
- 它们也不应被随意重排到所配对的 fixpipe `TPUSH` 之后

换句话说，PTOIR / lowering 不能只让 `pto.set_quant_*` 自己带 side-effect，而让
后继 fixpipe `TPUSH` 完全看不到这份 quant-state 依赖；否则后续优化 / 调度仍可能把
`TPUSH` 挪过 `pto.set_quant_*`，把量化状态错误地配到别的 push 上。

因此，实现层至少需要满足其一：

- `pto.set_quant_scalar` / `pto.set_quant_vector` 向同一份 producer quant-state
  resource 写入，而消费它们的 fixpipe `TPUSH` 显式读取或消费这同一份 resource；
  这份 resource 应至少按 logical pipe `id` 做可区分的建模
- 或者 `pto.set_quant_*` 显式产出 ordering token，并要求匹配的 fixpipe `TPUSH`
  显式依赖该 token

在 MLIR 实现层，文档应进一步建议：

- `pto.set_quant_scalar`
- `pto.set_quant_vector`

实现 `MemoryEffectsOpInterface`，并显式建模到一类专门的 producer quant-state
resource 上；同时，消费这些 quant state 的 fixpipe `TPUSH` 也必须对同一 resource
声明可观察的 Read / Write effect。若采用 resource 方案，建议至少按 logical pipe
`id` 对 quant-state resource 做参数化或分片；若不采用 resource 方案，则至少要提供
等价的 ordering token 机制，让 `pto.set_quant_*` 与匹配的同 id fixpipe `TPUSH`
建立显式依赖。

这样后续 canonicalize / CSE / DCE / scheduler 才有稳定依据，不会把“无 SSA result”
误判成“无副作用”。

### 6.5 `slot_size` 语义

对开启 `acc_push_epilogue` 的 fixpipe pipe，`slot_size` 应按 **post-fixpipe 的
consumer-visible entry 语义** 理解，而不是按 source acc tile 的原始元素类型理解。

也就是说：

- `slot_size` 描述的是这条 pipe 在物理 FIFO / slot 中承载的 consumer entry 大小
- 其计算应与该 logical pipe 解析出的 resolved consumer result type 一致
- 它不应再按 producer acc source element type 直接估算

例如：

- `acc<i32> + deqf16_scalar + nz2nd` 的 consumer entry 元素类型是 `f16`
- 则 `slot_size` 应与这条 `f16 + nz2nd` consumer entry 的物理承载大小一致

对不同 target，内部最终走 UB FIFO 还是 GM FIFO 可以不同；但前端 `slot_size`
语义应统一描述 fixpipe 转换后的 consumer entry，而不是 source acc tile。

这一点需要显式覆盖现有普通 pipe 文档里的既有表述：

- 对普通 pipe，仍沿用既有 `slot_size = pre-split full logical entry bytes`
  的语义
- 但对开启 `acc_push_epilogue` 的 fixpipe Acc-producer pipe，这里的 logical entry
  必须按 fixpipe 后的 consumer-visible entry 理解
- 因此旧文档里“按 source entry 直观估算 `slot_size`”的理解不再适用于 fixpipe
  Acc producer

为了让这一规则更可实现，第一版 verifier / lowering 至少应围绕下述下限语义工作：

- `required_slot_size >= physical_size(consumer_shape,
  resolved_consumer_elem_type, resolved_consumer_layout)`

其中：

- `consumer_shape` 由这条 pipe 的逻辑 entry shape 决定
- `resolved_consumer_elem_type` 由 peer consumer `tpop` 结果 dtype 提供，并校验与
  `acc_push_epilogue.quant + SrcElemType` 的前端类型规则一致
- `resolved_consumer_layout` 由 peer consumer `tpop` 结果 tile layout 提供，并校验与
  `acc_push_epilogue.layout` 的 layout contract 一致

如果目标平台对 slot entry 还要求额外对齐，那么实现可以在这个下限之上再叠加
target-specific alignment 约束；但前端 contract 至少应保证：用户给出的 `slot_size`
不能小于 post-fixpipe consumer entry 的物理承载需求。

## 7.  IR 形状

### 7.1 前端 IR

推荐把 fixpipe 配置挂到 `initialize_pipe`：

```mlir
func.func @cube_kernel() {
  pto.aic_initialize_pipe {
    id = 0,
    dir_mask = 1,
    slot_size = 1024,
    nosplit = true,
    acc_push_epilogue =
        #pto.acc_push_epilogue<layout = nz2nd, quant = deqf16_scalar, relu = normal_relu>
  }(...)
}

func.func @vector_kernel() {
  pto.aiv_initialize_pipe {
    id = 0,
    dir_mask = 1,
    slot_size = 1024,
    nosplit = true,
    acc_push_epilogue =
        #pto.acc_push_epilogue<layout = nz2nd, quant = deqf16_scalar, relu = normal_relu>
  }(...)
}
```

推荐新增如下复合 attr：

```tablegen
def PTO_AccPushEpilogueAttr : AttrDef<PTO_Dialect, "AccPushEpilogue"> {
  let mnemonic = "acc_push_epilogue";
  let parameters = (ins
    EnumParameter<PTO_FixpipeLayoutEnum>:$layout,
    EnumParameter<PTO_FixpipeQuantEnum>:$quant,
    EnumParameter<PTO_FixpipeReluEnum>:$relu
  );
  let assemblyFormat =
    "`<` `layout` `=` $layout `,` `quant` `=` $quant `,` `relu` `=` $relu `>`";
}
```

也就是说，前端不再用 3 个分散的 attr 表达这组语义，而是统一收敛到
`acc_push_epilogue` 这一个复合 attr。

当前实现使用 dedicated `PTO_FixpipeLayoutEnum` / `PTO_FixpipeQuantEnum` /
`PTO_FixpipeReluEnum` 作为 `acc_push_epilogue` 的前端公开值域：

- 这样可以在 parse surface 就把未公开的 fixpipe 枚举值挡掉，而不是先允许写入
  IR、再到 verifier 阶段二次收窄
- 与此同时，底层现有 `PTO_AccStoreQuantPreModeEnum` / `PTO_ReluPreModeEnum`
  仍可继续保留给其它非 fixpipe surface 使用

其中 `quant` / `relu` 的前端允许集合应与第 10 节 rule 8 / rule 7 保持一致：

- `quant` 允许集合应与第 10 节 rule 8 保持一致；当前 v1 只允许：
  `no_convert` / `f32_f16` / `req8_scalar` / `req8_vec` /
  `deqf16_scalar` / `deqf16_vec` / `f32_bf16` /
  `qf322b8_pre_scalar` / `qf322b8_pre_vec` /
  `qf322f16_pre_scalar` / `qf322bf16_pre_scalar` /
  `qs322bf16_pre_scalar` / `qs322bf16_pre_vec` /
  `qf322hif8_pre_scalar` / `qf322fp8_pre_scalar`
- `relu` 只允许：`no_relu` / `normal_relu`

其余现存 enum 值在 v1 中都应在前端 surface 直接拒绝，而不是默认为“语法合法但语义待定”。
在当前 dedicated enum 实现下，这一拒绝会发生在 parse 阶段，而不是等到 verifier
再兜底。

与之绑定的 `tpush` 保持轻量：

```mlir
pto.tpush_to_aiv(%acc_tile : !pto.tile_buf<...>) {id = 0, split = 0}
```

`tpop` 也不新增 config operand 或 attr：

```mlir
%recv = pto.tpop_from_aic {id = 0, split = 0} -> !pto.tile_buf<...>
```

这份 `acc_push_epilogue` 应作为同一条 logical pipe 的双端镜像 contract：

- producer Cube function 内的 `pto.aic_initialize_pipe` 显式携带
- consumer Vector function 内的 peer `pto.aiv_initialize_pipe` 也显式携带
- 两端保持逐字段一致
- `tpush` / `tpop` 仍不重复携带这组配置

这里的 “peer” 不建议抽象成“另一个刚好 `id` 相同的 init”。更具体地说：

- 第一版应复用 `reserve_buffer` / `import_reserved_buffer` 与 `peer_func`
  已经建立的 pipe peer 关系来找到对端 init
- `id` 只在各自 function 内承担 data op 到 local init 的查找职责
- 跨函数的 producer / consumer contract check 不应把 `id` 当作唯一配对键

### 7.2 producer-side quant config IR

为覆盖 `SET_QUANT_SCALAR` / `SET_QUANT_VECTOR`，第一版 PTOIR 已新增两类显式 op。

标量量化配置：

```mlir
pto.set_quant_scalar(%scale : f32) {id = 0}
```

向量量化配置：

```mlir
pto.set_quant_vector(%fp : !pto.tile_buf<loc=scaling, ...>) {id = 0}
```

推荐语义：

- `pto.set_quant_scalar {id = k}` 表示更新同一 producer context 中、logical pipe
  `id = k` 当前生效的 scalar quant state
- `pto.set_quant_vector {id = k}` 表示更新同一 producer context 中、logical pipe
  `id = k` 当前生效的 vector quant state
- 这里的 `id` 复用当前 producer function 内 `initialize_pipe` / `tpush` 的本地
  logical pipe id 命名空间；它只承担本地 producer-side 绑定职责，不单独承担跨函数
  peer pipe 配对职责
- 第一版 verifier 先收紧为“同一基本块内、按 `id` 与程序顺序建立最近生效绑定”，
  不要求跨 block 的一般化 dominance 推导
- 对任意一条需要 scalar/vector quant 的 fixpipe `TPUSH`，它实际读取的是同一基本块
  内、程序顺序上最近一条同类且同 id 的 `pto.set_quant_*`
- 这份绑定会一直持续到被下一条同类、同 id 的 `pto.set_quant_*` 覆盖；因此当同一
  pipe 的 quant payload 不变时，IR 层只需要绑定一次
- 与之配对的 fixpipe `TPUSH` 必须显式参与同一份 quant-state 依赖，无论实现选择
  shared resource 还是 ordering token，都不能只让 `pto.set_quant_*` 单方面带
  side-effect
- 二者都属于 producer-side config op，而不是 pipe op
- 二者都不返回 SSA result；它们建模的是“影响后续 producer-side fixpipe push 的
  machine state”
- 二者都必须保持 side-effect / ordering 语义，不能作为 `Pure` 无结果 op 参与
  DCE / CSE / 任意重排
- lowering / EmitC 可以保守地在每次命中的 quant `TPUSH` 前都重新发射一次
  `SET_QUANT_SCALAR` / `SET_QUANT_VECTOR`；至少在从其它 pipe 切回当前 `id = k`
  时必须重发，以反映底层全局 machine state 的重新配置
- 如果某条需要 scalar/vector quant 的 fixpipe `TPUSH` 在当前基本块内找不到同类、
  同 id、且程序顺序上先于它的 `pto.set_quant_*`，则该 IR 非法

`pto-isa` 当前底层接口是：

```cpp
SET_QUANT_SCALAR<OutType>(scalar);
```

但在 PTOIR 设计中，不建议把 `OutType` 再额外挂成 `pto.set_quant_scalar`
的独立 attr。更合适的做法是：

- `acc_push_epilogue.quant` 继续表达量化模式
- producer 侧 `tpush` 的 source acc tile element type 继续表达源类型
- consumer 侧同一 logical pipe 的 `tpop` 结果 dtype 继续表达 destination element
  type，并要求同一条 pipe 上的所有 `tpop` 结果类型一致
- lowering 在 peer contract verify 完成后，对每一条命中的 scalar quant `TPUSH`
  根据它的 `id` 找到当前 producer function 内目标 logical pipe，再为该 pipe 解析
  一份 resolved consumer element type，并从当前生效的
  `pto.set_quant_scalar {id = k}` 取出 payload，据此发射
  `SET_QUANT_SCALAR<OutType>(scalar)`

这样设计的原因是：

- 对本文当前已公开的 scalar quant 子集，这份 resolved consumer element type 会：
  - 在 `deqf16_scalar` / `qf322f16_pre_scalar` 上退化成 `f16`
  - 在 `qf322bf16_pre_scalar` / `qs322bf16_pre_scalar` 上退化成 `bf16`
  - 在 `qf322hif8_pre_scalar` 上退化成 `!pto.hif8`
  - 在 `qf322fp8_pre_scalar` 上退化成 `f8E4M3FN`
- 对 `req8_scalar` / `qf322b8_pre_scalar`，同一机制会直接从 peer consumer `tpop`
  dtype 中拿到
  `SET_QUANT_SCALAR<int8_t>` / `SET_QUANT_SCALAR<uint8_t>` 所需的 signedness
- 因而 `pto.set_quant_scalar` 不需要再平行暴露一份 `out_type`

这里的 `pto.set_quant_scalar(%scale : f32) {id = k}` 是刻意对齐当前 fixpipe
`SET_QUANT_SCALAR(float)` 这一 `pto-isa` 接口；它不等同于另一套 structured
acc-store `pre_quant` payload surface。

`pto.set_quant_vector` 同理不需要额外类型 attr。当前 PTOIR 公共 contract 已经把
`SET_QUANT_VECTOR(fpTile)` 的输入下限收紧为：

- `loc=scaling`
- element type 属于 `f16` / `bf16` / `f32` family

但它仍不需要像 scalar quant 那样额外暴露一份 `out_type`；更细的 payload
shape / layout 约束仍可继续留给 arch-specific verifier 与 lowering 处理。

### 7.3 前端 IR 使用样例

scalar quant 示例：

```mlir
func.func @cube_kernel() {
  pto.aic_initialize_pipe {
    id = 0,
    dir_mask = 1,
    slot_size = 1024,
    nosplit = true,
    acc_push_epilogue =
        #pto.acc_push_epilogue<layout = nz2nd, quant = deqf16_scalar, relu = normal_relu>
  }(...)

  pto.set_quant_scalar(%scale : f32) {id = 0}
  pto.tpush_to_aiv(%acc_tile : !pto.tile_buf<loc=acc, ...>) {id = 0, split = 0}
}

func.func @vector_kernel() {
  pto.aiv_initialize_pipe {
    id = 0,
    dir_mask = 1,
    slot_size = 1024,
    nosplit = true,
    acc_push_epilogue =
        #pto.acc_push_epilogue<layout = nz2nd, quant = deqf16_scalar, relu = normal_relu>
  }(...)

  %recv = pto.tpop_from_aic {id = 0, split = 0} -> !pto.tile_buf<...>
}
```

后续示例若只展开 producer 侧写法，仅仅是为了压缩篇幅；对应 peer consumer function
中的 `pto.aiv_initialize_pipe` 仍应镜像携带同一份 `acc_push_epilogue`。

对本文当前已公开的 quant 子集，consumer `tpop` 结果 dtype 的直观对应关系至少应满足：

- `deqf16_scalar` / `deqf16_vec` / `qf322f16_pre_scalar` -> `f16`
- `f32_bf16` / `qf322bf16_pre_scalar` / `qs322bf16_pre_scalar` / `qs322bf16_pre_vec`
  -> `bf16`
- `req8_scalar` / `qf322b8_pre_scalar` -> 显式 `si8` 或 `ui8`
- `req8_vec` / `qf322b8_pre_vec` -> 显式 `si8`
- `qf322hif8_pre_scalar` -> `!pto.hif8`
- `qf322fp8_pre_scalar` -> `f8E4M3FN`

vector quant 示例：

```mlir
pto.aic_initialize_pipe {
  id = 0,
  dir_mask = 1,
  slot_size = 1024,
  nosplit = true,
  acc_push_epilogue =
      #pto.acc_push_epilogue<layout = nz2nd, quant = deqf16_vec, relu = no_relu>
}(...)

pto.set_quant_vector(%fp_tile : !pto.tile_buf<loc=scaling, ...>) {id = 0}
pto.tpush_to_aiv(%acc_tile : !pto.tile_buf<loc=acc, ...>) {id = 0, split = 0}
```

`NoQuant` 示例则不需要任何 quant-config op：

```mlir
pto.aic_initialize_pipe {
  id = 0,
  dir_mask = 1,
  slot_size = 1024,
  nosplit = true,
  acc_push_epilogue =
      #pto.acc_push_epilogue<layout = nz2nd, quant = no_convert, relu = no_relu>
}(...)

pto.tpush_to_aiv(%acc_tile : !pto.tile_buf<loc=acc, ...>) {id = 0, split = 0}
```

同一 kernel 中允许多条 fixpipe pipe 各自挂不同 `acc_push_epilogue`。例如：

```mlir
pto.aic_initialize_pipe {
  id = 0,
  dir_mask = 1,
  slot_size = 1024,
  nosplit = true,
  acc_push_epilogue =
      #pto.acc_push_epilogue<layout = nz2nd, quant = deqf16_scalar, relu = normal_relu>
}(...)

pto.aic_initialize_pipe {
  id = 1,
  dir_mask = 1,
  slot_size = 1024,
  nosplit = true,
  acc_push_epilogue =
      #pto.acc_push_epilogue<layout = nz2dn, quant = deqf16_vec, relu = no_relu>
}(...)

pto.set_quant_scalar(%scale0 : f32) {id = 0}
pto.tpush_to_aiv(%acc0 : !pto.tile_buf<loc=acc, ...>) {id = 0, split = 0}

pto.set_quant_vector(%fp1 : !pto.tile_buf<loc=scaling, ...>) {id = 1}
pto.tpush_to_aiv(%acc1 : !pto.tile_buf<loc=acc, ...>) {id = 1, split = 0}
```

这时：

- `pipe0` 与 `pipe1` 的 `acc_push_epilogue` 可以不同
- `pto.set_quant_scalar(%scale0) {id = 0}` 绑定的是 `pipe0` 当前生效的 scalar quant
  payload；直到新的 `id = 0` scalar 配置出现前，后续 `pipe0` 的 scalar quant
  `TPUSH` 都可以继续复用它
- `pto.set_quant_vector(%fp1) {id = 1}` 同理绑定 `pipe1` 当前生效的 vector quant
  payload

如果同一基本块里交错使用多条 scalar quant pipe，则 IR 层不要求用户在“切回旧 pipe”
时再次重复写一条相同的 `pto.set_quant_scalar`。例如：

```mlir
pto.aic_initialize_pipe {
  id = 0,
  dir_mask = 1,
  slot_size = 1024,
  nosplit = true,
  acc_push_epilogue =
      #pto.acc_push_epilogue<layout = nz2nd, quant = deqf16_scalar, relu = normal_relu>
}(...)

pto.aic_initialize_pipe {
  id = 1,
  dir_mask = 1,
  slot_size = 1024,
  nosplit = true,
  acc_push_epilogue =
      #pto.acc_push_epilogue<layout = nz2nd, quant = qf322bf16_pre_scalar, relu = no_relu>
}(...)

pto.set_quant_scalar(%scale0 : f32) {id = 0}
pto.tpush_to_aiv(%acc0 : !pto.tile_buf<loc=acc, ...>) {id = 0, split = 0}

pto.set_quant_scalar(%scale1 : f32) {id = 1}
pto.tpush_to_aiv(%acc1 : !pto.tile_buf<loc=acc, ...>) {id = 1, split = 0}

pto.tpush_to_aiv(%acc2 : !pto.tile_buf<loc=acc, ...>) {id = 0, split = 0}
```

这时：

- `pipe0` 的 `%scale0` 绑定在 IR 中仍然持续有效，因为其间没有新的
  `pto.set_quant_scalar {id = 0}` 覆盖它
- 最后一条 `pto.tpush_to_aiv {id = 0}` 合法地复用这份 `pipe0` 绑定
- lowering / EmitC 为了匹配底层全局 quant machine state，必须在切回 `pipe0` 时
  自动重新 materialize 一次 `SET_QUANT_SCALAR<half>(scale0)`；用户不需要在 IR
  层手写第二条完全相同的 `pto.set_quant_scalar {id = 0}`

### 7.4 为什么不把 config 写到 `tpush`

不推荐如下形状：

```mlir
pto.tpush_to_aiv(%acc_tile : !pto.tile_buf<...>) {
  id = 0,
  split = 0,
  acc_push_epilogue = ...
}
```

因为它会把 pipe-level contract 拆散到多条 producer op 上。

同理，也不建议把 scalar/vector quant payload 直接塞进 `tpush` operand：

```mlir
pto.tpush_to_aiv(%acc_tile, %scale_or_fp_tile) { ... }
```

因为 `SET_QUANT_SCALAR` / `SET_QUANT_VECTOR` 在 `pto-isa` 中本来就是独立的
producer-side state config，而不是 `TPUSH` 的显式 operand。

## 8. 类型推导

对于 fixpipe pipe，logical pipe 的 consumer entry type 不是单纯由 source tile
type 决定，而是由：

- source tile type
- `acc_push_epilogue.quant`
- `acc_push_epilogue.layout`

共同决定。

其中：

- `acc_push_epilogue.quant` 决定结果元素类型
- `acc_push_epilogue.layout` 决定结果 tile layout，例如 `NZ2ND` 导向 vec row-major 视角
- `acc_push_epilogue.relu` 不改变类型，但改变数值语义

更具体地说，logical pipe 的 resolved consumer element type 应由 consumer 侧
`tpop` 结果 dtype 给出；对本文当前已纳入公开值域的 quant 子集，verifier 再要求这份
resolved consumer element type 与 `pto-isa` 当前
`FixpipeConsDType_t<quant, SrcElemType>::type` 的推导规则对齐。

如果同一 kernel 内存在多条 fixpipe pipe，则每一条 consumer / destination type
都应相对于“当前这条 `tpush` 所绑定的那条 pipe 的 `acc_push_epilogue`”独立推导，
而不是假定整个 kernel 只有一个全局唯一的 fixpipe 结果类型。

下文所说的 **consumer / destination element type**，指的是用户可见 consumer
`tpop` 结果 tile 的 element type；如果 lowering 先引入等价的内部 destination tile，
则该内部类型必须与用户可见 `tpop` 结果类型一致。

如果同一 logical pipe 在 consumer 侧出现多次 `tpop`，则这些 `tpop` 的结果 tile
type 不应彼此冲突。第一版至少应要求：

- result element type 一致
- result layout 一致
- 对 scalar 8-bit family，signedness 也必须一致，且结果 element type 只能是
  `si8` 或 `ui8`，不得使用 signless `i8`
- 对 vector 8-bit family，结果 element type 固定为 `si8`，不得使用 `ui8` 或
  signless `i8`

如果 consumer 侧不存在任何 `tpop`，则当前 contract 下无法解析这条 logical pipe 的
resolved consumer type；第一版应直接把这种 fixpipe pipe 视为非法 IR。

前端 verifier / peer contract check 应把这份一致的 `tpop` 结果类型视为该 logical
pipe 的 resolved consumer type。

如果 source element type 记为 `SrcElemType`，那么 v1 应按如下规则校验 logical
pipe 的 resolved consumer result type：

- `resolved_consumer_elem_type` 由 peer consumer `tpop` 结果 dtype 提供
- `consumer_layout` 由 `acc_push_epilogue.layout` 决定。第一版可直接按 consumer
  result tile type 的 layout config 参数做 verifier：
  - `layout = nz2nd` 时，consumer tile 应为 vec `BLayout::RowMajor`
  - `layout = nz2dn` 时，consumer tile 应为 vec `BLayout::ColMajor`
  - `layout = nz2nz` 时，consumer tile 应为 vec `BLayout::ColMajor`，并带
    `SLayout::RowMajor`
  对当前 A5 no-split C2V UB fixpipe 路径，这正好对应 `pto-isa` 现有
  `FixpipeVecTile` 规则；对走 GM FIFO 视图的路径，前端语义上仍应满足同一
  layout contract，只是内部可映射成对应的 ND / DN / NZ 全局布局表示
- `acc_push_epilogue.relu` 不参与类型推导

第一版文档建议至少按下面这组公共映射规则做 verifier 校验：

| `acc_push_epilogue.quant` | `resolved_consumer_elem_type` 约束 |
|---|---|
| `no_convert` | `SrcElemType` |
| `f32_f16` | `f16` |
| `req8_scalar` | `si8` 或 `ui8` |
| `req8_vec` | `si8` |
| `deqf16_scalar` / `deqf16_vec` | `f16` |
| `f32_bf16` | `bf16` |
| `qf322b8_pre_scalar` | `si8` 或 `ui8` |
| `qf322b8_pre_vec` | `si8` |
| `qf322f16_pre_scalar` | `f16` |
| `qf322bf16_pre_scalar` | `bf16` |
| `qs322bf16_pre_scalar` / `qs322bf16_pre_vec` | `bf16`（A5-only） |
| `qf322hif8_pre_scalar` | `!pto.hif8` |
| `qf322fp8_pre_scalar` | `f8E4M3FN` |

如果后续需要支持更多 `QuantMode_t` 枚举值，也应继续按
`FixpipeConsDType_t` 的同一套规则扩展，而不是单独定义另一份类型映射。

但仅靠上面的 resolved consumer element type 约束还不够；v1 verifier 还应同时
约束合法的 producer source acc element type。推荐至少明确如下规则：

| `acc_push_epilogue.quant` | producer acc `SrcElemType` 要求 | consumer `DstElemType` |
|---|---|---|
| `no_convert` | `f32` 或 `i32`，且 `DstElemType = SrcElemType` | `SrcElemType` |
| `f32_f16` | `f32` | `f16` |
| `req8_scalar` | `i32` | `si8` 或 `ui8` |
| `req8_vec` | `i32` | `si8` |
| `deqf16_scalar` | `i32` | `f16` |
| `deqf16_vec` | `i32` | `f16` |
| `f32_bf16` | `f32` | `bf16` |
| `qf322b8_pre_scalar` | `f32` | `si8` 或 `ui8` |
| `qf322b8_pre_vec` | `f32` | `si8` |
| `qf322f16_pre_scalar` | `f32` | `f16` |
| `qf322bf16_pre_scalar` | `f32` | `bf16` |
| `qs322bf16_pre_scalar` / `qs322bf16_pre_vec` | `i32` | `bf16`（A5-only） |
| `qf322hif8_pre_scalar` | `f32` | `!pto.hif8` |
| `qf322fp8_pre_scalar` | `f32` | `f8E4M3FN` |

原因是 `FixpipeConsDType_t<quant, SrcElemType>` 只负责“给定 quant 和 `SrcElemType`
时结果类型如何折叠”，并不会单独保证 `quant` 与 source dtype 的语义组合一定合法。
因此前端 verifier 需要显式拒绝例如：

- `no_convert + acc<f16>`
- `no_convert + acc<i16>`
- `f32_f16 + i32 acc`
- `req8_scalar + f32 acc`
- `req8_vec + f32 acc`
- `f32_bf16 + i32 acc`
- `deqf16_scalar + f32 acc`
- `deqf16_vec + f32 acc`
- `qf322b8_pre_scalar + i32 acc`
- `qf322b8_pre_vec + i32 acc`
- `qf322f16_pre_scalar + i32 acc`
- `qf322bf16_pre_scalar + i32 acc`
- `qs322bf16_pre_scalar + f32 acc`
- `qs322bf16_pre_vec + f32 acc`
- `qf322hif8_pre_scalar + i32 acc`
- `qf322fp8_pre_scalar + i32 acc`

需要单独说明的是：`pto-isa` 当前 `FixpipeConsDType_t` 在 8-bit family 上会折叠成
`int8_t`，因此它并不能单独承担全部 8-bit destination signedness 的前端语义。

- 对本文当前已经纳入公开值域的 `req8_scalar` / `qf322b8_pre_scalar`，这部分
  signedness 必须来自 peer consumer `tpop` 的结果 element type，并要求同一
  logical pipe 上的所有 `tpop` 结果类型一致；同时 consumer result element type
  必须显式写成 `si8` 或 `ui8`，不得使用 signless `i8`
- 对 `req8_vec` / `qf322b8_pre_vec`，当前 `pto-isa` fixpipe vector quant 路径与
  `SET_QUANT_VECTOR` 都没有类似 `SET_QUANT_SCALAR<OutType>` 的 unsigned 通道，
  因此前端 v1 应直接把 consumer result element type 收紧为显式 `si8`，不得使用
  `ui8` 或 signless `i8`
- 对 `qs322bf16_pre_scalar` / `qs322bf16_pre_vec`，当前 `pto-isa` 源码证据只在 A5
  路径坐实；因此虽然公共类型规则可写成 `bf16`，但前端 v1 仍应额外要求 target/profile
  支持这两条 A5 路径，否则 verifier 应直接拒绝

因此，v1 的精确类型一致性约束拆成两层：

- producer 侧本地检查：结合 producer `tpush` 的 source acc tile type 与本地
  `aic_initialize_pipe.acc_push_epilogue`，推导 logical pipe 的期望 consumer
  entry type
- peer logical pipe 一致性检查：再确认 consumer 侧 `tpop` 结果类型与 peer
  `aiv_initialize_pipe` 镜像 contract 与 producer 侧推导结果一致

也就是说，consumer 不需要在单个 `tpop_from_aic` verifier 中独自恢复 producer
`SrcElemType`；这项精确匹配更适合作为 peer logical pipe contract check 的一部分。

需要注意，`SET_QUANT_SCALAR` / `SET_QUANT_VECTOR` 属于 producer-side runtime
quant state 约束，不属于 `FixpipeConsDType_t` 的类型推导本身：

- consumer elem type 是否匹配：
  v1 公开值域内看 `acc_push_epilogue.quant` 与 resolved consumer `tpop` result
  dtype 是否满足上表约束
- scalar/vector quant config 是否齐备，看对应 `SET_QUANT_SCALAR` /
  `SET_QUANT_VECTOR` verifier 规则
- 对 `SET_QUANT_SCALAR<OutType>` 而言，`OutType` 不作为独立前端 attr 暴露；
  它应来自当前 logical pipe 的 resolved consumer element type。对本文当前已公开
  的 scalar quant 子集：
  - `deqf16_scalar` / `qf322f16_pre_scalar` -> `half`
  - `qf322bf16_pre_scalar` / `qs322bf16_pre_scalar` -> `bfloat16_t`
  - `qf322hif8_pre_scalar` -> `hifloat8_t`
  - `qf322fp8_pre_scalar` -> `float8_e4m3_t`
  - `req8_scalar` / `qf322b8_pre_scalar` 上由 `si8` / `ui8` 分别落成 `int8_t` /
    `uint8_t`

对 `no_convert` 还应单独收紧一层 source-type 合法值，而不是简单写成“任意
`SrcElemType`”：

- 当前 v1 fixpipe Acc producer 路径建议至少限制为 `SrcElemType in {f32, i32}`
- 并继续要求 `DstElemType = SrcElemType`

也就是说，v1 `no_convert` 更准确的公共语义应是：

- `f32 -> f32`
- `i32 -> i32`

而不是把 `acc<f16>`、`acc<i16>` 之类当前后端并不能稳定 lower 的组合提前放进
前端 contract。

## 9. Lowering 方案

### 9.1 前端到内部 pipe IR

前端 `initialize_pipe` lowering 到内部 `!pto.pipe` 时：

- 保留普通 pipe 字段：`id / dir_mask / slot_size / slot_num / nosplit`
- 额外挂上 `acc_push_epilogue` 字段

推荐内部 pipe 继续保持 opaque handle，不把这些字段塞进 `!pto.pipe` type 参数。

由于 `acc_push_epilogue` 在 producer / consumer 两端 init 上镜像存在，第一版更合适
的流程是：

- 先在 frontend `initialize_pipe` 仍然显式存在时，对 module 内同一 logical pipe
  的 producer / consumer peer init 做一次 contract verify
- 这一步更适合发生在 frontend init 被擦除之前，而不是等到所有 frontend init 都被
  lower / erase 之后再临时回溯恢复
- producer function 的 `aic_initialize_pipe` 在本地 lowering 时保留这份 contract
- consumer function 的 `aiv_initialize_pipe` 在本地 lowering 时也保留这份 contract
- frontend init lowering 后，内部 pipe handle / anchor metadata 仍应继续保留
  `acc_push_epilogue`，直到 EmitC 消费完毕
- 在 frontend `initialize_pipe` 仍可见阶段，还应一并收集同一 logical pipe 上所有
  consumer `tpop` 的结果类型；若一条 pipe 没有任何 `tpop`，或多次 `tpop`
  结果类型彼此不一致，都应在这一阶段报错；通过校验后再把这份 resolved consumer
  type 记录到内部 pipe metadata 中
- 在进入 EmitC 前，还应能再次基于这份保留下来的 metadata 做一致性 sanity check，
  确认 `acc_push_epilogue` 逐字段一致，且 consumer 结果类型满足 producer 侧推导
  的 logical pipe entry type

这样既复用了当前 frontend init 的 function-local 查找模型，也避免让单个 data op 的
verifier / lowering 去跨函数读取 peer init。

其中 “producer / consumer peer init” 的定位不应靠 module 级裸 `id` 猜测，而应复用
当前 frontend pipe 对 peer buffer 的建模关系：

- 对 C2V / V2C local buffer，按 `reserve_buffer` / `import_reserved_buffer` 的
  逻辑 key 配对
- `peer_func` 用来限定对端 function
- reserve/import 的逻辑 buffer 名称用于区分同一对 function 之间的多条 pipe

这样即使同一个 module 中有多组 cube/vector kernel 都使用 `id = 0`，peer contract
check 也不会误把不相关的 pipe init 配到一起。

如果某条开启 `acc_push_epilogue` 的 pipe 在 frontend IR 中给出的 `c2v_consumer_buf`
无法继续追溯到 `reserve_buffer` / `import_reserved_buffer`，那么这条 pipe 不应进入
“尝试做 peer contract verify”阶段，而应直接作为非法 IR 拒绝。

新增的 `pto.set_quant_scalar` / `pto.set_quant_vector` 不需要并入 `!pto.pipe`。
它们应继续作为普通 producer-side op 保留在 IR 中，并在后续 lowering 中按顺序
映射到 EmitC。

### 9.2 内部 pipe 到 EmitC

当 `pto.tpush(%tile, %pipe)` 命中 fixpipe pipe 时，EmitC 不再生成普通：

```cpp
TPUSH<Pipe, TileProd, TileSplitAxis::TILE_NO_SPLIT>(pipe, tile);
```

而是生成：

```cpp
using Pipe0FixpipeConfig = FixpipeParams<
    LayoutMode_t::NZ2ND,
    QuantMode_t::DEQF16,
    ReluPreMode::NormalRelu>;
TPUSH<Pipe, TileProd, Pipe0FixpipeConfig>(pipe, tile);
```

也就是说：

- 前端 IR 不直接暴露 `TConfig`
- EmitC 阶段根据 pipe-level `acc_push_epilogue` 组装 `FixpipeParams`，并发射对应的
  `TPUSH<Pipe, TileProd, TConfig>(pipe, tile)` 模板调用
- 若同一 kernel 中有多次命中同一条 fixpipe pipe 的 `TPUSH`，应复用同一个
  config type alias，而不是在每次 `TPUSH` 前重复声明同名 alias
- 若同一 kernel 中存在多条 fixpipe pipe，则应按 pipe 维度生成唯一别名，例如
  `Pipe0FixpipeConfig`、`Pipe1FixpipeConfig`
- 之所以推荐“每条 pipe 一个唯一 alias”，是为了避免在同一作用域下把同名
  alias 重复绑定到不同 `FixpipeParams<...>` 实参时引入 C++ 重定义冲突
- 第一版 EmitC 只从 PTOIR 读取 `acc_push_epilogue`，再从中拆出
  `layout`、`quant`、`relu`
- `layout` / `quant` lowering 都应使用**显式符号映射**，而不是依赖 PTOIR enum
  ordinal 与 `pto-isa` enum ordinal 恰好一致
- 特别是 `layout` 不能直接做 integer cast；推荐固定映射为：
  - `nz2nd -> LayoutMode_t::NZ2ND`
  - `nz2dn -> LayoutMode_t::NZ2DN`
  - `nz2nz -> LayoutMode_t::NZ2NZ`
- v1 `quant` 也应固定映射为：
  - `no_convert -> QuantMode_t::NoQuant`
  - `f32_f16 -> QuantMode_t::F322F16`
  - `req8_scalar -> QuantMode_t::REQ8`
  - `req8_vec -> QuantMode_t::VREQ8`
  - `deqf16_scalar -> QuantMode_t::DEQF16`
  - `deqf16_vec -> QuantMode_t::VDEQF16`
  - `f32_bf16 -> QuantMode_t::F322BF16`
  - `qf322b8_pre_scalar -> QuantMode_t::QF322B8_PRE`
  - `qf322b8_pre_vec -> QuantMode_t::VQF322B8_PRE`
  - `qf322f16_pre_scalar -> QuantMode_t::QF322F16_PRE`
  - `qf322bf16_pre_scalar -> QuantMode_t::QF322BF16_PRE`
  - `qs322bf16_pre_scalar -> QuantMode_t::QS322BF16_PRE`（A5-only）
  - `qs322bf16_pre_vec -> QuantMode_t::VQS322BF16_PRE`（A5-only）
  - `qf322hif8_pre_scalar -> QuantMode_t::QF322HIF8_PRE`
  - `qf322fp8_pre_scalar -> QuantMode_t::QF322FP8_PRE`
- `STPhase`、`AtomicType`、`SubBlockId`、`ClipReluMode_t`、`IsChannelSplit`
  虽然在 `pto-isa` 某些实现里可能被消费，但它们当前阶段暂不作为 PTOIR 前端公开配置项
- 因此这里的“兼容”是指继续复用 `TPUSH<Pipe, TileProd, FixpipeParams<...>>`
  这一调用形状，而不是承诺 PTOAS 前端已经完整覆盖全部 `FixpipeParams`
  模板参数语义
- 若实现上不希望引入别名，也可以直接内联
  `FixpipeParams<LayoutMode_t::..., QuantMode_t::..., ReluPreMode::...>` 作为
  `TPUSH` 的第三个模板实参；本文档只把“每条 pipe 一个唯一 alias”作为推荐生成形状
- 对当前暂未前端公开的 `FixpipeParams` 模板参数，第一版 EmitC 应**有意依赖**
  `pto-isa` 的默认模板实参，而不是把它们视为“无定义 / 无关字段”：
  - `STPhase = STPhase::Unspecified`
  - `SubBlockId = 0`
  - `AtomicType = AtomicType::AtomicNone`
  - `ClipReluMode_t = ClipReluMode_t::NOCLIP_RELU`
  - `IsChannelSplit = false`
- 其中 `SubBlockId = 0` 在 A5 no-split C2V UB fixpipe 路径上应理解为：
  v1 只覆盖 `AccToVecMode::SingleModeVec0`
- 这并不等价于“所有 no-split single-vector 消费场景都已被 v1 覆盖”
- 如果后续要支持 `SingleModeVec1`，或者支持更复杂的 dual-vector 消费 / target
  特化选择，就不能继续把 `SubBlockId` 简单视为“后端无关默认值”；届时应明确把它提升为
  target-specific lowering decision，或纳入更完整的前端 contract
- `SET_QUANT_SCALAR` / `SET_QUANT_VECTOR` 不属于 `FixpipeParams` 模板参数；
  它们应作为独立 producer-side op 保留在 IR 中，并按每条命中的 quant `TPUSH`
  做 rematerialization lowering。例如：
  - `pto.set_quant_scalar(%scale) {id = k}`
    -> 在后续命中的同 id scalar quant `TPUSH` 前，按需发射
       `SET_QUANT_SCALAR<OutType>(scale)`
  - `pto.set_quant_vector(%fp_tile) {id = k}`
    -> 在后续命中的同 id vector quant `TPUSH` 前，按需发射
       `SET_QUANT_VECTOR(fpTile)`
- 其中 `SET_QUANT_SCALAR<OutType>` 的 `OutType` 不来自独立前端 attr；它应来自该
  logical pipe 解析后的 resolved consumer element type
- 这份 resolved consumer type 由 peer consumer `tpop` 的结果 dtype 提供，并要求
  同一条 pipe 上的所有 `tpop` 结果类型一致
- 对本文当前已公开的 scalar quant 子集，这一 resolved consumer type 会稳定落成：
  - `deqf16_scalar` / `qf322f16_pre_scalar` -> `half`
  - `qf322bf16_pre_scalar` / `qs322bf16_pre_scalar` -> `bfloat16_t`
  - `qf322hif8_pre_scalar` -> `hifloat8_t`
  - `qf322fp8_pre_scalar` -> `float8_e4m3_t`
  - `req8_scalar` / `qf322b8_pre_scalar` -> `int8_t` 或 `uint8_t`
- 同一 kernel 中若存在多条带不同 `acc_push_epilogue` 的 fixpipe pipe，EmitC 应按
  每条 `TPUSH` / `pto.set_quant_*` 实际绑定的 pipe `id` 分别读取各自的
  `acc_push_epilogue`，而不是假设 kernel 内只有一份全局 fixpipe 配置
- `pto.set_quant_scalar` / `pto.set_quant_vector` 的 lowering 不应从“某个全局
  `TPOP` 类型”反推；更直接的做法是对每一条 quant `TPUSH` 建立
  `TPUSH{id = k} -> 同基本块内最近一条同类、同 id 的 pto.set_quant_*`
  的解析结果，再从该 `TPUSH` 所属 pipe 的 resolved consumer type + 当前生效
  payload 发射具体 EmitC
- 单条 `pto.set_quant_scalar {id = k}` / `pto.set_quant_vector {id = k}` 在 EmitC 中
  可能对应多次 `SET_QUANT_*` 发射；这是为了在跨 pipe 交错执行时正确恢复底层
  producer-side machine state，属于预期的 rematerialization，而不是 IR 重复表达
- 如果某条 quant `TPUSH` 在前端 IR 中找不到当前生效的同类、同 id 绑定，应在
  verifier 阶段报错，而不是把歧义推迟到 EmitC
- 例如下列交错 IR：

```mlir
pto.set_quant_scalar(%scale0 : f32) {id = 0}
pto.tpush_to_aiv(%acc0 : !pto.tile_buf<loc=acc, ...>) {id = 0, split = 0}

pto.set_quant_scalar(%scale1 : f32) {id = 1}
pto.tpush_to_aiv(%acc1 : !pto.tile_buf<loc=acc, ...>) {id = 1, split = 0}

pto.tpush_to_aiv(%acc2 : !pto.tile_buf<loc=acc, ...>) {id = 0, split = 0}
```

  可合法 lower 为：

```cpp
SET_QUANT_SCALAR<half>(scale0);
TPUSH<Pipe0, AccTile0, Pipe0FixpipeConfig>(pipe0, acc0);

SET_QUANT_SCALAR<bfloat16_t>(scale1);
TPUSH<Pipe1, AccTile1, Pipe1FixpipeConfig>(pipe1, acc1);

SET_QUANT_SCALAR<half>(scale0);
TPUSH<Pipe0, AccTile0, Pipe0FixpipeConfig>(pipe0, acc2);
```

  其中最后一次 `SET_QUANT_SCALAR<half>(scale0)` 就是“切回 `pipe0` 时自动重发”的
  rematerialization；IR 层不要求用户再手写第二条同内容的
  `pto.set_quant_scalar {id = 0}`
- 具体 `TPipe` 方向与数据路径由目标平台决定：
  - A2/A3：对齐现有 `pto-isa`，fixpipe 型 C2V `TPUSH` 走 GM FIFO 路径，
    Vector 侧 `TPOP` 再从 GM slot load 到本地 tile
  - A5：对齐现有 `pto-isa`，可映射到 no-split 的 C2V UB FIFO 路径；
    若后续扩展到 GM FIFO 方向，也应复用同一套 pipe-level `acc_push_epilogue`，
    只在 lowering / EmitC 侧区分目标 `TPipe` 形态

### 9.3 `tpop` EmitC

`tpop` 仍按当前 pipe entry 类型正常发射：

```cpp
TPOP<Pipe, TileCons, TileSplitAxis::TILE_NO_SPLIT>(pipe, tile);
```

它不需要生成额外 `TConfig`。

fixpipe 的生产语义由 producer `TPUSH` 和 pipe contract 保证，而不是由 `TPOP`
显式重复声明。

## 10. 验证规则

新增以下 verifier 规则：

1. 开启 `acc_push_epilogue` 的 frontend `initialize_pipe` 必须是单向 C2V pipe，因此
   `dir_mask` 必须等于 `1`。
2. 开启 `acc_push_epilogue` 时必须 `nosplit = true`，且所有绑定 data op 必须 `split = 0`。
3. 一条 C2V fixpipe logical pipe 的 producer `aic_initialize_pipe` 与 peer
   consumer `aiv_initialize_pipe` 必须同时显式携带 `acc_push_epilogue`，且
   `layout` / `quant` / `relu` 三字段逐项一致。
4. peer logical pipe 的配对不得仅依赖 module 级裸 `id`；第一版应复用
   `reserve_buffer` / `import_reserved_buffer` + `peer_func` 已建立的 peer key 来定位
   producer / consumer 对端 init。
5. 开启 `acc_push_epilogue` 的 C2V fixpipe pipe，其 `c2v_consumer_buf` 必须可追溯到
   `reserve_buffer` 或 `import_reserved_buffer`；否则 peer contract check 应直接报错。
6. 同一逻辑 pipe 上不允许出现多组 `acc_push_epilogue`。
7. `acc_push_epilogue.relu` 在 v1 中只允许 `no_relu` / `normal_relu`；若使用
   `scalar_relu` / `vector_relu` / `pwl`，前端 surface 必须直接拒绝。
8. `acc_push_epilogue.quant` 在 v1 中只允许：
   `no_convert` / `f32_f16` / `req8_scalar` / `req8_vec` /
   `deqf16_scalar` / `deqf16_vec` / `f32_bf16` /
   `qf322b8_pre_scalar` / `qf322b8_pre_vec` /
   `qf322f16_pre_scalar` / `qf322bf16_pre_scalar` /
   `qs322bf16_pre_scalar` / `qs322bf16_pre_vec` /
   `qf322hif8_pre_scalar` / `qf322fp8_pre_scalar`。
   其它现存 enum 值（包括 `*_hybrid_*`、`qf322f32_pre_*`、`*_s4_*`、`*_s16_*`，
   以及未纳入 v1 的其它 `*_vec` / `*_scalar` 变体）在 v1 中都必须由前端
   surface 直接拒绝；在当前 dedicated enum 实现下，这意味着 parse 阶段报错。
9. 开启 `acc_push_epilogue` 的 pipe，其 producer entry 必须是 `acc` tile。
10. `acc_push_epilogue.quant` 必须与 producer acc source element type 语义匹配：
   `f32_f16` / `f32_bf16` 要求 `SrcElemType = f32`；
   `qf322b8_pre_scalar` / `qf322b8_pre_vec` /
   `qf322f16_pre_scalar` / `qf322bf16_pre_scalar` /
   `qf322hif8_pre_scalar` / `qf322fp8_pre_scalar`
   要求 `SrcElemType = f32`；
   `req8_scalar` / `req8_vec` 要求 `SrcElemType = i32`；
   `deqf16_scalar` / `deqf16_vec` 要求 `SrcElemType = i32`；
   `qs322bf16_pre_scalar` / `qs322bf16_pre_vec` 要求 `SrcElemType = i32`；
   `no_convert` 要求 `SrcElemType in {f32, i32}`，且 consumer
   `DstElemType = SrcElemType`。另外，`qf322hif8_pre_scalar` /
   `qf322fp8_pre_scalar` 还要求当前 target/profile 支持 `!pto.hif8` /
   `f8E4M3FN` 作为 consumer destination dtype；`qs322bf16_pre_scalar` /
   `qs322bf16_pre_vec` 还要求当前 target/profile 坐实对应 A5 路径；否则 verifier
   必须报错。
11. 每一条开启 `acc_push_epilogue` 且被实际消费的 logical pipe，在 consumer 侧
   必须至少出现一次 `tpop`，以提供这条 pipe 的 resolved consumer result type。
   若出现多次 `tpop`，这些 `tpop` 的结果类型必须一致；第一版至少要求 result
   element type 与 layout 一致。对本文当前已纳入公开值域的 scalar 8-bit family，
   还应继续要求 signedness 一致；对 vector 8-bit family，还应继续要求结果
   element type 固定为 `si8`。
12. logical pipe 的 resolved consumer result element type 必须与
   `acc_push_epilogue.quant` 的前端类型规则一致，其中 `SrcElemType` 指 producer
   acc tile 的元素类型；对本文当前公开值域，可按前文表格理解为
   `FixpipeConsDType_t<acc_push_epilogue.quant, SrcElemType>::type` 的同值约束。
   对 `req8_scalar` / `qf322b8_pre_scalar`，这里还要额外要求 consumer result
   element type 只能是显式 `si8` 或 `ui8`，不得使用 signless `i8`；对
   `req8_vec` / `qf322b8_pre_vec`，这里还要额外要求 consumer result element
   type 只能是显式 `si8`，不得使用 `ui8` 或 signless `i8`。
13. `acc_push_epilogue.layout` 与 consumer result tile layout 必须一致。第一版可按
   tile type 的 layout config 参数校验：
   `nz2nd -> vec row_major`，
   `nz2dn -> vec col_major`，
   `nz2nz -> vec col_major + s_layout = row_major`。
14. fixpipe pipe 的 `slot_size` 必须与该 pipe 的 post-fixpipe consumer entry
    语义一致，而不是按 producer acc source tile 大小解释；第一版至少应满足
    `slot_size >= physical_size(consumer_shape, resolved_consumer_elem_type,
    resolved_consumer_layout)`，如有 target-specific 对齐要求，可在此基础上继续收紧。
15. 对普通 pipe，仍沿用既有 `slot_size = pre-split full logical entry bytes`
    语义；但对开启 `acc_push_epilogue` 的 fixpipe Acc-producer pipe，`slot_size`
    必须按 fixpipe 后的 consumer-visible entry 解释。
16. `deqf16_scalar` / `req8_scalar` / `qf322b8_pre_scalar` /
   `qf322f16_pre_scalar` / `qf322bf16_pre_scalar` / `qs322bf16_pre_scalar` /
   `qf322hif8_pre_scalar` / `qf322fp8_pre_scalar`
   这类 scalar quant 模式要求在对应 `TPUSH` 之前存在一条当前生效的
   `pto.set_quant_scalar {id = k}`；第一版要求二者位于同一 producer context、
   同一基本块内，且 `pto.set_quant_scalar.id` 必须匹配它所服务的 logical pipe `id`。
   对每一条这类 `TPUSH`，verifier 应解析同一基本块内、程序顺序上最近一条同类、
   同 id 的 `pto.set_quant_scalar` 作为它的 scalar quant binding；该 binding 会持续
   生效，直到被下一条同类、同 id 的 `pto.set_quant_scalar` 覆盖。若 `id = k`
   找不到同 function 内开启了 fixpipe scalar quant family 的 producer pipe，或该
   `TPUSH` 在当前基本块内不存在这样一条程序顺序上先于它的绑定，则 IR 非法。
17. `deqf16_vec` / `req8_vec` / `qf322b8_pre_vec` / `qs322bf16_pre_vec`
   这类 vector quant 模式要求在对应 `TPUSH` 之前存在一条当前生效的
   `pto.set_quant_vector {id = k}`；第一版要求二者位于同一 producer context、
   同一基本块内，且 `pto.set_quant_vector.id` 必须匹配它所服务的 logical pipe `id`。
   对每一条这类 `TPUSH`，verifier 应解析同一基本块内、程序顺序上最近一条同类、
   同 id 的 `pto.set_quant_vector` 作为它的 vector quant binding；该 binding 会持续
   生效，直到被下一条同类、同 id 的 `pto.set_quant_vector` 覆盖。若 `id = k`
   找不到同 function 内开启了 fixpipe vector quant family 的 producer pipe，或该
   `TPUSH` 在当前基本块内不存在这样一条程序顺序上先于它的绑定，则 IR 非法。
18. `pto.set_quant_scalar` / `pto.set_quant_vector` 都必须保留 side-effect /
    ordering 语义，不能作为 `Pure` 无结果 op 被 DCE / CSE / 任意重排；并且消费
    它们的 fixpipe `TPUSH` 也必须显式参与同一份 quant-state 依赖。实现上可选择：
    使用 `MemoryEffectsOpInterface` + 专门的 producer quant-state resource，并让
    fixpipe `TPUSH` 对同一 resource 声明 Read / Write effect；若采用 resource
    方案，这份 quant-state resource 至少应按 logical pipe `id` 可区分；或提供
    等价的 ordering token 机制。
19. 同一 kernel 可以存在多条带不同 `acc_push_epilogue` 的 fixpipe pipe；不同 `id`
   的 `pto.set_quant_*` 可以在同一基本块内交错出现。同一条
   `pto.set_quant_* {id = k}` 可以合法地服务于后续多条同 id、family 匹配的
   `TPUSH`，直到被下一条同类、同 id 的 `pto.set_quant_*` 覆盖。实现 / verifier
   应按“每条 `TPUSH` 读取最近一次生效绑定”来解释这类交错写法，而不是按
   “单次消费”解释。
20. scalar quant lowering 到 `SET_QUANT_SCALAR<OutType>` 时，`OutType` 应来自当前
   `pto.set_quant_scalar.id` 所绑定 logical pipe 的 resolved consumer result
   element type；这份类型由 peer consumer `tpop` 结果 dtype 提供，并要求同一条
   pipe 上的所有 `tpop` 结果类型一致。对本文当前公开的 scalar quant 子集：
   `deqf16_scalar` / `qf322f16_pre_scalar -> half`，
   `qf322bf16_pre_scalar` / `qs322bf16_pre_scalar -> bfloat16_t`，
   `qf322hif8_pre_scalar -> hifloat8_t`，
   `qf322fp8_pre_scalar -> float8_e4m3_t`；
   对 `req8_scalar` / `qf322b8_pre_scalar`，`si8 -> int8_t`，
   `ui8 -> uint8_t`，不得从 signless `i8` 继续推导。单条
   `pto.set_quant_scalar {id = k}` 在 codegen 中可以对应多次
   `SET_QUANT_SCALAR<OutType>` 发射；至少当执行顺序从别的 pipe 切回 `id = k` 时，
   lowering 必须重新发射一次以恢复该 pipe 的量化 machine state。
21. peer logical pipe 一致性检查必须在 frontend `initialize_pipe` 仍可见时完成首轮
    contract verify；同时 internal pipe lowering 后仍应保留 `acc_push_epilogue`
    metadata，供后续 EmitC / sanity check 使用。
22. peer logical pipe 一致性检查必须确认：
   producer 侧按 `SrcElemType + acc_push_epilogue.quant` 推导出的 logical pipe
   consumer entry type，与 consumer 侧 resolved `tpop` 结果类型一致。
23. A5 no-split C2V UB fixpipe 的 v1 范围内，还应额外收紧为：
    `SubBlockId = 0`，即仅覆盖 `AccToVecMode::SingleModeVec0`。`SingleModeVec1`
    及更复杂的 dual-vector 目标选择不属于当前前端 contract。
24. `pto.set_quant_vector` 的输入必须是 `loc=scaling` 的 tile；其 element type
    还必须属于 `f16` / `bf16` / `f32` family。其大小与更细的 layout 约束仍应满足
    目标平台 fixpipe vector quant payload 的参数布局要求。
    某些 target 当前可能以 `uint64_t` scaling tile 访问 packed scaling buffer，
    但这只应视为 target-specific lowering / payload 访问形式，不应上升为统一的
    前端公共语义类型约束。
25. 对 `pto.set_quant_vector`，第一版公共 verifier 至少检查 `loc=scaling` 与
    `f16` / `bf16` / `f32` element type family；其余 payload shape / layout 约束，
    可由 arch-specific verifier 按目标平台现有 fixpipe vector quant 规则继续收紧。
26. 第一版 PTOIR 只允许显式出现 `acc_push_epilogue` 这一个前端公开复合 attr，
    其内部字段限定为 `layout`、`quant`、`relu`。
27. `layout` / `quant` lowering 到 `FixpipeParams` 时必须使用显式符号映射，不得
    依赖前端 enum ordinal 与 `pto-isa` enum ordinal 一致。
28. `STPhase`、`AtomicType`、`SubBlockId`、`ClipReluMode_t`、`IsChannelSplit`
    当前阶段暂不作为 PTOIR 前端公开配置项，也不允许以 attr 形式半暴露半忽略。
29. 这些当前未公开字段在 v1 EmitC 中应按 `FixpipeParams` 默认模板实参处理，而不是
    留成未定义行为。

## 11. 测试建议

至少补以下测试：

- `DEQF16` + `NZ2ND` + `NormalRelu`
- `VDEQF16` + `NZ2ND` + `NoRelu`
- `NoQuant` + `NZ2ND`
- `NoQuant` + `f32 -> f32`
- `NoQuant` + `i32 -> i32`
- `DEQF16` + `pto.set_quant_scalar`
- `REQ8` + `pto.set_quant_scalar`
- `REQ8 scalar` + `ui8` consumer destination type 的正向 case
- `QF322B8_PRE` + `pto.set_quant_scalar`
- `QF322F16_PRE` + `pto.set_quant_scalar`
- `QF322BF16_PRE` + `pto.set_quant_scalar`
- `QS322BF16_PRE` + `pto.set_quant_scalar` 的 A5 正向 case
- `QF322HIF8_PRE` + `pto.set_quant_scalar` 的 A5 正向 case
- `QF322FP8_PRE` + `pto.set_quant_scalar` 的 A5 正向 case
- `VDEQF16` + `pto.set_quant_vector`
- `REQ8` + `pto.set_quant_vector`
- `QF322B8_PRE` + `pto.set_quant_vector`
- `VREQ8` / `VQF322B8_PRE` + `si8` consumer destination type 的正向 case
- `VQS322BF16_PRE` + `pto.set_quant_vector` 的 A5 正向 case
- `pto.set_quant_scalar {id = 0}` / `pto.tpush_to_aiv {id = 0}` 成功配对的正向 case
- `pto.set_quant_vector {id = 1}` / `pto.tpush_to_aiv {id = 1}` 成功配对的正向 case
- 单条 `pto.set_quant_scalar {id = 0}` 被同一基本块内两条 `id = 0` scalar quant
  `TPUSH` 复用的正向 case
- 单条 `pto.set_quant_vector {id = 1}` 被同一基本块内两条 `id = 1` vector quant
  `TPUSH` 复用的正向 case
- 同一 kernel 中两条 fixpipe pipe 使用不同 `acc_push_epilogue` 的正向 case
- 同一 kernel 中 scalar quant pipe 与 vector quant pipe 并存、各自配对
  `pto.set_quant_*` 的正向 case
- 同一基本块内不同 `id` 的 `pto.set_quant_*` 与 `TPUSH` 交错出现、仍可各自正确配对
  的正向 case
- `pto.set_quant_scalar {id = 0}`、`pto.set_quant_scalar {id = 1}` 与
  `TPUSH{id = 0} / TPUSH{id = 1} / TPUSH{id = 0}` 交错出现，并在 codegen 中自动
  rematerialize `SET_QUANT_SCALAR` 的正向 / codegen case
- 同一 `id` 上后一条 `pto.set_quant_scalar` 覆盖前一条绑定、后续 `TPUSH` 读取新
  payload 的正向 case
- producer / consumer 两侧镜像 `acc_push_epilogue` 一致的正向 case
- 基于 `reserve_buffer` / `import_reserved_buffer` + `peer_func` 正确找到 peer
  logical pipe 的正向 case
- 开启 `acc_push_epilogue` 时，`c2v_consumer_buf` 无法追溯到
  `reserve_buffer` / `import_reserved_buffer` 的 verifier 失败 case
- `layout = nz2nd` / `nz2dn` / `nz2nz` 分别 lowering 到
  `LayoutMode_t::NZ2ND` / `NZ2DN` / `NZ2NZ` 的 codegen case
- `quant = no_convert` / `f32_f16` / `req8_scalar` / `req8_vec` /
  `deqf16_scalar` / `deqf16_vec` / `f32_bf16` /
  `qf322b8_pre_scalar` / `qf322b8_pre_vec` /
  `qf322f16_pre_scalar` / `qf322bf16_pre_scalar` /
  `qs322bf16_pre_scalar` / `qs322bf16_pre_vec` /
  `qf322hif8_pre_scalar` / `qf322fp8_pre_scalar`
  分别 lowering 到对应 `QuantMode_t::*` 的 codegen case
- `DEQF16` / `REQ8` / `QF322B8_PRE` / `QF322F16_PRE` / `QF322BF16_PRE` /
  `QS322BF16_PRE` / `QF322HIF8_PRE` / `QF322FP8_PRE` 对应
  `pto.set_quant_scalar` 缺失时的 verifier 失败 case
- `VDEQF16` / `VREQ8` / `VQF322B8_PRE` / `VQS322BF16_PRE` 对应
  `pto.set_quant_vector` 缺失时的 verifier 失败 case
- `pto.set_quant_scalar {id = 0}` / `pto.set_quant_vector {id = 0}` 命中错误 quant
  family pipe 的 verifier 失败 case
- `pto.set_quant_scalar {id = 0}` 指向不存在的 local producer pipe 的 verifier
  失败 case
- `pto.set_quant_vector {id = 1}` 指向不存在的 local producer pipe 的 verifier
  失败 case
- `pto.set_quant_scalar` 输入不是 `f32` 的 verifier 失败 case
- `pto.set_quant_scalar.id` 与实际消费它的 `TPUSH.id` 不一致的 verifier 失败 case
- `pto.set_quant_vector.id` 与实际消费它的 `TPUSH.id` 不一致的 verifier 失败 case
- scalar quant `TPUSH` 在当前基本块内先于任何 `pto.set_quant_scalar {id = k}`
  出现的 verifier 失败 case
- vector quant `TPUSH` 在当前基本块内先于任何 `pto.set_quant_vector {id = k}`
  出现的 verifier 失败 case
- `pto.set_quant_*` 与其消费它的 fixpipe `TPUSH` 不在同一基本块内的 verifier
  失败 case
- producer / consumer 两侧 `acc_push_epilogue` 不一致的 verifier 失败 case
- 同一 module 中存在多组 cube/vector function、且各自都有 `id = 0` 时，
  仅靠 `id` 会歧义，但按 peer buffer key 仍能正确配对的 case
- 使用 `scalar_relu` / `vector_relu` / `pwl` 的 parse-surface 失败 case
- 使用 v1 未公开的 `quant` 枚举值（如 `qf322fp8_pre_vec`、
  `qf322hif8_pre_vec`、`qf322f16_pre_vec`、`qf322bf16_pre_vec`、`qf322f32_pre_*`）
  的 parse-surface 失败 case
- `no_convert + acc<f16>` 的 verifier 失败 case
- `no_convert + acc<i16>` 的 verifier 失败 case
- `f32_f16 + i32 acc` 的 verifier 失败 case
- `req8_scalar + f32 acc` 的 verifier 失败 case
- `req8_vec + f32 acc` 的 verifier 失败 case
- `f32_bf16 + i32 acc` 的 verifier 失败 case
- `deqf16_scalar + f32 acc` 的 verifier 失败 case
- `deqf16_vec + f32 acc` 的 verifier 失败 case
- `qf322b8_pre_scalar + i32 acc` 的 verifier 失败 case
- `qf322b8_pre_vec + i32 acc` 的 verifier 失败 case
- `qf322f16_pre_scalar + i32 acc` 的 verifier 失败 case
- `qf322bf16_pre_scalar + i32 acc` 的 verifier 失败 case
- `qs322bf16_pre_scalar + f32 acc` 的 verifier 失败 case
- `qs322bf16_pre_vec + f32 acc` 的 verifier 失败 case
- `qf322hif8_pre_scalar + i32 acc` 的 verifier 失败 case
- `qf322fp8_pre_scalar + i32 acc` 的 verifier 失败 case
- `qf322hif8_pre_scalar` / `qf322fp8_pre_scalar` 在不支持低精度目标 dtype 的
  target/profile 上的 verifier 失败 case
- `qs322bf16_pre_scalar` / `qs322bf16_pre_vec` 在非 A5 target/profile 上的 verifier
  失败 case
- producer 侧推导出的 logical pipe consumer type 与 consumer `tpop` 结果类型不一致
  的 peer-contract 失败 case
- `req8_*` / `qf322b8_pre_*` 使用 signless `i8` 作为 consumer `tpop` result dtype
  的 verifier 失败 case
- `req8_scalar` / `qf322b8_pre_scalar` 在同一 logical pipe 上混用 `si8` 与 `ui8`
  consumer `tpop` 结果类型的 verifier 失败 case
- `req8_vec` / `qf322b8_pre_vec` 使用 `ui8` 作为 consumer `tpop` result dtype 的
  verifier 失败 case
- 开启 `acc_push_epilogue` 的 logical pipe 在 consumer 侧没有任何 `tpop` 的
  verifier 失败 case
- 同一 logical pipe 上两个 `tpop` 返回不同结果 dtype / layout 的 verifier 失败
  case
- `pto.set_quant_vector` 输入不是 `loc=scaling` tile 的 verifier 失败 case
- `pto.set_quant_vector` 使用非法 payload element type 的 verifier 失败 case
- 不同 `acc_push_epilogue.quant` 与 consumer type 不匹配的 verifier 失败 case
- fixpipe pipe 与 `split = 1/2` 混用的 verifier 失败 case
- 用户尝试用一条 `dir_mask = 3` pipe 同时承载 `C2V fixpipe + V2C normal` 的
  verifier 失败 case
- 同一 pipe 上混入两组不同 `acc_push_epilogue` 的 verifier 失败 case
- `slot_size` 与 post-fixpipe consumer entry 语义不一致的 verifier / codegen case
- `pto.set_quant_*` 已声明 quant-state effect、但匹配的 fixpipe `TPUSH` 未声明同一
  resource / token 依赖时的防回归 case
- `pto.set_quant_*` 的 quant-state resource 未按 `id` 区分、导致不同 pipe 互相污染的
  防回归 case
- `pto.set_quant_*` 被当成普通无结果 op 删除 / 合并 / 重排的防回归 case
