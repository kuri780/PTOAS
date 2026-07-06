# PTOAS 内存一致性设计

本文说明 PTOAS 如何建模并校验 GM payload 与 signal 之间的内存一致性要求。

这里讨论的是内存一致性，不是自动同步。自动同步负责 pipe 之间的执行顺序，例如
`set_flag`、`wait_flag` 和 `pipe_barrier`。内存一致性负责回答另一个问题：当 signal
已经被对端观察到时，signal 之前发布的 payload 是否已经对正确的观察方可见。

## 1. 背景

`pto.comm.tnotify` 用来发布一个 signal。对端通过 `pto.comm.twait` 或
`pto.comm.ttest` 观察这个 signal，然后读取对应的 payload。

signal ready 不等价于 payload 一定已经可见。原因是 signal 和 payload 可能走不同的硬件路径：

- signal 通常是一个较小的通信同步标记。
- payload 通常是更大的 GM 数据，可能由 MTE3、FIX 或 comm macro op 写出。
- 不同路径之间只靠源码顺序不一定形成完整的 GM 可见性关系。

因此，PTOAS 需要在发布 signal 前校验 release 侧动作，在消费 signal 后校验 acquire 侧动作。

## 2. 当前暴露的 IR

当前 PR 只暴露两条内存一致性 IR：

| PTO IR | 语义 | EmitC lowering |
| --- | --- | --- |
| `pto.fence.barrier_all #pto.fence_scope<gm>` | GM visibility barrier，可用于 publish 或 acquire 边界 | `dsb(DSB_DDR)` |
| `pto.cmo.cacheinvalid all #pto.address_space<gm>` | 显式 GM cache maintenance 边界，可用于 release 或 acquire | `dcci((__gm__ void*)0, cache_line_t::ENTIRE_DATA_CACHE)` |

对应 PyPTO 写法：

```python
pto.FenceBarrierAllOp(pto.FenceScope.GM)
pto.CmoCacheInvalidOp(pto.AddressSpace.GM)
```

`FenceScopeAttr` 当前定义三个语义 scope：

- `local_memory`
- `gm`
- `all`

本 PR 只为 `gm` 和 `all` 提供 EmitC lowering；它们当前都 lower 成 `dsb(DSB_DDR)`。
`local_memory` 预留给后续 UB/local-memory 语义。

`cmo.cacheinvalid` 的 `all` 是当前 CMO op assembly format 的固定关键字，表示 whole-cache
粒度。第一阶段不传 GM 地址和范围。后续如果要支持精确 payload 范围，可以扩展成：

```mlir
pto.cmo.cacheinvalid %partition_view
```

再由 backend 根据 ABI 选择 `SINGLE_CACHE_LINE` 或循环覆盖 range。

## 3. DCCI 语义边界

`dcci` 是 cache maintenance 指令，不是 pipe drain，也不是 GM visibility fence。

PTOAS 当前对齐 PTO-ISA 的写法，只生成 two-argument CCE builtin：

```cpp
dcci(addr, cache_line);
```

其中 `cache_line_t::SINGLE_CACHE_LINE` 表示处理传入地址所在 cache line，
`cache_line_t::ENTIRE_DATA_CACHE` 表示处理整个 Data Cache。

公开 AscendC API 和部分底层 CCE 头文件还存在带 destination 参数的形式，例如
`CACHELINE_OUT`。但是 PTO-ISA 主线的 `TNotify`、`TWait`、`TTest` 和 ready-queue
实现都使用 two-argument `dcci`。因此本 PR 不把三参数 `dcci(..., CACHELINE_OUT)`
作为 PTOAS 的生成代码契约，也不在 PTO IR 中暴露 destination 选择。

当前 public IR `pto.cmo.cacheinvalid all #pto.address_space<gm>` 使用 two-argument
whole-cache 形式：

```cpp
dcci((__gm__ void*)0, cache_line_t::ENTIRE_DATA_CACHE);
```

它是 PTOAS 暴露给上游的显式 CMO 边界。虽然 op 名称沿用 `cacheinvalid`，但当前
PTO-ISA 对齐的 two-argument `dcci` 同时服务两类上下文：

- release 侧：放在 cacheable scalar GM store 之后、`fence.barrier_all` 之前，作为发布
  payload 前的 cache maintenance。
- acquire 侧：放在 `TWait` 或 `TTest` 之后、cacheable GM load 之前，避免读取本地 stale
  cache line。

本 PR 不新增 `pto.cmo.clean`。原因是当前 release 侧和 acquire 侧都对齐到 PTO-ISA
two-argument `dcci`，新增一个 lowering 相同的 public op 会让 PyPTO 和用户难以区分。
后续如果确认需要暴露不同 destination 或精确 writeback 语义，再单独扩展 CMO IR。

## 4. MemoryConsistency Pass

`pto-memory-consistency` 是一个 Module pass，运行在 shared mainline 上，因此 EmitC 和
VPTO backend 都会先经过这一步。

这个 pass 的职责是：

- 识别 signal publish 前是否存在 pending GM payload write。
- 识别 signal acquire 后是否存在 cacheable GM payload read。
- 校验用户或 PyPTO 是否已经插入必要的 `fence.barrier_all` 或 `cmo.cacheinvalid`。
- 在显式 `barrier_all` 前自动补齐必要的 MTE3 或 FIX pipe drain。
- 对缺失或顺序错误的场景报编译错误。

这个 pass 不负责分配 event id，也不属于 InsertSync 自动同步流水线。

## 5. 场景规则

### 5.1 MTE3、FIX 或 TPUT 写 Payload 后发布 Signal

适用场景：

- `TStore` 通过 `PIPE_MTE3` 写 GM。
- `TStore` 通过 `PIPE_FIX` 写 GM，例如 ACC tile 写回 GM。
- `TStoreFP` 通过 `PIPE_FIX` 写 GM。
- `TPUT` macro op 内部通过 MTE3 写 peer GM。
- 其他 macro op phase 中存在 MTE3 GM write。

PyPTO 或用户需要生成：

```mlir
// payload producer
pto.fence.barrier_all #pto.fence_scope<gm>
pto.comm.tnotify ...
```

PTOAS 会在 `pto.fence.barrier_all #pto.fence_scope<gm>` 前检查是否存在 pending MTE3 或
FIX GM write；如果存在，自动插入对应 pipe 的 drain：

```mlir
pto.barrier #pto.pipe<PIPE_MTE3>
// or
pto.barrier #pto.pipe<PIPE_FIX>
```

最终 EmitC 关键顺序是：

```cpp
pipe_barrier(PIPE_MTE3);
dsb(DSB_DDR);
pto::comm::TNOTIFY(...);
```

`pipe_barrier` 用来排空实际执行 GM write 的 pipe。`barrier_all` lower 出来的
`dsb(DSB_DDR)` 用来保证这些 GM 写入在 signal 发布前进入 GM visibility domain。

如果缺少 `pto.fence.barrier_all #pto.fence_scope<gm>`，PTOAS 会报错。PTOAS 可以推导
低层 pipe drain，但不会替用户猜测 payload publish 的语义边界。

### 5.2 MTE2 工作后发布 Signal

`TLoad` 或其他 `PIPE_MTE2` 工作出现在 `TNotify` 前时，当前仍允许 PTOAS 自动标注并在
EmitC lowering 中生成 `PIPE_MTE2` barrier。MTE2 是 GM read 方向，只需要 signal 前不要越过
前序 MTE2 工作，不需要 `barrier_all`。

### 5.3 TWait 或 TTest 后读取 Cacheable GM Payload

适用场景：

- `TWait` 返回后执行 `load_scalar` 读取 GM payload。
- `TTest` 成功观察到 signal 后执行 `load_scalar` 读取 GM payload。

需要的顺序：

```mlir
pto.comm.twait ...
pto.cmo.cacheinvalid all #pto.address_space<gm>
%value = pto.load_scalar ...
```

`cacheinvalid` 用来避免读取到本地 stale cache line。如果缺少该 op，PTOAS 会报错。

### 5.4 Cacheable Scalar GM Store 后发布 Signal

适用场景：

```mlir
pto.store_scalar %value, %payload[%idx] : !pto.ptr<i32>, i32
pto.cmo.cacheinvalid all #pto.address_space<gm>
pto.fence.barrier_all #pto.fence_scope<gm>
pto.comm.tnotify ...
```

最终 EmitC 关键顺序是：

```cpp
payload[idx] = value;
dcci((__gm__ void*)0, cache_line_t::ENTIRE_DATA_CACHE);
dsb(DSB_DDR);
pto::comm::TNOTIFY(...);
```

这里 `cmo.cacheinvalid` 表示 publish 前的显式 CMO 边界，`fence.barrier_all` 表示
GM visibility fence。二者顺序不能交换。如果缺少 `cmo.cacheinvalid`，或者把它放在
`fence.barrier_all` 后面，PTOAS 会报错。

## 6. PyPTO 对接说明

PyPTO 不需要手动生成 `pto.barrier #pto.pipe<PIPE_MTE3>` 或
`pto.barrier #pto.pipe<PIPE_FIX>`。这是低层 pipe drain 细节，由 PTOAS 根据 `barrier_all`
前的 pending GM write pipe 自动插入。这样可以保证最终顺序是对应 pipe barrier 先于
`dsb(DSB_DDR)`，不会出现先 fence、后 drain 的错误顺序。

PyPTO 生成规则：

| 场景 | PyPTO 需要生成 | PTOAS 自动补齐 |
| --- | --- | --- |
| `TStore`、`TStoreFP` 或 `TPUT` 后发布 signal | `pto.fence.barrier_all #pto.fence_scope<gm>` | `PIPE_MTE3` 或 `PIPE_FIX` drain |
| `TLoad` 后发布 signal | 不需要显式 fence | `PIPE_MTE2` drain |
| `TWait` 后读取 cacheable scalar GM payload | `pto.cmo.cacheinvalid all #pto.address_space<gm>` | 无 |
| `TTest` ready path 后读取 cacheable scalar GM payload | 在 ready path 的 payload load 前生成 `pto.cmo.cacheinvalid all #pto.address_space<gm>` | 无 |
| cacheable scalar GM store 后发布 signal | `pto.cmo.cacheinvalid all #pto.address_space<gm>`，随后 `pto.fence.barrier_all #pto.fence_scope<gm>` | 无 |

`pto.entry` launcher 可以调用多个 kernel 函数；每个 kernel 函数会被
`pto-memory-consistency` 独立分析。kernel body 内部若通过 `func.call` 调用包含 payload
访问、CMO、fence 或 signal op 的 helper，PyPTO 应在 `pto-memory-consistency` 前将 helper
inline，或者把 payload、CMO、fence 和 signal 保持在同一个 caller 中。否则 pass 会报错，
避免 caller 侧 `TNotify` 或 `TWait` 看不到 callee 内部的 memory-consistency 状态。

### 6.1 Issue #872：TPUT 发布 Signal

```mlir
pto.comm.tput ...
pto.fence.barrier_all #pto.fence_scope<gm>
pto.comm.tnotify ...
```

对应 PyPTO 写法：

```python
pto.TPutOp(...)
pto.FenceBarrierAllOp(pto.FenceScope.GM)
pto.TNotifyOp(...)
```

这个形态对应 #872 中的 `TPUT -> TNotify` 问题。`TPUT` macro op 内部会通过 MTE3 写
peer GM payload；如果直接发布 `TNotify`，receiver 可能先观察到 signal ready，但 payload
写入还没有完成 pipe drain 或进入 GM visibility domain。

PTOAS 会识别 `TPUT` macro model 中的 MTE3 GM write phase，并在 `barrier_all` 前自动补齐
低层 pipe drain：

```mlir
pto.comm.tput ...
pto.barrier #pto.pipe<PIPE_MTE3>
pto.fence.barrier_all #pto.fence_scope<gm>
pto.comm.tnotify ...
```

EmitC 最终生成的关键顺序是：

```cpp
TPUT(...);
pipe_barrier(PIPE_MTE3);
dsb(DSB_DDR);
TNOTIFY(...);
```

### 6.2 TWait 后读取 Scalar Payload

```mlir
pto.comm.twait ...
pto.cmo.cacheinvalid all #pto.address_space<gm>
%value = pto.load_scalar ...
```

对应 PyPTO 写法：

```python
pto.TWaitOp(...)
pto.CmoCacheInvalidOp(pto.AddressSpace.GM)
pto.LoadScalarOp(...)
```

### 6.3 Scalar Store 发布 Signal

```mlir
pto.store_scalar %value, %payload[%idx] : !pto.ptr<i32>, i32
pto.cmo.cacheinvalid all #pto.address_space<gm>
pto.fence.barrier_all #pto.fence_scope<gm>
pto.comm.tnotify ...
```

对应 PyPTO 写法：

```python
pto.StoreScalarOp(...)
pto.CmoCacheInvalidOp(pto.AddressSpace.GM)
pto.FenceBarrierAllOp(pto.FenceScope.GM)
pto.TNotifyOp(...)
```

这对应 cacheable scalar GM store 发布 payload 的场景。`CmoCacheInvalidOp` 和
`FenceBarrierAllOp` 都是必需的，且顺序必须是 CMO 在 fence 之前。

## 7. Backend Lowering 状态

### 7.1 EmitC

EmitC backend 当前支持：

- `pto.cmo.cacheinvalid` lower 到 `dcci((__gm__ void*)0, cache_line_t::ENTIRE_DATA_CACHE)`。
- `pto.fence.barrier_all #pto.fence_scope<gm>` lower 到 `dsb(DSB_DDR)`。

### 7.2 VPTO

VPTO backend 当前没有确认的 DSB 和 DCCI intrinsic ABI。

因此，VPTO lowering 中现在提供的是 fail-fast stub：

- `pto.cmo.cacheinvalid`
- `pto.fence.barrier_all`

如果这些 op 进入 VPTO LLVM lowering，PTOAS 会报错，提示 VPTO backend 尚不支持这些
memory-consistency op，需要确认 DSB/DCCI intrinsic ABI 后再接真实 lowering。

## 8. 当前限制

- `cmo.cacheinvalid` 是 whole-cache 粒度，不是精确地址范围。
- `TWait` 和 `TTest` acquire 侧当前只覆盖 `load_scalar`。
- VPTO 暂不支持 CMO 和 GM fence 的真实 lowering。
- 对复杂 CFG 的分析仍是保守近似，不做完整 path-sensitive 数据流。

## 9. 后续工作

1. 和 VPTO/Bisheng 对齐 DSB 和 DCCI intrinsic ABI，并补齐 VPTO lowering。
2. 将 whole-cache `cacheinvalid` 优化成精确 GM address range CMO。
3. 如果后续确认 release writeback 需要不同 destination 或更精确语义，再决定是否引入新的 CMO op。
4. 扩展 acquire 侧 consumer 范围，从 `load_scalar` 扩展到更多 cacheable GM read。
5. 将 macro op phase 的 memory descriptor 做得更精细，减少误报。
