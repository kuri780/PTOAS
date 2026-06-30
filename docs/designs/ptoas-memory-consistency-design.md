# PTOAS 内存一致性设计

本文说明 PTOAS 如何建模并校验 GM payload 与 signal 之间的内存一致性要求。

这里讨论的是内存一致性，不是自动同步。自动同步负责 pipe 之间的执行顺序，例如
`set_flag`、`wait_flag` 和 `pipe_barrier`。内存一致性负责回答另一个问题：当 signal
已经被对端观察到时，signal 之前发布的 payload 是否已经对正确的观察方可见。

## 1. 背景

`pto.comm.tnotify` 用来发布一个 signal。对端通过 `pto.comm.twait` 或
`pto.comm.ttest` 观察这个 signal，然后读取对应的 payload。

一个容易误解的点是：signal ready 不等价于 payload 一定已经可见。原因是 signal
和 payload 可能走不同的硬件路径：

- signal 通常是一个较小的通信同步标记。
- payload 通常是更大的 GM 数据，可能由 MTE3、TPUT 或 cacheable scalar store 写出。
- 不同路径之间只靠源码顺序不一定形成完整的可见性关系。

因此，PTOAS 需要在发布 signal 前校验 release 侧动作，在消费 signal 后校验
acquire 侧动作。

## 2. 关键概念

### 2.1 Payload

payload 是真正要被对端或后续代码读取的数据。例如：

- `TStore` 写出的 GM 数据。
- `TPUT` 内部写出的 peer GM 数据。
- `store_scalar` 写出的 GM 数据。

### 2.2 Signal

signal 是通知对端 payload 已经准备好的标记。例如：

- `TNotify` 发布 signal。
- `TWait` 等待 signal。
- `TTest` 轮询 signal 是否 ready。

signal 只表达“通知发生了”。如果 signal 前没有正确的 release 动作，signal 可能先被
对端观察到，而 payload 仍然没有进入对端能够正确读取的可见性状态。

### 2.3 Pipe drain

pipe drain 用来保证某条 pipe 上已经发出的工作完成到该 pipe 的边界。典型指令是：

```mlir
pto.barrier #pto.pipe<PIPE_MTE3>
```

它解决的是 pipe 内工作排空问题。它不等价于 cache clean，也不等价于 DDR-domain
visibility fence。

### 2.4 Cache maintenance operation

cache maintenance operation 用来处理 cacheable GM 访问造成的 cache line 状态。
当前 PTOAS 暴露两个语义 op：

```mlir
pto.cmo.clean all #pto.address_space<gm>
pto.cmo.invalidate all #pto.address_space<gm>
```

第一阶段采用 whole-cache 形式。也就是说，它不指定精确地址范围，而是对整个 GM
相关 data cache 做保守处理。这样优先保证正确性，后续再优化成精确 range。

### 2.5 DDR fence

DDR fence 用来把已经完成的 GM 写入或 cache maintenance 操作推进到 DDR visibility
domain，并约束它们发生在后续 signal publish 之前。当前 PTOAS 暴露两个语义 op：

```mlir
pto.fence.release #pto.fence_scope<ddr>
pto.fence.acquire #pto.fence_scope<ddr>
```

当前 release 和 acquire 都使用同一个 `ddr` scope。语义上，release 侧用于发布
payload，acquire 侧用于约束观察 signal 后的 payload 读取。

## 3. 整体模型

生产端的正确顺序是：

```mermaid
flowchart LR
  A["payload write"] --> B["pipe drain or cache clean"]
  B --> C["DDR release fence"]
  C --> D["TNotify publishes signal"]
```

消费端的正确顺序是：

```mermaid
flowchart LR
  A["TWait or successful TTest observes signal"] --> B["cache invalidate if needed"]
  B --> C["payload read"]
```

这两个方向配合起来，才能保证 signal 和 payload 的顺序关系对观察方成立。

## 4. 显式 IR 接口

PTOAS 选择把 cache maintenance 和 DDR fence 暴露成显式 PTO IR，而不是在 lowering
阶段偷偷插入 `dcci` 和 `dsb`。

原因如下：

- 这类动作有实际运行时成本，尤其 whole-cache CMO 成本较高。
- 用户或 PyPTO 更清楚 payload 的发布边界。
- PTOAS 可以负责校验契约，避免漏插或乱序，而不是猜测所有场景。
- VPTO 后端当前还没有确认的 DSB 和 DCCI intrinsic ABI，显式 IR 可以先稳定上层契约。

当前新增的语义 op 是：

| PTO IR | 语义 | EmitC lowering |
| --- | --- | --- |
| `pto.cmo.clean all #pto.address_space<gm>` | 清理 GM 相关 dirty cache line | `dcci((__gm__ void*)0, ENTIRE_DATA_CACHE, CACHELINE_OUT)` |
| `pto.cmo.invalidate all #pto.address_space<gm>` | 失效 GM 相关 stale cache line | `dcci((__gm__ void*)0, ENTIRE_DATA_CACHE)` |
| `pto.fence.release #pto.fence_scope<ddr>` | release 侧 DDR visibility fence | `dsb(DSB_DDR)` |
| `pto.fence.acquire #pto.fence_scope<ddr>` | acquire 侧 DDR visibility fence | `dsb(DSB_DDR)` |

## 5. MemoryConsistency pass

`pto-memory-consistency` 是一个 Module pass，运行在 shared mainline 上，因此 EmitC 和
VPTO backend 都会先经过这一步。

这个 pass 的职责是校验显式契约：

- 识别 signal publish 前是否存在 pending payload write。
- 识别 signal acquire 后是否存在 cacheable GM payload read。
- 校验用户或 PyPTO 是否已经插入必要的 CMO 和 fence。
- 在显式 release fence 前自动补齐必要的 MTE3 或 FIX pipe drain。
- 对缺失或顺序错误的场景报编译错误。
- 对不需要 `dcci` 和 `dsb` 的纯 pipe drain 场景，仍允许保留自动标注。

遍历策略是 region-scoped 的保守分析：单 block region 按顺序递归分析；复杂 CFG
region 暂不做 path-sensitive 数据流，但只在当前 region 内收集 pending state，不把同一个
parent op 的其他 sibling region 状态混入。外部函数声明没有函数体，pass 会直接跳过。

这个 pass 不负责分配 event id，也不属于 InsertSync 自动同步流水线。

## 6. 场景规则

### 6.1 MTE3、FIX 或 TPUT 写 payload 后发布 signal

适用场景：

- `TStore` 通过 `PIPE_MTE3` 写 GM。
- `TStore` 通过 `PIPE_FIX` 写 GM，例如 ACC tile 写回 GM。
- `TStoreFP` 通过 `PIPE_FIX` 写 GM。
- `TPUT` macro op 内部通过 MTE3 写 peer GM。
- 其他 macro op phase 中存在 MTE3 GM write。

需要的顺序：

```mlir
// payload producer
pto.fence.release #pto.fence_scope<ddr>
pto.comm.tnotify ...
```

PyPTO 或用户只需要表达 `pto.fence.release` 这个内存一致性边界。PTOAS 会在
`pto.fence.release #pto.fence_scope<ddr>` 前检查是否存在 pending MTE3 或 FIX GM write；如果存在，
自动插入对应 pipe 的 drain：

```mlir
pto.barrier #pto.pipe<PIPE_MTE3>
// or
pto.barrier #pto.pipe<PIPE_FIX>
```

最终 lowering 的顺序是：

```cpp
pipe_barrier(PIPE_MTE3);
dsb(DSB_DDR);
pto::comm::TNOTIFY(...);
```

`pipe_barrier(PIPE_MTE3)` 或 `pipe_barrier(PIPE_FIX)` 用来排空实际执行 GM write 的
pipe。`pto.fence.release` lower 出来的 `dsb(DSB_DDR)` 用来保证这些 GM 写入在 signal
发布前进入 DDR visibility domain。

这里不能把所有 `PIPE_FIX` op 都当成 release payload write。很多 FIX op 只是本地
ACC 到 MAT 或 ACC 到 VEC 的搬运，不需要 DDR release。PTOAS 只对确认写 GM payload 的
FIX 路径补 release drain。

同理，也不能把所有 `PIPE_MTE3` op 都当成 release payload write。例如 A5 的
Vec 到 Mat `TInsert` 是本地 UB 到 L1 的搬运，不发布 GM payload。PTOAS 只对
`TStore`、comm macro MTE3 phase 等确认写 GM payload 的路径补 release drain。

如果缺少 `pto.fence.release`，PTOAS 会报错。因为 PTOAS 可以推导 pipe drain，但不会凭空
猜测 payload publish 的语义边界。

### 6.2 MTE2 工作后发布 signal

适用场景：

- `TLoad` 或其他 `PIPE_MTE2` 工作出现在 `TNotify` 之前。

当前规则：

```mlir
// PTOAS 可以自动标注并在 EmitC lowering 中生成 PIPE_MTE2 barrier
pto.comm.tnotify ...
```

MTE2 是 GM read 方向。它需要的是 signal 前不要越过前序 MTE2 工作，但不需要 DDR
release fence。PTOAS 当前仍允许自动插入这类纯 pipe drain。

### 6.3 Cacheable scalar GM store 后发布 signal

适用场景：

- `store_scalar` 写 GM，并且该路径可能经过 cache。

需要的顺序：

```mlir
pto.store_scalar ...
pto.cmo.clean all #pto.address_space<gm>
pto.fence.release #pto.fence_scope<ddr>
pto.comm.tnotify ...
```

`pto.cmo.clean` 把 dirty cache line 推出。`pto.fence.release` 等待并约束 clean 的结果在
signal 发布前可见。

如果只插 `pto.fence.release`，PTOAS 会报错。因为 fence 不会替代 cache clean。

### 6.4 TWait 或 TTest 后读取 cacheable GM payload

适用场景：

- `TWait` 返回后执行 `load_scalar` 读取 GM payload。
- `TTest` 成功观察到 signal 后执行 `load_scalar` 读取 GM payload。

需要的顺序：

```mlir
pto.comm.twait ...
pto.cmo.invalidate all #pto.address_space<gm>
%value = pto.load_scalar ...
```

invalidate 用来避免读取到本地 stale cache line。

### 6.5 Acquire 前本地可能存在 dirty GM cache

适用场景：

- 同一个执行流中，等待 signal 前已经有 cacheable GM store。
- 后续又要在 signal acquire 后读取 GM payload。

需要的顺序：

```mlir
pto.store_scalar ...
pto.cmo.clean all #pto.address_space<gm>
pto.fence.release #pto.fence_scope<ddr>
pto.comm.twait ...
pto.cmo.invalidate all #pto.address_space<gm>
%value = pto.load_scalar ...
```

clean 和 release fence 用来处理本地 dirty cache。invalidate 用来处理 signal 后读取对端
payload 时可能遇到的 stale cache。

## 7. PyPTO 生成建议

PyPTO 需要在 payload publish 边界显式生成 CMO 和 fence。

PyPTO 不需要手动生成 `pto.barrier #pto.pipe<PIPE_MTE3>` 或
`pto.barrier #pto.pipe<PIPE_FIX>`。这是低层 pipe drain 细节，由 PTOAS 根据 release fence
前的 pending GM write pipe 自动插入。这样可以保证最终顺序是对应 pipe barrier 先于
`dsb(DSB_DDR)`，不会出现先 fence、后 drain 的错误顺序。

### 7.1 TPUT 发布 signal

```mlir
pto.comm.tput ...
pto.fence.release #pto.fence_scope<ddr>
pto.comm.tnotify ...
```

### 7.2 TStore 发布 signal

```mlir
pto.tstore ...
pto.fence.release #pto.fence_scope<ddr>
pto.comm.tnotify ...
```

### 7.3 Scalar store 发布 signal

```mlir
pto.store_scalar ...
pto.cmo.clean all #pto.address_space<gm>
pto.fence.release #pto.fence_scope<ddr>
pto.comm.tnotify ...
```

### 7.4 TWait 后读取 scalar payload

```mlir
pto.comm.twait ...
pto.cmo.invalidate all #pto.address_space<gm>
%value = pto.load_scalar ...
```

### 7.5 TTest polling 后读取 scalar payload

```mlir
%ready = pto.comm.ttest ...
scf.if %ready {
  pto.cmo.invalidate all #pto.address_space<gm>
  %value = pto.load_scalar ...
}
```

如果 PyPTO 使用 `pto.ldg` 或 `pto.stg` 并显式选择 uncache 路径，可以避免部分
cacheable scalar GM 问题。但这不是 `pto.cmo.clean` 或 `pto.cmo.invalidate` 的替代品。
如果之前已经存在 dirty 或 stale cache line，仍需要显式 CMO。

## 8. Backend lowering 状态

### 8.1 EmitC

EmitC backend 已经支持真实 lowering：

- `pto.cmo.clean` lower 到 `dcci(..., CACHELINE_OUT)`。
- `pto.cmo.invalidate` lower 到 `dcci(...)`。
- `pto.fence.release` lower 到 `dsb(DSB_DDR)`。
- `pto.fence.acquire` lower 到 `dsb(DSB_DDR)`。

### 8.2 VPTO

VPTO backend 当前没有确认的 DSB 和 DCCI intrinsic ABI。

因此，VPTO lowering 中现在提供的是 fail-fast stub：

- `pto.cmo.clean`
- `pto.cmo.invalidate`
- `pto.fence.release`
- `pto.fence.acquire`

如果这些 op 进入 VPTO LLVM lowering，PTOAS 会报错，提示 VPTO backend 尚不支持这些
memory-consistency op，需要确认 DSB/DCCI intrinsic ABI 后再接真实 lowering。

这样做的目的不是支持 VPTO 运行，而是避免 unsupported op 静默残留到后端 IR。

## 9. 当前限制

当前实现优先保证正确性，仍有以下限制：

- CMO 是 whole-cache 粒度，不是精确地址范围。
- `TWait` 和 `TTest` acquire 侧当前只覆盖 `load_scalar`。
- VPTO 暂不支持 CMO 和 DDR fence 的真实 lowering。
- 对复杂 CFG 的分析仍是保守近似，不做完整 path-sensitive 数据流。
- MemoryConsistency pass 校验的是显式内存一致性契约，不替代 InsertSync 的 alias 和 pipe
  同步分析。

## 10. 后续工作

后续可以分几步推进：

1. 和 VPTO/Bisheng 对齐 DSB 和 DCCI intrinsic ABI，并补齐 VPTO lowering。
2. 将 whole-cache CMO 优化成精确 GM address range CMO。
3. 扩展 acquire 侧 consumer 范围，从 `load_scalar` 扩展到更多 cacheable GM read。
4. 将 macro op phase 的 memory descriptor 做得更精细，减少误报。
5. 在 PyPTO 和 PTOAS 之间明确 cacheable 与 uncacheable GM 访问的 IR 契约。
