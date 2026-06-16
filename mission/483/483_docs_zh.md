# Issue 483 PTODSL RMSNorm 前端接口草案

本文档描述用于表达 RMSNorm SimtVF kernel 的 PTODSL 前端接口草案。目标是在尽量复用现有 PTODSL 概念的前提下，只补齐缺失的通用能力：连续 vector 访存、DSL 层 vector 值、vector 规约、SIMT all-reduce，以及 lane-local 指针/局部存储。

## 1. 扩展 scalar memory access

`scalar.load` 和 `scalar.store` 继续作为用户层面的内存访问 API。它们从“只支持标量读写”扩展为：当用户显式指定连续元素个数时，可以读写一段连续 vector。

`offset` 的单位是元素，不是字节。

#### `scalar.load(ptr, offset=0, *, contiguous=None) -> ScalarType | VecType`

**描述**：从 typed pointer 读取一个标量元素，或者读取 `contiguous` 个连续元素。

**参数**：

| 参数 | 类型 | 说明 |
|------|------|------|
| `ptr` | `PtrType` | 源 typed pointer。它可以是 kernel 入参中的 typed pointer，例如 `pto.ptr(T, "ub")`；也可以是 `pto.alloc_buffer(...)` 返回的 typed pointer。 |
| `offset` | index-like PTO 标量或 Python 整数 | 从 `ptr` 开始计算的元素偏移。默认值是 `0`。 |
| `contiguous` | `int` 或 `None` | 要读取的连续元素个数。`None` 和 `1` 表示标量 load；`N > 1` 表示 vector load。 |

**返回值**：

| 返回值 | 类型 | 说明 |
|--------|------|------|
| `value` | `ScalarType` | 当 `contiguous` 是 `None` 或 `1` 时返回；类型和源 pointer 的元素类型一致。 |
| `value` | `pto.vec(T, N)` | 当 `contiguous=N` 且 `N > 1` 时返回；`T` 是源 pointer 的元素类型。 |

**示例**：

```python
# 现有标量访问仍然有效。
x = scalar.load(x_ub, offset)

# 新增连续 vector 访问：从 UB pointer 连续读取 4 个 f32。
x4 = scalar.load(x_ub, offset, contiguous=4)

# 从 lane-local pointer 连续读取 4 个 f32。
w4 = scalar.load(w_frag, frag_offset, contiguous=4)
```

**约束**：

| 规则 | 说明 |
|------|------|
| 元素单位 | `offset` 是元素偏移，不是字节偏移。 |
| 元素类型 | vector 的元素类型从源 typed pointer 的元素类型推导。 |
| `contiguous` 取值 | 如果提供 `contiguous`，它必须是编译期已知的正整数。 |

---

#### `scalar.store(value, ptr, offset=0, *, contiguous=None) -> None`

**描述**：向 typed pointer 写入一个标量值，或者写入一个 vector 值。

**参数**：

| 参数 | 类型 | 说明 |
|------|------|------|
| `value` | `ScalarType` 或 `VecType` | 要写入的值。标量值写 1 个元素；vector 值写连续多个元素。 |
| `ptr` | `PtrType` | 目标 typed pointer。它可以是 kernel pointer，也可以是 `pto.alloc_buffer(...)` 返回的 typed pointer。 |
| `offset` | index-like PTO 标量或 Python 整数 | 从 `ptr` 开始计算的元素偏移。默认值是 `0`。 |
| `contiguous` | `int` 或 `None` | 可选的显式宽度校验。如果 `value` 是 vector，并且提供了 `contiguous=N`，则 `N` 必须和 vector lane 数一致。 |

**返回值**：无。

**示例**：

```python
# 现有标量 store 仍然有效。
scalar.store(rstd, rstd_ub, ping)

# 新增 vector store：向 UB pointer 连续写入 4 个 f32。
scalar.store(y4, y_ub, y_offset)

# 可选显式宽度校验。
scalar.store(y4, y_ub, y_offset, contiguous=4)

# 把 vector 写入 lane-local pointer。
scalar.store(x4, x_frag, frag_offset)
```

**约束**：

| 规则 | 说明 |
|------|------|
| 标量 store | 如果 `value` 是标量，则只写入 1 个元素。 |
| vector store | 如果 `value` 是 `pto.vec(T, N)`，则连续写入 `N` 个元素。 |
| 类型匹配 | 写入的标量类型或 vector 元素类型必须和目标 pointer 的元素类型一致。 |
| 显式宽度 | 如果对 vector store 提供 `contiguous=N`，则 `N` 必须和 vector lane 数一致。 |
| 元素单位 | `offset` 是元素偏移，不是字节偏移。 |

**RMSNorm 使用模式**：

```python
x4 = scalar.load(x_ub, x_offset, contiguous=4)
scalar.store(x4, x_frag, r * lanes)

x4 = scalar.load(x_frag, r * lanes, contiguous=4)
w4 = scalar.load(w_frag, r * lanes, contiguous=4)
y4 = x4 * rstd4 * w4
scalar.store(y4, y_ub, y_offset)
```

## 2. DSL vector 类型

PTODSL 需要一个前端 vector 类型，用来表达固定宽度的 builtin vector 值，例如 `vector<4xf32>`。这个 vector 类型和 PTO 硬件 vector register 类型不同，例如 `!pto.vreg<NxT>`。

#### `pto.vec(dtype, lanes) -> VecType`

**描述**：构造一个 DSL vector 类型，其中包含 `lanes` 个 `dtype` 类型元素。

**参数**：

| 参数 | 类型 | 说明 |
|------|------|------|
| `dtype` | PTODSL 标量 dtype | 元素类型，例如 `pto.f32` 或 `pto.i32`。 |
| `lanes` | `int` | vector 元素个数。RMSNorm float4 访问使用 `4`。 |

**返回值**：

| 返回值 | 类型 | 说明 |
|--------|------|------|
| `vec_type` | `VecType` | DSL vector 类型，例如 `pto.vec(pto.f32, 4)`。 |

**示例**：

```python
f32x4 = pto.vec(pto.f32, 4)
```

---

#### `pto.vec(dtype, lanes)(value) -> VecValue`

**描述**：构造一个 vector 值。当 `value` 是标量时，把这个标量广播到所有 vector lane。它替代之前提出的 `vf_splat_f32x4` helper。

**参数**：

| 参数 | 类型 | 说明 |
|------|------|------|
| `dtype` | PTODSL 标量 dtype | vector 元素类型。 |
| `lanes` | `int` | vector lane 数。 |
| `value` | 标量值或兼容 vector 值 | 用来构造 vector 的值；如果是标量，则广播到所有 lane。 |

**返回值**：

| 返回值 | 类型 | 说明 |
|--------|------|------|
| `result` | `pto.vec(dtype, lanes)` | 构造出来的 vector 值。 |

**示例**：

```python
# 把一个标量 rstd 广播成 [rstd, rstd, rstd, rstd]。
rstd4 = pto.vec(pto.f32, 4)(rstd)

# 然后使用 Python 普通算术表达逐元素 vector 计算。
y4 = x4 * rstd4 * w4
```

**约束**：

| 规则 | 说明 |
|------|------|
| 广播 | 标量输入会广播到 vector 的每个 lane。 |
| 算术 | 两个兼容 `VecValue` 对象上的 Python 算术是逐元素算术。 |
| 类型区分 | `pto.vec(dtype, lanes)` 是 builtin vector 值类型，不等同于 `pto.vreg_type(lanes, dtype)`。 |

## 3. Vector reduce

RMSNorm 需要先在一个 workitem 内部，把一个 lane-local vector 里的元素求和，然后再跨 SIMT lane 做 all-reduce。例如 `x4 * x4` 会在一个 workitem 内得到 4 个平方值，这 4 个值需要先合成一个标量 `local_sum`。

#### `scalar.reduce(value, *, op="add") -> ScalarType`

**描述**：把一个标量或 DSL vector 值规约成一个标量。对于 `pto.vec(T, N)` 输入，该操作会把 `N` 个 vector lane 合并，并返回元素类型为 `T` 的标量。

**参数**：

| 参数 | 类型 | 说明 |
|------|------|------|
| `value` | `ScalarType` 或 `VecType` | 要规约的输入值。RMSNorm 使用 `pto.vec(pto.f32, 4)`。 |
| `op` | `"add"` | 规约操作。RMSNorm 初始需求是 sum reduction。 |

**返回值**：

| 返回值 | 类型 | 说明 |
|--------|------|------|
| `result` | `ScalarType` | 标量规约结果。对于 `pto.vec(T, N)`，返回类型是 `T`。 |

**示例**：

```python
x4 = scalar.load(x_ub, x_offset, contiguous=4)
sq4 = x4 * x4
local_sum = local_sum + scalar.reduce(sq4, op="add")
```

**语义**：

```text
scalar.reduce([v0, v1, v2, v3], op="add") = v0 + v1 + v2 + v3
```

**约束**：

| 规则 | 说明 |
|------|------|
| 初始操作 | 初始必须支持的操作是 `op="add"`；其他规约操作可以后续扩展。 |
| vector 输入 | vector 输入只在当前 workitem 内部跨元素规约。 |
| 和 SIMT 规约的区别 | 这不是跨 SIMT workitem 的规约；跨 workitem 规约使用 `pto.simt_allreduce_sum`。 |

## 4. SIMT all-reduce sum

RMSNorm 还需要把每个参与的 SIMT workitem 各自产生的一个标量做求和，并把同一个求和结果返回给每个 workitem。这个接口和 `scalar.reduce` 不同：`scalar.reduce` 只是在一个 workitem 内部把一个 vector 的多个元素规约成一个标量，而 `pto.simt_allreduce_sum` 是跨多个 SIMT workitem 做规约。

#### `pto.simt_allreduce_sum(value, *, threads, scale=1, thread_offset=0, scratch=None, scratch_offset=0) -> ScalarType`

**描述**：对每个参与的 SIMT workitem 输入的一个标量做求和，并把总和返回给每个参与的 workitem。

**参数**：

| 参数 | 类型 | 说明 |
|------|------|------|
| `value` | PTO 标量 | 当前 workitem 贡献给 all-reduce 的标量值。RMSNorm 中通常是当前 lane 算出的局部平方和。 |
| `threads` | `int` | 参与规约的 SIMT workitem 数量。RMSNorm 常见值是 `128`。 |
| `scale` | `int` | 可选缩放因子，用来对齐 all-reduce 风格接口。默认值是 `1`。 |
| `thread_offset` | `int` | 逻辑 workitem 偏移。默认值是 `0`。 |
| `scratch` | typed UB pointer 或 `None` | 可选临时空间。某些实现方式可能需要用它暂存每个 workitem 的中间值。 |
| `scratch_offset` | index-like PTO 标量或 Python 整数 | `scratch` 内部的元素偏移。默认值是 `0`。 |

**返回值**：

| 返回值 | 类型 | 说明 |
|--------|------|------|
| `result` | `ScalarType` | 所有参与 workitem 的 `value` 求和结果。每个参与 workitem 都拿到同一个结果。 |

**示例**：

```python
local_sum = scalar.reduce(sq4, op="add")
sum_sq = pto.simt_allreduce_sum(
    local_sum,
    threads=128,
    scratch=reduce_scratch,
)
```

**语义**：

```text
对于每个参与的 workitem：
  result = sum(所有参与 workitem 的 value)
```

**约束**：

| 规则 | 说明 |
|------|------|
| 标量输入 | `value` 是每个 workitem 的一个标量，不是 vector。 |
| 参与线程数 | `threads` 必须和当前 SIMT launch/body 的规约范围匹配。 |
| 临时空间 | 如果传入 `scratch`，它必须有足够空间支撑对应的实现方式和 `threads` 数量。 |
| 和 vector reduce 的区别 | `scalar.reduce` 用于 `vector<4xf32> -> f32`；`pto.simt_allreduce_sum` 用于多个 workitem 的标量求和并广播。 |

## 5. Lane-local pointer / alloc_buffer

RMSNorm 需要每个 workitem 私有的局部数组，例如 `x_frag[32]` 和 `w_frag[32]`。这些数组属于当前 SIMT workitem 自己，不在不同 workitem 之间共享。接口上，`pto.alloc_buffer(...)` 返回 typed pointer，但它背后的分配语义仍然可以表示“每个 workitem 私有的一段局部存储”。

#### `pto.alloc_buffer(shape, dtype, *, scope, persistent=False) -> PtrType`

**描述**：在指定作用域分配一块线性可寻址存储，并返回指向这块存储的 typed pointer。对于 RMSNorm，`scope="ub"` 用于 UB scratch，例如 `x_ub` / `y_ub`；`scope="local"` 用于 lane-local pointer，例如 `x_frag` / `w_frag`。它替代之前提出的 lane-local 存储接口名 `alloc_fragment`。

**参数**：

| 参数 | 类型 | 说明 |
|------|------|------|
| `shape` | `tuple[int, ...]` 或 `list[int]` | 分配形状。例如对 `scope="local"`，`(32,)` 表示每个 workitem 拥有 32 个元素。 |
| `dtype` | PTODSL 标量 dtype | 元素类型，例如 `pto.f32`。 |
| `scope` | `"ub"` 或 `"local"` | 存储作用域。`"ub"` 表示 UB scratch storage；`"local"` 表示每个 SIMT workitem 私有的 storage instance。 |
| `persistent` | `bool` | local storage 是否需要跨多次 SIMT launch 保留内容。默认值是 `False`。 |

**返回值**：

| 返回值 | 类型 | 说明 |
|--------|------|------|
| `ptr` | `PtrType` | 指向已分配存储的 typed pointer。元素类型来自 `dtype`；地址空间来自 `scope`。 |

**示例**：

```python
# UB scratch pointer，用来放一个或两个 token row。
x_ub = pto.alloc_buffer((2, 4096), pto.f32, scope="ub")
y_ub = pto.alloc_buffer((2, 4096), pto.f32, scope="ub")

# 当前 token 的 x 临时值。每个 SIMT workitem 私有。
x_frag = pto.alloc_buffer((32,), pto.f32, scope="local")

# 权重 w 的临时值。每个 workitem 私有，并且希望跨 token 循环保留。
w_frag = pto.alloc_buffer((32,), pto.f32, scope="local", persistent=True)

# 通过 scalar.load/store 做 vector 访问。
scalar.store(x4, x_frag, r * 4)
x4 = scalar.load(x_frag, r * 4, contiguous=4)
```

**可选下标语法糖**：

```python
# 标量元素访问语法糖。
x_frag[i] = value
value = x_frag[i]

# 连续 vector 访问语法糖。
x_frag[i : i + 4] = x4
x4 = x_frag[i : i + 4]
```

这些下标形式本质上是 `scalar.load` 和 `scalar.store` 的语法糖：

```python
value = x_frag[i]                  # scalar.load(x_frag, i)
x_frag[i] = value                  # scalar.store(value, x_frag, i)
x4 = x_frag[i : i + 4]             # scalar.load(x_frag, i, contiguous=4)
x_frag[i : i + 4] = x4             # scalar.store(x4, x_frag, i, contiguous=4)
```

**约束**：

| 规则 | 说明 |
|------|------|
| 作用域所有权 | `scope="ub"` 返回的 pointer 指向 UB scratch storage；`scope="local"` 返回的 pointer 指向当前 SIMT workitem 私有的 storage instance。 |
| 形状 | `shape` 必须在编译期已知。 |
| persistent storage | `persistent=True` 主要用于 local storage，例如 RMSNorm 权重，希望一次读入后跨多次 SIMT launch 复用。 |
| 访问方式 | 返回的 pointer 通过 `scalar.load/store` 或等价下标语法糖访问。 |

## 6. RMSNorm 数据流示例

下面的草图展示这些接口如何组合起来表达一个 RMSNorm SIMT body。在这个示例中，`x_ub` 和 `y_ub` 是通过 `pto.alloc_buffer(..., scope="ub")` 分配的 UB pointer；`x_frag` 和 `w_frag` 是通过 `scope="local"` 分配的 lane-local pointer。

```python
@pto.simt
def rmsnorm_lane_body(
    x_ub,
    y_ub,
    reduce_scratch: pto.ptr(pto.f32, "ub"),
    x_frag,
    w_frag,
    eps: pto.f32,
    *,
    threads: pto.const_expr,
    rounds: pto.const_expr,
    lanes: pto.const_expr = 4,
    hidden_size: pto.const_expr = 4096,
):
    tx = pto.get_tid_x()
    local_sum = 0.0

    for r in pto.static_range(0, rounds):
        offset = r * threads * lanes + tx * lanes
        x4 = scalar.load(x_ub, offset, contiguous=lanes)
        scalar.store(x4, x_frag, r * lanes)

        sq4 = x4 * x4
        local_sum = local_sum + scalar.reduce(sq4, op="add")

    sum_sq = pto.simt_allreduce_sum(
        local_sum,
        threads=threads,
        scratch=reduce_scratch,
    )

    rstd = 1.0 / scalar.sqrt(sum_sq / hidden_size + eps)
    rstd4 = pto.vec(pto.f32, lanes)(rstd)

    for r in pto.static_range(0, rounds):
        offset = r * threads * lanes + tx * lanes
        x4 = scalar.load(x_frag, r * lanes, contiguous=lanes)
        w4 = scalar.load(w_frag, r * lanes, contiguous=lanes)
        y4 = x4 * rstd4 * w4
        scalar.store(y4, y_ub, offset)
```

## 7. 接口汇总

| 接口 | 用途 |
|------|------|
| `scalar.load(ptr, offset=0, *, contiguous=None)` | 从 typed pointer 读取一个标量或一段连续 vector。 |
| `scalar.store(value, ptr, offset=0, *, contiguous=None)` | 向 typed pointer 写入一个标量或一段连续 vector。 |
| `pto.vec(dtype, lanes)` | 定义 DSL builtin vector 类型。 |
| `pto.vec(dtype, lanes)(value)` | 构造 vector 值，包括把标量广播成 vector。 |
| `scalar.reduce(value, *, op="add")` | 在一个 workitem 内部把 vector 值规约成标量。 |
| `pto.simt_allreduce_sum(value, *, threads, scale=1, thread_offset=0, scratch=None, scratch_offset=0)` | 把每个参与 SIMT workitem 的一个标量求和，并把结果广播给所有参与 workitem。 |
| `pto.alloc_buffer(shape, dtype, *, scope, persistent=False)` | 分配线性 UB 或 lane-local storage，并返回 typed pointer。 |
