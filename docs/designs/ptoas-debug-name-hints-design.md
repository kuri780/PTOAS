# PTOAS 变量名保留与调试名提示设计

## 1. 文档范围

本文定义 issue `#337` 对应的能力：

- 前端为 PTO IR 附带“原始变量名”提示信息
- PTOAS 在 `.pto -> .cpp` 编译链路中输出稳定的溯源注释，提升问题定位效率

本文只讨论调试可定位性，不改变任何 IR 语义、优化行为或生成代码功能。

## 2. 背景问题

当前 PTOAS 在 `level3` 编译链路中会经过多轮 rewrite、CSE 和 EmitC 降低，最终 `kernel.cpp`
中的局部变量名通常被重新编号为 `v0`、`v1`、`v2`。

这会带来两个直接问题：

- 输入 `.pto` 与输出 `.cpp` 不容易一一对照
- 前端 Python 里的业务变量名无法保留下来，定位问题时大量 `vXXX` 可读性很差

issue `#337` 希望解决的是“可对照、可定位”，而不是改变编译结果本身。

## 3. 设计目标

本设计当前阶段的目标如下：

- 支持前端向 PTO IR 传递可选的变量名提示信息
- 生成 `kernel.cpp` 时为可恢复的值追加 `// pto: %name` 溯源注释
- 默认打印 IR 时不明显增加噪声，避免把 IR 变丑
- 在发生 CSE、控制流合并、lowering 新造临时值时，仍能给出稳定、可定位的来源提示

## 4. 非目标

本设计明确不保证以下行为：

- 不保证输入 `.pto` 中的 `%v37` 在输出 `.cpp` 中仍然严格叫 `v37`
- 不保证所有中间优化后仍保留一一对应关系
- 不为名字建立任何语义约束；名字仅用于调试与阅读
- 不要求所有前端都必须提供名字提示

原因很直接：当前链路存在 CSE、值合并、值拆分、新值物化与 `emitc.variable` 提升，最终 C++
中的很多值并不存在与输入 SSA 的严格一一映射关系。

## 5. 总体方案

### 5.1 核心思路

PTOAS 将“原始变量名”视为调试元数据，而不是语义属性。

当前阶段的具体做法：

- 前端把名字提示写入 op 的 `Location` 元数据
- 对直接输入的 textual `.pto`，PTOAS 额外从源码文本里提取 SSA 名、函数参数名和 block argument 名
- PTOAS 在 rewrite / lowering 时尽量传播该元数据
- EmitC lowering 后插入 provenance marker，并在最终 C++ 文本中转成 `// pto: %name` 注释

这样做的核心收益是：

- IR 语义不受影响
- 默认 textual IR 不需要把一堆 `name_hint` 属性直接打印出来
- 只有在显式看调试信息时，这些名字才会显式出现
- 不需要在 C++ 文本后处理阶段重建符号表和作用域来做重命名

### 5.2 为什么不会让 IR 变丑

本设计不建议把名字直接做成普通 op 属性，比如：

```mlir
%0 = pto.foo ... { pto.result_name_hints = ["query_tile"] }
```

这种方案虽然直观，但会让所有 IR 都充满调试字段。

本设计改为把名字放进 `Location`：

- 默认 IR 打印时，location 不会成为主要视觉噪声
- 只有在显式打开 debug info 打印时，名字才会展示出来
- 语义属性区保持干净

因此，IR 里是“多带了调试备注”，不是“把主体语法变复杂”。

## 6. IR 名字承载方式

### 6.1 单结果 op

单结果 op 的名字提示记录在 `NameLoc` 或 `FusedLoc` 元数据中。

示意：

```mlir
%0 = pto.tload ... loc("query_tile")
```

其含义是：该 op 的主结果推荐名字为 `query_tile`。

### 6.2 多结果 op

多结果 op 不能只靠单个 `NameLoc` 表达所有结果名，因此使用 `FusedLoc` 的 metadata
携带结果名数组。

逻辑示意：

```text
loc(fused<metadata=["lhs", "rhs", "acc"]>[...])
```

这里 metadata 只服务于调试命名，不参与语义。

### 6.3 前端接口约定

前端可以提供名字，也可以不提供。

- 提供时：PTOAS 尽量保留
- 不提供时：
  - 若输入是 textual `.pto`，PTOAS 会先尝试回收源码里的 SSA / 参数 / block arg 名
  - 若仍然没有名字提示，再使用现有 fallback 命名

因此该能力是增量增强，不破坏现有前端。

## 7. 名字传播规则

### 7.1 直接透传

如果一个 rewrite 基本是一对一替换：

- 新值继承原值名字

示例：

- `pto.xxx -> emitc.cast`
- `pto.xxx -> emitc.variable`
- `pto.xxx -> 某个一对一的 helper op`

### 7.2 派生命名

如果 lowering 会从一个源值派生多个新值，则在源名字基础上追加稳定后缀。

建议后缀包括：

- `_cast`
- `_addr`
- `_tile`
- `_shape`
- `_stride`
- `_tmp`

例如：

- `query_tile -> query_tile_addr`
- `query_tile -> query_tile_cast`

### 7.3 合并场景

如果多个值被合并成一个值，例如 CSE 或公共表达式复用：

- 优先保留支配值的已有名字
- 若名字冲突或为空，回退到稳定生成名

### 7.4 控制流与 hoist 场景

`scf` / `cf` / `emitc.variable` 路径会引入额外临时变量，这些变量在源 `.pto` 中通常没有严格对应项。

对这类值：

- 若来源明确，则使用来源名加后缀，如 `_phi`、`_cond`、`_tmp`
- 若来源不明确，则使用稳定 fallback 名

本次实现里，控制流合流出来的 block argument 采用保守传播策略：

- 先尽量把 block argument 名字传播到 EmitC value loc
- 若某条 lowering 路径无法稳定维持该关联，则 fail-closed 为不附带该条名字提示，而不是在 C++ 文本层猜测性回填
- 当前阶段的 provenance 注释只针对“有结果的 op”插 marker；因此 merged block argument 不保证一定落成单独的 `// pto: %name` 注释

## 8. C++ 输出策略

### 8.1 当前阶段

当前阶段只保证溯源注释，不保证把调试名重写成最终的 C++ 局部变量名。

原因是：重命名的正确性依赖完整的符号表、作用域和冲突检测，而这些信息属于
`CppEmitter` 内部命名逻辑。若在 `translateToCpp` 之后对 C++ 文本做字符串层重命名，
需要重新近似构建一套作用域/碰撞分析，风险较高。

因此本阶段只保证：

- provenance marker 会被清理干净，不会泄漏到最终输出
- 可恢复的值会带 `// pto: %name` 注释
- 对包含 `*/`、换行等特殊字符的原始名做 comment-safe 转义，保证输出合法
- 无法稳定恢复的值保持 EmitC/CppEmitter 原有 `vN` 命名

### 8.2 后续阶段

若后续要恢复“语义重命名”能力，应改为在 `CppEmitter::getOrCreateName(Value)` 一类
真正取名的位置消费这些调试名，并复用 emitter 自己的冲突/作用域逻辑，而不是在
`translateToCpp` 之后重写 C++ 文本。

## 9. 可见性与开关

### 9.1 默认行为

默认情况下：

- IR 语义不变
- 若前端提供了名字提示，PTOAS 尽量在 provenance 注释里保留
- 普通 IR 打印不要求显式展示这些提示

### 9.2 调试打印

若开发者需要查看 IR 中实际承载的调试名字，可通过调试打印模式展示 location。

也就是说：

- 平时看 IR：保持干净
- 排查名字传播问题时：打开 debug info 看 metadata

## 10. 与现有链路的关系

该设计需要覆盖以下环节：

- 前端生成 PTO IR 时附带名字提示
- textual `.pto` 输入时，基于 `AsmParserState` 恢复 SSA / 参数 / block arg 名
- PTOAS rewrite / lowering helper 在替换时传播名字
- `PTOToEmitC` 中新建 `emitc::VariableOp`、`emitc::CastOp` 等值时继承或派生 provenance
- 最终 `translateToCpp` 后在 PTOAS 包装层中仅把 provenance marker 转成安全注释

## 11. 溯源注释（issue #337 第 1 点：可定位性）

issue #337 的第 1 点要求“`.pto` 的 `%N` 与 `.cpp` 的 `vN` 序号一致以便定位”。
但 level3 链路有 CSE、值合并、新值物化与 `emitc.variable` 提升，输出 `cpp` 的 `vN`
编号由 EmitC 按遍历顺序重新分配，与输入 SSA 编号在原理上无法逐号对齐。
因此本设计当前阶段对该点采用“溯源注释”路线，而非强制的 `%N == vN`。

### 11.1 溯源注释

PTOAS 在 EmitC lowering 后，对每个结果可溯源到输入 `.pto` SSA 名的 op，额外插入
`/* PTOAS_PROVENANCE:rawname */` 标记。该标记携带**未经 sanitize 的原始 SSA 名**
（如 `0`、`24`、`query_tile`、`c0`）。

在最终 C++ 后处理阶段，该标记被转换为 marker 原地位置的独立注释：

```cpp
// pto: %0
int32_t v3 = helper(v1, v2);
// pto: %query_tile
LocalTensor<half> v12 = ...;
```

这样读者在 `kernel.cpp` 里看到任意一行，都能直接读到它对应的输入 SSA 名，从而在
`.pto` 里定位到来源，无需依赖序号一致。

### 11.2 限制

溯源注释解决的是“能定位来源”，不是“变量本身可读”。因此最终 C++ 里的参数名、局部名
仍可能是 `vN`。对被 CSE 合并、无法溯源到单一输入值的生成值，不强制挂注释（降级为无注释）。

## 12. 风险与限制

主要风险如下：

- 某些 pass 新建值但没有传播 provenance，会导致局部缺少 `// pto: ...`
- 多结果 op 的 metadata 约定若不统一，前后端容易理解不一致
- 名字传播若写成语义属性，容易污染 IR；因此必须坚持“调试元数据”定位

限制如下：

- 该设计当前阶段只能提供“定位来源”，不能提供“最终变量名语义化”
- 对 aggressive CSE 后的公共值，只能保留最终幸存值的名字
- 对 textual `.pto` 的 SSA 名恢复依赖 `AsmParserState` 暴露的解析结果；若 lowering 后的 CFG 形状不再稳定匹配，则相关 hint 会 fail-closed 丢弃，而不是猜测性错挂

## 13. 测试建议

建议至少覆盖以下测试：

- 单结果 op：前端名字能出现在最终 `// pto: %name` 注释
- 多结果 op：多个结果的 provenance 不会错位
- 名字含特殊字符：注释能安全转义
- 控制流 / `emitc.variable` / hoist：注释不会错挂到错误声明
- textual `.pto`：函数参数名、局部 SSA 名要能在 provenance 注释中保留；CFG block arg 至少要验证“不发生错挂”，不要求当前阶段每个 merged arg 都生成单独注释
- 未提供 hint：仍保持现有 `vN` 回退行为

## 14. 结论

本设计采用“名字作为调试元数据”的路线：

- 不把名字当语义
- 不要求逐号保真
- 不把 IR 默认打印搞得很吵
- 当前阶段重点解决 `.pto`、前端 Python 和最终 `kernel.cpp` 之间“能对得上来源”的问题

这条路线对现有编译链路侵入较小，也最符合 issue `#337` 的真实诉求。
