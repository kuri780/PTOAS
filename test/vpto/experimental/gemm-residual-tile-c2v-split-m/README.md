# gemm-residual-tile-c2v-split-m (实验性 / 非门禁)

## 状态：已知失败 — 非阻塞

本目录是 PTODSL/Tile PTO 调查的实验性用例，**不参与自动验证门禁**。

## 失败原因

### 输入结构

此用例采用 `pto.entry` 包装函数 + 多个 `pto.kernel_kind` 子函数的结构：

```mlir
module attributes {pto.target_arch = "a5"} {
  func.func @wrapper(...) attributes {pto.entry} {
    func.call @cube_func(...)
    func.call @vector_func(...)
    return
  }
  func.func private @cube_func(...)
    attributes {pto.kernel_kind = #pto.kernel_kind<cube>} { ... }
  func.func private @vector_func(...)
    attributes {pto.kernel_kind = #pto.kernel_kind<vector>} { ... }
}
```

### 已知问题

**VPTOSplitCVModule** pass 当前无法规范化 (normalize) 这种 `pto.entry` + 多个 function-level `pto.kernel_kind` 的 container 结构。该 pass 期望的是单一 `pto.kernel` 标记的顶层函数，内含 `pto.section.cube` / `pto.section.vector` 区域。

### 错误表现

编译时在 VPTO container normalization 阶段失败：
- 无法识别多个 `kernel_kind` 函数的 container 语义
- Cube/Vector 的 C2V pipe 建立和 subblock dispatch 无法正确展开

## 与 Phase 2 的关系

- **非 Phase 2 主线阻塞**：Phase 2 (GEMM+Residual+RMSNorm C2V) 的底层 vreg 原型已通过 `test/vpto/cases/kernels/gemm-residual-rmsnorm-c2v-split-m/` 验证。
- **独立问题**：此用例暴露的是 PTODSL 前端 container 支持问题，需要 VPTOSplitCVModule 扩展以支持 `pto.entry` 包装模式。
- **不依赖 ExpandTileOp**：底层 vreg 原型完全不使用 Tile compute ops (OpPipeInterface)，不触发 ExpandTileOp。Tile dialect 中的 TmatMul/Tadd/Tcvt 等操作是通过 PTODSL→VPTO lowering 链处理的，与底层 vreg 直写无关。

## 复现命令

```bash
ptoas --pto-arch=a5 --pto-backend=vpto --pto-level=level3 \
  --enable-tile-op-expand --enable-insert-sync \
  kernel.pto -o kernel.fatobj.o
```

预期错误：VPTOSplitCVModule 无法处理 `pto.entry` + function-level `pto.kernel_kind` 结构。

## 后续计划

- VPTOSplitCVModule 扩展支持 `pto.entry` 包装模式后，此用例应重新评估
- 届时可将此目录移回 `test/vpto/cases/kernels/` 并纳入门禁

## 文件清单

| 文件 | 说明 |
|------|------|
| `kernel.pto` | Tile dialect: tmatmul → tpush_to_aiv → tpop_from_aic → tadd → tstore |
| `golden.py` | BF16 GEMM + BF16 Residual → FP32 output 参考值 |
| `compare.py` | 数值比较脚本 |
| `launch.cpp` | ACL kernel launch wrapper |
| `main.cpp` | ACL host runner |
| `ptoas.flags` | ptoas 编译参数 |
