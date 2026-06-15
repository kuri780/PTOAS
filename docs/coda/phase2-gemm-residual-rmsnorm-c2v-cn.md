# Phase 2: GEMM + Residual + RMSNorm C2V Split-M（中文）

## 概述

在 Phase 1 的 GEMM + Residual Add C2V Split-M 基础上，于 Vector 侧叠加 FP32 RMSNorm，
形成端到端融合算子。GEMM 结果和 Residual Add 中间结果不写 GM，仅最终 RMSNorm 输出
执行 UB→GM。

**数学定义**：

```
x[m,n]       = bf16(A[m,k]) @ bf16(B[k,n]) + bf16(Residual[m,n])
sum_sq[m]    = Σₙ x[m,n]²
inv_rms[m]   = 1 / √(sum_sq[m] / N + ε)
Output[m,n]  = x[m,n] × inv_rms[m] × Gamma[n]
```

**固定参数**：M=16, K=64, N=256, ε=1e-6，静态形状。

**约束**：GEMM 和 Residual Add 的 FP32 中间结果禁止写入 GM。不允许 `pto.mte_l0c_gm`。

## 环境

| 项目 | 值 |
|------|-----|
| 服务器 | x86_64 Linux, CANN 9.0.0 |
| 编译器 | bisheng (clang 15.0.5) |
| 模拟器 | dav_3510 (Ascend950PR) |
| git rev | 基于 coda-style-fusion 分支，验证提交见测试记录 |

## 新增/相关文件

### 融合版（测试通过）

```
test/vpto/cases/kernels/gemm-residual-rmsnorm-c2v-split-m/
├── kernel.pto      # Cube: MAD + mte_l0c_ub split_m; Vector: 两遍 vreg RMSNorm
├── golden.py       # BF16截断 + chunked FP32 row reduction + RMSNorm + Gamma
├── compare.py      # atol=1e-4, rtol=1e-4 数值验证
├── launch.cpp      # ACL launch wrapper（7 参数）
└── main.cpp        # ACL host runner（含 Gamma buffer）
```

### 未融合基线（待实测对比）

```
test/vpto/cases/kernels/gemm-residual-rmsnorm-no-c2v/
├── kernel.pto      # GEMM→C2V→UB→GM→UB→RMSNorm（含 GM round-trip）
├── golden.py       # 与融合版相同的 golden 生成逻辑
├── compare.py      # 相同门限
├── launch.cpp      # ACL launch wrapper（9 参数，含 gemm_gm temp）
└── main.cpp        # ACL host runner
```

### 实验性用例（非门禁）

```
test/vpto/experimental/gemm-residual-tile-c2v-split-m/
├── README.md       # 已知失败说明与非阻塞状态
├── kernel.pto      # Tile dialect: tmatmul + tpush_to_aiv 等
└── ...             # 其他配套文件
```

## 数据流

```
[Cube Pipeline]
  BF16 A[16,64] GM ──MTE2──▶ L1 ──MTE1──▶ L0A
  BF16 B[64,256] GM ──MTE2──▶ L1 ──MTE1──▶ L0B (transpose)
      │                                    │
      └──────────── MAD ───────────────────┘
                      │
                 FP32 L0C[16,256]
                      │
            FIXPIPE C2V split_m
                      │
         ┌────────────┴────────────┐
         ▼                         ▼
  Vector UB (subblock 0)     Vector UB (subblock 1)
  GEMM tile [8,256] FP32     GEMM tile [8,256] FP32
         │                         │
    ┌────┴────┬──────────┐    ┌────┴────┬──────────┐
    │MTE2     │MTE2      │    │MTE2     │MTE2      │
    │Residual │Gamma     │    │Residual │Gamma     │
    │[8,256]  │[256]     │    │[8,256]  │[256]     │
    │BF16     │FP32      │    │BF16     │FP32      │
    └────┬────┴────┬─────┘    └────┬────┴────┬─────┘
         │         │               │         │
         ▼         ▼               ▼         ▼
    ┌─────────────────────────┐┌─────────────────────────┐
    │ Pass 1 (per row):       ││ Pass 1 (per row):       │
    │  vlds GEMM + vlds Res   ││  vlds GEMM + vlds Res   │
    │  vcvt BF16→FP32         ││  vcvt BF16→FP32         │
    │  vadd = x               ││  vadd = x               │
    │  vsts x (save for pass2)││  vsts x (save for pass2)│
    │  vmul x²                ││  vmul x²                │
    │  vcadd reduce (4 chunks)││  vcadd reduce (4 chunks)│
    │  vadd accumulate        ││  vadd accumulate        │
    │                         ││                         │
    │ Scalar chain:           ││ Scalar chain:           │
    │  vmuls(1/N)→vadds(ε)   ││  vmuls(1/N)→vadds(ε)   │
    │  →vsqrt→vdiv(1/root)   ││  →vsqrt→vdiv(1/root)   │
    │  →vdup(LOWEST)→inv_rms ││  →vdup(LOWEST)→inv_rms │
    │                         ││                         │
    │ Pass 2 (per row):       ││ Pass 2 (per row):       │
    │  vlds x                 ││  vlds x                 │
    │  vlds gamma             ││  vlds gamma             │
    │  vmul inv_rms           ││  vmul inv_rms           │
    │  vmul gamma             ││  vmul gamma             │
    │  vsts output            ││  vsts output            │
    └──────────┬──────────────┘└──────────┬──────────────┘
               │                         │
          MTE3 UB→GM                MTE3 UB→GM
          Output[0:8,:]             Output[8:16,:]

  Final Output: single contiguous FP32[16,256] in GM
```

**GEMM 中间结果全程不经过 GM**。L0C → UB 经过 FIXPIPE 直连。
Residual Add 的结果保留在 UB 中由 RMSNorm 直接消费。

## 使用的 vreg 操作

全部计算使用 VPTO vreg 操作实现，**无 tile compute op（无 OpPipeInterface）**，不触发 ExpandTileOp：

| 操作 | 用途 | VPTOOps.td 定义 |
|------|------|----------------|
| `pto.vcvt {part="EVEN"}` | BF16→FP32 转换 | PTO_VcvtOp |
| `pto.vadd` | GEMM + Residual 加法；vcadd 累加 | PTO_VaddOp |
| `pto.vmul` | x² 平方；inv_rms 广播乘；gamma 乘 | PTO_VmulOp |
| `pto.vcadd` | 64 元素行内归约求和（reduce to lowest lane） | PTO_VcaddOp |
| `pto.vmuls` | 乘标量 1/N | PTO_VmulsOp |
| `pto.vadds` | 加标量 epsilon | PTO_VaddsOp |
| `pto.vsqrt` | 平方根 | PTO_VsqrtOp |
| `pto.vdiv` | 1/root 倒数 | PTO_VdivOp |
| `pto.vdup {position="LOWEST"}` | 从 lane 0 广播 inv_rms 到所有 lane | PTO_VdupOp |
| `pto.vbr` | 广播标量 0.0 初始化累加器；广播 1.0 | PTO_VbrOp |

**不使用**：`pto.trowsum`、`pto.trecip`、`pto.trsqrt`、`pto.tmul`、`pto.trowexpandmul`、
`pto.tcolexpandmul` 等任何 OpPipeInterface tile compute op。

## RMSNorm 算法细节

每个 Vector subblock 含 8 行 × 256 列 FP32。RMSNorm 按行独立计算：

### Pass 1：计算 x 并累加 sum(x²)

```
for row in 0..7:
  acc = vbr(0.0)                          # 64-lane 累加器，仅 lane 0 有效
  for chunk in 0, 64, 128, 192:           # 4 chunks of 64 elements
    gemm     = vlds C2V_UB[row*256+chunk] # GEMM 结果
    res_bf16 = vlds Res_UB[row*256+chunk] {UNPK_B16}  # BF16 残差
    res_f32  = vcvt(res_bf16) {EVEN}      # BF16→FP32
    x        = vadd(gemm, res_f32)        # x = GEMM + residual
    vsts(x, C2V_UB[row*256+chunk])        # 保存 x 供 Pass 2 使用
    sq       = vmul(x, x)                 # x²
    chk_sum  = vcadd(sq)                  # 64→1 归约
    acc      = vadd(acc, chk_sum, one_lane_mask)  # 累加至 lane 0
```

### Scalar Chain

```
mean_sq = vmuls(acc, 1/256, one_lane_mask)   # 除以 N
var     = vadds(mean_sq, 1e-6, one_lane_mask) # 加 epsilon
root    = vsqrt(var, one_lane_mask)            # 平方根
one     = vbr(1.0)
inv_rms = vdiv(one, root, one_lane_mask)       # 1/root
inv_bc  = vdup(inv_rms, full_mask){LOWEST}     # 广播到全部 64 lane
```

### Pass 2：应用 inv_rms × gamma

```
for chunk in 0, 64, 128, 192:
  x      = vlds C2V_UB[row*256+chunk]    # 重新加载 x
  gamma  = vlds Gamma_UB[chunk]           # gamma[chunk:chunk+64]
  normed = vmul(x, inv_bc)                # x × inv_rms
  output = vmul(normed, gamma)            # × gamma
  vsts(output, C2V_UB[row*256+chunk])     # 写入最终结果
```

### chunked_fp32_row_sum 的硬件模拟

golden.py 严格模拟硬件归约顺序：
- 每行分成 4 个 64 元素 chunk；
- 每个 chunk 内使用 FP32 求和；
- 4 个结果按顺序使用 FP32 累加。

## UB 内存布局（每个 subblock）

| 偏移 | 大小 (bytes) | 内容 |
|------|-------------|------|
| 0 | 8,192 | C2V GEMM FP32 tile / x / 最终输出 [8,256] |
| 8,192 | 4,096 | Residual BF16 input [8,256] |
| 12,288 | 1,024 | Gamma FP32 [256] |

总 UB 用量：13,312 bytes（Phase 1 为 28,672 bytes，本阶段移除了 ub_residual_f32 区域）。

## 同步模式

与 Phase 1 完全相同，新增 Gamma 加载复用 MTE2→V 同步链：

```
Vector section:
  1. sync.wait <PIPE_MTE3>            # 等待 C2V 数据
  2. set_flag/wait_flag MTE3→V        # 通知 V pipeline
  3. mte_gm_ub Residual (per subblock) # 加载残差
  4. mte_gm_ub Gamma                  # 加载 gamma（两个 subblock 共享）
  5. set_flag/wait_flag MTE2→V        # 加载完成，通知 V
  6. get_buf / vecscope / rls_buf     # RMSNorm 计算（两遍）
  7. set_flag/wait_flag V→MTE3        # 计算完成，通知 MTE3
  8. get_buf / mte_ub_gm / rls_buf   # 最终输出写回
  9. barrier PIPE_ALL
```

## 测试结果

### 实测（已验证通过）

| 测试用例 | 结果 | Max Abs Error | Max Rel Error | Tick | 门限 |
|---------|------|---------------|---------------|------|------|
| gemm-residual-rmsnorm-c2v-split-m (融合版) | **PASS** | 见 compare.py 输出 | 见 compare.py 输出 | 4,690 | atol=1e-4, rtol=1e-4 |
| gemm-residual-c2v-split-m (Phase 1 回归) | **PASS** | 0.0 | 0.0 | 4,279 | atol=1e-4, rtol=1e-4 |

### 推测（基于分析的预期收益）

- C2V 融合避免了 GEMM 结果的 GM 写+读 round-trip
- 融合版在 UB 内完成 RMSNorm，消除额外的 MTE3/MTE2 传输开销

### 尚未完成

| 项目 | 说明 |
|------|------|
| 公平未融合性能基线 | `test/vpto/cases/kernels/gemm-residual-rmsnorm-no-c2v/` 已添加，待实测对比 |
| msprof 对比 | 融合版 vs 未融合基线的硬件性能计数器对比 |

### 编译命令

```bash
ptoas --pto-arch=a5 --pto-backend=vpto --pto-level=level3 \
  --enable-tile-op-expand --enable-insert-sync \
  kernel.pto -o kernel.fatobj.o
```

### 静态检查

```bash
# Tile compute ops: 应为 0
grep -cE 'pto\.t(rowsum|recip|mul|sqrt|colexpandmul|rowexpandmul)' kernel.pto

# mte_l0c_gm: 应为 0
grep -c 'mte_l0c_gm' kernel.pto

# ExpandTileOp 触发: 应为 0
# （编译日志中无 TileLang daemon 启动消息）
```

## 数值误差分析

- Max absolute error: 9.54e-07（约 1 ULP for FP32）
- Max relative error: 2.37e-07

误差来源：
1. BF16 输入截断（A/B/Residual 从 FP32 golden 值截断为 BF16）
2. GEMM MAD 内部 FP32 累加顺序差异
3. vcadd 64→1 归约的 FP32 累加顺序与 NumPy 不同
4. vdiv/vsqrt 硬件实现的精度极限

误差在 FP32 精度范围内（~1e-7），属于合理的数值噪声。

## 性能分析

### 与 Phase 1 对比

| 指标 | Phase 1 (Residual only) | Phase 2 (+RMSNorm) | Delta |
|------|------------------------|---------------------|-------|
| Total tick | 4,279 | 4,690 | +411 (+9.6%) |
| 操作 | vcvt + vadd | vcvt + vadd + vmul×2 + vcadd + scalar chain | — |

RMSNorm 增加 ~411 ticks（~9.6%），主要在：
- Pass 1: vmul(x²) + vcadd reduce + vadd accumulate per row
- Scalar chain: vmuls + vadds + vsqrt + vdiv + vdup per row
- Pass 2: vlds(x) + vlds(gamma) + vmul×2 per row

### 融合收益估计（待确认）

上述 4,690 ticks 是 **实测值**（GEMM+Residual+RMSNorm 全部融合在单个 kernel 内）。

**推测**：若拆分为「GEMM+Residual C2V → GM」+「GM → UB RMSNorm → GM」两个独立 kernel，
中间 GM round-trip 理论上会引入额外 MTE3 写 + MTE2 读开销。

**尚未完成**：与完整未融合基线（统一 kernel 内，GEMM 中间结果经 GM round-trip，详见
`test/vpto/cases/kernels/gemm-residual-rmsnorm-no-c2v/`）的公平性能对比。
在公平基线测试完成前，不对外宣称具体 tick 节省数字。

## 与 PTODSL / Tile PTO 的关系

本阶段刻意不接入 PTODSL 和 Tile PTO 编译链：

- **不使用** `pto.tload`、`pto.tstore`、`pto.tmatmul` 等 PTODSL op
- **不使用** `ExpandTileOp` pass
- **全部 vreg 操作**（`pto.vcvt`、`pto.vadd`、`pto.vmul`、`pto.vcadd`、`pto.vdiv`、
  `pto.vsqrt`、`pto.vmuls`、`pto.vadds`、`pto.vdup`、`pto.vbr`）
- 内存搬运使用底层 `pto.mte_*` 指令

### 底层 vreg 原型与 Tile PTO 的关系

当前底层 vreg 原型 **不依赖** ExpandTileOp。所有 RMSNorm 计算使用 `pto.v*` 操作
（vreg 级别），不触发任何 OpPipeInterface tile compute pass。

Tile PTO 的标准 module-level `pto.kernel_kind` 输入是可以用文本 IR 表达的（通过
`module attributes {pto.kernel_kind = ...}` 或单一 `pto.kernel` 标记函数内的
`pto.section.cube` / `pto.section.vector` 区域）。本阶段使用的正是这种形式，
无 C++ API 依赖。

### 已知限制

1. **VPTOSplitCVModule 不识别 `pto.entry` + 多个 function-level `pto.kernel_kind` 的
   container 结构**：PTODSL 前端生成的 `pto.entry` 包装函数 + 多个带有
   `pto.kernel_kind = #pto.kernel_kind<cube/vector>` 的子函数结构，尚不能被
   VPTO container 管道规范化。这是独立的前端 container 支持问题，不阻塞底层
   CODA-style vreg 原型。
2. **实验性用例**：相关 Tile dialect 调查已移至
   `test/vpto/experimental/gemm-residual-tile-c2v-split-m/`，不参与自动验证门禁。

## 下一步

- Phase 3：双缓冲（ping-pong）隐藏 C2V→Vector 同步延迟
- Phase 4：完成 `gemm-residual-rmsnorm-no-c2v` 未融合基线的实测对比
- 完成融合版 vs 未融合基线的 msprof 性能对比
- VPTOSplitCVModule 扩展支持 `pto.entry` container 结构后，
  评估 PTODSL tile 操作实现 RMSNorm 的可行性

## 编译器修改

**无。** 未修改 lib/、include/、tools/ 中的任何文件。
