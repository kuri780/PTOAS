# Phase 1: GEMM + Residual C2V Split-M Baseline (English)

## Overview

Establish a stable direct-VPTO correctness baseline for BF16 GEMM with FP32
Residual Add, using Cube-to-Vector (C2V) Split-M data movement on Ascend 950
(A5).  No PTODSL, no double-buffering, no RMSNorm — a single-buffered,
two-subblock prototype.

**Math**: `Output[M,N] = bf16(A[M,K]) @ bf16(B[K,N]) + bf16(Residual[M,N])`

**Shapes**: M=16, K=64, N=256 (static)

**Constraint**: The GEMM FP32 result must **not** be written to GM before the
Residual Add.  No `pto.mte_l0c_gm` is allowed for intermediate GEMM storage.

## Environment

| Item | Value |
|------|-------|
| Server | x86_64 Linux, CANN 9.0.0-beta.1 |
| Compiler | bisheng (clang 15.0.5, 2026-02-04) |
| Simulator | dav_3510 (Ascend950PR_9599 config) |
| msprof | Ascend950PR_9599 SoC |
| pto-isa | N/A (direct VPTO fatobj) |
| git rev | 7bdf3b464fcb423564dd15afd59456b30572fdda |

## Files Created

```
test/vpto/cases/kernels/gemm-residual-c2v-split-m/
├── kernel.pto      # PTO IR: Cube MAD + C2V split_m + Vector vcvt/vadd
├── golden.py       # NumPy golden: BF16 GEMM + Residual → FP32
├── compare.py      # FP32 comparison with first-error and max-error reporting
├── launch.cpp      # ACL launch wrapper (two subblock output pointers)
└── main.cpp        # ACL host runner, single contiguous output buffer
```

## Data Flow

```
[Cube Pipeline]
  BF16 A[16,64] GM ──MTE2──▶ L1 ──MTE1──▶ L0A
  BF16 B[64,256] GM ──MTE2──▶ L1 ──MTE1──▶ L0B (transpose)
      │                                    │
      └──────────── MMAD ──────────────────┘
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
    ┌────┴────┐              ┌────┴────┐
    │MTE2 load│              │MTE2 load│
    │BF16 Res │              │BF16 Res │
    │[8,256]  │              │[8,256]  │
    └────┬────┘              └────┬────┘
         │                         │
    vcvt BF16→FP32           vcvt BF16→FP32
         │                         │
    VADD GEMM+Res            VADD GEMM+Res
         │                         │
    MTE3 UB→GM               MTE3 UB→GM
    Output[0:8,:]            Output[8:16,:]

  Final Output: single contiguous FP32[16,256] in GM
```

**No GEMM intermediate data ever touches GM.**  The L0C → UB transfer goes
through the FIXPIPE directly.

## Critical Synchronization Pattern

The most important finding was that `pto.section.vector` requires explicit
cross-pipeline flag synchronization when performing Vector compute operations
(vcvt, vadd) on C2V data:

```
pto.section.vector {
  %subblock = pto.get_subblock_idx

  // 1. Wait for C2V data in UB
  pto.sync.wait <PIPE_MTE3>, 1

  // 2. Notify V pipeline that data is ready
  pto.set_flag["PIPE_MTE3", "PIPE_V", "EVENT_ID0"]
  pto.wait_flag["PIPE_MTE3", "PIPE_V", "EVENT_ID0"]

  // 3. MTE2: load residual from GM
  scf.if %is_subblock0 {
    pto.mte_gm_ub %residual_gm_0, %ub_residual_bf16, ...
  }
  scf.if %is_subblock1 {
    pto.mte_gm_ub %residual_gm_1, %ub_residual_bf16, ...
  }

  // 4. MTE2→V: residual load complete
  pto.set_flag["PIPE_MTE2", "PIPE_V", "EVENT_ID0"]
  pto.wait_flag["PIPE_MTE2", "PIPE_V", "EVENT_ID0"]

  // 5. Vector compute
  pto.get_buf "PIPE_V", 0, 0
  pto.get_buf "PIPE_V", 1, 0
  pto.get_buf "PIPE_V", 2, 0
  pto.vecscope {
    scf.for %offset = ... {
      %mask, %next = pto.plt_b32 %remaining
      %bf16 = pto.vlds %ub_residual_bf16[%offset] {dist = "UNPK_B16"}
      %f32  = pto.vcvt %bf16, %cvt_mask {part = "EVEN"}
      %gemm = pto.vlds %ub_c2v[%offset]
      %sum  = pto.vadd %gemm, %f32, %mask
      pto.vsts %sum, %ub_c2v[%offset], %mask
    }
  }
  pto.rls_buf "PIPE_V", 0, 0
  pto.rls_buf "PIPE_V", 1, 0
  pto.rls_buf "PIPE_V", 2, 0

  // 6. V→MTE3: vector compute complete
  pto.set_flag["PIPE_V", "PIPE_MTE3", "EVENT_ID0"]
  pto.wait_flag["PIPE_V", "PIPE_MTE3", "EVENT_ID0"]

  // 7. MTE3: write output to GM
  pto.get_buf "PIPE_MTE3", 2, 0
  scf.if %is_subblock0 {
    pto.mte_ub_gm %ub_c2v, %out_gm_0, ...
  }
  scf.if %is_subblock1 {
    pto.mte_ub_gm %ub_c2v, %out_gm_1, ...
  }
  pto.rls_buf "PIPE_MTE3", 2, 0
  pto.barrier #pto.pipe<PIPE_ALL>
}
```

**Without the MTE3→V and V→MTE3 flag pairs (steps 2 and 6), Vector compute
operations silently become no-ops in the split Vector kernel.**  This was
confirmed through binary search debugging: vlds+vsts (copy) works, but any
operation involving vadd becomes a no-op without the cross-pipeline flags.

The `pto.sync.wait <PIPE_MTE3>` notifies the MTE3 pipeline but does **not**
notify the V pipeline.  A separate flag chain is required to wake up Vector
compute.

## UB Memory Layout (per subblock)

| Offset | Size (bytes) | Content |
|--------|-------------|---------|
| 0 | 8,192 | C2V GEMM FP32 tile [8,256] |
| 8,192 | 4,096 | Residual BF16 input [8,256] |
| 20,480 | 8,192 | Residual FP32 converted [8,256] |

Total UB per subblock: 28,672 bytes.

## Test Results

| Test | Result | Max Error |
|------|--------|-----------|
| gemm-residual-c2v-split-m (new) | **PASS** | 0.0 |
| mad_bf16bf16f32 | **PASS** | — |
| fixpipe-acc-store-dual-ub-cv | **PASS** | — |
| binary-vector/vadd | **PASS** | — |

### Validation Command

```bash
source /usr/local/CANN/cann/set_env.sh
WORK_SPACE=/tmp/ptoas-gemm-residual-c2v \
ASCEND_HOME_PATH=/usr/local/CANN/cann-9.0.0-beta.1 \
PTOAS_BIN=$PWD/build/tools/ptoas/ptoas \
DEVICE=SIM \
CASE_NAME=kernels/gemm-residual-c2v-split-m \
bash test/vpto/scripts/run_host_vpto_validation.sh
```

### Static Checks

```bash
# Confirm no mte_l0c_gm
rg -n "mte_l0c_gm" test/vpto/cases/kernels/gemm-residual-c2v-split-m/kernel.pto
# → only found in the comment header (line 24), not in code

# Confirm required ops present
rg -n "mte_l0c_ub|dst_mode\(split_m\)|sync.set|sync.wait|get_subblock_idx|vadd|vcvt" \
  test/vpto/cases/kernels/gemm-residual-c2v-split-m/kernel.pto
# → all present
```

## msprof Profiling

### Collect

```bash
msprof op simulator \
  --application="$OUT_DIR/kernels_gemm-residual-c2v-split-m" \
  --kernel-name="gemm_residual_c2v_split_m_kernel" \
  --launch-count=1 \
  --soc-version="Ascend950PR_9599" \
  --timeout=120 \
  --output="$COLLECT_DIR/out"
```

### Export

```bash
msprof op simulator \
  --export="$OPPROF_DIR/dump" \
  --output="$EXPORT_ROOT"
```

### Cycle Breakdown (Simulator, Not Hardware)

| Phase | Core | Duration (cycles) |
|-------|------|-------------------|
| GM→L1 A+B | Cube/MTE2 | ~1,756 |
| L1→L0A/L0B | Cube/MTE1 | ~161 |
| **BF16 MMAD** | Cube/CUBE | **89** |
| **FIXPIPE C2V** | Cube/FIXP | **115** |
| Sync wait (C2V) | Vector/MTE3 | 2,627 (wall) |
| Residual load | Vector/MTE2 | 611 (parallel with Cube) |
| **BF16→FP32 vcvt** | Vector/RVECEX | **224** (32 iterations) |
| **FP32 VADD** | Vector/RVECEX | **224** (32 iterations) |
| Vector store | Vector/RVECST | **379** (32 iterations) |
| UB→GM output | Vector/MTE3 | 838 |
| **Total** | — | **4,292** |

### Core Timing (msprof report)

| Core | Duration (µs) | Running (µs) |
|------|--------------|--------------|
| core0.cubecore0 | 1.40 | 1.40 |
| core0.veccore0 | 1.95 | 1.95 |
| core0.veccore1 | 1.92 | 1.92 |

### Key Observations

1. **No L0C→GM instruction** present in Cube core CSV — constraint satisfied.
2. **C2V→Vector sync gap**: ~514 cycles between FIXPIPE completion (cycle ~2,113)
   and first Vector compute (cycle ~2,627).  This is inter-core sync latency.
3. **Residual load overlaps** with Cube section: MTE2 residual load completes
   at cycle 611 while C2V data arrives at cycle ~2,113.
4. **FP32 vcvt+vadd are fused in a single loop** (32 iterations), sharing the
   plt_b32 mask.
5. **All GM traffic confirmed correct**: Residual GM→UB via MTE2, Output UB→GM
   via MTE3.  No intermediate GEMM GM traffic.

### Export Artifacts

| File | Size (bytes) |
|------|-------------|
| simulator/trace.json | 154,428 |
| simulator/visualize_data.bin | 215,724 |
| core0.cubecore0_instr_exe_*.csv | 7,328 |
| core0.veccore0_instr_exe_*.csv | 6,883 |
| core0.veccore1_instr_exe_*.csv | 6,871 |

### Profile Artifact Locations

- Profile summary: `/tmp/ptoas-msprof-gemm-residual/profile_summary.md`
- Export root: `/tmp/ptoas-msprof-gemm-residual/insight_export/`
- Collect output: `/tmp/ptoas-msprof-gemm-residual/out/`

## Fused vs. Non-Fused Comparison (msprof Measured)

### Fused (C2V) — `gemm-residual-c2v-split-m`

```
GEMM L0C ──FIXPIPE──▶ UB ──vadd──▶ UB ──MTE3──▶ Output GM
                                   ▲
Residual GM ──MTE2──▶ UB ──vcvt──▶┘
```

- **Total cycles**: **4,292**
- **Vector core time**: 1.95 µs (veccore0), 1.92 µs (veccore1)
- **GM traffic**: A+B BF16 read + Residual BF16 read + Output FP32 write
  = 6,144 + 32,768 + 8,192 + 8,192 = 55,296 bytes _(~54 KB)_
- **Intermediate GEMM on GM**: **None** ✓
- **Vector section breakdown**: sync.wait(2627), residual MTE2(611),
  vcvt+vadd+vsts(~827), output MTE3(838)

### Non-Fused (GM round-trip) — `gemm-residual-no-c2v`

```
GEMM L0C ──C2V──▶ UB ──MTE3──▶ GM (temp)
                                    │
GM (temp) ──MTE2──▶ UB ──vadd──▶ UB ──MTE3──▶ Output GM
                                    ▲
Residual GM ──MTE2──▶ UB ──vcvt────▶┘
```

- **Total cycles**: **6,054** (+1,762, **+41%** vs fused)
- **Vector core time**: 2.92 µs (veccore0), 2.78 µs (veccore1)
- **GM traffic**: all of the above + GEMM temp write (8,192) + GEMM temp
  read (8,192) = **+16,384 bytes** (_total ~72 KB, +30%_)
- **Intermediate GEMM on GM**: **YES** ✗ (violates constraint)
- **Vector section breakdown**: sync.wait(2519), MTE3 GEMM write(585),
  MTE3→MTE2 flag(282), MTE2 GEMM reload(1166), MTE2 residual load(1306),
  vcvt+vadd+vsts(~827), output MTE3(827)

### Measured Overhead Breakdown

| Overhead Source | Cycles |
|-----------------|--------|
| MTE3 GEMM temp write (UB→GM) | +585 |
| MTE3→MTE2 cross-pipeline flag | +282 |
| MTE2 GEMM reload (GM→UB) | +555 |
| Additional barrier/sync | +340 |
| **Total GM round-trip overhead** | **+1,762 (+41%)** |

### Key Differences

| Metric | Fused (C2V) | Non-Fused (GM) | Delta |
|--------|------------|----------------|-------|
| GEMM→Residual path | FIXPIPE UB→UB | C2V→UB→GM→UB | — |
| Total cycles | **4,292** | **6,054** | **+41%** |
| Vector core time | 1.95 µs | 2.92 µs | **+50%** |
| GM traffic | ~55 KB | ~72 KB | **+30%** |
| MTE3 operations | 1 | 2 | +1 |
| MTE2 operations | 1 | 2 | +1 |
| GEMM on GM intermediate | No | Yes | — |

### Conclusion

The C2V fused version eliminates the GM round-trip for the GEMM intermediate
result, saving **1,762 simulator cycles (41%)** and **16 KB of GM bandwidth
(30%)** compared to the non-fused approach.  This confirms the architectural
benefit of FIXPIPE C2V for fusing GEMM with element-wise post-processing on
Ascend 950.

## PTOAS Compiler Modifications

**None.** No compiler code was changed. The kernel uses only existing VPTO
instructions and the split-CV module pass (`VPTOSplitCVModule`).  The
cross-pipeline flag requirement is a usage pattern, not a compiler defect.

## Next-Phase Risks (Tile PTO / tpush-tpop)

1. **Inter-core sync gap** (514 cycles): Tile PTO double-buffering will hide
   this gap by allowing Cube to work on tile N+1 while Vector processes tile N.
2. **Single buffering**: No compute/compute overlap.  Double-buffering is
   required for throughput.
3. **Static shapes**: M=16, K=64, N=256 are hard-coded.  Tile PTO handles
   dynamic tile sizes through `tpush`/`tpop`.
4. **VecScope in split kernels**: The explicit flag pattern (MTE3→V, V→MTE3)
   is undocumented and should be added to developer guides.
5. **No RMSNorm / no activation**: These will be added in later phases.  The
   FIXPIPE pipeline supports `pre_relu` and `pre_quant` inline, which may be
   usable for activation fusion.
