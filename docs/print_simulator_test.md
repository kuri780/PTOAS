# PTO Print 在 A5 模拟器上的测试

本文记录 `pto.print` 和 `pto.tprint` 在 Ascend A5 模拟器 (`Ascend950PR_950x`) 上的端到端测试过程和结果。

> **范围说明**：本文的详细步骤以标量 `pto.print` 的 EmitC 路径为例。`pto.tprint` 的 VPTO 路径测试用例位于 `test/vpto/cases/kernels/print-*/`（共 8 个 kernel 级用例），模拟器验证脚本位于 `test/vpto/scripts/run_vpto_tprint_validation.sh`。

## 1. 测试对象

`test/samples/Print/print_scalar.py` — 最简单的 scalar print 用例：

```python
# Python DSL 生成 PTO IR
fn = func.FuncOp("print_scalar_kernel", fn_ty)
scalar = entry.arguments[0]
pto.PrintOp("scalar = %+08.3f", scalar)
```

期待行为：A5 kernel 接收一个 `float`，调用 `cce::printf` 打印格式化字符串。

## 2. 端到端链路

```
print_scalar.py
  │  python3 (MLIR Python bindings)
  ▼
PTO IR (.pto)
  │  ptoas --pto-arch=a5 --pto-backend=emitc
  ▼
EmitC C++ (.cpp)
  │  generate_testcase.py → CMake → bisheng -xcce
  ▼
libprint_scalar_kernel.so + print_scalar_sim
  │  msprof op simulator --soc-version=Ascend950PR_950x
  ▼
模拟器执行 → printf 输出
```

## 3. 详细步骤

### 3.1 生成 PTO IR

```bash
PYTHONPATH=build-llvm21/python:<mlir_core路径> \
  python3 test/samples/Print/print_scalar.py
```

产物（注意：裸 `module` 不含 `pto.container`，VPTO 后端不可用。项目已提供 `print_scalar_a5.pto` 替代）：

```mlir
module {
  func.func @print_scalar_kernel(%arg0: f32) attributes {pto.entry} {
    pto.print ins("scalar = %+08.3f", %arg0 : f32)
    return
  }
}
```

### 3.2 PTO → EmitC C++

```bash
ptoas --pto-arch=a5 --pto-backend=emitc --pto-level=level2 \
  test/samples/Print/print_scalar_a5.pto \
  -o print_scalar_kernel.cpp
```

产物（关键部分）：

```cpp
extern "C" __global__ AICORE void print_scalar_kernel(float v1) {
  using T = float;
  cce::printf("scalar = %+08.3f\n", v1);
  return;
}
```

### 3.3 生成模拟器测试工程

```bash
python3 test/npu_validation/scripts/generate_testcase.py \
  --input print_scalar_kernel.cpp \
  --testcase print_scalar \
  --output-root /tmp/ptoas_sim_test \
  --run-mode sim \
  --soc-version Ascend950 \
  --aicore-arch dav-c310-vec
```

自动生成的文件：

| 文件 | 作用 |
|---|---|
| `main.cpp` | Host ACL runner：`aclInit` → 传参 → `LaunchKernel` → `aclrtSynchronizeStream` |
| `launch.cpp` | Kernel 启动封装：`print_scalar_kernel<<<1, nullptr, stream>>>(v1)` |
| `print_scalar_kernel.cpp` | Device 端 kernel 源码 |
| `CMakeLists.txt` | 编译配置（含 `--cce-enable-print` 等关键 flag） |
| `golden.py` / `compare.py` | Golden 生成和比对脚本 |
| `run.sh` | 一键运行脚本 |

### 3.4 编译

```bash
# CMake 配置
cmake -G Ninja -S /tmp/ptoas_sim_test/tmp/print_scalar -B build \
  -DSOC_VERSION=Ascend950PR_950x \
  -DPTO_ISA_ROOT=/home/wenxuekun/pto-isa

# 编译 simulator 目标
cmake --build build --target print_scalar_sim
```

产物：

| 产物 | 说明 |
|---|---|
| `libprint_scalar_kernel.so` | AICORE kernel 共享库 (`bisheng -xcce --cce-fatobj-link`) |
| `print_scalar_sim` | 模拟器可执行文件 (host 端 + 链接 `runtime_camodel`) |

编译时自动注入的关键选项：

```cmake
-xcce                           # CCE 语言模式（自动加载 __cce_half, __bf16 等内建类型）
--cce-enable-print              # 启用 cce::printf
-DPTOAS_ENABLE_CCE_PRINT=1      # include <ccelib/print/print.h>
--cce-aicore-arch=dav-c310-vec  # A5 vector core
-DREGISTER_BASE
-std=c++17                      # PTO 头文件需要 C++17
```

### 3.5 模拟器运行

```bash
# msprof 要求目录权限严格，不能用 /tmp
RUN_DIR="$HOME/ptoas_sim_run"
mkdir -p "$RUN_DIR" && chmod 700 "$RUN_DIR"

source /usr/local/Ascend/cann/set_env.sh

SIM_LIB_DIR="$ASCEND_HOME_PATH/x86_64-linux/simulator/Ascend950PR_950x/lib"
export LD_LIBRARY_PATH="build:$ASCEND_HOME_PATH/lib64:$SIM_LIB_DIR:$LD_LIBRARY_PATH"

cd "$RUN_DIR"

msprof op simulator \
  build/print_scalar_sim \
  --kernel-name=print_scalar_kernel \
  --launch-count=1 \
  --soc-version=Ascend950PR_950x \
  --timeout=120 \
  --output="$RUN_DIR/out"
```

## 4. 运行结果

```
2026-07-21 16:40:44 [INFO] <ProfInit> Start profiling on kernel: print_scalar_kernel
[info] [0000000024] [block_start] : AIV, task_id=0, core_id=0, block_id=0

-----------------------------------------------------------------------------
---------------------------------HiIPU Print---------------------------------
-----------------------------------------------------------------------------
=> Vec 0
scalar = +001.000

[info] [0000011590] [block_end] : AIV, task_id=0, core_id=0, block_id=0

core0.veccore0      5.71 µs (duration)    3.52 µs (running)

2026-07-21 16:42:17 [INFO] Profiling running finished. All task success.
```

✅ `cce::printf("scalar = %+08.3f\n", 1.0f)` → 输出 `scalar = +001.000`

## 5. 踩坑记录

### 5.1 `ccec` 直编失败 — 缺少 AICORE 内建类型

**症状**：`uint8_t` / `__ubuf__` / `bfloat16_t` / `PIPE_FIX` / `float8_e4m3_t` 等大量类型找不到。

**原因**：`ccec --cce-aicore-only -c` 在部分 CANN 版本下走的内部路径不等于 `bisheng -xcce`。后者会自动 `-include __clang_cce_runtime_wrapper.h`，拉入全部 AICORE 内建类型。

**解决**：不要手工调 `ccec`，用 `generate_testcase.py` 生成的 CMake 工程，它已正确配置 `-xcce` language mode。

### 5.2 C++ 标准版本

**症状**：`if constexpr`、`std::is_same_v`、`auto` 模板参数报错。

**解决**：CMakeLists 里已配 `-std=c++17`。PTO 头文件全面使用 C++17 特性。

### 5.3 `main.cpp` 中 `MrgSortExecutedNumList` 重复定义

**症状**：
```
error: redefinition of 'MrgSortExecutedNumList'
main.cpp:52:8 vs pto-isa/include/pto/common/type.hpp:521:8
```

**原因**：`generate_testcase.py` 生成的 `main.cpp` 带有 `#ifndef TMRGSORT_HPP` 守卫的 fallback 定义，但当前 pto-isa 版本已在 `type.hpp` 中提供了该类型。

**解决**：删除 `main.cpp` 中的 fallback 定义。

### 5.4 msprof 目录权限检查

**症状**：
```
[ERROR] /tmp is writable by any other users or group users.
```

**原因**：`msprof` 安全检查要求工作目录和输出目录不能是 other/group writable。

**解决**：在 `$HOME` 下创建 `chmod 700` 的专用目录。

### 5.5 SOC 版本不匹配

**症状**：
```
[warning] Core does not support AIV task
terminate called after throwing an instance of 'std::out_of_range'
```

**原因**：`dav_3102` 的模拟器配置对应 `Ascend610Lite`，不是 A5 的 vector core。

**解决**：改用 Ascend950 系列 SOC 版本（如 `Ascend950PR_950x`），其配置文件正确对接到 `dav-c310-vec`。

## 6. 关键文件

| 文件 | 作用 |
|---|---|
| `test/samples/Print/print_scalar.py` | Python DSL 入口 |
| `test/samples/Print/print_scalar_a5.pto` | A5 容器 PTO IR |
| `test/samples/Print/print_scalar.cpp` | EmitC 参考输出 |
| `test/npu_validation/scripts/generate_testcase.py` | 测试工程生成器 |
| `test/npu_validation/templates/main_template.cpp` | Host runner 模板 |
| `docs/msprof_op_simulator_usage_zh.md` | 模拟器使用文档 |

## 7. 结论

- `pto.print` → `cce::printf` 的 EmitC Lowering 链路在 A5 模拟器上验证通过。标量 `float` 参数能正确格式化输出，CCE printf 头文件 (`ccelib/print/print.h`) 和 `--cce-enable-print` 编译选项配合正常。
- `pto.tprint` → 内联 DebugTunnel 协议的 VPTO Lowering 已实现，包含 8 个 kernel 级端到端测试用例（覆盖 tile 打印、与向量计算/控制流/多 block 共存等场景），模拟器验证脚本位于 `test/vpto/scripts/run_vpto_tprint_validation.sh`。
