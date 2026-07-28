#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd. SPDX-License-Identifier: CANN-OSL-2.0
# case: kernels/print-all-features — full integration
# Kernel computes output[i] = input[i] * 2 (vector self-add).
# Golden = input * 2.0 for all 128 elements.
import argparse, numpy as np; from pathlib import Path
ELEM_COUNT, SEED, SCALE = 128, 999, 2.0
def generate(d: Path, seed: int):
    rng = np.random.default_rng(seed)
    inp = rng.uniform(-4.0, 4.0, size=(ELEM_COUNT,)).astype(np.float32)
    golden = (inp * SCALE).astype(np.float32)
    d.mkdir(parents=True, exist_ok=True)
    inp.tofile(d / "v1.bin")
    golden.tofile(d / "golden_v2.bin")
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", type=Path, default=Path("."))
    p.add_argument("--seed", type=int, default=SEED)
    generate(p.parse_args().output_dir, p.parse_args().seed)
if __name__ == "__main__": main()
