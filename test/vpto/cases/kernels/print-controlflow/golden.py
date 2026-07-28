#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: CANN-OSL-2.0
# case: kernels/print-controlflow
# Kernel is identity passthrough (GM→UB→GM) with conditional print.
# Golden output = input data.
import argparse, numpy as np
from pathlib import Path

ELEM_COUNT, SEED = 64, 123

def generate(output_dir: Path, seed: int) -> None:
    rng = np.random.default_rng(seed)
    inp = rng.uniform(-5.0, 5.0, size=(ELEM_COUNT,)).astype(np.float32)
    output_dir.mkdir(parents=True, exist_ok=True)
    inp.tofile(output_dir / "v1.bin")
    inp.tofile(output_dir / "golden_v2.bin")

def main():
    p = argparse.ArgumentParser(); p.add_argument("--output-dir", type=Path, default=Path(".")); p.add_argument("--seed", type=int, default=SEED)
    args = p.parse_args(); generate(args.output_dir, args.seed)

if __name__ == "__main__": main()
