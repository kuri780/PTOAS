#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import argparse
from pathlib import Path

import numpy as np

LANES = 2
OPS_PER_LANE = 2
ELEMS = LANES * OPS_PER_LANE * 6  # 3 regions × 2 vectors × 2 floats = 12 vectors = 24 floats


def generate(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    v1 = np.full(ELEMS, -1.0, dtype=np.float32)
    golden_v1 = np.full(ELEMS, -1.0, dtype=np.float32)

    for lane in range(LANES):
        base = lane * OPS_PER_LANE  # vector offset within each region
        # A region: vectors 0..3
        a0_idx = base * 2       # float index for A[base]
        a1_idx = (base + 1) * 2
        # B region: vectors 4..7
        b0_idx = (base + 4) * 2
        b1_idx = (base + 5) * 2
        # C region: vectors 8..11
        c0_idx = (base + 8) * 2
        c1_idx = (base + 9) * 2

        # Fill input A with known values
        a0_vals = np.array([1.0 + lane * 10, 2.0 + lane * 10], dtype=np.float32)
        a1_vals = np.array([3.0 + lane * 10, 4.0 + lane * 10], dtype=np.float32)

        # Fill input B with known values
        b0_vals = np.array([0.5 + lane, 1.5 + lane], dtype=np.float32)
        b1_vals = np.array([2.5 + lane, 3.5 + lane], dtype=np.float32)

        v1[a0_idx : a0_idx + 2] = a0_vals
        v1[a1_idx : a1_idx + 2] = a1_vals
        v1[b0_idx : b0_idx + 2] = b0_vals
        v1[b1_idx : b1_idx + 2] = b1_vals

        # Copy inputs to golden
        golden_v1[a0_idx : a0_idx + 2] = a0_vals
        golden_v1[a1_idx : a1_idx + 2] = a1_vals
        golden_v1[b0_idx : b0_idx + 2] = b0_vals
        golden_v1[b1_idx : b1_idx + 2] = b1_vals

        # Golden output: C = A + B
        golden_v1[c0_idx : c0_idx + 2] = a0_vals + b0_vals
        golden_v1[c1_idx : c1_idx + 2] = a1_vals + b1_vals

    v1.tofile(output_dir / "v1.bin")
    golden_v1.tofile(output_dir / "golden_v1.bin")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    args = parser.parse_args()
    generate(args.output_dir)


if __name__ == "__main__":
    main()
