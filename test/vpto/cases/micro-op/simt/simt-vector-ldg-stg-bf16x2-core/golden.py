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
ELEMS = LANES * OPS_PER_LANE * 6  # 3 regions × 2 vectors × 2 bf16s = 12 vectors = 24 bf16s


def f32_to_bf16_bits(v):
    """Round f32 to bf16 and return uint16 bit pattern."""
    w = v.astype(np.float32, copy=False).view(np.uint32)
    r = np.uint32(0x7FFF) + ((w >> 16) & np.uint32(1))
    return ((w + r) >> 16).astype(np.uint16)


def bf16_bits_to_f32(b):
    """Convert uint16 bf16 bit pattern back to f32."""
    return (b.astype(np.uint32) << 16).view(np.float32)


def generate(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    sentinel = np.uint16(0x7C00)  # bf16 -inf sentinel
    v1 = np.full(ELEMS, sentinel, dtype=np.uint16)
    golden_v1 = np.full(ELEMS, sentinel, dtype=np.uint16)

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

        a0_bits = f32_to_bf16_bits(a0_vals)
        a1_bits = f32_to_bf16_bits(a1_vals)
        b0_bits = f32_to_bf16_bits(b0_vals)
        b1_bits = f32_to_bf16_bits(b1_vals)

        v1[a0_idx : a0_idx + 2] = a0_bits
        v1[a1_idx : a1_idx + 2] = a1_bits
        v1[b0_idx : b0_idx + 2] = b0_bits
        v1[b1_idx : b1_idx + 2] = b1_bits

        # Copy inputs to golden
        golden_v1[a0_idx : a0_idx + 2] = a0_bits
        golden_v1[a1_idx : a1_idx + 2] = a1_bits
        golden_v1[b0_idx : b0_idx + 2] = b0_bits
        golden_v1[b1_idx : b1_idx + 2] = b1_bits

        # Golden output: C = A + B (compute in f32, then round to bf16)
        c0_vals = a0_vals + b0_vals
        c1_vals = a1_vals + b1_vals
        golden_v1[c0_idx : c0_idx + 2] = f32_to_bf16_bits(c0_vals)
        golden_v1[c1_idx : c1_idx + 2] = f32_to_bf16_bits(c1_vals)

    v1.tofile(output_dir / "v1.bin")
    golden_v1.tofile(output_dir / "golden_v1.bin")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    args = parser.parse_args()
    generate(args.output_dir)


if __name__ == "__main__":
    main()
