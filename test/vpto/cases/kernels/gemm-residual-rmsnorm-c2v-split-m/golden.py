#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under CANN Open Software License Agreement Version 2.0.
# case: kernels/gemm-residual-rmsnorm-c2v-split-m
# family: gemm-residual
# shapes: A[16,64] B[64,256] Residual[16,256] Gamma[256] Output[16,256]

import argparse
from pathlib import Path
import numpy as np

M = 16
N = 256
K = 64
EPSILON = 1e-6


def to_bf16_bits(values: np.ndarray) -> np.ndarray:
    f32 = values.astype(np.float32, copy=False)
    return (f32.view(np.uint32) >> 16).astype(np.uint16)


def bf16_bits_to_f32(bits: np.ndarray) -> np.ndarray:
    return (bits.astype(np.uint32) << 16).view(np.float32)


def chunked_row_sum_sq(x: np.ndarray) -> np.ndarray:
    """Simulate hardware: 4 chunks of 64 elements per row, FP32 accumulate."""
    rows = x.shape[0]
    result = np.zeros(rows, dtype=np.float32)
    for r in range(rows):
        acc = np.float32(0.0)
        for c_start in range(0, N, 64):
            chunk = x[r, c_start:c_start + 64].astype(np.float32)
            chunk_sum = np.sum(chunk)  # FP32 summation
            acc = np.float32(acc + chunk_sum)
        result[r] = acc
    return result


def generate(output_dir: Path) -> None:
    # Deterministic but diverse patterns
    row_m = np.arange(M, dtype=np.float32).reshape(M, 1)
    col_k = np.arange(K, dtype=np.float32).reshape(1, K)
    a_f32 = (((row_m * 5 + col_k * 3) % 23) - 11) / 8.0

    k_idx = np.arange(K, dtype=np.float32).reshape(K, 1)
    n_idx = np.arange(N, dtype=np.float32).reshape(1, N)
    b_f32 = (((k_idx * 2 - n_idx * 7) % 29) - 14) / 9.0

    m_idx_r = np.arange(M, dtype=np.float32).reshape(M, 1)
    n_idx_r = np.arange(N, dtype=np.float32).reshape(1, N)
    residual_f32 = (((m_idx_r * 3 + n_idx_r * 11) % 17) - 8) / 4.0

    # Gamma: deterministic pattern with positive values
    gamma_f32 = (((np.arange(N, dtype=np.float32) * 7 + 3) % 13) + 2) / 5.0

    # BF16 quantization
    a_bf16 = to_bf16_bits(a_f32)
    b_bf16 = to_bf16_bits(b_f32)
    residual_bf16 = to_bf16_bits(residual_f32)

    a_f32_q = bf16_bits_to_f32(a_bf16).astype(np.float32)
    b_f32_q = bf16_bits_to_f32(b_bf16).astype(np.float32)
    residual_f32_q = bf16_bits_to_f32(residual_bf16).astype(np.float32)

    # GEMM (FP32)
    gemm_f32 = a_f32_q @ b_f32_q  # [16,256]

    # x = GEMM + residual
    x_f32 = gemm_f32 + residual_f32_q  # [16,256]

    # RMSNorm per row
    sq = x_f32.astype(np.float32) * x_f32.astype(np.float32)  # x^2
    sum_sq = chunked_row_sum_sq(sq)  # [16] — chunked FP32 accumulation
    mean_sq = sum_sq / np.float32(N)
    variance = mean_sq + np.float32(EPSILON)
    root = np.sqrt(variance.astype(np.float32))
    inv_rms = np.float32(1.0) / root  # [16]

    # Apply RMSNorm + gamma
    output = x_f32.astype(np.float32) * inv_rms[:, np.newaxis] * gamma_f32[np.newaxis, :]
    output = output.astype(np.float32)

    # Write binary files
    output_dir.mkdir(parents=True, exist_ok=True)

    # v1.bin: A [M,K] BF16
    a_bf16.reshape(-1).tofile(output_dir / "v1.bin")
    # v2.bin: B [K,N] BF16
    b_bf16.reshape(-1).tofile(output_dir / "v2.bin")
    # v3.bin: Residual [M,N] BF16
    residual_bf16.reshape(-1).tofile(output_dir / "v3.bin")
    # v4.bin: Gamma [N] FP32
    gamma_f32.astype(np.float32).reshape(-1).tofile(output_dir / "v4.bin")
    # v5.bin: Output placeholder [M,N] FP32 (zeros)
    np.zeros((M, N), dtype=np.float32).reshape(-1).tofile(output_dir / "v5.bin")
    # golden_v5.bin: Expected Output [M,N] FP32
    output.reshape(-1).tofile(output_dir / "golden_v5.bin")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    args = parser.parse_args()
    generate(args.output_dir)


if __name__ == "__main__":
    main()
