#!/usr/bin/python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from pathlib import Path
import sys

import numpy as np

for search_root in (Path(__file__).resolve().parent, Path(__file__).resolve().parents[1]):
    if (search_root / "validation_runtime.py").is_file():
        sys.path.insert(0, str(search_root))
        break

from validation_runtime import default_buffers, load_case_meta, rng, single_output, write_buffers, write_golden


E4M3_BITS = np.asarray([0x00, 0x30, 0x38, 0x40, 0xB0, 0xB8, 0xC0], dtype=np.uint8)
E5M2_BITS = np.asarray([0x00, 0x38, 0x3C, 0x40, 0xB8, 0xBC, 0xC0], dtype=np.uint8)


def decode_e4m3fn(values):
    values = np.asarray(values, dtype=np.uint8)
    sign = np.where((values & 0x80) != 0, -1.0, 1.0).astype(np.float32)
    exponent = (values >> 3) & 0x0F
    mantissa = values & 0x07
    out = np.zeros(values.shape, dtype=np.float32)
    subnormal = exponent == 0
    normal = (exponent != 0) & (exponent != 0x0F)
    out[subnormal] = mantissa[subnormal].astype(np.float32) * (2.0 ** -9)
    out[normal] = (1.0 + mantissa[normal].astype(np.float32) / 8.0) * np.exp2(exponent[normal].astype(np.int32) - 7)
    out[exponent == 0x0F] = np.nan
    return sign * out


def decode_e5m2(values):
    values = np.asarray(values, dtype=np.uint8)
    sign = np.where((values & 0x80) != 0, -1.0, 1.0).astype(np.float32)
    exponent = (values >> 2) & 0x1F
    mantissa = values & 0x03
    out = np.zeros(values.shape, dtype=np.float32)
    subnormal = exponent == 0
    normal = (exponent != 0) & (exponent != 0x1F)
    out[subnormal] = mantissa[subnormal].astype(np.float32) * (2.0 ** -16)
    out[normal] = (1.0 + mantissa[normal].astype(np.float32) / 4.0) * np.exp2(exponent[normal].astype(np.int32) - 15)
    out[exponent == 0x1F] = np.nan
    return sign * out


def convert_scale_b_format(scale, block_size=16, c0_size_mx=2):
    k, n = scale.shape
    pad_n = (block_size - n % block_size) % block_size
    pad_k = (c0_size_mx - k % c0_size_mx) % c0_size_mx
    if pad_n > 0 or pad_k > 0:
        padded = np.pad(scale, ((0, pad_k), (0, pad_n)), mode="constant", constant_values=0)
    else:
        padded = scale
    k_padded, n_padded = padded.shape
    result = padded.reshape((k_padded // c0_size_mx, c0_size_mx, n_padded // 16, 16)).transpose(2, 0, 3, 1)
    return result.reshape(result.shape[1] * result.shape[3], result.shape[0] * result.shape[2])


def main():
    meta = load_case_meta()
    out_name = single_output(meta)
    if len(meta.inputs) != 4:
        raise ValueError(f"expected 4 input buffers, got {meta.inputs}")
    a_name, b_name, a_scale_name, b_scale_name = meta.inputs
    generator = rng()

    a_bits = generator.choice(E4M3_BITS, size=128).astype(np.uint8)
    b_bits = generator.choice(E5M2_BITS, size=128 * 16).astype(np.uint8)
    a_scale = generator.integers(127, 130, size=(1, 4), dtype=np.uint8)
    b_scale = generator.integers(127, 130, size=(4, 16), dtype=np.uint8)

    buffers = default_buffers(meta)
    buffers[a_name] = a_bits
    buffers[b_name] = b_bits
    buffers[a_scale_name] = a_scale.reshape(-1)

    packed_b_scale = convert_scale_b_format(b_scale).astype(np.uint8).reshape(-1)
    b_scale_buffer = np.zeros(meta.elem_counts[b_scale_name], dtype=meta.np_types[b_scale_name])
    b_scale_buffer[: packed_b_scale.size] = packed_b_scale
    buffers[b_scale_name] = b_scale_buffer
    buffers[out_name] = np.zeros(meta.elem_counts[out_name], dtype=meta.np_types[out_name])
    write_buffers(meta, buffers)

    a = decode_e4m3fn(a_bits).reshape(1, 128)
    b = decode_e5m2(b_bits).reshape(128, 16)
    a_scale_full = np.exp2(a_scale.astype(np.int16) - 127).astype(np.float32)
    b_scale_full = np.exp2(b_scale.astype(np.int16) - 127).astype(np.float32)

    a_real = a * a_scale_full[:, np.arange(128) // 32]
    b_real = b * b_scale_full[np.arange(128) // 32, :]
    golden = np.matmul(a_real.astype(np.float32), b_real.astype(np.float32)).astype(np.float32)

    write_golden(meta, {out_name: golden.reshape(-1)})


if __name__ == "__main__":
    main()
