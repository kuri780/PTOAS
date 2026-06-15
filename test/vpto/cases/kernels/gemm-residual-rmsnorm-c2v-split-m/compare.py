#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under CANN Open Software License Agreement Version 2.0.

import os
import sys
import numpy as np


def compare_bin(golden_path: str, output_path: str, label: str = "") -> bool:
    if not os.path.exists(golden_path) or not os.path.exists(output_path):
        print(f"[ERROR] missing file: golden={golden_path} output={output_path}")
        return False
    golden = np.fromfile(golden_path, dtype=np.float32)
    output = np.fromfile(output_path, dtype=np.float32)
    if golden.shape != output.shape:
        print(f"[ERROR] shape mismatch: golden={golden.shape} output={output.shape}")
        return False
    abs_diff = np.abs(golden - output)
    max_err = float(np.max(abs_diff))
    max_idx = int(np.argmax(abs_diff))
    rel_err = np.where(np.abs(golden) > 0, abs_diff / np.abs(golden), 0.0)
    max_rel_err = float(np.max(rel_err))
    print(f"[INFO] {label}max abs error = {max_err:.6e} at idx={max_idx}")
    print(f"[INFO]   golden[{max_idx}] = {float(golden[max_idx]):.6f}")
    print(f"[INFO]   output[{max_idx}] = {float(output[max_idx]):.6f}")
    print(f"[INFO]   max relative error = {max_rel_err:.6e}")
    if np.allclose(golden, output, atol=1e-2, rtol=1e-2):
        return True
    diff = np.where(abs_diff > (1e-2 + 1e-2 * np.abs(golden)))[0]
    idx = int(diff[0]) if diff.size else 0
    print(f"[ERROR] first mismatch at idx={idx}: golden={float(golden[idx])}, out={float(output[idx])}")
    return False


def main() -> None:
    strict = os.getenv("COMPARE_STRICT", "1") != "0"
    ok = compare_bin("golden_v5.bin", "v5.bin", label="RMSNorm: ")
    if not ok:
        if strict:
            print("[ERROR] compare failed")
            sys.exit(2)
        print("[WARN] compare failed (non-gating)")
        return
    print("[INFO] compare passed")


if __name__ == "__main__":
    main()
