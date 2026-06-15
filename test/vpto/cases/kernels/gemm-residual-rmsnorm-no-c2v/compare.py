#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under CANN Open Software License Agreement Version 2.0.

import os
import sys
import numpy as np


def compare_bin(golden_path: str, output_path: str, label: str = "",
                atol: float = 1e-4, rtol: float = 1e-4) -> bool:
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
    # Avoid div-by-zero: use abs(golden) safety threshold
    nonzero = np.abs(golden) > 0.0
    rel_err = np.where(nonzero, abs_diff / np.abs(golden), 0.0)
    max_rel_err = float(np.max(rel_err))
    print(f"[INFO] {label}max abs error = {max_err:.6e} at idx={max_idx}")
    print(f"[INFO]   golden[{max_idx}] = {float(golden[max_idx]):.6f}")
    print(f"[INFO]   output[{max_idx}] = {float(output[max_idx]):.6f}")
    print(f"[INFO]   max relative error = {max_rel_err:.6e}")
    if np.allclose(golden, output, atol=atol, rtol=rtol):
        return True
    # Report error distribution before failing
    threshold = atol + rtol * np.abs(golden)
    exceed = abs_diff > threshold
    num_exceed = int(np.sum(exceed))
    print(f"[ERROR] {label}{num_exceed} / {golden.size} elements exceed atol={atol}, rtol={rtol}")
    if num_exceed > 0:
        exceed_indices = np.where(exceed)[0]
        # Show up to 10 worst offenders
        show_n = min(10, num_exceed)
        # Sort by absolute error descending among exceeded
        exceed_abs = abs_diff[exceed]
        order = np.argsort(exceed_abs)[::-1][:show_n]
        print(f"[ERROR] worst {show_n} mismatches:")
        for rank, o in enumerate(order):
            i = exceed_indices[o]
            print(f"[ERROR]   #{rank+1} idx={i}: golden={float(golden[i]):.6e}"
                  f" output={float(output[i]):.6e}"
                  f" abs_err={float(abs_diff[i]):.6e}"
                  f" rel_err={float(rel_err[i]):.6e}")
        # Error distribution summary
        print(f"[ERROR] error distribution (all elements):")
        print(f"[ERROR]   abs_err: min={float(np.min(abs_diff)):.6e}"
              f" 25%={float(np.percentile(abs_diff, 25)):.6e}"
              f" 50%={float(np.percentile(abs_diff, 50)):.6e}"
              f" 75%={float(np.percentile(abs_diff, 75)):.6e}"
              f" max={max_err:.6e}")
        print(f"[ERROR]   rel_err: min={float(np.min(rel_err)):.6e}"
              f" 25%={float(np.percentile(rel_err, 25)):.6e}"
              f" 50%={float(np.percentile(rel_err, 50)):.6e}"
              f" 75%={float(np.percentile(rel_err, 75)):.6e}"
              f" max={max_rel_err:.6e}")
    return False


def main() -> None:
    strict = os.getenv("COMPARE_STRICT", "1") != "0"
    ok = compare_bin("golden_v5.bin", "v5.bin", label="RMSNorm(no-c2v): ")
    if not ok:
        if strict:
            print("[ERROR] compare failed")
            sys.exit(2)
        print("[WARN] compare failed (non-gating)")
        return
    print("[INFO] compare passed")


if __name__ == "__main__":
    main()
