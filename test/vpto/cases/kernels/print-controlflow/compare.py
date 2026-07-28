#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: CANN-OSL-2.0
# case: kernels/print-controlflow
import os, sys, numpy as np

def compare(golden_path, output_path, dtype, eps):
    if not os.path.exists(output_path): print(f"[ERROR] Output missing: {output_path}"); return False
    if not os.path.exists(golden_path): print(f"[ERROR] Golden missing: {golden_path}"); return False
    g = np.fromfile(golden_path, dtype=np.dtype(dtype)); o = np.fromfile(output_path, dtype=np.dtype(dtype))
    if g.shape != o.shape: print(f"[ERROR] Shape: {g.shape} vs {o.shape}"); return False
    if not np.allclose(g, o, atol=eps, rtol=eps, equal_nan=True):
        d = np.abs(g.astype(np.float64) - o.astype(np.float64)); idx = int(np.argmax(d))
        print(f"[ERROR] Mismatch at {idx}: golden={float(g[idx])} out={float(o[idx])} diff={float(d[idx])}"); return False
    return True

def main():
    ok = compare("golden_v2.bin", "v2.bin", np.float32, 1e-6)
    if not ok: print("[ERROR] compare failed"); sys.exit(2)
    print("[INFO] compare passed")

if __name__ == "__main__": main()
