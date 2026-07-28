#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd. SPDX-License-Identifier: CANN-OSL-2.0
# case: kernels/print-vector-ops — compare output*2x = golden*2x
import os, sys, numpy as np
def compare(gp, op, dt, eps):
    if not os.path.exists(op): print(f"[ERROR] missing {op}"); return False
    if not os.path.exists(gp): print(f"[ERROR] missing {gp}"); return False
    g=np.fromfile(gp,dtype=np.dtype(dt)); o=np.fromfile(op,dtype=np.dtype(dt))
    if g.shape!=o.shape: print(f"[ERROR] shape {g.shape} vs {o.shape}"); return False
    if not np.allclose(g,o,atol=eps,rtol=eps,equal_nan=True):
        d=np.abs(g.astype(np.float64)-o.astype(np.float64)); i=int(np.argmax(d))
        print(f"[ERROR] mismatch idx={i} golden={float(g[i])} out={float(o[i])} diff={float(d[i])}"); return False
    return True
def main():
    if not compare("golden_v2.bin","v2.bin",np.float32,1e-4): print("[ERROR] compare failed"); sys.exit(2)
    print("[INFO] compare passed")
if __name__=="__main__": main()
