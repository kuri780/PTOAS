#!/usr/bin/env python3
import sys, numpy as np

def main():
    golden = np.fromfile("golden_out.bin", dtype=np.float32)
    out = np.fromfile("out.bin", dtype=np.float32)
    if golden.shape != out.shape or not np.allclose(golden, out, rtol=1e-5, atol=1e-5):
        mismatches = np.nonzero(~np.isclose(golden, out, rtol=1e-5, atol=1e-5))[0]
        idx = int(mismatches[0]) if mismatches.size else 0
        print(f"[ERROR] mismatch at idx={{idx}}, golden={{golden[idx]:.6f}}, out={{out[idx]:.6f}}")
        sys.exit(2)
    print("[INFO] compare passed")

if __name__ == "__main__":
    main()
