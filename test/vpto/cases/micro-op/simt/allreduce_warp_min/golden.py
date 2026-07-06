#!/usr/bin/env python3
import argparse, numpy as np
from pathlib import Path

NLANES = 32
EXPECTED = 1.0

def generate(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    out = np.zeros(NLANES, dtype=np.float32)
    out.tofile(output_dir / "out.bin")
    golden = np.full(NLANES, EXPECTED, dtype=np.float32)
    golden.tofile(output_dir / "golden_out.bin")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", type=Path, default=Path("."))
    a = p.parse_args()
    generate(a.output_dir)

if __name__ == "__main__":
    main()
