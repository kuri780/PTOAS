#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: CANN-OSL-2.0
"""Validate the complete single-block scalar print integration output."""

import sys
from pathlib import Path


def main():
    log_path = Path("runtime.log")
    if not log_path.exists():
        print("[ERROR] missing runtime.log")
        sys.exit(2)
    output = log_path.read_text(errors="replace")
    expected = [
        "value = +003.250",
        "total_elems = 128",
        "block 0",
        "block 1",
    ]
    missing = [text for text in expected if text not in output]
    if missing:
        for text in missing:
            print(f"[ERROR] missing print output: {text}")
        sys.exit(2)
    print("[INFO] print output compare passed")


if __name__ == "__main__":
    main()
