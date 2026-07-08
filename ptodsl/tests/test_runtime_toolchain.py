#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import os
import tempfile
import unittest
from unittest import mock

from pathlib import Path

from ptodsl._runtime import toolchain


class ResolvePtoasBinaryTests(unittest.TestCase):
    def test_env_override_wins_over_repo_default(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            env_ptoas = temp_root / "env-ptoas"
            env_ptoas.write_text("", encoding="utf-8")

            repo_build_ptoas = temp_root / "build" / "tools" / "ptoas" / "ptoas"
            repo_build_ptoas.parent.mkdir(parents=True, exist_ok=True)
            repo_build_ptoas.write_text("", encoding="utf-8")

            fake_toolchain_file = temp_root / "repo" / "ptodsl" / "ptodsl" / "_runtime" / "toolchain.py"
            fake_toolchain_file.parent.mkdir(parents=True, exist_ok=True)
            fake_toolchain_file.write_text("", encoding="utf-8")

            with mock.patch.dict(os.environ, {"PTOAS_BIN": str(env_ptoas)}, clear=False), mock.patch.object(
                toolchain, "__file__", str(fake_toolchain_file)
            ):
                resolved = toolchain.resolve_ptoas_binary()

            self.assertEqual(resolved, env_ptoas)

    def test_invalid_env_override_raises(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            missing = Path(temp_dir) / "missing-ptoas"
            with mock.patch.dict(os.environ, {"PTOAS_BIN": str(missing)}, clear=False):
                with self.assertRaises(FileNotFoundError):
                    toolchain.resolve_ptoas_binary()


if __name__ == "__main__":
    unittest.main()
