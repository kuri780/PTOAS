# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""PTODSL TileLib template for pto.tprint."""

from ptodsl.tilelib import metadata as m
from ptodsl.tilelib.decorator import tile_template
from ptodsl.tilelib.templates.a5._common import NUMERIC_DTYPES
from ptodsl.tilelib.templates.a5._common import same_dtype_signatures

# pto.tprint is a debug operation that accepts a single tile_buf operand
# and emits DebugTunnel print output.  The template exists solely for the
# InsertTemplateAttributes pass; the actual lowering is handled by the
# VPTO / EmitC backend.


@tile_template(
    op="pto.tprint",
    name="template_ttprint",
    dtypes=same_dtype_signatures(1, dtypes=NUMERIC_DTYPES),
    iteration_axis="none",
    op_engine="other",
    op_class="other",
    id=0,
    loop_depth=0,
    is_post_update=False,
)
def template_ttprint(
    src: m.TileSpec,
) -> None:
    """Debug-tile-print — lowered by the VPTO/EmitC backend."""
    pass
