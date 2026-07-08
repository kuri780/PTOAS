# PTODSL TileLib ST Failure Tracker

This file tracks TileLang ST smoke failures seen while validating the A5 TileLib
templates with `PTOAS_TILE_LIB_BACKEND=ptodsl`.

Latest focused log run:

```bash
# Logs are under mani_log/<testcase>.log.
# Each testcase was run in an isolated /tmp/mani_st_isolated/<testcase>/smoke tree
# to avoid concurrent CMake build directory races.
PTOAS_TILE_LIB_BACKEND=ptodsl python3 test/tilelang_st/script/run_st.py \
  -r sim -v a5 \
  -p build-llvm21/tools/ptoas/ptoas \
  -t <testcase> --target-dir /tmp/mani_st_isolated/<testcase>/smoke
```

Latest fixed batch:

```text
mani_log/fixed_attr/summary.tsv
```

The following smoke testcases now build, run, and compare successfully with
PTODSL: `tcolexpand`, `tcolexpandmax`, `tcolexpandmin`, `tcolexpandmul`,
`softmax`, `tcolmax`, `tcolmin`, `tlrelu`, `tload`, `tmatmul`, and `tmov`.

| Testcase | Stage | Failure type | Main error / observation |
|---|---|---|---|
| softmax | Fixed | Passed smoke ST | Fixed by making `tstore` handle dynamic partition/tile metadata the same way as `tload`; isolated smoke compare passes. See `mani_log/softmax_tstore_fix.log`. |
| tcmp | Runtime | Wrong output | `f32_8x64_gt` mismatch: golden `115.0`, output `-128.0`, max diff `243.0`. |
| tcmps | Build | Template instantiation failure | `ExpandTileOp: failed to instantiate TileLib template for tcmps`. |
| tcolexpand | Fixed | Passed smoke ST | Fixed by preserving candidates through `PTOViewToMemref` and allowing `vec` row-major tiles. See `mani_log/fixed_attr/tcolexpand.log`. |
| tcolexpanddiv | Build | Unsupported dtype | `pto.tcolexpanddiv` has no legal template for dtype signature `('i32', 'i32', 'i32')`. |
| tcolexpandmax | Fixed | Passed smoke ST | Fixed by preserving candidates through `PTOViewToMemref` and allowing `vec` row-major tiles. See `mani_log/fixed_attr/tcolexpandmax.log`. |
| tcolexpandmin | Fixed | Passed smoke ST | Fixed by preserving candidates through `PTOViewToMemref` and allowing `vec` row-major tiles. See `mani_log/fixed_attr/tcolexpandmin.log`. |
| tcolexpandmul | Fixed | Passed smoke ST | Fixed by preserving candidates through `PTOViewToMemref` and allowing `vec` row-major tiles. See `mani_log/fixed_attr/tcolexpandmul.log`. |
| tcolmax | Fixed | Passed smoke ST | Fixed by preserving candidates through `PTOViewToMemref` and allowing `vec` row-major tiles. See `mani_log/fixed_attr/tcolmax.log`. |
| tcolmin | Fixed | Passed smoke ST | Fixed by preserving candidates through `PTOViewToMemref` and allowing `vec` row-major tiles. See `mani_log/fixed_attr/tcolmin.log`. |
| tcvt | Build | Unsupported dtype | Test wants `('f16', 'f32')`; current `tcvt` templates do not support that direction/signature. |
| tdiv | Build | Unsupported dtype | `pto.tdiv` template exists, but `('f16', 'f16', 'f16')` is not supported. |
| tdivs | Build | Python/runtime crash | Metadata RPC crashes with `OSError: failed to make path absolute`; `Fatal Python error: error evaluating path`. |
| textract | Build | Constraint failure | `template_textract_vec2vec_nd` exists, but custom constraints are not satisfied. |
| textract_fp | Build | Wrong dependency / instantiation failure | `ExpandTileOp: failed to instantiate TileLib template for tmatmul`. |
| tfillpad | Build | tfillpad instantiation failure | After the `tload` metadata fix, isolated smoke reaches `tfillpad` expansion and fails with `ExpandTileOp requires at least one template candidate` at `tfillpad.pto:57`. See `mani_log/tload_downstream/tfillpad.log`. |
| tfillpad_expand | Runtime | Wrong output | `u16_64x16_src_63x7` and `u16_260x32_src_259x7` mismatch at idx 7: golden `65535`, output `0`. |
| tfillpad_inplace | Build | Template instantiation failure | `ExpandTileOp: failed to instantiate TileLib template for tfillpad_inplace`. |
| tload | Fixed | Passed smoke ST | Fixed by preserving `pto.make_tensor_view` strides through `pto.partition_view` metadata for TileLib selection. DN padded load/store smoke compare passes. See `mani_log/tload_fix.log`. |
| tlog | Build | Template instantiation failure | `ExpandTileOp: failed to instantiate TileLib template for tlog`. |
| tlrelu | Fixed | Passed smoke ST | Fixed by preserving candidates through `PTOViewToMemref` and allowing `vec` row-major tiles. See `mani_log/fixed_attr/tlrelu.log`. |
| tmatmul | Fixed | Passed smoke ST | Fixed by preserving candidates through `PTOViewToMemref`; smoke ST passes. See `mani_log/fixed_attr/tmatmul.log`. |
| tmov | Fixed | Passed smoke ST | Fixed by preserving candidates through `PTOViewToMemref` and allowing `vec` row-major tiles. See `mani_log/fixed_attr/tmov.log`. |
| tmrgsort | Build | Operand/signature mismatch | `NoMatchingTemplate` for `pto.tmrgsort`; single-list expects 3 operands but got 5, multi-list2 rejects dtype signature `('f16', 'f16', 'f16', 'f16', 'i16')`, multi-list3/4 expect 6/7 operands. See `mani_log/tmrgsort.log`. |
| trowargmax | Build | Unsupported dtype | `NoMatchingTemplate` for `pto.trowargmax`; `template_trowargmax` rejects dtype signature `('f32', 'f32', 'ui32')`. See `mani_log/trowargmax.log`. |
| trowargmin | Build | Unsupported dtype | `NoMatchingTemplate` for `pto.trowargmin`; `template_trowargmin` rejects dtype signature `('f32', 'f32', 'ui32')`. See `mani_log/trowargmin.log`. |
| trowsum | Build | Unsupported dtype | `NoMatchingTemplate` for `pto.trowsum`; `template_trowsum` rejects dtype signature `('i16', 'i16', 'i16')`. See `mani_log/trowsum.log`. |
| tsel | Build | Constraint failure | `NoMatchingTemplate` for `pto.tsel`; `template_tsel` custom constraints are not satisfied. See `mani_log/tsel.log`. |
| tsels | Build | Unsupported dtype | `NoMatchingTemplate` for `pto.tsels`; `template_tsels` rejects dtype signature `('i16', 'i32', 'i32', 'i32', 'i32')`. See `mani_log/tsels.log`. |
| tsort32 | Build | Operand/signature mismatch | `NoMatchingTemplate` for `pto.tsort32`; `template_tsort32` expects 3 operands but got 4, and `template_tsort32_with_tmp` rejects dtype signature `('f32', 'ui32', 'f32', 'f32')`. See `mani_log/tsort32.log`. |

## Notes

- The first parallel attempt used shared `test/tilelang_st/.../smoke/build` and produced
  CMake races. The current `mani_log` entries were regenerated with isolated target
  directories under `/tmp/mani_st_isolated`.
- The rows above are focused on the current failing set. Passing cases are tracked by
  the ST summary output rather than repeated here.
- `tfillpad` previously failed through `tload`. Focused reruns show it now reaches
  its own template-candidate failure.
