# PTODSL TileLib Migration Test Checklist

This page tracks the tests used while migrating PTOAS TileLib expansion from
the legacy TileLang implementation to PTODSL. Run commands from the repository
root.

## Environment

Set up PTOAS, PTODSL, MLIR, and LLVM test-tool paths:

```bash
export PTOAS_ENV_SKIP_SMOKE_TEST=1
source scripts/ptoas_env.sh
export FILECHECK="$LLVM_BUILD_DIR/bin/FileCheck"
```

The Python-only tests do not require rebuilding PTOAS. Tests that invoke
`ptoas` must use a binary rebuilt after the corresponding C++ or TableGen
changes.

## Milestone coverage

| Milestone | Test | Purpose |
|---|---|---|
| Legacy baseline | `expand_tile_op_tilelang_tsub.pto` | Confirms the default TileLang backend still works |
| PTODSL TileLib package | `test_tilelib_constraints.py`, `test_tilelib_elementwise.py`, `test_tilelib_render.py`, `test_tilelib_select.py` | Covers legality constraints, template registration and selection, and rendering |
| PTODSL daemon | `test_tilelib_daemon.py` | Covers the Unix-socket protocol, metadata, rendering, candidate IDs, and caching |
| PTOAS daemon selection | `expand_tile_op_ptodsl_tsub.pto` | Confirms `--tile-lib-backend=ptodsl` starts and uses the PTODSL daemon |
| Two-call expansion | `expand_tile_op_ptodsl_tsub.pto` | Confirms metadata discovery followed by rendering with the sole candidate ID |
| Multi-candidate boundary | `expand_tile_op_ptodsl_tadd_requires_selection.pto` | Confirms four legal `tadd` candidates require a separate selection stage |

## Python TileLib tests

Run every Python TileLib test:

```bash
python3 -m unittest discover -s ptodsl/tests -p 'test_tilelib_*.py'
```

Run the layers individually:

```bash
python3 ptodsl/tests/test_tilelib_constraints.py
python3 ptodsl/tests/test_tilelib_elementwise.py
python3 ptodsl/tests/test_tilelib_render.py
python3 ptodsl/tests/test_tilelib_select.py
python3 ptodsl/tests/test_tilelib_daemon.py
```

Each command prints `OK` when successful.

## PTOAS integration tests

### PTODSL positive path: one legal candidate

`pto.tsub` has one legal PTODSL candidate. The test checks that PTOAS expands
it into vector operations:

```bash
ptoas --pto-arch=a5 --pto-backend=vpto --emit-vpto \
  --tile-lib-backend=ptodsl \
  test/lit/vpto/expand_tile_op_ptodsl_tsub.pto -o - 2>/dev/null |
"$FILECHECK" test/lit/vpto/expand_tile_op_ptodsl_tsub.pto
```

### PTODSL negative path: selection is still required

`pto.tadd` currently has four legal candidates. PTOAS is expected to reject it
after metadata discovery because version selection is not a separate stage yet:

```bash
ptoas --pto-arch=a5 --pto-backend=vpto --emit-vpto \
  --tile-lib-backend=ptodsl \
  test/lit/vpto/expand_tile_op_ptodsl_tadd_requires_selection.pto \
  -o /dev/null 2>&1 |
"$FILECHECK" \
  test/lit/vpto/expand_tile_op_ptodsl_tadd_requires_selection.pto
```

The `ptoas` process fails intentionally in this test. `FileCheck` succeeds
only when it sees the expected four-candidate diagnostic.

### Legacy backend regression

Omitting `--tile-lib-backend` must continue to select TileLang:

```bash
ptoas --pto-arch=a5 --pto-backend=vpto --emit-vpto \
  test/lit/vpto/expand_tile_op_tilelang_tsub.pto -o - 2>/dev/null |
"$FILECHECK" test/lit/vpto/expand_tile_op_tilelang_tsub.pto
```

## Reading the result

`FileCheck` is silent when it succeeds. Immediately check its status with:

```bash
echo $?
```

`0` means the check passed. When candidate selection is implemented, replace
the expected-failure `tadd` coverage with a positive selected-version test and
update the milestone table above.
