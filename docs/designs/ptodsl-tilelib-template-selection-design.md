# PTODSL TileLib Template Selection Design

## Background

PTOAS currently supports two TileLib backends for VPTO tile-op expansion:

- `tilelang`, the legacy TileLangDSL template implementation.
- `ptodsl`, the PTODSL-native template implementation.

The migration goal is for PTODSL to reach TileLangDSL parity for A5 TileLib
templates while keeping the VPTO compiler pipeline deterministic and
debuggable. A tile op may have several legal implementations for the same op
name. Those implementations can differ by dtype, layout, memory space,
attribute mode, tail behavior, temporary-buffer shape, or special algorithmic
version. This document defines how PTODSL discovers, filters, records, and
expands those template versions.

This is an implementation design document. User-facing tile-op semantics belong
in the ISA and user guide documents, not here.

## Goals

- Match the TileLangDSL version-selection model closely enough that ST coverage
  can migrate incrementally.
- Keep legality decisions in Python, where template metadata and predicates are
  authored with the template.
- Keep the IR-side candidate attribute compact and stable.
- Make `ExpandTileOp` specialization reuse safe across all operands that can
  change rendered helper bodies.
- Make debugging possible from either side of the boundary: PTODSL metadata,
  inserted IR attributes, after-expand IR, and emitted VPTO.

## Non-Goals

- This design does not define public PTODSL user syntax.
- This design does not make PTODSL the default TileLib backend.
- This design does not require every TileLangDSL version to be ported at once.
- This design does not store the full Python metadata object on every TileOp.

## Terminology

| Term | Meaning |
|---|---|
| Template | A Python function registered with `@tile_template(...)`. |
| Version | One registered implementation of an op, usually distinguished by metadata, constraints, or priority. |
| Candidate | A template version that targets the current op and target architecture. |
| Legal candidate | A candidate whose operand specs and context attributes satisfy its metadata and custom constraints. |
| Selected candidate | The first candidate recorded on the TileOp and requested by `ExpandTileOp` during rendering. |
| Specialization key | The C++ key used to deduplicate generated helper functions inside a module. |

## Pipeline

The PTODSL TileLib path has two compiler interactions with the Python daemon.

```text
TileOp in MLIR
  |
  | InsertTemplateAttributes
  |   - reconstruct operand specs from MLIR
  |   - collect context attributes
  |   - ask the PTODSL daemon for legal candidates
  |   - store compact candidate metadata on the TileOp
  v
TileOp with candidates attr
  |
  | ExpandTileOp
  |   - build a specialization key from current MLIR operands and attrs
  |   - choose candidate 0 from the compact candidates attr
  |   - ask the daemon to render that candidate
  |   - clone the generated helper and replace the TileOp with func.call
  v
VPTO-facing IR
```

The two-stage flow is intentional. `InsertTemplateAttributes` performs legality
before later passes can make candidate information harder to reconstruct.
`ExpandTileOp` still renders from the current MLIR operands so the helper body
matches the actual operand types and view metadata that survived to expansion.

## Template Metadata

PTODSL template authors register versions through `tilelib.tile_template`.
The registration metadata has two roles.

Hard legality fields:

- `op`
- `target`
- `dtypes`
- `layouts`
- `memory_spaces`
- `constraints`

Selection and reporting fields:

- `priority`
- `fusible`
- `loop_depth`
- `id`
- `Tail`
- `is_post_update`
- `iteration_axis`
- `op_engine`
- `op_class`
- `tags`

Only the fields needed after legality are persisted on the MLIR op. The rest
remain in Python metadata for selection, diagnostics, and future tooling.

## Operand Specs

Both `InsertTemplateAttributes` and `ExpandTileOp` reconstruct operand specs
from MLIR. The JSON shape sent to the daemon is deliberately close to
`TileSpec`, `ViewSpec`, `ScalarSpec`, and `VectorSpec`.

| Operand kind | Required metadata |
|---|---|
| tile | dtype, shape, valid shape, memory space, block layout, sub-layout, fractal size, pad value |
| view | dtype, shape, strides when known, memory space, optional layout |
| vector | dtype and vector shape |
| scalar | dtype and static integer value when recoverable |

Tile specs drive both legality and rendered tile-buffer entry types. View specs
are equally important: PTODSL templates often materialize `ViewSpec` shape or
stride values as constants in helper bodies. A view with the same dtype but a
different physical stride can require a different helper.

## Context Attributes

TileOp attributes that affect version selection or rendering are forwarded as
context attrs. Current examples include:

| Context attr | Typical users |
|---|---|
| `round_mode` | `tcvt` |
| `rounds` | `trandom` |
| `cmp_mode` | `tcmp`, `tcmps` |
| `mask_pattern` | gather-side paths |
| `precisionType` | high-precision math families |

When a new TileLangDSL version depends on an op attribute, the PTODSL migration
should first decide whether the attribute is a real context attr. If it changes
template legality or helper code generation, it must be forwarded before the
template is considered ported.

## Candidate Legality And Ranking

The daemon loads only the template module for the requested op and target. It
then evaluates each registered candidate:

1. Bind positional MLIR operands to the template parameter names.
2. Build a flat constraint context from the concrete specs.
3. Check op and target.
4. Check dtype signatures.
5. Check layout and memory-space metadata.
6. Merge context attributes.
7. Run custom constraint predicates.
8. Sort legal candidates by descending priority.

If no candidate is legal, the daemon reports a `NoMatchingTemplate` error with
per-candidate reasons. If multiple candidates tie for the highest priority and
no explicit candidate is requested, the registry reports ambiguity rather than
silently picking one.

For multi-candidate ops, candidate `id` values must be unique. The C++ pass
sorts persisted candidate metadata by `id` and then by name, so ids should be
stable and intentionally assigned.

## Compact Candidate Attribute

`InsertTemplateAttributes` stores a compact `candidates` array attribute on the
TileOp. Each entry contains:

- `id`
- `name`
- `loop_depth`
- `postupdate`
- `tail`

This attribute is intentionally not a copy of the full Python metadata object.
Legality has already happened in the daemon. The IR only needs a stable list of
legal render targets and the small amount of metadata consumed by downstream
passes.

Do not add fields to the IR candidate payload simply because they exist in
Python metadata. Add a field only when a C++ pass or IR-level test consumes it.

## Expansion And Specialization

`ExpandTileOp` uses the first candidate in the compact candidate list. For
PTODSL, it passes the selected candidate name back to the daemon so rendering
cannot accidentally choose a different legal template after the metadata pass.

The specialization key deduplicates generated helpers inside one module. It
must include every input that can change the rendered helper body:

- op name
- target architecture
- tile operand dtype, shape, valid shape, memory space, layouts, fractal size,
  and pad value
- view operand dtype, shape, strides, memory space, and layout
- vector operand dtype and shape
- scalar operand dtype and static value when known
- forwarded context attrs

The helper name should also carry enough of this information to make IR dumps
readable. It is not a semantic contract, but useful names make ST failures much
faster to inspect.

## View Metadata In The Specialization Key

PTODSL and TileLangDSL differ in how view metadata reaches helper bodies.

TileLangDSL helpers can keep a partition tensor view argument and read the
tensor-view stride in IR. PTODSL helpers often receive memref-shaped operands
and bake `ViewSpec` shape or stride values into the rendered function. Because
of that, the PTODSL specialization cache must not treat all views with the same
element dtype as equivalent.

A real failure exposed this requirement:

- A full `trowargmin` non-smoke run contained an earlier compact destination
  view with physical row width `1`.
- A later case used the same output tile type but a physical destination row
  width of `8`.
- The old PTODSL specialization key ignored view shape and strides, so
  `ExpandTileOp` reused the compact helper for the strided case.
- The row-arg computation was correct, but final writeback used a 4-byte GM row
  stride instead of 32 bytes.

The fix was to include view shape, view strides, view memory space, and view
layout in both specialization equality and hashing. A focused lit regression
uses two `tstore` ops with the same tile type but different destination view
strides to prove those helpers remain distinct.

## Failure Modes

| Symptom | Most likely layer |
|---|---|
| `NoMatchingTemplate` with dtype/signature reason | missing metadata coverage or too-narrow constraints |
| custom constraints are not satisfied | template predicate does not accept the real ST operand form |
| no `candidates` attr at expansion | metadata pass did not run or candidate attr was lost |
| `ExpandTileOp requires at least one template candidate` | candidate attr missing or empty |
| isolated case passes but full file fails | specialization cache may be missing an operand or context field |
| computed values look right but GM compare fails | store/load view shape, stride, or valid-shape metadata may be wrong |

## Validation Strategy

Each selection or specialization change should have a small regression before
relying on full ST.

Recommended layers:

1. Python TileLib tests for metadata, constraints, and render coverage.
2. Focused lit tests for inserted candidate attrs and expansion behavior.
3. Compiler-only emits for representative ST `.pto` files.
4. Smoke ST for quick end-to-end validation.
5. Non-smoke ST for parity claims.

The `trowargmin` view-stride cache bug is covered by
`test/lit/vpto/expand_tile_op_ptodsl_view_stride_cache.pto`. The test is
deliberately small: it checks helper specialization directly instead of
depending on a long row-reduction run.

## Rules For Future Version Work

- Register every intentionally supported version with explicit metadata.
- Keep custom constraints narrow enough to reject unsupported forms and broad
  enough to accept ST-proven TileLangDSL forms.
- Forward context attrs before porting a version that depends on them.
- Use stable candidate ids for multi-candidate ops.
- Put all helper-code-affecting operand metadata in the specialization key.
- Add a focused regression for each backend-selection bug.
- Treat full ST status files as snapshots, not design documentation.
