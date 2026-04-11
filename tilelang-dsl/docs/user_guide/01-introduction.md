# TileLang Python DSL Guide

The TileLang Python DSL provides a high-level, Pythonic interface for authoring vector compute kernels targeting the Ascend NPU hardware. This guide is intended for library developers and performance engineers who need to write efficient, hardware-aware kernels using the PTO micro instruction set.

The DSL is designed to generate MLIR function libraries rather than direct binary executables. These MLIR libraries are intended to be consumed by other compilation frameworks that transform high-level tile semantics into low-level vector operations. This enables library developers to focus on hardware-aware kernel authoring while relying on upstream compilers for tile-level optimizations and code generation.

## Language Tier

The DSL surface is organized into multiple maturity tiers, reflecting the stability and intended use of different language features. As the design evolves, the basic authoring path is being explicitly separated from more advanced surfaces. Refer to the following table when reading this guide:

| Surface Family | Tier | Usage Guidance |
|----------------|------|----------------|
| `TensorView` | `basic` | Default GM-facing data model for starter kernels. |
| `Tile` | `basic` | Default UB-facing compute tile for starter kernels. |
| Base vector ops (`make_mask`, `vlds`, `vsts`, `vadd`, `vmuls`, etc.) | `basic` | Default compute skeleton for starter kernels. |
| `strict_vecscope` | `advanced` | Explicit vector-scope management for expert authoring. |
| Raw pointer family (`ptr(...)`, `castptr`, `addptr`) | `advanced` | For expert authoring and migration; not required for Quick Start. |
| DMA family (`copy_*`, `set_loop*_stride_*`, `set_loop_size_*`) | `advanced` | Direct DMA engine control for expert authoring. |
| Tile pointer helper (`tile.as_ptr()`) | `advanced` | Expert-only helper when advanced authoring needs explicit typed pointers. |

For the authoritative tier classification, consult `tilelang-dsl/python/tilelang_dsl/support_matrix.py`. For known implementation gaps, refer to `tilelang-dsl/docs/unsupported-features.md`.

### Basic vs Advanced Authoring Modes

The TileLang DSL provides two distinct authoring modes:

**Basic Mode (default)**
- Uses **Tile element/slice semantics** for buffer access
- Direct tile indexing syntax: `tile[start:]`, `tile[row, col:]`, `tile[row:, col]` (Tile indexing sugar only supports open-ended vector slices; explicit `stop` and `step` forms are not accepted for `Tile` indexing)
- Vector operations use element-indexing syntax: `pto.vlds(tile[row, col:])`, `pto.vsts(vec, tile[start:], mask)`
- No pointer arithmetic or explicit offset calculations
- Suitable for most kernel authoring with high-level abstractions

**Advanced Mode (`advanced=True` in `@pto.vkernel`)**
- Uses **raw pointer semantics** for explicit memory management  
- Direct pointer operations correspond to `pto.ptr` types in MLIR
- Explicit pointer arithmetic: `ptr(...)`, `castptr`, `addptr`
- Manual DMA engine control with low-level copy operations
- Requires explicit buffer management and pointer arithmetic
- Intended for expert users and performance-critical optimizations

**Key Differences**
- **Basic mode**: Uses tile element-indexing syntax (`tile[row, col:]`, `tile[start:]`) for vector operations
- **Advanced mode**: Uses pointer byte-offset syntax (`pto.vlds(buf: ptr, offset)`) for vector operations
- Tile slices in basic mode correspond to MLIR `memref` types
- Raw pointers in advanced mode correspond to MLIR `pto.ptr` types
- No automatic conversion between tile and pointer semantics - choose the appropriate syntax for your authoring mode
