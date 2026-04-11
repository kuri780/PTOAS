### Enum Types for Vector Memory Operations

The following enum types provide type-safe parameter specification for vector memory operations:

- **`DeinterleaveDist`**: Deinterleave distribution modes for `pto.vldx2`
  - `B8`: 8-bit element deinterleave (for i8)
  - `B16`: 16-bit element deinterleave (for i16, f16, bf16)
  - `B32`: 32-bit element deinterleave (for i32, f32)
  - `BD`: Broadcast deinterleave mode

- **`InterleaveDist`**: Interleave distribution modes for `pto.vstx2`
  - `B8`: 8-bit element interleave (for i8)
  - `B16`: 16-bit element interleave (for i16, f16, bf16)
  - `B32`: 32-bit element interleave (for i32, f32)

- **`StrideMode`**: Stride modes for `pto.vsld`
  - `S3_B16`: Stride 3, block size 16
  - `S4_B64`: Stride 4, block size 64
  - `S8_B32`: Stride 8, block size 32
  - `S2_B64`: Stride 2, block size 64

### Address Generation Syntax Sugar

To simplify address calculation and reduce manual byte offset computation errors, TileLang DSL provides syntactic sugar for vector load/store operations using element-based indexing. This syntax automatically computes the byte offset based on tile shape, element type, and layout.

#### Indexing Syntax

The syntax supports two indexing modes for different operations:

1. **Vector-range indexing** (for vector load/store operations):
   - **Row-major layout (default)**: `tile[row_index, col_start:]`
     - `row_index`: Row index (0-based)
     - `col_start:`: Starting column index followed by colon, indicating a vector-width contiguous region starting from this column
     - The colon (`:`) indicates an implicit vector-width range determined by hardware vector size (256 bytes) and element type
   
   - **Column-major layout**: `tile[row_start:, col_index]`
     - `row_start:`: Starting row index followed by colon, indicating a vector-width contiguous region starting from this row
     - `col_index`: Column index (0-based)
     - Used for column-major tiles (`BLayout.COL_MAJOR`) where elements are stored column-wise
   
   - **1D tile indexing**: `tile[start:]` (or equivalently `tile[0, start:]` for row-major or `tile[start:, 0]` for column-major)
     - `start:`: Starting element index followed by colon

   Tile indexing sugar only accepts an open-ended vector slice. Python slice
   forms with an explicit `stop` or `step` are not supported for `Tile`
   indexing. For example, `tile[row, col:col_end]`, `tile[row, col::2]`,
   `tile[row_start:row_end, col]`, and `tile[start:stop:step]` are invalid.

2. **Single-element indexing** (for scalar load operations like `pto.vsld`):
   - **Row-major layout (default)**: `tile[row_index, col_index]`
     - `row_index`: Row index (0-based)
     - `col_index`: Column index (0-based)
     - Loads a single element at the specified position and broadcasts it to all vector lanes
   
   - **Column-major layout**: `tile[row_index, col_index]` (same syntax)
     - `row_index`: Row index (0-based)
     - `col_index`: Column index (0-based)
     - Same syntax as row-major; the layout determines how the offset is computed
   
   - **1D tile indexing**: `tile[pos]`
     - `pos`: Element index (0-based)
     - Loads a single element at the specified position and broadcasts it to all vector lanes

#### Vector Width Calculation

The number of elements loaded/stored in a single vector operation is determined by:

```
vector_lanes = 256 // element_size_bytes(element_type)
```

**Convenience API**: Use `pto.elements_per_vreg(dtype)` to compute the number of elements per vector register for a given element type (e.g., `pto.elements_per_vreg(pto.f32)` returns 64, `pto.elements_per_vreg(pto.f16)` returns 128). See [Type Query Operations](07-frontend-operations.md#type-query-operations) for full documentation.

Where `element_size_bytes` is:
- 1 byte for `i8`
- 2 bytes for `i16`, `f16`, `bf16`
- 4 bytes for `i32`, `f32`
- 8 bytes for `i64`

#### Offset Computation

The byte offset is automatically computed based on tile layout:

- **Row-major layout** (`BLayout.ROW_MAJOR`):
  ```
  offset = (row_index * stride_row + col_start) * element_size_bytes
  ```
  where `stride_row` is the row stride in elements (typically `tile.shape[1]` for contiguous tiles).

- **Column-major layout** (`BLayout.COL_MAJOR`):
  - For syntax `tile[row_start:, col_index]`:
    ```
    offset = (col_index * stride_col + row_start) * element_size_bytes
    ```
  - For backward compatibility with traditional offset calculation:
    ```
    offset = (col_start * stride_col + row_index) * element_size_bytes
    ```
  where `stride_col` is the column stride in elements (typically `tile.shape[0]` for contiguous tiles), `row_start` is the starting row index, and `col_index` is the column index.

**Note**: 
- For single-element indexing (`tile[row, col]` or `tile[pos]`), the same offset formulas apply with `col_start` replaced by `col_index` (or `start` replaced by `pos` for 1D tiles).
- For column-major vector-range indexing (`tile[row_start:, col_index]`), the offset formula uses `row_start` as the starting position along the contiguous dimension.
- The compiler automatically handles the appropriate substitution based on the indexing syntax and tile layout.

#### Constraints

1. **Boundary checks**: The requested region must be within tile bounds:
   - **For vector-range indexing** (`:` syntax):
     - **Row-major layout** (`tile[row_index, col_start:]`):
       - `row_index < tile.shape[0]` and `col_start + vector_lanes <= tile.shape[1]`
     - **Column-major layout** (`tile[row_start:, col_index]`):
       - `row_start + vector_lanes <= tile.shape[0]` and `col_index < tile.shape[1]`
     - **1D tile indexing**: `tile[start:]`
       - `start + vector_lanes <= tile.shape[0]` (or `tile.shape[1]` for 1D tiles)
   - **For single-element indexing** (no `:` syntax):
     - 2D: `row_index < tile.shape[0]` and `col_index < tile.shape[1]` (same for both layouts)
     - 1D: `pos < tile.shape[0]` (or `tile.shape[1]` for 1D tiles)

2. **Alignment**: The computed offset must satisfy hardware alignment requirements for the operation.

3. **Full vectors only**: The `:` syntax always loads/stores a full vector width. For partial vectors, use the traditional byte offset approach with explicit mask handling.

4. **Single-element operations**: The single-element indexing syntax (`tile[row, col]` or `tile[pos]`) is only supported for scalar load operations like `pto.vsld`. For other operations, use vector-range indexing with `:` syntax.

5. **No explicit slice bounds/stride for `Tile` indexing**: `Tile` vector-range
   indexing only supports the open-ended forms `tile[start:]`,
   `tile[row, col:]`, and `tile[row_start:, col_index]` (for column-major
   layout). `stop` and `step` syntax are not accepted in user-guide Tile
   indexing.

#### Supported Operations

The indexing syntax is supported for all vector load and store operations with the following syntax mapping:

- **Vector-range indexing** (`tile[row, col:]` or `tile[start:]`):
  - Load operations: `vlds`, `vldas`, `vldus`, `vldx2`
  - Store operations: `vsts`, `vsta`, `psts`, `vsst`, `vstx2`

- **Single-element indexing** (`tile[row, col]` or `tile[pos]`):
  - Load operations: `vsld` (scalar load with broadcast)

#### Examples

The following examples use row-major layout syntax. For column-major tiles, use `tile[row_start:, col_index]` syntax instead of `tile[row_index, col_start:]`.

```python
# 2D tile indexing (row-major layout)
vec = pto.vlds(tile[i, j:])          # Load vector from row i, columns j to j+vector_lanes-1
pto.vsts(vec, tile[i, j:], mask)     # Store vector with mask

# 1D tile indexing  
vec = pto.vlds(tile[k:])             # Load vector from elements k to k+vector_lanes-1
pto.vsts(vec, tile[k:], mask)        # Store vector with mask

# Dual load with deinterleave
low, high = pto.vldx2(tile[i, j:], DeinterleaveDist.B32)

# Aligned load with indexing
vec = pto.vldas(tile[i, j:], align)

# Scalar load (broadcast)
vec = pto.vsld(tile[i, j])          # Load scalar at tile[i,j] and broadcast to vector
```

#### Comparison with Manual Offset Calculation

**Traditional approach (error-prone):**
```python
# Manual byte offset calculation for f32 tile
rows, cols = tile.shape
row_offset = i * cols * 4  # Hard-coded 4 bytes for f32
col_offset = j * 4
offset = row_offset + col_offset
vec = pto.vlds(tile, offset)
```

**New syntax (type-safe):**
```python
# Automatic offset calculation
vec = pto.vlds(tile[i, j:])  # Compiler computes correct offset for any element type
```

The syntax sugar eliminates manual byte calculations, reduces errors, and makes code generic across different element types (e.g., the same kernel works for both `f16` and `f32` without modification).

### Vector Load Operations

Operations for loading data from memory into vector registers.

#### `pto.vlds(buf: ptr, offset: Index) -> VRegType`  [Advanced Tier]
#### `pto.vlds(tile[row, col:]) -> VRegType`  [Basic Tier]
#### `pto.vlds(tile[start:]) -> VRegType`  [Basic Tier]

**Description**: Stateless vector load from buffer. Supports both traditional byte-offset syntax and new element-indexing syntax.

**Parameters (pointer syntax)**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `buf` | `ptr` | Pointer to buffer in UB memory space (Advanced mode only - requires explicit pointer) |
| `offset` | `Index` | Byte offset |

**Parameters (element-indexing syntax)**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `tile[row, col:]` | `Tile` with indexing | 2D tile with row index and starting column (vector-width range) |
| `tile[start:]` | `Tile` with indexing | 1D tile with starting element index (vector-width range) |

**Returns**:
| Return Value | Type | Description |
|--------------|------|-------------|
| `vec` | `VRegType` | Loaded vector register |

**Constraints**:
- Buffer must be in UB memory space
- For byte-offset syntax: offset must be properly aligned based on element type
- For element-indexing syntax: the requested vector region must be within tile bounds and satisfy alignment requirements

**Examples**:
```python
# Traditional byte-offset syntax
vec = pto.vlds(ub_ptr, lane * 256)

# New element-indexing syntax
vec = pto.vlds(tile[i, j:])      # Load from row i, columns j to j+vector_lanes-1
vec = pto.vlds(tile[k:])         # Load from 1D tile, elements k to k+vector_lanes-1

# Generic kernel that works for both f16 and f32
@pto.vkernel(target="a5", op="scale", dtypes=[(pto.AnyFloat, pto.AnyFloat)], priority=10)
def generic_scale(src: pto.Tile, dst: pto.Tile, scale: pto.f32):
    rows, cols = src.shape
    all_mask = pto.make_mask(src.element_type, PAT.ALL)
    for i in range(0, rows):
        for j in range(0, cols, vector_lanes):  # vector_lanes computed from element type
            # No manual byte calculation needed!
            vec = pto.vlds(src[i, j:])
            scaled = pto.vmuls(vec, scale, all_mask)
            pto.vsts(scaled, dst[i, j:], all_mask)
```

#### `pto.vldas(buf: ptr) -> pto.align`  [Advanced Tier]
#### `pto.vldas(tile[row, col:]) -> pto.align`  [Basic Tier]
#### `pto.vldas(tile[start:]) -> pto.align`  [Basic Tier]

**Description**: Prime alignment buffer for subsequent unaligned load. Returns alignment state for use with `pto.vldus`. Supports both pointer syntax and element-indexing syntax.

**Parameters (pointer syntax)**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `buf` | `ptr` | Pointer to buffer in UB memory space (Advanced mode only - requires explicit pointer) |

**Parameters (element-indexing syntax)**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `tile[row, col:]` | `Tile` with indexing | 2D tile with row index and starting column |
| `tile[start:]` | `Tile` with indexing | 1D tile with starting element index |

**Returns**:
| Return Value | Type | Description |
|--------------|------|-------------|
| `align` | `pto.align` | Alignment state for use with `pto.vldus` |

**Examples**:
```python
# Pointer syntax
align = pto.vldas(ub_ptr)

# Element-indexing syntax
align = pto.vldas(tile[i, j:])
align = pto.vldas(tile[k:])
```

#### `pto.vldus(buf: ptr, align: pto.align) -> (VRegType, pto.align, ptr)`  [Advanced Tier]
#### `pto.vldus(tile[row, col:], align: pto.align) -> (VRegType, pto.align, ptr)`  [Basic Tier]
#### `pto.vldus(tile[start:], align: pto.align) -> (VRegType, pto.align, ptr)`  [Basic Tier]

**Description**: Unaligned load using primed align state. Requires alignment state from `pto.vldas` or previous `pto.vldus`. Updates alignment state and base pointer for subsequent loads. Supports both pointer syntax and element-indexing syntax.

**Parameters (pointer syntax)**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `buf` | `ptr` | Pointer to buffer in UB memory space (Advanced mode only - requires explicit pointer) |
| `align` | `pto.align` | Alignment state from `pto.vldas` or previous `pto.vldus` |

**Parameters (element-indexing syntax)**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `tile[row, col:]` | `Tile` with indexing | 2D tile with row index and starting column |
| `align` | `pto.align` | Alignment state from `pto.vldas` or previous `pto.vldus` |
| _or_ | | |
| `tile[start:]` | `Tile` with indexing | 1D tile with starting element index |
| `align` | `pto.align` | Alignment state from `pto.vldas` or previous `pto.vldus` |

**Returns**:
| Return Value | Type | Description |
|--------------|------|-------------|
| `vec` | `VRegType` | Assembled vector value |
| `align_out` | `pto.align` | Updated alignment state for next load |
| `base_out` | `ptr` | Post-update base pointer state |

**Constraints**:
- A matching `pto.vldas` must appear before the first dependent `pto.vldus` stream in the same vector loop
- Both alignment state and base address advance across the stream
- If DSL authoring uses explicit byte/element offsets, the frontend first rewrites them into pointer/index expressions before lowering to this VPTO form.

**Examples**:
```python
# Pointer syntax - requires alignment state priming
align = pto.vldas(ub_ptr)
vec, align_out, base_out = pto.vldus(ub_ptr, align)

# Element-indexing syntax
align = pto.vldas(tile[i, j:])
vec, align_out, base_out = pto.vldus(tile[i, j:], align)

# Multiple unaligned loads in a stream
align = pto.vldas(tile[k:])
for n in range(4):
    vec, align, base = pto.vldus(tile[k:], align)  # alignment state updates
```


#### `pto.vldx2(buf: ptr, offset: Index, dist: DeinterleaveDist) -> (VRegType, VRegType)`  [Advanced Tier]
#### `pto.vldx2(tile[row, col:], dist: DeinterleaveDist) -> (VRegType, VRegType)`  [Basic Tier]
#### `pto.vldx2(tile[start:], dist: DeinterleaveDist) -> (VRegType, VRegType)`  [Basic Tier]

**Description**: Dual vector load with deinterleave (AoS → SoA conversion). Loads interleaved data from a single buffer and deinterleaves into two vectors. Supports both byte-offset and element-indexing syntax.

**Parameters (pointer syntax)**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `buf` | `ptr` | Pointer to source buffer in UB memory space (Advanced mode only - requires explicit pointer) |
| `offset` | `Index` | Byte offset |
| `dist` | `DeinterleaveDist` | Deinterleave distribution mode: `DeinterleaveDist.B8`, `DeinterleaveDist.B16`, `DeinterleaveDist.B32`, `DeinterleaveDist.BD`. Determines element size and interleave pattern. |

**Parameters (element-indexing syntax)**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `tile[row, col:]` | `Tile` with indexing | 2D tile with row index and starting column (vector-width range) |
| `dist` | `DeinterleaveDist` | Deinterleave distribution mode: `DeinterleaveDist.B8`, `DeinterleaveDist.B16`, `DeinterleaveDist.B32`, `DeinterleaveDist.BD`. Determines element size and interleave pattern. |
| _or_ | | |
| `tile[start:]` | `Tile` with indexing | 1D tile with starting element index (vector-width range) |
| `dist` | `DeinterleaveDist` | Deinterleave distribution mode: `DeinterleaveDist.B8`, `DeinterleaveDist.B16`, `DeinterleaveDist.B32`, `DeinterleaveDist.BD`. |

**Returns**:
| Return Value | Type | Description |
|--------------|------|-------------|
| `low` | `VRegType` | First vector (even elements in interleaved stream) |
| `high` | `VRegType` | Second vector (odd elements in interleaved stream) |

**Constraints**:
- Source buffer must be in UB memory space
- Offset must satisfy alignment requirements for the selected distribution mode
- The requested vector region must be within tile bounds (for element-indexing syntax)
- Distribution mode must match element type (e.g., `DeinterleaveDist.B32` for 32-bit elements)

**Examples**:
```python
from pto import DeinterleaveDist

# Byte-offset syntax
low, high = pto.vldx2(ub_ptr, offset, DeinterleaveDist.B32)

# Element-indexing syntax
low, high = pto.vldx2(tile[i, j:], DeinterleaveDist.B32)
low, high = pto.vldx2(tile[k:], DeinterleaveDist.B16)

# Example: Load interleaved XY pairs into separate X/Y vectors
x_vec, y_vec = pto.vldx2(xy_tile[i, j:], DeinterleaveDist.B32)
```

#### `pto.vsld(buf: ptr, offset: Index, stride: StrideMode) -> VRegType`  [Advanced Tier]
#### `pto.vsld(tile[row, col], stride: StrideMode) -> VRegType`  [Basic Tier]
#### `pto.vsld(tile[pos], stride: StrideMode) -> VRegType`  [Basic Tier]

**Description**: Strided load with fixed stride pattern. Loads elements from memory with regular stride pattern. The offset parameter encodes displacement with selected stride mode. This is a deprecated compatibility family.

**Parameters (pointer syntax)**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `buf` | `ptr` | Pointer to buffer in UB memory space (Advanced mode only - requires explicit pointer) |
| `offset` | `Index` | Byte displacement encoded with selected stride mode |
| `stride` | `StrideMode` | Stride mode token: `StrideMode.S3_B16`, `StrideMode.S4_B64`, `StrideMode.S8_B32`, `StrideMode.S2_B64`. Determines which sub-elements are read from each source block. |

**Parameters (element-indexing syntax)**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `tile[row, col]` | `Tile` with indexing | 2D tile with row and column indices (single element) |
| `stride` | `StrideMode` | Stride mode token: `StrideMode.S3_B16`, `StrideMode.S4_B64`, `StrideMode.S8_B32`, `StrideMode.S2_B64`. |
| _or_ | | |
| `tile[pos]` | `Tile` with indexing | 1D tile with element index (single element) |
| `stride` | `StrideMode` | Stride mode token: `StrideMode.S3_B16`, `StrideMode.S4_B64`, `StrideMode.S8_B32`, `StrideMode.S2_B64`. |

**Returns**:
| Return Value | Type | Description |
|--------------|------|-------------|
| `vec` | `VRegType` | Loaded vector with strided pattern |

**Constraints**:
- The selected stride token determines which sub-elements are read from each source block
- This operation family is deprecated; prefer other load patterns when possible

**Examples**:
```python
from pto import StrideMode

# Byte-offset syntax
vec = pto.vsld(ub_ptr, offset, StrideMode.S4_B64)

# Element-indexing syntax
vec = pto.vsld(tile[i, j], StrideMode.S3_B16)
vec = pto.vsld(tile[k], StrideMode.S8_B32)
```

#### `pto.vgather2(buf: ptr, offsets: Index, active_lanes: Index) -> VRegType`  [Advanced Tier]

**Description**: Indexed gather from UB. Gathers elements from a single buffer using per-lane offsets, with participation bounded by active lanes count.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `buf` | `ptr` | Pointer to source buffer in UB memory space |
| `offsets` | `Index` | Per-lane element offsets (vector register) |
| `active_lanes` | `Index` | Number of lanes that participate (bounds gather operation) |

**Returns**:
| Return Value | Type | Description |
|--------------|------|-------------|
| `vec` | `VRegType` | Gathered vector |

**Constraints**:
- Only the first `active_lanes` offsets participate in the gather
- Index element width and interpretation must match selected gather form
- Each effective address must satisfy the gather form's alignment rules

**Example**:
```python
vec = pto.vgather2(buf, offsets, active_lanes)
```

#### `pto.vgather2_bc(buf: ptr, offsets: Index, mask: MaskType) -> VRegType`  [Advanced Tier]

**Description**: Gather with broadcast, conditioned by mask. Gathers elements from a single buffer using per-lane offsets, with mask gating lane participation.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `buf` | `ptr` | Pointer to source buffer in UB memory space |
| `offsets` | `Index` | Per-lane element offsets (vector register) |
| `mask` | `MaskType` | Mask gating which lanes participate |

**Returns**:
| Return Value | Type | Description |
|--------------|------|-------------|
| `vec` | `VRegType` | Gathered vector |

**Constraints**:
- Masked-off lanes do not participate in address coalescing and do not trigger address overflow exceptions
- Destination lanes for masked-off lanes are zero-filled
- This is a backward-compatible operation family

**Example**:
```python
vec = pto.vgather2_bc(buf, offsets, mask)
```

#### `pto.vgatherb(buf: ptr, offsets: Index) -> VRegType`  [Advanced Tier]

**Description**: Byte‑granularity gather load.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `buf` | `ptr` | Pointer to buffer |
| `offsets` | `Index` | Byte offsets |

**Returns**:
| Return Value | Type | Description |
|--------------|------|-------------|
| `vec` | `VRegType` | Gathered vector |

**Example**:
```python
vec = pto.vgatherb(buf, offsets)
```

#### `pto.vsldb(buf: ptr, offset: Index, mask: MaskType) -> VRegType`  [Advanced Tier]
#### `pto.vsldb(tile[row, col], offset: Index, mask: MaskType) -> VRegType`  [Basic Tier]
#### `pto.vsldb(tile[pos], offset: Index, mask: MaskType) -> VRegType`  [Basic Tier]

**Description**: Block-strided load for 2D tile access. Loads elements with block stride pattern controlled by packed offset word and mask.

**Parameters (byte-offset syntax)**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `buf` | `ptr` | Pointer to buffer in UB memory space |
| `offset` | `Index` | Packed stride/control word (not plain byte displacement) |
| `mask` | `MaskType` | Mask controlling which blocks participate |

**Parameters (element-indexing syntax)**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `tile[row, col]` | `Tile` with indexing | 2D tile with row and column indices (single element) |
| `offset` | `Index` | Packed stride/control word (not plain byte displacement) |
| `mask` | `MaskType` | Mask controlling which blocks participate |
| _or_ | | |
| `tile[pos]` | `Tile` with indexing | 1D tile with element index (single element) |
| `offset` | `Index` | Packed stride/control word (not plain byte displacement) |
| `mask` | `MaskType` | Mask controlling which blocks participate |

**Returns**:
| Return Value | Type | Description |
|--------------|------|-------------|
| `vec` | `VRegType` | Loaded vector with block-strided pattern |

**Constraints**:
- The offset encodes block stride and repeat pattern, not a plain byte displacement
- If a block is masked off, the corresponding destination block is zeroed
- Masked-off blocks must not raise address overflow exceptions

**Example**:
```python
# Byte-offset syntax
vec = pto.vsldb(ub_ptr, control_word, mask)

# Element-indexing syntax
vec = pto.vsldb(tile[i, j], control_word, mask)
vec = pto.vsldb(tile[k], control_word, mask)
```

### Vector Store Operations

Operations for storing data from vector registers to memory.

#### `pto.vsts(vec: VRegType, buf: ptr, offset: Index, mask: MaskType) -> None`  [Advanced Tier]
#### `pto.vsts(vec: VRegType, tile[row, col:], mask: MaskType) -> None`  [Basic Tier]
#### `pto.vsts(vec: VRegType, tile[start:], mask: MaskType) -> None`  [Basic Tier]

**Description**: Stateless vector store to buffer. Supports both byte-offset and element-indexing syntax.

**Parameters (byte-offset syntax)**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `vec` | `VRegType` | Vector to store |
| `buf` | `ptr` | Pointer to destination buffer in UB memory space (Advanced mode only - requires explicit pointer) |
| `offset` | `Index` | Byte offset |
| `mask` | `MaskType` | Predicate mask |

**Parameters (element-indexing syntax)**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `vec` | `VRegType` | Vector to store |
| `tile[row, col:]` | `Tile` with indexing | 2D tile with row index and starting column |
| `tile[start:]` | `Tile` with indexing | 1D tile with starting element index |
| `mask` | `MaskType` | Predicate mask |

**Returns**: None (side-effect operation)

**Constraints**:
- Buffer must be in UB memory space
- For byte-offset syntax: offset must be properly aligned based on element type
- For element-indexing syntax: the destination vector region must be within tile bounds and satisfy alignment requirements

**Examples**:
```python
# Byte-offset syntax
pto.vsts(vec_f32, ub_ptr, lane * 256, mask32)

# Element-indexing syntax
pto.vsts(vec, tile[i, j:], mask)      # Store to row i, columns j to j+vector_lanes-1
pto.vsts(vec, tile[k:], mask)         # Store to 1D tile, elements k to k+vector_lanes-1

# In a generic kernel
@pto.vkernel(target="a5", op="copy", dtypes=[(pto.AnyFloat, pto.AnyFloat)], priority=10)
def generic_store(src: pto.Tile, dst: pto.Tile):
    rows, cols = src.shape
    all_mask = pto.make_mask(src.element_type, PAT.ALL)
    for i in range(0, rows):
        for j in range(0, cols, vector_lanes):
            vec = pto.vlds(src[i, j:])
            pto.vsts(vec, dst[i, j:], all_mask)  # No manual offset calculation
```

#### `pto.psts(mask: MaskType, buf: ptr, offset: Index) -> None`  [Advanced Tier]
#### `pto.psts(mask: MaskType, tile[row, col:]) -> None`  
#### `pto.psts(mask: MaskType, tile[start:]) -> None`

**Description**: Predicate store to buffer. Supports both traditional byte-offset syntax and new element-indexing syntax.

**Parameters (byte-offset syntax)**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `mask` | `MaskType` | Mask to store |
| `buf` | `ptr` | Pointer to destination buffer (Advanced mode only - requires explicit pointer) |
| `offset` | `Index` | Byte offset |

**Parameters (element-indexing syntax)**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `mask` | `MaskType` | Mask to store |
| `tile[row, col:]` | `Tile` with indexing | 2D tile with row index and starting column (vector-width range) |

**Parameters (1D element-indexing syntax)**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `mask` | `MaskType` | Mask to store |
| `tile[start:]` | `Tile` with indexing | 1D tile with starting element index (vector-width range) |

**Returns**: None (side-effect operation)

#### `pto.vsst(scalar: ScalarType, buf: ptr, offset: Index, mask: MaskType) -> None`  [Advanced Tier]
#### `pto.vsst(scalar: ScalarType, tile[row, col:], mask: MaskType) -> None`  
#### `pto.vsst(scalar: ScalarType, tile[start:], mask: MaskType) -> None`

**Description**: Scalar to vector store (broadcast scalar to all lanes). Supports both traditional byte-offset syntax and new element-indexing syntax.

**Parameters (byte-offset syntax)**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `scalar` | `ScalarType` | Scalar value |
| `buf` | `ptr` | Pointer to destination buffer (Advanced mode only - requires explicit pointer) |
| `offset` | `Index` | Byte offset |
| `mask` | `MaskType` | Predicate mask |

**Parameters (element-indexing syntax)**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `scalar` | `ScalarType` | Scalar value |
| `tile[row, col:]` | `Tile` with indexing | 2D tile with row index and starting column (vector-width range) |
| `mask` | `MaskType` | Predicate mask |

**Parameters (1D element-indexing syntax)**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `scalar` | `ScalarType` | Scalar value |
| `tile[start:]` | `Tile` with indexing | 1D tile with starting element index (vector-width range) |
| `mask` | `MaskType` | Predicate mask |

**Returns**: None (side-effect operation)

#### `pto.vstx2(low: VRegType, high: VRegType, buf: ptr, offset: Index, dist: InterleaveDist, mask: MaskType) -> None`  [Advanced Tier]
#### `pto.vstx2(low: VRegType, high: VRegType, tile[row, col:], dist: InterleaveDist, mask: MaskType) -> None`  
#### `pto.vstx2(low: VRegType, high: VRegType, tile[start:], dist: InterleaveDist, mask: MaskType) -> None`

**Description**: Dual interleaved store (SoA → AoS conversion). Stores two vectors interleaved into a single buffer. Supports both byte-offset and element-indexing syntax.

**Parameters (byte-offset syntax)**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `low` | `VRegType` | First vector (even elements in interleaved stream) |
| `high` | `VRegType` | Second vector (odd elements in interleaved stream) |
| `buf` | `ptr` | Pointer to destination buffer in UB memory space (Advanced mode only - requires explicit pointer) |
| `offset` | `Index` | Byte offset |
| `dist` | `InterleaveDist` | Interleave distribution mode: `InterleaveDist.B8`, `InterleaveDist.B16`, `InterleaveDist.B32`. Determines element size and interleave pattern. |
| `mask` | `MaskType` | Predicate mask |

**Parameters (element-indexing syntax)**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `low` | `VRegType` | First vector (even elements in interleaved stream) |
| `high` | `VRegType` | Second vector (odd elements in interleaved stream) |
| `tile[row, col:]` | `Tile` with indexing | 2D tile with row index and starting column (vector-width range) |
| `dist` | `InterleaveDist` | Interleave distribution mode: `InterleaveDist.B8`, `InterleaveDist.B16`, `InterleaveDist.B32`. |
| `mask` | `MaskType` | Predicate mask |

**Parameters (1D element-indexing syntax)**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `low` | `VRegType` | First vector (even elements in interleaved stream) |
| `high` | `VRegType` | Second vector (odd elements in interleaved stream) |
| `tile[start:]` | `Tile` with indexing | 1D tile with starting element index (vector-width range) |
| `dist` | `InterleaveDist` | Interleave distribution mode: `InterleaveDist.B8`, `InterleaveDist.B16`, `InterleaveDist.B32`. |
| `mask` | `MaskType` | Predicate mask |

**Returns**: None (side-effect operation)

**Constraints**:
- Destination buffer must be in UB memory space
- Offset must satisfy alignment requirements for the selected distribution mode
- The destination vector region must be within tile bounds (for element-indexing syntax)
- Distribution mode must match element type (e.g., `InterleaveDist.B32` for 32-bit elements)
- The two source vectors form an ordered pair; interleave semantics must be preserved

**Examples**:
```python
from pto import InterleaveDist

# Byte-offset syntax
pto.vstx2(x_vec, y_vec, ub_ptr, offset, InterleaveDist.B32, mask)

# Element-indexing syntax
pto.vstx2(x_vec, y_vec, tile[i, j:], InterleaveDist.B32, mask)
pto.vstx2(x_vec, y_vec, tile[k:], InterleaveDist.B16, mask)

# Example: Store separate X/Y vectors as interleaved XY pairs
pto.vstx2(x_vec, y_vec, xy_tile[i, j:], InterleaveDist.B32, all_mask)
```

#### `pto.vsta(align: pto.align, buf: ptr, offset: Index) -> None`  [Advanced Tier]
#### `pto.vsta(align: pto.align, tile[row, col:]) -> None`  [Basic Tier]
#### `pto.vsta(align: pto.align, tile[start:]) -> None`  [Basic Tier]

**Description**: Flush alignment state to memory. Writes buffered tail bytes from alignment state to UB memory. Consumes the alignment state after flush.

**Parameters (byte-offset syntax)**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `align` | `pto.align` | Pending store-alignment state |
| `buf` | `ptr` | Pointer to destination buffer in UB memory space (Advanced mode only - requires explicit pointer) |
| `offset` | `Index` | Flush displacement |

**Parameters (element-indexing syntax)**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `align` | `pto.align` | Pending store-alignment state |
| `tile[row, col:]` | `Tile` with indexing | 2D tile with row index and starting column (vector-width range) |
| _or_ | | |
| `align` | `pto.align` | Pending store-alignment state |
| `tile[start:]` | `Tile` with indexing | 1D tile with starting element index (vector-width range) |

**Returns**: None (side-effect operation)

**Constraints**:
- The flush address must match the post-updated address expected by the preceding unaligned-store stream
- After the flush, the corresponding store alignment state is consumed
- A final flush operation is required to commit buffered bytes after unaligned-store sequences
- The `align` input should come from the latest `vstu`/`vstus`/`vstur` in the same stream

**Example**:
```python
# Byte-offset syntax
pto.vsta(align, ub_ptr, offset)

# Element-indexing syntax
pto.vsta(align, tile[i, j:])
pto.vsta(align, tile[k:])
```

#### `pto.vscatter(vec: VRegType, buf: ptr, offsets: Index, active_lanes: Index) -> None`  [Advanced Tier]

**Description**: Indexed scatter to UB. Stores vector elements to irregular locations using per-lane offsets, with participation bounded by active lanes count.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `vec` | `VRegType` | Source vector to scatter |
| `buf` | `ptr` | Pointer to destination buffer in UB memory space |
| `offsets` | `Index` | Per-lane element offsets (vector register) |
| `active_lanes` | `Index` | Number of lanes that participate (bounds scatter operation) |

**Returns**: None (side-effect operation)

**Constraints**:
- Only `b8`, `b16`, and `b32` element sizes are supported
- Index vector must use a supported integer element type and layout
- Each computed address must be element-aligned
- If indices alias, only one write is guaranteed (winning lane is implementation-defined)
- Only the first `active_lanes` offsets participate in the scatter

**Example**:
```python
pto.vscatter(vec, buf, offsets, active_lanes)
```

#### `pto.vsstb(scalar: ScalarType, buf: ptr, offset: Index, mask: MaskType) -> None`  [Advanced Tier]
#### `pto.vsstb(scalar: ScalarType, tile[row, col:], mask: MaskType) -> None`  [Basic Tier]
#### `pto.vsstb(scalar: ScalarType, tile[start:], mask: MaskType) -> None`  [Basic Tier]

**Description**: Scalar to vector store with broadcast (enhanced version of `vsst`). Supports both byte‑offset and element‑indexing syntax.

**Parameters (pointer syntax)**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `scalar` | `ScalarType` | Scalar value |
| `buf` | `ptr` | Pointer to destination buffer |
| `offset` | `Index` | Byte offset |
| `mask` | `MaskType` | Predicate mask |

**Parameters (element-indexing syntax)**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `scalar` | `ScalarType` | Scalar value |
| `tile[row, col:]` | `Tile` with indexing | 2D tile with row index and starting column (vector-width range) |
| `mask` | `MaskType` | Predicate mask |

**Parameters (1D element-indexing syntax)**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `scalar` | `ScalarType` | Scalar value |
| `tile[start:]` | `Tile` with indexing | 1D tile with starting element index (vector-width range) |
| `mask` | `MaskType` | Predicate mask |

**Returns**: None (side-effect operation)

**Example**:
```python
# Byte-offset syntax
pto.vsstb(pto.f32(0.0), ub_ptr, offset, mask)

# Element-indexing syntax
pto.vsstb(pto.f32(1.0), tile[i, j:], mask)
```

#### `pto.vstar(align: pto.align, buf: ptr) -> None`  [Advanced Tier]
#### `pto.vstar(align: pto.align, tile[row, col:]) -> None`  [Basic Tier]
#### `pto.vstar(align: pto.align, tile[start:]) -> None`  [Basic Tier]

**Description**: Flush alignment state using the register-update form. Writes buffered tail bytes from alignment state to UB memory. The implicit update state must correspond to the same store stream that produced the alignment state.

**Parameters (byte-offset syntax)**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `align` | `pto.align` | Pending store-alignment state |
| `buf` | `ptr` | Pointer to destination buffer in UB memory space |

**Parameters (element-indexing syntax)**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `align` | `pto.align` | Pending store-alignment state |
| `tile[row, col:]` | `Tile` with indexing | 2D tile with row index and starting column (vector-width range) |
| _or_ | | |
| `align` | `pto.align` | Pending store-alignment state |
| `tile[start:]` | `Tile` with indexing | 1D tile with starting element index (vector-width range) |

**Returns**: None (side-effect operation)

**Constraints**:
- The implicit update state consumed by this flush must correspond to the same store stream that produced the alignment state
- A final flush operation is required to commit buffered bytes after unaligned-store sequences
- The `align` input should come from the latest `vstu`/`vstus`/`vstur` in the same stream

**Example**:
```python
# Byte-offset syntax
pto.vstar(align, ub_ptr)

# Element-indexing syntax
pto.vstar(align, tile[i, j:])
pto.vstar(align, tile[k:])
```

#### `pto.vstas(align: pto.align, buf: ptr, offset: Index) -> None`  [Advanced Tier]
#### `pto.vstas(align: pto.align, tile[row, col:], offset: Index) -> None`  [Basic Tier]
#### `pto.vstas(align: pto.align, tile[start:], offset: Index) -> None`  [Basic Tier]

**Description**: Scalar-register-offset form of alignment-state flush. Writes buffered tail bytes from alignment state to UB memory with explicit scalar offset. Uses same buffered-tail semantics as `pto.vsta`.

**Parameters (pointer syntax)**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `align` | `pto.align` | Pending store-alignment state |
| `buf` | `ptr` | Pointer to destination buffer in UB memory space |
| `offset` | `Index` | Scalar-register style displacement |

**Parameters (element-indexing syntax)**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `align` | `pto.align` | Pending store-alignment state |
| `tile[row, col:]` | `Tile` with indexing | 2D tile with row index and starting column (vector-width range) |
| `offset` | `Index` | Scalar-register style displacement |
| _or_ | | |
| `align` | `pto.align` | Pending store-alignment state |
| `tile[start:]` | `Tile` with indexing | 1D tile with starting element index (vector-width range) |
| `offset` | `Index` | Scalar-register style displacement |

**Returns**: None (side-effect operation)

**Example**:
```python
# Byte-offset syntax
pto.vstas(align, ub_ptr, offset)

# Element-indexing syntax
pto.vstas(align, tile[i, j:], offset)
pto.vstas(align, tile[k:], offset)
```

### Stateful Store Operations

Operations for storing data with stateful semantics.

#### `pto.pstu(align_in: pto.align, mask: MaskType, buf: ptr) -> (pto.align, ptr)`  [Advanced Tier]

**Description**: Predicate unaligned store with align state update. Stores predicate mask with alignment state threading.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `align_in` | `pto.align` | Incoming store-alignment state |
| `mask` | `MaskType` | Predicate mask to store |
| `buf` | `ptr` | Pointer to destination buffer in UB memory space |

**Returns**:
| Return Value | Type | Description |
|--------------|------|-------------|
| `align_out` | `pto.align` | Updated alignment state |
| `base_out` | `ptr` | Post-update base pointer state |

**Constraints**:
- Part of stateful unaligned-store sequence with alignment state threading

#### `pto.vstu(align_in: pto.align, base_in: ptr, vec: VRegType, buf: ptr, mode: Index) -> (pto.align, ptr)`  [Advanced Tier]

**Description**: Unaligned store with explicit threaded alignment/base state. Models a stateful unaligned-store sequence in SSA form. Requires a final `pto.vsta`/`pto.vstas`/`pto.vstar` to flush trailing buffered bytes.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `align_in` | `pto.align` | Incoming store-alignment state |
| `base_in` | `ptr` | Current stream base pointer |
| `vec` | `VRegType` | Vector to store |
| `buf` | `ptr` | Destination buffer in UB memory space |
| `mode` | `Index` | Mode selecting post-update behavior |

**Returns**:
| Return Value | Type | Description |
|--------------|------|-------------|
| `align_out` | `pto.align` | Updated buffered-tail state |
| `base_out` | `ptr` | Post-update base pointer state |

**Constraints**:
- Models stateful unaligned-store sequence in SSA form
- Final flush operation required to commit buffered bytes

**Example**:
```python
# Stateful unaligned store + final flush (vsta form)
align1, base1 = pto.vstu(align0, base0, vec0, ub_ptr, mode)
align2, base2 = pto.vstu(align1, base1, vec1, ub_ptr, mode)
pto.vsta(align2, ub_ptr, tail_offset)
```

#### `pto.vstus(align_in: pto.align, base_in: ptr, vec: VRegType, buf: ptr, offset: Index) -> (pto.align, ptr)`  [Advanced Tier]

**Description**: Scalar-offset unaligned store with threaded state. Same roles as `pto.vstu` but with explicit scalar displacement.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `align_in` | `pto.align` | Incoming store-alignment state |
| `base_in` | `ptr` | Current stream base pointer |
| `vec` | `VRegType` | Vector to store |
| `buf` | `ptr` | Destination buffer in UB memory space |
| `offset` | `Index` | Scalar displacement |

**Returns**:
| Return Value | Type | Description |
|--------------|------|-------------|
| `align_out` | `pto.align` | Updated buffered-tail state |
| `base_out` | `ptr` | Post-update base pointer state |

**Constraints**:
- Same final flush requirement and state-threading constraints as `pto.vstu`

**Example**:
```python
# Scalar-offset threaded form + final flush (vstas form)
align1, base1 = pto.vstus(align0, base0, vec0, ub_ptr, offset0)
align2, base2 = pto.vstus(align1, base1, vec1, ub_ptr, offset1)
pto.vstas(align2, ub_ptr, flush_offset)
```

#### `pto.vstur(align_in: pto.align, vec: VRegType, buf: ptr) -> pto.align`  [Advanced Tier]

**Description**: Register-update unaligned store form. Updates only the residual alignment state without base pointer update. Requires matching flush operation to emit trailing bytes.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `align_in` | `pto.align` | Incoming store-alignment state |
| `vec` | `VRegType` | Vector to store |
| `buf` | `ptr` | Destination buffer in UB memory space |

**Returns**:
| Return Value | Type | Description |
|--------------|------|-------------|
| `align_out` | `pto.align` | Updated buffered-tail state |

**Constraints**:
- Updates only residual alignment state (no base pointer update)
- Matching flush operation still required to emit trailing bytes

**Example**:
```python
# Residual-state form + final flush (vstar form)
align1 = pto.vstur(align0, vec0, ub_ptr)
align2 = pto.vstur(align1, vec1, ub_ptr)
pto.vstar(align2, ub_ptr)
```

#### Align-State Store Closed Loop

For unaligned store families, the state must form a closed loop:

1. Start from an incoming `align` state.
2. Thread state through one or more `vstu` / `vstus` / `vstur` operations.
3. Terminate the stream with exactly one flush op: `vsta` or `vstas` or `vstar`.
4. Do not reuse a flushed `align` state in another stream.
