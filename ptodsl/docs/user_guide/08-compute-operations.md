# 8. Compute Operations

Chapters 6 and 7 covered scalars, pointers, and data movement. This chapter covers everything that actually *computes* — arithmetic, math functions, reductions, comparisons, and matrix multiplication — organized by abstraction level: tile ops (L1), vector ops (L3 SIMD), and cube ops (L3 cube).

## 8.1 Tile-level compute (L1)

Tile compute ops are the primary arithmetic surface inside `@pto.jit`. They operate on `Tile` buffers in UB and follow a consistent pattern: each op reads one or more source tiles, optionally a scalar, and writes a destination tile. Shapes and valid regions must be compatible across all operands.

### 8.1.1 Binary tile-tile arithmetic

Element-wise operations between two tiles of the same shape.

#### `pto.tile.add(src0: Tile, src1: Tile, dst: Tile) -> None`
#### `pto.tile.sub(src0: Tile, src1: Tile, dst: Tile) -> None`
#### `pto.tile.mul(src0: Tile, src1: Tile, dst: Tile) -> None`
#### `pto.tile.max(src0: Tile, src1: Tile, dst: Tile) -> None`
#### `pto.tile.min(src0: Tile, src1: Tile, dst: Tile) -> None`

**Description**: Element-wise `dst[i,j] = src0[i,j] <op> src1[i,j]`.

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `src0` | `Tile` | First source tile |
| `src1` | `Tile` | Second source tile |
| `dst` | `Tile` | Destination tile (must be pre-allocated, shape-compatible) |

**Returns**: None (writes to `dst`).

**Example**:

```python
pto.tile.add(a_tile, b_tile, o_tile)
pto.tile.mul(scale_tile, data_tile, scaled_tile)
```

---

#### `pto.tile.div(src0: Tile, src1: Tile, dst: Tile, *, div_precision: DivPrecision = DivPrecision.Default) -> None`

**Description**: Element-wise division. `div_precision` can be `Default` or `HighPrecision` (f16/f32 only).

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `src0` | `Tile` | Numerator tile |
| `src1` | `Tile` | Denominator tile |
| `dst` | `Tile` | Destination tile |
| `div_precision` | `DivPrecision` | `Default` (default) or `HighPrecision` |

**Returns**: None.

---

### 8.1.2 Tile-scalar arithmetic

Element-wise operations between a tile and a scalar.

#### `pto.tile.adds(src: Tile, scalar: ScalarType, dst: Tile) -> None`
#### `pto.tile.subs(src: Tile, scalar: ScalarType, dst: Tile) -> None`
#### `pto.tile.muls(src: Tile, scalar: ScalarType, dst: Tile) -> None`
#### `pto.tile.maxs(src: Tile, scalar: ScalarType, dst: Tile) -> None`
#### `pto.tile.mins(src: Tile, scalar: ScalarType, dst: Tile) -> None`

**Description**: Element-wise `dst[i,j] = src[i,j] <op> scalar`.

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `src` | `Tile` | Source tile |
| `scalar` | `ScalarType` | Scalar operand (Python number or PTO scalar) |
| `dst` | `Tile` | Destination tile |

**Returns**: None.

---

#### `pto.tile.divs(src: Tile, scalar: ScalarType, dst: Tile, *, div_precision: DivPrecision = DivPrecision.Default) -> None`

**Description**: Element-wise tile-scalar division: `dst[i,j] = src[i,j] / scalar`.

---

### 8.1.3 Unary math

Single-source element-wise math functions.

#### `pto.tile.exp(src: Tile, dst: Tile, *, exp_precision: ExpPrecision = ExpPrecision.Default) -> None`
#### `pto.tile.log(src: Tile, dst: Tile, *, log_precision: LogPrecision = LogPrecision.Default) -> None`
#### `pto.tile.sqrt(src: Tile, dst: Tile, *, sqrt_precision: SqrtPrecision = SqrtPrecision.Default) -> None`
#### `pto.tile.rsqrt(src: Tile, dst: Tile, *, rsqrt_precision: RsqrtPrecision = RsqrtPrecision.Default) -> None`
#### `pto.tile.recip(src: Tile, dst: Tile, *, recip_precision: RecipPrecision = RecipPrecision.Default) -> None`

**Description**: Element-wise `exp`, `ln`, `sqrt`, `1/sqrt`, `1/x`.

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `src` | `Tile` | Source tile |
| `dst` | `Tile` | Destination tile |
| `*_precision` | op-specific precision enum | `Default` or `HighPrecision` |

**Returns**: None.

---

#### `pto.tile.abs(src: Tile, dst: Tile) -> None`
#### `pto.tile.neg(src: Tile, dst: Tile) -> None`

**Description**: Element-wise absolute value and negation. No precision mode attribute.

---

### 8.1.4 Activation

#### `pto.tile.relu(src: Tile, dst: Tile) -> None`

**Description**: `dst[i,j] = max(0, src[i,j])`. Supported on f16, f32, i32.

#### `pto.tile.lrelu(src: Tile, slope: float, dst: Tile) -> None`

**Description**: Leaky ReLU — `dst[i,j] = src[i,j] >= 0 ? src[i,j] : slope * src[i,j]`.

---

### 8.1.5 Row and column reductions

Reductions collapse one dimension of a 2D tile, producing a tile with one row or one column.

#### Row reductions

#### `pto.tile.rowsum(src: Tile, dst: Tile, *, tmp: Tile | None = None) -> None`
#### `pto.tile.rowmax(src: Tile, dst: Tile, *, tmp: Tile | None = None) -> None`
#### `pto.tile.rowmin(src: Tile, dst: Tile, *, tmp: Tile | None = None) -> None`
#### `pto.tile.rowprod(src: Tile, dst: Tile, *, tmp: Tile | None = None) -> None`
#### `pto.tile.rowargmax(src: Tile, dst: Tile, *, tmp: Tile | None = None) -> None`
#### `pto.tile.rowargmin(src: Tile, dst: Tile, *, tmp: Tile | None = None) -> None`

**Description**: For each row `i`, reduce across columns: `dst[i, 0] = <reduce>_j src[i, j]`. `tile.rowargmax`/`tile.rowargmin` return the column index of the extremum. In the public PTODSL wrapper, `tmp` is optional; when omitted, PTODSL allocates a matching scratch tile automatically.

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `src` | `Tile` | Source tile (`[rows, cols]`) |
| `dst` | `Tile` | Destination tile (`[rows, 1]`) |
| `tmp` | `Tile | None` | Optional scratch tile for intermediate reduction state; when omitted, PTODSL synthesizes a matching scratch tile automatically |

**Returns**: None.

---

#### Column reductions

#### `pto.tile.colsum(src: Tile, dst: Tile) -> None`
#### `pto.tile.colmax(src: Tile, dst: Tile) -> None`
#### `pto.tile.colmin(src: Tile, dst: Tile) -> None`
#### `pto.tile.colprod(src: Tile, dst: Tile) -> None`

**Description**: For each column `j`, reduce across rows: `dst[0, j] = <reduce>_i src[i, j]`.

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `src` | `Tile` | Source tile (`[rows, cols]`) |
| `dst` | `Tile` | Destination tile (`[1, cols]`) |

**Returns**: None.

---

### 8.1.6 Broadcast and expansion

Expansion ops take a narrow source (scalar, row vector, or column vector) and broadcast it to a full tile shape. They are useful for applying per-row or per-column coefficients to a tile.

#### Scalar broadcast

#### `pto.tile.expands(scalar: ScalarType, dst: Tile) -> None`

**Description**: `dst[i,j] = scalar` — fills every element of `dst` with the same scalar value.

---

#### Row expansion

#### `pto.tile.rowexpand(src: Tile, dst: Tile) -> None`

**Description**: `dst[row, col] = src[row, 0]` — broadcasts each row's single value across all columns of `dst`.

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `src` | `Tile` | Source tile (`[rows, 1]`) |
| `dst` | `Tile` | Destination tile (`[rows, cols]`) |

**Returns**: None.

---

#### Column expansion

#### `pto.tile.colexpand(src: Tile, dst: Tile) -> None`

**Description**: `dst[row, col] = src[0, col]` — broadcasts each column's single value across all rows of `dst`.

---

#### Row-expand arithmetic

These combine broadcasting with an arithmetic operation: `src1` is a per-row coefficient tile (`[rows, 1]`) that gets expanded row-wise before the element-wise op with `src0`.

| Op | Semantics |
|----|-----------|
| `pto.tile.rowexpandadd(src0, src1, dst)` | `dst = src0 + expand_rows(src1)` |
| `pto.tile.rowexpandsub(src0, src1, dst)` | `dst = src0 - expand_rows(src1)` |
| `pto.tile.rowexpandmul(src0, src1, dst)` | `dst = src0 * expand_rows(src1)` |
| `pto.tile.rowexpanddiv(src0, src1, dst)` | `dst = src0 / expand_rows(src1)` (f-only) |
| `pto.tile.rowexpandmax(src0, src1, dst)` | `dst = max(src0, expand_rows(src1))` |
| `pto.tile.rowexpandmin(src0, src1, dst)` | `dst = min(src0, expand_rows(src1))` |
| `pto.tile.rowexpandexpdif(src0, src1, dst)` | `dst = exp(src0 - expand_rows(src1))` (f-only) |

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `src0` | `Tile` | Full-shape source tile (`[rows, cols]`) |
| `src1` | `Tile` | Per-row coefficient tile (`[rows, 1]`) |
| `dst` | `Tile` | Destination tile (`[rows, cols]`) |

**Returns**: None.

**Example** — apply per-row scale and bias:

```python
# alpha_tile: [rows, 1], beta_tile: [rows, 1], data_tile: [rows, cols]
pto.tile.rowexpandmul(data_tile, alpha_tile, scaled_tile)
pto.tile.rowexpandadd(scaled_tile, beta_tile, result_tile)
```

---

#### Column-expand arithmetic

Same pattern as row-expand arithmetic, but `src1` is a per-column coefficient tile (`[1, cols]`):

| Op | Semantics |
|----|-----------|
| `pto.tile.colexpandadd(src0, src1, dst)` | `dst = src0 + expand_cols(src1)` |
| `pto.tile.colexpandsub(src0, src1, dst)` | `dst = src0 - expand_cols(src1)` |
| `pto.tile.colexpandmul(src0, src1, dst)` | `dst = src0 * expand_cols(src1)` |
| `pto.tile.colexpanddiv(src0, src1, dst)` | `dst = src0 / expand_cols(src1)` (f-only) |
| `pto.tile.colexpandmax(src0, src1, dst)` | `dst = max(src0, expand_cols(src1))` |
| `pto.tile.colexpandmin(src0, src1, dst)` | `dst = min(src0, expand_cols(src1))` |
| `pto.tile.colexpandexpdif(src0, src1, dst)` | `dst = exp(src0 - expand_cols(src1))` (f-only) |

---

### 8.1.7 Selection

#### `pto.tile.sel(mask: Tile, src0: Tile, src1: Tile, dst: Tile, *, tmp: Tile | None = None) -> None`

**Description**: Element-wise ternary: `dst[i,j] = mask[i,j] ? src0[i,j] : src1[i,j]`. The `mask` is an integer tile where zero means false and non-zero means true. `tmp` is an optional scratch tile override; when omitted, PTODSL synthesizes any architecture-specific scratch tile automatically.

#### `pto.tile.sels(mask: Tile, src: Tile, scalar: ScalarType, dst: Tile, *, tmp: Tile | None = None) -> None`

**Description**: Element-wise select with scalar fallback: `dst[i,j] = mask[i,j] ? src[i,j] : scalar`. As with `tile.sel`, `tmp` is optional and PTODSL synthesizes any required scratch tile automatically when it is omitted.

---

### 8.1.8 Type conversion

#### `pto.tile.cvt(src: Tile, dst: Tile, *, rmode: RoundMode = RoundMode.NONE) -> None`

**Description**: Element-wise type conversion. The destination tile's `dtype` determines the target type.

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `src` | `Tile` | Source tile |
| `dst` | `Tile` | Destination tile (with target dtype) |
| `rmode` | `RoundMode` | Rounding mode: `NONE`, `RINT`, `ROUND`, `FLOOR`, `CEIL`, `TRUNC`, `ODD`, `CAST_RINT` |

**Returns**: None.

---

### 8.1.9 Bitwise ops

Bitwise operations on integer tiles (i8, i16, i32, etc.). All follow the standard `(src, dst)` or `(src0, src1, dst)` pattern.

#### Unary bitwise

#### `pto.tile.bit_not(src: Tile, dst: Tile) -> None`

**Description**: Element-wise bitwise NOT: `dst[i,j] = ~src[i,j]`. Integer types only.

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `src` | `Tile` | Source tile (integer dtype) |
| `dst` | `Tile` | Destination tile |

**Returns**: None.

---

#### Binary bitwise (tile-tile)

#### `pto.tile.bit_and(src0: Tile, src1: Tile, dst: Tile) -> None`
#### `pto.tile.bit_or(src0: Tile, src1: Tile, dst: Tile) -> None`
#### `pto.tile.bit_shl(src0: Tile, src1: Tile, dst: Tile) -> None`
#### `pto.tile.bit_shr(src0: Tile, src1: Tile, dst: Tile) -> None`

**Description**: Element-wise bitwise `dst[i,j] = src0[i,j] <op> src1[i,j]`. Integer types only.

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `src0` | `Tile` | First source tile |
| `src1` | `Tile` | Second source tile |
| `dst` | `Tile` | Destination tile |

**Returns**: None.

---

#### `pto.tile.bit_xor(src0: Tile, src1: Tile, dst: Tile, *, tmp: Tile | None = None) -> None`

**Description**: Element-wise bitwise XOR. Requires an additional scratch buffer `tmp` of the same type as `dst`. When `tmp` is omitted, PTODSL synthesizes a matching scratch tile automatically.

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `src0` | `Tile` | First source tile |
| `src1` | `Tile` | Second source tile |
| `dst` | `Tile` | Destination tile |
| `tmp` | `Tile | None` | Optional scratch tile; when omitted, PTODSL synthesizes one automatically |

**Returns**: None.

---

#### Binary bitwise (tile-scalar)

#### `pto.tile.bit_ands(src: Tile, scalar: ScalarType, dst: Tile) -> None`
#### `pto.tile.bit_ors(src: Tile, scalar: ScalarType, dst: Tile) -> None`
#### `pto.tile.bit_shls(src: Tile, scalar: ScalarType, dst: Tile) -> None`
#### `pto.tile.bit_shrs(src: Tile, scalar: ScalarType, dst: Tile) -> None`

**Description**: Element-wise `dst[i,j] = src[i,j] <op> scalar`. The scalar is broadcast to all elements. Integer types only.

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `src` | `Tile` | Source tile |
| `scalar` | `ScalarType` | Scalar operand (Python int or PTO scalar) |
| `dst` | `Tile` | Destination tile |

**Returns**: None.

---

#### `pto.tile.bit_xors(src: Tile, scalar: ScalarType, dst: Tile, *, tmp: Tile | None = None) -> None`

**Description**: Element-wise bitwise XOR with scalar. Requires an additional scratch buffer `tmp` of the same type as `dst`. When `tmp` is omitted, PTODSL synthesizes a matching scratch tile automatically.

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `src` | `Tile` | Source tile |
| `scalar` | `ScalarType` | Scalar operand |
| `dst` | `Tile` | Destination tile |
| `tmp` | `Tile | None` | Optional scratch tile; when omitted, PTODSL synthesizes one automatically |

**Returns**: None.

---

### 8.1.10 Partial elementwise ops

Partial elementwise ops compute over the **intersection** of the valid regions of two source tiles. This allows element-wise arithmetic between tiles that have different `valid_shape`s — only the overlapping area is computed.

#### `pto.tile.partadd(src0: Tile, src1: Tile, dst: Tile) -> None`
#### `pto.tile.partmul(src0: Tile, src1: Tile, dst: Tile) -> None`
#### `pto.tile.partmax(src0: Tile, src1: Tile, dst: Tile) -> None`
#### `pto.tile.partmin(src0: Tile, src1: Tile, dst: Tile) -> None`

**Description**: Element-wise `dst[i,j] = src0[i,j] <op> src1[i,j]` over the intersection of `src0.valid_shape` and `src1.valid_shape`.

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `src0` | `Tile` | First source tile (may have a partial valid region) |
| `src1` | `Tile` | Second source tile (may have a partial valid region) |
| `dst` | `Tile` | Destination tile |

**Returns**: None.

**Example** — adding tiles with different valid regions:

```python
# a_tile: valid_shape = [64, 32], b_tile: valid_shape = [64, 64]
# The partial add only operates on the intersection: 64 columns × min(32, 64) = 32 columns
pto.tile.partadd(a_tile, b_tile, result_tile)
```

---

### 8.1.11 Fill/padding

Fill-padding ops copy a source tile's valid region into a destination tile, filling the remaining physical elements (outside `src.valid_shape`) with a configured pad value. The pad value is specified at tile allocation time via the tile's `PadValue` attribute (`Null`, `Zero`, `Max`, or `Min`).

#### `pto.tile.fillpad(src: Tile, dst: Tile) -> None`

**Description**: Copies `src`'s valid region into `dst` and fills extra elements of `dst` with the pad value configured on `dst`'s type. The `dst` physical shape must be at least as large as `src.valid_shape`.

#### `pto.tile.fillpad_expand(src: Tile, dst: Tile) -> None`

**Description**: Like `fillpad`, but the destination tile may have a different shape in the partition/tensor view. The src valid region is copied and the expanded area is filled with the pad value. Useful when expanding a tile into a larger buffer for downstream processing.

#### `pto.tile.fillpad_inplace(src: Tile, dst: Tile) -> None`

**Description**: In-place variant of `fillpad`. `src` and `dst` may refer to the same tile buffer, padding the tile's own valid region in place.

**Parameters** (all three ops):

| Parameter | Type | Description |
|-----------|------|-------------|
| `src` | `Tile` | Source tile (with valid region to copy) |
| `dst` | `Tile` | Destination tile (carries `PadValue` attribute set at allocation) |

**Returns**: None.

**Example** — padding a partial tile to full shape:

```python
# tile has valid_shape [32, 16] in a physical buffer of [32, 32]
# pad=Zero at allocation time fills extra columns with zeros
pto.tile.fillpad(partial_tile, padded_tile)
```

---

### 8.1.12 Tile compute quick reference

| Category | Operations |
|----------|------------|
| Binary tile-tile | `tile.add`, `tile.sub`, `tile.mul`, `tile.div`, `tile.max`, `tile.min` |
| Tile-scalar | `tile.adds`, `tile.subs`, `tile.muls`, `tile.divs`, `tile.maxs`, `tile.mins` |
| Unary math | `tile.exp`, `tile.log`, `tile.sqrt`, `tile.rsqrt`, `tile.recip`, `tile.abs`, `tile.neg` |
| Activation | `tile.relu`, `tile.lrelu` |
| Row reductions | `tile.rowsum`, `tile.rowmax`, `tile.rowmin`, `tile.rowprod`, `tile.rowargmax`, `tile.rowargmin` |
| Column reductions | `tile.colsum`, `tile.colmax`, `tile.colmin`, `tile.colprod` |
| Broadcast | `tile.expands`, `tile.rowexpand`, `tile.colexpand` |
| Row-expand arith | `tile.rowexpandadd`, `tile.rowexpandsub`, `tile.rowexpandmul`, `tile.rowexpanddiv`, `tile.rowexpandmax`, `tile.rowexpandmin`, `tile.rowexpandexpdif` |
| Col-expand arith | `tile.colexpandadd`, `tile.colexpandsub`, `tile.colexpandmul`, `tile.colexpanddiv`, `tile.colexpandmax`, `tile.colexpandmin`, `tile.colexpandexpdif` |
| Selection | `tile.sel`, `tile.sels` |
| Type conversion | `tile.cvt` |
| Bitwise | `tile.bit_not`, `tile.bit_and`, `tile.bit_or`, `tile.bit_xor`, `tile.bit_shl`, `tile.bit_shr`, `tile.bit_ands`, `tile.bit_ors`, `tile.bit_xors`, `tile.bit_shls`, `tile.bit_shrs` |
| Partial elementwise | `tile.partadd`, `tile.partmul`, `tile.partmax`, `tile.partmin` |
| Fill/padding | `tile.fillpad`, `tile.fillpad_expand`, `tile.fillpad_inplace` |

---

## 8.2 Vector compute (L3 — `@pto.simd`)

Vector compute ops operate on `VRegType` values inside `@pto.simd` sub-kernels. Every vector op takes a `MaskType` predicate that gates which lanes participate; masked-off lanes produce an unspecified result (use the result only where the mask is true, or feed it to a masked store).

All vector ops in this section follow the pattern established in Section 7.3 for tile-index and pointer-form addressing. The signatures below use the vector-register form — tile-index forms load into `vreg` first, then compute.

### 8.2.1 Unary vector ops

#### `pto.vexp(vec: VRegType, mask: MaskType) -> VRegType`
#### `pto.vln(vec: VRegType, mask: MaskType) -> VRegType`
#### `pto.vsqrt(vec: VRegType, mask: MaskType) -> VRegType`
#### `pto.vabs(vec: VRegType, mask: MaskType) -> VRegType`
#### `pto.vneg(vec: VRegType, mask: MaskType) -> VRegType`
#### `pto.vrec(vec: VRegType, mask: MaskType) -> VRegType`
#### `pto.vrsqrt(vec: VRegType, mask: MaskType) -> VRegType`
#### `pto.vrelu(vec: VRegType, mask: MaskType) -> VRegType`
#### `pto.vnot(vec: VRegType, mask: MaskType) -> VRegType`

**Description**: Element-wise unary operation under mask. `vrec` = reciprocal, `vrsqrt` = inverse square root, `vrelu` = `max(0, x)`.

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `vec` | `VRegType` | Input vector |
| `mask` | `MaskType` | Predicate mask (granularity must match element type) |

**Returns**:

| Return Value | Type | Description |
|--------------|------|-------------|
| `result` | `VRegType` | Result vector |

**Example**:

<!-- ptodsl-doc-test: {"mode":"compile_fragment","fixture":"compute_ops.vector_compute","symbol":"compute_ops_vector_probe","compile":{"BLOCK":128}} -->
```python
exp_vec = pto.vexp(s_row, col_mask)
```

---

### 8.2.2 Binary vector ops

#### `pto.vadd(v0: VRegType, v1: VRegType, mask: MaskType) -> VRegType`
#### `pto.vsub(v0: VRegType, v1: VRegType, mask: MaskType) -> VRegType`
#### `pto.vmul(v0: VRegType, v1: VRegType, mask: MaskType) -> VRegType`
#### `pto.vdiv(v0: VRegType, v1: VRegType, mask: MaskType) -> VRegType`
#### `pto.vmax(v0: VRegType, v1: VRegType, mask: MaskType) -> VRegType`
#### `pto.vmin(v0: VRegType, v1: VRegType, mask: MaskType) -> VRegType`

**Description**: Element-wise binary operation: `result[i] = v0[i] <op> v1[i]` for lanes where `mask[i]` is true.

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `v0` | `VRegType` | First operand vector |
| `v1` | `VRegType` | Second operand vector |
| `mask` | `MaskType` | Predicate mask |

**Returns**:

| Return Value | Type | Description |
|--------------|------|-------------|
| `result` | `VRegType` | Result vector |

---

**Bitwise binary ops** (integer types only):

| Op | Semantics |
|----|-----------|
| `pto.vand(v0, v1, mask) -> VRegType` | `v0 & v1` |
| `pto.vor(v0, v1, mask) -> VRegType` | `v0 \| v1` |
| `pto.vxor(v0, v1, mask) -> VRegType` | `v0 ^ v1` |
| `pto.vshl(vec, shift, mask) -> VRegType` | `vec << shift` (per-element) |
| `pto.vshr(vec, shift, mask) -> VRegType` | `vec >> shift` (per-element) |

---

### 8.2.3 Vector-scalar ops

#### `pto.vadds(vec: VRegType, scalar: ScalarType, mask: MaskType) -> VRegType`
#### `pto.vsubs(vec: VRegType, scalar: ScalarType, mask: MaskType) -> VRegType`
#### `pto.vmuls(vec: VRegType, scalar: ScalarType, mask: MaskType) -> VRegType`
#### `pto.vmaxs(vec: VRegType, scalar: ScalarType, mask: MaskType) -> VRegType`
#### `pto.vmins(vec: VRegType, scalar: ScalarType, mask: MaskType) -> VRegType`

**Description**: Element-wise `result[i] = vec[i] <op> scalar`. The scalar is broadcast to all active lanes.

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `vec` | `VRegType` | Input vector |
| `scalar` | `ScalarType` | Scalar operand (uniform across all lanes) |
| `mask` | `MaskType` | Predicate mask |

**Returns**:

| Return Value | Type | Description |
|--------------|------|-------------|
| `result` | `VRegType` | Result vector |

**Example** — subtract row max from score row (online softmax):

<!-- ptodsl-doc-test: {"mode":"compile_fragment","fixture":"compute_ops.vector_compute","symbol":"compute_ops_vector_probe","compile":{"BLOCK":128}} -->
```python
s_shifted = pto.vsubs(s_row, m_next, col_mask)
```

---

#### `pto.vlrelu(vec: VRegType, alpha: ScalarType, mask: MaskType) -> VRegType`

**Description**: Leaky ReLU — `vec[i] >= 0 ? vec[i] : alpha * vec[i]`.

---

### 8.2.4 Full-vector and group reductions

#### Full-vector reductions

#### `pto.vcadd(vec: VRegType, mask: MaskType) -> VRegType`

**Description**: Full-vector sum reduction. Result placed in lane 0.

#### `pto.vcmax(vec: VRegType, mask: MaskType) -> VRegType`

**Description**: Full-vector max with argmax. Result lane 0 = max value, lane 1 = max index.

#### `pto.vcmin(vec: VRegType, mask: MaskType) -> VRegType`

**Description**: Full-vector min with argmin. Result lane 0 = min value, lane 1 = min index.

---

#### Group reductions (per-VLane)

These reduce within each hardware vector lane group (typically 8 groups per vector). Useful when a vector register holds multiple independent sub-vectors that need separate reductions.

#### `pto.vcgadd(vec: VRegType, mask: MaskType) -> ScalarType`
#### `pto.vcgmax(vec: VRegType, mask: MaskType) -> ScalarType`
#### `pto.vcgmin(vec: VRegType, mask: MaskType) -> ScalarType`

**Description**: Per-group sum, max, or min. The underlying vector reduction places each group's result in the first lane of that group; the ptodsl surface extracts lane 0 and returns it as a runtime scalar.

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `vec` | `VRegType` | Input vector |
| `mask` | `MaskType` | Predicate mask |

**Returns**:

| Return Value | Type | Description |
|--------------|------|-------------|
| `result` | `ScalarType` | Lane-0 scalar extracted from the grouped reduction result |

**Example** — row max and row sum from online softmax:

<!-- ptodsl-doc-test: {"mode":"compile_fragment","fixture":"compute_ops.vector_compute","symbol":"compute_ops_vector_probe","compile":{"BLOCK":128}} -->
```python
row_max = pto.vcgmax(s_row, col_mask)   # grouped reduction, surfaced as a runtime scalar
row_sum = pto.vcgadd(p_row, col_mask)   # grouped reduction, surfaced as a runtime scalar
```

---

#### `pto.vcpadd(vec: VRegType, mask: MaskType) -> VRegType`

**Description**: Inclusive prefix sum (scan). `result[i] = sum_{k=0}^{i} vec[k]` for active lanes. f16 and f32 only.

---

### 8.2.5 Fused and compound ops

These combine an arithmetic operation with a math function or activation in a single instruction.

#### `pto.vexpdif(vec: VRegType, max_vec: VRegType, mask: MaskType, *, part: PartMode = PartMode.ODD) -> VRegType`

**Description**: `exp(vec[i] - max_vec[i])` — the stable softmax numerator. `part` controls which half of the vector is computed: `EVEN` or `ODD`. The result keeps the same `VRegType` as the input vector.

---

#### `pto.vaxpy(alpha: ScalarType, x: VRegType, y: VRegType, mask: MaskType) -> VRegType`

**Description**: Fused multiply-add: `alpha * x[i] + y[i]`.

---

#### `pto.vaddrelu(v0: VRegType, v1: VRegType, mask: MaskType) -> VRegType`

**Description**: `max(0, v0[i] + v1[i])` — fused add + ReLU.

#### `pto.vsubrelu(v0: VRegType, v1: VRegType, mask: MaskType) -> VRegType`

**Description**: `max(0, v0[i] - v1[i])` — fused sub + ReLU.

---

### 8.2.6 Comparison and selection

#### `pto.vcmp(v0: VRegType, v1: VRegType, seed_mask: MaskType, cmp_mode: CmpMode) -> MaskType`

**Description**: Element-wise comparison producing a predicate mask. `seed_mask` selects which lanes participate; the result inherits its granularity (e.g., `mask_b32` for f32).

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `v0` | `VRegType` | First operand |
| `v1` | `VRegType` | Second operand |
| `seed_mask` | `MaskType` | Seed mask gating participation |
| `cmp_mode` | `CmpMode` | `EQ`, `NE`, `LT`, `LE`, `GT`, `GE` |

**Returns**:

| Return Value | Type | Description |
|--------------|------|-------------|
| `pred` | `MaskType` | Result predicate mask |

---

#### `pto.vcmps(vec: VRegType, scalar: ScalarType, seed_mask: MaskType, cmp_mode: CmpMode) -> MaskType`

**Description**: Vector-scalar comparison. Same semantics as `vcmp` with a uniform scalar second operand.

---

#### `pto.vsel(true_v: VRegType, false_v: VRegType, mask: MaskType) -> VRegType`

**Description**: Per-lane select: `mask[i] ? true_v[i] : false_v[i]`.

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `true_v` | `VRegType` | Values when mask is true |
| `false_v` | `VRegType` | Values when mask is false |
| `mask` | `MaskType` | Selection predicate |

**Returns**:

| Return Value | Type | Description |
|--------------|------|-------------|
| `result` | `VRegType` | Selected vector |

---

### 8.2.7 Vector compute quick reference

| Category | Operations |
|----------|------------|
| Unary | `vexp`, `vln`, `vsqrt`, `vabs`, `vneg`, `vrec`, `vrsqrt`, `vrelu`, `vnot` |
| Binary | `vadd`, `vsub`, `vmul`, `vdiv`, `vmax`, `vmin`, `vand`, `vor`, `vxor`, `vshl`, `vshr` |
| Vector-scalar | `vadds`, `vsubs`, `vmuls`, `vmaxs`, `vmins`, `vlrelu` |
| Broadcast | `vbr`, `vdup` |
| Full reduction | `vcadd`, `vcmax`, `vcmin` |
| Group reduction | `vcgadd`, `vcgmax`, `vcgmin` |
| Scan | `vcpadd` |
| Fused | `vexpdif`, `vaxpy`, `vaddrelu`, `vsubrelu` |
| Compare/select | `vcmp`, `vcmps`, `vsel` |
| Conversion | `vbitcast`, `pbitcast` |

---

## 8.3 Cube compute (L3 — `@pto.cube`)

The Cube unit performs matrix multiplication. Its operands are typed pointers into cube-local buffers — L0A (left operand), L0B (right operand), L0C (accumulator), and BIAS. Cube data movement (`mte_l1_l0a`, `mte_l1_l0b`, `mte_l0c_ub`, etc.) was covered in Section 7.5; this section covers the compute instruction itself.

### 8.3.1 Matrix multiply: `pto.mad`

#### `pto.mad(lhs: PtrType, rhs: PtrType, dst: PtrType, m: int, n: int, k: int, *, unit_flag: pto.MadUnitFlagMode | None = None, disable_gemv: bool = False, sat: pto.SatMode | None = None, tf32_mode: pto.Tf32Mode | None = None, n_dir: bool = False) -> None`

**Description**: Zero-initialized matrix multiply: `dst[M×N] = lhs[M×K] * rhs[K×N]`. `lhs` is an L0A pointer, `rhs` is an L0B pointer, `dst` is an L0C pointer.

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `lhs` | `PtrType` (L0A) | Left operand matrix (M × K) |
| `rhs` | `PtrType` (L0B) | Right operand matrix (K × N) |
| `dst` | `PtrType` (L0C) | Destination accumulator (M × N) |
| `m` | `int` | M dimension size |
| `k` | `int` | K dimension (inner/reduction dimension) |
| `n` | `int` | N dimension size |
| `unit_flag` | `pto.MadUnitFlagMode` or `None` | Optional producer unit-flag clause: `CHECK_ONLY` or `CHECK_AND_SET` |
| `disable_gemv` | `bool` | Force normal matmul operand layout instead of GEMV specialization |
| `sat` | `pto.SatMode` or `None` | Optional saturation clause: `ON` or `OFF` |
| `tf32_mode` | `pto.Tf32Mode` or `None` | Optional TF32 rounding mode for f32/f32/f32 `mad*`: `ROUND_EVEN` or `ROUND_AWAY` |
| `n_dir` | `bool` | Request N-direction production order for compatible schedules |

**Returns**: None (writes to `dst` in L0C).

---

#### `pto.mad_acc(lhs: PtrType, rhs: PtrType, dst: PtrType, m: int, n: int, k: int, *, unit_flag: pto.MadUnitFlagMode | None = None, disable_gemv: bool = False, sat: pto.SatMode | None = None, tf32_mode: pto.Tf32Mode | None = None, n_dir: bool = False) -> None`

**Description**: Accumulating matrix multiply: `dst[M×N] += lhs[M×K] * rhs[K×N]`. `dst` must already hold a prior accumulation result.

---

#### `pto.mad_bias(lhs: PtrType, rhs: PtrType, dst: PtrType, bias: PtrType, m: int, n: int, k: int, *, unit_flag: pto.MadUnitFlagMode | None = None, disable_gemv: bool = False, sat: pto.SatMode | None = None, tf32_mode: pto.Tf32Mode | None = None, n_dir: bool = False) -> None`

**Description**: Bias-initialized matrix multiply: `dst[M×N] = lhs[M×K] * rhs[K×N] + bias[M×N]`. `bias` is a BIAS pointer.

---

#### `pto.mad_mx(lhs: PtrType, rhs: PtrType, dst: PtrType, m: int, n: int, k: int, *, unit_flag: pto.MadUnitFlagMode | None = None, disable_gemv: bool = False, sat: pto.SatMode | None = None, n_dir: bool = False) -> None`

**Description**: MX-format zero-initialized matrix multiply. This variant is intended for MX-enabled operand formats such as f8 payloads with their associated scale data already staged into cube-local buffers.

---

#### `pto.mad_mx_acc(lhs: PtrType, rhs: PtrType, dst: PtrType, m: int, n: int, k: int, *, unit_flag: pto.MadUnitFlagMode | None = None, disable_gemv: bool = False, sat: pto.SatMode | None = None, n_dir: bool = False) -> None`

**Description**: MX-format accumulating matrix multiply: `dst[M×N] += lhs[M×K] * rhs[K×N]`.

---

#### `pto.mad_mx_bias(lhs: PtrType, rhs: PtrType, dst: PtrType, bias: PtrType, m: int, n: int, k: int, *, unit_flag: pto.MadUnitFlagMode | None = None, disable_gemv: bool = False, sat: pto.SatMode | None = None, n_dir: bool = False) -> None`

**Description**: MX-format bias-initialized matrix multiply: `dst[M×N] = lhs[M×K] * rhs[K×N] + bias[M×N]`.

MX variants intentionally do not expose `tf32_mode`; that clause is only valid for f32/f32/f32 non-MX `mad`, `mad_acc`, and `mad_bias`.

---

### 8.3.2 MAD common clauses

All `mad*` APIs accept TileLang-compatible keyword clauses. The wrapper lowers these keywords to the VPTO custom assembly clauses shown below.

| Keyword | Values | Lowered clause |
|---------|--------|----------------|
| `unit_flag` | `pto.MadUnitFlagMode.CHECK_ONLY`, `CHECK_AND_SET`, or `None` | `unit_flag(check_only)` / `unit_flag(check_and_set)` |
| `disable_gemv` | `True` / `False` | `disable_gemv` when true |
| `sat` | `pto.SatMode.ON`, `OFF`, or `None` | `sat` / `nosat` |
| `tf32_mode` | `pto.Tf32Mode.ROUND_EVEN`, `ROUND_AWAY`, or `None` | `tf32_mode(round_even)` / `tf32_mode(round_away)`; f32/f32/f32 non-MX only |
| `n_dir` | `True` / `False` | `n_dir` when true |

Example:

```python
pto.mad(
    lhs_l0a.as_ptr(),
    rhs_l0b.as_ptr(),
    acc_l0c.as_ptr(),
    m,
    n,
    k,
    unit_flag=pto.MadUnitFlagMode.CHECK_ONLY,
    disable_gemv=True,
    sat=pto.SatMode.OFF,
    tf32_mode=pto.Tf32Mode.ROUND_EVEN,
    n_dir=True,
)
```

### 8.3.3 Typical cube matmul pattern

A full cube matmul follows a three-stage pattern: stage operands into L0A/L0B, compute, write back to UB.

<!-- ptodsl-doc-test: {"mode":"compile_fragment","fixture":"data_movement.cube_helper","symbol":"data_movement_cube_helper_probe","compile":{"BLOCK_M":16,"BLOCK_K":16,"BLOCK_N":16}} -->
```python
@pto.cube
def qk_matmul(q_tile, k_tile, q_l0a, k_l0b, s_acc, s_tile):
    m = q_tile.valid_shape[0]
    k = q_tile.valid_shape[1]
    n = k_tile.valid_shape[1]

    # Stage: source tiles → L0A / L0B
    pto.mte_l1_l0a(q_tile.as_ptr(), q_l0a.as_ptr(), m, k)
    pto.mte_l1_l0b(k_tile.as_ptr(), k_l0b.as_ptr(), k, n, transpose=True)

    # Compute: L0A × L0B → L0C
    pto.mad(q_l0a.as_ptr(), k_l0b.as_ptr(), s_acc.as_ptr(), m, n, k)

    # Writeback: L0C → UB
    pto.mte_l0c_ub(s_acc.as_ptr(), s_tile.as_ptr(), m, n, n, n, 0)
```

The `mte_l1_l0a`/`mte_l1_l0b` stage operands from the authored source tiles into cube-local buffers. `mad` performs the matrix multiply into L0C. `mte_l0c_ub` writes the result back to a UB tile for downstream processing. At this micro-op layer, the operands are explicit pointer views obtained with `.as_ptr()`.

---

### 8.3.4 Cube compute quick reference

| Operation | Semantics |
|-----------|-----------|
| `pto.mad(lhs, rhs, dst, m, n, k, **clauses)` | `dst = lhs * rhs` (zero-init) |
| `pto.mad_acc(lhs, rhs, dst, m, n, k, **clauses)` | `dst += lhs * rhs` (accumulating) |
| `pto.mad_bias(lhs, rhs, dst, bias, m, n, k, **clauses)` | `dst = lhs * rhs + bias` |
| `pto.mad_mx(lhs, rhs, dst, m, n, k, **clauses)` | MX-format zero-init matmul |
| `pto.mad_mx_acc(lhs, rhs, dst, m, n, k, **clauses)` | MX-format accumulating matmul |
| `pto.mad_mx_bias(lhs, rhs, dst, bias, m, n, k, **clauses)` | MX-format bias-init matmul |

MX variants require MX-enabled dtypes (f8) and pre-loaded scale payloads. For most users, the standard `mad`, `mad_acc`, and `mad_bias` are the primary interface.
