# Issue 483 RMSNorm PTODSL kernel draft.
#
# This draft follows mission/483_docs.md:
# - pto.alloc_buffer(...) returns a typed pointer.
#   - UB scratch and lane-local storage both use alloc_buffer.
# - scalar.load/store support contiguous vector access.
# - pto.vec(dtype, lanes)(scalar) type def & broadcasts a scalar to a vector.
# - scalar.reduce(vector, op="add") reduces inside one workitem.(未找到合适接口，再看看或新增)
# - pto.simt_allreduce_sum(...) reduces across SIMT workitems.
# - SIMT launch is written as the requested context form: with pto.simt(x, y, z).
#
# This file is a frontend contract/example. Some APIs are proposed and are not
# expected to run until the corresponding PTODSL implementation lands.

from ptodsl import pto, scalar


# =============================================================================
# Reusable body logic: load W once from UB into lane-local persistent storage.
# =============================================================================
def init_weight_fragment_body(
    w_ub,
    w_frag,
    *,
    threads: pto.const_expr,
    rounds: pto.const_expr,
    lanes: pto.const_expr = 4,
):
    tx = pto.get_tid_x()

    for r in pto.static_range(0, rounds):
        ub_offset = r * threads * lanes + tx * lanes
        frag_offset = r * lanes

        w4 = scalar.load(w_ub, ub_offset, contiguous=lanes)
        scalar.store(w4, w_frag, frag_offset)


# =============================================================================
# Reusable body logic: one-token RMSNorm inside a SIMT context.
# =============================================================================
def rmsnorm_4096_token_body(
    x_ub,
    y_ub,
    rstd_ub,
    reduce_scratch,
    x_frag,
    w_frag,
    eps: pto.f32,
    ping: pto.i32,
    *,
    threads: pto.const_expr,
    rounds: pto.const_expr,
    lanes: pto.const_expr = 4,
    hidden_size: pto.const_expr = 4096,
):
    tx = pto.get_tid_x()
    local_sum = 0.0

    # 1. Each workitem reads small contiguous chunks and accumulates sum(x^2).
    for r in pto.static_range(0, rounds):
        lane_offset = r * threads * lanes + tx * lanes
        x_offset = ping * hidden_size + lane_offset
        frag_offset = r * lanes

        x4 = scalar.load(x_ub, x_offset, contiguous=lanes)
        scalar.store(x4, x_frag, frag_offset)

        sq4 = x4 * x4
        local_sum = local_sum + scalar.reduce(sq4, op="add")

    # 2. Reduce the per-workitem local sums across the SIMT launch.
    sum_sq = pto.simt_allreduce_sum(
        local_sum,
        threads=threads,
        scratch=reduce_scratch,
    )

    # 3. RMSNorm scale for this token: rstd = rsqrt(sum(x^2) / hidden + eps).
    rstd = 1.0 / scalar.sqrt(sum_sq / hidden_size + eps)

    # Save one scalar rstd per token. Every workitem sees the same rstd, so only
    # workitem 0 writes it.
    if tx == 0:
        scalar.store(rstd, rstd_ub, ping)

    # 4. y = x * rstd * w. Broadcast scalar rstd to vector<lanesxf32> first.
    rstd_vec = pto.vec(pto.f32, lanes)(rstd)

    for r in pto.static_range(0, rounds):
        lane_offset = r * threads * lanes + tx * lanes
        y_offset = ping * hidden_size + lane_offset
        frag_offset = r * lanes

        x4 = scalar.load(x_frag, frag_offset, contiguous=lanes)
        w4 = scalar.load(w_frag, frag_offset, contiguous=lanes)
        y4 = x4 * rstd_vec * w4
        scalar.store(y4, y_ub, y_offset)


# =============================================================================
# Kernel template: same body, different SIMT launch sizes.
# =============================================================================
@pto.jit(target="a5", mode="explicit")
def rmsnorm_4096_alloc_buffer_simt_context_kernel(
    X: pto.ptr(pto.f32, "gm"),
    W: pto.ptr(pto.f32, "gm"),
    Y: pto.ptr(pto.f32, "gm"),
    RSTD: pto.ptr(pto.f32, "gm"),
    eps: pto.f32,
    batch: pto.i32,
    *,
    threads: pto.const_expr,
    rounds: pto.const_expr,
    lanes: pto.const_expr = 4,
    hidden_size: pto.const_expr = 4096,
    n_cores: pto.const_expr = 64,
    tokens_per_core: pto.const_expr = 64,
):
    core_id = pto.get_block_idx()
    frag_elems: pto.const_expr = rounds * lanes

    # UB scratch storage. alloc_buffer returns typed pointers.
    w_ub = pto.alloc_buffer((hidden_size,), pto.f32, scope="ub")
    x_ub = pto.alloc_buffer((2, hidden_size), pto.f32, scope="ub")
    y_ub = pto.alloc_buffer((2, hidden_size), pto.f32, scope="ub")
    rstd_ub = pto.alloc_buffer((2,), pto.f32, scope="ub")
    reduce_scratch = pto.alloc_buffer((threads,), pto.f32, scope="ub")

    # Lane-local storage. Each SIMT workitem gets its own x_frag and w_frag.
    x_frag = pto.alloc_buffer((frag_elems,), pto.f32, scope="local")
    w_frag = pto.alloc_buffer((frag_elems,), pto.f32, scope="local", persistent=True)

    # Move W once from GM to UB, then cache it in persistent lane-local storage.
    # Contract note: MTE helpers need to accept alloc_buffer returned pointers.
    pto.mte_gm_ub(
        W,
        w_ub,
        0,
        hidden_size * 4,
        nburst=(1, hidden_size * 4, hidden_size * 4),
    )

    with pto.simt(threads, 1, 1):
        init_weight_fragment_body(
            w_ub,
            w_frag,
            threads=threads,
            rounds=rounds,
            lanes=lanes,
        )

    for local_token in pto.static_range(0, tokens_per_core):
        token_id = local_token * n_cores + core_id
        ping = local_token & 1

        # Move one token row X[token, :] from GM to the ping/pong UB region.
        pto.mte_gm_ub(
            X + token_id * hidden_size,
            x_ub,
            ping * hidden_size * 4,
            hidden_size * 4,
            nburst=(1, hidden_size * 4, hidden_size * 4),
        )

        with pto.simt(threads, 1, 1):
            rmsnorm_4096_token_body(
                x_ub,
                y_ub,
                rstd_ub,
                reduce_scratch,
                x_frag,
                w_frag,
                eps,
                ping,
                threads=threads,
                rounds=rounds,
                lanes=lanes,
                hidden_size=hidden_size,
            )

        # Move Y[token, :] and RSTD[token] back to GM.
        pto.mte_ub_gm(
            y_ub,
            Y + token_id * hidden_size,
            hidden_size * 4,
            nburst=(1, hidden_size * 4, hidden_size * 4),
        )

        pto.mte_ub_gm(
            rstd_ub,
            RSTD + token_id,
            4,
            nburst=(1, 4, 4),
        )


# =============================================================================
# Concrete launch-specialized wrappers.
# =============================================================================
def rmsnorm_4096_x128(
    X: pto.ptr(pto.f32, "gm"),
    W: pto.ptr(pto.f32, "gm"),
    Y: pto.ptr(pto.f32, "gm"),
    RSTD: pto.ptr(pto.f32, "gm"),
    eps: pto.f32,
    batch: pto.i32,
):
    return rmsnorm_4096_alloc_buffer_simt_context_kernel(
        X,
        W,
        Y,
        RSTD,
        eps,
        batch,
        threads=128,
        rounds=8,
        lanes=4,
        tokens_per_core=64,
    )


def rmsnorm_4096_x64(
    X: pto.ptr(pto.f32, "gm"),
    W: pto.ptr(pto.f32, "gm"),
    Y: pto.ptr(pto.f32, "gm"),
    RSTD: pto.ptr(pto.f32, "gm"),
    eps: pto.f32,
    batch: pto.i32,
):
    return rmsnorm_4096_alloc_buffer_simt_context_kernel(
        X,
        W,
        Y,
        RSTD,
        eps,
        batch,
        threads=64,
        rounds=16,
        lanes=4,
        tokens_per_core=64,
    )
