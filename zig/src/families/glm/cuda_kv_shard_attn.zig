//! TF_GLM_KV_SHARD's owner-computed attention (GLM patch 0118 by Robert Wojciechowski, GitHub @wojo, X @rwojo):
//! per DSA layer, a segment whose latent rows only one rank holds (its owner, with no twin on the other) has that rank
//! attend for both ranks' heads. After the replicated projections, the other rank sends its absorbed queries for the
//! segment's rows; the holder attends its own heads and then the peer's (the same kernels, token lists and call shapes
//! the peer would have run) and sends those outputs back into the peer's rows. The forward is eager (no graphs), so
//! every exchange is sized to the round's segments on the host: grouped NCCL send/recv of whole row ranges, one group
//! a phase, on the main stream. A segment both ranks hold (a prompt pass's twin) attends locally, as unsharded.
const std = @import("std");
const cuda = @import("cuda");
const kvs = @import("cuda_kv_shard.zig");
const fw = @import("cuda_forward.zig");
const dsa_block = @import("cuda_dsa_block.zig");
const latent = @import("cuda_latent.zig");
const weights = @import("cuda_weights.zig");
const prof = @import("cuda_prof.zig");

/// A segment's part on this rank: both ranks hold its rows (`local`), this rank lacks them (`give`: its queries out,
/// its outputs in), or this rank alone holds them (`take`: the peer's queries in, its outputs out).
pub const Role = enum { local, give, take };

pub fn role(hold: ?kvs.Hold, rank: usize) Role {
    const h = hold orelse return .local;
    if (h.twin) return .local;
    return if (h.owner == rank) .take else .give;
}

/// One segment of a layer: rows [row0, row0 + rows) of the window, and its role here.
pub const Seg = struct { row0: usize, rows: usize, role: Role };

/// Which buffer an exchange moves: this rank's queries (`qa`) or outputs (`ol`) rows, the peer queries this rank
/// received (`pq`), or the peer outputs it computed (`po`).
pub const Buf = enum { qa, ol, pq, po };

/// The pass an `attend` makes over a segment: its own heads (queries `qa` into `ol`), the peer's (`pq` into `po`), or
/// none (the rows' writes and token lists alone).
pub const Pass = enum { own, peer, none };

/// Whether any segment exchanges rows (else the layer runs as unsharded, with no exchange posted).
pub fn exchanges(segs: []const Seg) bool {
    for (segs) |g| if (g.role != .local) return true;
    return false;
}

/// One DSA layer's attention over `segs` on this rank, through `d` (the device, or a host model in tests):
/// 1. queries: every give segment's `qa` rows to the peer, every take segment's into `pq`, in segment order;
/// 2. each segment in order: local, its own heads; take, its own heads then the peer's; give, its writes and lists;
/// 3. outputs: every take segment's `po` rows to the peer, every give segment's into `ol`, in segment order.
/// Both ranks see the same segments with opposite roles, so every send meets the peer's receive of the same rows.
pub fn run(comptime D: type, d: D, segs: []const Seg) !void {
    if (!exchanges(segs)) {
        for (segs) |g| try d.attend(g, .own);
        return;
    }
    try d.group(.begin);
    for (segs) |g| switch (g.role) {
        .give => try d.move(.qa, g, .send),
        .take => try d.move(.pq, g, .recv),
        .local => {},
    };
    try d.group(.end);
    for (segs) |g| switch (g.role) {
        .local => try d.attend(g, .own),
        .take => {
            try d.attend(g, .own);
            if (!mut(.peer_skipped)) try d.attend(g, .peer);
        },
        .give => try d.attend(g, .none),
    };
    try d.group(.begin);
    for (segs) |g| switch (g.role) {
        .take => try d.move(.po, g, .send),
        .give => try d.move(if (mut(.output_recv_into_qa)) .qa else .ol, g, .recv),
        .local => {},
    };
    try d.group(.end);
}

pub const Dir = enum { send, recv };
pub const Edge = enum { begin, end };

/// The engine's exchange room and this rank's place (null in a Forward unsharded).
pub const Ctx = struct {
    rank: usize,
    lib: *const cuda.nccl.Library,
    comm: cuda.nccl.Comm,
    pq: u64, // bf16 [rows][heads][lw]: the peer's queries for the rows this rank attends for it
    po: u64, // bf16 [rows][heads][lw]: this rank's outputs of the peer's heads
    rows: usize, // the rows pq and po hold
    counts: ?*kvs.Counts = null, // rank 0's counters (the other rank's own are not reported)

    pub fn peer(c: *const Ctx) usize {
        return 1 - c.rank;
    }
};

/// Bytes one row of a rank's absorbed queries or latent outputs takes: its heads x lw bf16.
pub fn rowBytes(l: dsa_block.Layer) usize {
    return l.heads * l.lw * 2;
}

/// The exchange room's bytes: two buffers of `rows` rows of `heads` x `lw` bf16.
pub fn roomBytes(rows: usize, heads: usize, lw: usize) usize {
    return 2 * rows * heads * lw * 2;
}

/// The device side of `run`: the CUDA forward's buffers and kernels for one DSA layer over one sequence or a shared
/// window's segments.
const Gpu = struct {
    f: *const fw.Forward,
    kv: *const Ctx,
    k: dsa_block.Kernels,
    a: dsa_block.Layer,
    b: *fw.Buffers,
    caches: []const dsa_block.Caches, // each segment's sequence's cache of this layer
    states: []const *fw.State,
    nch: usize, // single-sequence: the call's chunks (a shared window computes each segment's)
    single: bool,

    fn base(g: Gpu, which: Buf) u64 {
        return switch (which) {
            .qa => g.b.dsa.qa,
            .ol => g.b.dsa.ol,
            .pq => g.kv.pq,
            .po => g.kv.po,
        };
    }

    pub fn group(g: Gpu, e: Edge) !void {
        const lib = g.kv.lib;
        switch (e) {
            .begin => try lib.check(lib.api.ncclGroupStart(), "ncclGroupStart"),
            .end => try lib.check(lib.api.ncclGroupEnd(), "ncclGroupEnd"),
        }
    }

    pub fn move(g: Gpu, which: Buf, s: Seg, dir: Dir) !void {
        const rb = rowBytes(g.a);
        if (s.row0 + s.rows > g.kv.rows and (which == .pq or which == .po)) return error.KvShardRoom;
        const at = g.base(which) + s.row0 * rb;
        const lib = g.kv.lib;
        const peer: c_int = @intCast(g.kv.peer());
        switch (dir) {
            .send => try lib.check(lib.api.ncclSend(at, s.rows * rb, .u8, peer, g.kv.comm, g.f.stream.handle), "ncclSend"),
            .recv => try lib.check(lib.api.ncclRecv(at, s.rows * rb, .u8, peer, g.kv.comm, g.f.stream.handle), "ncclRecv"),
        }
        if (g.kv.counts) |c| if (dir == .send) {
            c.owner_bytes += s.rows * rb;
        };
    }

    fn index(g: Gpu, s: Seg) usize {
        if (g.single) return 0;
        for (g.f.ranks[0].segs, 0..) |x, i| if (x.row0 == s.row0) return i;
        unreachable;
    }

    pub fn attend(g: Gpu, s: Seg, pass: Pass) !void {
        const i = g.index(s);
        const st = g.states[i];
        const c = g.caches[i];
        const nch = if (g.single) g.nch else @min(latent.chunksFor(st.pos + s.rows), g.b.nch);
        switch (pass) {
            .own => try dsa_block.attend(g.k, g.f.stream, g.a, c, g.b.dsa, s.row0, s.rows, st.pos_dev, st.pos, nch, g.b.part_rows),
            .none => _ = try dsa_block.attendNoLatent(g.k, g.f.stream, g.a, c, g.b.dsa, s.row0, s.rows, st.pos_dev, st.pos),
            .peer => { // right after the own pass: its token lists are in the scratch's first rows
                const rb = rowBytes(g.a);
                try dsa_block.attendOnly(g.k, g.f.stream, g.a, c, g.b.dsa, s.rows, st.pos_dev, st.pos, nch, g.b.part_rows, g.kv.pq + s.row0 * rb, g.kv.po + s.row0 * rb, g.b.dsa.tokens, g.b.dsa.counts);
            },
        }
    }
};

/// Whether this forward's rank attends any segment owner-computed this layer (null Ctx: never).
pub fn wanted(f: *const fw.Forward, r: fw.Rank) bool {
    const kv = f.kv orelse return false;
    if (r.segs.len == 0) return role(r.st.hold, kv.rank) != .local;
    for (r.segs) |g| if (role(g.st.hold, kv.rank) != .local) return true;
    return false;
}

/// Forward.mixer's DSA block on one sequence whose latent rows one rank holds: the projections, the exchange, its
/// attention or the peer's, expand_v (o_proj stays the caller's fill).
pub fn single(f: *const fw.Forward, r: fw.Rank, a: dsa_block.Layer, di: usize, rows: usize, nch: usize, k: dsa_block.Kernels) !void {
    const kv = f.kv.?;
    const st = r.st;
    try dsa_block.prepareProfile(k.set, f.stream, a, r.b.dsa, r.b.normed, r.w.config.hidden, rows, dsa_block.sparseRows(a, st.pos, rows), k.prof);
    const caches = [_]dsa_block.Caches{st.caches[di]};
    const states = [_]*fw.State{st};
    const d: Gpu = .{ .f = f, .kv = kv, .k = k, .a = a, .b = r.b, .caches = &caches, .states = &states, .nch = nch, .single = true };
    const seg = [_]Seg{.{ .row0 = 0, .rows = rows, .role = role(st.hold, kv.rank) }};
    try run(Gpu, d, &seg);
    note(kv, &seg);
    const stamp = prof.span(k.prof, f.stream, .projections);
    defer stamp.end();
    try latent.expandV(k.set, f.stream, r.b.dsa.ol, a.wv, r.b.dsa.vn, rows, a.heads, a.dv, a.lw);
}

/// The shared window's DSA block (cuda_shared.mixer's) with owner-computed segments: every segment on its own
/// launches (the batched ones' bits), the exchanges around them, then expand_v for every row.
pub fn shared(f: *const fw.Forward, r: fw.Rank, a: dsa_block.Layer, di: usize, rows: usize) !void {
    const kv = f.kv.?;
    const b = r.b;
    var sparse_any = false;
    for (r.segs) |g| sparse_any = sparse_any or dsa_block.sparseRows(a, g.st.pos, g.rows);
    try dsa_block.prepare(f.set(), f.stream, a, b.dsa, b.normed, r.w.config.hidden, rows, sparse_any);
    var caches: [latent.max_segs]dsa_block.Caches = undefined;
    var states: [latent.max_segs]*fw.State = undefined;
    var segs: [latent.max_segs]Seg = undefined;
    if (r.segs.len > latent.max_segs) return error.TooManySegments;
    for (r.segs, 0..) |g, i| {
        caches[i] = g.st.caches[di];
        states[i] = g.st;
        segs[i] = .{ .row0 = g.row0, .rows = g.rows, .role = role(g.st.hold, kv.rank) };
    }
    const n = r.segs.len;
    const d: Gpu = .{ .f = f, .kv = kv, .k = .{ .set = f.set(), .ops = f.m.ops() }, .a = a, .b = b, .caches = caches[0..n], .states = states[0..n], .nch = 0, .single = false };
    try run(Gpu, d, segs[0..n]);
    note(kv, segs[0..n]);
    try latent.expandV(f.set(), f.stream, b.dsa.ol, a.wv, b.dsa.vn, rows, a.heads, a.dv, a.lw);
}

/// Rank 0's counters for a layer's exchange.
fn note(kv: *const Ctx, segs: []const Seg) void {
    const c = kv.counts orelse return;
    if (!exchanges(segs)) return;
    c.owner_layers += 1;
    for (segs) |g| c.owner_rows += if (g.role != .local) g.rows else 0;
}

// -- mutants (test builds only) -------------------------------------------------------------------------------------

pub const Mutant = enum { none, peer_skipped, output_recv_into_qa };
pub var mutant: Mutant = .none;

fn mut(comptime m: Mutant) bool {
    return @import("builtin").is_test and mutant == m;
}
