//! A compact prefix of a rank's committed GLM state; live sequence addresses and kernel layouts stay fixed.
const std = @import("std");
const State = @import("cuda_state.zig").State;
const page_rows = @import("cuda_state.zig").page_rows;
const Config = @import("config.zig").Config;
const cache = @import("cuda_dsa_block.zig");
const sparse = @import("cuda_sparse.zig");

/// `len` bytes from `live`, or on a paged sequence's pages `span` bytes a page `page` bytes apart, from byte `from`;
/// `lat`: a sharded sequence's latent rows (TF_GLM_KV_SHARD), whose pages are its latent table's (`Plan.lids`).
pub const Part = struct { live: u64, len: usize, span: usize = 0, page: u64 = 0, lat: bool = false, from: usize = 0 };
pub const Plan = struct {
    pub const max_parts = 5 * (@import("config.zig").max_layers + 1) + 3;
    parts: [max_parts]Part = undefined, // and one outside the arena (`outside`)
    count: usize = 0,
    len: usize = 0,
    ids: []const u32 = &.{}, // a paged sequence's pages in row order
    lids: []const u32 = &.{}, // sharded: its latent pages in row order (this rank's, cuda_kv_shard.latentIds)
    sizing: bool = false, // `size`: the rows counted, not read, so no page need be mapped
    latent: bool = true, // the latent codes and scales are parts (sharded: only where this rank keeps them)
    shared: usize = 0, // TF_KV_SHARE: a paged sequence's first pages the shared store holds: their rows stay on them

    /// The state at `at` (a kept copy's plan; `initForShared` past a shared store's pages).
    pub fn init(st: *const State, c: *const Config, at: usize, head: ?usize) !Plan {
        return make(st, c, at, head, 0, false, true);
    }

    /// `init` past the shared store's `shared` pages (TF_KV_SHARE, a state's own copy).
    pub fn initShared(st: *const State, c: *const Config, at: usize, head: ?usize, shared: usize) !Plan {
        return make(st, c, at, head, shared, false, true);
    }

    /// `init` with or without the DSA caches' latent codes and scales (`latent`: a sharded state off its home rank
    /// holds every other part; cuda_snapshots).
    pub fn initFor(st: *const State, c: *const Config, at: usize, head: ?usize, latent: bool) !Plan {
        return initForShared(st, c, at, head, 0, latent);
    }

    /// `initFor` past the shared store's `shared` pages (TF_KV_SHARE; a resume's state leaves their rows out).
    pub fn initForShared(st: *const State, c: *const Config, at: usize, head: ?usize, shared: usize, latent: bool) !Plan {
        return make(st, c, at, head, shared, false, latent);
    }

    /// A prefix's bytes at `at`, rows to `capacity` counted held or not (a paged sequence holds its request's alone).
    pub fn size(st: *const State, c: *const Config, capacity: usize, at: usize, head: ?usize) !usize {
        return sizeFor(st, c, capacity, at, head, true);
    }

    /// `size` with or without the latent codes and scales (`initFor`'s `latent`).
    pub fn sizeFor(st: *const State, c: *const Config, capacity: usize, at: usize, head: ?usize, latent: bool) !usize {
        return sizeForShared(st, c, capacity, at, head, 0, latent);
    }

    /// `sizeFor` past the shared store's `shared` pages (TF_KV_SHARE).
    pub fn sizeForShared(st: *const State, c: *const Config, capacity: usize, at: usize, head: ?usize, shared: usize, latent: bool) !usize {
        var whole = st.*;
        whole.capacity = capacity;
        return (try make(&whole, c, at, head, shared, true, latent)).len;
    }

    fn make(st: *const State, c: *const Config, at: usize, head: ?usize, shared: usize, sizing: bool, latent: bool) !Plan {
        if (at > st.capacity or st.caches.len > @import("config.zig").max_layers) return error.SnapshotShape;
        if (st.rings.len != 2 * (st.caches.len + 1)) return error.SnapshotShape;
        if (shared > 0 and (st.table == 0 or shared * page_rows > at)) return error.SnapshotShape; // pages below `at` only
        var p: Plan = .{ .ids = st.pages, .lids = st.lpages, .sizing = sizing, .latent = latent, .shared = shared };
        // a paged sequence's caches are its pages: its fields up to them end at its first ring
        const first = if (st.table != 0) st.rings[0] else if (st.caches.len > 0) st.caches[0].lc else st.mtp_caches.lc;
        if (first < st.arena.buf.ptr) return error.SnapshotShape;
        try p.add(st, st.arena.buf.ptr, @intCast(first - st.arena.buf.ptr));
        for (st.caches, 0..) |x, j| try p.rows(st, c, x, j, at);
        if (head) |n| {
            if (n > st.capacity) return error.SnapshotShape;
            try p.rows(st, c, st.mtp_caches, st.caches.len, n);
        }
        if (st.mtp_pos_dev < st.arena.buf.ptr or st.mtp_pos_dev - st.arena.buf.ptr > st.arena.buf.len)
            return error.SnapshotShape;
        const end = if (st.table != 0) st.table else st.arena.buf.ptr + st.arena.buf.len; // its table stays its own
        if (end < st.mtp_pos_dev) return error.SnapshotShape;
        try p.add(st, st.mtp_pos_dev, @intCast(end - st.mtp_pos_dev));
        return p;
    }

    fn rows(p: *Plan, st: *const State, c: *const Config, x: cache.Caches, j: usize, at: usize) !void {
        const latent = try std.math.mul(usize, at, @as(usize, c.kv_lora) * @as(usize, if (x.ls == 0) 2 else 1));
        const ring = try std.math.mul(usize, @min(at, st.ring_rows), st.ring_row);
        const pooled = try std.math.mul(usize, @min(at / sparse.pool + 2, x.pk_rows), @as(usize, c.i_dim) * 2);
        const latent_row = @as(usize, c.kv_lora) * @as(usize, if (x.ls == 0) 2 else 1);
        if (p.latent) {
            try p.addLatent(st, x, x.lc, latent, page_rows * latent_row);
            // E4M3 scales: beside the codes in the arena, or on pages a part's 512 after its codes
            if (x.ls != 0) try p.addLatent(st, x, x.ls, try std.math.mul(usize, at, @sizeOf(f32)), page_rows * @sizeOf(f32));
        }
        try p.add(st, st.rings[2 * j], ring);
        try p.add(st, st.rings[2 * j + 1], ring);
        try p.addCache(st, x, x.pk, pooled, page_rows / sparse.pool * @as(usize, c.i_dim) * 2);
    }

    /// A cache's rows: in the arena, or on the sequence's pages, `span` bytes a page (past the store's `shared`).
    fn addCache(p: *Plan, st: *const State, x: cache.Caches, live: u64, len: usize, span: usize) !void {
        if (x.pt == 0) return p.add(st, live, len);
        const from = @min(len, p.shared * span);
        if (len == from) return;
        if ((!p.sizing and (len - 1) / span >= p.ids.len) or p.count == p.parts.len) return error.SnapshotShape;
        p.len = try std.math.add(usize, p.len, len - from);
        p.parts[p.count] = .{ .live = live, .len = len - from, .span = span, .page = x.ps, .from = from };
        p.count += 1;
    }

    /// A cache's latent codes or scales: `addCache`, and sharded (a latent table of its own) on its latent pages.
    fn addLatent(p: *Plan, st: *const State, x: cache.Caches, live: u64, len: usize, span: usize) !void {
        if (x.lpt == 0) return p.addCache(st, x, live, len, span);
        if (len == 0) return;
        if ((!p.sizing and (len - 1) / span >= p.lids.len) or p.count == p.parts.len) return error.SnapshotShape;
        p.len = try std.math.add(usize, p.len, len);
        p.parts[p.count] = .{ .live = live, .len = len, .span = span, .page = x.lps, .lat = true };
        p.count += 1;
    }

    /// `len` bytes the sequence's arena does not hold (DFlash2's context ring for the slot), after every arena part.
    pub fn outside(p: *Plan, live: u64, len: usize) !void {
        if (len == 0) return;
        if (live == 0 or p.count == p.parts.len) return error.SnapshotShape;
        p.len = try std.math.add(usize, p.len, len);
        p.parts[p.count] = .{ .live = live, .len = len };
        p.count += 1;
    }

    /// Where a part's bytes from `done` on sit, as many as lie in a row there.
    pub fn piece(p: *const Plan, part: Part, done: usize) Part {
        if (part.span == 0) return .{ .live = part.live + done, .len = part.len - done };
        const at = part.from + done;
        const off = at % part.span;
        const ids = if (part.lat) p.lids else p.ids;
        return .{ .live = part.live + @as(u64, ids[at / part.span]) * part.page + off, .len = @min(part.span - off, part.len - done) };
    }

    fn add(p: *Plan, st: *const State, live: u64, len: usize) !void {
        if (len == 0) return;
        if (live < st.arena.buf.ptr) return error.SnapshotShape;
        const off = live - st.arena.buf.ptr;
        if (off > st.arena.buf.len or len > st.arena.buf.len - off or p.count == p.parts.len)
            return error.SnapshotShape;
        p.len = try std.math.add(usize, p.len, len);
        p.parts[p.count] = .{ .live = live, .len = len };
        p.count += 1;
    }
};
