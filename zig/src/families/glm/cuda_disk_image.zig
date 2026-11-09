//! A GLM kept state's disk image on one rank: its parts at a position, and where each image byte sits in its lease.
const std = @import("std");
const cuda = @import("cuda");
const st_mod = @import("cuda_state.zig");
const sparse = @import("cuda_sparse.zig");
const Config = @import("config.zig").Config;
const pages = @import("cuda_snapshot_pages.zig");
const State = st_mod.State;
const window_rows = st_mod.window_rows;

/// The image's first bytes: the format, the position, the head's rows, both ring bases and the KDA copy bits. A
/// KV-sharded state (TF_GLM_KV_SHARD) also names its home part and the parts its pool had in the format word; the
/// image's bytes are the same either way, as its latent rows come from the home rank alone (`route`).
pub const Words = struct {
    at: u64,
    mtp_len: u64,
    base: u64,
    mtp_base: u64,
    cur: u64,
    home: ?u8 = null, // the part that held its latent rows when it was written (null: unsharded)

    pub const count = 6;
    pub const bytes: u64 = count * 8;
    const format: u64 = 1;
    const sharded: u64 = 3; // | home << 8 | kv_parts << 16
    pub const kv_parts: u64 = 2; // a sharded pool's parts (TP=2 alone)

    pub fn encode(w: Words) [count]u64 {
        const f = if (w.home) |h| if (mut(.words_no_home)) format else sharded | @as(u64, h) << 8 | kv_parts << 16 else format;
        return .{ f, w.at, w.mtp_len, w.base, w.mtp_base, w.cur };
    }

    /// Words read back: refused unless the format is one of these and the state stands at `at`; a sharded state's
    /// home within its parts, and those parts the two of TP=2.
    pub fn decode(x: [count]u64, at: u32) !Words {
        const home: ?u8 = switch (x[0] & 0xff) {
            format => if (x[0] == format) null else return error.DiskWords,
            sharded => if (x[0] >> 24 == 0 and (x[0] >> 16 & 0xff) == kv_parts and (x[0] >> 8 & 0xff) < kv_parts) @intCast(x[0] >> 8 & 0xff) else return error.DiskWords,
            else => return error.DiskWords,
        };
        if (x[1] != at) return error.DiskWords;
        return .{ .at = x[1], .mtp_len = x[2], .base = x[3], .mtp_base = x[4], .cur = x[5], .home = home };
    }
};

/// A part as the rows store takes it: `row` bytes for each `per` tokens from position 0, or the state's own (row 0).
pub const Part = struct { bytes: u64, row: u32 = 0, per: u32 = 1 };

/// `len` image bytes from `img`: the words, `lease` bytes into the state's lease, zeros the lease does not hold, or
/// latent rows this rank's copy of a sharded state never held (`absent`: another rank's, never read or written here).
pub const Seg = struct { img: u64, len: u64, from: union(enum) { words, lease: u64, zeros, absent } };

/// What a rank's kept state holds, read from a sequence of the engine (its addresses are not used past here).
pub const Shape = struct {
    prefix: u64, // the sequence's own fields before its caches: KDA conv and states, a window's saved rows
    tail: u64, // its fields after them: the head's position and first-draft row
    layers: u32, // DSA layers, whose rows are shared by position
    latent: u32, // a latent row's bytes
    scales: bool, // fp8 latent rows: a 4-byte scale each
    ring_rows: u32,
    ring_row: u32,
    pk_rows: u32,
    pooled_row: u32,
    capacity: u32,
    head: bool, // the MTP head's caches are kept (drafting)

    pub fn of(st: *const State, c: *const Config, capacity: usize, head: bool) !Shape {
        if (st.rings.len == 0) return error.SnapshotShape;
        const base = st.arena.buf.ptr;
        const first = if (st.table != 0) st.rings[0] else if (st.caches.len > 0) st.caches[0].lc else st.mtp_caches.lc;
        const end = if (st.table != 0) st.table else base + st.arena.buf.len;
        if (first < base or st.mtp_pos_dev < first or end < st.mtp_pos_dev) return error.SnapshotShape;
        const fp8 = st.mtp_caches.ls != 0;
        return .{
            .prefix = first - base,
            .tail = end - st.mtp_pos_dev,
            .layers = @intCast(st.caches.len),
            .latent = c.kv_lora * @as(u32, if (fp8) 1 else 2),
            .scales = fp8,
            .ring_rows = @intCast(st.ring_rows),
            .ring_row = @intCast(st.ring_row),
            .pk_rows = @intCast(capacity / sparse.pool + 2),
            .pooled_row = c.i_dim * 2,
            .capacity = @intCast(capacity),
            .head = head,
        };
    }

    /// Rows the image keeps of the head's caches at `at`: a verify window past it, as the cache charges a state.
    pub fn headRows(sh: Shape, at: u32) u32 {
        return @min(sh.capacity, at + window_rows);
    }
};

/// One cache's parts in the lease's order, for `n` rows: latent, scales, the two index rings, the pooled keys.
fn cache(sh: Shape, n: u64) [5]u64 {
    return .{
        n * sh.latent,
        if (sh.scales) n * 4 else 0,
        @min(n, sh.ring_rows) * sh.ring_row,
        @min(n, sh.ring_rows) * sh.ring_row,
        @min(n / sparse.pool + 2, sh.pk_rows) * sh.pooled_row,
    };
}

/// The image at `at` (a chunk start): the words, the prefix, each DSA layer's rows, the head's caches, the tail.
pub fn parts(sh: Shape, at: u32, out: *std.ArrayList(Part), a: std.mem.Allocator) !void {
    if (at == 0 or at % sparse.pool != 0 or at > sh.capacity) return error.SnapshotShape;
    try out.appendSlice(a, &.{ .{ .bytes = Words.bytes }, .{ .bytes = sh.prefix } });
    const pooled: u64 = at / sparse.pool;
    for (0..sh.layers) |_| {
        const c = cache(sh, at);
        try out.append(a, .{ .bytes = c[0], .row = sh.latent });
        if (sh.scales) try out.append(a, .{ .bytes = c[1], .row = 4 });
        try out.appendSlice(a, &.{ .{ .bytes = c[2] }, .{ .bytes = c[3] } });
        // whole pools are rows by position; the pool past them and the spare row stay the state's own
        try out.appendSlice(a, &.{ .{ .bytes = pooled * sh.pooled_row, .row = sh.pooled_row, .per = sparse.pool }, .{ .bytes = c[4] - pooled * sh.pooled_row } });
    }
    if (sh.head) for (cache(sh, sh.headRows(at))) |n| if (n > 0) try out.append(a, .{ .bytes = n });
    try out.append(a, .{ .bytes = sh.tail });
}

/// The image's segments in order for a state at `at` whose head holds `mtp_len` rows (0 without a head).
pub fn segments(sh: Shape, at: u32, mtp_len: u64, out: *std.ArrayList(Seg), a: std.mem.Allocator) !void {
    return segmentsFor(sh, at, mtp_len, true, out, a);
}

/// `segments` of a rank's copy with its latent rows or without them (`latent` false: a KV-sharded state off its home
/// rank, whose lease packs every other part back to back, cuda_snapshots.keepAs): the image keeps their place, absent.
pub fn segmentsFor(sh: Shape, at: u32, mtp_len: u64, latent: bool, out: *std.ArrayList(Seg), a: std.mem.Allocator) !void {
    if (sh.head and mtp_len > sh.headRows(at)) return error.SnapshotShape;
    var img: u64 = Words.bytes;
    var lease: u64 = 0;
    try out.append(a, .{ .img = 0, .len = Words.bytes, .from = .words });
    if (latent or mut(.copy_keeps_latent)) {
        const f = front(sh, at);
        try out.append(a, .{ .img = img, .len = f, .from = .{ .lease = 0 } });
        img += f;
        lease += f;
    } else {
        if (sh.prefix > 0) try out.append(a, .{ .img = img, .len = sh.prefix, .from = .{ .lease = 0 } });
        img += sh.prefix;
        lease += sh.prefix;
        const c = cache(sh, at);
        for (0..sh.layers) |_| {
            if (c[0] + c[1] > 0) try out.append(a, .{ .img = img, .len = c[0] + c[1], .from = .absent });
            img += c[0] + c[1];
            const rest = c[2] + c[3] + c[4];
            if (rest > 0) try out.append(a, .{ .img = img, .len = rest, .from = .{ .lease = lease } });
            img += rest;
            lease += rest;
        }
    }
    if (sh.head) for (cache(sh, sh.headRows(at)), cache(sh, mtp_len)) |full, held| {
        if (held > 0) try out.append(a, .{ .img = img, .len = held, .from = .{ .lease = lease } });
        if (full > held) try out.append(a, .{ .img = img + held, .len = full - held, .from = .zeros });
        img += full;
        lease += held;
    };
    if (sh.tail > 0) try out.append(a, .{ .img = img, .len = sh.tail, .from = .{ .lease = lease } });
}

/// The image bytes after the words that sit in the lease as in the image: the prefix and every DSA layer's parts.
pub fn front(sh: Shape, at: u32) u64 {
    var n: u64 = sh.prefix;
    for (0..sh.layers) |_| for (cache(sh, at)) |x| {
        n += x;
    };
    return n;
}

/// The lease bytes of a state at `at` whose head holds `mtp_len` rows (layout.Plan's length for it).
pub fn leaseBytes(sh: Shape, at: u32, mtp_len: u64) u64 {
    var n: u64 = sh.prefix + sh.tail;
    for (0..sh.layers) |_| for (cache(sh, at)) |x| {
        n += x;
    };
    if (sh.head) for (cache(sh, mtp_len)) |x| {
        n += x;
    };
    return n;
}

/// Rank 0's image bytes at `at` (its head counted at a verify window past `at`).
pub fn imageBytes(sh: Shape, at: u32) u64 {
    var n: u64 = Words.bytes + front(sh, at) + sh.tail;
    if (sh.head) for (cache(sh, sh.headRows(at))) |x| {
        n += x;
    };
    return n;
}

/// `len` bytes at `off` of rank `rank`'s image; `shared`: past rank 0's prefix, the same bytes every rank holds.
pub const Piece = struct { rank: u32, off: u64, len: u64, shared: bool };

/// The store's image at `at`: rank 0's, then each other rank's prefix (rank 0's bytes stand for all the rest).
pub fn storeParts(sh: Shape, prefixes: []const u64, at: u32, out: *std.ArrayList(Part), a: std.mem.Allocator) !void {
    try parts(sh, at, out, a);
    for (prefixes[1..]) |p| try out.append(a, .{ .bytes = p });
}

/// Store image bytes [off, off + len) split by rank: rank 0's image, then rank r's prefix after its words. `home`: a
/// KV-sharded state's home part (null: unsharded), whose rank alone holds the latent rows; their bytes are its own
/// there (shared false), at the same place past its prefix as in rank 0's image.
pub fn route(sh: Shape, prefixes: []const u64, at: u32, off: u64, len: u64, home: ?u8, out: *std.ArrayList(Piece), a: std.mem.Allocator) !void {
    const own = Words.bytes + sh.prefix; // rank 0's bytes no other rank holds
    if (home) |h| if (h >= prefixes.len) return error.SnapshotShape;
    var start: u64 = 0; // the section's first store byte
    for (prefixes, 0..) |p, r| {
        const size = if (r == 0) imageBytes(sh, at) else p;
        defer start += size;
        const lo = @max(off, start);
        const hi = @min(off + len, start + size);
        if (lo >= hi) continue;
        if (r > 0) {
            try out.append(a, .{ .rank = @intCast(r), .off = Words.bytes + lo - start, .len = hi - lo, .shared = false });
            continue;
        }
        if (lo < own) try out.append(a, .{ .rank = 0, .off = lo, .len = @min(hi, own) - lo, .shared = false });
        if (hi <= own) continue;
        const h = (if (mut(.route_ignores_home)) null else home) orelse {
            try out.append(a, .{ .rank = 0, .off = @max(lo, own), .len = hi - @max(lo, own), .shared = true });
            continue;
        };
        var x = @max(lo, own);
        while (x < hi) {
            const run = latentRun(sh, at, x);
            const end = @min(hi, run.end);
            if (run.latent) {
                try out.append(a, .{ .rank = h, .off = x - prefixes[0] + prefixes[h], .len = end - x, .shared = false });
            } else try out.append(a, .{ .rank = 0, .off = x, .len = end - x, .shared = true });
            x = end;
        }
    }
    if (off + len > start) return error.SnapshotShape;
}

/// Whether rank 0's image byte `x` (past its prefix) lies in a DSA layer's latent codes or scales, and where that run
/// of latent or other bytes ends.
fn latentRun(sh: Shape, at: u32, x: u64) struct { latent: bool, end: u64 } {
    const first = Words.bytes + sh.prefix;
    const c = cache(sh, at);
    const layer = c[0] + c[1] + c[2] + c[3] + c[4];
    if (x < first or layer == 0 or x >= first + sh.layers * layer) return .{ .latent = false, .end = std.math.maxInt(u64) };
    const j = (x - first) / layer;
    const o = (x - first) % layer;
    const lat = c[0] + c[1];
    if (o < lat) return .{ .latent = true, .end = first + j * layer + lat };
    return .{ .latent = false, .end = first + (j + 1) * layer };
}

/// The bytes past a rank's prefix that every rank's image shares and a rank's copy holds, as (offset past its prefix,
/// length): all of them, or (`latent` false: a sharded state off its home rank) all but the latent codes and scales.
pub fn heldShared(sh: Shape, at: u32, latent: bool, out: *std.ArrayList([2]u64), a: std.mem.Allocator) !void {
    const end = imageBytes(sh, at) - Words.bytes - sh.prefix;
    if (latent) return out.append(a, .{ 0, end });
    const c = cache(sh, at);
    var x: u64 = 0;
    for (0..sh.layers) |_| {
        x += c[0] + c[1];
        const rest = c[2] + c[3] + c[4];
        if (rest > 0) try out.append(a, .{ x, rest });
        x += rest;
    }
    if (end > x) try out.append(a, .{ x, end - x });
}

/// Host test mutants, each caught by a disk image test (cuda_disk_shard_test).
pub const Mutant = enum { none, words_no_home, copy_keeps_latent, route_ignores_home };
pub var mutant: Mutant = .none;

fn mut(comptime m: Mutant) bool {
    return @import("builtin").is_test and mutant == m;
}

/// Copies between `host` and image bytes [off, off + host.len) of a lease: out (lease to host) or in.
pub const Copy = struct {
    d: *const cuda.Driver,
    pool: *const pages.Pool,
    lease: *const pages.Lease,
    segs: []const Seg,
    words: *[Words.count]u64,

    pub fn run(c: Copy, off: u64, host: []u8, out: bool) !void {
        const end = off + host.len;
        for (c.segs) |s| {
            const lo = @max(off, s.img);
            const hi = @min(end, s.img + s.len);
            if (lo >= hi) continue;
            const h = host[@intCast(lo - off)..][0..@intCast(hi - lo)];
            switch (s.from) {
                .words => {
                    const w = std.mem.asBytes(c.words)[@intCast(lo - s.img)..][0..h.len];
                    if (out) @memcpy(h, w) else @memcpy(w, h);
                },
                .zeros => if (out) @memset(h, 0),
                .absent => return error.KvShardAbsent, // another rank's latent rows (route sends none here)
                .lease => |at| try c.leased(at + (lo - s.img), h, out),
            }
        }
    }

    /// Lease bytes from `at`, a page at a time: the lease's device pages first, then its pinned host pages.
    fn leased(c: Copy, at: u64, host: []u8, out: bool) !void {
        if (at + host.len > c.lease.count() * pages.page) return error.SnapshotSize;
        const nd = c.lease.dev.ids.items.len;
        var done: usize = 0;
        while (done < host.len) {
            const x = at + done;
            const k: usize = @intCast(x / pages.page);
            const n = @min(host.len - done, pages.page - x % pages.page);
            const p = host[done..][0..n];
            if (k < nd) {
                const arena = c.pool.arena orelse return error.NoSnapshotRoom;
                const dev = arena.ptr + @as(u64, c.lease.dev.ids.items[k]) * pages.page + x % pages.page;
                if (out) {
                    try c.d.check(c.d.api.cuMemcpyDtoH_v2(p.ptr, dev, n), "cuMemcpyDtoH");
                } else try c.d.check(c.d.api.cuMemcpyHtoD_v2(dev, p.ptr, n), "cuMemcpyHtoD");
            } else {
                const buf = c.pool.host orelse return error.NoSnapshotRoom;
                const from = @as(usize, c.lease.host.ids.items[k - nd]) * pages.page + @as(usize, @intCast(x % pages.page));
                const h = buf.bytes[from..][0..n];
                if (out) @memcpy(p, h) else @memcpy(h, p);
            }
            done += n;
        }
    }
};

test {
    _ = @import("cuda_disk_image_test.zig");
}
