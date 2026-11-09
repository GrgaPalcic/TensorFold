//! A GLM kept state's disk image across ranks: rank 0's engine thread leases, fills, reads and seals each rank's copy.
//! A KV-sharded state (TF_GLM_KV_SHARD) is written and read as each rank holds it: its latent rows on its home rank
//! alone, every other byte on every rank; the store image is the same as an unsharded state's.
const std = @import("std");
const follow = @import("cuda_follow.zig");
const image = @import("cuda_disk_image.zig");
const Engine = @import("cuda_engine.zig").Engine;

/// A disk step's op (Header.flags); its id is `a`, its image offset `seed`, its bytes `depth`, its rank `top_k`; a
/// blank's home part + 1 in `depth` (0: unsharded).
pub const Op = enum(u32) { blank = 1, put, take, seal, prefixes };
const kind = @backingInt(follow.Kind.disk); // each op crosses the follow link as a disk step, its bytes as the ids

/// This rank's image shape (every sequence of the engine has the same).
pub fn shape(e: anytype) !image.Shape {
    return image.Shape.of(&e.cur.st, &e.c, e.capacity, e.mtp);
}

/// A kept state's home part under TF_GLM_KV_SHARD, the part whose rank holds its latent rows (null: unsharded).
pub fn homeOf(e: anytype, s: anytype) ?u8 {
    return if (e.settings.kv_shard.on) s.home else null;
}

/// Whether this rank's copy of a kept state holds its latent rows: unsharded every rank's does, sharded its home's.
fn holdsLatent(e: anytype, s: anytype) bool {
    const h = homeOf(e, s) orelse return true;
    return h == e.share.rank;
}

/// The words of a kept state on this rank: its position, head rows, ring bases, KDA copy bits and home part.
pub fn wordsOf(e: anytype, id: u64) !image.Words {
    const s = e.snaps.get(id) orelse return error.NoKeptState;
    if (s.cur.len > 64) return error.SnapshotShape;
    var cur: u64 = 0;
    for (s.cur, 0..) |b, i| cur |= @as(u64, b) << @intCast(i);
    return .{ .at = s.pos, .mtp_len = s.mtp_len, .base = s.bases[0], .mtp_base = s.bases[1], .cur = cur, .home = homeOf(e, s) };
}

/// Bytes [off, off + host.len) of this rank's image of `id`, out (lease to host) or in; unsealed: before its head only.
/// Off its home rank a sharded state's latent rows are absent (KvShardAbsent): `route` asks only the home for them.
pub fn copy(e: anytype, id: u64, off: u64, host: []u8, out: bool) !void {
    const s = e.snaps.get(id) orelse return error.NoKeptState;
    const sh = try shape(e);
    const at: u32 = @intCast(s.pos);
    if (s.len == 0 and off + host.len > image.Words.bytes + image.front(sh, at)) return error.DiskOrder;
    var segs: std.ArrayList(image.Seg) = .empty;
    defer segs.deinit(e.gpa);
    try image.segmentsFor(sh, at, s.mtp_len, holdsLatent(e, s), &segs, e.gpa);
    var w = (try wordsOf(e, id)).encode();
    const pool = if (e.snap_pages) |*p| p else return error.NoSnapshotCache;
    try (image.Copy{ .d = e.d, .pool = pool, .lease = &s.lease, .segs = segs.items, .words = &w }).run(off, host, out);
}

/// Rank 0's link at TP, else null (one rank: every op is local).
fn leader(e: *Engine) ?follow.Leader {
    const l = e.link orelse return null;
    return .{ .e = e, .link = @ptrCast(@alignCast(l)) };
}

/// Rank 0: whether the other ranks sleep on their lines (no stream), and putting them back to sleep after disk steps.
pub fn asleep(e: *Engine) bool {
    return if (leader(e)) |x| x.link.asleep else false;
}

pub fn rest(e: *Engine) !void {
    if (leader(e)) |x| try x.idle();
}

/// Rank 0: a disk read's home part under TF_GLM_KV_SHARD, the part a new stream would take now (Parts.choose with no
/// home: fewer live streams, then more free pages; part 0 when neither has a page). The words' home is where the
/// state was written, not where the stream that reads it will run, so it does not bind: the image is the same for
/// either home. Null unsharded.
pub fn readHome(e: anytype) ?u8 {
    if (!e.settings.kv_shard.on or mut(.read_unsharded)) return null;
    const p = e.pages orelse return null;
    const x = p.shard orelse return null;
    return x.parts.choose(1, null) orelse 0;
}

/// Rank 0: a new id leased at `at` on every rank (rank 0 first: the ranks' pools match, so theirs cannot fail apart),
/// under TF_GLM_KV_SHARD in its home part (`readHome`) on every rank.
pub fn blank(e: *Engine, at: u32) !u64 {
    if (e.next_snap > std.math.maxInt(u32)) return error.SnapshotIdLimit;
    const id = e.next_snap;
    const home = readHome(e);
    try e.blankAs(id, at, home);
    e.next_snap += 1;
    if (leader(e)) |x| try x.disk(.{ .kind = kind, .flags = @backingInt(Op.blank), .a = id, .seed = at, .depth = if (home) |h| @as(u32, h) + 1 else 0 }, &.{});
    return id;
}

/// Rank 0: `bytes` into rank `rank`'s image of `id` at `off` (rank 0's own lease, or through the link in pieces).
pub fn put(e: *Engine, rank: u32, id: u64, off: u64, bytes: []const u8) !void {
    if (rank == 0) return copy(e, id, off, @constCast(bytes), false);
    const x = leader(e) orelse return error.NoRank;
    const room = x.link.ids.len * 4;
    var done: usize = 0;
    while (done < bytes.len) {
        const n = @min(room, bytes.len - done);
        const words = (n + 3) / 4;
        @memcpy(std.mem.sliceAsBytes(x.link.ids[0..words])[0..n], bytes[done..][0..n]);
        try x.disk(.{ .kind = kind, .flags = @backingInt(Op.put), .a = id, .seed = off + done, .depth = @intCast(n), .top_k = rank }, x.link.ids[0..words]);
        done += n;
    }
}

/// Rank 0: rank `rank`'s image bytes of `id` from `off` into `out`.
pub fn take(e: *Engine, rank: u32, id: u64, off: u64, out: []u8) !void {
    if (rank == 0) return copy(e, id, off, out, true);
    const x = leader(e) orelse return error.NoRank;
    const room = x.link.ids.len * 4;
    var done: usize = 0;
    while (done < out.len) {
        const n = @min(room, out.len - done);
        try x.disk(.{ .kind = kind, .flags = @backingInt(Op.take), .a = id, .seed = off + done, .depth = @intCast(n), .top_k = rank }, &.{});
        try x.link.from(rank, n);
        try x.link.buf.download(64, out[done..][0..n]);
        done += n;
    }
}

/// Rank 0: every rank's blank `id` sealed with `w` (its head's rows, ring bases and KDA copy bits).
pub fn seal(e: *Engine, id: u64, w: image.Words) !void {
    try e.sealState(id, w.mtp_len, .{ w.base, w.mtp_base }, w.cur);
    if (leader(e)) |x| {
        const words = w.encode();
        try x.disk(.{ .kind = kind, .flags = @backingInt(Op.seal), .a = id }, std.mem.bytesAsSlice(u32, std.mem.sliceAsBytes(&words)));
    }
}

/// Rank 0: every rank's prefix bytes in rank order (the part of a state each rank holds alone).
pub fn prefixes(e: *Engine, out: []u64) !void {
    const mine = (try shape(e)).prefix;
    const x = leader(e) orelse {
        if (out.len != 1) return error.NoRank;
        out[0] = mine;
        return;
    };
    try x.disk(.{ .kind = kind, .flags = @backingInt(Op.prefixes) }, &.{});
    try gather(x.link, e, mine, out);
}

fn gather(l: *follow.Link, e: *const Engine, mine: u64, out: []u64) !void {
    try @import("cuda_ranks.zig").gather(u64, l.d, l.lib, l.comm, l.stream, e.share.rank, mine, out);
}

/// Store image bytes of a state at `at` (home part `home`) over `ranges` (the rows store's, each an `off` and `len`)
/// into `out`, each piece from the rank that holds it: `io.take(rank, off, bytes)` reads rank `rank`'s image.
pub fn gatherWith(io: anytype, sh: image.Shape, pre: []const u64, at: u32, home: ?u8, ranges: anytype, out: []u8, gpa: std.mem.Allocator) !void {
    var pieces: std.ArrayList(image.Piece) = .empty;
    defer pieces.deinit(gpa);
    var o: usize = 0;
    for (ranges) |g| {
        pieces.clearRetainingCapacity();
        try image.route(sh, pre, at, g.off, g.len, home, &pieces, gpa);
        for (pieces.items) |p| {
            const n: usize = @intCast(p.len);
            try io.take(p.rank, p.off, out[o..][0..n]);
            o += n;
        }
    }
}

/// Store image bytes over `ranges` from `in` into every rank's image of a blank state that holds them (the words
/// skipped: they went with the seal): rank 0's shared bytes to every rank, a sharded state's latent rows to its home's.
/// `io.put(rank, off, bytes)` writes rank `rank`'s image.
pub fn scatterWith(io: anytype, sh: image.Shape, pre: []const u64, at: u32, home: ?u8, ranges: anytype, in: []const u8, gpa: std.mem.Allocator) !void {
    var pieces: std.ArrayList(image.Piece) = .empty;
    defer pieces.deinit(gpa);
    var o: usize = 0;
    for (ranges) |g| {
        pieces.clearRetainingCapacity();
        try image.route(sh, pre, at, g.off, g.len, home, &pieces, gpa);
        for (pieces.items) |p| {
            const n: usize = @intCast(p.len);
            const bytes = in[o..][0..n];
            o += n;
            if (p.rank == 0 and p.off < image.Words.bytes) continue; // the words went with the seal
            try io.put(p.rank, p.off, bytes);
            if (p.shared) for (pre[1..], 1..) |q, r| try io.put(@intCast(r), p.off - pre[0] + q, bytes);
        }
    }
}

/// Host test mutants, each caught by a disk image test (cuda_disk_shard_test).
pub const Mutant = enum { none, read_unsharded };
pub var mutant: Mutant = .none;

fn mut(comptime m: Mutant) bool {
    return @import("builtin").is_test and mutant == m;
}

/// A rank other than 0: rank 0's disk step on this rank's copy of its state.
pub fn step(e: *Engine, l: *follow.Link, h: follow.Header, ids: []const u32) !void {
    if (h.flags < @backingInt(Op.blank) or h.flags > @backingInt(Op.prefixes)) return error.BadStep;
    const op: Op = @fromBackingInt(h.flags);
    const rank: u32 = @intCast(e.share.rank);
    switch (op) {
        .blank => {
            if (h.depth > image.Words.kv_parts or (h.depth != 0 and !e.settings.kv_shard.on)) return error.BadStep;
            try e.blankAs(h.a, @intCast(h.seed), if (h.depth == 0) null else @intCast(h.depth - 1));
        },
        .seal => {
            if (ids.len != 2 * image.Words.count) return error.BadStep;
            var x: [image.Words.count]u64 = undefined;
            @memcpy(std.mem.sliceAsBytes(&x), std.mem.sliceAsBytes(ids));
            const w = try image.Words.decode(x, @intCast((e.snaps.get(h.a) orelse return error.NoKeptState).pos));
            try e.sealState(h.a, w.mtp_len, .{ w.base, w.mtp_base }, w.cur);
        },
        .put => if (h.top_k == rank) {
            if (h.depth > ids.len * 4) return error.BadStep;
            try copy(e, h.a, h.seed, @constCast(std.mem.sliceAsBytes(ids))[0..h.depth], false);
        },
        .take => {
            if (h.depth > l.ids.len * 4) return error.BadStep;
            if (h.top_k == rank) {
                const host = std.mem.sliceAsBytes(l.ids)[0..h.depth];
                try copy(e, h.a, h.seed, host, true);
                try l.buf.uploadAsync(64, host, l.stream.handle);
            }
            try l.from(h.top_k, h.depth);
        },
        .prefixes => {
            var out: [64]u64 = undefined;
            if (e.share.world > out.len) return error.BadStep;
            try gather(l, e, (try shape(e)).prefix, out[0..e.share.world]);
        },
    }
}
