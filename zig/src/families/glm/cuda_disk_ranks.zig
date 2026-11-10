//! A GLM kept state's disk image across ranks: rank 0's engine thread leases, fills, reads and seals each rank's copy.
//! A KV-sharded state (TF_GLM_KV_SHARD) is written and read as each rank holds it: its latent rows on its home rank
//! alone, every other byte on every rank; the store image is the same as an unsharded state's.
const std = @import("std");
const cuda = @import("cuda");
const follow = @import("cuda_follow.zig");
const image = @import("cuda_disk_image.zig");
const pages_mod = @import("cuda_pages.zig");
const snapshots = @import("cuda_snapshots.zig");
const kvs = @import("cuda_kv_shard.zig");
const Engine = @import("cuda_engine.zig").Engine;

/// A disk step's op (Header.flags); its id is `a`, its image offset `seed`, its bytes `depth`, its rank `top_k`; a
/// blank's home part + 1 in `depth` and the pages a recall holds in `top_k`. A page op's page is `a` (a key page's
/// global id, a latent page's part-local one), the byte in the page `seed`: rank 0's pool pages are read and written
/// through them (the other ranks' beside TF_GLM_KV_SHARD and on a read-back), a slot's lease through `slot`.
pub const Op = enum(u32) { blank = 1, put, take, seal, prefixes, key_put, lat_put, lat_take, slot_put, slot_take, page_put, page_take };
const kind = @backingInt(follow.Kind.disk); // each op crosses the follow link as a disk step, its bytes as the ids

/// A take's answer crosses the link as its bytes and one status word past them (0 ok, else the op failed on the
/// rank that answered): rank 0 fails the op here instead of gathering what a broken rank's copy sent. A seal's
/// answer is the status word alone, rank order: a rank whose ops failed for the id fails rank 0's seal with it.
const status_bytes: usize = @sizeOf(u32);

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
    return .{ .at = s.pos, .mtp_len = s.mtp_len, .base = s.bases[0], .mtp_base = s.bases[1], .cur = cur, .home = homeOf(e, s), .pages = s.shared > 0 };
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
/// under TF_GLM_KV_SHARD in its home part on every rank, and with the pages layout's shared page count recorded on
/// every rank (a recall's: its slot's parts past the pages, on every rank as a keep records its own).
pub fn blank(e: *Engine, at: u32, home: ?u8, shared: u32) !u64 {
    if (e.next_snap > std.math.maxInt(u32)) return error.SnapshotIdLimit;
    const id = e.next_snap;
    try e.blankAs(id, at, home, shared);
    e.next_snap += 1;
    if (leader(e)) |x| try x.disk(.{ .kind = kind, .flags = @backingInt(Op.blank), .a = id, .seed = at, .depth = if (home) |h| @as(u32, h) + 1 else 0, .top_k = shared }, &.{});
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

/// Rank 0: rank `rank`'s image bytes of `id` from `off` into `out`; its answer carries this rank's status word past
/// the bytes, a rank whose op failed failing the take here (its state poisoned, its half-made copy dropped there).
pub fn take(e: *Engine, rank: u32, id: u64, off: u64, out: []u8) !void {
    if (rank == 0) return copy(e, id, off, out, true);
    const x = leader(e) orelse return error.NoRank;
    const room = x.link.ids.len * 4 - status_bytes;
    var done: usize = 0;
    while (done < out.len) {
        const n = @min(room, out.len - done);
        try x.disk(.{ .kind = kind, .flags = @backingInt(Op.take), .a = id, .seed = off + done, .depth = @intCast(n), .top_k = rank }, &.{});
        try x.link.from(rank, n + status_bytes);
        try x.link.buf.download(64, out[done..][0..n]);
        if (try status(x.link, n) != 0) return error.DiskRankFailed;
        done += n;
    }
}

/// The status word a take or seal answered with, `past` bytes into the link's buffer after its header.
fn status(l: *follow.Link, past: usize) !u32 {
    var ok: u32 = 0;
    try l.buf.download(64 + past, std.mem.asBytes(&ok));
    return ok;
}

/// Rank 0: every rank's blank `id` sealed with `w` (its head's rows, ring bases and KDA copy bits). Each rank's seal
/// answers with its step's status: a rank whose disk ops failed for this id (or whose seal refused the words) fails
/// the seal here, so its broken copy costs rank 0's recall a cold pass, never the pair's step loops.
pub fn seal(e: *Engine, id: u64, w: image.Words) !void {
    try e.sealState(id, w.mtp_len, .{ w.base, w.mtp_base }, w.cur);
    if (leader(e)) |x| {
        const words = w.encode();
        try x.disk(.{ .kind = kind, .flags = @backingInt(Op.seal), .a = id }, std.mem.bytesAsSlice(u32, std.mem.sliceAsBytes(&words)));
        var r: u32 = 1;
        while (r < e.share.world) : (r += 1) {
            try x.link.from(r, status_bytes);
            if (try status(x.link, 0) != 0) return error.DiskRankFailed;
        }
    }
}

/// This rank's shard (null: unsharded): its parts' pools, its key pages (every page's, global ids) and its own part's
/// latent pages (part-local ids, the trash page last).
fn shard(e: *Engine) !*pages_mod.Shard {
    const p = e.pages orelse return error.NoPages;
    return p.shard orelse error.SnapshotShape;
}

/// Rank `rank`'s key page `id` (its global id; both ranks hold every page's keys) bytes [off, off + host.len), out
/// (its pages to host) or in; rank 0's directly, another rank's through the link in pieces.
pub fn keyPage(e: *Engine, rank: u32, id: u32, off: u64, host: []u8, out: bool) !void {
    const x0 = try shard(e);
    if (rank == 0) return shardBytes(e, x0.keys orelse return error.NoPages, x0.g.keyPage(), id, off, host, out);
    const x = leader(e) orelse return error.NoRank;
    const room = x.link.ids.len * 4 - status_bytes;
    var done: usize = 0;
    while (done < host.len) {
        const n = @min(room, host.len - done);
        if (out) {
            try x.disk(.{ .kind = kind, .flags = @backingInt(Op.lat_take), .a = id, .seed = off + done, .depth = @intCast(n), .top_k = rank }, &.{});
            try x.link.from(rank, n + status_bytes);
            try x.link.buf.download(64, host[done..][0..n]);
            if (try status(x.link, n) != 0) return error.DiskRankFailed;
        } else {
            @memcpy(std.mem.sliceAsBytes(x.link.ids[0..words4(n)])[0..n], host[done..][0..n]);
            try x.disk(.{ .kind = kind, .flags = @backingInt(Op.key_put), .a = id, .seed = off + done, .depth = @intCast(n), .top_k = rank }, x.link.ids[0..words4(n)]);
        }
        done += n;
    }
}

/// Rank `rank`'s latent page `id` (its part-local id; the home rank's alone hold a sharded state's latent rows) bytes
/// [off, off + host.len), out or in; rank 0's directly, another rank's through the link in pieces.
pub fn latPage(e: *Engine, rank: u32, id: u32, off: u64, host: []u8, out: bool) !void {
    const x0 = try shard(e);
    if (rank == 0) return shardBytes(e, x0.lat orelse return error.NoPages, x0.g.latPage(), id, off, host, out);
    const x = leader(e) orelse return error.NoRank;
    const room = x.link.ids.len * 4 - status_bytes;
    var done: usize = 0;
    while (done < host.len) {
        const n = @min(room, host.len - done);
        if (out) {
            try x.disk(.{ .kind = kind, .flags = @backingInt(Op.lat_take), .a = id, .seed = off + done, .depth = @intCast(n), .top_k = rank }, &.{});
            try x.link.from(rank, n + status_bytes);
            try x.link.buf.download(64, host[done..][0..n]);
            if (try status(x.link, n) != 0) return error.DiskRankFailed;
        } else {
            @memcpy(std.mem.sliceAsBytes(x.link.ids[0..words4(n)])[0..n], host[done..][0..n]);
            try x.disk(.{ .kind = kind, .flags = @backingInt(Op.lat_put), .a = id, .seed = off + done, .depth = @intCast(n), .top_k = rank }, x.link.ids[0..words4(n)]);
        }
        done += n;
    }
}

fn words4(n: usize) usize {
    return (n + 3) / 4;
}

/// One page's bytes [off, off + host.len) in a rank's key or latent pages (a run never crosses a page).
fn shardBytes(e: *Engine, buf: cuda.DeviceBuffer, stride: u64, id: u32, off: u64, host: []u8, out: bool) !void {
    if (off + host.len > stride) return error.SnapshotShape;
    const base = buf.ptr + @as(u64, id) * stride + off;
    if (out) try e.d.check(e.d.api.cuMemcpyDtoH_v2(host.ptr, base, host.len), "cuMemcpyDtoH") else try e.d.check(e.d.api.cuMemcpyHtoD_v2(base, host.ptr, host.len), "cuMemcpyHtoD");
}

/// Rank `rank`'s slot lease bytes [off, off + host.len) of `id`, out or in (its raw lease offsets, a pages layout's
/// fixed parts packed past its prefix); rank 0's directly, another rank's through the link in pieces.
pub fn slot(e: *Engine, rank: u32, id: u64, off: u64, host: []u8, out: bool) !void {
    if (rank == 0) return snapshots.slotBytes(e, e.snaps.getPtr(id) orelse return error.NoKeptState, @intCast(off), host, out);
    const x = leader(e) orelse return error.NoRank;
    const room = x.link.ids.len * 4 - status_bytes;
    var done: usize = 0;
    while (done < host.len) {
        const n = @min(room, host.len - done);
        if (out) {
            try x.disk(.{ .kind = kind, .flags = @backingInt(Op.slot_take), .a = @intCast(id), .seed = off + done, .depth = @intCast(n), .top_k = rank }, &.{});
            try x.link.from(rank, n + status_bytes);
            try x.link.buf.download(64, host[done..][0..n]);
            if (try status(x.link, n) != 0) return error.DiskRankFailed;
        } else {
            @memcpy(std.mem.sliceAsBytes(x.link.ids[0..words4(n)])[0..n], host[done..][0..n]);
            try x.disk(.{ .kind = kind, .flags = @backingInt(Op.slot_put), .a = @intCast(id), .seed = off + done, .depth = @intCast(n), .top_k = rank }, x.link.ids[0..words4(n)]);
        }
        done += n;
    }
}

/// Rank `rank`'s pool page `id` (unsharded: its whole pages in its arena) bytes [off, off + host.len), out or in;
/// rank 0's directly, another rank's through the link in pieces (a read-back's rows to every rank's pages).
pub fn poolPage(e: *Engine, rank: u32, id: u32, off: u64, host: []u8, out: bool) !void {
    if (rank == 0) {
        const p = e.pages orelse return error.NoPages;
        if (p.shard != null) return error.SnapshotShape;
        const arena = if (p.arena) |a| a.ptr else return error.NoPages;
        const stride = p.layout.pageBytes();
        if (off + host.len > stride) return error.SnapshotShape;
        const base = arena + @as(u64, id) * stride + off;
        if (out) try e.d.check(e.d.api.cuMemcpyDtoH_v2(host.ptr, base, host.len), "cuMemcpyDtoH") else try e.d.check(e.d.api.cuMemcpyHtoD_v2(base, host.ptr, host.len), "cuMemcpyHtoD");
        return;
    }
    const x = leader(e) orelse return error.NoRank;
    const room = x.link.ids.len * 4 - status_bytes;
    var done: usize = 0;
    while (done < host.len) {
        const n = @min(room, host.len - done);
        if (out) {
            try x.disk(.{ .kind = kind, .flags = @backingInt(Op.page_take), .a = id, .seed = off + done, .depth = @intCast(n), .top_k = rank }, &.{});
            try x.link.from(rank, n + status_bytes);
            try x.link.buf.download(64, host[done..][0..n]);
            if (try status(x.link, n) != 0) return error.DiskRankFailed;
        } else {
            @memcpy(std.mem.sliceAsBytes(x.link.ids[0..words4(n)])[0..n], host[done..][0..n]);
            try x.disk(.{ .kind = kind, .flags = @backingInt(Op.page_put), .a = id, .seed = off + done, .depth = @intCast(n), .top_k = rank }, x.link.ids[0..words4(n)]);
        }
        done += n;
    }
}

/// A pages-layout sharded state's row part gathered or scattered through the pool's pages: byte `b` of the part sits
/// in page `pages[b / span]` at `b % span` past the part's place in its page (`part.at`); the pooled keys are rank
/// 0's key pages (every rank's key pages get them on the way back in), the latent codes and scales the home rank's
/// latent pages. `out`: gather (the pages to host) or scatter.
pub fn pageRows(e: *Engine, part: image.Part, sh: image.Shape, home: u8, pages: []const u32, b: u64, host: []u8, out: bool) !void {
    const x = try shard(e);
    const span = image.pageSpan(sh, part);
    const keys = part.row != sh.latent and part.row != 4;
    const h: u32 = @intCast(x.parts.h);
    var done: usize = 0;
    while (done < host.len) {
        const y = b + done;
        const k = y / span;
        if (k >= pages.len) return error.SnapshotShape;
        const n: usize = @intCast(@min(host.len - done, span - y % span));
        const chunk = host[done..][0..n];
        const off = part.at + y % span;
        if (keys) {
            try keyPage(e, 0, pages[k], off, chunk, out);
            if (!out) for (1..e.share.world) |r| try keyPage(e, @intCast(r), pages[k], off, chunk, out);
        } else {
            const g = pages[k];
            if (kvs.partOf(g, x.parts.h) != home) return error.SnapshotShape; // a kept state's pages lie in its home part
            try latPage(e, home, g - home * h, off, chunk, out);
        }
        done += n;
    }
}

/// A pages-layout unsharded state's row part written into every rank's pool pages (every rank holds the rows; a
/// read-back's scatter): rank 0's directly, the others through the link.
pub fn poolRows(e: *Engine, part: image.Part, sh: image.Shape, pages: []const u32, b: u64, host: []const u8) !void {
    const p = e.pages orelse return error.NoPages;
    if (p.shard != null) return error.SnapshotShape;
    const span = image.pageSpan(sh, part);
    var done: usize = 0;
    while (done < host.len) {
        const y = b + done;
        const k = y / span;
        if (k >= pages.len) return error.SnapshotShape;
        const n: usize = @intCast(@min(host.len - done, span - y % span));
        const off = part.at + y % span;
        try poolPage(e, 0, pages[k], off, @constCast(host[done..][0..n]), false);
        for (1..e.share.world) |r| try poolPage(e, @intCast(r), pages[k], off, @constCast(host[done..][0..n]), false);
        done += n;
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

/// A disk op that failed on this rank: the id poisoned (its later writes skipped, its seal and takes answer with
/// the failure) and its half-made state dropped — rank 0's own drop covers the pair when its recall gives up. The
/// error stays here: out of the step loop it would end this rank and with it the lane (a failed disk recall or
/// save costs rank 0 a cold pass or a skipped save, never the ranks).
fn poison(e: *Engine, id: u64, err: anyerror) void {
    std.log.warn("rank {d}: a disk step for state {d} failed ({s}); the state is dropped here and rank 0 gives its recall or save up", .{ e.share.rank, id, @errorName(err) });
    e.disk_failed.put(e.gpa, id, {}) catch {};
    e.dropState(id);
}

/// A take's answer on this rank: its bytes into the link's ids with this rank's status word past them (a poisoned
/// id's bytes are stale and its status says so; a failing op poisons the id and answers with its failure), then the
/// link's broadcast the op's answer crosses by. The step loop runs on either way.
fn answered(e: *Engine, l: *follow.Link, h: follow.Header, rank: u32, state: ?u64, body: *const fn (*Engine, follow.Header, []u8) anyerror!void) !void {
    if (h.depth > l.ids.len * 4 - status_bytes) return error.BadStep;
    var ok: u32 = 0;
    if (h.top_k == rank) {
        const bytes = std.mem.sliceAsBytes(l.ids);
        const host = bytes[0..h.depth];
        // a failed read answers with its failure and nothing more: the state it read stays (rank 0 still holds its
        // copy and may resume from it; a save of it is what is skipped). A state a failed recall dropped here fails
        // its reads by being gone.
        if (state != null and e.disk_failed.contains(state.?)) ok = 1 else if (body(e, h, host)) |_| {} else |err| {
            std.log.warn("rank {d}: a disk read for {d} failed ({s}); rank 0's save of it is skipped", .{ e.share.rank, h.a, @errorName(err) });
            ok = 1;
        }
        std.mem.writeInt(u32, bytes[h.depth..][0..status_bytes], ok, .little);
        l.buf.uploadAsync(64, bytes[0 .. h.depth + status_bytes], l.stream.handle) catch {};
    }
    try l.from(h.top_k, h.depth + status_bytes);
}

fn takeBody(e: *Engine, h: follow.Header, host: []u8) anyerror!void {
    return copy(e, h.a, h.seed, host, true);
}

fn latTakeBody(e: *Engine, h: follow.Header, host: []u8) anyerror!void {
    return shardBytes(e, (try shard(e)).lat orelse return error.NoPages, (try shard(e)).g.latPage(), @intCast(h.a), h.seed, host, true);
}

fn slotTakeBody(e: *Engine, h: follow.Header, host: []u8) anyerror!void {
    return snapshots.slotBytes(e, e.snaps.getPtr(h.a) orelse return error.NoKeptState, @intCast(h.seed), host, true);
}

fn pageTakeBody(e: *Engine, h: follow.Header, host: []u8) anyerror!void {
    return poolPage(e, 0, @intCast(h.a), h.seed, host, true);
}

fn putBody(e: *Engine, h: follow.Header, host: []u8) anyerror!void {
    return copy(e, h.a, h.seed, host, false);
}

/// One of rank 0's page and slot writes on this rank's copy (the op's body, unchanged).
fn pagePutBody(e: *Engine, op: Op, h: follow.Header, host: []u8) anyerror!void {
    switch (op) {
        .key_put => try shardBytes(e, (try shard(e)).keys orelse return error.NoPages, (try shard(e)).g.keyPage(), @intCast(h.a), h.seed, host, false),
        .lat_put => try shardBytes(e, (try shard(e)).lat orelse return error.NoPages, (try shard(e)).g.latPage(), @intCast(h.a), h.seed, host, false),
        .slot_put => try snapshots.slotBytes(e, e.snaps.getPtr(h.a) orelse return error.NoKeptState, @intCast(h.seed), host, false),
        .page_put => try poolPage(e, 0, @intCast(h.a), h.seed, host, false),
        else => unreachable,
    }
}

/// The seal's words decoded and sealed on this rank's copy of `id` (the op's body, unchanged).
fn sealBody(e: *Engine, id: u64, ids: []const u32) !void {
    var x: [image.Words.count]u64 = undefined;
    @memcpy(std.mem.sliceAsBytes(&x), std.mem.sliceAsBytes(ids));
    const w = try image.Words.decode(x, @intCast((e.snaps.get(id) orelse return error.NoKeptState).pos));
    try e.sealState(id, w.mtp_len, .{ w.base, w.mtp_base }, w.cur);
}

/// A rank other than 0: rank 0's disk step on this rank's copy of its state. An op that fails here poisons its id
/// (poison) and the step loop runs on; only a step that breaks the protocol itself ends this rank.
pub fn step(e: *Engine, l: *follow.Link, h: follow.Header, ids: []const u32) !void {
    if (h.flags < @backingInt(Op.blank) or h.flags > @backingInt(Op.page_take)) return error.BadStep;
    const op: Op = @fromBackingInt(h.flags);
    const rank: u32 = @intCast(e.share.rank);
    const write_op = switch (op) {
        .blank, .put, .key_put, .lat_put, .slot_put, .page_put => true,
        else => false,
    };
    // a page op's `a` is a pool page id, not a state's: its state is the one this rank last blanked (rank 0 recalls
    // one state at a time: blank, its pieces, seal)
    const state: u64 = switch (op) {
        .key_put, .lat_put, .page_put => e.disk_cur,
        else => h.a,
    };
    if (op == .blank) e.disk_cur = h.a;
    if (write_op and e.disk_failed.contains(state)) return; // a poisoned id's writes are skipped: its seal says so
    switch (op) {
        .blank => {
            if (h.depth > image.Words.kv_parts or (h.depth != 0 and !e.settings.kv_shard.on)) return error.BadStep;
            if (e.blankAs(h.a, @intCast(h.seed), if (h.depth == 0) null else @intCast(h.depth - 1), h.top_k)) |_| {} else |err| poison(e, h.a, err);
        },
        .seal => {
            if (ids.len != 2 * image.Words.count) return error.BadStep;
            var ok: u32 = if (e.disk_failed.contains(h.a)) 1 else 0; // a poisoned id's seal answers with its failure
            if (ok == 0) if (sealBody(e, h.a, ids)) |_| {} else |err| {
                poison(e, h.a, err);
                ok = 1;
            };
            // the seal's answer crosses the link, rank order: each rank's status word (0 its seal stood, 1 it failed)
            const bytes = std.mem.sliceAsBytes(l.ids);
            std.mem.writeInt(u32, bytes[0..status_bytes], ok, .little);
            var r: u32 = 1;
            while (r < e.share.world) : (r += 1) {
                if (r == rank) l.buf.uploadAsync(64, bytes[0..status_bytes], l.stream.handle) catch {};
                try l.from(r, status_bytes);
            }
        },
        .put => if (h.top_k == rank) {
            if (h.depth > ids.len * 4) return error.BadStep;
            if (putBody(e, h, @constCast(std.mem.sliceAsBytes(ids))[0..h.depth])) |_| {} else |err| poison(e, h.a, err);
        },
        .take => try answered(e, l, h, rank, h.a, takeBody),
        // rank 0's pool-page and slot writes and reads (a pages layout's rows and fixed parts, beside the shard its
        // keys and its home's latent rows): the bytes ride the ids there and the link's buffer back
        .key_put, .lat_put, .slot_put, .page_put => if (h.top_k == rank) {
            if (h.depth > ids.len * 4) return error.BadStep;
            if (pagePutBody(e, op, h, @constCast(std.mem.sliceAsBytes(ids))[0..h.depth])) |_| {} else |err| poison(e, state, err);
        },
        .lat_take => try answered(e, l, h, rank, null, latTakeBody),
        .slot_take => try answered(e, l, h, rank, h.a, slotTakeBody),
        .page_take => try answered(e, l, h, rank, null, pageTakeBody),
        .prefixes => {
            var out: [64]u64 = undefined;
            if (e.share.world > out.len) return error.BadStep;
            try gather(l, e, (try shape(e)).prefix, out[0..e.share.world]);
        },
    }
}
