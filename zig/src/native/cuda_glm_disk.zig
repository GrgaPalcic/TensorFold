//! GLM-5.3-Flash's CUDA kept states on disk (--learn-conversations): every rank's part in rank 0's files. Under
//! TF_GLM_KV_SHARD a state's latent rows come from and go to its home rank alone (cuda_disk_ranks).
const std = @import("std");
const cuda = @import("cuda");
const api = @import("engine_api");
const lanes = @import("lanes");
const glm = @import("glm_cuda");
const cuda_disk = @import("cuda_disk.zig");
const prompt_pages = @import("cuda_prompt_pages.zig");
const img = glm.disk_image;
const ranks = glm.disk_ranks;
const rows = api.prompt_cache.rows;
const Engine = glm.engine.Engine;
const Kept = glm.engine.Kept;
const Allocator = std.mem.Allocator;

const t = std.testing;

pub const image: rows.Image = .{ .parts = parts, .gather = gather, .blank = blank, .scatter = scatter, .whole = whole, .blank_pages = blankPages };

/// Every rank's prefix bytes, read at attach (one GLM engine a process; rank 0 alone serves).
var prefix_buf: [8]u64 = undefined;
var prefixes: []const u64 = &.{};

/// The store's layout: its states' rows in the pool's pages (TF_KV_SHARE's --learn-conversations), set once at the
/// share attach; without it every state is a copy, as 0109's store has always written. `pages_sharded`
/// (TF_GLM_KV_SHARD): the pool's two parts, a state's latent rows on its home part's rank alone.
var pages_mode: bool = false;
var pages_sharded: bool = false;

/// Whether the store keeps the pages layout (a test's and the image's read of the mode).
pub fn sharedLayout() bool {
    return pages_mode;
}

fn engineOf(ptr: *anyopaque) *Engine {
    return @ptrCast(@alignCast(ptr));
}

fn parts(ptr: *anyopaque, at: u32, out: *std.ArrayList(rows.image.Part), a: Allocator) anyerror!void {
    var list: std.ArrayList(img.Part) = .empty;
    defer list.deinit(a);
    const sh = try ranks.shape(engineOf(ptr));
    if (pages_mode) try img.pageParts(sh, at, &list, a, pages_sharded) else try img.parts(sh, at, &list, a);
    for (list.items) |p| try out.append(a, .{ .bytes = p.bytes, .row = p.row, .per = p.per });
    for (prefixes[1..]) |p| try out.append(a, .{ .bytes = p });
}

/// The other ranks asleep before a disk call go back to sleep after it (their lines, not NCCL, wait meanwhile).
const Rest = struct {
    e: *Engine,
    asleep: bool,

    fn of(e: *Engine) Rest {
        return .{ .e = e, .asleep = ranks.asleep(e) };
    }

    fn done(r: Rest) void {
        if (r.asleep) ranks.rest(r.e) catch |err| std.log.warn("glm: the ranks did not sleep after a disk step ({s})", .{@errorName(err)});
    }
};

/// Each rank's image of one state, through rank 0's disk steps (cuda_disk_ranks.gatherWith, scatterWith).
const RankIo = struct {
    e: *Engine,
    id: u64,

    pub fn take(io: RankIo, rank: u32, off: u64, out: []u8) !void {
        return ranks.take(io.e, rank, io.id, off, out);
    }

    pub fn put(io: RankIo, rank: u32, off: u64, bytes: []const u8) !void {
        return ranks.put(io.e, rank, io.id, off, bytes);
    }
};

/// The home part of rank 0's copy of a kept state (TF_GLM_KV_SHARD; null: unsharded).
fn homeOf(e: *Engine, id: u64) !?u8 {
    return ranks.homeOf(e, e.snaps.get(id) orelse return error.NoKeptState);
}

fn gather(ptr: *anyopaque, saved: api.prompt_cache.Saved, pages: []const u32, ranges: []const rows.image.Range, out: []u8) anyerror!void {
    const e = engineOf(ptr);
    const k: *Kept = @ptrCast(@alignCast(saved));
    const rest = Rest.of(e);
    defer rest.done();
    if (pages_mode) return if (pages_sharded) gatherPagesSharded(e, k, pages, ranges, out) else gatherPages(e, k, pages, ranges, out);
    try ranks.gatherWith(RankIo{ .e = e, .id = k.id }, try ranks.shape(e), prefixes, k.at, try homeOf(e, k.id), ranges, out, e.gpa);
}

/// A pages-layout state's image bytes over `ranges` into `out`: the words from the engine's fields, the rows from the
/// pool's pages (a piece a page-run, as snapshot_pages.moveParts copies), the fixed parts from the slot's lease in the
/// image's fixed order, and each other rank's prefix through the link (rank 0's files, as every store's).
fn gatherPages(e: *Engine, k: *Kept, pages: []const u32, ranges: []const rows.image.Range, out: []u8) anyerror!void {
    if (pages.len == 0) return error.NoPages;
    const s = e.snaps.get(k.id) orelse return error.NoKeptState;
    const sh = try ranks.shapeOf(e, s);
    var list: std.ArrayList(img.Part) = .empty;
    defer list.deinit(e.gpa);
    try img.pageParts(sh, k.at, &list, e.gpa, false);
    const p = e.pages orelse return error.NoPages;
    const page = p.layout.pageBytes();
    const arena: u64 = if (p.arena) |a| a.ptr else return error.NoPages;
    var words = (try ranks.wordsOf(e, k.id)).encode();
    var own: u64 = 0; // rank 0's image bytes: the words, the rows and the fixed parts; the rank prefixes follow
    for (list.items) |part| own += part.bytes;
    var o: usize = 0;
    for (ranges) |g| {
        const end = g.off + g.len;
        var base: u64 = 0; // the part's first image byte
        var fixed: u64 = 0; // the fixed-part image bytes before this one, past the words
        for (list.items, 0..) |part, pi| {
            defer {
                base += part.bytes;
                if (pi > 0 and part.row == 0) fixed += part.bytes;
            }
            const lo = @max(g.off, base);
            const hi = @min(end, base + part.bytes);
            if (lo >= hi) continue;
            const host = out[o + @as(usize, @intCast(lo - g.off)) ..][0..@intCast(hi - lo)];
            const within = lo - base;
            if (pi == 0) {
                @memcpy(host, std.mem.sliceAsBytes(&words)[@intCast(within)..][0..host.len]);
            } else if (part.row > 0) {
                try pageRows(e, part, sh, pages, within, host, arena, page);
            } else if (img.fixedLease(sh, fixed + within)) |at| {
                try ranks.slot(e, 0, k.id, at, host, true);
            } else @memset(host, 0); // the dead half's place: a live-half keep's lease holds no bytes there
        }
        try rankPrefixes(e, k.id, g, end, own, o, out, true);
        o += @intCast(g.len);
    }
}

/// A sharded pages-layout state's image bytes over `ranges` into `out`: the words from the engine's fields, the
/// pooled keys from rank 0's key pages, the latent codes and scales from the home rank's latent pages (through the
/// link when the home is rank 1), the fixed parts from rank 0's slot, and each other rank's prefix through the link:
/// the gather walks the ranks a piece's bytes come from, all written through rank 0's files.
fn gatherPagesSharded(e: *Engine, k: *Kept, pages: []const u32, ranges: []const rows.image.Range, out: []u8) anyerror!void {
    if (pages.len == 0) return error.NoPages;
    const s = e.snaps.get(k.id) orelse return error.NoKeptState;
    const sh = try ranks.shapeOf(e, s);
    var list: std.ArrayList(img.Part) = .empty;
    defer list.deinit(e.gpa);
    try img.pageParts(sh, k.at, &list, e.gpa, true);
    const home = s.home orelse return error.SnapshotShape; // a sharded state's latent rows are its home part's
    var words = (try ranks.wordsOf(e, k.id)).encode();
    var own: u64 = 0; // rank 0's image bytes: the words, the rows and the fixed parts; the rank prefixes follow
    for (list.items) |part| own += part.bytes;
    var o: usize = 0;
    for (ranges) |g| {
        const end = g.off + g.len;
        var base: u64 = 0; // the part's first image byte
        var fixed: u64 = 0; // the fixed-part image bytes before this one, past the words
        for (list.items, 0..) |part, pi| {
            defer {
                base += part.bytes;
                if (pi > 0 and part.row == 0) fixed += part.bytes;
            }
            const lo = @max(g.off, base);
            const hi = @min(end, base + part.bytes);
            if (lo >= hi) continue;
            const host = out[o + @as(usize, @intCast(lo - g.off)) ..][0..@intCast(hi - lo)];
            const within = lo - base;
            if (pi == 0) {
                @memcpy(host, std.mem.sliceAsBytes(&words)[@intCast(within)..][0..host.len]);
            } else if (part.row > 0) {
                try ranks.pageRows(e, part, sh, home, pages, within, host, true);
            } else if (img.fixedLease(sh, fixed + within)) |at| {
                try ranks.slot(e, 0, k.id, at, host, true);
            } else @memset(host, 0); // the dead half's place: a live-half keep's lease holds no bytes there
        }
        try rankPrefixes(e, k.id, g, end, own, o, out, true);
        o += @intCast(g.len);
    }
}

/// Each other rank's prefix bytes of the store image's range [g.off, end) (`own` rank 0's bytes before them), read
/// from (`out` true) or written to its slot's lease.
fn rankPrefixes(e: *Engine, id: u64, g: rows.image.Range, end: u64, own: u64, o: usize, buf: []const u8, out: bool) !void {
    var pre = own; // each other rank's prefix through the link, as the flat store's
    var r: u32 = 1;
    while (r < prefixes.len) : (r += 1) {
        const lo = @max(g.off, pre);
        const hi = @min(end, pre + prefixes[r]);
        if (lo < hi) {
            const host = buf[o + @as(usize, @intCast(lo - g.off)) ..][0..@intCast(hi - lo)];
            if (out) try ranks.take(e, r, id, img.Words.bytes + (lo - pre), @constCast(host)) else try ranks.put(e, r, id, img.Words.bytes + (lo - pre), host);
        }
        pre += prefixes[r];
    }
    if (end > pre) return error.SnapshotShape;
}

/// A row part's bytes [within, within + host.len) read from rank 0's pool pages (unsharded: its arena's whole pages):
/// byte b in page ids[b / span] at the part's place in its page plus b % span (`pageSpan`), a `cuMemcpyDtoH` a page-run.
fn pageRows(e: *Engine, part: img.Part, sh: img.Shape, pages: []const u32, within: u64, host: []u8, arena: u64, page: u64) !void {
    const span = img.pageSpan(sh, part);
    var done: usize = 0;
    while (done < host.len) {
        const b = within + done;
        const k = b / span;
        if (k >= pages.len) return error.SnapshotShape;
        const n: usize = @intCast(@min(host.len - done, span - b % span));
        try e.d.check(e.d.api.cuMemcpyDtoH_v2(host[done..].ptr, arena + @as(u64, pages[k]) * page + part.at + b % span, n), "cuMemcpyDtoH");
        done += n;
    }
}

/// A new id leased at `at` on every rank, filled by `scatter` and sealed by the words in its last one.
fn blank(ptr: *anyopaque, at: u32) anyerror!api.prompt_cache.Saved {
    const e = engineOf(ptr);
    const rest = Rest.of(e);
    defer rest.done();
    const k = try e.gpa.create(Kept);
    errdefer e.gpa.destroy(k);
    k.* = .{ .id = try ranks.blank(e, at, ranks.readHome(e), 0), .at = at };
    return @ptrCast(k);
}

/// A new kept state of `at` tokens whose leading rows go into pool pages the store holds (`out` their ids, each held
/// for the caller's tree), its slot past them: a recall's blank under TF_KV_SHARE. The pages lie in one part under
/// TF_GLM_KV_SHARD (the part with room, as a fresh stream's `Parts.choose`), a recall's home part — the blank's home,
/// whose rank alone holds its latent rows on the write and the read.
fn blankPages(ptr: *anyopaque, at: u32, out: *std.ArrayList(u32)) anyerror!api.prompt_cache.Saved {
    const e = engineOf(ptr);
    const rest = Rest.of(e);
    defer rest.done();
    const p = e.pages orelse return error.NoPages;
    const n = at / glm.state.page_rows;
    if (n == 0) return error.SnapshotShape;
    const ids = try e.gpa.alloc(u32, n);
    errdefer e.gpa.free(ids);
    const home: ?u8 = if (p.shard) |x| x.parts.choose(n, null) else null; // the recall's home part
    if (!holdPages(p, home, ids)) return error.LaneMemory;
    errdefer givePages(p, ids);
    const id = try ranks.blank(e, at, home, @intCast(n));
    errdefer e.dropState(id);
    const k = try e.gpa.create(Kept);
    errdefer e.gpa.destroy(k);
    k.* = .{ .id = id, .at = at };
    try out.appendSlice(e.gpa, ids);
    e.gpa.free(ids);
    return @ptrCast(k);
}

/// `n` pool pages held for a recalled state in one part with room (its pools reclaiming first), global ids. `home`:
/// the part chosen for it (its pools reclaim first); null: the unsharded pool's.
fn holdPages(p: anytype, home: ?u8, out: []u32) bool {
    if (p.shard) |x| {
        const part = home orelse x.parts.choose(out.len, null) orelse return false;
        if (!x.parts.pools[part].hold(out)) return false;
        for (out) |*id| id.* += @as(u32, @intCast(part * x.parts.h));
        return true;
    }
    return p.pool.hold(out);
}

/// A recalled state's pages given back (its tree did not take them: a failed read or insert).
fn givePages(p: anytype, ids: []const u32) void {
    if (p.shard) |x| {
        for (ids) |id| x.parts.release(id);
        return;
    }
    for (ids) |id| p.pool.release(id);
}

/// The last scatter brings the state's own file, words first: every rank seals it then (the words place the head).
fn scatter(ptr: *anyopaque, saved: api.prompt_cache.Saved, pages: []const u32, ranges: []const rows.image.Range, in: []const u8) anyerror!void {
    const e = engineOf(ptr);
    const k: *Kept = @ptrCast(@alignCast(saved));
    const rest = Rest.of(e);
    defer rest.done();
    if (ranges.len > 0 and ranges[0].off == 0) {
        if (ranges[0].len < img.Words.bytes) return error.DiskWords;
        var x: [img.Words.count]u64 = undefined;
        @memcpy(std.mem.sliceAsBytes(&x), in[0..img.Words.bytes]);
        try ranks.seal(e, k.id, try img.Words.decode(x, k.at));
    }
    if (pages_mode) return if (pages_sharded) scatterPagesSharded(e, k, pages, ranges, in) else scatterPages(e, k, pages, ranges, in);
    // the blank's home (cuda_disk_ranks.readHome), not the words': a sharded state's latent rows to that rank alone
    try ranks.scatterWith(RankIo{ .e = e, .id = k.id }, try ranks.shape(e), prefixes, k.at, try homeOf(e, k.id), ranges, in, e.gpa);
}

/// A pages-layout state's image bytes from `in` into its pool pages and its slot's lease (the words went with the
/// seal), and each other rank's prefix through the link: the read-back of a recall into pool pages. Unsharded, every
/// rank's pool pages get the rows (each holds them) and its lease the fixed parts past its prefix.
fn scatterPages(e: *Engine, k: *Kept, pages: []const u32, ranges: []const rows.image.Range, in: []const u8) anyerror!void {
    if (pages.len == 0) return error.NoPages;
    const s = e.snaps.get(k.id) orelse return error.NoKeptState;
    const sh = try ranks.shapeOf(e, s);
    var list: std.ArrayList(img.Part) = .empty;
    defer list.deinit(e.gpa);
    try img.pageParts(sh, k.at, &list, e.gpa, false);
    var own: u64 = 0;
    for (list.items) |part| own += part.bytes;
    var o: usize = 0;
    for (ranges) |g| {
        const end = g.off + g.len;
        var base: u64 = 0;
        var fixed: u64 = 0; // the fixed-part image bytes before this one, past the words
        for (list.items, 0..) |part, pi| {
            defer {
                base += part.bytes;
                if (pi > 0 and part.row == 0) fixed += part.bytes;
            }
            const lo = @max(g.off, base);
            const hi = @min(end, base + part.bytes);
            if (lo >= hi) continue;
            const bytes = in[o + @as(usize, @intCast(lo - g.off)) ..][0..@intCast(hi - lo)];
            const within = lo - base;
            if (pi == 0) continue; // the words went with the seal
            if (part.row > 0) try ranks.poolRows(e, part, sh, pages, within, bytes) else if (img.fixedLease(sh, fixed + within)) |at| {
                try fixedParts(e, k.id, at, bytes);
            } else {} // the dead half's place: the file's zeros there have no bytes of the lease's
        }
        try rankPrefixes(e, k.id, g, end, own, o, in, false);
        o += @intCast(g.len);
    }
}

/// A sharded pages-layout state's image bytes from `in` into the pool's pages and the slots' leases (the words went
/// with the seal), and each other rank's prefix through the link: the pooled keys into every rank's key pages (each
/// holds them), the latent codes and scales into the home rank's latent pages alone, the fixed parts into rank 0's
/// lease and every rank's past its prefix.
fn scatterPagesSharded(e: *Engine, k: *Kept, pages: []const u32, ranges: []const rows.image.Range, in: []const u8) anyerror!void {
    if (pages.len == 0) return error.NoPages;
    const s = e.snaps.get(k.id) orelse return error.NoKeptState;
    const sh = try ranks.shapeOf(e, s);
    var list: std.ArrayList(img.Part) = .empty;
    defer list.deinit(e.gpa);
    try img.pageParts(sh, k.at, &list, e.gpa, true);
    const home = s.home orelse return error.SnapshotShape;
    var own: u64 = 0;
    for (list.items) |part| own += part.bytes;
    var o: usize = 0;
    for (ranges) |g| {
        const end = g.off + g.len;
        var base: u64 = 0;
        var fixed: u64 = 0; // the fixed-part image bytes before this one, past the words
        for (list.items, 0..) |part, pi| {
            defer {
                base += part.bytes;
                if (pi > 0 and part.row == 0) fixed += part.bytes;
            }
            const lo = @max(g.off, base);
            const hi = @min(end, base + part.bytes);
            if (lo >= hi) continue;
            const bytes = in[o + @as(usize, @intCast(lo - g.off)) ..][0..@intCast(hi - lo)];
            const within = lo - base;
            if (pi == 0) continue; // the words went with the seal
            if (part.row > 0) {
                try ranks.pageRows(e, part, sh, home, pages, within, @constCast(bytes), false);
            } else if (img.fixedLease(sh, fixed + within)) |at| {
                try fixedParts(e, k.id, at, bytes);
            } else {} // the dead half's place: the file's zeros there have no bytes of the lease's
        }
        try rankPrefixes(e, k.id, g, end, own, o, in, false);
        o += @intCast(g.len);
    }
}

/// A fixed part's bytes at slot-lease offset `at` written into rank 0's lease, and past rank 0's prefix into every
/// other rank's (the rings, the pools past the mark and the tail are every rank's own; each rank's prefix came with
/// the image's prefix section).
fn fixedParts(e: *Engine, id: u64, at: u64, bytes: []const u8) !void {
    try ranks.slot(e, 0, id, at, @constCast(bytes), false);
    if (at < prefixes[0]) return; // rank 0's prefix alone: the others' came with their own section
    for (prefixes[1..], 1..) |_, r| try ranks.slot(e, @intCast(r), id, at, @constCast(bytes), false);
}

fn whole(ptr: *anyopaque, saved: api.prompt_cache.Saved) bool {
    const k: *Kept = @ptrCast(@alignCast(saved));
    const s = engineOf(ptr).snaps.get(k.id) orelse return false;
    return s.len != 0;
}

/// A fixed prompt's store image hashed, and each other rank's shared bytes it holds checked equal to the image's.
fn probe(a: Allocator, io: std.Io, backend: lanes.backend.Backend, e: *Engine) !struct { hash: u64, rate: f64 } {
    const mark: u32 = glm.native.prompt_rows;
    var toks: [glm.native.prompt_rows + 40]u32 = undefined;
    for (&toks, 0..) |*x, i| x.* = @intCast(1000 + i * 7919 % 50000);
    const Hook = struct {
        e: *Engine,
        saved: ?*anyopaque = null,
        failed: ?anyerror = null,
        fn at(p: *anyopaque, _: *lanes.Stream, m: u32) void {
            const h: *@This() = @ptrCast(@alignCast(p));
            h.saved = glm.native.snap_save(h.e, null, m) catch |err| {
                h.failed = err;
                return;
            };
        }
    };
    var hook: Hook = .{ .e = e };
    const marks = [_]u32{mark};
    var s = try lanes.Stream.init(a, .{ .id = "probe", .prompt = &toks, .max_new = 1, .drafts = e.mtp, .reuse = .{ .marks = &marks, .hook = .{ .ptr = &hook, .at = Hook.at } } });
    defer s.deinit(a);
    const t0 = std.Io.Clock.awake.now(io).toNanoseconds();
    backend.vtable.prefill(backend.ptr, &s) catch |err| {
        backend.vtable.release(backend.ptr, &s);
        return err;
    };
    const ns = std.Io.Clock.awake.now(io).toNanoseconds() - t0;
    backend.vtable.release(backend.ptr, &s);
    const saved = hook.saved orelse return hook.failed orelse error.ProbeUnmarked;
    defer glm.native.snap_drop(e, saved);
    const k: *Kept = @ptrCast(@alignCast(saved));
    const sh = try ranks.shape(e);
    const own = img.imageBytes(sh, mark);
    var total = own;
    for (prefixes[1..]) |p| total += p;
    const bytes = try a.alloc(u8, @intCast(total));
    try gather(e, saved, &.{}, &.{.{ .off = 0, .len = total }}, bytes); // the probe is a copied state: no pages
    const shared = bytes[@intCast(img.Words.bytes + sh.prefix)..@intCast(own)];
    const theirs = try a.alloc(u8, shared.len);
    // each rank's copy of what it holds (under TF_GLM_KV_SHARD the latent rows on the probe's home alone)
    const home = try homeOf(e, k.id);
    var held: std.ArrayList([2]u64) = .empty;
    defer held.deinit(a);
    for (prefixes[1..], 1..) |q, r| {
        held.clearRetainingCapacity();
        try img.heldShared(sh, mark, if (home) |h| h == r else true, &held, a);
        for (held.items) |x| {
            const part = theirs[@intCast(x[0])..][0..@intCast(x[1])];
            try ranks.take(e, @intCast(r), k.id, img.Words.bytes + q + x[0], part);
            if (!std.mem.eql(u8, shared[@intCast(x[0])..][0..part.len], part)) return error.RanksDiffer;
        }
    }
    try ranks.rest(e);
    return .{ .hash = std.hash.Wyhash.hash(0x676c6d, bytes), .rate = @as(f64, @floatFromInt(toks.len)) / (@as(f64, @floatFromInt(@max(ns, 1))) / 1e9) };
}

/// The store identity's facts: the share's world and shape, the drafter, the caller's layout word (`extra`), and
/// each rank's prefix — with TF_GLM_KV_LIVE_HALF's own beside them (its store is another directory; with the
/// switch off the facts, and the identity, are as they have always been).
fn identityFacts(a: Allocator, e: anytype, sh: img.Shape, extra: []const u64) !std.ArrayList(u64) {
    var facts: std.ArrayList(u64) = .empty;
    errdefer facts.deinit(a);
    try facts.appendSlice(a, &.{ e.share.world, sh.capacity, sh.ring_rows, sh.tail, @intFromBool(sh.head), @intFromBool(sh.scales), @intFromBool(e.drafter != null) });
    try facts.appendSlice(a, extra);
    if (glm.snapshots.layout.liveHalfEnabled()) try facts.append(a, 1);
    try facts.appendSlice(a, prefixes);
    return facts;
}

/// --learn-conversations: every rank's prefix read, the identity probed on this thread, the store attached.
pub fn attach(a: Allocator, io: std.Io, ctx: *const cuda.Context, o: api.Open, root: []const u8, kernels: ?[]const u8, loaded: anytype, store: *api.prompt_cache.Store) !void {
    const e: *Engine = loaded.target;
    if (e.share.world > prefix_buf.len) return error.TooManyRanks;
    try ranks.prefixes(e, prefix_buf[0..e.share.world]);
    prefixes = prefix_buf[0..e.share.world];
    const sh = try ranks.shape(e);
    var facts = try identityFacts(a, e, sh, &.{});
    defer facts.deinit(a);
    const p = try probe(a, io, loaded.backend, e);
    const id = try cuda_disk.identityOf(a, io, ctx, o.dir, kernels, facts.items, p.hash);
    const gibs = struct {
        fn bytes(g: f64) u64 {
            return @intFromFloat(g * (1 << 30));
        }
    }.bytes;
    // spill stays off: an eviction would gather on rank 0 mid-pass while the other ranks wait in the gate's all-reduce
    const raw = std.c.getenv("TF_GLM_LEARN_STAGING_GIB");
    const named = raw != null and std.mem.trim(u8, std.mem.span(raw.?), " \t").len > 0;
    const staging_bytes = try stagingOf(if (raw) |v| std.mem.span(v) else null);
    const r = try rows.Rows.open(store.gpa, store.family, .{ .root = root, .identity = id, .cap = gibs(o.learn_gib), .floor = gibs(o.learn_min_free_gib), .daily = gibs(o.learn_daily_gib), .lookahead = store.rules.lookahead, .planned = store.rules.planned, .grid = store.rules.grid, .staging_bytes = staging_bytes });
    r.prefill_rate = p.rate;
    store.disk = r;
    var permit_buf: [48]u8 = undefined;
    const permit = if (named) try std.fmt.bufPrint(&permit_buf, "; staging {d} GiB", .{staging_bytes >> 30}) else "";
    std.log.info("glm: conversation states on disk in {s} ({d} states, {d} MiB of {d} MiB; {d} ranks' parts{s})", .{ r.dir, r.states.items.len, r.held() >> 20, r.cap >> 20, prefixes.len, permit });
}

/// The store's staging permit when TF_GLM_LEARN_STAGING_GIB names none: rows_store.Options' 1 GiB default.
const staging_default: u64 = (rows.Options{ .root = "", .identity = 0, .cap = 0, .floor = 0, .daily = 0 }).staging_bytes;

/// TF_GLM_LEARN_STAGING_GIB: GiB the store stages at once (unset or empty: the store's 1 GiB; a number: that many GiB, whole pages; anything else: refused).
fn stagingOf(text: ?[]const u8) error{BadStagingGib}!u64 {
    const s = std.mem.trim(u8, text orelse "", " \t");
    if (s.len == 0) return staging_default;
    const g = std.fmt.parseFloat(f64, s) catch return error.BadStagingGib;
    if (!std.math.isFinite(g) or g < 0 or g > 4096) return error.BadStagingGib;
    return std.mem.alignBackward(u64, @intFromFloat(g * (1 << 30)), rows.file.alignment);
}

/// --learn-conversations with TF_KV_SHARE: the shared store's states on disk, their rows in the pool's pages and
/// their slots' parts in rank 0's files (the pages layout: its own identity, so the two stores' sets never mix).
pub fn attachShare(a: Allocator, io: std.Io, ctx: *const cuda.Context, o: api.Open, root: []const u8, kernels: ?[]const u8, loaded: anytype, x: *prompt_pages.Adapter) !void {
    const e: *Engine = loaded.target;
    const sharded = e.settings.kv_shard.on; // TF_GLM_KV_SHARD: the pool's two parts, a state's latent rows its home part's
    if (e.share.world > prefix_buf.len) return error.TooManyRanks;
    try ranks.prefixes(e, prefix_buf[0..e.share.world]);
    prefixes = prefix_buf[0..e.share.world];
    const sh = try ranks.shape(e);
    // 1: the pages layout, 2 beside the shard — the copy layout's store, the pages one and the sharded pages one never mix
    var facts = try identityFacts(a, e, sh, &.{1 + @intFromBool(sharded)});
    defer facts.deinit(a);
    const was = e.share_pages;
    const held = if (e.pages) |pg| pg.prompts else null;
    // the probe keeps a whole copy through its own mark hook, as without the share: the store stands aside for it
    e.share_pages = false;
    if (e.pages) |pg| pg.prompts = null;
    if (e.link) |l| try (glm.follow.Leader{ .e = e, .link = @ptrCast(@alignCast(l)) }).sharePages(false); // the other ranks keep it whole too
    const p = probe(a, io, loaded.backend, e) catch |err| {
        e.share_pages = was;
        if (e.pages) |pg| pg.prompts = held;
        return err;
    };
    e.share_pages = was;
    if (e.pages) |pg| pg.prompts = held;
    if (e.link) |l| try (glm.follow.Leader{ .e = e, .link = @ptrCast(@alignCast(l)) }).sharePages(was);
    pages_mode = true;
    pages_sharded = sharded;
    const id = try cuda_disk.identityOf(a, io, ctx, o.dir, kernels, facts.items, p.hash);
    const gibs = struct {
        fn bytes(g: f64) u64 {
            return @intFromFloat(g * (1 << 30));
        }
    }.bytes;
    const raw = std.c.getenv("TF_GLM_LEARN_STAGING_GIB");
    const named = raw != null and std.mem.trim(u8, std.mem.span(raw.?), " \t").len > 0;
    const staging_bytes = try stagingOf(if (raw) |v| std.mem.span(v) else null);
    const r = try rows.Rows.open(x.gpa, x.store.tree.family, .{ .root = root, .identity = id, .cap = gibs(o.learn_gib), .floor = gibs(o.learn_min_free_gib), .daily = gibs(o.learn_daily_gib), .lookahead = x.store.rules.lookahead, .planned = x.store.rules.planned, .grid = x.store.rules.grid, .staging_bytes = staging_bytes });
    r.prefill_rate = p.rate;
    x.store.attachDisk(r);
    x.disk = r;
    var permit_buf: [48]u8 = undefined;
    const permit = if (named) try std.fmt.bufPrint(&permit_buf, "; staging {d} GiB", .{staging_bytes >> 30}) else "";
    std.log.info("glm: conversation states on disk in {s} ({d} states, {d} MiB of {d} MiB; {d} ranks' parts, pages and parts{s})", .{ r.dir, r.states.items.len, r.held() >> 20, r.cap >> 20, prefixes.len, permit });
}

test {
    std.testing.refAllDecls(@This());
}

test "the store identity's facts: TF_GLM_KV_LIVE_HALF's own beside them; off, the facts as before" {
    const layout = glm.snapshots.layout;
    const fake = .{ .share = .{ .world = 2 }, .drafter = @as(?*u8, null) };
    const sh = std.mem.zeroes(img.Shape);
    defer layout.test_switch = null;
    layout.test_switch = false;
    const off = try identityFacts(t.allocator, &fake, sh, &.{});
    defer off.deinit(t.allocator);
    const again = try identityFacts(t.allocator, &fake, sh, &.{});
    defer again.deinit(t.allocator);
    try t.expectEqualSlices(u64, again.items, off.items); // the switch off (or unset): the identity as before
    layout.test_switch = true;
    const on = try identityFacts(t.allocator, &fake, sh, &.{1}); // attachShare's layout word beside the rest
    defer on.deinit(t.allocator);
    try t.expectEqual(off.items.len + 1, on.items.len);
    try t.expectEqualSlices(u64, off.items[0..7], on.items[0..7]); // the share's facts first, as they have been
    try t.expectEqual(@as(u64, 1), on.items[7]); // the pages layout's word, then the live half's
    try t.expectEqual(@as(u64, 1), on.items[8]); // its own: another directory, the sets never mix
    try t.expectEqual(@as(u64, 1), on.items[on.items.len - 1]); // rank 1's prefix last, as before
}

test "TF_GLM_LEARN_STAGING_GIB: unset keeps the store's 1 GiB, a number gives whole pages of GiB, anything else is refused" {
    try t.expectEqual(staging_default, try stagingOf(null));
    try t.expectEqual(staging_default, try stagingOf(""));
    try t.expectEqual(staging_default, try stagingOf("  "));
    try t.expectEqual(@as(u64, 2 << 30), try stagingOf("2"));
    try t.expectEqual(@as(u64, 12 << 30), try stagingOf("12"));
    try t.expectEqual(@as(u64, 4 << 30), try stagingOf("4.0001")); // down to whole pages
    try t.expectEqual(@as(u64, 0), try stagingOf("0"));
    try t.expectEqual(@as(u64, 0), try stagingOf("0.000001"));
    for ([_][]const u8{ "x", "-1", "nan", "inf", "5000" }) |bad| try t.expectError(error.BadStagingGib, stagingOf(bad));
}
