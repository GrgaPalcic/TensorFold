//! GLM-5.3-Flash's CUDA kept states on disk (--learn-conversations): every rank's part in rank 0's files. Under
//! TF_GLM_KV_SHARD a state's latent rows come from and go to its home rank alone (cuda_disk_ranks).
const std = @import("std");
const cuda = @import("cuda");
const api = @import("engine_api");
const lanes = @import("lanes");
const glm = @import("glm_cuda");
const cuda_disk = @import("cuda_disk.zig");
const img = glm.disk_image;
const ranks = glm.disk_ranks;
const rows = api.prompt_cache.rows;
const Engine = glm.engine.Engine;
const Kept = glm.engine.Kept;
const Allocator = std.mem.Allocator;

pub const image: rows.Image = .{ .parts = parts, .gather = gather, .blank = blank, .scatter = scatter, .whole = whole };

/// Every rank's prefix bytes, read at attach (one GLM engine a process; rank 0 alone serves).
var prefix_buf: [8]u64 = undefined;
var prefixes: []const u64 = &.{};

fn engineOf(ptr: *anyopaque) *Engine {
    return @ptrCast(@alignCast(ptr));
}

fn parts(ptr: *anyopaque, at: u32, out: *std.ArrayList(rows.image.Part), a: Allocator) anyerror!void {
    var list: std.ArrayList(img.Part) = .empty;
    defer list.deinit(a);
    try img.storeParts(try ranks.shape(engineOf(ptr)), prefixes, at, &list, a);
    for (list.items) |p| try out.append(a, .{ .bytes = p.bytes, .row = p.row, .per = p.per });
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

fn gather(ptr: *anyopaque, saved: api.prompt_cache.Saved, ranges: []const rows.image.Range, out: []u8) anyerror!void {
    const e = engineOf(ptr);
    const k: *Kept = @ptrCast(@alignCast(saved));
    const rest = Rest.of(e);
    defer rest.done();
    try ranks.gatherWith(RankIo{ .e = e, .id = k.id }, try ranks.shape(e), prefixes, k.at, try homeOf(e, k.id), ranges, out, e.gpa);
}

/// A new id leased at `at` on every rank, filled by `scatter` and sealed by the words in its last one.
fn blank(ptr: *anyopaque, at: u32) anyerror!api.prompt_cache.Saved {
    const e = engineOf(ptr);
    const rest = Rest.of(e);
    defer rest.done();
    const k = try e.gpa.create(Kept);
    errdefer e.gpa.destroy(k);
    k.* = .{ .id = try ranks.blank(e, at), .at = at };
    return @ptrCast(k);
}

/// The last scatter brings the state's own file, words first: every rank seals it then (the words place the head).
fn scatter(ptr: *anyopaque, saved: api.prompt_cache.Saved, ranges: []const rows.image.Range, in: []const u8) anyerror!void {
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
    // the blank's home (cuda_disk_ranks.readHome), not the words': a sharded state's latent rows to that rank alone
    try ranks.scatterWith(RankIo{ .e = e, .id = k.id }, try ranks.shape(e), prefixes, k.at, try homeOf(e, k.id), ranges, in, e.gpa);
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
    try gather(e, saved, &.{.{ .off = 0, .len = total }}, bytes);
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

/// --learn-conversations: every rank's prefix read, the identity probed on this thread, the store attached.
pub fn attach(a: Allocator, io: std.Io, ctx: *const cuda.Context, o: api.Open, root: []const u8, kernels: ?[]const u8, loaded: anytype, store: *api.prompt_cache.Store) !void {
    const e: *Engine = loaded.target;
    if (e.share.world > prefix_buf.len) return error.TooManyRanks;
    try ranks.prefixes(e, prefix_buf[0..e.share.world]);
    prefixes = prefix_buf[0..e.share.world];
    const sh = try ranks.shape(e);
    var facts: std.ArrayList(u64) = .empty;
    try facts.appendSlice(a, &.{ e.share.world, sh.capacity, sh.ring_rows, sh.tail, @intFromBool(sh.head), @intFromBool(sh.scales), @intFromBool(e.drafter != null) });
    try facts.appendSlice(a, prefixes);
    const p = try probe(a, io, loaded.backend, e);
    const id = try cuda_disk.identityOf(a, io, ctx, o.dir, kernels, facts.items, p.hash);
    const gibs = struct {
        fn bytes(g: f64) u64 {
            return @intFromFloat(g * (1 << 30));
        }
    }.bytes;
    // spill stays off: an eviction would gather on rank 0 mid-pass while the other ranks wait in the gate's all-reduce
    const r = try rows.Rows.open(store.gpa, store.family, .{ .root = root, .identity = id, .cap = gibs(o.learn_gib), .floor = gibs(o.learn_min_free_gib), .daily = gibs(o.learn_daily_gib), .lookahead = store.rules.lookahead, .planned = store.rules.planned, .grid = store.rules.grid });
    r.prefill_rate = p.rate;
    store.disk = r;
    std.log.info("glm: conversation states on disk in {s} ({d} states, {d} MiB of {d} MiB; {d} ranks' parts)", .{ r.dir, r.states.items.len, r.held() >> 20, r.cap >> 20, prefixes.len });
}

test {
    std.testing.refAllDecls(@This());
}
