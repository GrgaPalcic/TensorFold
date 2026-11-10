//! GLM-5.3-Flash's KV sharding (TF_GLM_KV_SHARD, GLM patch 0118 by Robert Wojciechowski, GitHub @wojo, X @rwojo) on
//! two ranks: a sharded paged stream against its twin on a whole window (unsharded), byte for byte, through every path
//! the sharded stream can take: a prompt with a twin (both ranks hold its rows) and without (owner-computed), each
//! part as its owner, decode windows of 1 to 16 rows after the twin goes back (owner-computed), a kept state resumed
//! in its home part and outside it (the carry), with and without a twin, and the counters and layout after. A kept
//! state also goes through its --learn-conversations disk image (each rank's held bytes over the disk steps, read back
//! into a home part rank 0 picks) and resumes from the copy read back, at home and carried.
const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");
const lanes_tests = @import("glm_lanes_tests.zig");
const Gpu = check.Gpu;

/// A window of 16,384 and a verify window, as cuda_native.capacity makes it: room for four prompt chunks.
const capacity = 16448;
const sharded_slot = 0; // the engine's own sequence, on pages
const other_slot = 1; // a second paged stream: the parts' placement sees a live stream beside it
const twin_slot = 4; // past the paged sequences: a whole window, unsharded
const streams = 3;

/// The paths a prompt pass takes: its part (the owner), whether it may borrow a twin, and a resume.
const Case = struct { name: []const u8, len: usize, home: u8, twin: bool, resume_part: ?u8 = null, resume_twin: bool = true, disk: bool = false };
const cases = [_]Case{
    .{ .name = "part 0, a twin", .len = 3000, .home = 0, .twin = true },
    .{ .name = "part 1, a twin", .len = 3000, .home = 1, .twin = true },
    .{ .name = "part 0, owner-computed", .len = 2049, .home = 0, .twin = false },
    .{ .name = "part 1, owner-computed past the dense limit", .len = 9000, .home = 1, .twin = false },
    .{ .name = "kept in part 0, resumed at home with a twin", .len = 9000, .home = 0, .twin = true, .resume_part = 0 },
    .{ .name = "kept in part 0, resumed in part 1 (carry)", .len = 9000, .home = 0, .twin = false, .resume_part = 1, .resume_twin = false },
    .{ .name = "kept in part 1, resumed in part 0 with a twin (carry to both)", .len = 9000, .home = 1, .twin = true, .resume_part = 0 },
    // through the disk image: the read's home is the part a new stream takes then (the kept stream's, the other busy)
    .{ .name = "kept in part 1, through its disk image, resumed at the read's home", .len = 9000, .home = 1, .twin = false, .resume_part = 1, .resume_twin = false, .disk = true },
    .{ .name = "kept in part 1, through its disk image, resumed in part 0 (carry)", .len = 9000, .home = 1, .twin = false, .resume_part = 0, .resume_twin = false, .disk = true },
    .{ .name = "kept in part 0, through its disk image, resumed in part 1 (carry)", .len = 9000, .home = 0, .twin = false, .resume_part = 1, .resume_twin = false, .disk = true },
};

/// Rank 0: kept state `id`'s store image gathered over the disk steps, the state dropped, and the image read back as
/// the rows store reads it (a blank, the rows by position, the seal by its words, the state's own bytes); its new id.
fn throughDisk(a: std.mem.Allocator, e: *@import("glm_cuda").engine.Engine, id: u64) !u64 {
    const glm = @import("glm_cuda");
    const R = glm.disk_ranks;
    const I = glm.disk_image;
    const Range = struct { off: u64, len: u64 };
    const Io = struct {
        e: *glm.engine.Engine,
        id: u64,
        pub fn take(io: @This(), rank: u32, off: u64, out: []u8) !void {
            return R.take(io.e, rank, io.id, off, out);
        }
        pub fn put(io: @This(), rank: u32, off: u64, bytes: []const u8) !void {
            return R.put(io.e, rank, io.id, off, bytes);
        }
    };
    var pre: [2]u64 = undefined;
    try R.prefixes(e, &pre);
    const sh = try R.shape(e);
    const s = e.snaps.get(id) orelse return error.NoKeptState;
    const at: u32 = @intCast(s.pos);
    const home = R.homeOf(e, s);
    var list: std.ArrayList(I.Part) = .empty;
    try I.storeParts(sh, &pre, at, &list, a);
    var total: u64 = 0;
    var rows: std.ArrayList(Range) = .empty;
    var own: std.ArrayList(Range) = .empty;
    for (list.items) |part| {
        if (part.bytes > 0) try (if (part.row != 0) &rows else &own).append(a, .{ .off = total, .len = part.bytes });
        total += part.bytes;
    }
    const img = try a.alloc(u8, @intCast(total));
    try R.gatherWith(Io{ .e = e, .id = id }, sh, &pre, at, home, &[_]Range{.{ .off = 0, .len = total }}, img, a);
    var x: [I.Words.count]u64 = undefined;
    @memcpy(std.mem.sliceAsBytes(&x), img[0..I.Words.bytes]);
    const w = try I.Words.decode(x, at);
    if (w.home != home) return error.DiskWords;
    e.dropState(id);
    const nid = try R.blank(e, at);
    const read_home = R.homeOf(e, e.snaps.get(nid).?);
    for ([_][]const Range{ rows.items, own.items }, 0..) |ranges, k| {
        if (k == 1) try R.seal(e, nid, w);
        var n: usize = 0;
        for (ranges) |g| n += @intCast(g.len);
        const in = try a.alloc(u8, n);
        var o: usize = 0;
        for (ranges) |g| {
            @memcpy(in[o..][0..@intCast(g.len)], img[@intCast(g.off)..][0..@intCast(g.len)]);
            o += @intCast(g.len);
        }
        try R.scatterWith(Io{ .e = e, .id = nid }, sh, &pre, at, read_home, ranges, in, a);
    }
    std.debug.print("kv shard: a {d}-token state written at home {?d} ({d} MiB image), read back at home {?d}\n", .{ at, home, total >> 20, read_home });
    return nid;
}

var kept_id: u64 = 0;

fn keepAt(ctx: *anyopaque, at: u32) void {
    const e: *@import("glm_cuda").engine.Engine = @ptrCast(@alignCast(ctx));
    _ = at;
    kept_id = e.keepState() catch 0;
}

/// `glm-kv-shard-check <ckpt> <rank> <world> <idfile>`: TF_GLM_KV_SHARD=1 is set here; both ranks run it.
pub fn shardCheck(gpu: Gpu, checkpoint: []const u8, rank: c_int, world: c_int, id_path: []const u8) !void {
    const glm = @import("glm_cuda");
    var arena = std.heap.ArenaAllocator.init(gpu.gpa);
    defer arena.deinit();
    const a = arena.allocator();
    try check.expect(world == 2, "glm-kv-shard-check runs on two ranks (TP=2), not {d}", .{world});
    var lib = try cuda.nccl.Library.open();
    defer lib.close();
    const comm = try lanes_tests.openComm(gpu, &lib, rank, world, id_path);
    defer _ = lib.api.ncclCommDestroy(comm);
    var problem: []const u8 = "";
    var set = glm.ranks.Settings.read(&glm.ranks.environment, &problem) catch |err| {
        std.debug.print("REFUSE {s}\n", .{problem});
        return err;
    };
    set.pages = true;
    set.kv8 = true; // the recipe's KV=fp8
    set.kv_shard = .{ .on = true };
    // no drafts: the target MTP head is refused beside sharding, and DFlash2's verify windows are these windows
    const e = try glm.engine.Engine.init(gpu.gpa, gpu.io, gpu.d, gpu.ctx, checkpoint, &lib, comm, @intCast(rank), @intCast(world), .{ .capacity = capacity, .drafts = false, .settings = set });
    defer e.deinit();
    var link = try glm.follow.Link.init(gpu.gpa, gpu.d, &lib, comm, e.stream, capacity);
    defer link.deinit();
    if (rank != 0) {
        try glm.follow.follow(e, &link);
        check.pass("rank {d} mapped rank 0's sharded pages and twins, attended for both ranks' heads, carried and checked the layouts to the end", .{rank});
        return;
    }
    const x: glm.follow.Leader = .{ .e = e, .link = &link };
    defer x.end() catch {};
    e.link = &link; // the disk steps (cuda_disk_ranks) cross on it
    const P = glm.pages;
    const l = try P.layout(gpu.gpa, e, 0);
    const count = 4 * P.tableLen(capacity) + 4; // two parts of two windows each
    try x.pool(streams, count, l.pageBytes());
    e.pages = try P.Pages.init(gpu.gpa, e, l, streams, count, true);
    const pages = e.pages.?;
    e.kv_ctx.counts = &pages.shard.?.parts.counts;
    pages.shard.?.switches.check = 4; // a layout digest every 4 decode rounds (rank 0 decides; the follower compares)
    pages.poison = true; // a row read before its write turns the reply to NaN
    _ = try x.snapshots(512 << 20, 0);
    const V = e.share.vocab.len();
    const mine = try a.alloc(u16, 16 * V);
    const twin = try a.alloc(u16, 16 * V);
    var failed = false;
    var compared: usize = 0;
    for (cases, 0..) |c, k| {
        const ids = try a.alloc(u32, c.len);
        for (ids, 0..) |*u, i| u.* = @intCast(100 + (i * 7919 + k * 31) % 5000);
        // a live stream in the other part, so the placement is the case's, not an empty pool's
        try x.bind(other_slot);
        const busy = try pages.openWith(other_slot, 600, 0, 1 - c.home, &.{});
        try x.beginOn(null, busy, 0);
        pages.shard.?.switches.twins = c.twin;
        var first: [2]u32 = undefined;
        const at: usize = glm.engine.prompt_rows; // the kept state's mark (a chunk start)
        for ([_]u32{ sharded_slot, twin_slot }, 0..) |slot, j| {
            try x.bind(slot);
            if (slot == sharded_slot) try x.beginOn(null, try pages.openWith(slot, c.len, 64, c.home, &.{}), 0) else try x.begin(null);
            const marks: ?glm.engine.Marks = if (c.resume_part != null and slot == sharded_slot) .{ .at = &.{@as(u32, @intCast(at))}, .ctx = e, .call = keepAt } else null;
            first[j] = try x.prefill(ids, null, null, marks);
            try gpu.d.check(gpu.d.api.cuMemcpyDtoH_v2(if (j == 0) mine.ptr else twin.ptr, e.bp.logits, V * 2), "cuMemcpyDtoH");
        }
        if (first[0] != first[1] or !std.mem.eql(u16, mine[0..V], twin[0..V])) {
            std.debug.print("FAIL kv shard: {s}: a {d}-token prompt's first token {d} or logits differ from the unsharded twin's ({d})\n", .{ c.name, c.len, first[0], first[1] });
            failed = true;
        }
        if (c.resume_part) |rp| { // the same prompt again from the kept state, in part `rp`
            try x.release(sharded_slot);
            pages.shard.?.switches.twins = c.resume_twin;
            if (c.disk and kept_id != 0) kept_id = try throughDisk(a, e, kept_id);
            try x.bind(sharded_slot);
            try x.beginOn(null, try pages.openWith(sharded_slot, c.len, 64, rp, &.{}), 0);
            if (kept_id == 0) {
                std.debug.print("FAIL kv shard: {s}: no state was kept at {d}\n", .{ c.name, at });
                failed = true;
            } else {
                const again = try x.prefill(ids, .{ .id = kept_id, .at = at }, null, null);
                try gpu.d.check(gpu.d.api.cuMemcpyDtoH_v2(mine.ptr, e.bp.logits, V * 2), "cuMemcpyDtoH");
                if (again != first[1] or !std.mem.eql(u16, mine[0..V], twin[0..V])) {
                    std.debug.print("FAIL kv shard: {s}: the resumed prompt's first token {d} or logits differ from the twin's ({d})\n", .{ c.name, again, first[1] });
                    failed = true;
                }
                e.dropState(kept_id);
                kept_id = 0;
            }
        }
        const widths = [_]usize{ 1, 2, 4, 16, 3, 1, 8, 16 };
        for (widths, 0..) |rows, round| { // windows of 1 to 16 rows: owner-computed once the twin is given back
            var w: [16]u32 = undefined;
            for (w[0..rows], 0..) |*u, r| u.* = @intCast(100 + (round * 31 + k * 7 + r * 13) % 5000);
            var outs: [2][16]u32 = undefined;
            for ([_]u32{ sharded_slot, twin_slot }, 0..) |slot, j| {
                try x.bind(slot);
                try x.verify(w[0..rows], outs[j][0..rows], 0);
                try gpu.d.check(gpu.d.api.cuMemcpyDtoH_v2(if (j == 0) mine.ptr else twin.ptr, e.bd.logits, rows * V * 2), "cuMemcpyDtoH");
                try x.keep(1 + (round + k) % rows);
            }
            if (!std.mem.eql(u16, mine[0 .. rows * V], twin[0 .. rows * V]) or !std.mem.eql(u32, outs[0][0..rows], outs[1][0..rows])) {
                std.debug.print("FAIL kv shard: {s}: window {d} ({d} rows) differs from the twin's\n", .{ c.name, round, rows });
                failed = true;
            }
            compared += rows;
        }
        if (pages.twinCount(sharded_slot) != 0) {
            std.debug.print("FAIL kv shard: {s}: the twin was not given back at the first window\n", .{c.name});
            failed = true;
        }
        try x.release(sharded_slot);
        try x.release(twin_slot);
        try x.release(other_slot);
    }
    const counts = pages.shard.?.parts.counts;
    for ([_]struct { []const u8, u64 }{ .{ "owner layers", counts.owner_layers }, .{ "owner prompts", counts.owner_prompts }, .{ "twins made", counts.twins_made }, .{ "carries", counts.carries }, .{ "layout checks", counts.checks } }) |c| if (c[1] == 0) {
        std.debug.print("FAIL kv shard: the check never reached a path: {s} is 0\n", .{c[0]});
        failed = true;
    };
    for (0..2) |i| if (pages.shard.?.parts.pools[i].available() != pages.shard.?.parts.h) {
        std.debug.print("FAIL kv shard: part {d} holds pages after every stream ended\n", .{i});
        failed = true;
    };
    if (failed) return error.TestFailed;
    check.pass("kv shard at TP=2: {d} prompt paths (twins, owner-computed, both parts, resumes at home and carried) and {d} window rows equal the unsharded twin's logits and tokens byte for byte; {d} owner layers ({d} MiB sent by rank 0), {d} twins, {d} carries ({d} MiB), {d} layout checks; both parts empty after", .{ cases.len, compared, counts.owner_layers, counts.owner_bytes >> 20, counts.twins_made, counts.carries, counts.carry_bytes >> 20, counts.checks });
}
