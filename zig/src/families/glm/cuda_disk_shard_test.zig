//! A KV-sharded kept state's disk image (TF_GLM_KV_SHARD beside --learn-conversations) on two ranks whose device
//! memory is host memory and whose NCCL sends queue bytes for the peer's receive: each rank writes the bytes it holds,
//! the store image is the same as an unsharded state's, and a read lands in its home part, or is carried to the stream
//! when placed elsewhere, as a memory restore does. Sharded and unsharded images read each other's. Each rank's kept
//! states take a page of device memory, then pinned host pages (0015's two tiers), so every copy crosses both.
const std = @import("std");
const cuda = @import("cuda");
const weights = @import("cuda_weights.zig");
const shard = @import("cuda_shard.zig");
const pages_mod = @import("cuda_pages.zig");
const snapshots = @import("cuda_snapshots.zig");
const snap_pages = @import("cuda_snapshot_pages.zig");
const layout = @import("cuda_snapshot_layout.zig");
const image = @import("cuda_disk_image.zig");
const ranks = @import("cuda_disk_ranks.zig");
const follow = @import("cuda_follow.zig");
const kvs = @import("cuda_kv_shard.zig");
const kv_keep = @import("cuda_kv_shard_keep.zig");
const Engine = @import("cuda_engine.zig").Engine;
const expect = std.testing.expect;
const expectEqual = std.testing.expectEqual;
const expectError = std.testing.expectError;

/// Device addresses are host addresses; NCCL's sends queue bytes by sender, its receives take them in order.
const H = struct {
    var held: [96]struct { ptr: u64 = 0, len: usize = 0 } = @splat(.{});
    var queue: [2]std.ArrayList(u8) = .{ .empty, .empty }; // bytes rank r sent its peer, not yet received
    var taken: [2]usize = .{ 0, 0 };

    fn bytes(p: u64, n: usize) []u8 {
        return @as([*]u8, @ptrFromInt(p))[0..n];
    }
    fn alloc(out: *u64, n: usize) callconv(.c) cuda.abi.Result {
        const m = std.heap.page_allocator.alloc(u8, @max(n, 1)) catch return 2;
        @memset(m, 0);
        for (&held) |*h| if (h.ptr == 0) {
            h.* = .{ .ptr = @intFromPtr(m.ptr), .len = m.len };
            out.* = h.ptr;
            return 0;
        };
        std.heap.page_allocator.free(m);
        return 2;
    }
    fn free(p: u64) callconv(.c) cuda.abi.Result {
        for (&held) |*h| if (h.ptr == p) {
            std.heap.page_allocator.free(bytes(h.ptr, h.len));
            h.* = .{};
            return 0;
        };
        return 1;
    }
    fn set(p: u64, v: u8, n: usize) callconv(.c) cuda.abi.Result {
        @memset(bytes(p, n), v);
        return 0;
    }
    fn set32(p: u64, v: u32, n: usize, _: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
        @memset(@as([*]align(1) u32, @ptrFromInt(p))[0..n], v);
        return 0;
    }
    fn dtod(dst: u64, src: u64, n: usize, _: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
        @memmove(bytes(dst, n), bytes(src, n));
        return 0;
    }
    fn dtoh(dst: ?*anyopaque, src: u64, n: usize) callconv(.c) cuda.abi.Result {
        @memcpy(@as([*]u8, @ptrCast(dst.?))[0..n], bytes(src, n));
        return 0;
    }
    fn htod(dst: u64, src: ?*const anyopaque, n: usize) callconv(.c) cuda.abi.Result {
        @memcpy(bytes(dst, n), @as([*]const u8, @ptrCast(src.?))[0..n]);
        return 0;
    }
    fn wait(_: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
        return 0;
    }
    fn eventCreate(out: *cuda.abi.Event, _: c_uint) callconv(.c) cuda.abi.Result {
        out.* = @ptrFromInt(8);
        return 0;
    }
    fn eventDestroy(_: cuda.abi.Event) callconv(.c) cuda.abi.Result {
        return 0;
    }
    fn eventRecord(_: cuda.abi.Event, _: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
        return 0;
    }
    fn eventQuery(_: cuda.abi.Event) callconv(.c) cuda.abi.Result {
        return 0;
    }
    fn hostAlloc(out: *?*anyopaque, n: usize, _: c_uint) callconv(.c) cuda.abi.Result {
        var p: u64 = 0;
        const r = alloc(&p, n);
        out.* = @ptrFromInt(p);
        return r;
    }
    fn hostFree(p: ?*anyopaque) callconv(.c) cuda.abi.Result {
        return free(@intFromPtr(p.?));
    }
    fn toHost(dst: ?*anyopaque, src: u64, n: usize, _: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
        return dtoh(dst, src, n);
    }
    fn fromHost(dst: u64, src: ?*const anyopaque, n: usize, _: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
        return htod(dst, src, n);
    }
    fn rankOf(comm: cuda.nccl.Comm) usize {
        return @intFromPtr(comm) - 1;
    }
    var link: [2]u64 = .{ 0, 0 }; // each rank's link buffer (the test notes it after Link.init)
    fn bcast(buf: u64, _: u64, n: usize, _: cuda.nccl.DataType, root: c_int, comm: cuda.nccl.Comm, _: cuda.abi.Stream) callconv(.c) cuda.nccl.Result {
        const r = rankOf(comm);
        if (root < 0 or root >= link.len) return 3;
        const g: usize = @intCast(root);
        if (link[r] == 0 or link[g] == 0 or buf < link[r] or buf - link[r] + n > (64 + 256 * 4)) return 3;
        const off = buf - link[r];
        @memmove(bytes(link[r] + off, n), bytes(link[g] + off, n));
        return 0;
    }
    fn send(buf: u64, n: usize, _: cuda.nccl.DataType, peer: c_int, comm: cuda.nccl.Comm, _: cuda.abi.Stream) callconv(.c) cuda.nccl.Result {
        const r = rankOf(comm);
        if (@as(usize, @intCast(peer)) != 1 - r) return 3;
        queue[r].appendSlice(std.heap.page_allocator, bytes(buf, n)) catch return 2;
        return 0;
    }
    fn recv(buf: u64, n: usize, _: cuda.nccl.DataType, peer: c_int, comm: cuda.nccl.Comm, _: cuda.abi.Stream) callconv(.c) cuda.nccl.Result {
        const from: usize = @intCast(peer);
        if (from != 1 - rankOf(comm) or taken[from] + n > queue[from].items.len) return 3; // a receive no send meets
        @memcpy(bytes(buf, n), queue[from].items[taken[from]..][0..n]);
        taken[from] += n;
        return 0;
    }
    fn drained() bool {
        return taken[0] == queue[0].items.len and taken[1] == queue[1].items.len;
    }
    fn reset() void {
        for (&queue) |*q| q.clearAndFree(std.heap.page_allocator);
        taken = .{ 0, 0 };
        link = .{ 0, 0 };
    }
};

const at: u32 = 1024; // a kept state's mark, on the chunk grid
const prompt: usize = 1100;
const reply: usize = 100;

/// One rank's engine of TP=2's shapes on a KDA, a DSA and a KDA layer, 4,096 rows, E4M3 pages, sharded or not.
const Rank = struct {
    d: cuda.Driver,
    lib: cuda.nccl.Library,
    layers: [3]weights.Layer,
    e: Engine,
    p: *pages_mod.Pages,
    counts: kvs.Counts = .{},

    fn init(r: *Rank, gpa: std.mem.Allocator, rank: usize, on: bool, carry: bool) !void {
        r.counts = .{};
        r.d = undefined;
        r.d.api.cuMemAlloc_v2 = H.alloc;
        r.d.api.cuMemFree_v2 = H.free;
        r.d.api.cuMemsetD8_v2 = H.set;
        r.d.api.cuMemsetD32Async = H.set32;
        r.d.api.cuMemcpyHtoD_v2 = H.htod;
        r.d.api.cuMemcpyDtoH_v2 = H.dtoh;
        r.d.api.cuMemcpyDtoDAsync_v2 = H.dtod;
        r.d.api.cuStreamSynchronize = H.wait;
        r.d.api.cuEventCreate = H.eventCreate;
        r.d.api.cuEventDestroy_v2 = H.eventDestroy;
        r.d.api.cuEventRecord = H.eventRecord;
        r.d.api.cuEventQuery = H.eventQuery;
        r.d.api.cuMemHostAlloc = H.hostAlloc;
        r.d.api.cuMemFreeHost = H.hostFree;
        r.d.api.cuMemcpyDtoHAsync_v2 = H.toHost;
        r.d.api.cuMemcpyHtoDAsync_v2 = H.fromHost;
        r.lib = undefined;
        r.lib.api.ncclBroadcast = H.bcast;
        r.lib.api.ncclSend = H.send;
        r.lib.api.ncclRecv = H.recv;
        for (&r.layers, [_]bool{ false, true, false }) |*l, mla| l.mixer = if (mla) .{ .mla = undefined } else .{ .kda = undefined };
        const e = &r.e;
        e.* = undefined;
        e.gpa = gpa;
        e.d = &r.d;
        e.stream = .{ .d = &r.d, .handle = null };
        e.capacity = 4096;
        e.seqs = .empty;
        e.pages = null;
        e.images = null;
        e.drafter = null;
        e.settings = .{ .kv8 = true, .pages = true, .kv_shard = .{ .on = on, .carry = carry, .twins = false } };
        e.mtp = false;
        e.w = undefined;
        e.w.config = .{};
        e.c = e.w.config;
        e.w.share = try shard.Share.of(&e.w.config, rank, 2);
        e.share = e.w.share;
        e.w.layers = &r.layers;
        e.snaps = .empty;
        e.snap_pages = null;
        e.next_snap = 1;
        e.announce = 0;
        e.dropped = .empty;
        e.disk_failed = .empty;
        e.link = null;
        e.lib = &r.lib;
        e.comm = @ptrFromInt(rank + 1);
        e.kv_room = null;
        e.kv_stage = try cuda.DeviceBuffer.alloc(&r.d, kv_keep.piece_bytes);
        e.kv_ctx = .{ .rank = rank, .lib = &r.lib, .comm = e.comm, .pq = 0, .po = 0, .rows = 0, .counts = &r.counts };
        try e.snaps.ensureTotalCapacity(gpa, 16);
        e.snap_pages = try snap_pages.Pool.initTiers(gpa, &r.d, snap_pages.page, 32 << 20); // a device page, then host
        const l = try pages_mod.layout(gpa, e, 0);
        try pages_mod.ownSequence(e);
        r.p = try pages_mod.Pages.init(gpa, e, l, 4, 20, rank == 0);
        e.pages = r.p;
    }

    fn deinit(r: *Rank) void {
        const e = &r.e;
        var it = e.snaps.valueIterator();
        while (it.next()) |s| snapshots.free(e, s);
        e.snaps.deinit(e.gpa);
        e.disk_failed.deinit(e.gpa);
        r.p.deinit();
        for (e.seqs.items) |q| if (q) |s| {
            s.st.deinit();
            e.gpa.destroy(s);
        };
        e.seqs.deinit(e.gpa);
        e.snap_pages.?.deinit();
        e.kv_stage.?.free();
    }
};

/// Two ranks, sharded or not, with or without carries.
const Pair = struct {
    r: [2]Rank,

    fn init(p: *Pair, on: bool, carry: bool) !void {
        try p.r[0].init(std.testing.allocator, 0, on, carry);
        errdefer p.r[0].deinit();
        try p.r[1].init(std.testing.allocator, 1, on, carry);
    }

    fn deinit(p: *Pair) void {
        p.r[1].deinit();
        p.r[0].deinit();
        H.reset();
    }

    /// A stream on slot `slot` on both ranks: rank 0 places it (in `home` when it has the room), rank 1 maps its ids.
    fn open(p: *Pair, slot: u32, n_prompt: usize, home: ?u8) !void {
        const ids = try p.r[0].p.openWith(slot, n_prompt, reply, home, &.{});
        const twin = p.r[0].p.twinCount(slot);
        if (p.r[0].e.settings.kv_shard.on) try p.r[1].p.mapWith(slot, ids, twin, 0) else try p.r[1].p.map(slot, ids, 0);
    }

    fn close(p: *Pair, slot: u32) void {
        for (&p.r) |*x| x.p.close(slot);
    }

    fn bind(p: *Pair, slot: u32) !void {
        for (&p.r) |*x| try x.e.bind(slot);
    }

    /// The owner part of the bound stream (null: unsharded).
    fn owner(p: *Pair) ?u8 {
        return if (p.r[0].e.cur.st.hold) |h| h.owner else null;
    }
};

/// Part `k` of a full plan: a byte pattern of its own; the prefix (part 0) also by rank, as each rank's heads differ.
fn pattern(rank: usize, k: usize, j: usize) u8 {
    return if (k == 0) @truncate(rank *% 97 +% j *% 13 +% 1) else @truncate(k *% 131 +% j *% 7 +% 3);
}

/// The parity a fill's KDA copy bits take (null: mixed, as the other tests'): a live-half keep's.
var fill_parity: ?u1 = null;

/// Each part of the bound sequence's full plan at `at` (on a rank that does not hold its latent rows, those land on
/// the trash page), filled with `pattern`; then its position and head rows set as a pass leaves them.
fn fill(e: *Engine) !void {
    const plan = try layout.Plan.initFor(&e.cur.st, &e.c, at, null, true, e.cur.st.cur);
    for (plan.parts[0..plan.count], 0..) |part, k| {
        var done: usize = 0;
        while (done < part.len) {
            const x = plan.piece(part, done);
            for (H.bytes(x.live, x.len), 0..) |*b, i| b.* = pattern(e.share.rank, k, done + i);
            done += x.len;
        }
    }
    const bits = try e.gpa.alloc(u1, e.cur.st.cur.len);
    defer e.gpa.free(bits);
    for (bits, 0..) |*b, i| b.* = if (fill_parity) |h| h else @intCast(i & 1);
    try e.cur.st.restoreHost(e.stream, at, 0, bits, .{ 0, 0 });
}

/// The bound sequence's parts at `at` (full plan), copied out.
const Parts = struct {
    bytes: [layout.Plan.max_parts][]u8 = undefined,
    lat: [layout.Plan.max_parts]bool = undefined,
    count: usize = 0,

    fn of(e: *Engine) !Parts {
        const plan = try layout.Plan.initFor(&e.cur.st, &e.c, at, null, true, e.cur.st.cur);
        var out: Parts = .{ .count = plan.count };
        for (plan.parts[0..plan.count], 0..) |part, k| {
            out.lat[k] = part.lat; // the DSA layer's codes and scales: the home rank's alone (sharded; a live-half keep's head in two parts shifts them)
            out.bytes[k] = try e.gpa.alloc(u8, part.len);
            var done: usize = 0;
            while (done < part.len) {
                const x = plan.piece(part, done);
                @memcpy(out.bytes[k][done..][0..x.len], H.bytes(x.live, x.len));
                done += x.len;
            }
        }
        return out;
    }

    fn deinit(x: *Parts, gpa: std.mem.Allocator) void {
        for (x.bytes[0..x.count]) |b| gpa.free(b);
    }
};

/// The bound streams' parts on both ranks equal the kept ones: every rank its own prefix and shared parts, and the
/// latent rows (the state's, from rank `from`) wherever the new stream's rows are held.
fn expectRestored(p: *Pair, kept: *const [2]Parts, from: usize) !void {
    for (&p.r, 0..) |*x, rank| {
        var now = try Parts.of(&x.e);
        defer now.deinit(x.e.gpa);
        if (now.count != kept[rank].count) return error.Mismatch;
        const holds = if (x.e.cur.st.hold) |h| h.has(rank) else true;
        for (0..now.count) |k| {
            if (now.lat[k] and !holds) continue; // its latent table reaches the trash page alone
            const want = if (now.lat[k]) kept[from].bytes[k] else kept[rank].bytes[k];
            if (!std.mem.eql(u8, want, now.bytes[k])) return error.Mismatch;
        }
    }
}

/// Both ranks' image views of state `id`, as rank 0's disk steps reach them (cuda_disk_ranks.copy on each rank).
const Io = struct {
    p: *Pair,
    id: u64,

    pub fn take(io: Io, rank: u32, off: u64, out: []u8) !void {
        return ranks.copy(&io.p.r[rank].e, io.id, off, out, true);
    }

    pub fn put(io: Io, rank: u32, off: u64, bytes: []const u8) !void {
        return ranks.copy(&io.p.r[rank].e, io.id, off, @constCast(bytes), false);
    }
};

const Range = struct { off: u64, len: u64 };

fn prefixesOf(p: *Pair) ![2]u64 {
    return .{ (try ranks.shape(&p.r[0].e)).prefix, (try ranks.shape(&p.r[1].e)).prefix };
}

// Rank 1's disk steps with rank 0's link beside it: an op that fails here poisons its id, answers rank 0's takes
// and seal with the failure, and the step loop runs on — a failed recall costs rank 0 a cold pass, not the rank.
test "GLM disk steps: a follower's failed op poisons the id, answers the seal with a failure and the loop runs on" {
    const a = std.testing.allocator;
    var p: Pair = undefined;
    try p.init(false, false);
    defer p.deinit();
    try p.open(1, prompt, null); // the bound stream the blanks' states stand beside
    defer p.close(1);
    try p.bind(1);
    var link: [2]follow.Link = undefined;
    for (&p.r, 0..) |*r, i| {
        link[i] = try follow.Link.init(a, &r.d, &r.lib, r.e.comm, r.e.stream, 256);
        H.link[i] = link[i].buf.ptr;
    }
    defer for (&link) |*l| l.deinit();
    const e = &p.r[1].e;
    const disk: u32 = @backingInt(follow.Kind.disk);
    // its blank for id 3 stands
    try ranks.step(e, &link[1], .{ .kind = disk, .flags = @backingInt(ranks.Op.blank), .a = 3, .seed = at, .depth = 0, .top_k = 0 }, &.{});
    try expect(e.snaps.contains(3));
    // a write into it past the blank's front fails on this rank: the id poisoned, its half-made state dropped,
    // and no error out of the step
    var bytes: [2]u32 = undefined; // the op's 8 bytes of ids
    const sh = try ranks.shape(e);
    const put: follow.Header = .{ .kind = disk, .flags = @backingInt(ranks.Op.put), .a = 3, .seed = image.Words.bytes + image.front(sh, at) + 8, .depth = 8, .top_k = 1 };
    try ranks.step(e, &link[1], put, &bytes);
    try expect(!e.snaps.contains(3)); // the half-made state dropped
    try expect(e.disk_failed.contains(3));
    try ranks.step(e, &link[1], put, &bytes); // skipped, not an error
    // a take for the poisoned id still crosses the link and answers with its status
    try ranks.step(e, &link[1], .{ .kind = disk, .flags = @backingInt(ranks.Op.take), .a = 3, .seed = 0, .depth = 8, .top_k = 1 }, &.{});
    try link[0].from(1, 8 + @sizeOf(u32));
    var answer: [8 + @sizeOf(u32)]u8 = undefined;
    try link[0].buf.download(64, &answer);
    try expectEqual(@as(u32, 1), std.mem.readInt(u32, answer[8..12], .little));
    // its seal answers the failure too
    var seal_ids: [2 * image.Words.count]u32 = undefined;
    @memset(&seal_ids, 0);
    try ranks.step(e, &link[1], .{ .kind = disk, .flags = @backingInt(ranks.Op.seal), .a = 3 }, &seal_ids);
    try link[0].from(1, @sizeOf(u32));
    var ok: u32 = 0;
    try link[0].buf.download(64, std.mem.asBytes(&ok));
    try expectEqual(@as(u32, 1), ok);
    // a healthy blank's seal answers 0 and stands sealed
    try ranks.step(e, &link[1], .{ .kind = disk, .flags = @backingInt(ranks.Op.blank), .a = 8, .seed = at, .depth = 0, .top_k = 0 }, &.{});
    const words = (image.Words{ .at = at, .mtp_len = 0, .base = 0, .mtp_base = 0, .cur = 0 }).encode();
    try ranks.step(e, &link[1], .{ .kind = disk, .flags = @backingInt(ranks.Op.seal), .a = 8 }, std.mem.bytesAsSlice(u32, std.mem.sliceAsBytes(&words)));
    try link[0].from(1, @sizeOf(u32));
    try link[0].buf.download(64, std.mem.asBytes(&ok));
    try expectEqual(@as(u32, 0), ok);
    try expect(e.snaps.get(8).?.len != 0);
    try expect(!e.disk_failed.contains(8));
}

/// The store image of kept state `id` (rank 0's image, then rank 1's prefix), gathered in ranges across the seams.
fn write(p: *Pair, id: u64) ![]u8 {
    const pre = try prefixesOf(p);
    const sh = try ranks.shape(&p.r[0].e);
    const total = image.imageBytes(sh, at) + pre[1];
    const out = try std.testing.allocator.alloc(u8, @intCast(total));
    errdefer std.testing.allocator.free(out);
    var cuts = [_]u64{ 0, 40, image.Words.bytes + sh.prefix + 100, total / 3, total / 2 + 7, total - pre[1] - 3, total };
    std.mem.sort(u64, &cuts, {}, std.sort.asc(u64));
    var ranges: [cuts.len - 1]Range = undefined;
    for (&ranges, cuts[0 .. cuts.len - 1], cuts[1..]) |*g, lo, hi| g.* = .{ .off = lo, .len = hi - lo };
    const home = ranks.homeOf(&p.r[0].e, p.r[0].e.snaps.get(id).?);
    try ranks.gatherWith(Io{ .p = p, .id = id }, sh, &pre, at, home, &ranges, out, std.testing.allocator);
    return out;
}

/// A disk read of store image `img` as id `id` on both ranks, in the rows store's order: a blank in rank 0's chosen
/// home, the rows by position, then the state's own bytes with its words first (every rank seals); the words back.
fn read(p: *Pair, img: []const u8, id: u64) !image.Words {
    const a = std.testing.allocator;
    const home = ranks.readHome(&p.r[0].e);
    for (&p.r) |*x| try snapshots.blankAs(&x.e, id, at, home, 0);
    const pre = try prefixesOf(p);
    const sh = try ranks.shape(&p.r[0].e);
    var list: std.ArrayList(image.Part) = .empty;
    defer list.deinit(a);
    try image.storeParts(sh, &pre, at, &list, a);
    var rows: std.ArrayList(Range) = .empty;
    defer rows.deinit(a);
    var own: std.ArrayList(Range) = .empty;
    defer own.deinit(a);
    var off: u64 = 0;
    for (list.items) |part| {
        if (part.bytes > 0) try (if (part.row != 0) &rows else &own).append(a, .{ .off = off, .len = part.bytes });
        off += part.bytes;
    }
    try expectEqual(@as(u64, img.len), off);
    try scatter(p, sh, &pre, home, id, rows.items, img);
    var x: [image.Words.count]u64 = undefined;
    @memcpy(std.mem.sliceAsBytes(&x), img[0..image.Words.bytes]);
    const w = try image.Words.decode(x, at);
    for (&p.r) |*r| try snapshots.seal(&r.e, id, w.mtp_len, .{ w.base, w.mtp_base }, w.cur);
    try scatter(p, sh, &pre, home, id, own.items, img);
    return w;
}

/// `ranges` of `img` into both ranks, as the store's scatter hands them (the bytes of those ranges, in order).
fn scatter(p: *Pair, sh: image.Shape, pre: []const u64, home: ?u8, id: u64, ranges: []const Range, img: []const u8) !void {
    const a = std.testing.allocator;
    var n: usize = 0;
    for (ranges) |g| n += @intCast(g.len);
    const in = try a.alloc(u8, n);
    defer a.free(in);
    var o: usize = 0;
    for (ranges) |g| {
        @memcpy(in[o..][0..@intCast(g.len)], img[@intCast(g.off)..][0..@intCast(g.len)]);
        o += @intCast(g.len);
    }
    try ranks.scatterWith(Io{ .p = p, .id = id }, sh, pre, at, home, ranges, in, a);
}

/// Restores state `id` into the bound streams: its home rank first (a carry's sender, so the receive finds its bytes).
fn restore(p: *Pair, id: u64) !void {
    const home = p.r[0].e.snaps.get(id).?.home orelse 0;
    try snapshots.restore(&p.r[home].e, id);
    try snapshots.restore(&p.r[1 - home].e, id);
    if (!H.drained()) return error.Mismatch;
}

/// A stream placed in part `home`, filled and kept on both ranks as id 1; its parts copied out on each rank.
fn keepIn(p: *Pair, home: ?u8, kept: *[2]Parts) !void {
    try p.open(1, prompt, home);
    try p.bind(1);
    if (home) |h| try expectEqual(h, p.owner().?);
    for (&p.r) |*x| try fill(&x.e);
    for (&p.r) |*x| try snapshots.keepAs(&x.e, 1);
    for (&p.r) |*x| if (x.e.snaps.get(1).?.lease.host.ids.items.len == 0) return error.Mismatch; // it runs onto host pages
    for (&p.r, 0..) |*x, rank| kept[rank] = try Parts.of(&x.e);
}

/// A state kept by `writer` (sharded at `home`, or unsharded) written to an image, read by `reader` as id 2 and restored
/// into a new stream placed in its read home part, or (`away`) in the other part with the home part full; true when
/// the latent rows were carried.
fn roundTrip(writer: *Pair, reader: *Pair, home: ?u8, away: bool) !bool {
    const a = std.testing.allocator;
    var kept: [2]Parts = undefined;
    try keepIn(writer, home, &kept);
    defer for (&kept) |*k| k.deinit(a);
    const img = try write(writer, 1);
    defer a.free(img);
    // the words name the home the state was written in; the bytes past them are the same for either home
    var x: [image.Words.count]u64 = undefined;
    @memcpy(std.mem.sliceAsBytes(&x), img[0..image.Words.bytes]);
    const words = try image.Words.decode(x, at);
    if (words.home != home) return error.Mismatch;
    writer.close(1);
    const w = try read(reader, img, 2);
    if (w.home != home) return error.Mismatch;
    const read_home = reader.r[0].e.snaps.get(2).?.home;
    for (&reader.r) |*r| if (r.e.snaps.get(2).?.home != read_home) return error.Mismatch;
    if (reader.r[0].e.settings.kv_shard.on and read_home == null) return error.Mismatch;
    if (away) try reader.open(3, 3000, read_home.?); // 7 of the home part's 8 pages: the new stream goes elsewhere
    try reader.open(2, prompt, read_home);
    try reader.bind(2);
    if (read_home) |h| if ((reader.owner().? == h) == away) return error.Mismatch;
    const carries = reader.r[0].counts.carries + reader.r[1].counts.carries;
    try restore(reader, 2);
    try expectRestored(reader, &kept, home orelse 0);
    reader.close(2);
    if (away) reader.close(3);
    return reader.r[0].counts.carries + reader.r[1].counts.carries > carries;
}

test "GLM disk image: a sharded kept state at home 0 and at home 1 round-trips through an image into its home part" {
    for ([_]u8{ 0, 1 }) |home| {
        var p: Pair = undefined;
        try p.init(true, true);
        defer p.deinit();
        try expect(!try roundTrip(&p, &p, home, false));
    }
}

test "GLM disk image: a sharded state read with its home part full is carried to the stream's part, as from memory" {
    for ([_]u8{ 0, 1 }) |home| {
        var p: Pair = undefined;
        try p.init(true, true);
        defer p.deinit();
        try expect(try roundTrip(&p, &p, home, true));
    }
}

test "GLM disk image: each rank writes only what it holds; the store image is an unsharded state's, words aside" {
    const a = std.testing.allocator;
    var imgs: [3][]u8 = undefined;
    var n: usize = 0;
    defer for (imgs[0..n]) |x| a.free(x);
    for ([_]?u8{ null, 0, 1 }) |home| {
        var p: Pair = undefined;
        try p.init(home != null, true);
        defer p.deinit();
        var kept: [2]Parts = undefined;
        try keepIn(&p, home, &kept);
        defer for (&kept) |*k| k.deinit(a);
        // the rank off home holds no latent rows: its image reads none (KvShardAbsent), and its lease is the packed
        // plan's alone (cuda_snapshots.keepAs)
        if (home) |h| {
            const off = p.r[1 - h].e.snaps.get(1).?;
            const sh = try ranks.shape(&p.r[1 - h].e);
            var buf: [8]u8 = undefined;
            try expectError(error.KvShardAbsent, ranks.copy(&p.r[1 - h].e, 1, image.Words.bytes + sh.prefix, &buf, true));
            const full = image.leaseBytes(sh, at, 0);
            try expectEqual(full - at * (512 + 4), @as(u64, off.len)); // one DSA layer's codes and scales fewer
        }
        imgs[n] = try write(&p, 1);
        n += 1;
    }
    try expectEqual(imgs[0].len, imgs[1].len);
    try expectEqual(imgs[0].len, imgs[2].len);
    for (imgs[1..]) |x| try std.testing.expectEqualSlices(u8, imgs[0][8..], x[8..]);
    // the format word alone differs: 1 unsharded, 3 | home << 8 | 2 parts << 16 sharded
    try expectEqual(@as(u64, 1), std.mem.readInt(u64, imgs[0][0..8], .little));
    try expectEqual(@as(u64, 3 | 0 << 8 | 2 << 16), std.mem.readInt(u64, imgs[1][0..8], .little));
    try expectEqual(@as(u64, 3 | 1 << 8 | 2 << 16), std.mem.readInt(u64, imgs[2][0..8], .little));
}

test "GLM disk image: sharded and unsharded images read each other's: the reader's own layout, the same rows" {
    // written unsharded, read sharded (in the reader's chosen home, and carried when that part is full) ...
    for ([_]bool{ false, true }) |away| {
        var w: Pair = undefined;
        try w.init(false, true);
        defer w.deinit();
        var r: Pair = undefined;
        try r.init(true, true);
        defer r.deinit();
        try expectEqual(away, try roundTrip(&w, &r, null, away));
    }
    // ... and written sharded at either home, read unsharded: every rank holds the latent rows, nothing carried
    for ([_]u8{ 0, 1 }) |home| {
        var w: Pair = undefined;
        try w.init(true, true);
        defer w.deinit();
        var r: Pair = undefined;
        try r.init(false, true);
        defer r.deinit();
        try expect(!try roundTrip(&w, &r, home, false));
        try expectEqual(@as(?u8, null), r.r[1].e.snaps.get(2).?.home);
    }
}

test "GLM disk image: a sharded read's words: format 3 names a home within two parts; others are refused" {
    const plain: image.Words = .{ .at = at, .mtp_len = 0, .base = 0, .mtp_base = 0, .cur = 5 };
    try expectEqual(@as(u64, 1), plain.encode()[0]); // unsharded: format 1, as before sharding
    try expectEqual(@as(?u8, null), (try image.Words.decode(plain.encode(), at)).home);
    const w: image.Words = .{ .at = at, .mtp_len = 0, .base = 0, .mtp_base = 0, .cur = 5, .home = 1 };
    const x = w.encode();
    try expectEqual(@as(?u8, 1), (try image.Words.decode(x, at)).home);
    // the pages layout (TF_KV_SHARE's store): formats 4 and 5 say so, a home beside it as sharding's does
    const pw: image.Words = .{ .at = at, .mtp_len = 0, .base = 0, .mtp_base = 0, .cur = 5, .pages = true };
    const ps: image.Words = .{ .at = at, .mtp_len = 0, .base = 0, .mtp_base = 0, .cur = 5, .home = 1, .pages = true };
    try expectEqual(@as(u64, 4), pw.encode()[0]);
    try expectEqual(@as(u64, 5 | 1 << 8 | 2 << 16), ps.encode()[0]);
    try expect((try image.Words.decode(pw.encode(), at)).pages);
    try expectEqual(@as(?u8, 1), (try image.Words.decode(ps.encode(), at)).home);
    for ([_]u64{ 3 | 2 << 8 | 2 << 16, 3 | 0 << 8 | 3 << 16, 3 | 1 << 8 | 2 << 16 | 1 << 24, 1 | 1 << 8, 2, 4 | 1 << 8, 5, 6 }) |bad| {
        var y = x;
        y[0] = bad;
        try expectError(error.DiskWords, image.Words.decode(y, at));
    }
}

test "GLM disk image: TF_GLM_KV_SHARD_CARRY=0 refuses a read state's resume outside its home, as a memory state's" {
    const Lanes = @import("cuda_lanes.zig").Lanes;
    const a = std.testing.allocator;
    var p: Pair = undefined;
    try p.init(true, false);
    defer p.deinit();
    var kept: [2]Parts = undefined;
    try keepIn(&p, 1, &kept); // a memory state, id 1, home 1
    defer for (&kept) |*k| k.deinit(a);
    const img = try write(&p, 1);
    defer a.free(img);
    p.close(1);
    _ = try read(&p, img, 2); // its disk copy, id 2, read where a new stream would go now: part 0 (both parts empty)
    try expectEqual(@as(?u8, 0), p.r[0].e.snaps.get(2).?.home);
    try p.open(3, 3000, 0); // part 0 full, so the next stream goes to part 1
    try p.open(2, prompt, 0);
    try p.bind(2);
    try expectEqual(@as(u8, 1), p.owner().?);
    var ln: Lanes = .{ .x = .{ .e = &p.r[0].e, .link = undefined } };
    defer ln.lanes.deinit(a);
    const sh = p.r[0].p.shard.?;
    try expect(ln.resumesHere(1)); // the memory state's home is part 1: here
    try expect(!ln.resumesHere(2)); // the disk state's is part 0, and carries are off: from 0 (KvShardNotHome)
    try expectEqual(@as(u64, 1), sh.parts.counts.resumes_dropped);
    sh.switches.carry = true;
    try expect(ln.resumesHere(2)); // with carries on it resumes, its rows carried
    try restore(&p, 2);
    try expectRestored(&p, &kept, 1);
    try expectEqual(@as(u64, 1), p.r[0].counts.carries);
}

test "GLM disk image: each sharded disk mutant is caught" {
    defer image.mutant = .none;
    defer ranks.mutant = .none;
    inline for (.{ image.Mutant.words_no_home, image.Mutant.copy_keeps_latent, image.Mutant.route_ignores_home }) |m| {
        image.mutant = m;
        var p: Pair = undefined;
        try p.init(true, true);
        defer p.deinit();
        if (roundTrip(&p, &p, 1, true)) |_| return error.MutantSurvived else |_| {}
        image.mutant = .none;
    }
    ranks.mutant = .read_unsharded; // a read with no home: nothing to carry, so the placement has no part to prefer
    var p: Pair = undefined;
    try p.init(true, true);
    defer p.deinit();
    try expectError(error.Mismatch, roundTrip(&p, &p, 1, true));
}

test "GLM disk image: a live-half keep round-trips through an image, sharded and unsharded, carried too" {
    defer layout.test_switch = null;
    defer fill_parity = null;
    layout.test_switch = true;
    fill_parity = 1; // every KDA layer on half 1: the keeps carry the live half alone
    for ([_]u8{ 0, 1 }) |home| for ([_]bool{ false, true }) |away| {
        var p: Pair = undefined;
        try p.init(true, true);
        defer p.deinit();
        try expectEqual(away, try roundTrip(&p, &p, home, away)); // the half keep written, read and restored
    };
    var p: Pair = undefined;
    try p.init(false, true);
    defer p.deinit();
    try expect(!try roundTrip(&p, &p, null, false)); // unsharded: the same, no home and no carry
}

// The startup probe's shape (cuda_glm_disk.probe, run with the store standing aside): a keep at the mark, the
// whole store image gathered in one range, and each rank's shared bytes compared equal to the image's.
test "GLM disk image: the startup probe's keep and whole-image gather pass under TF_GLM_KV_LIVE_HALF, the ranks' shared bytes the image's" {
    const a = std.testing.allocator;
    defer layout.test_switch = null;
    defer fill_parity = null;
    layout.test_switch = true;
    fill_parity = 1;
    for ([_]bool{ false, true }) |on| {
        var p: Pair = undefined;
        try p.init(on, true);
        defer p.deinit();
        try p.open(1, prompt, null);
        defer p.close(1);
        try p.bind(1);
        for (&p.r) |*x| try fill(&x.e);
        for (&p.r) |*x| try snapshots.keepAs(&x.e, 1);
        const pre = try prefixesOf(&p);
        const sh = try ranks.shape(&p.r[0].e);
        const home = ranks.homeOf(&p.r[0].e, p.r[0].e.snaps.get(1).?);
        const own = image.imageBytes(sh, at);
        const total = own + pre[1];
        const bytes = try a.alloc(u8, @intCast(total));
        defer a.free(bytes);
        const one = [1]Range{.{ .off = 0, .len = total }};
        try ranks.gatherWith(Io{ .p = &p, .id = 1 }, sh, &pre, at, home, &one, bytes, a); // one range, as the probe's
        const shared = bytes[@intCast(image.Words.bytes + sh.prefix)..@intCast(own)];
        const theirs = try a.alloc(u8, shared.len);
        defer a.free(theirs);
        for (pre[1..], 1..) |q, r| {
            var held: std.ArrayList([2]u64) = .empty;
            defer held.deinit(a);
            try image.heldShared(sh, at, if (home) |h| h == r else true, &held, a);
            for (held.items) |x| {
                const part = theirs[@intCast(x[0])..][0..@intCast(x[1])];
                try ranks.copy(&p.r[r].e, 1, image.Words.bytes + q + x[0], part, true); // that rank's copy of what it holds
                try expect(std.mem.eql(u8, shared[@intCast(x[0])..][0..part.len], part));
            }
        }
    }
}
