//! TF_GLM_KV_SHARD's pages on a driver of addresses alone: every page's keys and this rank's latent pages, each
//! stream's two tables, twins, a follower's map, the layout digest, and the unsharded pages unchanged.
const std = @import("std");
const cuda = @import("cuda");
const weights = @import("cuda_weights.zig");
const shard = @import("cuda_shard.zig");
const pages_mod = @import("cuda_pages.zig");
const state = @import("cuda_state.zig");
const kvs = @import("cuda_kv_shard.zig");
const native = @import("cuda_native.zig");
const share_mod = @import("cuda_share.zig");
const types = @import("cuda_engine_types.zig");
const kp = @import("core").kv_prompts;
const radix = @import("engine_api").prompt_radix;
const Engine = @import("cuda_engine.zig").Engine;
const expect = std.testing.expect;
const expectEqual = std.testing.expectEqual;
const expectEqualSlices = std.testing.expectEqualSlices;

/// The driver: each allocation a fresh address and its size, the tables written kept by address.
const F = struct {
    var next: u64 = 1 << 30;
    var live: usize = 0;
    var sizes: [32][2]u64 = undefined; // (address, bytes) of each allocation
    var n_alloc: usize = 0;
    var at: [16]u64 = @splat(0);
    var words: [16][512]u32 = undefined; // a table the capacity's entries fit (the store's resumes run long)
    var fills: usize = 0;
    fn alloc(out: *u64, n: usize) callconv(.c) cuda.abi.Result {
        out.* = next;
        if (n_alloc < sizes.len) sizes[n_alloc] = .{ next, n };
        n_alloc += 1;
        next += std.mem.alignForward(u64, n + 1, 1 << 20);
        live += 1;
        return 0;
    }
    fn free(_: u64) callconv(.c) cuda.abi.Result {
        live -= 1;
        return 0;
    }
    fn clear(_: u64, _: u8, _: usize) callconv(.c) cuda.abi.Result {
        return 0;
    }
    fn fill(_: u64, v: c_uint, _: usize, _: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
        fills += @intFromBool(v == 0xFFFF_FFFF);
        return 0;
    }
    fn put(dst: u64, src: ?*const anyopaque, n: usize) callconv(.c) cuda.abi.Result {
        const ids: [*]const u32 = @ptrCast(@alignCast(src.?));
        for (&at, &words) |*a, *w| if (a.* == 0 or (dst >= a.* and dst < a.* + 256)) {
            if (a.* == 0) a.* = dst;
            const from = (dst - a.*) / 4;
            @memcpy(w[from..][0 .. n / 4], ids[0 .. n / 4]);
            return 0;
        };
        return 1;
    }
    fn wait(_: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
        return 0;
    }
    fn table(t: u64, n: usize) []const u32 {
        for (at, &words) |a, *w| if (a == t) return w[0..n];
        return &.{};
    }
    fn size(ptr: u64) u64 {
        for (sizes[0..@min(n_alloc, sizes.len)]) |s| if (s[0] == ptr) return s[1];
        return 0;
    }
};

/// An engine of TP=2's shapes on a KDA, a DSA and a KDA layer, 4,096 rows, E4M3 beside DFlash2, sharded or not.
const Fake = struct {
    d: cuda.Driver,
    layers: [3]weights.Layer,
    e: Engine,

    /// A fresh driver's records (each test's first fake; a second one shares them).
    fn reset() void {
        F.live = 0;
        F.n_alloc = 0;
        F.at = @splat(0);
        F.fills = 0;
    }

    fn init(f: *Fake, gpa: std.mem.Allocator, rank: usize, on: bool) !void {
        f.d = undefined;
        f.d.api.cuMemAlloc_v2 = F.alloc;
        f.d.api.cuMemFree_v2 = F.free;
        f.d.api.cuMemsetD8_v2 = F.clear;
        f.d.api.cuMemsetD32Async = F.fill;
        f.d.api.cuMemcpyHtoD_v2 = F.put;
        f.d.api.cuStreamSynchronize = F.wait;
        for (&f.layers, [_]bool{ false, true, false }) |*l, mla| l.mixer = if (mla) .{ .mla = undefined } else .{ .kda = undefined };
        f.e = undefined;
        f.e.gpa = gpa;
        f.e.d = &f.d;
        f.e.stream = .{ .d = &f.d, .handle = null };
        f.e.capacity = 4096;
        f.e.seqs = .empty;
        f.e.pages = null;
        f.e.images = null;
        f.e.settings = .{ .kv8 = true, .pages = true, .kv_shard = .{ .on = on } };
        f.e.mtp = false;
        f.e.w = undefined;
        f.e.w.config = .{};
        f.e.w.share = try shard.Share.of(&f.e.w.config, rank, 2);
        f.e.share = f.e.w.share;
        f.e.w.layers = &f.layers;
    }

    fn deinit(f: *Fake, p: ?*pages_mod.Pages) void {
        if (p) |q| q.deinit();
        for (f.e.seqs.items) |q| if (q) |s| {
            s.st.deinit();
            f.e.gpa.destroy(s);
        };
        f.e.seqs.deinit(f.e.gpa);
    }

    fn pool(f: *Fake, streams: usize, pages: usize, leader: bool) !*pages_mod.Pages {
        const l = try pages_mod.layout(f.e.gpa, &f.e, 0);
        try pages_mod.ownSequence(&f.e);
        const p = try pages_mod.Pages.init(f.e.gpa, &f.e, l, streams, pages, leader);
        f.e.pages = p;
        return p;
    }

    fn st(f: *Fake, slot: usize) *state.State {
        return &f.e.seqs.items[slot].?.st;
    }
};

test "kv shard pages: a token's bytes are every page's keys and half its latent rows; a stream adds a latent table" {
    var f: Fake = undefined;
    Fake.reset();
    try f.init(std.testing.allocator, 0, false);
    const whole = try pages_mod.layout(std.testing.allocator, &f.e, 0);
    try expectEqual(@as(u64, 512 + 4 + 64), whole.token_bytes); // unsharded: as before
    f.e.settings.kv_shard.on = true;
    const l = try pages_mod.layout(std.testing.allocator, &f.e, 0);
    const g = pages_mod.geometry(&f.e);
    try expectEqual(@as(u64, 264_192), g.latPage());
    try expectEqual(@as(u64, 32_768), g.keyPage());
    try expectEqual(@as(u64, (32_768 + 132_096) / 512), l.token_bytes); // 322
    try expectEqual(whole.fixed_bytes + 256, l.fixed_bytes); // the latent table, 256-aligned
}

test "kv shard pages: every page's keys, this rank's part's latent pages and a trash page; a stream's two tables" {
    const gpa = std.testing.allocator;
    var f: Fake = undefined;
    Fake.reset();
    try f.init(gpa, 0, true);
    const p = try f.pool(3, 20, true); // 20 pages: two parts of 8
    defer f.deinit(p);
    const x = p.shard.?;
    try expectEqual(@as(usize, 8), x.parts.h);
    try expectEqual(@as(u64, 16 * 32_768), F.size(x.keys.?.ptr));
    try expectEqual(@as(u64, 9 * 264_192), F.size(x.lat.?.ptr));
    try expectEqual(@as(usize, 0), p.pool.total()); // the unsharded pool holds no page
    try expect(p.fits(600, 100));
    const wire = try p.openWith(1, 600, 100, null, &.{}); // 2 pages in part 0, a twin of 2 in part 1
    try expectEqualSlices(u32, &.{ 0, 1, 8, 9 }, wire);
    try expectEqual(@as(usize, 2), p.twinCount(1));
    const s1 = f.st(1);
    try expectEqualSlices(u32, &.{ 0, 1 }, F.table(s1.table, 2));
    try expectEqualSlices(u32, &.{ 0, 1 }, F.table(s1.ltable, 2)); // its owner: its own pages, part-local
    try expectEqual(kvs.Hold{ .owner = 0, .twin = true }, s1.hold.?);
    const c = s1.caches[0];
    try expectEqual(x.lat.?.ptr, c.lc); // the only DSA layer: latent page 0's start
    try expectEqual(c.lc + 512 * 512, c.ls);
    try expectEqual(x.keys.?.ptr, c.pk);
    try expectEqual(s1.table, c.pt);
    try expectEqual(@as(u64, 32_768), c.ps);
    try expectEqual(s1.ltable, c.latPt());
    try expectEqual(@as(u64, 264_192), c.latPs());
    try expectEqual(@as(usize, 1024), s1.capacity);
    // the next stream goes to part 1 (fewer live streams), its twin of one page in part 0: rank 0's latent table on it
    const wire2 = try p.openWith(2, 300, 100, null, &.{});
    try expectEqualSlices(u32, &.{ 10, 2 }, wire2);
    try expectEqual(@as(usize, 1), p.twinCount(2));
    try expectEqualSlices(u32, &.{2}, F.table(f.st(2).ltable, 1));
    try p.giveTwin(2);
    try expectEqualSlices(u32, &.{8}, F.table(f.st(2).ltable, 1)); // the twin given back: the trash page
    // a resume's home part first, though it has more streams
    p.close(2);
    _ = try p.openWith(2, 300, 100, 0, &.{});
    try expectEqual(@as(u8, 0), f.st(2).hold.?.owner);
    try p.giveTwin(1);
    try expectEqual(@as(usize, 0), p.twinCount(1));
    try expectEqual(kvs.Hold{ .owner = 0 }, f.st(1).hold.?);
    p.close(1);
    p.close(2);
    try expectEqual(@as(usize, 8), x.parts.pools[0].available());
    try expectEqual(@as(usize, 8), x.parts.pools[1].available());
}

test "kv shard pages: a request past a part is admitted at once and refused, never left waiting" {
    const gpa = std.testing.allocator;
    var f: Fake = undefined;
    Fake.reset();
    try f.init(gpa, 0, true);
    const p = try f.pool(2, 14, true); // parts of 5 pages
    defer f.deinit(p);
    try expect(p.fits(2900, 100)); // 6 pages
    try std.testing.expectError(error.PoolTooSmall, p.openWith(1, 2900, 100, null, &.{}));
    try expect(!p.used[1] and p.shard.?.parts.pools[0].available() == 5);
}

test "kv shard pages: a follower maps rank 0's ids and twin, its latent table on the twin, the digests alike" {
    const gpa = std.testing.allocator;
    var lead: Fake = undefined;
    Fake.reset();
    try lead.init(gpa, 0, true);
    const p0 = try lead.pool(3, 20, true);
    defer lead.deinit(p0);
    const wire = try gpa.dupe(u32, try p0.openWith(1, 900, 100, null, &.{}));
    defer gpa.free(wire);
    const twin = p0.twinCount(1);
    try expectEqual(@as(usize, 2), twin);
    var other: Fake = undefined;
    try other.init(gpa, 1, true);
    const p1 = try other.pool(3, 20, false);
    defer other.deinit(p1);
    try p1.mapWith(1, wire, twin, 0);
    const s = other.st(1);
    try expectEqualSlices(u32, wire[0..2], F.table(s.table, 2));
    try expectEqualSlices(u32, &.{ 0, 1 }, F.table(s.ltable, 2)); // twin pages 8 and 9: part 1's first two, part-local
    try expect(s.hold.?.has(0) and s.hold.?.has(1));
    try expectEqual(p0.digest(), p1.digest());
    try p1.giveTwin(1);
    try expectEqualSlices(u32, &.{ 8, 8 }, F.table(s.ltable, 2)); // the trash page: rank 0 alone holds its rows
    try expect(p0.digest() != p1.digest());
    try p0.giveTwin(1);
    try expectEqual(p0.digest(), p1.digest());
    try std.testing.expectError(error.BadStep, p1.mapWith(2, wire, 5, 0)); // a twin longer than the step
    try std.testing.expectError(error.KvShardPlacement, p1.mapWith(2, &.{ 1, 9 }, 0, 0)); // across the parts
    try std.testing.expect(!p1.used[2]);
}

test "kv shard pages: a twin the other part has no room for is refused and counted, the stream placed alone" {
    const gpa = std.testing.allocator;
    var f: Fake = undefined;
    Fake.reset();
    try f.init(gpa, 0, true);
    const p = try f.pool(3, 20, true);
    defer f.deinit(p);
    _ = try p.openWith(1, 3000, 100, 1, &.{}); // 7 pages of part 1, a twin of 6 in part 0
    try expectEqual(@as(usize, 6), p.twinCount(1));
    const wire = try p.openWith(2, 600, 100, null, &.{}); // 2 pages: part 0 alone has them; part 1 has 1 left for a twin of 2
    try expectEqual(@as(usize, 2), wire.len);
    try expectEqual(@as(usize, 0), kvs.partOf(wire[0], 8));
    try expectEqual(@as(usize, 0), p.twinCount(2));
    try expectEqual(kvs.Hold{ .owner = 0 }, f.st(2).hold.?);
    try expectEqual(@as(u64, 1), p.shard.?.parts.counts.twins_refused);
    try expect(!p.fits(1500, 100)); // 4 pages: neither part has them
}

test "kv shard pages: TF_KV_POISON fills a stream's key pages and this rank's latent pages, never the trash page" {
    const gpa = std.testing.allocator;
    var f: Fake = undefined;
    Fake.reset();
    try f.init(gpa, 1, true);
    const p = try f.pool(2, 20, false);
    defer f.deinit(p);
    p.poison = true;
    try p.mapWith(1, &.{ 0, 1, 2 }, 0, 0); // part 0's stream on rank 1: keys only
    try expectEqual(@as(usize, 3), F.fills);
    p.close(1);
    try p.mapWith(1, &.{ 9, 10 }, 0, 0); // its own part: keys and latent pages
    try expectEqual(@as(usize, 3 + 4), F.fills);
    p.close(1);
    try p.mapWith(1, &.{ 0, 1, 2, 9, 11 }, 2, 0); // part 0's stream with a twin here: its keys and the twin's pages
    try expectEqual(@as(usize, 7 + 3 + 2), F.fills);
    try p.giveTwin(1); // the twin given back: the rows stay, nothing filled again
    try expectEqual(@as(usize, 12), F.fills);
}

test "kv shard pages: a kept state's plan walks the latent rows by the latent table, and without them on the other rank" {
    const gpa = std.testing.allocator;
    const layout = @import("cuda_snapshot_layout.zig");
    var f: Fake = undefined;
    Fake.reset();
    try f.init(gpa, 1, true);
    const p = try f.pool(3, 20, false);
    defer f.deinit(p);
    try p.mapWith(1, &.{ 12, 9, 14 }, 0, 0); // part 1's stream on rank 1: latent pages 4, 1, 6
    const st = f.st(1);
    const at = 1200;
    const full = try layout.Plan.initFor(st, &f.e.w.config, at, null, true, st.cur);
    const bare = try layout.Plan.initFor(st, &f.e.w.config, at, null, false, st.cur);
    try expectEqual(full.len - at * (512 + 4), bare.len); // one DSA layer's codes and scales
    try expectEqual(full.len, try layout.Plan.size(st, &f.e.w.config, st.capacity, at, null));
    var lat: usize = 0;
    for (full.parts[0..full.count]) |part| {
        if (!part.lat) continue;
        lat += 1;
        try expectEqual(@as(u64, 264_192), part.page);
        const x = p.shard.?;
        // row 600 lies on the stream's second page: latent page 1 of this rank's arena
        const span = part.span;
        const pc = full.piece(part, span + 5);
        try expectEqual(part.live + 1 * part.page + 5, pc.live);
        try expect(pc.live >= x.lat.?.ptr and pc.live < x.lat.?.ptr + 9 * 264_192);
    }
    try expectEqual(@as(usize, 2), lat); // codes, then scales
    for (bare.parts[0..bare.count]) |part| try expect(!part.lat);
    // the pooled keys' part walks the key pages by the stream's ids
    for (full.parts[0..full.count]) |part| if (part.span != 0 and !part.lat) {
        try expectEqual(@as(u64, 32_768), part.page);
        try expectEqual(part.live + 9 * part.page, full.piece(part, part.span).live);
    };
}

test "kv shard pages: a kept state's plan past the shared store's pages leaves their latent rows on the pages" {
    const gpa = std.testing.allocator;
    const layout = @import("cuda_snapshot_layout.zig");
    var f: Fake = undefined;
    Fake.reset();
    try f.init(gpa, 1, true);
    const p = try f.pool(3, 20, false);
    defer f.deinit(p);
    try p.mapWith(1, &.{ 12, 9, 14 }, 0, 0);
    const st = f.st(1);
    const at = 1200;
    const whole = try layout.Plan.initForShared(st, &f.e.w.config, at, null, 0, true, st.cur);
    const past = try layout.Plan.initForShared(st, &f.e.w.config, at, null, 2, true, st.cur); // the store holds 2 pages
    // the two pages' 1,024 rows of codes and scales stay on the pages, with their pooled keys
    try expect(whole.len - past.len >= 1024 * (512 + 4));
    var lat: usize = 0;
    for (past.parts[0..past.count]) |part| {
        if (!part.lat) continue;
        lat += 1;
        try expectEqual(2 * part.span, part.from);
        try expectEqual((at - 1024) * (part.span / 512), part.len);
        // the first byte it copies is row 1,024: the stream's third page (latent page 6 of this rank's arena)
        try expectEqual(part.live + 6 * part.page, past.piece(part, 0).live);
    }
    try expectEqual(@as(usize, 2), lat);
    // a state wholly on the store's pages copies no latent row, sized or planned
    const grid = try layout.Plan.sizeForShared(st, &f.e.w.config, st.capacity, 1024, null, 2, true, st.cur);
    const bare = try layout.Plan.sizeForShared(st, &f.e.w.config, st.capacity, 1024, null, 2, false, st.cur);
    try expectEqual(bare, grid);
}

test "kv shard pages: a stream resuming a kept state is seated in the state's home part, the lanes' view of it" {
    const gpa = std.testing.allocator;
    const Lanes = @import("cuda_lanes.zig").Lanes;
    const lanes = @import("lanes");
    var f: Fake = undefined;
    Fake.reset();
    try f.init(gpa, 0, true);
    f.e.snaps = .empty;
    defer f.e.snaps.deinit(gpa);
    const p = try f.pool(3, 20, true);
    defer f.deinit(p);
    try f.e.snaps.put(gpa, 7, .{ .lease = .{}, .len = 0, .pos = 4096, .mtp_len = 0, .cur = &.{}, .bases = .{ 0, 0 }, .home = 1 });
    var ln: Lanes = .{ .x = .{ .e = &f.e, .link = undefined } };
    defer ln.lanes.deinit(gpa);
    var kept: types.Kept = .{ .id = 7, .at = 4096 };
    var streams: [2]lanes.Stream = undefined;
    streams[0].reuse = .{ .saved = &kept, .at = 4096 };
    streams[1].reuse = .{};
    _ = try ln.seat(&streams[0], 600, 100, &.{}); // both parts empty: the tie would pick part 0, its home is part 1
    try expectEqual(@as(u8, 1), f.st(ln.lanes.get(&streams[0]).?.slot).hold.?.owner);
    _ = try ln.seat(&streams[1], 600, 100, &.{}); // no kept state: the part with fewer live streams
    try expectEqual(@as(u8, 0), f.st(ln.lanes.get(&streams[1]).?.slot).hold.?.owner);
}

test "kv shard pages: a share resume seats in its adopted pages' part, no twin, and waits there when it lacks room" {
    const gpa = std.testing.allocator;
    const Lanes = @import("cuda_lanes.zig").Lanes;
    const lanes = @import("lanes");
    const kv = @import("core").kv_pool;
    var f: Fake = undefined;
    Fake.reset();
    try f.init(gpa, 0, true);
    const p = try f.pool(4, 20, true); // two parts of 8
    defer f.deinit(p);
    const x = p.shard.?;
    // the store's two pages in part 1 (global 8, 9): a keep left them, a begin holds them once more
    var lease: kv.Lease = .{};
    try x.parts.pools[1].promise(&lease, 2);
    try x.parts.pools[1].cover(&lease, 2, null);
    const adopted = [_]u32{ lease.ids.items[0] + 8, lease.ids.items[1] + 8 };
    for (adopted) |id| {
        x.parts.retain(id); // the tree's own hold on the page
        x.parts.retain(id); // the begin's hold for the resume it offered the pages to
    }
    x.parts.pools[1].finish(&lease); // the stream's lease goes, the store's two holds stay
    var ln: Lanes = .{ .x = .{ .e = &f.e, .link = undefined } };
    defer ln.lanes.deinit(gpa);
    var s: lanes.Stream = undefined;
    s.reuse = .{};
    _ = try ln.seat(&s, 600, 100, &adopted); // part 0 has more free: the adopted pages' part wins
    const slot = ln.lanes.get(&s).?.slot;
    try expectEqual(@as(u8, 1), f.st(slot).hold.?.owner);
    try expectEqualSlices(u32, &adopted, f.st(slot).pages[0..2]);
    for (adopted) |id| try expectEqual(@as(u32, 2), x.parts.holders(id)); // the tree's and the lease's
    try expectEqual(@as(usize, 0), p.twinCount(slot)); // no twin: the pass runs owner-computed
    p.close(slot);
    for (adopted) |id| try expectEqual(@as(u32, 1), x.parts.holders(id)); // the lease's hold went, the tree's stays
    // part 1 full of other streams' pages: the resume waits there (LaneMemory), it never moves to part 0
    var full: kv.Lease = .{};
    try x.parts.pools[1].promise(&full, 6);
    try x.parts.pools[1].cover(&full, 6, null);
    for (adopted) |id| x.parts.retain(id); // a fresh begin's hold for this resume
    var w: lanes.Stream = undefined;
    w.reuse = .{};
    try std.testing.expectError(error.LaneMemory, ln.seat(&w, 1500, 100, &adopted)); // 4 pages: 2 fresh past the adopted
    for (adopted) |id| try expectEqual(@as(u32, 1), x.parts.holders(id)); // the failed seat gave the begin's hold back
    x.parts.pools[1].finish(&full);
}

test "kv shard pages: the follower maps a begin whose leading ids are the store's, and the digests agree" {
    const gpa = std.testing.allocator;
    const kv = @import("core").kv_pool;
    var lead: Fake = undefined;
    Fake.reset();
    try lead.init(gpa, 0, true);
    const p0 = try lead.pool(3, 20, true); // two parts of 8
    defer lead.deinit(p0);
    const x0 = p0.shard.?;
    // the store's two pages in part 1, a begin's holds beside the tree's own (a keep left them)
    var lease: kv.Lease = .{};
    try x0.parts.pools[1].promise(&lease, 2);
    try x0.parts.pools[1].cover(&lease, 2, null);
    const adopted = [_]u32{ lease.ids.items[0] + 8, lease.ids.items[1] + 8 };
    for (adopted) |id| {
        x0.parts.retain(id);
        x0.parts.retain(id);
    }
    x0.parts.pools[1].finish(&lease);
    const wire = try gpa.dupe(u32, try p0.openWith(1, 1500, 100, null, &adopted)); // 4 pages in part 1, no twin
    defer gpa.free(wire);
    try expectEqualSlices(u32, &adopted, wire[0..2]);
    try expectEqual(@as(usize, 0), p0.twinCount(1));
    var other: Fake = undefined;
    try other.init(gpa, 1, true);
    const p1 = try other.pool(3, 20, false);
    defer other.deinit(p1);
    try p1.mapWith(1, wire, 0, adopted.len); // the begin's ids, the store's first: mapped, not poisoned
    const s = other.st(1);
    try expectEqualSlices(u32, wire, F.table(s.table, wire.len));
    try expectEqual(kvs.Hold{ .owner = 1 }, s.hold.?); // owner-computed pass: rank 0 alone holds the rows
    try expectEqual(p0.digest(), p1.digest()); // the adopted pages are in the slot's ids, so in the digest
}

test "kv shard pages: a share resume whose untwinned tail does not pay runs cold, the kept state stays; one that pays adopts" {
    const gpa = std.testing.allocator;
    const Lanes = @import("cuda_lanes.zig").Lanes;
    const lanes = @import("lanes");
    const kv = @import("core").kv_pool;
    var f: Fake = undefined;
    Fake.reset();
    try f.init(gpa, 0, true);
    const p = try f.pool(4, 20, true); // two parts of 8
    defer f.deinit(p);
    const x = p.shard.?;
    // the store's page in part 1 (global 8): a keep left it, a begin holds it once more for the resume it offered
    var lease: kv.Lease = .{};
    try x.parts.pools[1].promise(&lease, 1);
    try x.parts.pools[1].cover(&lease, 1, null);
    const page = lease.ids.items[0] + 8;
    x.parts.retain(page); // the tree's own hold on the page
    x.parts.pools[1].finish(&lease); // the stream's lease goes, the tree's and the begins' holds stay
    var ln: Lanes = .{ .x = .{ .e = &f.e, .link = undefined } };
    defer ln.lanes.deinit(gpa);
    var adopt: std.ArrayList(u32) = .empty;
    defer adopt.deinit(gpa);
    var kept: types.Kept = .{ .id = 7, .at = 4096 };
    var owner: share_mod.Owner = .{ .pending = &kept };

    // 4,096 kept against a 126,407-token prompt: the tail would not pay, the resume is skipped and counted
    x.parts.retain(page); // the begin's hold for the resume
    try adopt.append(gpa, page);
    const cold = share_mod.gate(&f.e, &owner, &adopt, 126407); // the kept prefix's part, remembered for the seat
    try expectEqual(@as(usize, 0), adopt.items.len);
    try expect(owner.pending == null);
    try expectEqual(@as(u64, 1), x.parts.counts.resumes_skipped);
    try expectEqual(@as(u32, 1), x.parts.holders(page)); // the tree's hold: the kept state is still in the store
    try expectEqual(@as(u8, 1), cold.?);
    var s: lanes.Stream = undefined;
    s.reuse = .{};
    _ = try ln.seatCold(&s, 600, 100, adopt.items, cold); // the cold pass: the kept prefix's part, its twin borrowed
    const slot = ln.lanes.get(&s).?.slot;
    try expectEqual(@as(u8, 1), f.st(slot).hold.?.owner); // the prefix's part, not the parts' free choice
    try expect(p.twinCount(slot) > 0);
    p.close(slot);

    // 122,880 kept against the same prompt: the tail pays, the resume adopts with no twin
    x.parts.retain(page); // a fresh begin's hold
    var kept2: types.Kept = .{ .id = 7, .at = 122880 };
    owner = .{ .pending = &kept2 };
    try adopt.append(gpa, page);
    try expectEqual(@as(?u8, null), share_mod.gate(&f.e, &owner, &adopt, 126407)); // adopting: no cold home
    try expectEqual(@as(usize, 1), adopt.items.len);
    try expect(owner.pending == &kept2);
    try expectEqual(@as(u64, 1), x.parts.counts.resumes_skipped); // the adopted one is not a skip
    var s2: lanes.Stream = undefined;
    s2.reuse = .{};
    _ = try ln.seat(&s2, 600, 100, adopt.items); // the resume: its adopted pages' part, no twin
    const slot2 = ln.lanes.get(&s2).?.slot;
    try expectEqual(@as(u8, 1), f.st(slot2).hold.?.owner);
    try expectEqualSlices(u32, &.{page}, f.st(slot2).pages[0..1]);
    try expectEqual(@as(usize, 0), p.twinCount(slot2));
    try expectEqual(@as(u32, 2), x.parts.holders(page)); // the tree's and the lease's
    p.close(slot2);

    // at 100 the short-kept resume adopts too
    f.e.settings.share_untwinned_pct = 100;
    x.parts.retain(page); // a fresh begin's hold
    var kept3: types.Kept = .{ .id = 7, .at = 4096 };
    owner = .{ .pending = &kept3 };
    try expectEqual(@as(?u8, null), share_mod.gate(&f.e, &owner, &adopt, 126407));
    try expectEqual(@as(usize, 1), adopt.items.len);
    try expect(owner.pending == &kept3);
    try expectEqual(@as(u64, 1), x.parts.counts.resumes_skipped);
    var s3: lanes.Stream = undefined;
    s3.reuse = .{};
    _ = try ln.seat(&s3, 600, 100, adopt.items);
    try expectEqual(@as(u8, 1), f.st(ln.lanes.get(&s3).?.slot).hold.?.owner);
    try expectEqual(@as(usize, 0), p.twinCount(ln.lanes.get(&s3).?.slot));
    p.close(ln.lanes.get(&s3).?.slot);
}

test "kv shard pages: an unsharded share resume adopts always (the gate leaves the begin's pages alone)" {
    const gpa = std.testing.allocator;
    var f: Fake = undefined;
    Fake.reset();
    try f.init(gpa, 0, false); // unsharded
    const p = try f.pool(4, 20, true);
    defer f.deinit(p);
    var kept: types.Kept = .{ .id = 7, .at = 4096 };
    var owner: share_mod.Owner = .{ .pending = &kept };
    var adopt: std.ArrayList(u32) = .empty;
    defer adopt.deinit(gpa);
    try adopt.append(gpa, 3); // a page id nobody holds: the gate must not touch it
    _ = share_mod.gate(&f.e, &owner, &adopt, 126407);
    try expectEqual(@as(usize, 1), adopt.items.len);
    try expect(owner.pending == &kept);
}

/// The shared store over the fake's sharded parts, as native/cuda_prompt_pages.zig wires it on rank 0
/// (TF_GLM_KV_SHARD): the parts the tree's pages, a kept state's path never spanning them.
const Stored = struct {
    gpa: std.mem.Allocator,
    e: *Engine,
    src: kp.Parts,
    store: radix.Store,
    part: [2]Part = undefined,
    live: usize = 0,
    next: u64 = 1,

    /// Each part's reclaim hook (the adapter's): the store evicts this part's idle leaves until `want` free there.
    const Part = struct { x: *Stored, at: usize };

    fn of(ctx: *anyopaque) *Stored {
        return @ptrCast(@alignCast(ctx));
    }

    fn open(x: *Stored, gpa: std.mem.Allocator, e: *Engine, snaps: usize) !void {
        const src = native.share_pool(e).?.parts;
        const rules: radix.Rules = .{ .page = state.page_rows, .planned = true, .grid = native.share_grid, .lookahead = 1, .min_prompt = 0, .min_gap = 256, .tail = true };
        x.* = .{ .gpa = gpa, .e = e, .src = src, .store = undefined };
        x.store = try radix.Store.init(gpa, .{ .ptr = x, .vtable = &Family.vtable }, .{ .ptr = &x.src, .bytes = src.page_bytes, .vtable = &PagesV.vtable }, rules);
        x.store.parts = .{ .ctx = x, .one = Parting.one, .refused = Parting.refused }; // a kept state's path stays in one part
        for (0..2) |i| { // a part's promise evicts its own leaves first, as the adapter wires it
            x.part[i] = .{ .x = x, .at = i };
            src.setReclaim(src.ctx, i, .{ .ctx = &x.part[i], .call = Reclaim.part });
        }
        x.store.limit(.{ .pages = src.total, .snaps = snaps });
        native.share_attach(e, .{ .ctx = x, .vtable = &vtable });
    }

    fn close(x: *Stored, e: *Engine) void {
        native.share_attach(e, null);
        for (0..2) |i| x.src.setReclaim(x.src.ctx, i, null);
        x.store.deinit();
    }

    const Reclaim = struct {
        fn part(ctx: *anyopaque, want: usize) bool {
            const h: *Part = @ptrCast(@alignCast(ctx));
            return h.x.store.reclaim(want, .{ .ctx = h, .f = leafIn, .free = freeIn });
        }
        fn leafIn(ctx: *anyopaque, first: u32) bool {
            const h: *Part = @ptrCast(@alignCast(ctx));
            return h.x.src.inPart(h.x.src.ctx, h.at, first);
        }
        fn freeIn(ctx: *anyopaque) usize {
            const h: *Part = @ptrCast(@alignCast(ctx));
            return h.x.src.free(h.x.src.ctx, h.at);
        }
    };

    /// The states: `types.Kept`s (a resume's owner reads their `at` through the family's restore).
    const Family = struct {
        fn bytes(_: *anyopaque, _: u32) u64 {
            return 1;
        }
        fn save(p: *anyopaque, _: ?*anyopaque, at: u32) anyerror!*anyopaque {
            const x = of(p);
            const k = try x.gpa.create(types.Kept);
            k.* = .{ .id = x.next, .at = at };
            x.next += 1;
            x.live += 1;
            return k;
        }
        fn drop(p: *anyopaque, saved: *anyopaque) void {
            const x = of(p);
            x.gpa.destroy(@as(*types.Kept, @ptrCast(@alignCast(saved))));
            x.live -= 1;
        }
        fn restore(p: *anyopaque, owner: ?*anyopaque, saved: *anyopaque) anyerror!void {
            return native.snap_restore(of(p).e, owner, saved); // the resume waits in the share's owner, as ever
        }
        const vtable: radix.Snapshots.VTable = .{ .bytes = bytes, .save = save, .restore = restore, .drop = drop };
    };

    /// The tree's pages over the parts: each call through the family's own (hooks.Parts', ctx its parts).
    const PagesV = struct {
        fn parts(ptr: *anyopaque) *const kp.Parts {
            return @ptrCast(@alignCast(ptr));
        }
        fn retain(ptr: *anyopaque, id: u32) void {
            const s = parts(ptr);
            s.retain(s.ctx, id);
        }
        fn release(ptr: *anyopaque, id: u32) void {
            const s = parts(ptr);
            s.release(s.ctx, id);
        }
        fn holders(ptr: *anyopaque, id: u32) u32 {
            const s = parts(ptr);
            return s.holders(s.ctx, id);
        }
        fn available(ptr: *anyopaque) usize {
            const s = parts(ptr);
            return s.available(s.ctx);
        }
        const vtable: radix.Pages.VTable = .{ .retain = retain, .release = release, .holders = holders, .available = available };
    };

    /// The parts as the store's keep sees them: one part or refused, the refusal the family counts.
    const Parting = struct {
        fn one(ptr: *anyopaque, ids: []const u32) bool {
            const x = of(ptr);
            return x.src.onePart(x.src.ctx, ids);
        }
        fn refused(ptr: *anyopaque) void {
            const x = of(ptr);
            x.src.refused(x.src.ctx);
        }
    };

    fn begin(ctx: *anyopaque, prompt: []const u32, history_len: u32, shared: []const u32, owner: *anyopaque, adopt: *std.ArrayList(u32)) anyerror!kp.Plan {
        const x = of(ctx);
        const plan = try x.store.begin(x.gpa, prompt, history_len, shared, owner, adopt);
        return .{ .from = plan.from, .marks = plan.marks };
    }
    fn keep(ctx: *anyopaque, prompt: []const u32, at: u32, owner: *anyopaque, mine: []const u32, path: []u32) usize {
        return of(ctx).store.keep(prompt, at, owner, mine, path).shared;
    }
    fn report(_: *anyopaque, _: usize, _: u32, _: u64) void {}
    fn forget(ctx: *anyopaque, prompt: []const u32, at: u32) void {
        of(ctx).store.forget(prompt, at);
    }
    fn free(ctx: *anyopaque, marks: []const u32) void {
        of(ctx).gpa.free(marks);
    }
    const vtable: kp.Prompts.VTable = .{ .begin = begin, .keep = keep, .report = report, .forget = forget, .free = free };
};

test "kv shard pages: the GPU's sequence — a skipped resume seats at its prefix's part, its keep and the next resume stay in one part, the follower maps it" {
    const gpa = std.testing.allocator;
    const Lanes = @import("cuda_lanes.zig").Lanes;
    const lanes = @import("lanes");
    const kv = @import("core").kv_pool;
    var f: Fake = undefined;
    Fake.reset();
    try f.init(gpa, 0, true);
    f.e.capacity = 262144; // the 126k prompt's table
    const p = try f.pool(4, 520, true); // two parts of 258
    defer f.deinit(p);
    const x = p.shard.?;
    var st: Stored = undefined;
    try st.open(gpa, &f.e, 8);
    defer st.close(&f.e);
    const prompts = share_mod.of(&f.e).?;
    const big = try gpa.alloc(u32, 126407);
    defer gpa.free(big);
    for (big, 1..) |*t, i| t.* = @intCast(i);
    const big2 = try gpa.alloc(u32, 126408);
    defer gpa.free(big2);
    @memcpy(big2[0..126407], big);
    big2[126407] = 999_999;

    // the 4,096-token state kept in part A: its 8 pages the tree holds
    var owner: share_mod.Owner = .{};
    var lease: kv.Lease = .{};
    try x.parts.pools[0].promise(&lease, 8);
    try x.parts.pools[0].cover(&lease, 8, null);
    var path0: [8]u32 = undefined;
    try expect(st.store.keep(big, 4096, &owner, lease.ids.items, &path0).held);
    x.parts.pools[0].finish(&lease);
    for (path0) |id| try expectEqual(@as(u32, 1), x.parts.holders(id));

    // the 126,407-token prompt: the resume adopts, the gate skips it (the tail would not pay) and remembers the part
    var adopt: std.ArrayList(u32) = .empty;
    defer adopt.deinit(gpa);
    const plan = try prompts.begin(big, 0, &.{}, &owner, &adopt);
    defer prompts.free(plan.marks);
    try expectEqual(@as(u32, 4096), plan.from);
    try expectEqual(@as(usize, 8), adopt.items.len);
    const cold = share_mod.gate(&f.e, &owner, &adopt, 126407);
    try expectEqual(@as(u8, 0), cold.?); // the kept prefix's part, before the pages go back
    try expectEqual(@as(usize, 0), adopt.items.len);
    try expect(owner.pending == null);
    try expectEqual(@as(u64, 1), x.parts.counts.resumes_skipped);
    for (path0) |id| try expectEqual(@as(u32, 1), x.parts.holders(id)); // the tree's keep it resumable

    // the cold pass seats in that part (a kept state's home is today), its twin borrowed
    var ln: Lanes = .{ .x = .{ .e = &f.e, .link = undefined } };
    defer ln.lanes.deinit(gpa);
    var s: lanes.Stream = undefined;
    s.reuse = .{};
    _ = try ln.seatCold(&s, 600, 100, adopt.items, cold);
    const slot = ln.lanes.get(&s).?.slot;
    try expectEqual(@as(u8, 0), f.st(slot).hold.?.owner);
    try expect(p.twinCount(slot) > 0);
    p.close(slot);

    // kept at 122,880: the tail's pages join the prefix's under one node, all part A
    var tail: kv.Lease = .{};
    try x.parts.pools[0].promise(&tail, 240);
    try x.parts.pools[0].cover(&tail, 240, null);
    var path1: [240]u32 = undefined;
    try expect(st.store.keep(big, 122880, &owner, tail.ids.items, &path1).held);
    x.parts.pools[0].finish(&tail);
    try expectEqual(@as(u64, 0), x.parts.counts.keeps_refused);
    for (path1) |id| {
        try expectEqual(@as(usize, 0), kvs.partOf(id, x.parts.h));
        try expectEqual(@as(u32, 1), x.parts.holders(id));
    }

    // the next request with the prefix: the kept state at 122,880 adopts, its tail pays
    var owner2: share_mod.Owner = .{};
    var adopt2: std.ArrayList(u32) = .empty;
    defer adopt2.deinit(gpa);
    const plan2 = try prompts.begin(big2, 0, &.{}, &owner2, &adopt2);
    defer prompts.free(plan2.marks);
    try expectEqual(@as(u32, 122880), plan2.from);
    try expectEqual(@as(usize, 240), adopt2.items.len);
    try expect(owner2.pending != null);
    try expectEqual(@as(?u8, null), share_mod.gate(&f.e, &owner2, &adopt2, 126408));
    try expectEqual(@as(usize, 240), adopt2.items.len);

    // the resume opens in the adopted pages' part, the follower's mapping accepts it, the digests agree
    const wire = try gpa.dupe(u32, try p.openWith(1, 126408, 100, null, adopt2.items));
    defer gpa.free(wire);
    try expectEqualSlices(u32, adopt2.items, wire[0..240]); // the store's pages first
    try expectEqual(@as(usize, 0), p.twinCount(1)); // no twin: the pass runs owner-computed
    var other: Fake = undefined;
    try other.init(gpa, 1, true);
    other.e.capacity = 262144; // the follower maps the same 126k begin step
    const p1 = try other.pool(4, 520, false);
    defer other.deinit(p1);
    try p1.mapWith(1, wire, 0, adopt2.items.len);
    try expectEqual(p.digest(), p1.digest());
    p.close(1);
}

test "kv shard pages: part A full, the skipped resume's cold pass seats in part B and its keep replaces the part-A prefix" {
    const gpa = std.testing.allocator;
    const Lanes = @import("cuda_lanes.zig").Lanes;
    const lanes = @import("lanes");
    const kv = @import("core").kv_pool;
    var f: Fake = undefined;
    Fake.reset();
    try f.init(gpa, 0, true);
    f.e.capacity = 262144;
    const p = try f.pool(4, 520, true); // two parts of 258
    defer f.deinit(p);
    const x = p.shard.?;
    var st: Stored = undefined;
    try st.open(gpa, &f.e, 8);
    defer st.close(&f.e);
    const prompts = share_mod.of(&f.e).?;
    const big = try gpa.alloc(u32, 126407);
    defer gpa.free(big);
    for (big, 1..) |*t, i| t.* = @intCast(i);
    const big2 = try gpa.alloc(u32, 126408);
    defer gpa.free(big2);
    @memcpy(big2[0..126407], big);
    big2[126407] = 999_999;

    // the 4,096-token state kept in part A, then part A full of another pass's pages
    var owner: share_mod.Owner = .{};
    var lease: kv.Lease = .{};
    try x.parts.pools[0].promise(&lease, 8);
    try x.parts.pools[0].cover(&lease, 8, null);
    var path0: [8]u32 = undefined;
    try expect(st.store.keep(big, 4096, &owner, lease.ids.items, &path0).held);
    x.parts.pools[0].finish(&lease);
    var full: kv.Lease = .{};
    try x.parts.pools[0].promise(&full, 250);
    try x.parts.pools[0].cover(&full, 250, null);

    // the 126,407-token prompt skipped: the cold pass needs more pages than the prefix's idle ones free, so the
    // part cannot seat it even by reclaiming them — it takes part B
    var adopt: std.ArrayList(u32) = .empty;
    defer adopt.deinit(gpa);
    const plan = try prompts.begin(big, 0, &.{}, &owner, &adopt);
    defer prompts.free(plan.marks);
    try expectEqual(@as(u32, 4096), plan.from);
    const cold = share_mod.gate(&f.e, &owner, &adopt, 126407);
    try expectEqual(@as(u8, 0), cold.?);
    try expectEqual(@as(u64, 1), x.parts.counts.resumes_skipped);
    var ln: Lanes = .{ .x = .{ .e = &f.e, .link = undefined } };
    defer ln.lanes.deinit(gpa);
    var s: lanes.Stream = undefined;
    s.reuse = .{};
    _ = try ln.seatCold(&s, 6000, 100, adopt.items, cold); // the parts choose: part A has no room for the pass
    const slot = ln.lanes.get(&s).?.slot;
    try expectEqual(@as(u8, 1), f.st(slot).hold.?.owner);
    p.close(slot);

    // its keep at 122,880: the tree's part-A prefix would span the parts, and the pass is longer and newer —
    // the prefix chain has no other child, so it is replaced whole: the new path all part B
    var tail: kv.Lease = .{};
    try x.parts.pools[1].promise(&tail, 240);
    try x.parts.pools[1].cover(&tail, 240, null);
    var mine: [240]u32 = undefined;
    for (tail.ids.items, 0..) |id, i| mine[i] = id + 258; // part B's global ids
    var path1: [240]u32 = undefined;
    try expect(st.store.keep(big, 122880, &owner, &mine, &path1).held);
    x.parts.pools[1].finish(&tail);
    x.parts.pools[0].finish(&full);
    try expectEqual(@as(u64, 0), x.parts.counts.keeps_refused);
    for (path0) |id| try expectEqual(@as(u32, 0), x.parts.holders(id)); // the replaced chain's pages went back
    for (path1) |id| {
        try expectEqual(@as(usize, 1), kvs.partOf(id, x.parts.h));
        try expectEqual(@as(u32, 1), x.parts.holders(id));
    }
    try expectEqual(@as(usize, 1), st.live); // the 4,096 state went with its chain

    // the store's one state resumes whole
    var owner2: share_mod.Owner = .{};
    var adopt2: std.ArrayList(u32) = .empty;
    defer adopt2.deinit(gpa);
    const plan2 = try prompts.begin(big2, 0, &.{}, &owner2, &adopt2);
    defer prompts.free(plan2.marks);
    try expectEqual(@as(u32, 122880), plan2.from);
    try expectEqual(@as(usize, 240), adopt2.items.len);
    for (adopt2.items) |id| try expectEqual(@as(usize, 1), kvs.partOf(id, x.parts.h));
    for (adopt2.items) |id| x.parts.release(id); // the begin's holds go back, the tree's stay
}

test "kv shard pages: a keep under a branched prefix of the other part is refused and counted, the tree unchanged" {
    const gpa = std.testing.allocator;
    const kv = @import("core").kv_pool;
    var f: Fake = undefined;
    Fake.reset();
    try f.init(gpa, 0, true);
    f.e.capacity = 262144;
    const p = try f.pool(4, 520, true); // two parts of 258
    defer f.deinit(p);
    const x = p.shard.?;
    var st: Stored = undefined;
    try st.open(gpa, &f.e, 8);
    defer st.close(&f.e);
    const prompts = share_mod.of(&f.e).?;
    const big = try gpa.alloc(u32, 126407);
    defer gpa.free(big);
    for (big, 1..) |*t, i| t.* = @intCast(i);

    // the 4,096-token state kept in part A, and a longer one below it in part A: the prefix node has another child
    var owner: share_mod.Owner = .{};
    var lease: kv.Lease = .{};
    try x.parts.pools[0].promise(&lease, 8);
    try x.parts.pools[0].cover(&lease, 8, null);
    var path0: [8]u32 = undefined;
    try expect(st.store.keep(big, 4096, &owner, lease.ids.items, &path0).held);
    x.parts.pools[0].finish(&lease);
    const branched = try gpa.alloc(u32, 8193);
    defer gpa.free(branched);
    @memcpy(branched, big[0..8193]);
    for (branched[4096..]) |*t| t.* += 1_000_000; // it parts from the 4,096 prefix's next page
    var lease1: kv.Lease = .{};
    try x.parts.pools[0].promise(&lease1, 16);
    try x.parts.pools[0].cover(&lease1, 16, null);
    var path1: [16]u32 = undefined;
    try expect(st.store.keep(branched, 8192, &owner, lease1.ids.items, &path1).held);
    x.parts.pools[0].finish(&lease1);

    // a part-B pass over the same 4,096 prefix, longer: its pages would span the parts, the prefix node's other
    // child stands, so the keep is refused and counted, and nothing of the tree moves
    var tail: kv.Lease = .{};
    try x.parts.pools[1].promise(&tail, 240);
    try x.parts.pools[1].cover(&tail, 240, null);
    var mine: [240]u32 = undefined;
    for (tail.ids.items, 0..) |id, i| mine[i] = id + 258;
    var path2: [240]u32 = undefined;
    const refused = st.store.keep(big, 122880, &owner, &mine, &path2);
    try expect(!refused.held);
    try expectEqual(@as(usize, 8), refused.shared); // the pages the tree already held
    x.parts.pools[1].finish(&tail);
    try expectEqual(@as(u64, 1), x.parts.counts.keeps_refused);
    try expectEqual(@as(usize, 2), st.live);
    for (path0) |id| try expectEqual(@as(u32, 1), x.parts.holders(id));
    for (path1[8..]) |id| try expectEqual(@as(u32, 1), x.parts.holders(id));

    // both kept states still resume
    var adopt: std.ArrayList(u32) = .empty;
    defer adopt.deinit(gpa);
    const plan = try prompts.begin(branched, 0, &.{}, &owner, &adopt);
    defer prompts.free(plan.marks);
    try expectEqual(@as(u32, 8192), plan.from);
    try expectEqual(@as(usize, 16), adopt.items.len);
    for (adopt.items) |id| x.parts.release(id);
    adopt.clearRetainingCapacity();
    const plan2 = try prompts.begin(big, 0, &.{}, &owner, &adopt);
    defer prompts.free(plan2.marks);
    try expectEqual(@as(u32, 4096), plan2.from);
    try expectEqual(@as(usize, 8), adopt.items.len);
    for (adopt.items) |id| x.parts.release(id);
}

test "kv shard pages: a hand-built cross-part kept state is not adopted by begin, it is evicted" {
    const gpa = std.testing.allocator;
    const lanes = @import("lanes");
    const kv = @import("core").kv_pool;
    var f: Fake = undefined;
    Fake.reset();
    try f.init(gpa, 0, true);
    const p = try f.pool(4, 20, true); // two parts of 8
    defer f.deinit(p);
    const x = p.shard.?;
    var st: Stored = undefined;
    try st.open(gpa, &f.e, 8);
    defer st.close(&f.e);
    const big = try gpa.alloc(u32, 8193);
    defer gpa.free(big);
    for (big, 1..) |*t, i| t.* = @intCast(i);

    // the bug's tree: the parts' check off, a keep whose pages span the parts goes in
    const check = st.store.parts;
    st.store.parts = null;
    var a: kv.Lease = .{};
    try x.parts.pools[0].promise(&a, 8);
    try x.parts.pools[0].cover(&a, 8, null);
    var b: kv.Lease = .{};
    try x.parts.pools[1].promise(&b, 8);
    try x.parts.pools[1].cover(&b, 8, null);
    var mine: [16]u32 = undefined;
    for (a.ids.items, 0..) |id, i| mine[i] = id; // part A: global == local
    for (b.ids.items, 0..) |id, i| mine[8 + i] = id + 8; // part B's
    var path: [16]u32 = undefined;
    var owner: share_mod.Owner = .{};
    try expect(st.store.keep(big, 8192, &owner, &mine, &path).held);
    x.parts.pools[0].finish(&a);
    x.parts.pools[1].finish(&b);
    st.store.parts = check;
    for (mine) |id| try expectEqual(@as(u32, 1), x.parts.holders(id)); // the tree's, spanning the parts

    // begin refuses it every build mode: nothing adopted, the state evicted, one error logged
    var s: lanes.Stream = undefined;
    s.reuse = .{};
    s.key = big;
    s.prompt_len = big.len;
    s.history_len = 0;
    s.shared_prefixes = &.{};
    var adopt: std.ArrayList(u32) = .empty;
    defer adopt.deinit(gpa);
    const plan = try share_mod.begin(&f.e, &s, &owner, &adopt);
    defer f.e.gpa.free(plan.marks);
    try expectEqual(@as(u64, 0), x.parts.counts.keeps_refused); // the keep's refusal, not this
    try expectEqual(@as(usize, 0), adopt.items.len);
    try expect(owner.pending == null);
    for (mine) |id| try expectEqual(@as(u32, 0), x.parts.holders(id)); // evicted: the tree's holds went
    try expectEqual(@as(usize, 0), st.live);

    // the next pass with the prefix runs cold, as it should
    var owner2: share_mod.Owner = .{};
    var adopt2: std.ArrayList(u32) = .empty;
    defer adopt2.deinit(gpa);
    const plan2 = try share_mod.begin(&f.e, &s, &owner2, &adopt2);
    defer f.e.gpa.free(plan2.marks);
    try expectEqual(@as(u32, 0), plan2.from);
    try expectEqual(@as(usize, 0), adopt2.items.len);
    try expect(owner2.pending == null);
}
