//! TF_GLM_KV_SHARD's kept states on a driver of addresses alone: a restore that skips its latent parts leaves their
//! bytes in place and every other part's offsets as they were; a carry's staging copy walks device pages; and the two
//! parts' leases over random admissions, twins and ends stay whole, placed alike on both ranks.
const std = @import("std");
const cuda = @import("cuda");
const snap_pages = @import("cuda_snapshot_pages.zig");
const layout = @import("cuda_snapshot_layout.zig");
const kvs = @import("cuda_kv_shard.zig");
const expect = std.testing.expect;
const expectEqual = std.testing.expectEqual;

const MiB = 1 << 20;

/// The driver: one arena at a fixed address, every device-to-device copy recorded.
const F = struct {
    const base: u64 = 1 << 40;
    var copies: [32][3]u64 = undefined; // (dst, src, bytes)
    var n: usize = 0;
    fn alloc(out: *u64, _: usize) callconv(.c) cuda.abi.Result {
        out.* = base;
        return 0;
    }
    fn free(_: u64) callconv(.c) cuda.abi.Result {
        return 0;
    }
    fn copy(dst: u64, src: u64, bytes: usize, _: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
        copies[n] = .{ dst, src, bytes };
        n += 1;
        return 0;
    }
};

fn driver() cuda.Driver {
    var d: cuda.Driver = undefined;
    d.api.cuMemAlloc_v2 = F.alloc;
    d.api.cuMemFree_v2 = F.free;
    d.api.cuMemcpyDtoDAsync_v2 = F.copy;
    return d;
}

test "kv shard keep: a restore without its latent parts copies the others from their own offsets" {
    const gpa = std.testing.allocator;
    var d = driver();
    var pool = try snap_pages.Pool.init(gpa, &d, 3 * snap_pages.page);
    defer pool.deinit();
    var plan: layout.Plan = .{};
    const live: [3]u64 = .{ 0x1000_0000, 0x2000_0000, 0x3000_0000 };
    plan.parts[0] = .{ .live = live[0], .len = 1 * MiB };
    plan.parts[1] = .{ .live = live[1], .len = 3 * MiB, .lat = true };
    plan.parts[2] = .{ .live = live[2], .len = MiB / 2 };
    plan.count = 3;
    plan.len = 4 * MiB + MiB / 2;
    var lease = try pool.take(plan.len);
    defer pool.give(&lease);
    const stream: cuda.Stream = .{ .d = &d, .handle = null };
    F.n = 0;
    try pool.moveParts(stream, &lease, &plan, false, false);
    try expectEqual(@as(usize, 2), F.n); // the first part, then the last: the latent part's 3 MiB skipped
    try expectEqual([3]u64{ live[0], F.base, MiB }, F.copies[0]);
    try expectEqual([3]u64{ live[2], F.base + 2 * snap_pages.page, MiB / 2 }, F.copies[1]); // still at byte 4 MiB
    F.n = 0;
    try pool.move(stream, &lease, &plan, false);
    try expectEqual(@as(usize, 4), F.n); // the latent part too, over its two pages
    try expectEqual([3]u64{ live[1], F.base + MiB, MiB }, F.copies[1]);
    try expectEqual([3]u64{ live[1] + MiB, F.base + snap_pages.page, 2 * MiB }, F.copies[2]);
    // a carry's staging copy: 1 MiB from byte 1.5 MiB, across a page's end
    F.n = 0;
    try pool.copyOut(stream, &lease, 3 * MiB / 2, 0x9000_0000, MiB);
    try expectEqual(@as(usize, 2), F.n);
    try expectEqual([3]u64{ 0x9000_0000, F.base + 3 * MiB / 2, MiB / 2 }, F.copies[0]);
    try expectEqual([3]u64{ 0x9000_0000 + MiB / 2, F.base + snap_pages.page, MiB / 2 }, F.copies[1]);
    try std.testing.expectError(error.SnapshotSize, pool.copyOut(stream, &lease, 5 * MiB, 0x9000_0000, 2 * MiB));
}

test "kv shard parts: random admissions, twins and ends on two ranks: every page held once, placed alike" {
    const gpa = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(0x0118);
    const rnd = prng.random();
    const h = 40;
    const slots = 6;
    var lead = try kvs.Parts.init(gpa, h, 4096, slots, true);
    defer lead.deinit();
    var other = try kvs.Parts.init(gpa, h, 4096, slots, false);
    defer other.deinit();
    for (0..3000) |_| {
        const slot = rnd.uintLessThan(usize, slots);
        const s = &lead.slots[slot];
        switch (rnd.uintLessThan(u8, 4)) {
            0, 1 => if (!s.used) {
                const n = rnd.intRangeAtMost(usize, 1, 16);
                const home: ?u8 = if (rnd.boolean()) rnd.uintLessThan(u8, 2) else null;
                const part = lead.choose(n, home) orelse continue;
                if (home) |hp| if (lead.pools[hp].fits(n)) try expectEqual(hp, part);
                try lead.open(slot, part, n, if (rnd.boolean()) rnd.intRangeAtMost(usize, 1, n) else 0);
                try other.map(slot, s.ids.items, s.twin.items);
            },
            2 => if (s.used) {
                _ = lead.giveBack(slot);
                _ = other.giveBack(slot);
            },
            else => if (s.used) {
                lead.close(slot);
                other.close(slot);
            },
        }
        try expectEqual(lead.digest(), other.digest());
        // every page of each part held by one slot at most, and the free counts the rest
        var held: [2][h]bool = @splat(@splat(false));
        var count: [2]usize = .{ 0, 0 };
        for (lead.slots) |x| if (x.used) {
            for (x.ids.items) |id| {
                const p = kvs.partOf(id, h);
                try expectEqual(@as(usize, x.part), p);
                try expect(!held[p][id - p * h]);
                held[p][id - p * h] = true;
                count[p] += 1;
            }
            for (x.twin.items) |id| {
                const p = kvs.partOf(id, h);
                try expect(p != x.part);
                try expect(!held[p][id - p * h]);
                held[p][id - p * h] = true;
                count[p] += 1;
            }
        };
        for (0..2) |p| try expectEqual(h - count[p], lead.pools[p].available());
    }
    for (0..slots) |slot| {
        lead.close(slot);
        other.close(slot);
    }
    try expectEqual(@as(usize, h), lead.pools[0].available());
    try expectEqual(@as(usize, h), lead.pools[1].available());
}
