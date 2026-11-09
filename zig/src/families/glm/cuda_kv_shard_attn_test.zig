//! Owner-computed attention (cuda_kv_shard_attn.run) on a two-rank host model: each rank on its own thread, NCCL's
//! grouped send/recv as a barrier at each group's end, attention as a function of a row's query and its segment's
//! latent rows. Every rank's outputs must equal what it computes unsharded; a rank never attends rows it lacks; every
//! send meets the peer's receive of the same rows.
const std = @import("std");
const attn = @import("cuda_kv_shard_attn.zig");
const kvs = @import("cuda_kv_shard.zig");

const max_rows = 64;
const max_phases = 8;

fn mix(a: u64, b: u64) u64 {
    var h = std.hash.Wyhash.init(a);
    h.update(std.mem.asBytes(&b));
    return h.final();
}

/// What both ranks see: the segments' holds and rows, and the exchanges in flight, a mailbox a phase.
const World = struct {
    mutex: std.Io.Mutex = .init,
    cond: std.Io.Condition = .init,
    holds: []const kvs.Hold,
    rows: []const usize, // each segment's rows
    gen: [2]usize = .{ 0, 0 },
    mail: [2][max_phases]std.ArrayList([]u64) = @splat(@splat(.empty)),
    gpa: std.mem.Allocator,
    failed: ?anyerror = null,

    fn deinit(w: *World) void {
        for (&w.mail) |*r| for (r) |*m| {
            for (m.items) |x| w.gpa.free(x);
            m.deinit(w.gpa);
        };
    }
};

/// One rank: its query and output rows, the peer's queries and the outputs it computed for the peer.
const Rank = struct {
    w: *World,
    rank: usize,
    qa: [max_rows]u64 = undefined,
    ol: [max_rows]u64 = @splat(0),
    pq: [max_rows]u64 = @splat(0),
    po: [max_rows]u64 = @splat(0),
    out: std.ArrayList([]u64) = .empty,
    want: std.ArrayList(struct { buf: attn.Buf, seg: attn.Seg }) = .empty,

    fn buf(r: *Rank, b: attn.Buf) *[max_rows]u64 {
        return switch (b) {
            .qa => &r.qa,
            .ol => &r.ol,
            .pq => &r.pq,
            .po => &r.po,
        };
    }

    fn segIndex(r: *const Rank, s: attn.Seg) usize {
        var at: usize = 0;
        for (r.w.rows, 0..) |n, i| {
            if (at == s.row0) return i;
            at += n;
        }
        unreachable;
    }
};

/// cuda_kv_shard_attn.run's device: `Rank` behind a pointer.
const Model = struct {
    r: *Rank,

    pub fn group(m: Model, e: attn.Edge) !void {
        const r = m.r;
        const w = r.w;
        switch (e) {
            .begin => {
                r.out.clearRetainingCapacity();
                r.want.clearRetainingCapacity();
            },
            .end => {
                const io = std.testing.io;
                w.mutex.lockUncancelable(io);
                defer w.mutex.unlock(io);
                const phase = w.gen[r.rank];
                const peer = 1 - r.rank;
                try w.mail[peer][phase].appendSlice(w.gpa, r.out.items);
                r.out.clearRetainingCapacity();
                w.gen[r.rank] += 1;
                w.cond.broadcast(io);
                while (w.gen[peer] <= phase and w.failed == null) w.cond.waitUncancelable(io, &w.mutex);
                if (w.failed) |err| return err;
                const got = w.mail[r.rank][phase].items;
                if (got.len != r.want.items.len) return error.UnpairedExchange;
                for (got, r.want.items) |rows, x| {
                    if (rows.len != x.seg.rows) return error.UnpairedExchange;
                    @memcpy(r.buf(x.buf)[x.seg.row0..][0..x.seg.rows], rows);
                }
            },
        }
    }

    pub fn move(m: Model, which: attn.Buf, s: attn.Seg, dir: attn.Dir) !void {
        const r = m.r;
        switch (dir) {
            .send => try r.out.append(r.w.gpa, try r.w.gpa.dupe(u64, r.buf(which)[s.row0..][0..s.rows])),
            .recv => try r.want.append(r.w.gpa, .{ .buf = which, .seg = s }),
        }
    }

    pub fn attend(m: Model, s: attn.Seg, pass: attn.Pass) !void {
        const r = m.r;
        const i = r.segIndex(s);
        if (pass == .none) return;
        if (!r.w.holds[i].has(r.rank)) return error.AttendedWithoutRows;
        const lat = mix(0x1a7e, i);
        const q = if (pass == .own) &r.qa else &r.pq;
        const o = if (pass == .own) &r.ol else &r.po;
        for (s.row0..s.row0 + s.rows) |row| o[row] = mix(q[row], lat);
    }
};

fn rankMain(r: *Rank, segs: []const attn.Seg) void {
    attn.run(Model, .{ .r = r }, segs) catch |err| {
        const io = std.testing.io;
        r.w.mutex.lockUncancelable(io);
        defer r.w.mutex.unlock(io);
        if (r.w.failed == null) r.w.failed = err;
        r.w.cond.broadcast(io);
    };
}

/// Both ranks over segments of `rows` rows held as `holds`; every rank's outputs against its unsharded ones.
fn check(gpa: std.mem.Allocator, holds: []const kvs.Hold, rows: []const usize) !void {
    var w: World = .{ .holds = holds, .rows = rows, .gpa = gpa };
    defer w.deinit();
    var ranks: [2]Rank = .{ .{ .w = &w, .rank = 0 }, .{ .w = &w, .rank = 1 } };
    defer for (&ranks) |*r| {
        for (r.out.items) |x| gpa.free(x);
        r.out.deinit(gpa);
        r.want.deinit(gpa);
    };
    var segs: [2][16]attn.Seg = undefined;
    var total: usize = 0;
    for (rows, holds, 0..) |n, h, i| {
        for (0..2) |rk| segs[rk][i] = .{ .row0 = total, .rows = n, .role = attn.role(h, rk) };
        total += n;
    }
    for (&ranks) |*r| for (0..total) |row| {
        r.qa[row] = mix(r.rank + 1, row);
    };
    var threads: [2]std.Thread = undefined;
    for (&threads, &ranks) |*t, *r| t.* = try std.Thread.spawn(.{}, rankMain, .{ r, segs[r.rank][0..rows.len] });
    for (threads) |t| t.join();
    if (w.failed) |err| return err;
    var at: usize = 0;
    for (rows, 0..) |n, i| {
        const lat = mix(0x1a7e, i);
        for (&ranks) |*r| for (at..at + n) |row| {
            if (r.ol[row] != mix(r.qa[row], lat)) return error.WrongOutput;
        };
        at += n;
    }
}

test "kv shard attention: both ranks' outputs are their unsharded ones: owners, twins, both ranks owning" {
    const gpa = std.testing.allocator;
    try check(gpa, &.{.{ .owner = 0 }}, &.{16}); // one stream on rank 0: rank 1 gives
    try check(gpa, &.{.{ .owner = 1 }}, &.{1});
    try check(gpa, &.{.{ .owner = 1, .twin = true }}, &.{4096 % 64}); // a twin: both attend locally
    try check(gpa, &.{ .{ .owner = 0 }, .{ .owner = 1 }, .{ .owner = 0, .twin = true }, .{ .owner = 1 } }, &.{ 3, 16, 1, 9 });
    try check(gpa, &.{ .{ .owner = 1 }, .{ .owner = 1 }, .{ .owner = 1 } }, &.{ 2, 2, 2 });
}

test "kv shard attention: random windows of up to 16 segments" {
    const gpa = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(0x5aded);
    const rnd = prng.random();
    for (0..150) |_| {
        const n = rnd.intRangeAtMost(usize, 1, 16);
        var holds: [16]kvs.Hold = undefined;
        var rows: [16]usize = undefined;
        var left: usize = max_rows;
        for (0..n) |i| {
            holds[i] = .{ .owner = rnd.intRangeAtMost(u8, 0, 1), .twin = rnd.uintLessThan(u8, 4) == 0 };
            rows[i] = rnd.intRangeAtMost(usize, 1, @max(1, @min(16, left - (n - 1 - i))));
            left -= rows[i];
        }
        try check(gpa, holds[0..n], rows[0..n]);
    }
}

test "kv shard attention: roles: no hold or a twin attends locally; the owner takes; the other rank gives" {
    try std.testing.expectEqual(attn.Role.local, attn.role(null, 0));
    try std.testing.expectEqual(attn.Role.local, attn.role(.{ .owner = 0, .twin = true }, 1));
    try std.testing.expectEqual(attn.Role.take, attn.role(.{ .owner = 1 }, 1));
    try std.testing.expectEqual(attn.Role.give, attn.role(.{ .owner = 1 }, 0));
    try std.testing.expect(!attn.exchanges(&.{ .{ .row0 = 0, .rows = 1, .role = .local }, .{ .row0 = 1, .rows = 2, .role = .local } }));
    try std.testing.expect(attn.exchanges(&.{ .{ .row0 = 0, .rows = 1, .role = .local }, .{ .row0 = 1, .rows = 2, .role = .give } }));
    try std.testing.expectEqual(@as(usize, 2 * 4096 * 32 * 512 * 2), attn.roomBytes(4096, 32, 512)); // 256 MiB at TP2
}

test "kv shard attention: each guard's mutant is caught" {
    const gpa = std.testing.allocator;
    defer attn.mutant = .none;
    inline for (.{ attn.Mutant.peer_skipped, attn.Mutant.output_recv_into_qa }) |m| {
        attn.mutant = m;
        try std.testing.expectError(error.WrongOutput, check(gpa, &.{ .{ .owner = 0 }, .{ .owner = 1 } }, &.{ 4, 4 }));
    }
}
