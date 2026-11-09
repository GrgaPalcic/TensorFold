//! TF_GLM_KV_SHARD's kept states (GLM patch 0118 by Robert Wojciechowski, GitHub @wojo, X @rwojo): a state's home is
//! its stream's owner part; the home rank's copy holds every part as unsharded, the other rank's every part but the
//! DSA caches' latent codes and scales. Both lease the unsharded size, so rank 0's admission, which charges its own
//! lease, never passes a keep the other rank's pages cannot hold. A resume restores each rank's own parts, then the
//! home rank sends the latent rows to every rank that holds the new stream's rows but is not the home (`carry`): a
//! stream placed outside the home part (TF_GLM_KV_SHARD_CARRY), or a resumed pass's twin. Host side: who keeps,
//! who sends, and the pieces both ranks cut alike.
const std = @import("std");
const kvs = @import("cuda_kv_shard.zig");

/// Bytes a carry moves in one piece (the staging buffer's size on each rank).
pub const piece_bytes: usize = 32 << 20;

/// Whether this rank's copy of a kept state holds its latent rows: unsharded always; sharded on the stream's owner.
pub fn keepsLatent(hold: ?kvs.Hold, rank: usize) bool {
    const h = hold orelse return true;
    return h.owner == rank or mut(.everyone_keeps);
}

/// This rank's side of a resume's carry: the home rank sends when the other rank holds the new stream's rows; the other
/// rank receives when it does. `home` null: an unsharded state (no carry).
pub const Side = enum { none, send, recv };

pub fn carry(home: ?u8, hold: ?kvs.Hold, rank: usize) Side {
    const hp = home orelse return .none;
    const h = hold orelse return .none;
    const peer = 1 - rank;
    if (rank == hp) return if (h.has(peer) and peer != hp) .send else .none;
    return if (h.has(rank) or mut(.recv_without_rows)) .recv else .none;
}

/// Whether a resume restores the latent parts of its own copy into the live sequence: on its home, when it holds the
/// new stream's rows (else its copy's latent parts are only sent, or skipped).
pub fn restoresLatent(home: ?u8, hold: ?kvs.Hold, rank: usize) bool {
    const hp = home orelse return true;
    const h = hold orelse return true;
    return rank == hp and h.has(rank);
}

/// The pieces of a carry over latent parts of `lens` bytes, `piece` at most: (part, offset in it, bytes), in order.
/// The sender cuts its copy's parts and the receiver its live parts by this one rule, so each send meets its receive.
pub fn pieces(gpa: std.mem.Allocator, lens: []const usize, piece: usize) ![]const [3]usize {
    var out: std.ArrayList([3]usize) = .empty;
    errdefer out.deinit(gpa);
    for (lens, 0..) |len, i| {
        var done: usize = 0;
        while (done < len) {
            const n = @min(piece, len - done);
            try out.append(gpa, .{ i, done, n });
            done += n;
        }
    }
    return out.toOwnedSlice(gpa);
}

pub const Mutant = enum { none, everyone_keeps, recv_without_rows };
pub var mutant: Mutant = .none;

fn mut(comptime m: Mutant) bool {
    return @import("builtin").is_test and mutant == m;
}

test "kv shard keep: the owner's copy alone keeps the latent rows; unsharded every copy does" {
    try std.testing.expect(keepsLatent(null, 0) and keepsLatent(null, 1));
    try std.testing.expect(keepsLatent(.{ .owner = 1 }, 1));
    try std.testing.expect(!keepsLatent(.{ .owner = 1 }, 0));
    try std.testing.expect(!keepsLatent(.{ .owner = 0, .twin = true }, 1)); // a twin's rows are not the state's
}

test "kv shard keep: a resume's carry: the home rank sends to the rank that holds the new stream's rows" {
    for ([_]u8{ 0, 1 }) |home| for ([_]u8{ 0, 1 }) |owner| for ([_]bool{ false, true }) |twin| {
        const h: kvs.Hold = .{ .owner = owner, .twin = twin };
        const a = carry(home, h, home);
        const b = carry(home, h, 1 - home);
        const needs = h.has(1 - home); // the other rank holds the new stream's rows
        try std.testing.expectEqual(if (needs) Side.send else Side.none, a);
        try std.testing.expectEqual(if (needs) Side.recv else Side.none, b);
        try std.testing.expectEqual(h.has(home), restoresLatent(home, h, home));
        try std.testing.expect(!restoresLatent(home, h, 1 - home));
    };
    try std.testing.expectEqual(Side.none, carry(null, .{ .owner = 0 }, 1)); // unsharded state
    try std.testing.expectEqual(Side.none, carry(1, null, 0));
    try std.testing.expect(restoresLatent(null, .{ .owner = 0 }, 1));
}

test "kv shard keep: both ranks cut a carry's pieces alike, whole parts covered once in order" {
    const gpa = std.testing.allocator;
    const lens = [_]usize{ 70 << 20, 2048, 0, 64 << 20 };
    const p = try pieces(gpa, &lens, piece_bytes);
    defer gpa.free(p);
    try std.testing.expectEqual(@as(usize, 3 + 1 + 2), p.len);
    var sum: [4]usize = @splat(0);
    for (p) |x| {
        try std.testing.expect(x[2] > 0 and x[2] <= piece_bytes);
        try std.testing.expectEqual(sum[x[0]], x[1]);
        sum[x[0]] += x[2];
    }
    try std.testing.expectEqualSlices(usize, &lens, &sum);
}

test "kv shard keep: each guard's mutant is caught" {
    defer mutant = .none;
    mutant = .everyone_keeps;
    try std.testing.expect(keepsLatent(.{ .owner = 1 }, 0));
    mutant = .recv_without_rows;
    try std.testing.expectEqual(Side.recv, carry(0, .{ .owner = 0 }, 1)); // the peer would wait on a send never made
}
