//! Kept states written behind: a block (or a block's worth of a state's own file) a step, only while the engine idles.
const std = @import("std");
const platform = @import("sys");
const pc = @import("prompt_cache.zig");
const image = @import("rows_image.zig");
const chain = @import("rows_chain.zig");
const file = @import("rows_file.zig");
const rows_store = @import("rows_store.zig");
const Rows = rows_store.Rows;
const Name = chain.Name;
const Span = @import("cache_modes.zig").Span;

/// A state's own file goes out in pieces of this many bytes, so a request waits for one piece at most.
const piece: u64 = 16 << 20;

/// One whole block a state names: its name, group (window), index and file bytes.
pub const Want = struct { name: Name, window: u32, k: u32, bytes: u64 };

/// A kept state waiting to be written, with the chunk starts below it (a planned family's state depends on them).
pub const Queued = struct {
    kept: rows_store.Kept,
    starts: []u32,
    waited: bool = false, // the jobs in flight once left it no room, so losing its entry counts as a skipped save
};

/// A kept state being written: the blocks the disk lacks, then its own file a piece at a time.
pub const Job = struct {
    kept: rows_store.Kept,
    starts: []u32,
    name: Name,
    parts: []image.Part,
    wants: []Want,
    next: usize = 0, // the next want to look at
    own: []image.Range,
    own_i: usize = 0,
    own_off: u64 = 0,
    writer: ?file.Writer = null,
    bytes: u64 = 0, // new bytes this job writes, reserved under the free-disk floor
    reserved: bool = false,
    blocks: u32 = 0, // blocks it wrote
    began: i128 = 0,
};

/// A newly kept state joins the queue (the store calls this on every new keep), with the request's chunk starts.
pub fn queued(r: *Rows, kept: rows_store.Kept, starts: []const u32) void {
    const own = rows_store.startsBelow(r, r.gpa, starts, kept.at) catch return;
    r.queue.append(r.gpa, .{ .kept = kept, .starts = own }) catch r.gpa.free(own);
}

/// The store's hold on a queued save given up: its pages released (nothing held: the flat store's).
pub fn gave(_: *Rows, q: *const Queued) void {
    if (q.kept.freed) |f| f.call(f.ctx, q.kept.owner);
}

/// A state leaving the RAM store leaves the queue, and its write is given up if under way.
pub fn dropped(r: *Rows, owner: *anyopaque) void {
    for (r.queue.items, 0..) |*q, i| if (q.kept.owner == owner) {
        r.gpa.free(q.starts);
        gave(r, q);
        _ = r.queue.orderedRemove(i);
        break;
    };
    if (r.job) |j| if (j.kept.owner == owner) abort(r, j);
}

/// Whether writing waits; spill-only RAM entries wait for eviction or the close-time flush.
pub fn pending(r: *const Rows) bool {
    return r.job != null or ((r.spill != .evicted or r.flush_clock != null) and r.queue.items.len > 0);
}

/// --spill-gib: an entry evicted from RAM is written now, unless the disk holds it already; the write under way first.
pub fn evicted(r: *Rows, owner: *anyopaque) void {
    if (r.spill == .off) return;
    if (r.job) |j| if (j.kept.owner != owner) while (r.job != null) {
        _ = step(r);
    };
    if (r.job == null) {
        const i = for (r.queue.items, 0..) |*q, k| {
            if (q.kept.owner == owner) break k;
        } else return;
        if (!begin(r, r.queue.orderedRemove(i))) return;
    }
    while (r.job != null) _ = step(r);
}

/// A paged store's eviction: a save still queued is written now (its pages are held; the state leaves the tree).
pub fn evicting(r: *Rows, owner: *anyopaque) void {
    if (r.job) |j| if (j.kept.owner != owner) while (r.job != null) {
        _ = step(r);
    };
    if (r.job == null) {
        const i = for (r.queue.items, 0..) |*q, k| {
            if (q.kept.owner == owner) break k;
        } else return;
        if (!begin(r, r.queue.orderedRemove(i))) return;
    }
    while (r.job != null) _ = step(r);
}

/// Whether the job under way names the block of stem `n` (collection keeps it while its state is written).
pub fn wanted(r: *const Rows, n: chain.Stem) bool {
    const j = r.job orelse return false;
    for (j.wants) |w| if (std.mem.eql(u8, w.name[0..16], &n)) return true;
    return false;
}

/// One unit of writing: a job started, a block, or a piece of a state's file; false when nothing was done.
pub fn step(r: *Rows) bool {
    const j = r.job orelse return start(r);
    unit(r, j) catch |err| {
        r.counts.failed += 1;
        pc.note("imprint: writing {d} tokens failed ({s}); they stay in RAM only", .{ j.kept.at, @errorName(err) });
        end(r, j, false);
    };
    return true;
}

/// Write until nothing waits or `deadline_ns` (monotonic) passes; the states left unwritten.
pub fn drain(r: *Rows, deadline_ns: i128) usize {
    while (pending(r) and nowNs() < deadline_ns) _ = step(r);
    const left = r.queue.items.len + @intFromBool(r.job != null);
    if (left > 0) pc.note("imprint: the stop's deadline passed with {d} kept states unwritten", .{left});
    return left;
}

pub fn nowNs() i128 {
    var t: platform.c.timespec = undefined;
    _ = platform.c.clock_gettime(platform.c.CLOCK.MONOTONIC, &t);
    return @as(i128, t.sec) * std.time.ns_per_s + t.nsec;
}

/// The newest queued entry made a job (its blocks serve older ones); one on disk, or past budget, cap or floor, skips.
fn start(r: *Rows) bool {
    if (r.spill == .evicted and r.flush_clock == null) return false;
    const q = r.queue.pop() orelse return false;
    _ = begin(r, q);
    return true;
}

/// `q` made the job under way; false when it is on disk already, or past budget, cap or floor.
fn begin(r: *Rows, q: Queued) bool {
    const j = (prepare(r, q) catch |err| {
        r.counts.skipped += 1;
        pc.note("imprint: {d} tokens stay in RAM only ({s})", .{ q.kept.at, @errorName(err) });
        gave(r, &q);
        return false;
    }) orelse return false;
    r.job = j;
    admit(r, j) catch |err| {
        r.counts.skipped += 1;
        pc.note("imprint: {d} tokens stay in RAM only ({s})", .{ j.kept.at, @errorName(err) });
        abort(r, j);
        return false;
    };
    return true;
}

/// A job for `q` (its starts now the job's), or null when the disk holds its state already.
pub fn prepare(r: *Rows, q: Queued) !?*Job {
    const gpa = r.gpa;
    errdefer gpa.free(q.starts);
    const kept = q.kept;
    const name = chain.state(r.keying, kept.tokens, kept.spans, q.starts, kept.at);
    if (r.find(name)) |i| {
        r.touch(i);
        gpa.free(q.starts);
        gave(r, &q);
        return null;
    }
    if (r.img.whole) |whole| if (!whole(r.family.ptr, kept.saved)) return error.NotWhole;
    const parts = try r.parts(gpa, kept.at);
    errdefer gpa.free(parts);
    var wants: std.ArrayList(Want) = .empty;
    errdefer wants.deinit(gpa);
    try r.blockNames(gpa, parts, kept.tokens, kept.spans, q.starts, kept.at, &wants);
    var own: std.ArrayList(image.Range) = .empty;
    errdefer own.deinit(gpa);
    try image.ownRanges(parts, kept.at, r.keying.block, &own, gpa);
    const j = try gpa.create(Job);
    j.* = .{ .kept = kept, .starts = q.starts, .name = name, .parts = parts, .wants = &.{}, .own = &.{}, .began = nowNs() };
    j.wants = wants.toOwnedSlice(gpa) catch |err| {
        gpa.destroy(j);
        return err;
    };
    j.own = own.toOwnedSlice(gpa) catch |err| {
        gpa.free(j.wants);
        gpa.destroy(j);
        return err;
    };
    return j;
}

/// The job's new bytes checked against today's budget, the cap (evicting by value) and the free-disk floor.
fn admit(r: *Rows, j: *Job) !void {
    const kept = j.kept;
    for (j.wants) |w| if (!r.blocks.contains(chain.stem(w.name))) {
        j.bytes += w.bytes;
    };
    j.bytes += image.total(j.own) + 4 * (3 + kept.tokens.len + 2 * kept.spans.len + j.starts.len) + @sizeOf(file.Trailer);
    if (!r.budget(j.bytes)) return error.DailyBudgetSpent;
    if (!r.room(j.bytes)) return error.OverCap;
    switch (r.admission.reserve(r.dir, j.bytes)) {
        .ready => j.reserved = true,
        else => return error.UnderFreeDiskFloor,
    }
}

/// The job's next unit of work.
fn unit(r: *Rows, j: *Job) !void {
    if (r.img.whole) |whole| if (!whole(r.family.ptr, j.kept.saved)) return error.NotWhole;
    const gather = r.img.gather;
    while (j.next < j.wants.len and r.blocks.contains(chain.stem(j.wants[j.next].name))) j.next += 1;
    if (j.next < j.wants.len) {
        const w = j.wants[j.next];
        var ranges: std.ArrayList(image.Range) = .empty;
        defer ranges.deinit(r.gpa);
        try image.blockRanges(j.parts, r.keying.block, w.window, w.k, &ranges, r.gpa);
        const n = image.total(ranges.items);
        const stage = try r.staged(n);
        try gather(r.family.ptr, j.kept.saved, j.kept.pages, ranges.items, stage[0..@intCast(n)]);
        var wr = try file.Writer.open(r.dir, &chain.hex(w.name), ".blk");
        errdefer wr.abort();
        wr.io = r.file_io;
        try wr.append(stage[0..@intCast(n)]);
        const bytes = try wr.finish(.{ .kind = @backingInt(file.Kind.block), .payload = 0, .payload_hash = 0, .name = w.name, .block = r.keying.block }, &.{}, false);
        try r.blocks.put(r.gpa, chain.stem(w.name), .{ .bytes = bytes });
        r.spent(bytes);
        j.blocks += 1;
        j.next += 1;
        return;
    }
    if (j.writer == null) {
        j.writer = try file.Writer.open(r.dir, &chain.hex(j.name), ".st");
        j.writer.?.io = r.file_io;
    }
    if (j.own_i < j.own.len) {
        var pieces: std.ArrayList(image.Range) = .empty;
        defer pieces.deinit(r.gpa);
        var n: u64 = 0;
        while (j.own_i < j.own.len and n < piece) {
            const g = j.own[j.own_i];
            const take = @min(g.len - j.own_off, piece - n);
            try pieces.append(r.gpa, .{ .off = g.off + j.own_off, .len = take });
            n += take;
            j.own_off += take;
            if (j.own_off == g.len) {
                j.own_i += 1;
                j.own_off = 0;
            }
        }
        const stage = try r.staged(n);
        try gather(r.family.ptr, j.kept.saved, j.kept.pages, pieces.items, stage[0..@intCast(n)]);
        try j.writer.?.append(stage[0..@intCast(n)]);
        return;
    }
    const kept = j.kept;
    const counts = [3]u32{ @intCast(kept.tokens.len), @intCast(kept.spans.len), @intCast(j.starts.len) };
    const meta = [_][]const u8{ std.mem.sliceAsBytes(&counts), std.mem.sliceAsBytes(kept.tokens), std.mem.sliceAsBytes(kept.spans), std.mem.sliceAsBytes(j.starts) };
    const bytes = try j.writer.?.finish(.{ .kind = @backingInt(file.Kind.state), .payload = 0, .payload_hash = 0, .name = j.name, .at = kept.at, .block = r.keying.block, .flags = @intFromBool(kept.shared) }, &meta, true);
    j.writer = null;
    r.spent(bytes);
    var s: rows_store.State = .{ .name = j.name, .at = kept.at, .tokens = &.{}, .spans = &.{}, .starts = &.{}, .own = bytes, .rows = 0, .used = 0, .shared = kept.shared };
    s.tokens = try r.gpa.dupe(u32, kept.tokens);
    s.spans = r.gpa.dupe(Span, kept.spans) catch |err| {
        r.gpa.free(s.tokens);
        return err;
    };
    s.starts = j.starts;
    j.starts = &.{};
    for (j.wants) |w| s.rows += w.bytes;
    try r.states.append(r.gpa, s);
    r.ref(j.wants, 1);
    r.touch(r.states.items.len - 1);
    r.counts.writes += 1;
    const ms = @as(f64, @floatFromInt(nowNs() - j.began)) / 1e6;
    if (!@import("builtin").is_test) std.log.info("imprint: {d} tokens on disk: {d} new blocks and its own {d} MiB ({d} MiB this state, {d} MiB of rows it shares) in {d:.0} ms", .{ kept.at, j.blocks, bytes >> 20, j.bytes >> 20, s.rows >> 20, ms });
    if (r.verify) @import("rows_verify.zig").state(r, r.states.items.len - 1, kept.saved, kept.pages);
    finishJob(r, j, true);
}

/// The job under way given up (its entry left RAM, or the store closes): its file in progress goes.
pub fn abort(r: *Rows, j: *Job) void {
    end(r, j, true);
}

/// A job ended unwritten; `disk_ok` false (a write failed) backs the free-disk floor's admission off.
fn end(r: *Rows, j: *Job, disk_ok: bool) void {
    if (j.writer) |*w| w.abort();
    j.writer = null;
    finishJob(r, j, disk_ok);
    r.collect();
}

fn finishJob(r: *Rows, j: *Job, ok: bool) void {
    if (j.reserved) r.admission.finish(r.dir, j.bytes, ok);
    if (j.kept.freed) |f| f.call(f.ctx, j.kept.owner); // the write no longer reads the state: its pages go back
    r.gpa.free(j.starts);
    r.gpa.free(j.parts);
    r.gpa.free(j.wants);
    r.gpa.free(j.own);
    r.job = null;
    r.gpa.destroy(j);
}
