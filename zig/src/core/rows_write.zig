//! The engine stages owned images and publishes completions; one I/O thread writes and syncs their files.
const std = @import("std");
const pc = @import("prompt_cache.zig");
const store = @import("rows_store.zig");
const image = @import("rows_image.zig");
const file = @import("rows_file.zig");
const chain = @import("rows_chain.zig");
const sync = @import("rows_write_sync.zig");
pub const io = @import("rows_io.zig");
const Rows = store.Rows;
const Kept = store.Kept;
pub const Want = sync.Want;
pub const Queued = sync.Queued;
pub const Job = sync.Job;
pub const nowNs = io.nowNs;

pub const Failure = struct { id: u64, generation: u64, name: chain.Name, at: u32, err: anyerror };

/// Recent failures retain each job's identity and outcome, including failures that follow an earlier failure.
pub const Failures = struct {
    records: [64]?Failure = @splat(null),
    count: u64 = 0,

    pub fn add(f: *Failures, record: Failure) void {
        f.records[@intCast(f.count % f.records.len)] = record;
        f.count += 1;
    }
};

pub fn queued(r: *Rows, kept: Kept, starts: []const u32) void {
    if (r.stopped) return;
    sync.queued(r, kept, starts);
}

/// The store's hold on a queued save given up: its pages released, its tokens freed (the store's own copies).
pub fn gave(_: *Rows, q: *const sync.Queued) void {
    if (q.kept.freed) |f| f.call(f.ctx, q.kept.owner);
}

/// Eviction removes only a save still queued (staged work owns no entry); one that waited for room counts as skipped.
pub fn dropped(r: *Rows, owner: *anyopaque) void {
    if (r.waiting == owner) r.waiting = null;
    for (r.queue.items, 0..) |*q, i| if (q.kept.owner == owner) {
        if (q.waited and !r.stopped) {
            r.next_job +%= 1; // the id a save staged now would take, so no job in flight shares it
            skip(r, chain.state(r.keying, q.kept.tokens, q.kept.spans, q.starts, q.kept.at), q.kept.at, error.StagingFull);
        }
        r.gpa.free(q.starts);
        gave(r, q);
        _ = r.queue.orderedRemove(i);
        break;
    };
    sync.dropped(r, owner);
}

/// A paged store's eviction: a save still queued is written now (its pages are held; the state leaves the tree).
pub fn evicting(r: *Rows, owner: *anyopaque) void {
    if (r.sync_writes) return sync.evicting(r, owner);
    if (r.stopped) return;
    reap(r);
    for (r.queue.items, 0..) |*q, i| if (q.kept.owner == owner) {
        _ = stage(r, r.queue.orderedRemove(i), false);
        return;
    };
}

pub fn pending(r: *const Rows) bool {
    if (r.sync_writes) return sync.pending(r);
    return (if (r.worker) |w| w.pending() > 0 else false) or (!r.stopped and (r.spill != .evicted or r.flush_clock != null) and r.queue.items.len > 0);
}

pub fn ready(r: *const Rows) bool {
    if (r.sync_writes) return sync.pending(r);
    return (if (r.worker) |w| w.ready() else false) or (!r.stopped and r.spill != .evicted and r.queue.items.len > 0 and !blocked(r));
}

/// Whether the newest queued save was refused room by the jobs in flight and none has been taken since.
fn blocked(r: *const Rows) bool {
    const o = r.waiting orelse return false;
    return r.queue.items.len > 0 and r.queue.items[r.queue.items.len - 1].kept.owner == o;
}

/// A spill copies the entry before RAM releases it, without waiting for filesystem work or a byte permit.
pub fn evicted(r: *Rows, owner: *anyopaque) void {
    if (r.sync_writes) return sync.evicted(r, owner);
    if (r.spill == .off or r.stopped) return;
    reap(r);
    for (r.queue.items, 0..) |*q, i| if (q.kept.owner == owner) {
        // The entry leaves RAM now, so a full permit skips it rather than hold it.
        _ = stage(r, r.queue.orderedRemove(i), false);
        return;
    };
}

pub fn wanted(r: *const Rows, n: chain.Stem) bool {
    if (r.sync_writes) return sync.wanted(r, n);
    return if (r.worker) |w| w.wanted(n) else false;
}

/// Reconcile finished jobs and stage at most one queued entry; a save waiting for room is work only a reap can free.
pub fn step(r: *Rows) bool {
    if (r.sync_writes) return sync.step(r);
    reap(r);
    if (r.stopped or blocked(r) or (r.spill == .evicted and r.flush_clock == null)) return false;
    const q = r.queue.pop() orelse return false;
    return stage(r, q, true);
}

pub fn abort(r: *Rows, j: *Job) void {
    sync.abort(r, j);
}

/// Invalidate old completions before an unlink or a change of model generation.
pub fn fence(r: *Rows) void {
    if (r.worker) |w| w.fence();
}

/// Stop admits no further jobs; an I/O syscall cannot extend the owner's shutdown deadline.
pub fn stop(r: *Rows, deadline: i128) void {
    if (r.fetch) |f| f.stopAt(deadline);
    if (r.stopped) return;
    r.stopped = true;
    fence(r);
    if (r.worker) |w| {
        var jobs: [io.max_jobs]?*io.Job = undefined;
        w.cancellations(&jobs);
        for (jobs) |job| if (job) |j| {
            const err = if (j.done.load(.acquire)) j.failure orelse error.Cancelled else error.Cancelled;
            if (err == error.Cancelled) r.failures.add(.{ .id = j.id, .generation = j.generation, .name = j.name, .at = j.at, .err = err }) else failed(r, j, err);
            r.admission.finish(r.dir, j.bytes, err == error.Cancelled);
        };
        r.worker = null;
        w.close(deadline);
    }
}

/// A spill-enabled close also drains RAM, newest first; completed saves keep their cap space over older entries.
pub fn drain(r: *Rows, deadline: i128) usize {
    const previous = r.flush_clock;
    defer r.flush_clock = previous;
    if (r.spill != .off) {
        if (r.sync_writes) {
            while (r.job != null and nowNs() < deadline) _ = sync.step(r);
        } else while (r.worker) |w| {
            reap(r);
            if (w.pending() == 0 or nowNs() >= deadline) break;
            io.nap();
        }
        r.flush_clock = r.clock;
    }
    if (r.sync_writes) return sync.drain(r, deadline);
    while (pending(r) and nowNs() < deadline) {
        if (r.flush_clock != null) {
            reap(r);
            if (nowNs() >= deadline) break;
            if (r.worker) |w| if (w.pending() > 0) {
                io.nap();
                continue;
            };
        }
        if (!step(r)) io.nap();
    }
    reap(r);
    const left = r.queue.items.len + (if (r.worker) |w| w.pending() else @as(usize, 0));
    stop(r, deadline);
    if (left > 0) pc.note("imprint: the stop's deadline passed with {d} kept states unwritten", .{left});
    return left;
}

fn failed(r: *Rows, j: *const io.Job, err: anyerror) void {
    r.counts.failed += 1;
    if (err == error.VerifyFailed) r.counts.differed += 1;
    r.failures.add(.{ .id = j.id, .generation = j.generation, .name = j.name, .at = j.at, .err = err });
    pc.note("imprint: writing {d} tokens failed ({s}); the save is skipped", .{ j.at, @errorName(err) });
}

/// Only this engine-thread function changes indexes or exposes completed files, after all writes and syncs succeeded.
pub fn reap(r: *Rows) void {
    const w = r.worker orelse return;
    while (w.take()) |j| {
        defer j.destroy();
        r.waiting = null;
        const err: ?anyerror = j.failure orelse if (j.generation != w.generation.load(.acquire)) error.Cancelled else null;
        if (err) |e| {
            failed(r, j, e);
            r.admission.finish(r.dir, j.bytes, e == error.Cancelled);
            r.collect();
            continue;
        }
        publish(r, j) catch |e| {
            failed(r, j, e);
            // The job that owed the block already backed admission off.
            r.admission.finish(r.dir, j.bytes, e == error.MissingBlock);
            r.collect();
            continue;
        };
        r.admission.finish(r.dir, j.bytes, true);
    }
}

fn publish(r: *Rows, j: *io.Job) !void {
    if (!j.final) {
        // A piece's blocks: complete files, published as any state's (its state's entry comes with the final job).
        try r.blocks.ensureUnusedCapacity(r.gpa, @intCast(j.files.len));
        for (j.files) |*f| {
            try f.writer.?.publish();
            const refs = if (r.blocks.get(chain.stem(f.trailer.name))) |b| b.refs else 0;
            r.blocks.putAssumeCapacity(chain.stem(f.trailer.name), .{ .bytes = f.bytes, .refs = refs });
            r.spent(f.bytes);
        }
        return;
    }
    if (r.find(j.name)) |_| return;
    if (!complete(r, j)) return error.MissingBlock;
    var s: store.State = .{ .name = j.name, .at = j.at, .tokens = &.{}, .spans = &.{}, .starts = &.{}, .own = j.files[j.files.len - 1].bytes, .rows = 0, .used = 0, .shared = j.shared };
    s.tokens = try r.gpa.dupe(u32, j.tokens);
    errdefer r.gpa.free(s.tokens);
    s.spans = try r.gpa.dupe(@import("cache_modes.zig").Span, j.spans);
    errdefer r.gpa.free(s.spans);
    s.starts = try r.gpa.dupe(u32, j.starts);
    errdefer r.gpa.free(s.starts);
    try r.states.ensureUnusedCapacity(r.gpa, 1);
    try r.blocks.ensureUnusedCapacity(r.gpa, @intCast(j.files.len - 1));
    for (j.files[0 .. j.files.len - 1]) |*f| {
        try f.writer.?.publish();
        const refs = if (r.blocks.get(chain.stem(f.trailer.name))) |b| b.refs else 0;
        r.blocks.putAssumeCapacity(chain.stem(f.trailer.name), .{ .bytes = f.bytes, .refs = refs });
        r.spent(f.bytes);
    }
    try j.files[j.files.len - 1].writer.?.publish();
    r.spent(s.own);
    for (j.wants) |want| s.rows += want.bytes;
    r.states.appendAssumeCapacity(s);
    r.ref(j.wants, 1);
    r.touch(r.states.items.len - 1);
    r.counts.writes += 1;
    if (j.verify) r.counts.verified += 1;
    if (!@import("builtin").is_test) std.log.info("imprint: {d} tokens on disk: {d} new blocks and its own {d} MiB ({d} MiB this state, {d} MiB of rows it shares) in {d:.0} ms", .{ j.at, j.new_blocks, s.own >> 20, j.bytes >> 20, s.rows >> 20, @as(f64, @floatFromInt(nowNs() - j.began)) / 1e6 });
}

/// Whether every block the job's state names is on disk or among the files it publishes now.
fn complete(r: *const Rows, j: *const io.Job) bool {
    blocks: for (j.wants) |want| {
        const n = chain.stem(want.name);
        if (r.blocks.contains(n)) continue;
        for (j.blocks()) |f| if (std.mem.eql(u8, &chain.stem(f.trailer.name), &n)) continue :blocks;
        return false;
    }
    return true;
}

/// Whether `q` left the queue, staged or skipped; false when it waits for room and is queued again (`may_wait`).
fn stage(r: *Rows, q: Queued, may_wait: bool) bool {
    const at = q.kept.at;
    r.next_job +%= 1;
    const name = chain.state(r.keying, q.kept.tokens, q.kept.spans, q.starts, at);
    const j = sync.prepare(r, q) catch |err| {
        gave(r, &q);
        skip(r, name, at, err);
        return true;
    } orelse return true;
    var queued_again = false;
    defer {
        if (!queued_again) {
            r.gpa.free(j.starts);
            gave(r, &q); // staged or skipped: the save no longer reads the state
        }
        r.gpa.free(j.parts);
        r.gpa.free(j.wants);
        r.gpa.free(j.own);
        r.gpa.destroy(j);
    }
    stageJob(r, j, may_wait) catch |err| {
        if (err == error.StagingBusy) {
            queued_again = true;
            r.waiting = q.kept.owner;
            r.queue.appendAssumeCapacity(.{ .kept = q.kept, .starts = q.starts, .waited = true });
            return false;
        }
        skip(r, j.name, at, err);
        if (err == error.GatherFailed or err == error.SnapshotGpuFailure) r.counts.failed += 1;
        return true;
    };
    return true;
}

/// A save given up: counted, recorded under the job id it would have had, and noted.
fn skip(r: *Rows, name: chain.Name, at: u32, err: anyerror) void {
    r.counts.skipped += 1;
    r.failures.add(.{ .id = r.next_job, .generation = if (r.worker) |w| w.generation.load(.acquire) else 0, .name = name, .at = at, .err = err });
    pc.note("imprint: saving {d} tokens skipped ({s})", .{ at, @errorName(err) });
}

fn stageJob(r: *Rows, prepared: *const Job, may_wait: bool) !void {
    if (r.worker == null) r.worker = try io.Worker.open(r.dir, r.staging_bytes, r.file_io);
    const w = r.worker.?;
    if (w.names(prepared.name)) return;
    const kept = prepared.kept;
    const metadata = 4 * (3 + kept.tokens.len + 2 * kept.spans.len + prepared.starts.len);
    var promised: io.Promised = .empty;
    defer promised.deinit(r.gpa);
    // The whole state as the one job a save has always staged, when the permit holds it; else its missing blocks a
    // piece at a time, block edges between pieces, its own file and metadata last (a 600k state under a 1 GiB permit).
    if (try whole(r, w, prepared, kept, &promised, metadata, may_wait)) return;
    var wi: usize = 0;
    var new_blocks: u32 = 0;
    while (wi < prepared.wants.len) {
        try w.promised(r.gpa, &promised);
        var end = wi;
        var bytes: u64 = 0;
        var count: usize = 0;
        while (end < prepared.wants.len) : (end += 1) {
            const want = prepared.wants[end];
            if (!lacks(r, &promised, want)) continue;
            const next = bytes + want.bytes + @sizeOf(io.Job) + (count + 1) * @sizeOf(io.File) + 4 * @sizeOf([]const u8);
            if (count > 0 and !w.fits(next)) break; // the permit holds this piece's blocks alone
            if (count == 0 and !w.fits(next)) return if (may_wait and w.busy(next)) error.StagingBusy else error.StagingFull;
            bytes += want.bytes;
            count += 1;
        }
        if (count == 0) { // everything from `wi` on is on disk or in flight: the next run of missing blocks
            wi = end;
            continue;
        }
        try piece(r, w, prepared, kept, &promised, wi, end, bytes, count, may_wait);
        new_blocks += @intCast(count);
        wi = end;
    }
    // The state's own file and metadata last: its publish makes the entry, once every block it names is on disk.
    try w.promised(r.gpa, &promised);
    const own_bytes = image.total(prepared.own) + metadata + @sizeOf(file.Trailer);
    const charge = own_bytes + @sizeOf(io.Job) + @sizeOf(io.File) + prepared.wants.len * @sizeOf(Want) + 4 * @sizeOf([]const u8);
    if (!w.fits(charge)) return if (may_wait and w.busy(charge)) error.StagingBusy else error.StagingFull;
    const reserved = w.reserved();
    if (!r.budget(own_bytes + reserved)) return error.DailyBudgetSpent;
    if (!r.room(own_bytes + reserved)) return error.OverCap;
    if (r.admission.reserve(r.dir, own_bytes) != .ready) return error.UnderFreeDiskFloor;
    errdefer r.admission.finish(r.dir, own_bytes, true);
    const j = try io.allocator.create(io.Job);
    errdefer io.allocator.destroy(j);
    var arena = std.heap.ArenaAllocator.init(io.allocator);
    errdefer arena.deinit();
    const a = arena.allocator();
    const files = try a.alloc(io.File, 1);
    files[0] = .{ .trailer = .{ .kind = @backingInt(file.Kind.state), .payload = 0, .payload_hash = 0, .name = prepared.name, .at = kept.at, .block = r.keying.block, .flags = @intFromBool(kept.shared) }, .payload = try a.alloc(u8, @intCast(image.total(prepared.own))) };
    if (files[0].payload.len > 0) try r.img.gather(r.family.ptr, kept.saved, kept.pages, prepared.own, files[0].payload);
    const tokens = try a.dupe(u32, kept.tokens);
    const spans = try a.dupe(@import("cache_modes.zig").Span, kept.spans);
    const starts = try a.dupe(u32, prepared.starts);
    const counts = try a.create([3]u32);
    counts.* = .{ @intCast(tokens.len), @intCast(spans.len), @intCast(starts.len) };
    files[0].meta = try a.dupe([]const u8, &.{ std.mem.asBytes(counts), std.mem.sliceAsBytes(tokens), std.mem.sliceAsBytes(spans), std.mem.sliceAsBytes(starts) });
    const wants = try a.dupe(Want, prepared.wants);
    r.next_job +%= 1;
    j.* = .{ .arena = arena, .id = r.next_job, .generation = w.generation.load(.acquire), .name = prepared.name, .at = kept.at, .shared = kept.shared, .tokens = tokens, .spans = spans, .starts = starts, .wants = wants, .files = files, .bytes = own_bytes, .staged = charge, .final = true, .new_blocks = new_blocks, .verify = r.verify, .began = prepared.began };
    w.submit(j);
}

/// The whole state as one job, gathered and submitted now; false when the staging permit cannot hold it, so its
/// save goes in pieces (the budget, the cap or the floor refuse the save as they always have).
fn whole(r: *Rows, w: *io.Worker, prepared: *const Job, kept: Kept, promised: *io.Promised, metadata: usize, _: bool) !bool {
    var bytes: u64 = image.total(prepared.own) + metadata + @sizeOf(file.Trailer);
    var count: usize = 1;
    var charge: u64 = 0;
    while (true) {
        count = 1;
        bytes = image.total(prepared.own) + metadata + @sizeOf(file.Trailer);
        for (prepared.wants) |want| if (lacks(r, promised, want)) {
            bytes += want.bytes;
            count += 1;
        };
        charge = bytes + @sizeOf(io.Job) + count * @sizeOf(io.File) + prepared.wants.len * @sizeOf(Want) + 4 * @sizeOf([]const u8);
        if (!w.fits(charge)) return false;
        const reserved = w.reserved();
        if (!r.budget(bytes + reserved)) return error.DailyBudgetSpent;
        if (!r.room(bytes + reserved)) return error.OverCap;
        try w.promised(r.gpa, promised); // a fence inside room frees the blocks cancelled jobs promised
        var again: u64 = image.total(prepared.own) + metadata + @sizeOf(file.Trailer);
        var again_count: usize = 1;
        for (prepared.wants) |want| if (lacks(r, promised, want)) {
            again += want.bytes;
            again_count += 1;
        };
        if (again == bytes and again_count == count) break;
    }
    if (r.admission.reserve(r.dir, bytes) != .ready) return error.UnderFreeDiskFloor;
    errdefer r.admission.finish(r.dir, bytes, true);
    const j = try io.allocator.create(io.Job);
    errdefer io.allocator.destroy(j);
    var arena = std.heap.ArenaAllocator.init(io.allocator);
    errdefer arena.deinit();
    const a = arena.allocator();
    const files = try a.alloc(io.File, count);
    var built: usize = 0;
    for (prepared.wants) |want| if (lacks(r, promised, want)) {
        var ranges: std.ArrayList(image.Range) = .empty;
        defer ranges.deinit(r.gpa);
        try image.blockRanges(prepared.parts, r.keying.block, want.window, want.k, &ranges, r.gpa);
        files[built] = .{ .trailer = .{ .kind = @backingInt(file.Kind.block), .payload = 0, .payload_hash = 0, .name = want.name, .block = r.keying.block }, .payload = try a.alloc(u8, @intCast(image.total(ranges.items))) };
        try r.img.gather(r.family.ptr, kept.saved, kept.pages, ranges.items, files[built].payload);
        built += 1;
    };
    const own = &files[built];
    own.* = .{ .trailer = .{ .kind = @backingInt(file.Kind.state), .payload = 0, .payload_hash = 0, .name = prepared.name, .at = kept.at, .block = r.keying.block, .flags = @intFromBool(kept.shared) }, .payload = try a.alloc(u8, @intCast(image.total(prepared.own))) };
    if (own.payload.len > 0) try r.img.gather(r.family.ptr, kept.saved, kept.pages, prepared.own, own.payload);
    const tokens = try a.dupe(u32, kept.tokens);
    const spans = try a.dupe(@import("cache_modes.zig").Span, kept.spans);
    const starts = try a.dupe(u32, prepared.starts);
    const counts = try a.create([3]u32);
    counts.* = .{ @intCast(tokens.len), @intCast(spans.len), @intCast(starts.len) };
    own.meta = try a.dupe([]const u8, &.{ std.mem.asBytes(counts), std.mem.sliceAsBytes(tokens), std.mem.sliceAsBytes(spans), std.mem.sliceAsBytes(starts) });
    const wants = try a.dupe(Want, prepared.wants);
    j.* = .{ .arena = arena, .id = r.next_job, .generation = w.generation.load(.acquire), .name = prepared.name, .at = kept.at, .shared = kept.shared, .tokens = tokens, .spans = spans, .starts = starts, .wants = wants, .files = files, .bytes = bytes, .staged = charge, .final = true, .new_blocks = @intCast(built), .verify = r.verify, .began = prepared.began };
    w.submit(j);
    return true;
}

/// One piece of a state's missing blocks, gathered now and submitted (its bytes reserved under the free-disk floor).
fn piece(r: *Rows, w: *io.Worker, prepared: *const Job, kept: Kept, promised: *io.Promised, from: usize, end: usize, bytes_in: u64, count_in: usize, may_wait: bool) !void {
    var bytes = bytes_in;
    var count = count_in;
    var charge: u64 = 0;
    while (true) {
        charge = bytes + @sizeOf(io.Job) + count * @sizeOf(io.File) + 4 * @sizeOf([]const u8);
        if (!w.fits(charge)) return if (may_wait and w.busy(charge)) error.StagingBusy else error.StagingFull;
        const reserved = w.reserved();
        if (!r.budget(bytes + reserved)) return error.DailyBudgetSpent;
        if (!r.room(bytes + reserved)) return error.OverCap;
        try w.promised(r.gpa, promised); // a fence inside room frees the blocks cancelled jobs promised
        var again: u64 = 0;
        var again_count: usize = 0;
        for (prepared.wants[from..end]) |want| if (lacks(r, promised, want)) {
            again += want.bytes;
            again_count += 1;
        };
        if (again_count == count and again == bytes) break;
        if (again_count == 0) return; // the piece's blocks all went to another job in flight: the next piece's
        bytes = again;
        count = again_count;
    }
    if (r.admission.reserve(r.dir, bytes) != .ready) return error.UnderFreeDiskFloor;
    errdefer r.admission.finish(r.dir, bytes, true);
    const j = try io.allocator.create(io.Job);
    errdefer io.allocator.destroy(j);
    var arena = std.heap.ArenaAllocator.init(io.allocator);
    errdefer arena.deinit();
    const a = arena.allocator();
    const files = try a.alloc(io.File, count);
    var built: usize = 0;
    var wants: std.ArrayList(Want) = .empty;
    errdefer wants.deinit(r.gpa);
    for (prepared.wants[from..end]) |want| if (lacks(r, promised, want)) {
        var ranges: std.ArrayList(image.Range) = .empty;
        defer ranges.deinit(r.gpa);
        try image.blockRanges(prepared.parts, r.keying.block, want.window, want.k, &ranges, r.gpa);
        files[built] = .{ .trailer = .{ .kind = @backingInt(file.Kind.block), .payload = 0, .payload_hash = 0, .name = want.name, .block = r.keying.block }, .payload = try a.alloc(u8, @intCast(image.total(ranges.items))) };
        try r.img.gather(r.family.ptr, kept.saved, kept.pages, ranges.items, files[built].payload);
        built += 1;
        try wants.append(r.gpa, want);
    };
    const want_slice = a.dupe(Want, wants.items) catch |err| {
        wants.deinit(r.gpa);
        return err;
    };
    wants.deinit(r.gpa);
    r.next_job +%= 1;
    j.* = .{ .arena = arena, .id = r.next_job, .generation = w.generation.load(.acquire), .name = prepared.name, .at = kept.at, .shared = kept.shared, .tokens = &.{}, .spans = &.{}, .starts = &.{}, .wants = want_slice, .files = files, .bytes = bytes, .staged = charge, .final = false };
    w.submit(j);
}

/// Whether the state's block is for this job to write: the disk lacks it and no job in flight is writing it.
fn lacks(r: *const Rows, promised: *const io.Promised, want: Want) bool {
    const n = chain.stem(want.name);
    return !r.blocks.contains(n) and !promised.contains(n);
}
