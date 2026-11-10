//! Prompt-sized snapshots on each rank; the existing chunk grid and the live sequence's layout do not change.
//! A state takes device pages first, then pinned host pages (cuda_snapshot_host); beside DFlash2 it also carries the
//! drafter's context rows up to its mark, so a resumed pass drafts from the context a fresh pass would have.
const std = @import("std");
const cuda = @import("cuda");
const types = @import("cuda_engine_types.zig");
const dflash = @import("dflash.zig");
pub const pages = @import("cuda_snapshot_pages.zig");
pub const layout = @import("cuda_snapshot_layout.zig");
pub const host_tier = @import("cuda_snapshot_host.zig");
const kv_keep = @import("cuda_kv_shard_keep.zig");
const kvs = @import("cuda_kv_shard.zig");

/// TF_GLM_KV_SHARD: which ranks hold the bound stream's latent rows (null: unsharded, or the switch off).
fn holdOf(e: anytype) ?kvs.Hold {
    return if (e.settings.kv_shard.on) e.cur.st.hold else null;
}

pub const Setup = struct { ptr: *anyopaque, run: *const fn (*anyopaque, u64) anyerror!void };

pub fn prepare(e: anytype, budget: u64) !void {
    return prepareTiers(e, budget, 0);
}

/// `budget` bytes of device pages and `host` bytes of pinned host pages, made once.
pub fn prepareTiers(e: anytype, budget: u64, host: u64) !void {
    if (e.snap_pages != null or e.snaps.count() != 0) return error.SnapshotCacheMade;
    const count = std.math.cast(u32, budget / pages.page + host / pages.page) orelse return error.SnapshotBudget;
    try e.snaps.ensureTotalCapacity(e.gpa, count);
    try e.dropped.ensureTotalCapacity(e.gpa, @as(usize, count) + 1);
    e.snap_pages = try pages.Pool.initTiers(e.gpa, e.d, budget, host);
}

/// The pages made by `prepareTiers` freed again (no state holds any): a peer could not make its share.
pub fn unprepare(e: anytype) void {
    if (e.snap_pages) |*p| p.deinit();
    e.snap_pages = null;
}

/// Every rank's device pages at `budget`, beside the host tier: the least any rank may pin of `asked` (auto: its share
/// of the host's available memory), or, should a rank fail to pin that, the device pages alone. Each step is one
/// agreement on every rank, so no peer waits on a missing one; the pinned bytes a rank holds come back (0: none).
pub fn agreeTiers(e: anytype, budget: u64, asked: u64) !u64 {
    return agreeTiersWith(e, budget, asked, if (asked == 0) null else host_tier.available());
}

/// `agreeTiers` with this rank's reading of the host's available memory (null: unread).
pub fn agreeTiersWith(e: anytype, budget: u64, asked: u64, available: ?u64) !u64 {
    if (asked == 0) {
        try agree(e, budget);
        return 0;
    }
    const mine = host_tier.allowance(asked, available, e.share.world);
    const least = -(try e.shared(-@as(f64, @floatFromInt(mine)))); // the ranks' maximum of minus theirs
    const host = host_tier.pages(if (least > 0) @intFromFloat(least) else 0);
    if (host > 0) {
        var failed: ?anyerror = null;
        e.prepareTiers(budget, host) catch |err| {
            failed = err;
        };
        if (try e.shared(if (failed != null) 1 else 0) == 0) return host;
        if (failed == null) e.unprepareSnapshots();
        if (e.share.rank == 0) std.log.warn("prompt cache: a rank could not pin {d} MiB of host memory ({s}); device pages alone", .{ host >> 20, if (failed) |err| @errorName(err) else "a peer's failure" });
    }
    try agree(e, budget);
    return 0;
}

/// A failed allocation still joins the startup agreement, so a peer never waits for a missing participant.
pub fn agree(e: anytype, budget: u64) !void {
    var failed: ?anyerror = null;
    e.prepareSnapshots(budget) catch |err| {
        failed = err;
    };
    if (try e.shared(if (failed != null) 1 else 0) > 0) return failed orelse error.PeerSnapshotCache;
}

/// The target MTP rows a state at `at` holds, as `keepAs` will save them: none without the head; beside a drafter its
/// prompts leave the head's caches empty (cuda_engine prefillChunk), so mtp_len stays 0; else up to a window past `at`.
fn mtpHead(e: anytype, at: usize) error{Overflow}!?usize {
    if (!e.mtp) return null;
    if (e.drafter != null) return 0;
    return @min(e.capacity, try std.math.add(usize, at, @import("cuda_state.zig").window_rows));
}

pub fn bytes(e: anytype, at: usize) u64 {
    const head = mtpHead(e, at) catch return std.math.maxInt(u64);
    // a paged sequence holds its request's rows: its prefix is counted to the capacity, in a window's bytes
    const n = layout.Plan.sizeForShared(&e.cur.st, &e.c, e.capacity, at, head, heldPages(e, at), true, e.cur.st.cur) catch return std.math.maxInt(u64);
    const ring: usize = if (e.drafter != null) dflash.ring_bytes else 0; // at most: DFlash2's ring when it holds rows
    return pages.rounded(n + ring) catch std.math.maxInt(u64);
}

/// The pages below `at` the shared prompt store holds (TF_KV_SHARE): a kept state's copy leaves them out.
fn heldPages(e: anytype, at: usize) usize {
    return if (e.share_pages) at / @import("cuda_state.zig").page_rows else 0;
}

/// DFlash2's ring for the bound slot as the plan's last part, when it holds the context rows right up to `at`; where
/// those rows start (null: no drafter, or nothing worth keeping).
fn drafterPart(e: anytype, plan: *layout.Plan, at: usize) !?i64 {
    const dr = e.drafter orelse return null;
    if (std.c.getenv("TF_GLM_SNAP_DRAFTER")) |v| if (std.mem.eql(u8, std.mem.span(v), "0")) return null; // A/B: resume from the mark
    const from = dr.keptFrom(e.slot, at) orelse return null;
    try plan.outside(try dr.ringPtr(e.slot), dflash.ring_bytes);
    return from;
}

/// A monotonic clock in nanoseconds, for the copies' one log line.
fn now() u64 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * 1_000_000_000 + @as(u64, @intCast(ts.nsec));
}

/// Rank 0's line for a copy that touched host pages (device copies stay quiet, as before).
fn note(e: anytype, comptime what: []const u8, s: *const types.Snap, t0: u64) void {
    if (e.share.rank != 0 or @import("builtin").is_test) return;
    const host = s.lease.hostBytes(s.len);
    if (host == 0) return;
    std.log.info("prompt cache: " ++ what ++ " {d} tokens, {d} MiB ({d} MiB in host memory{s}) in {d:.1} ms", .{ s.pos, s.len >> 20, host >> 20, if (s.ring_from != null) ", with DFlash2's context" else "", @as(f64, @floatFromInt(now() - t0)) / 1e6 });
}

pub fn keep(e: anytype) !u64 {
    if (e.next_snap > std.math.maxInt(u32)) return error.SnapshotIdLimit;
    const id = e.next_snap;
    try keepAs(e, id);
    e.next_snap += 1;
    e.announce = id;
    return id;
}

pub fn keepAs(e: anytype, id: u64) !void {
    if (e.snaps.contains(id)) return error.StateKept;
    if (e.cur.st.pending != 0 or e.cur.st.mtp_len > e.cur.st.pos + @import("cuda_state.zig").window_rows)
        return error.NotAtMark;
    const pool = if (e.snap_pages) |*p| p else return error.NoSnapshotCache;
    const hold = holdOf(e);
    const head = if (e.mtp) e.cur.st.mtp_len else null;
    // TF_GLM_KV_SHARD: the latent rows on their owner's copy alone; every rank leases the whole state's bytes.
    // TF_KV_SHARE: the shared store's pages below the mark stay out (the state's copy resumes from them by reference).
    const own = kv_keep.keepsLatent(hold, e.share.rank);
    const held = heldPages(e, e.cur.st.pos);
    var plan = try layout.Plan.initForShared(&e.cur.st, &e.c, e.cur.st.pos, head, held, own, e.cur.st.cur);
    const latent_bytes = if (own) 0 else (try layout.Plan.initForShared(&e.cur.st, &e.c, e.cur.st.pos, head, held, true, e.cur.st.cur)).len - plan.len;
    const ring = try drafterPart(e, &plan, e.cur.st.pos);
    var lease = try pool.take(plan.len + latent_bytes);
    errdefer {
        cuda.completion.drain(e.d, e.stream, "a failed GLM snapshot copy did not drain");
        pool.give(&lease);
    }
    const cur = try e.gpa.dupe(u1, e.cur.st.cur);
    errdefer e.gpa.free(cur);
    const t0 = now();
    try pool.move(e.stream, &lease, &plan, true);
    try e.stream.synchronize();
    const s: types.Snap = .{ .lease = lease, .len = plan.len, .pos = e.cur.st.pos, .mtp_len = e.cur.st.mtp_len, .cur = cur, .bases = .{ e.cur.st.base, e.cur.st.mtp_base }, .shared = plan.shared, .ring_from = ring, .home = if (hold) |h| h.owner else null };
    note(e, "kept", &s, t0);
    e.snaps.putAssumeCapacityNoClobber(id, s);
}

/// A disk read's state: `at`'s pages leased under rank 0's id on each rank, refused by `restore` until `seal`.
/// `home`: TF_GLM_KV_SHARD's home part, recorded on every rank as a keep records it (null: unsharded); every rank
/// leases the whole size, as a keep does. `shared`: the pages layout's store pages under the mark (TF_KV_SHARE's
/// recall; its slot's parts past them on every rank, so a rank's restore leaves the rows to the pages).
pub fn blankAs(e: anytype, id: u64, at: usize, home: ?u8, shared: usize) !void {
    if (e.snaps.contains(id)) return error.StateKept;
    const pool = if (e.snap_pages) |*p| p else return error.NoSnapshotCache;
    var lease = try pool.take(bytes(e, at));
    errdefer pool.give(&lease);
    const cur = try e.gpa.alloc(u1, e.cur.st.cur.len);
    @memset(cur, 0);
    e.snaps.putAssumeCapacityNoClobber(id, .{ .lease = lease, .len = 0, .pos = at, .mtp_len = 0, .cur = cur, .bases = .{ 0, 0 }, .shared = shared, .home = home });
}

/// Whether an index ring of `rows` rows based at `base` holds every row up to `len`, as the image's ring bytes do.
fn ringHolds(base: usize, len: usize, rows: usize) bool {
    return base <= len and len - base <= rows;
}

/// A blank state made whole from its image's words (its bytes are in or follow): a restore takes it from here.
pub fn seal(e: anytype, id: u64, mtp_len: usize, bases: [2]usize, cur: u64) !void {
    const s = e.snaps.getPtr(id) orelse return error.NoKeptState;
    if (s.len != 0) return error.StateKept;
    if (s.cur.len > 64 or (s.cur.len < 64 and cur >> @intCast(s.cur.len) != 0)) return error.DiskWords;
    // the words' parity first: the blank's is zeros, and the state's size and half follow the keep's, not the slot's
    for (s.cur, 0..) |*b, i| b.* = @intCast(cur >> @intCast(i) & 1);
    // TF_GLM_KV_LIVE_HALF: the disk image lays a state's parts out as a live-half keep's lease, whose dead half
    // has no bytes; a state whose words say both halves (a keep with layers on both) has no such image here
    if (layout.liveHalfEnabled() and layout.halfOf(s.cur, e.cur.st.kda_layers, true) == null) return error.DiskWords;
    const head = if (e.mtp) @min(e.capacity, s.pos + @import("cuda_state.zig").window_rows) else 0;
    if (mtp_len > head) return error.DiskWords;
    const rows = e.cur.st.ring_rows;
    if (!ringHolds(bases[0], s.pos, rows) or !ringHolds(bases[1], mtp_len, rows)) return error.DiskWords;
    // TF_GLM_KV_SHARD: off its home rank the copy packs every part but the latent rows, as keepAs leaves it
    const own = if (s.home) |h| h == e.share.rank else true;
    // a state whose leading rows are the shared store's pages (TF_KV_SHARE's recall): its slot's parts past them
    const n = if (s.shared > 0)
        try layout.Plan.sizeForShared(&e.cur.st, &e.c, e.capacity, s.pos, if (e.mtp) mtp_len else null, s.shared, own, s.cur)
    else
        try layout.Plan.sizeFor(&e.cur.st, &e.c, e.capacity, s.pos, if (e.mtp) mtp_len else null, own, s.cur);
    if (n == 0 or n > s.lease.count() * pages.page) return error.SnapshotSize;
    s.len = n;
    s.mtp_len = mtp_len;
    s.bases = bases;
}

pub fn restore(e: anytype, id: u64) !void {
    const s = e.snaps.getPtr(id) orelse return error.NoKeptState;
    if (s.len == 0) return error.UnsealedState;
    const pool = if (e.snap_pages) |*p| p else return error.NoSnapshotCache;
    const hold = holdOf(e);
    const home: ?u8 = if (e.settings.kv_shard.on) s.home else null;
    const own = if (home) |h| h == e.share.rank else true; // this rank's copy holds the latent rows
    // the kept state's own parity picks the half a live-half keep carries; the slot's dead half stays as it is
    var plan = try layout.Plan.initForShared(&e.cur.st, &e.c, s.pos, if (e.mtp) s.mtp_len else null, s.shared, own, s.cur);
    if (s.ring_from != null) try plan.outside(try (e.drafter orelse return error.SnapshotSize).ringPtr(e.slot), dflash.ring_bytes);
    if (plan.len != s.len) return error.SnapshotSize;
    errdefer cuda.completion.drain(e.d, e.stream, "a failed GLM snapshot restore did not drain");
    const t0 = now();
    try pool.moveParts(e.stream, &s.lease, &plan, false, !own or kv_keep.restoresLatent(home, hold, e.share.rank));
    // TF_GLM_KV_SHARD: the latent rows to the rank that holds the new stream's rows but not the state's copy of them
    switch (kv_keep.carry(home, hold, e.share.rank)) {
        .none => {},
        .send => try carrySend(e, s, &plan),
        .recv => try carryRecv(e, s),
    }
    try e.cur.st.restoreHost(e.stream, s.pos, s.mtp_len, s.cur, s.bases);
    e.cur.in_head = 0;
    try e.stream.synchronize();
    // DFlash2 drafts on from the kept context rows, as a fresh pass would; without them from the mark (never tapped)
    if (e.drafter) |dr| if (s.ring_from) |from| try dr.resumeAt(e.slot, from, s.pos) else try dr.restart(e.slot, s.pos);
    note(e, "restored", s, t0);
}

/// The latent parts' lengths of `plan`, in order (a carry's pieces are cut from them alike on both ranks).
fn latentLens(plan: *const layout.Plan, out: []usize) []usize {
    var n: usize = 0;
    for (plan.parts[0..plan.count]) |part| if (part.lat) {
        out[n] = part.len;
        n += 1;
    };
    return out[0..n];
}

/// The home rank's side of a carry: each latent part of its copy, a piece at a time, into the staging buffer and to
/// the peer (`plan`: its copy's whole layout, the parts at their offsets in the lease).
fn carrySend(e: anytype, s: *const types.Snap, plan: *const layout.Plan) !void {
    const stage = (e.kv_stage orelse return error.KvShardRoom).ptr;
    var offs: [layout.Plan.max_parts]usize = undefined;
    var lens: [layout.Plan.max_parts]usize = undefined;
    var off: usize = 0;
    var n: usize = 0;
    for (plan.parts[0..plan.count]) |part| {
        if (part.lat) {
            offs[n] = off;
            lens[n] = part.len;
            n += 1;
        }
        off += part.len;
    }
    const cut = try kv_keep.pieces(e.gpa, lens[0..n], kv_keep.piece_bytes);
    defer e.gpa.free(cut);
    const peer: c_int = @intCast(1 - e.share.rank);
    var sent: u64 = 0;
    for (cut) |x| {
        try e.snap_pages.?.copyOut(e.stream, &s.lease, offs[x[0]] + x[1], stage, x[2]);
        try e.lib.check(e.lib.api.ncclSend(stage, x[2], .u8, peer, e.comm, e.stream.handle), "ncclSend");
        sent += x[2];
    }
    if (e.kv_ctx.counts) |c| {
        c.carries += 1;
        c.carry_bytes += sent;
    }
}

/// The other rank's side: its live sequence's latent parts at the state's position (its own pages or a twin's), a
/// piece at a time from the peer into the staging buffer and on into the rows.
fn carryRecv(e: anytype, s: *const types.Snap) !void {
    const stage = (e.kv_stage orelse return error.KvShardRoom).ptr;
    const live = try layout.Plan.initFor(&e.cur.st, &e.c, s.pos, if (e.mtp) s.mtp_len else null, true, e.cur.st.cur);
    var buf: [layout.Plan.max_parts]usize = undefined;
    const lens = latentLens(&live, &buf);
    var parts: [layout.Plan.max_parts]layout.Part = undefined;
    var n: usize = 0;
    for (live.parts[0..live.count]) |part| if (part.lat) {
        parts[n] = part;
        n += 1;
    };
    const cut = try kv_keep.pieces(e.gpa, lens, kv_keep.piece_bytes);
    defer e.gpa.free(cut);
    const peer: c_int = @intCast(1 - e.share.rank);
    for (cut) |x| {
        try e.lib.check(e.lib.api.ncclRecv(stage, x[2], .u8, peer, e.comm, e.stream.handle), "ncclRecv");
        var done: usize = 0;
        while (done < x[2]) {
            const at = live.piece(parts[x[0]], x[1] + done);
            const len = @min(at.len, x[2] - done);
            try e.d.check(e.d.api.cuMemcpyDtoDAsync_v2(at.live, stage + done, len, e.stream.handle), "cuMemcpyDtoDAsync");
            done += len;
        }
    }
}

pub fn holds(e: anytype, id: u64) bool {
    return e.snaps.contains(id);
}

/// A kept state's slot lease bytes [at, at + host.len) read into `host` (out) or written from it, on this rank's
/// engine thread: rank 0's pages-layout fixed parts, and another rank's through the disk step that calls it.
pub fn slotBytes(e: anytype, s: *types.Snap, at: usize, host: []u8, out: bool) !void {
    const pool = if (e.snap_pages) |*p| p else return error.NoSnapshotCache;
    const page = pages.page;
    if (at + host.len > s.lease.count() * page) return error.SnapshotSize;
    const nd = s.lease.dev.ids.items.len;
    var done: usize = 0;
    while (done < host.len) {
        const x = at + done;
        const k: usize = @intCast(x / page);
        const n: usize = @intCast(@min(host.len - done, page - x % page));
        if (k < nd) {
            const base = (pool.arena orelse return error.NoSnapshotRoom).ptr + @as(u64, s.lease.dev.ids.items[k]) * page + x % page;
            if (out) try e.d.check(e.d.api.cuMemcpyDtoH_v2(host[done..].ptr, base, n), "cuMemcpyDtoH") else try e.d.check(e.d.api.cuMemcpyHtoD_v2(base, host[done..].ptr, n), "cuMemcpyHtoD");
        } else {
            const buf = (pool.host orelse return error.NoSnapshotRoom).bytes;
            const from: usize = @as(usize, @intCast(s.lease.host.ids.items[k - nd])) * page + @as(usize, @intCast(x % page));
            const h = buf[from..][0..n];
            if (out) @memcpy(host[done..][0..n], h) else @memcpy(h, host[done..][0..n]);
        }
        done += n;
    }
}

pub fn drop(e: anytype, id: u64) void {
    var kv = e.snaps.fetchRemove(id) orelse return;
    free(e, &kv.value);
    if (e.share.rank == 0 and e.announce == id) {
        e.announce = 0; // the store failed to publish this keep, so no follower has it yet
        return;
    }
    if (e.share.rank == 0 and e.share.world > 1) e.dropped.appendAssumeCapacity(@intCast(id));
}

pub fn free(e: anytype, s: *types.Snap) void {
    e.snap_pages.?.give(&s.lease);
    e.gpa.free(s.cur);
}

/// Odd words release an evicted id before an even word announces its replacement; an idle gate still sends one word.
pub fn gate(e: anytype) !void {
    const leader = e.share.rank == 0;
    var dropped: usize = 0;
    while (true) {
        const word: u64 = if (!leader) 0 else if (dropped < e.dropped.items.len)
            2 * @as(u64, e.dropped.items[dropped]) + 1
        else
            2 * e.announce;
        const shared = try e.shared(@floatFromInt(word));
        if (!std.math.isFinite(shared) or shared < 0 or shared > 2 * @as(f64, std.math.maxInt(u32)) + 1 or @floor(shared) != shared)
            return error.BadSnapshotControl;
        const got: u64 = @intFromFloat(shared);
        if (got & 1 != 0) {
            if (leader) dropped += 1 else e.dropState(got / 2);
            continue;
        }
        if (leader) {
            e.dropped.clearRetainingCapacity();
            e.announce = 0;
        } else if (got > 0) try e.keepAs(got / 2);
        return;
    }
}

/// After all ranks agree to cancel, publish the last completed state before any rank releases or rebinds its slot.
pub fn interrupted(e: anytype, marks: ?types.Marks, at: u32) !void {
    if (marks) |m| if (m.interrupted) |save_prefix| save_prefix(m.ctx, at);
    try gate(e);
}
