//! GLM-5.3-Flash's KV pages on CUDA: the core pool's pages in one device arena, and the paged sequences that map them.
const std = @import("std");
const cuda = @import("cuda");
const kv = @import("core").kv_pool;
const fw = @import("cuda_forward.zig");
const state = @import("cuda_state.zig");
const types = @import("cuda_engine_types.zig");
const Engine = @import("cuda_engine.zig").Engine;
const kvs = @import("cuda_kv_shard.zig");
const kv_parts = @import("lanes").kv_parts;

/// Table entries a sequence of `capacity` rows takes.
pub fn tableLen(capacity: usize) usize {
    return (capacity + state.page_rows - 1) / state.page_rows;
}

/// What a stream of engine `e` holds: its fixed part and table (and `extra`, its other buffers), and a token's bytes
/// on the pages: 13,056 in bf16 with the target MTP head, 6,380 in E4M3 beside DFlash2 (no head part).
/// A layout of no token bytes (no DSA layer, no target MTP head) is error.NoPagedCaches: the pool divides by its page.
/// Sharded (TF_GLM_KV_SHARD): a token's bytes are every page's pooled keys and half its latent rows (3,542 on
/// GLM-5.3-Flash beside DFlash2), and a stream holds a second (latent) table.
pub fn layout(gpa: std.mem.Allocator, e: *const Engine, extra: u64) !kv.Layout {
    if (!pageable(e)) return error.NoPagedCaches;
    const sharded = e.settings.kv_shard.on;
    const fixed = try state.State.pagedBytesFor(gpa, &e.w, e.capacity, tableLen(e.capacity), e.mtp, sharded);
    const page = pageBytes(e);
    const token = if (sharded) geometry(e).tokenBytes() else page / state.page_rows;
    return .{ .fixed_bytes = fixed + extra, .token_bytes = token, .page_tokens = state.page_rows, .snap_bytes = 0 };
}

/// The sharded page's parts for engine `e`: its DSA layers' latent rows and pooled keys (the MTP head is refused).
pub fn geometry(e: *const Engine) kvs.Geometry {
    return kvs.Geometry.of(&e.w.config, state.mlaLayers(&e.w));
}

/// A page's bytes for engine `e`: its cache format's parts, the target MTP head's only when it is loaded.
pub fn pageBytes(e: *const Engine) u64 {
    return state.pageBytesOf(&e.w, e.settings.kv8, e.mtp);
}

/// Whether engine `e` has caches to page: a thin model of KDA layers alone (TF_GLM_LAYERS) without the target MTP head
/// has none, so its sequences keep their windows (of no KV rows) whatever TF_KV_PAGES says.
pub fn pageable(e: *const Engine) bool {
    return pageBytes(e) > 0;
}

/// A paged sequence of engine `e`'s format on the pages at `at`.
fn pagedState(e: *const Engine, at: state.Paged) !fw.State {
    return fw.State.initPagedFormat(e.gpa, e.d, &e.w, e.capacity, at, e.settings.kv8, e.mtp);
}

/// A request's need: its prompt, its reply and one last verify window.
pub fn need(l: kv.Layout, prompt: usize, max_tokens: usize) usize {
    return l.need(prompt, max_tokens, fw.window_rows);
}

/// The engine's own sequence (slot 0) when it pages, bound: its fixed part and table, no rows until its pool places it.
pub fn ownSequence(e: *Engine) !void {
    if (e.seqs.items.len == 0) try e.seqs.append(e.gpa, null);
    if (e.seqs.items[0] != null) return error.SlotTaken;
    const q = try e.gpa.create(types.Seq);
    errdefer e.gpa.destroy(q);
    q.* = .{ .st = try pagedState(e, if (e.settings.kv_shard.on) onShard(e, 0, 0) else on(e, 0, pageBytes(e))) };
    e.seqs.items[0] = q;
    e.cur = q;
    e.slot = 0;
}

/// A sequence's place on the pages at `kv`: a table as long as a stream's window.
fn on(e: *const Engine, kv_at: u64, page: u64) state.Paged {
    return .{ .kv = kv_at, .pages = tableLen(e.capacity), .page_bytes = page };
}

/// Sharded: every page's pooled keys at `keys`, this rank's latent pages at `lat`, a latent table beside the table.
fn onShard(e: *const Engine, keys: u64, lat: u64) state.Paged {
    const g = geometry(e);
    return .{ .kv = keys, .pages = tableLen(e.capacity), .page_bytes = g.keyPage(), .lat = lat, .lat_page = g.latPage(), .shard = true };
}

/// A sharded pool's own parts on a rank (TF_GLM_KV_SHARD): the two parts' leases (rank 0's) or mapped ids, every
/// page's keys, this rank's latent pages and its trash page, each slot's latent pages, a begin's ids as they cross.
pub const Shard = struct {
    parts: kvs.Parts,
    g: kvs.Geometry,
    rank: usize,
    switches: kvs.Switches,
    keys: ?cuda.DeviceBuffer = null,
    lat: ?cuda.DeviceBuffer = null,
    lids: []std.ArrayList(u32),
    wire: std.ArrayList(u32) = .empty,
    rounds: u64 = 0, // rank 0: decode rounds, for TF_GLM_KV_SHARD_CHECK's digest
};

pub const Pages = struct {
    gpa: std.mem.Allocator,
    e: *Engine,
    layout: kv.Layout,
    pool: kv.Pool, // rank 0 takes and gives back pages; the other ranks map the ids it sends
    arena: ?cuda.DeviceBuffer,
    leases: []kv.Lease, // slot i's: rank 0's lease, or the ids another rank mapped
    used: []bool,
    leader: bool,
    poison: bool, // TF_KV_POISON=1: a stream's pages start as NaN, so a row read before its write shows in the reply
    waiting: ?usize = null, // rank 0: the pages the request admission last held back needs (one log line a wait)
    shard: ?*Shard = null, // TF_GLM_KV_SHARD: the pool's two parts (then `pool` holds no page)

    /// `streams` sequences from slot 0 on `pages` pages: the engine's own (`ownSequence`) placed, the others made now.
    pub fn init(gpa: std.mem.Allocator, e: *Engine, l: kv.Layout, streams: usize, pages: usize, leader: bool) !*Pages {
        const own = if (e.seqs.items.len > 0) e.seqs.items[0] else null;
        if (own == null or own.?.st.table == 0) return error.OwnNotPaged; // slot 0 holds a window, or none was made
        const p = try gpa.create(Pages);
        errdefer gpa.destroy(p);
        const page = l.pageBytes();
        const sharded = e.settings.kv_shard.on;
        p.* = .{ .gpa = gpa, .e = e, .layout = l, .pool = try kv.Pool.init(gpa, if (sharded) 0 else pages, page), .arena = null, .leases = &.{}, .used = &.{}, .leader = leader, .poison = poisoned() };
        errdefer p.pool.deinit();
        if (pages > 0 and !sharded) p.arena = try cuda.DeviceBuffer.alloc(e.d, pages * page);
        errdefer if (p.arena) |*a| a.free();
        if (sharded) p.shard = try initShard(gpa, e, streams, pages, leader);
        errdefer if (p.shard) |x| freeShard(gpa, x);
        p.leases = try gpa.alloc(kv.Lease, streams);
        errdefer gpa.free(p.leases);
        @memset(p.leases, .{});
        p.used = try gpa.alloc(bool, streams);
        errdefer gpa.free(p.used);
        @memset(p.used, false);
        if (e.seqs.items.len < streams) try e.seqs.appendNTimes(e.gpa, null, streams - e.seqs.items.len);
        const seqs = e.seqs.items[1..@max(streams, 1)];
        var made: usize = 0;
        errdefer for (seqs[0..made]) |*q| {
            q.*.?.st.deinit();
            e.gpa.destroy(q.*.?);
            q.* = null;
        };
        errdefer cuda.completion.drain(e.d, e.stream, "GLM page construction did not drain");
        const at = if (p.shard) |x| onShard(e, x.keys.?.ptr, x.lat.?.ptr) else on(e, if (p.arena) |a| a.ptr else 0, page);
        for (seqs) |*q| {
            if (q.* != null) return error.SlotTaken; // attach comes before any stream
            const s = try e.gpa.create(types.Seq);
            errdefer e.gpa.destroy(s);
            s.* = .{ .st = try pagedState(e, at) };
            q.* = s;
            made += 1;
        }
        try e.stream.synchronize();
        own.?.st.place(&e.w, at); // last: a failure before it leaves the engine's own as it was
        return p;
    }

    /// Another rank: the pool rank 0's `pool` step names; a page of other bytes than rank 0's is error.RankMismatch.
    pub fn follow(gpa: std.mem.Allocator, e: *Engine, streams: usize, pages: usize, page_bytes: u64) !*Pages {
        const l = try layout(gpa, e, 0);
        if (l.pageBytes() != page_bytes) {
            if (!@import("builtin").is_test) std.log.err("rank {d}: a KV page is {d} bytes here and {d} on rank 0", .{ e.share.rank, l.pageBytes(), page_bytes });
            return error.RankMismatch;
        }
        return init(gpa, e, l, streams, pages, false);
    }

    fn poisoned() bool {
        const v = std.c.getenv("TF_KV_POISON") orelse return false;
        return std.mem.eql(u8, std.mem.span(v), "1");
    }

    /// Sharded: the parts on `pages` pages (a part each half of them less the trash page's room), their arenas.
    fn initShard(gpa: std.mem.Allocator, e: *Engine, streams: usize, pages: usize, leader: bool) !*Shard {
        const g = geometry(e);
        const h = kvs.Geometry.half(pages);
        if (h == 0) return error.PoolTooSmall;
        const x = try gpa.create(Shard);
        errdefer gpa.destroy(x);
        x.* = .{ .parts = try kvs.Parts.init(gpa, h, g.keyPage(), streams, leader), .g = g, .rank = e.share.rank, .switches = e.settings.kv_shard, .lids = &.{} };
        errdefer x.parts.deinit();
        x.lids = try gpa.alloc(std.ArrayList(u32), streams);
        @memset(x.lids, .empty);
        errdefer gpa.free(x.lids);
        x.keys = try cuda.DeviceBuffer.alloc(e.d, 2 * h * g.keyPage());
        errdefer x.keys.?.free();
        x.lat = try cuda.DeviceBuffer.alloc(e.d, (h + 1) * g.latPage()); // the trash page last
        return x;
    }

    fn freeShard(gpa: std.mem.Allocator, x: *Shard) void {
        if (x.lat) |*b| b.free();
        if (x.keys) |*b| b.free();
        for (x.lids) |*l| l.deinit(gpa);
        gpa.free(x.lids);
        x.wire.deinit(gpa);
        x.parts.deinit();
        gpa.destroy(x);
    }

    /// The pool and the pages' arena; the paged sequences are the engine's (it frees every slot).
    pub fn deinit(p: *Pages) void {
        cuda.completion.drain(p.e.d, p.e.stream, "GLM page ownership did not drain");
        if (p.shard) |x| freeShard(p.gpa, x);
        for (p.leases, p.used) |*l, u| {
            if (p.leader and u and p.shard == null) p.pool.finish(l) else l.ids.deinit(p.gpa);
        }
        p.gpa.free(p.leases);
        p.gpa.free(p.used);
        if (p.arena) |*a| a.free();
        p.pool.deinit();
        p.gpa.destroy(p);
    }

    /// Whether `slot` is one of the paged sequences.
    pub fn paged(p: *const Pages, slot: u32) bool {
        return slot < p.used.len;
    }

    /// Whether a request's pages fit now and a paged sequence is free for it. One that needs more pages than the whole
    /// pool holds is admitted at once, so its opening refuses it (error.PoolTooSmall) instead of it waiting forever.
    pub fn fits(p: *Pages, prompt: usize, max_tokens: usize) bool {
        const n = need(p.layout, prompt, max_tokens);
        const free_seq = for (p.used) |u| {
            if (!u) break true;
        } else false;
        if (p.shard) |x| return free_seq and x.parts.fits(n); // a stream's pages lie in one part
        const ok = free_seq and (n > p.pool.total() or p.pool.fits(n));
        p.noteWait(n, prompt, max_tokens, ok, free_seq);
        return ok;
    }

    /// One line when admission starts holding a request back, saying for what, and one when it fits again.
    fn noteWait(p: *Pages, n: usize, prompt: usize, max_tokens: usize, ok: bool, free_seq: bool) void {
        if (@import("builtin").is_test) return;
        if (ok) { // the held-back request's own admission ends its wait (another request fitting beside it does not)
            if (p.waiting == n) {
                std.log.info("kv pool: the request waiting for {d} pages fits now ({d} free of {d})", .{ n, p.pool.available(), p.pool.total() });
                p.waiting = null;
            }
            return;
        }
        if (p.waiting == n) return;
        p.waiting = n;
        if (!free_seq) return std.log.info("kv pool: a request ({d} prompt + {d} reply tokens, {d} pages) waits for a stream: all {d} paged sequences are busy", .{ prompt, max_tokens, n, p.used.len });
        std.log.info("kv pool: a request ({d} prompt + {d} reply tokens) waits for {d} pages: {d} free and {d} reclaimable of {d}, while the running streams hold the rest", .{ prompt, max_tokens, n, p.pool.available(), p.pool.reclaimable(), p.pool.total() });
    }

    /// Rank 0: paged slot `slot` takes the pages of a prompt, its reply and a last window; LaneMemory: not yet.
    pub fn open(p: *Pages, slot: u32, prompt: usize, max_tokens: usize) ![]const u32 {
        return p.openFor(slot, prompt, max_tokens, null);
    }

    /// `open`, sharded in `home` (a kept state's part) when it has the room (cuda_kv_shard.Parts.choose), with a twin
    /// of the prompt's pages in the other part under TF_GLM_KV_SHARD_TWINS when it has them. The ids that cross with
    /// the begin step: the stream's, then its twin's (`twinCount`).
    pub fn openFor(p: *Pages, slot: u32, prompt: usize, max_tokens: usize, home: ?u8) ![]const u32 {
        if (!p.paged(slot) or p.used[slot]) return error.SlotTaken;
        if (p.shard) |x| {
            const n = need(p.layout, prompt, max_tokens);
            if (n > tableLen(p.e.capacity)) return error.PromptTooLong;
            if (n > x.parts.h) return error.PoolTooSmall; // a stream's pages lie in one part
            const part = x.parts.choose(n, home) orelse return error.LaneMemory;
            try x.parts.open(slot, part, n, if (x.switches.twins) p.layout.pages(prompt) else 0);
            errdefer x.parts.close(slot);
            try p.mappedShard(slot, true);
            const sl = &x.parts.slots[slot];
            x.wire.clearRetainingCapacity();
            try x.wire.appendSlice(p.gpa, sl.ids.items);
            try x.wire.appendSlice(p.gpa, sl.twin.items);
            return x.wire.items;
        }
        const n = need(p.layout, prompt, max_tokens);
        if (n > tableLen(p.e.capacity)) return error.PromptTooLong;
        if (n > p.pool.total()) return error.PoolTooSmall; // never fits, even with every other stream done
        const l = &p.leases[slot];
        try p.pool.promise(l, n);
        errdefer p.pool.finish(l);
        var up: Upload = .{ .p = p, .slot = slot };
        try p.pool.cover(l, n, .{ .ctx = &up, .write = Upload.write });
        try p.mapped(slot, l.ids.items);
        return l.ids.items;
    }

    /// Another rank: paged slot `slot` maps rank 0's page ids (sent with the stream's begin step).
    pub fn map(p: *Pages, slot: u32, ids: []const u32) !void {
        return p.mapWith(slot, ids, 0);
    }

    /// `map` of a begin step whose last `twin` ids are a sharded stream's twin (cuda_follow: the header's depth).
    pub fn mapWith(p: *Pages, slot: u32, all: []const u32, twin: usize) !void {
        if (p.shard) |x| {
            if (!p.paged(slot) or twin > all.len or p.used[slot]) return error.BadStep;
            const ids = all[0 .. all.len - twin];
            if (ids.len == 0 or ids.len > tableLen(p.e.capacity)) return error.BadStep;
            try x.parts.map(slot, ids, all[ids.len..]);
            errdefer x.parts.close(slot);
            return p.mappedShard(slot, true);
        }
        if (twin != 0) return error.BadStep;
        const ids = all;
        if (!p.paged(slot) or ids.len == 0 or ids.len > tableLen(p.e.capacity)) return error.BadStep;
        if (p.used[slot]) return error.BadStep;
        for (ids) |id| if (id >= p.pool.total()) return error.BadStep;
        const l = &p.leases[slot];
        l.ids.clearRetainingCapacity();
        try l.ids.appendSlice(p.gpa, ids);
        errdefer l.ids.clearRetainingCapacity();
        var up: Upload = .{ .p = p, .slot = slot };
        try Upload.write(&up, l.ids.items, 0);
        try p.mapped(slot, l.ids.items);
    }

    /// The slot's sequence reaches its pages' rows (NaN first under TF_KV_POISON).
    fn mapped(p: *Pages, slot: u32, ids: []const u32) !void {
        // every id names a page of this pool, on every rank (the kernels read pt[pos / 512] for pos under the capacity
        // set below, and State.ready refuses a pass past it), so no table can send a kernel past the pages' arena
        for (ids) |id| if (id >= p.pool.total()) return error.BadStep;
        const st = &p.e.seqs.items[slot].?.st;
        const page = p.layout.pageBytes();
        if (p.poison) for (ids) |id| try p.e.d.check(p.e.d.api.cuMemsetD32Async(p.arena.?.ptr + id * page, 0xFFFF_FFFF, page / 4, p.e.stream.handle), "cuMemsetD32Async");
        st.pages = ids;
        st.capacity = @min(ids.len * state.page_rows, p.e.capacity);
        p.used[slot] = true;
    }

    /// The stream on paged slot `slot` ended: its sequence reaches no rows, and rank 0's lease gives its pages back.
    pub fn close(p: *Pages, slot: u32) void {
        if (!p.paged(slot) or !p.used[slot]) return;
        const st = &p.e.seqs.items[slot].?.st;
        st.pages = &.{};
        st.capacity = 0;
        p.used[slot] = false;
        if (p.shard) |x| {
            st.lpages = &.{};
            st.hold = null;
            x.parts.close(slot);
            x.lids[slot].clearRetainingCapacity();
            return;
        }
        if (!p.leader) return p.leases[slot].ids.clearRetainingCapacity();
        p.pool.finish(&p.leases[slot]);
        for (p.used) |u| if (u) return;
        p.report();
    }

    /// Sharded: slot `slot`'s pages mapped on this rank: its table (key pages) and its latent table (this rank's own
    /// pages, a twin's, or the trash page), uploaded, and who holds its latent rows; `fresh` (a new stream): the
    /// pages NaN-filled first under TF_KV_POISON.
    fn mappedShard(p: *Pages, slot: u32, fresh: bool) !void {
        const x = p.shard.?;
        const sl = &x.parts.slots[slot];
        const ids = sl.ids.items;
        const lids = &x.lids[slot];
        lids.clearRetainingCapacity();
        try lids.resize(p.gpa, ids.len);
        kvs.latentIds(x.rank, x.parts.h, ids, sl.twin.items, lids.items);
        const st = &p.e.seqs.items[slot].?.st;
        var up: Upload = .{ .p = p, .slot = slot };
        try Upload.write(&up, ids, 0);
        var lup: Upload = .{ .p = p, .slot = slot, .latent = true };
        try Upload.write(&lup, lids.items, 0);
        if (p.poison and fresh) for (ids, lids.items) |id, lid| {
            try p.e.d.check(p.e.d.api.cuMemsetD32Async(x.keys.?.ptr + id * x.g.keyPage(), 0xFFFF_FFFF, x.g.keyPage() / 4, p.e.stream.handle), "cuMemsetD32Async");
            if (lid != x.parts.h) try p.e.d.check(p.e.d.api.cuMemsetD32Async(x.lat.?.ptr + lid * x.g.latPage(), 0xFFFF_FFFF, x.g.latPage() / 4, p.e.stream.handle), "cuMemsetD32Async");
        };
        st.pages = ids;
        st.lpages = lids.items;
        st.hold = sl.hold();
        st.capacity = @min(ids.len * state.page_rows, p.e.capacity);
        p.used[slot] = true;
    }

    /// The ids a sharded begin step carries after the stream's own: its twin's (0 unsharded or without one).
    pub fn twinCount(p: *const Pages, slot: u32) usize {
        const x = p.shard orelse return 0;
        if (!p.paged(slot)) return 0;
        return x.parts.slots[slot].twin.items.len;
    }

    /// Every rank, where a prompt pass ends (its last chunk committed): slot `slot`'s twin goes back, and this rank's
    /// latent table points at its own pages or the trash page again. No-op unsharded or without a twin.
    pub fn giveTwin(p: *Pages, slot: u32) !void {
        const x = p.shard orelse return;
        if (!p.paged(slot) or !p.used[slot]) return;
        if (!x.parts.giveBack(slot)) return;
        try p.mappedShard(slot, false); // its rows stay: no TF_KV_POISON fill
    }

    /// Sharded: every live stream's place hashed (rank 0's sent every TF_GLM_KV_SHARD_CHECK rounds; 0 unsharded).
    pub fn digest(p: *const Pages) u64 {
        const x = p.shard orelse return 0;
        return x.parts.digest();
    }

    /// The parts' stats for /health and /metrics (lanes.kv_parts.Source's read).
    pub fn readStats(ctx: *anyopaque, out: *kv_parts.Stats) void {
        const p: *const Pages = @ptrCast(@alignCast(ctx));
        const x = p.shard orelse return;
        x.parts.stats(out);
    }

    /// The idle server's line: nothing promised, every page free (the copy store holds none).
    fn report(p: *const Pages) void {
        if (@import("builtin").is_test) return;
        const c = p.pool.census();
        const ok = p.pool.idle() and c.free == c.total;
        std.log.info("kv pool idle{s}: promised {d}, free {d} + store {d} == total {d}", .{ if (ok) "" else " (UNEVEN)", c.promised, c.free, c.kept, c.total });
    }

    /// A stream's table entries written to its sequence's device table (blocking: the stream waits for the copy);
    /// `latent`: its latent table (sharded).
    const Upload = struct {
        p: *Pages,
        slot: u32,
        latent: bool = false,

        fn write(ctx: *anyopaque, ids: []const u32, from: usize) anyerror!void {
            const u: *Upload = @ptrCast(@alignCast(ctx));
            const e = u.p.e;
            const st = &e.seqs.items[u.slot].?.st;
            const bytes = std.mem.sliceAsBytes(ids[from..]);
            try e.d.check(e.d.api.cuMemcpyHtoD_v2((if (u.latent) st.ltable else st.table) + from * 4, bytes.ptr, bytes.len), "cuMemcpyHtoD");
        }
    };
};
