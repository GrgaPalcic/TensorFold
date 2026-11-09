//! TF_GLM_KV_SHARD (GLM patch 0118 by Robert Wojciechowski, GitHub @wojo, X @rwojo; docs/glm-port/kv-shard-1104.md):
//! at TP=2 each stream's E4M3 latent rows live on one rank, its owner. The KV page ids are cut in two parts, part r
//! owned by rank r; a stream takes its pages from one part. Every page's pooled index keys stay on both ranks (the
//! indexer runs replicated), so a rank holds every page's keys and its own part's latent rows plus one trash page,
//! where a row it does not hold writes. A prompt pass may borrow the other part's pages (a twin) so both ranks hold its
//! rows while it fills. Host side only: switches, sizes, the parts' leases, latent tables, checks and counters.
const std = @import("std");
const builtin = @import("builtin");
const kv = @import("core").kv_pool;
const kv_parts = @import("lanes").kv_parts;

/// Whether this build runs sharding's device paths (owner-computed attention, twins, carries: cuda_kv_shard_attn,
/// cuda_kv_shard_keep). Without them TF_GLM_KV_SHARD=1 is refused at start, after the ranks agree on it.
pub const gpu_paths = true;

/// TF_GLM_KV_SHARD_CHECK's default: decode rounds between the ranks' layout digests.
pub const check_rounds: u32 = 1024;

/// Rows a page holds (cuda_state.page_rows; one latent chunk).
pub const page_rows: usize = 512;

/// TF_GLM_KV_SHARD ("1": on, "0" or unset: off) and its knobs, as each rank reads them; they ride the ranks'
/// fingerprint (`word`, `check`). Off reads no knob, and nothing changes.
pub const Switches = struct {
    on: bool = false,
    carry: bool = true, // TF_GLM_KV_SHARD_CARRY: a resumed state's rows go to a stream outside its home part
    twins: bool = true, // TF_GLM_KV_SHARD_TWINS: prompt passes borrow the other part's pages (no decode twins)
    check: u32 = check_rounds, // TF_GLM_KV_SHARD_CHECK: decode rounds between layout digests (0: none)

    pub fn parse(on: ?[]const u8, carry: ?[]const u8, twins: ?[]const u8, check: ?[]const u8) !Switches {
        if (!try flag(on, false, error.BadKvShard)) return .{};
        return .{
            .on = true,
            .carry = try flag(carry, true, error.BadKvShardCarry),
            .twins = try flag(twins, true, error.BadKvShardTwins),
            .check = try count(check, check_rounds, error.BadKvShardCheck),
        };
    }

    /// `get`'s environment (cuda_ranks.Settings.read's lookup).
    pub fn read(get: *const fn ([:0]const u8) ?[]const u8) !Switches {
        return parse(get("TF_GLM_KV_SHARD"), get("TF_GLM_KV_SHARD_CARRY"), get("TF_GLM_KV_SHARD_TWINS"), get("TF_GLM_KV_SHARD_CHECK"));
    }

    /// The fingerprint's word: 0 off; on, 1 | carry << 1 | twins << 2.
    pub fn word(s: Switches) u32 {
        if (!s.on) return 0;
        return 1 | @as(u32, @intFromBool(s.carry)) << 1 | @as(u32, @intFromBool(s.twins)) << 2;
    }

    /// The fingerprint's check word: 0 off (the knob is not read then).
    pub fn checkWord(s: Switches) u32 {
        return if (s.on) s.check else 0;
    }

    /// Whether decode round `round` (from 1) sends a layout digest: every `check` rounds (0: never).
    pub fn digestAt(s: Switches, round: u64) bool {
        return s.on and s.check != 0 and round != 0 and round % s.check == 0;
    }
};

/// "0" or "1" (blank or unset: `default`), else `bad`.
fn flag(raw: ?[]const u8, default: bool, bad: anyerror) !bool {
    const t = std.mem.trim(u8, raw orelse "", " \t\r\n");
    if (t.len == 0) return default;
    if (std.mem.eql(u8, t, "1")) return true;
    if (std.mem.eql(u8, t, "0")) return false;
    return bad;
}

/// Decimal digits only, a u32 (blank or unset: `default`), else `bad`.
fn count(raw: ?[]const u8, default: u32, bad: anyerror) !u32 {
    const t = std.mem.trim(u8, raw orelse "", " \t\r\n");
    if (t.len == 0) return default;
    for (t) |c| if (!std.ascii.isDigit(c)) return bad;
    return std.fmt.parseInt(u32, t, 10) catch bad;
}

/// What the start-up refusals read: every input is agreed between the ranks before the weights load.
pub const Facts = struct {
    world: usize,
    kv8: bool, // --kv-cache fp8
    pages: bool, // every sequence on KV pages (TF_KV_PAGES, and the loaded layers page)
    mtp_drafts: bool, // the target MTP head drafts (drafts without --drafter)
    built: bool = gpu_paths,
};

/// The settings sharding cannot serve, refused alike on every rank: two ranks (a part a rank), E4M3 latent rows on
/// pages (the only layout it splits), no target MTP head (its caches are a part of every page its own layer reads).
pub fn refuse(s: Switches, f: Facts) !void {
    if (!s.on) return;
    if (f.world != 2) return error.KvShardNeedsTwoRanks;
    if (!f.kv8) return error.KvShardNeedsFp8;
    if (!f.pages) return error.KvShardNeedsPages;
    if (f.mtp_drafts) return error.KvShardWithMtp;
    if (!f.built) return error.KvShardNotBuilt;
}

/// Rank 0's refusals at the pool's attach: more than one stream (one would leave a part idle), a part long enough.
pub fn refusePool(s: Switches, streams: usize, part_pages: usize, capacity: usize) !void {
    if (!s.on) return;
    if (streams < 2) return error.KvShardNeedsParallel;
    if (part_pages * page_rows < @max(min_part, capacity / 4)) return error.KvShardPartTooSmall;
}

/// A part shorter than this many tokens (or a quarter of the capacity) is refused: the old port's floor.
pub const min_part: usize = 131072;

/// The start-up's words for a refusal of TF_GLM_KV_SHARD (null: not one of its errors).
pub fn why(err: anyerror) ?[]const u8 {
    return switch (err) {
        error.BadKvShard => "TF_GLM_KV_SHARD: 0 (off, the default) or 1",
        error.BadKvShardCarry => "TF_GLM_KV_SHARD_CARRY: 1 (the default) or 0",
        error.BadKvShardTwins => "TF_GLM_KV_SHARD_TWINS: 1 (the default) or 0",
        error.BadKvShardCheck => "TF_GLM_KV_SHARD_CHECK: decode rounds between the ranks' layout digests (1024, the default; 0: none)",
        error.KvShardNeedsTwoRanks => "TF_GLM_KV_SHARD=1 needs two ranks (TP=2)",
        error.KvShardNeedsFp8 => "TF_GLM_KV_SHARD=1 needs --kv-cache fp8 (KV=fp8): it shards the E4M3 latent rows",
        error.KvShardNeedsPages => "TF_GLM_KV_SHARD=1 needs KV pages (TF_KV_PAGES unset or auto, and DSA layers loaded)",
        error.KvShardWithMtp => "TF_GLM_KV_SHARD=1 does not run with the target MTP head drafting (DRAFTER=mtp); DFlash2 is fine",
        error.KvShardNotBuilt => "TF_GLM_KV_SHARD=1: KV sharding's GPU paths are not in this build yet",
        error.KvShardNeedsParallel => "TF_GLM_KV_SHARD=1 needs --parallel above 1",
        error.KvShardPartTooSmall => "TF_GLM_KV_SHARD=1: a part of the KV pool would hold under 131,072 tokens (or a quarter of the context); give the pool more memory or turn sharding off",
        error.KvShardPlacement => "TF_GLM_KV_SHARD=1: a stream's page ids from rank 0 do not lie in one part (its twin's in the other)",
        error.KvShardLayout => "TF_GLM_KV_SHARD=1: the ranks' page layouts differ (TF_GLM_KV_SHARD_CHECK's digest)",
        else => null,
    };
}

// -- sizes ----------------------------------------------------------------------------------------------------------

/// A page's parts a DSA layer: its latent codes and scales (sharded) and its pooled keys (replicated).
pub const Geometry = struct {
    layers: usize, // DSA layers on the pages
    lat_layer: u64, // a page's E4M3 latent codes then fp32 scales, a layer: 512 x (kv_lora + 4)
    key_layer: u64, // a page's 128 pooled keys, a layer: 128 x i_dim x 2

    /// From the config's latent and index widths (`c`: kv_lora, i_dim) and the DSA layers on the pages.
    pub fn of(c: anytype, layers: usize) Geometry {
        return .{ .layers = layers, .lat_layer = page_rows * (@as(u64, c.kv_lora) + 4), .key_layer = page_rows / 4 * @as(u64, c.i_dim) * 2 };
    }

    pub fn latPage(g: Geometry) u64 {
        return g.layers * g.lat_layer;
    }

    pub fn keyPage(g: Geometry) u64 {
        return g.layers * g.key_layer;
    }

    /// An unsharded page's bytes (cuda_state.pageBytesOf without the MTP head).
    pub fn wholePage(g: Geometry) u64 {
        return g.latPage() + g.keyPage();
    }

    /// The pool layout's bytes a token sharded: a page's keys on both ranks, its latent rows on one of the two.
    pub fn tokenBytes(g: Geometry) u64 {
        const lat_half = (g.latPage() + 1) / 2;
        return (g.keyPage() + lat_half + page_rows - 1) / page_rows;
    }

    /// The pages of a part out of `pages` the plan counted at `tokenBytes` a token: four fewer (the trash page and the
    /// halves' rounding fit there), then halved.
    pub fn half(pages: usize) usize {
        return (pages -| 4) / 2;
    }

    /// Bytes a rank allocates for parts of `h` pages: every page's keys, its part's latent rows and the trash page.
    pub fn bytes(g: Geometry, h: usize) u64 {
        return 2 * h * g.keyPage() + (h + 1) * g.latPage();
    }
};

// -- parts ----------------------------------------------------------------------------------------------------------

/// The part of page `id` of a pool cut in parts of `h` pages.
pub fn partOf(id: u32, h: usize) usize {
    return @as(usize, id) / h;
}

/// A rank's latent page for each of a stream's key pages: its own part's page (part-local), the twin's when the stream
/// lies in the other part and this rank holds a twin page for that entry, else the trash page (`h`).
pub fn latentIds(rank: usize, h: usize, ids: []const u32, twin: []const u32, out: []u32) void {
    std.debug.assert(out.len >= ids.len);
    const lo = rank * h;
    for (ids, 0..) |id, i| {
        out[i] = if (partOf(id, h) == rank and !mut(.latent_all_trash))
            @intCast(id - lo)
        else if (i < twin.len and partOf(twin[i], h) == rank)
            @intCast(twin[i] - lo)
        else
            @intCast(h);
    }
}

/// The part a stream's begin step names (all its ids in one part, its twin's all in the other, no longer than it),
/// else KvShardPlacement.
pub fn placement(h: usize, ids: []const u32, twin: []const u32) !usize {
    if (ids.len == 0 or h == 0) return error.KvShardPlacement;
    const part = partOf(ids[0], h);
    if (part > 1) return error.KvShardPlacement;
    for (ids) |id| if (partOf(id, h) != part) return error.KvShardPlacement;
    if (twin.len > ids.len) return error.KvShardPlacement;
    for (twin) |id| if (partOf(id, h) != 1 - part and !mut(.placement_twin_any)) return error.KvShardPlacement;
    return part;
}

/// Who holds a stream's latent rows: its owner part's rank, and with a twin the other rank too.
pub const Hold = struct {
    owner: u8,
    twin: bool = false,

    pub fn has(h: Hold, rank: usize) bool {
        return h.owner == rank or h.twin;
    }
};

/// One paged slot's place: its part, its key page ids (in row order), its twin's (none outside a prompt pass), and on
/// rank 0 the leases in the parts' pools those ids came from.
pub const Slot = struct {
    used: bool = false,
    part: u8 = 0,
    ids: std.ArrayList(u32) = .empty,
    twin: std.ArrayList(u32) = .empty,
    lease: kv.Lease = .{}, // rank 0: in parts.pools[part], part-local ids
    twin_lease: kv.Lease = .{}, // rank 0: in parts.pools[1 - part]

    pub fn hold(s: *const Slot) Hold {
        return .{ .owner = s.part, .twin = s.twin.items.len > 0 };
    }

    fn deinit(s: *Slot, gpa: std.mem.Allocator) void {
        s.ids.deinit(gpa);
        s.twin.deinit(gpa);
    }
};

/// The two parts' page pools (rank 0 takes and gives back pages; the other rank maps the ids it sends) and each paged
/// slot's place.
pub const Parts = struct {
    gpa: std.mem.Allocator,
    h: usize, // pages a part
    pools: [2]kv.Pool,
    slots: []Slot,
    leader: bool,
    counts: Counts = .{},

    pub fn init(gpa: std.mem.Allocator, h: usize, page_bytes: u64, streams: usize, leader: bool) !Parts {
        var a = try kv.Pool.init(gpa, h, page_bytes);
        errdefer a.deinit();
        var b = try kv.Pool.init(gpa, h, page_bytes);
        errdefer b.deinit();
        const slots = try gpa.alloc(Slot, streams);
        @memset(slots, .{});
        return .{ .gpa = gpa, .h = h, .pools = .{ a, b }, .slots = slots, .leader = leader };
    }

    pub fn deinit(p: *Parts) void {
        for (p.slots) |*s| {
            if (p.leader and s.used) p.finish(s);
            s.deinit(p.gpa);
        }
        p.gpa.free(p.slots);
        for (&p.pools) |*x| x.deinit();
        p.* = undefined;
    }

    /// Live streams placed in part `i`.
    pub fn live(p: *const Parts, i: usize) usize {
        var n: usize = 0;
        for (p.slots) |s| n += @intFromBool(s.used and s.part == i);
        return n;
    }

    /// Whether a stream of `n` pages fits a part now.
    pub fn fits(p: *const Parts, n: usize) bool {
        if (n > p.h) return true; // never fits: admitted, so its opening refuses it (PoolTooSmall) instead of waiting
        return p.pools[0].fits(n) or p.pools[1].fits(n);
    }

    /// The part a stream of `n` pages goes to: `home` (a kept state's) when it has the room, else the part with fewer
    /// live streams, then more free pages, then the lower index; null: neither has the room.
    pub fn choose(p: *const Parts, n: usize, home: ?u8) ?u8 {
        if (home) |hp| if (hp < 2 and p.pools[hp].fits(n) and !mut(.home_ignored)) return hp;
        var best: ?u8 = null;
        for (0..2) |i| {
            if (!p.pools[i].fits(n)) continue;
            const b = best orelse {
                best = @intCast(i);
                continue;
            };
            const fewer = p.live(i) < p.live(b);
            const tie = p.live(i) == p.live(b);
            if ((fewer and !mut(.live_ignored)) or ((tie or mut(.live_ignored)) and p.pools[i].available() > p.pools[b].available())) best = @intCast(i);
        }
        return best;
    }

    /// Rank 0: slot `slot` takes `n` pages of part `part` (`ids` the global ids, row order), and with `twin` pages a
    /// twin of that many in the other part when it has them (counted when it has not).
    pub fn open(p: *Parts, slot: usize, part: u8, n: usize, twin: usize) !void {
        const s = &p.slots[slot];
        if (s.used or part > 1) return error.SlotTaken;
        const pool = &p.pools[part];
        try pool.promise(&s.lease, n);
        errdefer pool.finish(&s.lease);
        try pool.cover(&s.lease, n, null);
        s.ids.clearRetainingCapacity();
        try s.ids.ensureTotalCapacity(p.gpa, n);
        for (s.lease.ids.items) |id| s.ids.appendAssumeCapacity(@intCast(id + part * p.h));
        s.part = part;
        s.used = true;
        s.twin.clearRetainingCapacity();
        if (twin == 0) return;
        if (!p.borrow(slot, twin)) p.counts.twins_refused += 1;
    }

    /// Rank 0: a twin of `n` pages in the other part for slot `slot`, all or none; false when that part lacks them.
    pub fn borrow(p: *Parts, slot: usize, n: usize) bool {
        const s = &p.slots[slot];
        if (n == 0 or n > s.ids.items.len or s.twin.items.len > 0) return false;
        const other: u8 = 1 - s.part;
        const pool = &p.pools[other];
        if (!pool.fits(n)) return false;
        pool.promise(&s.twin_lease, n) catch return false;
        pool.cover(&s.twin_lease, n, null) catch {
            pool.finish(&s.twin_lease);
            return false;
        };
        s.twin.ensureTotalCapacity(p.gpa, n) catch {
            pool.finish(&s.twin_lease);
            return false;
        };
        for (s.twin_lease.ids.items) |id| s.twin.appendAssumeCapacity(@intCast(id + other * p.h));
        p.counts.twins_made += 1;
        return true;
    }

    /// Another rank: slot `slot` maps rank 0's ids (`twin` its twin's), checked to lie in one part, the twin's in the
    /// other.
    pub fn map(p: *Parts, slot: usize, ids: []const u32, twin: []const u32) !void {
        const s = &p.slots[slot];
        if (s.used) return error.SlotTaken;
        for (ids) |id| if (id >= 2 * p.h) return error.KvShardPlacement;
        for (twin) |id| if (id >= 2 * p.h) return error.KvShardPlacement;
        const part = try placement(p.h, ids, twin);
        s.ids.clearRetainingCapacity();
        try s.ids.appendSlice(p.gpa, ids);
        errdefer s.ids.clearRetainingCapacity();
        s.twin.clearRetainingCapacity();
        try s.twin.appendSlice(p.gpa, twin);
        s.part = @intCast(part);
        s.used = true;
    }

    /// The prompt pass ended (or the stream did): slot `slot`'s twin goes back; true when it held one.
    pub fn giveBack(p: *Parts, slot: usize) bool {
        const s = &p.slots[slot];
        if (s.twin.items.len == 0) return false;
        s.twin.clearRetainingCapacity();
        if (p.leader) p.pools[1 - s.part].finish(&s.twin_lease);
        p.counts.twins_given += 1;
        return true;
    }

    /// The stream on slot `slot` ended: its twin and its pages go back.
    pub fn close(p: *Parts, slot: usize) void {
        const s = &p.slots[slot];
        if (!s.used) return;
        _ = p.giveBack(slot);
        if (p.leader) p.finish(s);
        s.ids.clearRetainingCapacity();
        s.used = false;
    }

    fn finish(p: *Parts, s: *Slot) void {
        p.pools[1 - s.part].finish(&s.twin_lease);
        p.pools[s.part].finish(&s.lease);
    }

    /// Every live slot's place, hashed (slot, part, ids, twin ids): the same on every rank or a step was missed.
    pub fn digest(p: *const Parts) u64 {
        var h = std.hash.Wyhash.init(0x6b765f7368617264);
        for (p.slots, 0..) |s, i| {
            if (!s.used) continue;
            const head = [_]u64{ i, s.part, s.ids.items.len };
            h.update(std.mem.sliceAsBytes(&head));
            h.update(std.mem.sliceAsBytes(s.ids.items));
            if (mut(.digest_no_twin)) continue;
            h.update(std.mem.asBytes(&@as(u64, s.twin.items.len)));
            h.update(std.mem.sliceAsBytes(s.twin.items));
        }
        return h.final();
    }

    /// The /health and /metrics view: rank 0's parts and counters (`pages_tokens`: positions a page).
    pub fn stats(p: *const Parts, out: *kv_parts.Stats) void {
        out.* = .{ .parts = 2, .part_pages = p.h, .page_tokens = page_rows };
        for (0..2) |i| {
            var live_pages: u64 = 0;
            var twin_pages: u64 = 0;
            for (p.slots) |s| {
                if (!s.used) continue;
                if (s.part == i) live_pages += s.ids.items.len;
                if (s.part != i) twin_pages += s.twin.items.len;
            }
            out.part[i] = .{ .free = p.pools[i].available(), .live_pages = live_pages, .twin_pages = twin_pages, .live = p.live(i) };
        }
        p.counts.into(out);
    }

    // -- the shared store's pages over the parts (TF_KV_SHARE with sharding, core.kv_prompts.Pool.parts) ---------------

    /// The store's one more holder of page `id`: its part's pool, the part-local id.
    pub fn retain(p: *Parts, id: u32) void {
        const part = partOf(id, p.h);
        p.pools[part].retain(id - @as(u32, @intCast(part * p.h)));
    }

    /// The store's hold of page `id` gone; the last holder's release frees the page in its part's pool.
    pub fn release(p: *Parts, id: u32) void {
        const part = partOf(id, p.h);
        p.pools[part].release(id - @as(u32, @intCast(part * p.h)));
    }

    /// How many hold page `id`: each lease that maps it, and the store.
    pub fn holders(p: *const Parts, id: u32) u32 {
        const part = partOf(id, p.h);
        return p.pools[part].holders(id - @as(u32, @intCast(part * p.h)));
    }

    /// Pages free and not promised, both parts together (the store's logging only).
    pub fn available(p: *const Parts) usize {
        return p.pools[0].available() + p.pools[1].available();
    }

    /// Free pages in part `part` now: its part's reclaim evicts its leaves until `want` of these.
    pub fn free(p: *const Parts, part: usize) usize {
        return p.pools[part].available();
    }

    /// Whether a tree leaf whose first page is `first` lies in part `part` (a leaf's pages always do).
    pub fn inPart(p: *const Parts, part: usize, first: u32) bool {
        return partOf(first, p.h) == part;
    }

    /// Whether `ids` all lie in one part (a stream's pages do; the tree takes a leaf's pages as one).
    pub fn onePart(p: *const Parts, ids: []const u32) bool {
        const part = partOf(ids[0], p.h);
        for (ids) |id| if (partOf(id, p.h) != part) return false;
        return true;
    }

    /// Part `part`'s pool takes the store's reclaim hook (or drops it): a promise there evicts the part's idle
    /// leaves before it fails.
    pub fn setReclaim(p: *Parts, part: usize, hook: ?kv.Reclaim) void {
        p.pools[part].reclaim = hook;
    }
};

/// The ownership counters rank 0 reports (each rank counts its own exchanges; rank 0's are reported).
pub const Counts = struct {
    owner_layers: u64 = 0, // DSA layers whose attention one rank ran for both ranks' heads (a segment or more)
    owner_rows: u64 = 0, // rows those layers exchanged queries and outputs for
    owner_bytes: u64 = 0, // bytes this rank sent for them (queries and outputs)
    owner_prompts: u64 = 0, // prompt chunks that ran owner-computed (no twin)
    twins_made: u64 = 0,
    twins_given: u64 = 0,
    twins_refused: u64 = 0, // prompt passes the other part had no room to twin
    carries: u64 = 0, // resumes whose latent rows went to the rank lacking them
    carry_bytes: u64 = 0,
    resumes_dropped: u64 = 0, // resumes that started from 0: the stream could not go home and carries are off
    checks: u64 = 0, // layout digests sent (rank 0) or compared

    pub fn into(c: Counts, out: *kv_parts.Stats) void {
        inline for (@typeInfo(Counts).@"struct".field_names) |n| @field(out, n) = @field(c, n);
    }
};

// -- mutants (test builds only: each turns one guard back off, the test written for it must fail) --------------------

pub const Mutant = enum { none, latent_all_trash, placement_twin_any, home_ignored, live_ignored, digest_no_twin };
pub var mutant: Mutant = .none;

fn mut(comptime m: Mutant) bool {
    return builtin.is_test and mutant == m;
}

test "kv shard: off unless 1; the knobs only when on; a typo is refused" {
    try std.testing.expectEqual(Switches{}, try Switches.parse(null, "0", "x", "y"));
    try std.testing.expectEqual(Switches{}, try Switches.parse("0", null, null, null));
    try std.testing.expectEqual(Switches{}, try Switches.parse(" ", null, null, null));
    try std.testing.expectEqual(Switches{ .on = true }, try Switches.parse("1", null, null, null));
    try std.testing.expectEqual(Switches{ .on = true, .carry = false, .twins = false, .check = 0 }, try Switches.parse("1", "0", "0", "0"));
    try std.testing.expectEqual(@as(u32, 64), (try Switches.parse("1", null, null, "64")).check);
    try std.testing.expectError(error.BadKvShard, Switches.parse("on", null, null, null));
    try std.testing.expectError(error.BadKvShardCarry, Switches.parse("1", "yes", null, null));
    try std.testing.expectError(error.BadKvShardTwins, Switches.parse("1", null, "2", null));
    try std.testing.expectError(error.BadKvShardCheck, Switches.parse("1", null, null, "-1"));
    try std.testing.expectError(error.BadKvShardCheck, Switches.parse("1", null, null, "1e3"));
    try std.testing.expectError(error.BadKvShardCheck, Switches.parse("1", null, null, "99999999999"));
}

test "kv shard: the fingerprint's words are 0 off and name each knob on" {
    try std.testing.expectEqual(@as(u32, 0), (Switches{}).word());
    try std.testing.expectEqual(@as(u32, 0), (Switches{ .check = 7 }).checkWord());
    try std.testing.expectEqual(@as(u32, 7), (Switches{ .on = true }).word());
    try std.testing.expectEqual(@as(u32, 5), (Switches{ .on = true, .carry = false }).word());
    try std.testing.expectEqual(@as(u32, 3), (Switches{ .on = true, .twins = false }).word());
    try std.testing.expectEqual(@as(u32, 1024), (Switches{ .on = true }).checkWord());
    const s: Switches = .{ .on = true, .check = 4 };
    try std.testing.expect(!s.digestAt(0) and !s.digestAt(3) and s.digestAt(4) and s.digestAt(8));
    try std.testing.expect(!(Switches{ .on = true, .check = 0 }).digestAt(1024));
    try std.testing.expect(!(Switches{}).digestAt(1024));
}

test "kv shard: refused at start alike on every rank: TP 2, E4M3 pages, no target MTP head" {
    const on: Switches = .{ .on = true };
    const ok: Facts = .{ .world = 2, .kv8 = true, .pages = true, .mtp_drafts = false, .built = true };
    try refuse(on, ok);
    try refuse(.{}, .{ .world = 3, .kv8 = false, .pages = false, .mtp_drafts = true, .built = false }); // off: nothing
    var f = ok;
    f.world = 1;
    try std.testing.expectError(error.KvShardNeedsTwoRanks, refuse(on, f));
    f = ok;
    f.world = 4;
    try std.testing.expectError(error.KvShardNeedsTwoRanks, refuse(on, f));
    f = ok;
    f.kv8 = false;
    try std.testing.expectError(error.KvShardNeedsFp8, refuse(on, f));
    f = ok;
    f.pages = false;
    try std.testing.expectError(error.KvShardNeedsPages, refuse(on, f));
    f = ok;
    f.mtp_drafts = true;
    try std.testing.expectError(error.KvShardWithMtp, refuse(on, f));
    f = ok;
    f.built = false;
    try std.testing.expectError(error.KvShardNotBuilt, refuse(on, f));
    try refusePool(on, 2, 256, 1 << 19);
    try std.testing.expectError(error.KvShardNeedsParallel, refusePool(on, 1, 4096, 1 << 20));
    try std.testing.expectError(error.KvShardPartTooSmall, refusePool(on, 4, 255, 1 << 20)); // 130,560 tokens
    try std.testing.expectError(error.KvShardPartTooSmall, refusePool(on, 4, 600, 4 << 20)); // under a quarter of 4M
    try refusePool(.{}, 1, 0, 1 << 20);
    inline for (.{ error.BadKvShard, error.KvShardNeedsTwoRanks, error.KvShardWithMtp, error.KvShardPartTooSmall, error.KvShardLayout }) |e| try std.testing.expect(why(e) != null);
    try std.testing.expect(why(error.OutOfMemory) == null);
}

test "kv shard: GLM-5.3-Flash's sharded page is 3,542 bytes a token, 1.8 times the tokens, inside what the plan counts" {
    const g = Geometry.of(.{ .kv_lora = 512, .i_dim = 128 }, 11);
    try std.testing.expectEqual(@as(u64, 264_192), g.lat_layer);
    try std.testing.expectEqual(@as(u64, 32_768), g.key_layer);
    try std.testing.expectEqual(@as(u64, 6380 * 512), g.wholePage()); // cuda_state.pageBytesOf beside DFlash2
    try std.testing.expectEqual(@as(u64, 3542), g.tokenBytes());
    try std.testing.expect(@as(f64, 6380) / 3542 > 1.8);
    for ([_]usize{ 4, 5, 6, 7, 100, 101, 4095, 1 << 20 }) |pages| {
        const h = Geometry.half(pages);
        try std.testing.expect(g.bytes(h) <= pages * g.tokenBytes() * page_rows);
        try std.testing.expect(2 * h + 4 >= pages - 1);
    }
    // an odd geometry rounds the token's bytes up, so the parts still fit
    const odd = Geometry.of(.{ .kv_lora = 512, .i_dim = 128 }, 3);
    for ([_]usize{ 9, 10, 333 }) |pages| try std.testing.expect(odd.bytes(Geometry.half(pages)) <= pages * odd.tokenBytes() * page_rows);
}

test "kv shard: latent tables: own part's pages part-local, a twin's on the other rank, else the trash page" {
    const h = 8;
    var out: [4]u32 = undefined;
    latentIds(0, h, &.{ 3, 1, 7 }, &.{}, &out); // part 0's stream
    try std.testing.expectEqualSlices(u32, &.{ 3, 1, 7 }, out[0..3]);
    latentIds(1, h, &.{ 3, 1, 7 }, &.{}, &out); // the other rank: trash
    try std.testing.expectEqualSlices(u32, &.{ 8, 8, 8 }, out[0..3]);
    latentIds(1, h, &.{ 3, 1, 7 }, &.{ 9, 15 }, &out); // a twin of two pages in part 1: part-local, the rest trash
    try std.testing.expectEqualSlices(u32, &.{ 1, 7, 8 }, out[0..3]);
    latentIds(0, h, &.{ 3, 1, 7 }, &.{ 9, 15 }, &out); // the owner keeps its own
    try std.testing.expectEqualSlices(u32, &.{ 3, 1, 7 }, out[0..3]);
    latentIds(1, h, &.{ 12, 9 }, &.{}, &out); // part 1's stream on its owner
    try std.testing.expectEqualSlices(u32, &.{ 4, 1 }, out[0..2]);
}

test "kv shard: a begin's ids lie in one part and its twin's in the other" {
    try std.testing.expectEqual(@as(usize, 0), try placement(8, &.{ 0, 7, 3 }, &.{}));
    try std.testing.expectEqual(@as(usize, 1), try placement(8, &.{ 8, 15 }, &.{ 2, 0 }));
    try std.testing.expectError(error.KvShardPlacement, placement(8, &.{ 7, 8 }, &.{}));
    try std.testing.expectError(error.KvShardPlacement, placement(8, &.{ 1, 2 }, &.{3}));
    try std.testing.expectError(error.KvShardPlacement, placement(8, &.{9}, &.{ 1, 2 })); // a twin longer than its stream
    try std.testing.expectError(error.KvShardPlacement, placement(8, &.{16}, &.{}));
    try std.testing.expectError(error.KvShardPlacement, placement(8, &.{}, &.{}));
}

test "kv shard: parts place home first, then fewer live streams, then more free pages; a twin is all or none" {
    const gpa = std.testing.allocator;
    var p = try Parts.init(gpa, 8, 4096, 4, true);
    defer p.deinit();
    try std.testing.expectEqual(@as(?u8, 0), p.choose(2, null)); // a tie: the lower index
    try std.testing.expectEqual(@as(?u8, 1), p.choose(2, 1)); // home first
    try p.open(0, 0, 3, 0);
    try std.testing.expectEqualSlices(u32, &.{ 0, 1, 2 }, p.slots[0].ids.items);
    try std.testing.expectEqual(@as(?u8, 1), p.choose(2, null)); // fewer live streams
    try std.testing.expectEqual(@as(?u8, 0), p.choose(2, 0)); // home, though it has more streams
    try p.open(1, 1, 7, 0);
    try std.testing.expectEqual(@as(usize, 8), p.slots[1].ids.items[0] / 8 * 8); // part 1's ids from 8
    try std.testing.expectEqual(@as(?u8, 0), p.choose(2, 1)); // home lacks the room: the other
    try p.open(2, 0, 2, 0); // part 0: 5 of 8 pages, 2 streams; part 1: 7 of 8, 1 stream
    try std.testing.expectEqual(@as(?u8, 1), p.choose(1, null)); // fewer live streams, though fewer free pages
    try std.testing.expectEqual(@as(?u8, null), p.choose(4, null));
    // a twin of 2 pages for slot 1 (part 1) in part 0, all or none: 3 free there
    try std.testing.expect(p.borrow(1, 2));
    try std.testing.expectEqual(@as(usize, 2), p.slots[1].twin.items.len);
    for (p.slots[1].twin.items) |id| try std.testing.expectEqual(@as(usize, 0), partOf(id, 8));
    try std.testing.expect(!p.borrow(0, 2)); // part 1 has 1 free
    try std.testing.expectEqual(@as(usize, 1), p.pools[0].available());
    try std.testing.expect(p.giveBack(1));
    try std.testing.expect(!p.giveBack(1));
    try std.testing.expectEqual(@as(usize, 3), p.pools[0].available());
    p.close(0);
    p.close(1);
    p.close(2);
    try std.testing.expectEqual(@as(usize, 8), p.pools[0].available());
    try std.testing.expectEqual(@as(usize, 8), p.pools[1].available());
    try std.testing.expectEqual(@as(u64, 1), p.counts.twins_made);
    try std.testing.expectEqual(@as(u64, 1), p.counts.twins_given);
}

test "kv shard: a follower maps rank 0's places and digests them alike; a missed step shows" {
    const gpa = std.testing.allocator;
    var lead = try Parts.init(gpa, 16, 4096, 3, true);
    defer lead.deinit();
    var other = try Parts.init(gpa, 16, 4096, 3, false);
    defer other.deinit();
    try lead.open(0, 1, 5, 3);
    try lead.open(2, 0, 4, 0);
    try other.map(0, lead.slots[0].ids.items, lead.slots[0].twin.items);
    try other.map(2, lead.slots[2].ids.items, lead.slots[2].twin.items);
    try std.testing.expectEqual(lead.digest(), other.digest());
    try std.testing.expectEqual(Hold{ .owner = 1, .twin = true }, other.slots[0].hold());
    try std.testing.expect(other.slots[0].hold().has(0) and other.slots[0].hold().has(1));
    try std.testing.expect(other.slots[2].hold().has(0) and !other.slots[2].hold().has(1));
    _ = lead.giveBack(0); // a twin given back on one rank alone
    try std.testing.expect(lead.digest() != other.digest());
    _ = other.giveBack(0);
    try std.testing.expectEqual(lead.digest(), other.digest());
    try std.testing.expectError(error.KvShardPlacement, other.map(1, &.{ 3, 20 }, &.{}));
    try std.testing.expectError(error.KvShardPlacement, other.map(1, &.{40}, &.{}));
    try std.testing.expect(!other.slots[1].used);
    var st: kv_parts.Stats = .{};
    lead.stats(&st);
    try std.testing.expectEqual(@as(u64, 2), st.parts);
    try std.testing.expectEqual(@as(u64, 16), st.part_pages);
    try std.testing.expectEqual(@as(u64, 4), st.part[0].live_pages);
    try std.testing.expectEqual(@as(u64, 5), st.part[1].live_pages);
    try std.testing.expectEqual(@as(u64, 12), st.part[0].free);
    try std.testing.expectEqual(@as(u64, 1), st.twins_made);
}

test "kv shard: each guard's mutant is caught by its test" {
    const gpa = std.testing.allocator;
    defer mutant = .none;
    var out: [3]u32 = undefined;
    mutant = .latent_all_trash;
    latentIds(0, 8, &.{ 3, 1, 7 }, &.{}, &out);
    try std.testing.expect(!std.mem.eql(u32, &.{ 3, 1, 7 }, &out));
    mutant = .placement_twin_any;
    try std.testing.expect(if (placement(8, &.{ 1, 2 }, &.{3})) |_| true else |_| false);
    var p = try Parts.init(gpa, 8, 4096, 2, true);
    defer p.deinit();
    mutant = .home_ignored;
    try std.testing.expect(p.choose(2, 1).? != 1);
    mutant = .none;
    try p.open(0, 0, 2, 2); // part 0: 6 free, 1 stream; part 1: 6 free (its twin), none
    try std.testing.expectEqual(@as(?u8, 1), p.choose(1, null));
    mutant = .live_ignored;
    try std.testing.expectEqual(@as(?u8, 0), p.choose(1, null)); // more free pages alone
    mutant = .none;
    var q = try Parts.init(gpa, 8, 4096, 2, false);
    defer q.deinit();
    try q.map(0, p.slots[0].ids.items, &.{});
    mutant = .digest_no_twin;
    try std.testing.expectEqual(p.digest(), q.digest()); // the twin's absence on q goes unseen
}

test "kv shard: the parts are the shared store's pages: a global id routes to its part's pool, the sum is free" {
    const gpa = std.testing.allocator;
    var p = try Parts.init(gpa, 8, 4096, 2, true);
    defer p.deinit();
    var lease: kv.Lease = .{};
    try p.pools[1].promise(&lease, 3);
    try p.pools[1].cover(&lease, 3, null); // a part-1 stream: part-local ids 0, 1, 2, global 8, 9, 10
    defer p.pools[1].finish(&lease);
    try std.testing.expectEqualSlices(u32, &.{ 0, 1, 2 }, lease.ids.items);
    try std.testing.expectEqual(@as(u32, 1), p.holders(8)); // the lease's alone, by its global id
    p.retain(8); // the store's hold: part 1's pool, its page 0
    try std.testing.expectEqual(@as(u32, 2), p.holders(8));
    try std.testing.expectEqual(@as(usize, 13), p.available()); // the sum of the parts
    try std.testing.expectEqual(@as(usize, 13), p.free(0) + p.free(1));
    p.release(8);
    try std.testing.expectEqual(@as(u32, 1), p.holders(8));
    try std.testing.expect(p.onePart(&.{ 8, 10 }) and !p.onePart(&.{ 8, 3 }));
    try std.testing.expect(p.inPart(1, 9) and !p.inPart(0, 9));
    var probe: u8 = 0;
    p.setReclaim(1, .{ .ctx = &probe, .call = partReclaimProbe });
    try std.testing.expect(p.pools[1].reclaim != null and p.pools[0].reclaim == null);
    p.setReclaim(1, null);
    try std.testing.expect(p.pools[1].reclaim == null);
}

fn partReclaimProbe(_: *anyopaque, _: usize) bool {
    return false;
}
