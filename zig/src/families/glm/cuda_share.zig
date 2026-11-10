//! TF_KV_SHARE: GLM's side of the shared prompt store (native/cuda_prompt_pages.zig) over rank 0's KV pages.
const std = @import("std");
const kp = @import("core").kv_prompts;
const engine = @import("cuda_engine.zig");
const page_rows = @import("cuda_state.zig").page_rows;
const Stream = @import("lanes").Stream;

/// Image passes require the server's content key; text without a key retains its token identity.
pub fn cacheKey(s: *const Stream) ?[]const u32 {
    if (s.key.len != 0) return if (s.key.len == s.prompt_len) s.key else null;
    return if (s.images.len == 0) s.prompt() else null;
}

pub fn cacheable(s: *const Stream) bool {
    return s.prompt_logprobs == null and !s.untaught and s.embedding == null and cacheKey(s) != null;
}

/// Below 4 states a stream, live streams evict each other's: that many, and the --prompt-cache-gib holding their slots.
pub fn advice(states: usize, parallel: usize, slot: u64) ?struct { states: usize, gib: u64 } {
    const want = 4 * parallel;
    if (states >= want) return null;
    return .{ .states = want, .gib = @import("core").kv_pool.holdingGib(want, slot) };
}

/// More kept states than a startup plan sets slots aside for (kv_pool.most_snaps): refused by name at start.
pub fn tooMany(states: usize) bool {
    return states > @import("core").kv_pool.most_snaps;
}

test "TF_KV_SHARE: under 4 planned states a GLM stream, the advice names 4 a stream and the cache their slots take" {
    const mib: u64 = 1 << 20;
    try std.testing.expect(advice(16, 4, 180 * mib) == null);
    const a = advice(13, 4, 180 * mib).?; // a 2.4 GiB cache holds 13 of the 16 at --parallel 4
    try std.testing.expectEqual(@as(usize, 16), a.states);
    try std.testing.expectEqual(@as(u64, 3), a.gib); // 16 x 180 MiB is 2.8 GiB
    try std.testing.expectEqual(@as(usize, 32), advice(0, 8, 180 * mib).?.states); // a cache holding none
}

test "TF_KV_SHARE: up to 64 states is a plan's, 65 is refused" {
    try std.testing.expect(!tooMany(64) and tooMany(65) and !tooMany(0));
}

/// The store the server attached to rank 0's pages (null: none, or another rank).
pub fn of(e: *const engine.Engine) ?kp.Prompts {
    const p = e.pages orelse return null;
    return p.prompts;
}

/// A resume's state: restored by the prompt pass's start on every rank (engine.Resume), once its pages are mapped.
pub const Owner = struct { pending: ?*engine.Kept = null };

/// Before a prompt pass: the deepest kept state's pages in `adopt` (held for the stream), the state in `owner`.
/// Sharded (TF_GLM_KV_SHARD, on rank 0 where the store is, every build mode): the adopted ids must lie in one
/// part — on a violation nothing is adopted, the state is evicted (a cold pass is correct, a cross-part resume
/// is not) and one warning logged.
pub fn begin(e: *engine.Engine, s: *const Stream, owner: *Owner, adopt: *std.ArrayList(u32)) !kp.Plan {
    const prompts = of(e) orelse return error.NoPromptStore;
    const key = cacheKey(s) orelse return error.ImageCacheKeyRequired;
    const plan = try prompts.begin(key, s.history_len, s.shared_prefixes, owner, adopt);
    defer prompts.free(plan.marks);
    if (e.pages) |p| if (p.shard) |x| if (adopt.items.len > 0 and !x.parts.onePart(adopt.items)) {
        const n = adopt.items.len;
        for (adopt.items) |id| x.parts.release(id); // the begin's holds go back
        adopt.clearRetainingCapacity();
        owner.pending = null;
        prompts.forget(key, plan.from); // the mixed state goes: a later pass runs cold
        std.log.warn("kv shard: a kept state at {d} tokens holds {d} pages in both parts (TF_GLM_KV_SHARD); it is evicted, the pass runs cold", .{ plan.from, n });
    };
    const mine = e.gpa.dupe(u32, plan.marks) catch |err| { // the lane's own marks, freed by its allocator
        e.pages.?.give(adopt.items);
        adopt.clearRetainingCapacity();
        return err;
    };
    return .{ .from = plan.from, .marks = mine };
}

/// TF_GLM_KV_SHARE_UNTWINNED_PCT's default: an untwinned prefill's rate as a percentage of the twinned rate.
pub const untwinned_pct_default: u32 = 75;

/// The setting's value (blank or unset: the default), `error.BadKvShareUntwinned` outside 1 to 100.
pub fn untwinnedPct(raw: ?[]const u8) error{BadKvShareUntwinned}!u32 {
    const t = std.mem.trim(u8, raw orelse "", " \t\r\n");
    if (t.len == 0) return untwinned_pct_default;
    const v = std.fmt.parseInt(u32, t, 10) catch return error.BadKvShareUntwinned;
    if (v < 1 or v > 100) return error.BadKvShareUntwinned;
    return v;
}

/// Whether a resume adopts a kept prefix of `adopted` tokens before a prompt of `prompt` tokens: the tail it
/// prefills without a twin pays for the prefix it skips when `new × (100 − pct) ≤ adopted × pct`, the untwinned
/// rate as `pct`% of the twinned (75: a tail of at most three times the adopted prefix; 100: always adopt).
pub fn adoptPays(adopted: u32, prompt: usize, pct: u32) bool {
    const new = prompt -| adopted;
    return @as(u64, new) * (100 - @as(u64, pct)) <= @as(u64, adopted) * @as(u64, pct);
}

/// TF_KV_SHARE sharded: a resume adopts only when its untwinned tail pays (TF_GLM_KV_SHARE_UNTWINNED_PCT). A
/// skipped one runs cold as the pass would without a kept state (its twin borrowed, the kept state left in the
/// store and still resumable), counted in `kv_shard_resumes_skipped_total`, and seated in the kept prefix's part
/// (returned, so the seat takes it as a kept state's home; there the tree's prefix pages and the tail's lie in
/// one part when the keep comes). Rank 0 decides — with nothing adopted the begin step opens the follower's pass
/// cold as it does today. Unsharded (no parts) a resume adopts always.
pub fn gate(e: *engine.Engine, owner: *Owner, adopt: *std.ArrayList(u32), prompt: usize) ?u8 {
    if (adopt.items.len == 0) return null;
    const p = e.pages orelse return null;
    const x = p.shard orelse return null;
    const at = if (owner.pending) |k| k.at else return null;
    const pct = e.settings.share_untwinned_pct;
    if (adoptPays(at, prompt, pct)) return null;
    const home = p.partOf(adopt.items[0]); // the kept prefix's part, remembered before the pages go back
    for (adopt.items) |id| x.parts.release(id); // the begin's holds go back, the tree's keep the state resumable
    adopt.clearRetainingCapacity();
    owner.pending = null;
    x.parts.counts.resumes_skipped += 1;
    std.log.debug("kv shard: a share resume's kept {d} tokens stay in the store, its {d}-token tail untwinned does not pay (TF_GLM_KV_SHARE_UNTWINNED_PCT {d}); the pass runs cold", .{ at, prompt -| at, pct });
    return home;
}

/// A pass's marks (the engine's Marks.ctx): the mark it stands at, or the last it completed when cancelled, keeps.
pub const Keeps = struct {
    e: *engine.Engine,
    slot: u32,
    prompt: []const u32,

    pub fn at(ctx: *anyopaque, mark: u32) void {
        const k: *Keeps = @ptrCast(@alignCast(ctx));
        k.keep(mark);
    }

    /// Slot `slot`'s pass at mark `mark`: the store takes the pages below it it lacks, and the state (no table swaps).
    pub fn keep(k: *const Keeps, mark: u32) void {
        const prompts = of(k.e) orelse return;
        const p = k.e.pages.?;
        if (!p.paged(k.slot)) return;
        const n = mark / page_rows;
        // sharded (TF_GLM_KV_SHARD): the stream's pages are its part slot's global ids; unsharded its lease's
        const ids: []const u32 = if (p.shard) |x| x.parts.slots[k.slot].ids.items else p.leases[k.slot].ids.items;
        if (n > ids.len) return;
        if (p.shard) |x| std.debug.assert(x.parts.onePart(ids[0..n])); // a leaf's pages always lie in one part
        const path = k.e.gpa.alloc(u32, n) catch return;
        defer k.e.gpa.free(path);
        var owner: Owner = .{};
        _ = prompts.keep(k.prompt, mark, &owner, ids[0..n], path);
    }
};

/// The same content key names lookup and every completed or interrupted keep.
pub fn marks(e: *engine.Engine, slot: u32, s: *const Stream, at: []const u32, keeps: *Keeps) ?engine.Marks {
    keeps.* = .{ .e = e, .slot = slot, .prompt = cacheKey(s) orelse return null };
    return if (at.len == 0) null else .{ .at = at, .ctx = keeps, .call = Keeps.at, .interrupted = Keeps.at };
}

test "TF_GLM_KV_SHARE_UNTWINNED_PCT: 75 the default, 1 to 100 taken, anything else refused" {
    try std.testing.expectEqual(@as(u32, 75), try untwinnedPct(null));
    try std.testing.expectEqual(@as(u32, 75), try untwinnedPct(""));
    try std.testing.expectEqual(@as(u32, 75), try untwinnedPct("  "));
    try std.testing.expectEqual(@as(u32, 1), try untwinnedPct("1"));
    try std.testing.expectEqual(@as(u32, 100), try untwinnedPct("100"));
    try std.testing.expectError(error.BadKvShareUntwinned, untwinnedPct("0"));
    try std.testing.expectError(error.BadKvShareUntwinned, untwinnedPct("101"));
    try std.testing.expectError(error.BadKvShareUntwinned, untwinnedPct("-1"));
    try std.testing.expectError(error.BadKvShareUntwinned, untwinnedPct("half"));
}

test "TF_GLM_KV_SHARE_UNTWINNED_PCT 75: a resume adopts when the tail is at most three times the kept prefix" {
    try std.testing.expect(!adoptPays(4096, 126407, 75)); // 122,311 new tokens over 3 x 4,096 adopted: cold
    try std.testing.expect(adoptPays(122880, 126407, 75)); // 3,527 new tokens: adopted
    try std.testing.expect(!adoptPays(122880, 610128, 75)); // 487,248 new tokens over 3 x 122,880: cold
    try std.testing.expect(adoptPays(126407, 126407, 75)); // the whole prompt kept: an empty tail pays
    try std.testing.expect(!adoptPays(0, 4096, 75)); // nothing kept: cold
}

test "TF_GLM_KV_SHARE_UNTWINNED_PCT 100: a resume adopts however long its tail; at 1 almost never" {
    try std.testing.expect(adoptPays(4096, 126407, 100));
    try std.testing.expect(adoptPays(0, 4096, 100));
    try std.testing.expect(adoptPays(126407, 126407, 1)); // the tail empty: adopted
    try std.testing.expect(!adoptPays(4096, 126407, 1)); // a tail a hundredfold the prefix: cold
}
