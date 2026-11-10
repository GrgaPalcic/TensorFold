//! TF_KV_SHARE=n: a CUDA family on the core pool keeps prompts as its pool's pages in one prompt_radix store.
const std = @import("std");
const api = @import("engine_api");
const core = @import("core");
const snaps = @import("cuda_snaps.zig");
const radix = api.prompt_radix;
const kv = core.kv_pool;
const hooks = core.kv_prompts;
const Allocator = std.mem.Allocator;

/// Whether the family keeps prompts this way: it declares the hooks, its streams are on pages and TF_KV_SHARE is set.
pub fn wanted(comptime F: type, loaded: anytype) !bool {
    return could(F, loaded) and try hooks.sharing() > 0;
}

/// Whether TF_KV_SHARE would keep the family's prompts this way: it declares the hooks and its streams are on pages.
pub fn could(comptime F: type, loaded: anytype) bool {
    if (comptime !@hasDecl(F, "share_attach")) return false;
    return loaded.layout != null;
}

/// What the unpaged store's note on a state past its whole budget adds: the setting, where `could` holds.
pub fn hint(comptime F: type, loaded: anytype) []const u8 {
    return if (could(F, loaded)) "; TF_KV_SHARE=n keeps a prompt's rows as the pool's pages instead of copying them" else "";
}

/// The store over the family's pool: its pages in the tree, the rest of a state from the family's `snap_*` decls.
pub const Adapter = struct {
    gpa: Allocator,
    src: hooks.Pool, // the family's pages: one pool, or a sharded pool's parts (each part its own reclaim below)
    store: radix.Store,
    part: [2]Part = undefined, // each part's reclaim hook (the parts variant)
    detach: *const fn (*anyopaque) void,
    target: *anyopaque,

    /// The family's store, its pool's reclaim evicting the tree's idle pages — one pool's all of them, a sharded
    /// pool's each part its own leaves (below); null: the family has no pool.
    pub fn open(comptime F: type, gpa: Allocator, target: anytype, page_tokens: u32, snap_limit: usize) !?*Adapter {
        const src = F.share_pool(target) orelse return null;
        const x = try gpa.create(Adapter);
        errdefer gpa.destroy(x);
        var rules = snaps.cacheRules(F);
        rules.page = page_tokens;
        if (comptime @hasDecl(F, "share_grid")) {
            rules.planned = true;
            rules.grid = F.share_grid;
        }
        rules.min_gap = 256;
        rules.tail = true;
        const copies: radix.Snapshots = .{ .ptr = target, .vtable = &snaps.Snaps(F, null).vtable };
        x.* = .{ .gpa = gpa, .src = src, .store = undefined, .detach = Detach(F).call, .target = target };
        x.store = try radix.Store.init(gpa, copies, x.pages(), rules);
        switch (src) {
            .one => |pool| pool.reclaim = .{ .ctx = x, .call = reclaim }, // a promise or a hold evicts the tree's idle pages before it fails
            .parts => |s| for (0..s.count) |i| { // a part's promise evicts its own leaves first (its hook below)
                x.part[i] = .{ .x = x, .at = i };
                s.setReclaim(s.ctx, i, .{ .ctx = &x.part[i], .call = reclaimPart });
            },
        }
        x.store.limit(.{ .pages = switch (src) {
            .one => |pool| pool.total(),
            .parts => |s| s.total,
        }, .snaps = snap_limit });
        F.share_attach(target, .{ .ctx = x, .vtable = &vtable });
        return x;
    }

    /// The tree's page ids over this source: one pool's, or a sharded pool's parts' (its calls on its place here).
    fn pages(x: *Adapter) radix.Pages {
        return switch (x.src) {
            .one => |pool| radix.pagesOf(kv.Pool, pool, pool.page_bytes),
            .parts => |*s| .{ .ptr = s, .bytes = s.page_bytes, .vtable = &partsPages },
        };
    }

    /// The parts' pages as the tree takes them: each call through the family's own (hooks.Parts', ctx its parts).
    const partsPages: radix.Pages.VTable = .{
        .retain = PartsPages.retain,
        .release = PartsPages.release,
        .holders = PartsPages.holders,
        .available = PartsPages.available,
    };

    const PartsPages = struct {
        fn parts(ptr: *anyopaque) *const hooks.Parts {
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
    };

    fn Detach(comptime F: type) type {
        return struct {
            fn call(target: *anyopaque) void {
                F.share_attach(target, null);
            }
        };
    }

    /// The family stops calling, the pool's reclaim goes, the tree drops its states and releases its pages.
    pub fn close(x: *Adapter) void {
        x.detach(x.target);
        switch (x.src) {
            .one => |pool| pool.reclaim = null,
            .parts => |s| for (0..s.count) |i| s.setReclaim(s.ctx, i, null),
        }
        x.store.deinit();
        x.gpa.destroy(x);
    }

    fn of(ctx: *anyopaque) *Adapter {
        return @ptrCast(@alignCast(ctx));
    }

    fn reclaim(ctx: *anyopaque, want: usize) bool {
        return of(ctx).store.reclaim(want, null);
    }

    /// A sharded pool's part as its pool's reclaim hook: evict the tree's idle leaves of this part until `want`
    /// pages are free in it — never the other part's, whose rank's streams do not wait on this one.
    const Part = struct { x: *Adapter, at: usize };

    fn reclaimPart(ctx: *anyopaque, want: usize) bool {
        const h: *Part = @ptrCast(@alignCast(ctx));
        return h.x.store.reclaim(want, .{ .ctx = h, .f = leafInPart, .free = partFree });
    }

    fn leafInPart(ctx: *anyopaque, first: u32) bool {
        const h: *Part = @ptrCast(@alignCast(ctx));
        const s = h.x.src.parts;
        return s.inPart(s.ctx, h.at, first);
    }

    fn partFree(ctx: *anyopaque) usize {
        const h: *Part = @ptrCast(@alignCast(ctx));
        const s = h.x.src.parts;
        return s.free(s.ctx, h.at);
    }

    fn begin(ctx: *anyopaque, prompt: []const u32, history_len: u32, shared: []const u32, owner: *anyopaque, adopt: *std.ArrayList(u32)) anyerror!hooks.Plan {
        const x = of(ctx);
        const p = try x.store.begin(x.gpa, prompt, history_len, shared, owner, adopt);
        return .{ .from = p.from, .marks = p.marks };
    }

    fn keep(ctx: *anyopaque, prompt: []const u32, at: u32, owner: *anyopaque, mine: []const u32, path: []u32) usize {
        return of(ctx).store.keep(prompt, at, owner, mine, path).shared;
    }

    fn report(ctx: *anyopaque, prompt: usize, from: u32, kept: u64) void {
        of(ctx).store.report(prompt, from, kept);
    }

    fn free(ctx: *anyopaque, marks: []const u32) void {
        of(ctx).gpa.free(marks);
    }

    const vtable: hooks.Prompts.VTable = .{ .begin = begin, .keep = keep, .report = report, .free = free };
};

const testing = std.testing;

/// A family over a real pool whose states are a byte each, its pages its streams' (a host test's stand-in).
const Fake = struct {
    pool: kv.Pool,
    prompts: ?hooks.Prompts = null,
    saved: u8 = 0,
    restored: ?*anyopaque = null,

    fn of(p: *anyopaque) *Fake {
        return @ptrCast(@alignCast(p));
    }
    pub fn share_pool(p: *anyopaque) ?hooks.Pool {
        return .{ .one = &of(p).pool };
    }
    pub fn share_attach(p: *anyopaque, prompts: ?hooks.Prompts) void {
        of(p).prompts = prompts;
    }
    pub fn snap_bytes(_: *anyopaque, _: u32) u64 {
        return 1;
    }
    pub fn snap_save(p: *anyopaque, _: ?*anyopaque, _: u32) anyerror!*anyopaque {
        return @ptrCast(&of(p).saved);
    }
    pub fn snap_restore(p: *anyopaque, _: ?*anyopaque, saved: *anyopaque) anyerror!void {
        of(p).restored = saved;
    }
    pub fn snap_drop(_: *anyopaque, _: *anyopaque) void {}
    pub const snap_lookahead: u32 = 0;
};

test "TF_KV_SHARE's adapter: a kept prompt's pages resume a later one, adopted; closing gives every page back" {
    var f: Fake = .{ .pool = try kv.Pool.init(testing.allocator, 8, 1) };
    defer f.pool.deinit();
    const x = (try Adapter.open(Fake, testing.allocator, &f, 4, 4)).?;
    const p = f.prompts.?;
    const prompt = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
    var lease: kv.Lease = .{};
    try f.pool.promise(&lease, 3);
    try f.pool.cover(&lease, 3, null);
    var path: [2]u32 = undefined;
    var owner: u8 = 0;
    try testing.expectEqual(@as(usize, 2), p.keep(&prompt, 8, &owner, lease.ids.items[0..2], &path));
    try testing.expectEqualSlices(u32, lease.ids.items[0..2], &path);
    f.pool.finish(&lease);
    var adopt: std.ArrayList(u32) = .empty;
    defer adopt.deinit(testing.allocator);
    const plan = try p.begin(&prompt, 10, &.{}, &owner, &adopt);
    defer p.free(plan.marks);
    try testing.expectEqual(@as(u32, 8), plan.from);
    try testing.expectEqualSlices(u32, &path, adopt.items);
    try testing.expect(f.restored != null);
    for (adopt.items) |id| f.pool.release(id);
    x.close();
    try testing.expect(f.prompts == null and f.pool.idle() and f.pool.available() == 8);
}

test "TF_KV_SHARE: a state past the unpaged cache's whole budget names it where the family on pages could keep it" {
    const l: kv.Layout = .{ .fixed_bytes = 0, .token_bytes = 1, .page_tokens = 4, .snap_bytes = 0 };
    const paged: struct { layout: ?kv.Layout } = .{ .layout = l };
    const windows: struct { layout: ?kv.Layout } = .{ .layout = null };
    try testing.expect(std.mem.indexOf(u8, hint(Fake, paged), "TF_KV_SHARE=n") != null);
    try testing.expectEqualStrings("", hint(Fake, windows)); // a window a stream: the store cannot share
    try testing.expectEqualStrings("", hint(struct {}, paged)); // a family without the store's hooks
}

/// A sharded pool's stand-in (TF_GLM_KV_SHARD): two pools of `h` pages a part, `share_pool` handing them over as
/// hooks.Pool.parts, its pages view routing a global id to its part's pool with the part-local id.
const Sharded = struct {
    h: usize = 4,
    pools: [2]kv.Pool,
    prompts: ?hooks.Prompts = null,
    saved: u8 = 0,
    restored: ?*anyopaque = null,

    fn of(p: *anyopaque) *Sharded {
        return @ptrCast(@alignCast(p));
    }
    pub fn share_pool(p: *anyopaque) ?hooks.Pool {
        return .{ .parts = .{
            .retain = shRetain,
            .release = shRelease,
            .holders = shHolders,
            .available = shAvailable,
            .page_bytes = 1,
            .total = 2 * of(p).h,
            .count = 2,
            .free = partFree,
            .inPart = partIn,
            .onePart = partOne,
            .setReclaim = partSet,
            .ctx = p,
        } };
    }
    pub fn share_attach(p: *anyopaque, prompts: ?hooks.Prompts) void {
        of(p).prompts = prompts;
    }
    pub fn snap_bytes(_: *anyopaque, _: u32) u64 {
        return 1;
    }
    pub fn snap_save(p: *anyopaque, _: ?*anyopaque, _: u32) anyerror!*anyopaque {
        return @ptrCast(&of(p).saved);
    }
    pub fn snap_restore(p: *anyopaque, _: ?*anyopaque, saved: *anyopaque) anyerror!void {
        of(p).restored = saved;
    }
    pub fn snap_drop(_: *anyopaque, _: *anyopaque) void {}
    pub const snap_lookahead: u32 = 0;

    fn pool(p: *Sharded, id: u32) *kv.Pool {
        return &p.pools[id / p.h];
    }
    fn local(p: *const Sharded, id: u32) u32 {
        return id - @as(u32, @intCast(id / p.h * p.h));
    }
    pub fn retain(p: *Sharded, id: u32) void {
        p.pool(id).retain(p.local(id));
    }
    pub fn release(p: *Sharded, id: u32) void {
        p.pool(id).release(p.local(id));
    }
    pub fn holders(p: *const Sharded, id: u32) u32 {
        return p.pools[id / p.h].holders(p.local(id));
    }
    pub fn available(p: *const Sharded) usize {
        return p.pools[0].available() + p.pools[1].available();
    }

    fn shRetain(ctx: *anyopaque, id: u32) void {
        of(ctx).retain(id);
    }
    fn shRelease(ctx: *anyopaque, id: u32) void {
        of(ctx).release(id);
    }
    fn shHolders(ctx: *anyopaque, id: u32) u32 {
        return of(ctx).holders(id);
    }
    fn shAvailable(ctx: *anyopaque) usize {
        return of(ctx).available();
    }

    fn partFree(ctx: *anyopaque, part: usize) usize {
        return of(ctx).pools[part].available();
    }
    fn partIn(ctx: *anyopaque, part: usize, first: u32) bool {
        return first / of(ctx).h == part;
    }
    fn partOne(ctx: *anyopaque, ids: []const u32) bool {
        const h = of(ctx).h;
        const one = ids[0] / h;
        for (ids) |id| if (id / h != one) return false;
        return true;
    }
    fn partSet(ctx: *anyopaque, part: usize, hook: ?kv.Reclaim) void {
        of(ctx).pools[part].reclaim = hook;
    }
};

test "TF_KV_SHARE's adapter over a sharded pool: the parts are the store's pages, a keep and its resume cross them" {
    var f: Sharded = .{ .pools = .{ try kv.Pool.init(testing.allocator, 4, 1), try kv.Pool.init(testing.allocator, 4, 1) } };
    defer for (&f.pools) |*p| p.deinit();
    const x = (try Adapter.open(Sharded, testing.allocator, &f, 4, 4)).?;
    const p = f.prompts.?;
    const prompt = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
    var lease: kv.Lease = .{};
    try f.pools[1].promise(&lease, 3);
    try f.pools[1].cover(&lease, 3, null); // a stream in part 1: part-local 0, 1, 2, global 4, 5, 6
    const mine = [_]u32{ lease.ids.items[0] + 4, lease.ids.items[1] + 4 }; // the stream's pages, as the store sees them
    var path: [2]u32 = undefined;
    var owner: u8 = 0;
    try testing.expectEqual(@as(usize, 2), p.keep(&prompt, 8, &owner, &mine, &path));
    try testing.expectEqualSlices(u32, &mine, &path);
    f.pools[1].finish(&lease); // the stream's pages past the mark free, those below it the tree's
    for (path) |id| try testing.expectEqual(@as(u32, 1), f.holders(id)); // the tree's alone, in part 1
    var adopt: std.ArrayList(u32) = .empty;
    defer adopt.deinit(testing.allocator);
    const plan = try p.begin(&prompt, 10, &.{}, &owner, &adopt);
    defer p.free(plan.marks);
    try testing.expectEqual(@as(u32, 8), plan.from);
    try testing.expectEqualSlices(u32, &path, adopt.items); // the resume adopts them where they lie
    for (adopt.items) |id| f.release(id);
    x.close(); // the tree's holds go, each part's pool idle again
    try testing.expect(f.prompts == null and f.pools[0].idle() and f.pools[1].idle());
}
