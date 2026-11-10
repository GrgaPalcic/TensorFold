//! The paged prompt store's disk tier (TF_KV_SHARE with --learn-conversations): a kept state's rows are the tree's
//! pages, written behind and recalled into pool pages, over a fake pool and a tiny image, as rows_store's tests do.

const std = @import("std");
const pc = @import("prompt_cache.zig");
const radix = @import("prompt_radix.zig");
const rows = @import("rows_store.zig");
const Allocator = std.mem.Allocator;
const t = std.testing;

const page_tokens = 4; // tokens a page holds (rules.page)
const page_bytes = 8; // bytes of row data a page holds: its 4 tokens, 2 bytes a row
const page_count = 16; // pages the pool has
const words_bytes = 16; // the image's words: the position and the running sum
const fixed_bytes = 8; // the image's fixed part past the rows: the slot's own

/// A page pool as kv_pool.Pool's holds: `hold` takes free pages, `retain` one more holder.
const Pool = struct {
    refs: [page_count]u32 = std.mem.zeroes([page_count]u32),
    mem: []u8 = &.{}, // the pages' bytes, the rig's

    pub fn hold(p: *Pool, out: []u32) bool {
        var n: usize = 0;
        for (&p.refs, 0..) |*h, id| {
            if (h.* != 0) continue;
            out[n] = @intCast(id);
            h.* = 1;
            n += 1;
            if (n == out.len) return true;
        }
        for (out[0..n]) |id| p.refs[id] = 0; // nothing held: what it took goes back
        return false;
    }

    pub fn retain(p: *Pool, id: u32) void {
        p.refs[id] += 1;
    }

    pub fn release(p: *Pool, id: u32) void {
        p.refs[id] -= 1;
    }

    pub fn holders(p: *const Pool, id: u32) u32 {
        return p.refs[id];
    }

    pub fn available(p: *const Pool) usize {
        var n: usize = 0;
        for (&p.refs) |h| if (h == 0) {
            n += 1;
        };
        return n;
    }
};

/// The family: the pool's pages' bytes in host memory, a state the words and fixed part past its pages' rows.
const Fake = struct {
    gpa: Allocator,
    prompt: []const u32 = &.{},
    pool: *Pool,
    mem: []u8, // the pages' bytes: a page's rows at [0, page_bytes), as the paged caches sit on a page
    live: usize = 0,
    blanks: usize = 0,

    fn of(ptr: *anyopaque) *Fake {
        return @ptrCast(@alignCast(ptr));
    }

    /// A token row's 2 bytes, as a pass writes it into its page.
    fn row(f: *const Fake, p: u32) [2]u8 {
        const x = std.hash.Wyhash.hash(4, std.mem.sliceAsBytes(&[_]u32{f.prompt[p]}));
        return .{ @truncate(x), @truncate(x >> 8) };
    }

    /// The stream's rows into the pages it took (a pass's writes), tokens [0, at).
    fn fill(f: *Fake, pages: []const u32, at: u32) void {
        for (0..at) |p| {
            const r = f.row(@intCast(p));
            const base = @as(usize, pages[p / page_tokens]) * page_bytes + (p % page_tokens) * 2;
            @memcpy(f.mem[base..][0..2], &r);
        }
    }

    /// The words and fixed part a fresh pass leaves: a state past its pages' rows.
    fn build(f: *const Fake, at: u32, out: []u8) void {
        std.mem.writeInt(u64, out[0..8], at, .little);
        var sum: u64 = 0;
        for (f.prompt[0..at]) |tok| sum = sum *% 31 +% tok;
        std.mem.writeInt(u64, out[8..16], sum, .little);
        @memcpy(out[16..24], "slotpart");
    }

    fn bytes(_: *anyopaque, _: u32) u64 {
        return words_bytes + fixed_bytes; // the slot's bytes past the pages' rows
    }

    fn save(ptr: *anyopaque, _: ?*anyopaque, at: u32) anyerror!pc.Saved {
        const f = of(ptr);
        if (at == 0 or at % page_tokens != 0 or at > f.prompt.len) return error.NotAtMark;
        const k = try f.gpa.create(Kept);
        errdefer f.gpa.destroy(k);
        k.* = .{ .at = at, .slot = try f.gpa.alloc(u8, words_bytes + fixed_bytes) };
        f.build(at, k.slot);
        f.live += 1;
        return k;
    }

    fn restore(ptr: *anyopaque, _: ?*anyopaque, saved: pc.Saved) anyerror!void {
        const f = of(ptr);
        const k: *Kept = @ptrCast(@alignCast(saved));
        const want = try f.gpa.alloc(u8, k.slot.len);
        defer f.gpa.free(want);
        f.build(k.at, want);
        if (!std.mem.eql(u8, want, k.slot)) return error.RestoreDiffered;
    }

    fn drop(ptr: *anyopaque, saved: pc.Saved) void {
        const f = of(ptr);
        const k: *Kept = @ptrCast(@alignCast(saved));
        f.gpa.free(k.slot);
        f.gpa.destroy(k);
        f.live -= 1;
    }

    fn family(f: *Fake) pc.Snapshots {
        return .{ .ptr = f, .vtable = &.{ .bytes = bytes, .save = save, .restore = restore, .drop = drop, .rows = &image } };
    }

    const Kept = struct { at: u32, slot: []u8 };

    /// The store's tiny image: the words, a state's rows a row a token (its pages' rows), the fixed part.
    const image: rows.Image = .{ .parts = partsFn, .gather = gatherFn, .blank = blankFn, .scatter = scatterFn, .blank_pages = blankPagesFn };

    fn partsFn(_: *anyopaque, at: u32, out: *std.ArrayList(rows.image.Part), a: Allocator) anyerror!void {
        try out.appendSlice(a, &.{
            .{ .bytes = words_bytes },
            .{ .bytes = @as(u64, at) * 2, .row = 2 },
            .{ .bytes = fixed_bytes },
        });
    }

    /// Which bytes of the image [off, off + len) are: the words, a run of a page's rows, or the fixed part.
    const Seg = union(enum) { words: usize, page: struct { id: u32, at: usize }, fixed: usize };

    fn seg(_: *Fake, k: *Kept, pages: []const u32, off: u64, _: usize) !Seg {
        const rows_end = words_bytes + @as(u64, k.at) * 2;
        if (off < words_bytes) return .{ .words = @intCast(off) };
        if (off < rows_end) {
            const b = off - words_bytes;
            const k_i = b / page_bytes;
            if (k_i >= pages.len) return error.BadKeptRange;
            return .{ .page = .{ .id = pages[@intCast(k_i)], .at = @intCast(b % page_bytes) } };
        }
        if (off < rows_end + fixed_bytes) return .{ .fixed = @intCast(off - rows_end) };
        return error.BadKeptRange;
    }

    fn gatherFn(ptr: *anyopaque, saved: pc.Saved, pages: []const u32, ranges: []const rows.image.Range, out: []u8) anyerror!void {
        const f = of(ptr);
        const k: *Kept = @ptrCast(@alignCast(saved));
        var o: usize = 0;
        for (ranges) |g| {
            var done: u64 = 0;
            while (done < g.len) {
                const s = try seg(f, k, pages, g.off + done, @intCast(g.len - done));
                const n: usize = switch (s) {
                    .words => |at| @min(@as(usize, @intCast(g.len - done)), words_bytes - at),
                    .page => |p| @min(@as(usize, @intCast(g.len - done)), page_bytes - p.at),
                    .fixed => |at| @min(@as(usize, @intCast(g.len - done)), fixed_bytes - at),
                };
                const host = out[o..][0..n];
                switch (s) {
                    .words => |at| @memcpy(host, k.slot[at..][0..n]),
                    .page => |p| @memcpy(host, f.mem[@as(usize, p.id) * page_bytes + p.at ..][0..n]),
                    .fixed => |at| @memcpy(host, k.slot[words_bytes + at ..][0..n]),
                }
                done += n;
                o += n;
            }
        }
    }

    fn blankFn(_: *anyopaque, _: u32) anyerror!pc.Saved {
        return error.NoSlot; // the paged store recalls into pages, never into a copy alone
    }

    fn blankPagesFn(ptr: *anyopaque, at: u32, out: *std.ArrayList(u32)) anyerror!pc.Saved {
        const f = of(ptr);
        if (at % page_tokens != 0) return error.BadKeptRange;
        const k = try f.gpa.create(Kept);
        errdefer f.gpa.destroy(k);
        const slot = try f.gpa.alloc(u8, words_bytes + fixed_bytes);
        errdefer f.gpa.free(slot);
        @memset(slot, 0);
        out.resize(f.gpa, at / page_tokens) catch return error.LaneMemory;
        if (!f.pool.hold(out.items)) return error.LaneMemory; // the pages held for the caller's tree
        k.* = .{ .at = at, .slot = slot };
        f.live += 1;
        f.blanks += 1;
        return k;
    }

    fn scatterFn(ptr: *anyopaque, saved: pc.Saved, pages: []const u32, ranges: []const rows.image.Range, in: []const u8) anyerror!void {
        const f = of(ptr);
        const k: *Kept = @ptrCast(@alignCast(saved));
        var o: usize = 0;
        for (ranges) |g| {
            var done: u64 = 0;
            while (done < g.len) {
                const s = try seg(f, k, pages, g.off + done, @intCast(g.len - done));
                const n: usize = switch (s) {
                    .words => |at| @min(@as(usize, @intCast(g.len - done)), words_bytes - at),
                    .page => |p| @min(@as(usize, @intCast(g.len - done)), page_bytes - p.at),
                    .fixed => |at| @min(@as(usize, @intCast(g.len - done)), fixed_bytes - at),
                };
                const part = in[o..][0..n];
                switch (s) {
                    .words => |at| @memcpy(k.slot[at..][0..n], part),
                    .page => |p| @memcpy(f.mem[@as(usize, p.id) * page_bytes + p.at ..][0..n], part),
                    .fixed => |at| @memcpy(k.slot[words_bytes + at ..][0..n], part),
                }
                done += n;
                o += n;
            }
        }
    }
};

/// The store over a fake pool: `mem` the pages' bytes, the disk tier attached.
const Rig = struct {
    gpa: Allocator,
    f: Fake,
    pool: Pool,
    s: radix.Store,
    r: *rows.Rows,

    fn open(gpa: Allocator, root: []const u8, identity: u64) !*Rig {
        const x = try gpa.create(Rig);
        errdefer gpa.destroy(x);
        const mem = try gpa.alloc(u8, page_count * page_bytes);
        errdefer gpa.free(mem);
        @memset(mem, 0);
        x.* = .{ .gpa = gpa, .f = undefined, .pool = .{ .mem = mem }, .s = undefined, .r = undefined };
        x.f = .{ .gpa = gpa, .pool = &x.pool, .mem = mem };
        const family = x.f.family();
        x.s = try radix.Store.init(gpa, family, radix.pagesOf(Pool, &x.pool, page_bytes), .{ .page = page_tokens, .min_gap = 8, .min_prompt = 0 });
        errdefer x.s.deinit();
        x.s.limit(.{ .pages = page_count, .snaps = 8 });
        x.r = try rows.Rows.open(gpa, family, .{
            .root = root,
            .identity = identity,
            .cap = 1 << 30,
            .floor = 0,
            .daily = 1 << 30,
            .sync_writes = true,
        });
        errdefer x.r.deinit();
        x.r.prefill_rate = 1.0; // a token a second: reading from disk pays over prefilling
        x.s.attachDisk(x.r);
        return x;
    }

    fn close(x: *Rig) void {
        x.s.deinit(); // the writer's queued saves go first (their pages' holds back)
        x.r.deinit();
        x.gpa.free(x.f.mem);
        x.gpa.destroy(x);
    }

    /// A pass of `prompt`: the pages a stream takes and fills with its rows, a keep at each mark, the stream's end
    /// (its holds go; the tree's remain on the kept pages).
    fn pass(x: *Rig, prompt: []const u32, marks: []const u32) !void {
        x.f.prompt = prompt;
        const np = prompt.len / page_tokens;
        var mine: [page_count]u32 = undefined;
        try t.expect(x.pool.hold(mine[0..np]));
        x.f.fill(mine[0..np], @intCast(prompt.len));
        for (marks) |m| {
            var path: [page_count]u32 = undefined;
            const kept = x.s.keep(prompt, m, null, mine[0 .. m / page_tokens], path[0 .. m / page_tokens]);
            try t.expect(kept.held);
        }
        for (mine[0..np]) |id| x.pool.release(id);
    }

    /// The writer runs (sync writes): the queued saves land.
    fn drain(x: *Rig) void {
        while (rows.write.step(x.r)) {}
    }

    /// Whether the pages' rows of a state at `at` are a fresh pass's.
    fn rowsOk(x: *Rig, pages: []const u32, at: u32) !void {
        for (0..at) |p| {
            const r = x.f.row(@intCast(p));
            const base = @as(usize, pages[p / page_tokens]) * page_bytes + (p % page_tokens) * 2;
            try t.expectEqualSlices(u8, &r, x.f.mem[base..][0..2]);
        }
    }
};

test "TF_KV_SHARE's disk tier: a 2-page state written behind, its pages held while the save reads them" {
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    const root = try tmp.dir.realPathFileAlloc(t.io, ".", t.allocator);
    defer t.allocator.free(root);
    const x = try Rig.open(t.allocator, root, 9);
    defer x.close();
    const prompt = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
    try x.pass(&prompt, &.{8});
    x.drain();
    try t.expectEqual(@as(usize, 1), x.s.tree.snaps);
    try t.expectEqual(@as(usize, 2), x.s.tree.held_pages);
    // the save ran: the state on disk, its pages the tree's alone again
    try t.expectEqual(@as(usize, 1), x.r.states.items.len);
    try t.expectEqual(@as(u64, 1), x.r.counts.writes);
    try t.expectEqual(@as(usize, 0), x.s.saves.items.len);
    for (x.s.find(&prompt).?.pages) |id| try t.expectEqual(@as(u32, 1), x.pool.holders(id));
}

test "TF_KV_SHARE's disk tier: an eviction while the save is queued stages it, its pages held to the end" {
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    const root = try tmp.dir.realPathFileAlloc(t.io, ".", t.allocator);
    defer t.allocator.free(root);
    const x = try Rig.open(t.allocator, root, 10);
    defer x.close();
    const prompt = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
    try x.pass(&prompt, &.{8});
    // The save is queued, its pages held once more: the tree may drop the leaf under the writer and the pool
    // cannot reuse the pages under it.
    const node = x.s.find(&prompt).?;
    try t.expectEqual(@as(u32, 2), x.pool.holders(node.pages[0])); // the tree's and the writer's
    var full: [page_count - 2]u32 = undefined;
    try t.expect(x.pool.hold(&full)); // the rest of the pool: no page free but the save's
    x.s.tree.evict(node); // the store's evicting hook stages the queued save now
    try t.expectEqual(@as(usize, 0), x.s.tree.snaps);
    try t.expectEqual(@as(usize, 1), x.r.states.items.len); // the state landed before its pages went
    try t.expectEqual(@as(u64, 1), x.r.counts.writes);
    for (&full) |id| x.pool.release(id);
    try t.expectEqual(@as(usize, page_count), x.pool.available()); // the writer's holds went with the write
}

test "TF_KV_SHARE's disk tier: a miss recalls the state into pool pages, its pages adopted like any kept state's" {
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    const root = try tmp.dir.realPathFileAlloc(t.io, ".", t.allocator);
    defer t.allocator.free(root);
    {
        const x = try Rig.open(t.allocator, root, 11);
        defer x.close();
        const prompt = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
        try x.pass(&prompt, &.{8});
        x.drain();
        try t.expectEqual(@as(usize, 1), x.r.states.items.len);
    }
    // A restart: a new store over the same pool reads the state back into pool pages it holds.
    const x = try Rig.open(t.allocator, root, 11);
    defer x.close();
    const longer = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 };
    x.f.prompt = &longer;
    var adopt: std.ArrayList(u32) = .empty;
    defer {
        for (adopt.items) |id| x.pool.release(id); // the caller's holds; the tree's go with the store
        adopt.deinit(t.allocator);
    }
    const plan = try x.s.begin(t.allocator, &longer, 12, &.{}, null, &adopt);
    defer t.allocator.free(plan.marks);
    try t.expectEqual(@as(u32, 8), plan.from); // the recalled state is the resume
    try t.expectEqual(@as(usize, 2), adopt.items.len); // its 2 pages adopted
    try t.expectEqual(@as(usize, 1), x.f.blanks);
    try t.expectEqual(@as(u32, 2), x.pool.holders(adopt.items[0])); // the tree's and the caller's
    try x.rowsOk(adopt.items, 8); // the rows came back into the pages, a fresh pass's bytes
    try t.expectEqual(@as(usize, 1), x.f.live);
    // The tree's state goes; a pool with no room left cannot recall: the pass runs from the start.
    x.s.tree.evict(x.s.find(&longer).?);
    var full: [page_count - 2]u32 = undefined;
    try t.expect(x.pool.hold(&full));
    const again = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
    var adopt2: std.ArrayList(u32) = .empty;
    defer adopt2.deinit(t.allocator);
    const plan2 = try x.s.begin(t.allocator, &again, 8, &.{}, null, &adopt2);
    defer t.allocator.free(plan2.marks);
    try t.expectEqual(@as(u32, 0), plan2.from);
    try t.expectEqual(@as(usize, 0), adopt2.items.len);
}

test "TF_KV_SHARE's disk tier: two identities over one root keep their own directories and their own sets" {
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    const root = try tmp.dir.realPathFileAlloc(t.io, ".", t.allocator);
    defer t.allocator.free(root);
    const x = try Rig.open(t.allocator, root, 12);
    defer x.close();
    const y = try Rig.open(t.allocator, root, 13);
    defer y.close();
    try t.expect(std.mem.eql(u8, x.r.root, y.r.root));
    try t.expect(!std.mem.eql(u8, x.r.dir, y.r.dir)); // each identity word its own directory
    const prompt = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
    try x.pass(&prompt, &.{8});
    x.drain();
    try t.expectEqual(@as(usize, 1), x.r.states.items.len);
    try t.expectEqual(@as(usize, 0), y.r.states.items.len); // the other identity's store holds none of it
}
