//! The sharded pool's disk tier (TF_KV_SHARE with TF_GLM_KV_SHARD beside --learn-conversations): a kept state's rows
//! split by rank over a fake Parts — the owner part's latent rows, both parts' keys — written behind in pieces under
//! the staging permit, held through the parts while the save reads them, and recalled wholly into one part, as
//! cuda_glm_disk's sharded writer gathers and scatters them. A page here is 4 tokens, so the state is 128 pages
//! (two GLM pages' worth of blocks: 256-token blocks stay the rows store's).

const std = @import("std");
const pc = @import("prompt_cache.zig");
const radix = @import("prompt_radix.zig");
const rows = @import("rows_store.zig");
const Allocator = std.mem.Allocator;
const t = std.testing;

const page_tokens = 4; // tokens a page holds (rules.page)
const lat_row = 16; // a token's latent bytes (the owner rank's alone; wide so a block piece dwarfs the metadata)
const key_row = 8; // a pool's key bytes (both ranks hold them; a page holds one pool)
const lat_page = page_tokens * lat_row;
const words_bytes = 16; // the image's words: the position and the running sum
const fixed_bytes = 8; // the image's fixed part past the rows: the slot's own
const h = 256; // pages a part
const page_count = 2 * h; // pages the pool holds
const at_state = 512; // the kept state's tokens: 128 pages, two whole blocks
const block_tokens = rows.block_tokens; // 256: a block's rows a piece under a small permit
const block_bytes = block_tokens * lat_row + (block_tokens / page_tokens) * key_row;

/// A part's pages as the rig's holds: `hold` takes free pages, `retain` one more holder.
const Pool = struct {
    refs: [h]u32 = @splat(0),

    fn hold(p: *Pool, out: []u32) bool {
        var n: usize = 0;
        for (&p.refs, 0..) |*r, id| {
            if (r.* != 0) continue;
            out[n] = @intCast(id);
            r.* = 1;
            n += 1;
            if (n == out.len) return true;
        }
        for (out[0..n]) |id| p.refs[id] = 0; // nothing held: what it took goes back
        return false;
    }

    fn retain(p: *Pool, id: u32) void {
        p.refs[id] += 1;
    }

    fn release(p: *Pool, id: u32) void {
        p.refs[id] -= 1;
    }

    fn holders(p: *const Pool, id: u32) u32 {
        return p.refs[id];
    }

    fn available(p: *const Pool) usize {
        var n: usize = 0;
        for (&p.refs) |r| if (r == 0) {
            n += 1;
        };
        return n;
    }
};

/// Two parts' pools as the store's pages (core.kv_prompts.Parts'): a global id routes to its part's pool with the
/// part-local id, `choose` picks a part with room (home first, then the freer), `holdPages` holds a recall's pages
/// there as global ids, and a kept state's path never spans them (`onePart`).
const Parts = struct {
    pools: [2]Pool = .{ .{}, .{} },
    refused_count: u64 = 0,

    fn pool(p: *Parts, id: u32) *Pool {
        return &p.pools[id / h];
    }

    fn local(id: u32) u32 {
        return id - @as(u32, @intCast(id / h)) * h;
    }

    pub fn retain(p: *Parts, id: u32) void {
        p.pool(id).retain(local(id));
    }

    pub fn release(p: *Parts, id: u32) void {
        p.pool(id).release(local(id));
    }

    pub fn holders(p: *Parts, id: u32) u32 {
        return p.pool(id).holders(local(id));
    }

    pub fn available(p: *Parts) usize {
        return p.pools[0].available() + p.pools[1].available();
    }

    pub fn hold(p: *Parts, out: []u32) bool {
        return p.holdPages(out);
    }

    /// `out.len` pages in one part with room (its pools reclaiming first is the caller's), as global ids.
    pub fn holdPages(p: *Parts, out: []u32) bool {
        const part = p.choose(out.len, null) orelse return false;
        if (!p.pools[part].hold(out)) return false;
        for (out) |*id| id.* += @as(u32, part) * h;
        return true;
    }

    pub fn choose(p: *Parts, n: usize, home: ?u8) ?u8 {
        if (home) |hp| if (p.pools[hp].available() >= n) return hp;
        var best: ?u8 = null;
        for (0..2) |i| {
            if (p.pools[i].available() < n) continue;
            if (best == null or p.pools[i].available() > p.pools[best.?].available()) best = @intCast(i);
        }
        return best;
    }

    pub fn onePart(_: *Parts, ids: []const u32) bool {
        const one = ids[0] / h;
        for (ids) |id| if (id / h != one) return false;
        return true;
    }
};

/// The family: the parts' pages' bytes in host memory, a state the words and fixed part past its pages' rows — its
/// latent rows the home part's rank holds alone, its keys both ranks do (the gather reads rank 0's, the read-back
/// writes both).
const Fake = struct {
    gpa: Allocator,
    prompt: []const u32 = &.{},
    pts: *Parts,
    keys: [2][]u8, // rank r's key pages: every page's keys, a page's at its global id
    lat: [2][]u8, // rank r's latent pages: its own part's, a page's at its part-local id
    live: usize = 0,
    blanks: usize = 0,
    gathers: usize = 0,
    stream_home: u8 = 0, // the part the live stream's pages lie in (a keep's home, as ranks.readHome's)
    recall_home: ?u8 = null, // the part a recall's pages are held in (null: `choose`'s)

    fn of(ptr: *anyopaque) *Fake {
        return @ptrCast(@alignCast(ptr));
    }

    /// A token's latent bytes, as a pass writes them into its page on the owner rank.
    fn row(f: *const Fake, p: u32) [lat_row]u8 {
        var out: [lat_row]u8 = undefined;
        var x = std.hash.Wyhash.hash(4, std.mem.sliceAsBytes(&[_]u32{f.prompt[p]}));
        for (&out) |*b| {
            x = x *% 0x100000001b3 +% 1;
            b.* = @truncate(x >> 24);
        }
        return out;
    }

    /// A pool's key bytes, as both ranks' passes write them.
    fn key(f: *const Fake, p: u32) [key_row]u8 {
        var out: [key_row]u8 = undefined;
        var x = std.hash.Wyhash.hash(5, std.mem.sliceAsBytes(&[_]u32{f.prompt[p]}));
        for (&out) |*b| {
            x = x *% 0x100000001b3 +% 3;
            b.* = @truncate(x >> 24);
        }
        return out;
    }

    /// The stream's rows into its pages: the latent rows on its part's rank, the keys on both.
    fn fill(f: *Fake, pages: []const u32, at: u32, part: u8) void {
        for (0..at) |tok| {
            const r = f.row(@intCast(tok));
            const base = @as(usize, pages[tok / page_tokens] - @as(u32, part) * h) * lat_page + (tok % page_tokens) * lat_row;
            @memcpy(f.lat[part][base..][0..lat_row], &r);
        }
        for (0..at / page_tokens) |i| {
            const k = f.key(@intCast(i * page_tokens));
            const base = @as(usize, pages[i]) * key_row;
            @memcpy(f.keys[0][base..][0..key_row], &k);
            @memcpy(f.keys[1][base..][0..key_row], &k);
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
        const slot = try f.gpa.alloc(u8, words_bytes + fixed_bytes);
        errdefer f.gpa.free(slot);
        f.build(at, slot);
        k.* = .{ .at = at, .slot = slot, .home = f.stream_home };
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

    const Kept = struct { at: u32, slot: []u8, home: u8 = 0 };

    /// Which bytes of the image [off, off + len) are: the words, a run of a page's latent rows or keys, or the fixed part.
    const Seg = union(enum) { words: usize, lat: struct { id: u32, at: usize }, keys: struct { id: u32, at: usize }, fixed: usize };

    fn seg(_: *Fake, k: *Kept, pages: []const u32, off: u64) !Seg {
        const lat_end = words_bytes + @as(u64, k.at) * lat_row;
        const keys_end = lat_end + @as(u64, k.at / page_tokens) * key_row;
        if (off < words_bytes) return .{ .words = @intCast(off) };
        if (off < lat_end) {
            const b = off - words_bytes;
            const i = b / lat_page;
            if (i >= pages.len) return error.BadKeptRange;
            return .{ .lat = .{ .id = pages[@intCast(i)], .at = @intCast(b % lat_page) } };
        }
        if (off < keys_end) {
            const b = off - lat_end;
            const i = b / key_row;
            if (i >= pages.len) return error.BadKeptRange;
            return .{ .keys = .{ .id = pages[@intCast(i)], .at = 0 } };
        }
        if (off < keys_end + fixed_bytes) return .{ .fixed = @intCast(off - keys_end) };
        return error.BadKeptRange;
    }

    /// The store's tiny image: the words, a state's latent rows a row a token (its pages', the home part's rank's),
    /// its pools' keys a row a pool (both ranks'), the fixed part.
    const image: rows.Image = .{ .parts = partsFn, .gather = gatherFn, .blank = blankFn, .scatter = scatterFn, .blank_pages = blankPagesFn };

    fn partsFn(_: *anyopaque, at: u32, out: *std.ArrayList(rows.image.Part), a: Allocator) anyerror!void {
        try out.appendSlice(a, &.{
            .{ .bytes = words_bytes },
            .{ .bytes = @as(u64, at) * lat_row, .row = lat_row },
            .{ .bytes = @as(u64, at / page_tokens) * key_row, .row = key_row, .per = page_tokens },
            .{ .bytes = fixed_bytes },
        });
    }

    fn gatherFn(ptr: *anyopaque, saved: pc.Saved, pages: []const u32, ranges: []const rows.image.Range, out: []u8) anyerror!void {
        const f = of(ptr);
        const k: *Kept = @ptrCast(@alignCast(saved));
        f.gathers += 1;
        var o: usize = 0;
        for (ranges) |g| {
            var done: u64 = 0;
            while (done < g.len) {
                const s = try seg(f, k, pages, g.off + done);
                const n: usize = switch (s) {
                    .words => |at| @intCast(@min(g.len - done, words_bytes - at)),
                    .lat => |p| @intCast(@min(g.len - done, lat_page - p.at)),
                    .keys => |p| @intCast(@min(g.len - done, key_row - p.at)),
                    .fixed => |at| @intCast(@min(g.len - done, fixed_bytes - at)),
                };
                const host = out[o..][0..n];
                switch (s) {
                    .words => |at| @memcpy(host, k.slot[at..][0..n]),
                    .lat => |p| @memcpy(host, f.lat[k.home][@as(usize, p.id - @as(u32, k.home) * h) * lat_page + p.at ..][0..n]),
                    .keys => |p| @memcpy(host, f.keys[0][@as(usize, p.id) * key_row + p.at ..][0..n]),
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
        const home: ?u8 = if (f.recall_home) |part| part else f.pts.choose(out.items.len, null);
        const part = home orelse return error.LaneMemory;
        if (!f.pts.pools[part].hold(out.items)) return error.LaneMemory; // the pages held for the caller's tree
        for (out.items) |*id| id.* += @as(u32, part) * h;
        k.* = .{ .at = at, .slot = slot, .home = part };
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
                const s = try seg(f, k, pages, g.off + done);
                const n: usize = switch (s) {
                    .words => |at| @intCast(@min(g.len - done, words_bytes - at)),
                    .lat => |p| @intCast(@min(g.len - done, lat_page - p.at)),
                    .keys => |p| @intCast(@min(g.len - done, key_row - p.at)),
                    .fixed => |at| @intCast(@min(g.len - done, fixed_bytes - at)),
                };
                const part = in[o..][0..n];
                switch (s) {
                    .words => |at| @memcpy(k.slot[at..][0..n], part),
                    // the latent rows land on the new owner's pages alone
                    .lat => |p| @memcpy(f.lat[k.home][@as(usize, p.id - @as(u32, k.home) * h) * lat_page + p.at ..][0..n], part),
                    // the keys land on both ranks' key pages
                    .keys => |p| {
                        @memcpy(f.keys[0][@as(usize, p.id) * key_row + p.at ..][0..n], part);
                        @memcpy(f.keys[1][@as(usize, p.id) * key_row + p.at ..][0..n], part);
                    },
                    .fixed => |at| @memcpy(k.slot[words_bytes + at ..][0..n], part),
                }
                done += n;
                o += n;
            }
        }
    }
};

/// The store over a fake sharded pool: the parts the tree's pages, the disk tier attached (async writes, so a save
/// runs in pieces under the staging permit).
const Rig = struct {
    gpa: Allocator,
    f: Fake,
    pts: Parts,
    s: radix.Store,
    r: *rows.Rows,
    keys_mem: [2][]u8,
    lat_mem: [2][]u8,

    fn open(gpa: Allocator, root: []const u8, identity: u64, staging: ?u64) !*Rig {
        const x = try gpa.create(Rig);
        errdefer gpa.destroy(x);
        const keys0 = try gpa.alloc(u8, page_count * key_row);
        errdefer gpa.free(keys0);
        const keys1 = try gpa.alloc(u8, page_count * key_row);
        errdefer gpa.free(keys1);
        const lat0 = try gpa.alloc(u8, h * lat_page);
        errdefer gpa.free(lat0);
        const lat1 = try gpa.alloc(u8, h * lat_page);
        errdefer gpa.free(lat1);
        @memset(keys0, 0);
        @memset(keys1, 0);
        @memset(lat0, 0);
        @memset(lat1, 0);
        x.* = .{ .gpa = gpa, .f = undefined, .pts = .{}, .s = undefined, .r = undefined, .keys_mem = .{ keys0, keys1 }, .lat_mem = .{ lat0, lat1 } };
        x.f = .{ .gpa = gpa, .pts = &x.pts, .keys = .{ keys0, keys1 }, .lat = .{ lat0, lat1 } };
        const family = x.f.family();
        x.s = try radix.Store.init(gpa, family, radix.pagesOf(Parts, &x.pts, lat_page), .{ .page = page_tokens, .min_gap = 8, .min_prompt = 0 });
        errdefer x.s.deinit();
        x.s.parts = .{ .ctx = &x.pts, .one = partingOne, .refused = partingRefused }; // a kept state's path stays in one part
        x.s.limit(.{ .pages = page_count, .snaps = 8 });
        x.r = try rows.Rows.open(gpa, family, .{
            .root = root,
            .identity = identity,
            .cap = 1 << 30,
            .floor = 0,
            .daily = 1 << 30,
            .staging_bytes = staging orelse (1 << 30),
        });
        errdefer x.r.deinit();
        x.r.prefill_rate = 1.0; // a token a second: reading from disk pays over prefilling
        x.s.attachDisk(x.r);
        return x;
    }

    fn close(x: *Rig) void {
        x.s.deinit(); // the writer's queued saves go first (their pages' holds back)
        x.r.deinit();
        x.gpa.free(x.keys_mem[0]);
        x.gpa.free(x.keys_mem[1]);
        x.gpa.free(x.lat_mem[0]);
        x.gpa.free(x.lat_mem[1]);
        x.gpa.destroy(x);
    }

    fn partingOne(ptr: *anyopaque, ids: []const u32) bool {
        const p: *Parts = @ptrCast(@alignCast(ptr));
        return p.onePart(ids);
    }

    fn partingRefused(ptr: *anyopaque) void {
        const p: *Parts = @ptrCast(@alignCast(ptr));
        p.refused_count += 1;
    }

    /// A pass of `prompt` in `part`: the pages a stream of that part takes and fills with its rows (its latent rows
    /// on its rank, its keys on both), a keep at each mark, the stream's end (its holds go; the tree's remain).
    fn pass(x: *Rig, prompt: []const u32, marks: []const u32, part: u8) !void {
        x.f.prompt = prompt;
        x.f.stream_home = part;
        const np = prompt.len / page_tokens;
        var mine: [page_count]u32 = undefined;
        try t.expect(x.pts.pools[part].hold(mine[0..np]));
        for (mine[0..np]) |*id| id.* += @as(u32, part) * h;
        x.f.fill(mine[0..np], @intCast(prompt.len), part);
        for (marks) |m| {
            var path: [page_count]u32 = undefined;
            const kept = x.s.keep(prompt, m, null, mine[0 .. m / page_tokens], path[0 .. m / page_tokens]);
            try t.expect(kept.held);
        }
        for (mine[0..np]) |id| x.pts.release(id);
    }

    /// The writer runs: the queued saves land (their pages' holds back).
    fn drain(x: *Rig) void {
        const deadline = rows.write.nowNs() + 30 * std.time.ns_per_s;
        while (rows.write.pending(x.r) and rows.write.nowNs() < deadline) {
            if (!rows.write.step(x.r)) std.Io.sleep(t.io, .fromMilliseconds(1), .awake) catch {};
        }
        t.expect(!rows.write.pending(x.r)) catch {};
    }

    /// Whether a state's rows are a fresh pass's: its latent rows on its home rank, its keys on both.
    fn rowsOk(x: *Rig, pages: []const u32, at: u32, part: u8) !void {
        for (0..at) |tok| {
            const r = x.f.row(@intCast(tok));
            const base = @as(usize, pages[tok / page_tokens] - @as(u32, part) * h) * lat_page + (tok % page_tokens) * lat_row;
            try t.expectEqualSlices(u8, &r, x.f.lat[part][base..][0..lat_row]);
        }
        for (0..at / page_tokens) |i| {
            const k = x.f.key(@intCast(i * page_tokens));
            const base = @as(usize, pages[i]) * key_row;
            try t.expectEqualSlices(u8, &k, x.f.keys[0][base..][0..key_row]);
            try t.expectEqualSlices(u8, &k, x.f.keys[1][base..][0..key_row]);
        }
    }
};

const prompt512: [at_state + 8]u32 = blk: {
    var tokens: [at_state + 8]u32 = undefined;
    for (&tokens, 0..) |*tok, i| tok.* = @intCast(1 + i % 97);
    break :blk tokens;
};

test "the sharded pool's disk tier: a state in part 1 written behind in pieces under a small permit, its owner's latent rows and both ranks' keys" {
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    const root = try tmp.dir.realPathFileAlloc(t.io, ".", t.allocator);
    defer t.allocator.free(root);
    // a permit that holds one block piece and the slot's own file, but not both blocks beside it: the rows go in
    // pieces, one rank's part of them at a time (the whole state's charge past the pieces waits for room)
    const permit = 2 * block_bytes;
    const x = try Rig.open(t.allocator, root, 21, permit);
    defer x.close();
    try x.pass(&prompt512, &.{at_state}, 1);
    try t.expectEqual(@as(usize, 1), x.s.tree.snaps);
    try t.expectEqual(@as(usize, 1), x.s.saves.items.len); // the save queued, its pages held through the parts
    for (x.s.find(&prompt512).?.pages) |id| try t.expectEqual(@as(u32, 2), x.pts.holders(id)); // the tree's and the writer's
    x.drain();
    try t.expectEqual(@as(usize, 1), x.r.states.items.len);
    try t.expectEqual(@as(u64, 1), x.r.counts.writes);
    try t.expectEqual(@as(usize, 0), x.s.saves.items.len);
    for (x.s.find(&prompt512).?.pages) |id| try t.expectEqual(@as(u32, 1), x.pts.holders(id)); // the tree's alone again
    // the permit never held the whole state's charge (its rows, its own file and its metadata), yet the save landed:
    // it went in pieces, a block a piece, and both are on disk
    try t.expect(permit < 2 * block_bytes + words_bytes + fixed_bytes + 4 * at_state);
    try t.expect(x.r.states.items[0].rows >= 2 * block_bytes); // both blocks' rows are on disk
}

test "the sharded pool's disk tier: a part-0 promise while a part-0 state is mid-write holds its pages, the promise waits" {
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    const root = try tmp.dir.realPathFileAlloc(t.io, ".", t.allocator);
    defer t.allocator.free(root);
    const x = try Rig.open(t.allocator, root, 22, null);
    defer x.close();
    try x.pass(&prompt512, &.{at_state}, 0);
    try t.expectEqual(@as(usize, 1), x.s.saves.items.len); // the save queued: its pages held once more
    const node = x.s.find(&prompt512).?;
    // part 0 full but the state's pages: a promise there cannot take them under the writer, and the state's own
    // leaf is not idle (the writer holds its pages), so the part's reclaim frees nothing
    var rest: [h - at_state / page_tokens]u32 = undefined;
    try t.expect(x.pts.pools[0].hold(&rest));
    try t.expectEqual(@as(usize, 0), x.pts.pools[0].available());
    try t.expect(!x.s.reclaim(1, .{ .ctx = &x.pts, .f = part0, .free = part0Free }));
    for (node.pages) |id| try t.expectEqual(@as(u32, 2), x.pts.holders(id)); // the tree's and the writer's
    try t.expectEqual(@as(u64, 0), x.r.counts.writes);
    for (&rest) |id| x.pts.release(id);
    x.drain(); // the save lands once the writer is through: its pages back to the tree's alone
    try t.expectEqual(@as(u64, 1), x.r.counts.writes);
    for (node.pages) |id| try t.expectEqual(@as(u32, 1), x.pts.holders(id));
}

fn part0(_: *anyopaque, first: u32) bool {
    return first / h == 0;
}

fn part0Free(ptr: *anyopaque) usize {
    const p: *Parts = @ptrCast(@alignCast(ptr));
    return p.pools[0].available();
}

test "the sharded pool's disk tier: an eviction while the save is queued writes it first; a part's reclaim evicts its own leaves alone" {
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    const root = try tmp.dir.realPathFileAlloc(t.io, ".", t.allocator);
    defer t.allocator.free(root);
    const x = try Rig.open(t.allocator, root, 23, null);
    defer x.close();
    // a part-0 state whose save is queued: its eviction stages the save now (its pages held to the end), and the
    // write lands even though the tree's entry is gone
    try x.pass(&prompt512, &.{at_state}, 0);
    try t.expectEqual(@as(usize, 1), x.s.saves.items.len);
    try t.expectEqual(@as(u64, 0), x.r.counts.writes);
    x.s.tree.evict(x.s.find(&prompt512).?);
    x.drain();
    try t.expectEqual(@as(u64, 1), x.r.counts.writes);
    try t.expectEqual(@as(usize, 0), x.s.saves.items.len);
    try t.expectEqual(@as(usize, 1), x.r.states.items.len);
    // a second part-0 leaf and a part-1 leaf (other tokens, so each keep is its own leaf), both written: a part-0
    // reclaim evicts part 0's leaf alone and leaves part 1's
    const again = blk: {
        var toks = prompt512;
        for (&toks) |*tok| tok.* = tok.* % 97 + 101;
        break :blk toks;
    };
    const part1 = blk: {
        var toks = prompt512;
        for (&toks) |*tok| tok.* = tok.* % 89 + 201;
        break :blk toks;
    };
    try x.pass(&again, &.{at_state}, 0);
    try x.pass(&part1, &.{at_state}, 1);
    x.drain();
    try t.expectEqual(@as(u64, 3), x.r.counts.writes);
    const part1_node = x.s.find(&part1).?;
    try t.expectEqual(@as(usize, 2), x.s.tree.snaps);
    var rest0: [h - at_state / page_tokens]u32 = undefined;
    try t.expect(x.pts.pools[0].hold(&rest0)); // part 0 full but its leaf's pages: they are the tree's alone
    for (x.s.find(&again).?.pages) |id| try t.expectEqual(@as(u32, 1), x.pts.holders(id));
    try t.expect(x.s.reclaim(1, .{ .ctx = &x.pts, .f = part0, .free = part0Free }));
    try t.expectEqual(@as(usize, 1), x.s.tree.snaps); // part 0's leaf went, part 1's stands
    try t.expectEqual(part1_node, x.s.find(&part1).?);
    for (part1_node.pages) |id| try t.expectEqual(@as(u32, 1), x.pts.holders(id));
    for (&rest0) |id| x.pts.release(id);
}

test "the sharded pool's disk tier: a restart recalls a state into one part, its latent rows on the new owner alone, the keys on both ranks" {
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    const root = try tmp.dir.realPathFileAlloc(t.io, ".", t.allocator);
    defer t.allocator.free(root);
    {
        const x = try Rig.open(t.allocator, root, 24, null);
        defer x.close();
        try x.pass(&prompt512, &.{at_state}, 1); // saved from part 1
        x.drain();
        try t.expectEqual(@as(usize, 1), x.r.states.items.len);
    }
    // A restart: a new store over the same parts reads the state back, this time into part 0 (the other part than it
    // was saved from): its pages wholly part 0's, its latent rows on rank 0 alone, its keys on both ranks.
    const x = try Rig.open(t.allocator, root, 24, null);
    defer x.close();
    x.f.recall_home = 0;
    const longer = prompt512 ++ [_]u32{ 91, 92, 93, 94, 95, 96, 97, 98 };
    x.f.prompt = &longer;
    var adopt: std.ArrayList(u32) = .empty;
    defer {
        for (adopt.items) |id| x.pts.release(id); // the caller's holds; the tree's go with the store
        adopt.deinit(t.allocator);
    }
    const plan = try x.s.begin(t.allocator, &longer, at_state, &.{}, null, &adopt);
    defer t.allocator.free(plan.marks);
    try t.expectEqual(@as(u32, at_state), plan.from); // the recalled state is the resume
    try t.expectEqual(@as(usize, at_state / page_tokens), adopt.items.len);
    try t.expectEqual(@as(usize, 1), x.f.blanks);
    try t.expect(x.pts.onePart(adopt.items)); // the recalled path is one part's: T05d's invariant holds
    for (adopt.items) |id| try t.expectEqual(@as(u32, 2), x.pts.holders(id)); // the tree's and the caller's
    try x.rowsOk(adopt.items, at_state, 0); // the rows came back, a fresh pass's bytes, the latent rows on rank 0
    try t.expectEqual(@as(usize, 1), x.f.live);
}

test "the sharded pool's disk tier: two identities over one root keep their own directories and their own sets" {
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    const root = try tmp.dir.realPathFileAlloc(t.io, ".", t.allocator);
    defer t.allocator.free(root);
    const x = try Rig.open(t.allocator, root, 25, null);
    defer x.close();
    const y = try Rig.open(t.allocator, root, 26, null);
    defer y.close();
    try t.expect(std.mem.eql(u8, x.r.root, y.r.root));
    try t.expect(!std.mem.eql(u8, x.r.dir, y.r.dir)); // each identity word its own directory
    try x.pass(&prompt512, &.{at_state}, 1);
    x.drain();
    try t.expectEqual(@as(usize, 1), x.r.states.items.len);
    try t.expectEqual(@as(usize, 0), y.r.states.items.len); // the other identity's store holds none of it
}
