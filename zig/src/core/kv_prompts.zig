//! TF_KV_SHARE: what a family on the core pool calls of one prompt store over its pages (native/cuda_prompt_pages.zig).
const std = @import("std");
const kv = @import("kv_pool.zig");

/// Where a pass resumes and the marks it keeps states at (`marks` the store's allocation, freed by `free`).
pub const Plan = struct { from: u32 = 0, marks: []const u32 = &.{} };

/// The store as a family sees it: a prompt pass begins, keeps at its marks, and reports once its prompt is in.
pub const Prompts = struct {
    ctx: *anyopaque,
    vtable: *const VTable,

    pub const VTable = struct {
        /// The deepest state `prompt` resumes: its pages into `adopt` (held once more), the rest restored into `owner`.
        begin: *const fn (ctx: *anyopaque, prompt: []const u32, history_len: u32, shared: []const u32, owner: *anyopaque, adopt: *std.ArrayList(u32)) anyerror!Plan,
        /// At `at` the store takes the pages of `mine` it lacks and the state; `path`'s first n replace the stream's.
        keep: *const fn (ctx: *anyopaque, prompt: []const u32, at: u32, owner: *anyopaque, mine: []const u32, path: []u32) usize,
        /// One log line after a prompt pass: where it resumed and how many states it kept.
        report: *const fn (ctx: *anyopaque, prompt: usize, from: u32, kept: u64) void,
        /// The kept state at `at` for `prompt` goes (a peer could not resume it, or its pages would span the parts).
        forget: *const fn (ctx: *anyopaque, prompt: []const u32, at: u32) void,
        free: *const fn (ctx: *anyopaque, marks: []const u32) void,
    };

    pub fn begin(p: Prompts, prompt: []const u32, history_len: u32, shared: []const u32, owner: *anyopaque, adopt: *std.ArrayList(u32)) !Plan {
        return p.vtable.begin(p.ctx, prompt, history_len, shared, owner, adopt);
    }

    pub fn keep(p: Prompts, prompt: []const u32, at: u32, owner: *anyopaque, mine: []const u32, path: []u32) usize {
        return p.vtable.keep(p.ctx, prompt, at, owner, mine, path);
    }

    pub fn report(p: Prompts, prompt: usize, from: u32, kept: u64) void {
        p.vtable.report(p.ctx, prompt, from, kept);
    }

    pub fn forget(p: Prompts, prompt: []const u32, at: u32) void {
        p.vtable.forget(p.ctx, prompt, at);
    }

    pub fn free(p: Prompts, marks: []const u32) void {
        p.vtable.free(p.ctx, marks);
    }
};

/// A kept state's copied rows in store-held pages: byte o in chunk c = o / chunk, region c % regions, page c / regions.
pub const Blob = struct {
    ids: []const u32,
    chunk: u64, // the bytes one page holds in one region (a layer's K or V)
    regions: usize,

    /// Pages that hold `bytes` this way.
    pub fn pagesFor(bytes: u64, chunk: u64, regions: usize) usize {
        const page = chunk * regions;
        return @intCast((bytes + page - 1) / page);
    }

    /// Bytes [off, off + len) as pieces inside one chunk: f(ctx, region, page id, in-chunk offset, from `off`, len).
    pub fn each(b: Blob, off: u64, len: u64, ctx: anytype, comptime f: fn (@TypeOf(ctx), usize, u32, u64, u64, u64) anyerror!void) !void {
        var o = off;
        while (o < off + len) {
            const c = o / b.chunk;
            const within = o % b.chunk;
            const n = @min(off + len - o, b.chunk - within);
            const k: usize = @intCast(c / b.regions);
            if (k >= b.ids.len) return error.PastBlob;
            try f(ctx, @intCast(c % b.regions), b.ids[k], within, o - off, n);
            o += n;
        }
    }
};

/// The pages a family's store keeps prompts in (`share_pool`'s): one pool, or a sharded pool's parts
/// (TF_GLM_KV_SHARD's parts, each a kv_pool.Pool).
pub const Pool = union(enum) {
    one: *kv.Pool,
    parts: Parts,
};

/// A sharded pool's parts as the store's pages (TF_GLM_KV_SHARD's, each part a kv_pool.Pool): global page ids in,
/// each routed to its part's pool with the part-local id; the store's page limit is every part's, and each part's
/// pool takes its own reclaim hook (a promise there evicts that part's idle leaves first).
pub const Parts = struct {
    /// The tree's page ids over the parts, the pages' four calls on `ctx` (prompt_cache.Pages').
    retain: *const fn (ctx: *anyopaque, id: u32) void,
    release: *const fn (ctx: *anyopaque, id: u32) void,
    holders: *const fn (ctx: *anyopaque, id: u32) u32,
    /// Pages free and not promised, both parts together (the store's logging only).
    available: *const fn (ctx: *anyopaque) usize,
    /// Bytes of one page, over every layer (the tree's held count).
    page_bytes: u64,
    /// The store's page limit: every part's pages.
    total: usize,
    /// The parts (two at TP=2, lanes.kv_parts.max_parts).
    count: usize,
    /// Free pages in `part` now (its part's reclaim evicts its leaves until `want` of these).
    free: *const fn (ctx: *anyopaque, part: usize) usize,
    /// Whether a tree leaf whose first page is `first` lies in `part` (a leaf's pages always lie in one).
    inPart: *const fn (ctx: *anyopaque, part: usize, first: u32) bool,
    /// Whether `ids` all lie in one part (a stream's pages do; the tree takes a leaf's pages as one).
    onePart: *const fn (ctx: *anyopaque, ids: []const u32) bool,
    /// `part`'s pool takes the store's reclaim hook, or drops it.
    setReclaim: *const fn (ctx: *anyopaque, part: usize, hook: ?kv.Reclaim) void,
    /// `out.len` pages held for a recalled state in one part with room (its pools reclaiming first), global ids
    /// (false: none; a recall's home part, cuda_kv_shard.Parts.choose). Null: the parts cannot hold pages.
    holdPages: ?*const fn (ctx: *anyopaque, out: []u32) bool = null,
    /// The store refused a keep whose pages would span the parts (TF_GLM_KV_SHARD): the family counts it.
    refused: *const fn (ctx: *anyopaque) void,
    ctx: *anyopaque,
};

/// TF_KV_SHARE=n: keep prompts as pool pages and at most n states past them (0 or unset: copies, as before).
pub fn sharing() error{BadKvShare}!usize {
    const v = std.c.getenv("TF_KV_SHARE") orelse return 0;
    return std.fmt.parseInt(usize, std.mem.span(v), 10) catch error.BadKvShare;
}

const testing = std.testing;

const Pieces = struct {
    got: *std.ArrayList([5]u64),
    fn add(p: Pieces, region: usize, id: u32, within: u64, from: u64, n: u64) anyerror!void {
        try p.got.append(testing.allocator, .{ region, id, within, from, n });
    }
};

test "a blob's bytes walk its regions a chunk at a time, then its next page; a run splits at chunk edges" {
    const b: Blob = .{ .ids = &.{ 7, 3 }, .chunk = 10, .regions = 2 };
    try testing.expectEqual(@as(usize, 2), Blob.pagesFor(21, 10, 2));
    var got: std.ArrayList([5]u64) = .empty;
    defer got.deinit(testing.allocator);
    try b.each(5, 30, Pieces{ .got = &got }, Pieces.add); // bytes 5..35: chunks 0 (page 7, K), 1 (page 7, V), 2 (page 3, K), 3 (page 3, V)
    try testing.expectEqualSlices([5]u64, &.{ .{ 0, 7, 5, 0, 5 }, .{ 1, 7, 0, 5, 10 }, .{ 0, 3, 0, 15, 10 }, .{ 1, 3, 0, 25, 5 } }, got.items);
    try testing.expectError(error.PastBlob, b.each(39, 2, Pieces{ .got = &got }, Pieces.add));
}
