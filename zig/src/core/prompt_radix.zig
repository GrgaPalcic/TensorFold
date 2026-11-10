//! Prompt reuse as a radix tree over token pages: a request takes its longest match and resumes its deepest snapshot.

const std = @import("std");
const pc = @import("prompt_cache.zig");
const tree_mod = @import("prompt_radix_tree.zig");
const rows = @import("rows_store.zig");
const select = @import("rows_select.zig");
const Allocator = std.mem.Allocator;

pub const Saved = tree_mod.Saved;
pub const Snapshots = tree_mod.Snapshots;
pub const Pages = tree_mod.Pages;
pub const LeafFilter = tree_mod.LeafFilter;
pub const Counts = tree_mod.Counts;
pub const Node = tree_mod.Node;
pub const Tree = tree_mod.Tree;

/// The rules of prompt_cache.zig, paged: `page` set; a planned store cuts and resumes only on `grid`, a multiple of it.
pub const Rules = pc.Rules;

/// What the tree may hold: pages, and snapshots.
pub const Limits = struct { pages: usize = 0, snaps: usize = 0 };

/// An owner's pages as the store holds them: `T` has retain, release, holders and available (kv_pool.Pool).
pub fn pagesOf(comptime T: type, owner: *T, bytes: u64) Pages {
    const V = struct {
        fn of(p: *anyopaque) *T {
            return @ptrCast(@alignCast(p));
        }
        fn retain(p: *anyopaque, id: u32) void {
            of(p).retain(id);
        }
        fn release(p: *anyopaque, id: u32) void {
            of(p).release(id);
        }
        fn holders(p: *anyopaque, id: u32) u32 {
            return of(p).holders(id);
        }
        fn available(p: *anyopaque) usize {
            return of(p).available();
        }
        fn hold(p: *anyopaque, out: []u32) bool {
            return of(p).hold(out);
        }
    };
    return .{ .ptr = owner, .bytes = bytes, .vtable = &.{ .retain = V.retain, .release = V.release, .holders = V.holders, .available = V.available, .hold = V.hold } };
}

pub const Plan = pc.Plan;
/// A paged store's `keep`: whether the mark holds a state, and how many leading pages the stream swaps for the store's.
pub const Kept = struct { held: bool, shared: usize };

pub const Store = struct {
    tree: Tree,
    rules: Rules,
    limits: Limits = .{},
    /// The tokens of recently planned shared cuts, hashed (at most SHARED_KEYS).
    shared_keys: std.ArrayList(u64) = .empty,
    /// --learn-conversations: every kept state written behind to disk, read back where it pays (prompt_cache.Store's).
    disk: ?*rows.Rows = null,
    /// Saves of nodes under way or queued: their pages held once more, their tokens copied (rows_write's Kept.freed).
    saves: std.ArrayList(Save) = .empty,

    /// A save of a node the writer reads: the pages held for it, and the tokens its blocks are named by.
    const Save = struct { node: *Node, pages: []u32, tokens: []u32 };

    const SHARED_KEYS = 64;

    /// --learn-conversations on a paged store: every keep written behind (its pages held while the save reads them),
    /// an eviction's still-queued save staged, and a begin that misses reading the longest state on disk that pays.
    pub fn attachDisk(s: *Store, r: *rows.Rows) void {
        s.disk = r;
        s.tree.on_evict = .{ .ctx = s, .call = evicting };
    }

    pub fn init(gpa: Allocator, family: Snapshots, pages: Pages, rules: Rules) !Store {
        if (rules.page == 0 or (rules.planned and (rules.grid == 0 or rules.grid % rules.page != 0))) return error.BadPromptRadixRules;
        return .{ .tree = try Tree.init(gpa, family, pages, rules.page, rules.lookahead), .rules = rules };
    }

    /// Where a pass may cut and resume: a page edge, or a planned store's grid.
    fn step(s: *const Store) u32 {
        return if (s.rules.planned) s.rules.grid else s.rules.page;
    }

    /// Tokens a resume leaves past its state: the lookahead it read, and at least one to prefill.
    fn past(s: *const Store) u32 {
        return @max(1, s.rules.lookahead);
    }

    pub fn deinit(s: *Store) void {
        while (s.saves.items.len > 0) if (s.disk) |r| { // the writer's queue lets the saves go first (their pages back)
            rows.write.dropped(r, s.saves.items[s.saves.items.len - 1].node);
        } else break;
        s.saves.deinit(s.tree.gpa);
        s.shared_keys.deinit(s.tree.gpa);
        s.tree.deinit();
    }

    /// The tree may hold `limits` from now on: what passes them goes first.
    pub fn limit(s: *Store, limits: Limits) void {
        s.limits = limits;
        while (s.tree.snaps > limits.snaps) s.tree.evict(s.victim(&.{}, null) orelse break);
        while (s.tree.held_pages > limits.pages) if (!s.tree.evictLeaf(null)) break;
    }

    /// Whether anything can be kept.
    pub fn on(s: *const Store) bool {
        return s.limits.pages > 0 and s.limits.snaps > 0;
    }

    /// The deepest node whose state `prompt` resumes exactly: its tokens and lookahead a prefix, a token left.
    pub fn find(s: *const Store, prompt: []const u32) ?*Node {
        if (!s.on() or prompt.len < s.past() + 1) return null;
        return s.tree.deepest(prompt, (prompt.len - s.past()) / s.rules.page * s.rules.page, s.step());
    }

    /// Restores the longest state `prompt` resumes into `owner`, its pages into `adopt` (held once more); plans marks.
    /// With a disk store, a miss (or a tree state shorter than one on disk) recalls the longest state there that pays.
    pub fn begin(s: *Store, a: Allocator, prompt: []const u32, history_len: u32, shared: []const u32, owner: ?*anyopaque, adopt: *std.ArrayList(u32)) !Plan {
        if (!s.on()) return .{ .marks = try a.alloc(u32, 0) };
        const found0 = s.find(prompt);
        if (found0 == null) s.tree.counts.misses += 1;
        const found = if (s.disk) |r| s.recall(r, prompt, found0) else found0;
        const resume_at: u32 = if (found) |n| n.end else 0;
        const planned = try s.marks(a, prompt, resume_at, history_len, shared, if (found) |n| n.entry.?.last else &.{});
        const marks_ = try s.fitting(a, planned);
        errdefer a.free(marks_);
        for (shared) |w| { // the shared cuts this pass keeps: their states serve other conversations too
            const cut = w - w % s.step();
            if (std.mem.indexOfScalar(u32, marks_, cut) != null) s.noteShared(prompt[0 .. cut + s.rules.lookahead]);
        }
        s.reserve(prompt, found, marks_);
        var from: u32 = 0;
        if (found) |n| {
            const family = s.tree.family;
            const ok = if (family.vtable.restore(family.ptr, owner, n.entry.?.saved)) |_| true else |err| blk: {
                note("restoring {d} tokens failed ({s}); prefilling from the start", .{ n.end, @errorName(err) });
                break :blk false;
            };
            if (ok) {
                try s.tree.take(n, adopt);
                from = n.end;
            }
            s.resumed(n, prompt, ok);
        }
        return .{ .from = from, .marks = marks_ };
    }

    /// A restored state: it is now the prompt's (a failed restore drops it).
    pub fn resumed(s: *Store, n: *Node, prompt: []const u32, ok: bool) void {
        if (!ok) {
            s.tree.counts.failed += 1;
            return s.tree.evict(n);
        }
        const e = &n.entry.?;
        s.tree.clock += 1;
        e.used = s.tree.clock;
        e.hit = true;
        e.branched = e.branched or superseded(n, e.born); // a turn several continuations share (re-sent, edited, forked)
        s.tree.counts.hits += 1;
        const last = s.tree.gpa.dupe(u32, prompt) catch return;
        s.tree.gpa.free(e.last);
        e.last = last;
    }

    /// Room for every state the pass keeps, made before it: the snapshots and the new pages of its last mark.
    fn reserve(s: *Store, prompt: []const u32, from: ?*Node, marks_: []const u32) void {
        if (marks_.len == 0) return;
        const last = marks_[marks_.len - 1];
        while (true) {
            const new = last / s.rules.page - s.tree.have(prompt[0 .. last + s.rules.lookahead]);
            if (s.tree.held_pages + new <= s.limits.pages and s.tree.snaps + marks_.len <= s.limits.snaps) return;
            s.tree.evict(s.victim(prompt, from) orelse return);
        }
    }

    /// A kept state a peer could not resume goes, so later prompts do not ask for it again.
    pub fn forget(s: *Store, prompt: []const u32, at: u32) void {
        const n = s.find(prompt[0..@min(prompt.len, at + s.past())]) orelse return;
        if (n.end != at) return;
        s.tree.counts.failed += 1;
        s.tree.evict(n);
    }

    fn sharedKey(tokens: []const u32) u64 {
        return std.hash.Wyhash.hash(0, std.mem.sliceAsBytes(tokens));
    }

    fn noteShared(s: *Store, tokens: []const u32) void {
        const k = sharedKey(tokens);
        if (std.mem.indexOfScalar(u64, s.shared_keys.items, k) != null) return;
        if (s.shared_keys.items.len == SHARED_KEYS) _ = s.shared_keys.orderedRemove(0);
        s.shared_keys.append(s.tree.gpa, k) catch {};
    }

    /// The marks whose state the limits can hold (a new slice in `a`); the rest are refused now.
    fn fitting(s: *Store, a: Allocator, marks_: []const u32) ![]const u32 {
        defer a.free(marks_);
        var out: std.ArrayList(u32) = .empty;
        errdefer out.deinit(a);
        for (marks_) |m| {
            if (m / s.rules.page <= s.limits.pages and s.limits.snaps > 0) try out.append(a, m) else {
                s.tree.counts.refused += 1;
                note("kept nothing at {d} tokens: {d} pages pass the limit of {d}", .{ m, m / s.rules.page, s.limits.pages });
            }
        }
        return out.toOwnedSlice(a);
    }

    /// Where a pass from `from` keeps states: the history, then min_gap apart the stable prefix, blocks, last page.
    pub fn marks(s: *const Store, a: Allocator, prompt: []const u32, from: u32, history_len: u32, shared: []const u32, previous: []const u32) ![]const u32 {
        if (prompt.len < s.rules.min_prompt or prompt.len < 2) return a.alloc(u32, 0);
        var out: std.ArrayList(u32) = .empty;
        errdefer out.deinit(a);
        var want: std.ArrayList(u32) = .empty;
        defer want.deinit(a);
        try want.append(a, history_len);
        if (previous.len > 0) {
            const stable: u32 = @intCast(std.mem.indexOfDiff(u32, previous, prompt) orelse @min(previous.len, prompt.len));
            if (stable > 0 and stable < history_len and stable >= history_len / 2) try want.append(a, stable);
        }
        try want.appendSlice(a, shared);
        if (s.rules.tail and prompt.len > s.past()) try want.append(a, @intCast(prompt.len - s.past()));
        for (want.items, 0..) |w, k| {
            const m = w - w % s.step();
            if (m <= from or m >= prompt.len or m + s.rules.lookahead > prompt.len) continue;
            if (k > 0) { // the history's mark always; the others only away from it, each other and the resume point
                if (m - from < s.rules.min_gap) continue;
                const near = for (out.items) |o| {
                    if (@max(o, m) - @min(o, m) < s.rules.min_gap) break true;
                } else false;
                if (near) continue;
            }
            if (std.mem.indexOfScalar(u32, out.items, m) == null) try out.append(a, m);
        }
        std.mem.sort(u32, out.items, {}, std.sort.asc(u32));
        return out.toOwnedSlice(a);
    }

    /// At `at` the tree takes the pages of `mine` it lacks and the snapshot, evicting to fit; `path` gets its pages.
    pub fn keep(s: *Store, prompt: []const u32, at: u32, owner: ?*anyopaque, mine: []const u32, path: []u32) Kept {
        const t = &s.tree;
        if (!s.on() or at == 0 or at % s.step() != 0 or at + s.rules.lookahead > prompt.len or mine.len != at / s.rules.page or path.len != mine.len) return .{ .held = false, .shared = 0 };
        t.clock += 1;
        const tokens = prompt[0 .. at + s.rules.lookahead]; // a page past which the lookahead differs is another page
        const flagged = std.mem.indexOfScalar(u64, s.shared_keys.items, sharedKey(tokens)) != null;
        while (true) {
            var last: ?*Node = null;
            const got = t.heldPrefix(tokens, mine.len, path, &last);
            if (got == mine.len and last != null and last.?.end == at and last.?.entry != null) {
                const e = &last.?.entry.?;
                e.used = t.clock; // the same state again: no copy
                e.shared = e.shared or flagged;
                t.touch(last.?);
                return .{ .held = true, .shared = mine.len };
            }
            if (t.held_pages + (mine.len - got) <= s.limits.pages and t.snaps + 1 <= s.limits.snaps) break;
            const v = s.victim(prompt, null) orelse {
                t.counts.refused += 1;
                note("kept nothing at {d} tokens: no room", .{at});
                return .{ .held = false, .shared = got };
            };
            t.evict(v);
        }
        const node = t.insert(tokens, mine, path) catch |err| return s.fail(at, err, 0);
        const family = t.family;
        const saved = family.vtable.save(family.ptr, owner, at) catch |err| return s.fail(at, err, mine.len);
        const last = t.gpa.dupe(u32, prompt) catch {
            family.vtable.drop(family.ptr, saved);
            return s.fail(at, error.OutOfMemory, mine.len);
        };
        node.entry = .{ .saved = saved, .bytes = family.vtable.bytes(family.ptr, at), .born = @intCast(prompt.len), .used = t.clock, .last = last, .shared = flagged };
        t.snaps += 1;
        t.counts.kept += 1;
        if (s.disk) |r| s.writeBehind(r, node, saved);
        return .{ .held = true, .shared = mine.len };
    }

    /// The state just kept written behind (prompt_cache.Store's `queued`): its pages held once more while the save
    /// reads them, so the tree may drop the leaf but the pool cannot reuse the pages under the writer.
    fn writeBehind(s: *Store, r: *rows.Rows, node: *Node, saved: pc.Saved) void {
        const t = &s.tree;
        const e = &node.entry.?;
        const pages = t.gpa.dupe(u32, node.pages) catch return;
        errdefer t.gpa.free(pages);
        const tokens = t.gpa.dupe(u32, e.last[0..@min(e.last.len, node.end + t.ahead)]) catch return;
        errdefer t.gpa.free(tokens);
        for (pages) |id| t.pages.vtable.retain(t.pages.ptr, id);
        s.saves.append(t.gpa, .{ .node = node, .pages = pages, .tokens = tokens }) catch {
            for (pages) |id| t.pages.vtable.release(t.pages.ptr, id);
            return;
        };
        rows.write.queued(r, .{
            .owner = node,
            .tokens = tokens,
            .at = node.end,
            .saved = saved,
            .shared = e.shared,
            .pages = pages,
            .freed = .{ .ctx = s, .call = gaveBack },
        }, &.{});
    }

    /// The writer no longer reads a save's state (staged, ended or dropped): its pages go back, its tokens freed.
    fn gaveBack(ctx: *anyopaque, owner: *anyopaque) void {
        const s: *Store = @ptrCast(@alignCast(ctx));
        const t = &s.tree;
        const n: *Node = @ptrCast(@alignCast(owner));
        for (s.saves.items, 0..) |save, i| if (save.node == n) {
            for (save.pages) |id| t.pages.vtable.release(t.pages.ptr, id);
            t.gpa.free(save.pages);
            t.gpa.free(save.tokens);
            _ = s.saves.swapRemove(i);
            return;
        };
    }

    /// A leaf evicted while its save is still queued: the save is written now (its pages are held; rows_write).
    fn evicting(ctx: *anyopaque, n: *Node) void {
        const s: *Store = @ptrCast(@alignCast(ctx));
        const r = s.disk orelse return;
        for (s.saves.items) |save| if (save.node == n) return rows.write.evicting(r, n);
    }

    /// The longest state on disk below `prompt` that reaches past the tree's deepest match, when reading it pays:
    /// read into pool pages the store holds and a slot, and keep it as the tree's (its pages adopted like any kept
    /// state's; the store's holds go once the tree holds them). `found` (the tree's deepest match) otherwise.
    fn recall(s: *Store, r: *rows.Rows, prompt: []const u32, found: ?*Node) ?*Node {
        const t = &s.tree;
        const from: u32 = if (found) |n| n.end else 0;
        const i = r.best(prompt, &.{}, &.{}, from) orelse return found;
        const st = &r.states.items[i];
        if (st.at <= from) return found;
        const pay = select.savedSeconds(r, i, from) orelse return found; // the read beats prefilling what the tree lacks
        if (!(pay > 0)) return found;
        const np = st.at / s.rules.page;
        while (t.held_pages + np > s.limits.pages or t.snaps + 1 > s.limits.snaps) {
            t.evict(s.victim(prompt, found) orelse return found);
        }
        var pages: std.ArrayList(u32) = .empty;
        const got = rows.read.readPages(r, i, &pages) catch {
            pages.deinit(t.gpa);
            return found;
        };
        const give = struct {
            fn go(t2: *Tree, ps: *std.ArrayList(u32)) void {
                for (ps.items) |id| t2.pages.vtable.release(t2.pages.ptr, id);
                ps.deinit(t2.gpa);
            }
        }.go;
        const drop = struct {
            fn go(f2: pc.Snapshots, saved2: pc.Saved) void {
                f2.vtable.drop(f2.ptr, saved2);
            }
        }.go;
        const path = t.gpa.alloc(u32, np) catch {
            give(t, &pages);
            drop(t.family, got);
            return found;
        };
        defer t.gpa.free(path);
        const node = t.insert(prompt[0..@min(prompt.len, st.at + s.rules.lookahead)], pages.items, path) catch {
            give(t, &pages);
            drop(t.family, got);
            return found;
        };
        if (node.entry != null) { // the tree keeps this state's tokens already: its own copy wins
            give(t, &pages);
            drop(t.family, got);
            return found;
        }
        const last = t.gpa.dupe(u32, prompt) catch {
            give(t, &pages);
            drop(t.family, got);
            return found;
        };
        t.clock += 1;
        node.entry = .{ .saved = got, .bytes = t.family.vtable.bytes(t.family.ptr, st.at), .born = @intCast(prompt.len), .used = t.clock, .last = last, .shared = st.shared };
        t.snaps += 1;
        t.counts.kept += 1;
        give(t, &pages); // the tree holds the pages now; the recall's holds go
        return node;
    }

    fn fail(s: *Store, at: u32, err: anyerror, shared: usize) Kept {
        s.tree.counts.failed += 1;
        note("keeping {d} tokens failed ({s}); a later turn prefills them", .{ at, @errorName(err) });
        return .{ .held = false, .shared = shared };
    }

    /// Whether `a` is a less valuable entry than `b`: one never resumed before one that was, then the older.
    fn worse(a: *const Node, b: *const Node) bool {
        const x = a.entry.?;
        const y = b.entry.?;
        if (x.hit != y.hit) return !x.hit;
        return x.used < y.used;
    }

    /// Whether `prompt` repeats every token `n` holds: up to its end, and its lookahead.
    fn extends(n: *const Node, prompt: []const u32) bool {
        const parent = n.parent orelse return true;
        return prompt.len >= parent.end + n.tokens.len and std.mem.eql(u32, n.tokens, prompt[parent.end..][0..n.tokens.len]) and extends(parent, prompt);
    }

    /// Whether a later prompt kept a state below `n`.
    fn superseded(n: *const Node, born: u32) bool {
        for (n.kids.items) |k| {
            if ((k.entry != null and k.entry.?.born > born) or superseded(k, born)) return true;
        }
        return false;
    }

    fn pick(n: *Node, prompt: []const u32, skip: ?*Node, moved: *?*Node, oldest: *?*Node) void {
        if (n.entry) |e| if (n != skip) {
            // a state its conversation moved past goes first; a shared cut or a branch point only as the least valuable
            const gone = !e.shared and !e.branched and ((prompt.len > e.born and prompt.len > n.end and extends(n, prompt)) or superseded(n, e.born));
            if (gone and (moved.* == null or e.used < moved.*.?.entry.?.used)) moved.* = n;
            if (oldest.* == null or worse(n, oldest.*.?)) oldest.* = n;
        };
        for (n.kids.items) |k| pick(k, prompt, skip, moved, oldest);
    }

    /// The entry to free first, never `skip`: one a later prompt extends (no branch point), oldest first; else worst.
    fn victim(s: *const Store, prompt: []const u32, skip: ?*Node) ?*Node {
        var moved: ?*Node = null;
        var oldest: ?*Node = null;
        pick(s.tree.root, prompt, skip, &moved, &oldest);
        return moved orelse oldest;
    }

    /// Evicts until `want` pages are free where `in`'s leaves lie (null: the pages' available) and not promised;
    /// false when the tree cannot give that many.
    pub fn reclaim(s: *Store, want: usize, in: ?tree_mod.LeafFilter) bool {
        while (freeIn(s.tree.pages, want, in)) if (!s.tree.evictLeaf(in)) return false;
        return true;
    }

    /// Whether fewer than `want` pages are free where `in`'s leaves lie (null: the pages' available).
    fn freeIn(pages: tree_mod.Pages, want: usize, in: ?tree_mod.LeafFilter) bool {
        const n = if (in) |x| (if (x.free) |f| f(x.ctx) else pages.vtable.available(pages.ptr)) else pages.vtable.available(pages.ptr);
        return n < want;
    }

    /// One log line after a prompt pass: where it resumed, how many states it kept, and what the tree holds.
    pub fn report(s: *const Store, prompt: usize, from: u32, kept: u64) void {
        if (@import("builtin").is_test) return;
        const t = &s.tree;
        std.log.info("prompt cache: {d} tokens, resumed at {d}, kept {d}; {d} states and {d} pages ({d} MiB) of {d} states and {d} pages (hits {d}, misses {d}, evicted {d}, refused {d}, failed {d})", .{ prompt, from, kept, t.snaps, t.held_pages, t.held() >> 20, s.limits.snaps, s.limits.pages, t.counts.hits, t.counts.misses, t.counts.evicted, t.counts.refused, t.counts.failed });
    }
};

/// A line on the engine's log; tests stay quiet.
fn note(comptime fmt: []const u8, args: anytype) void {
    if (@import("builtin").is_test) return;
    std.log.warn("prompt cache: " ++ fmt, args);
}

test {
    _ = @import("prompt_radix_test.zig");
    _ = @import("prompt_radix_disk_test.zig");
}
