//! A backend's KV-ownership stats for /health and /metrics (family-neutral): a page pool cut into parts, one a rank
//! (GLM's TF_GLM_KV_SHARD, patch 0118 by Robert Wojciechowski), every part's pages by what holds them, and the sharded
//! attention's exchanges, twins and carries. A backend that shards hands the host a `Source`; the host reads it on its
//! engine thread (the backend's own) and publishes the copy for status readers.
const std = @import("std");

pub const max_parts = 2;

/// One part's pages now (gauges).
pub const Part = struct {
    free: u64 = 0, // pages nothing holds or promised
    live_pages: u64 = 0, // live streams' pages
    twin_pages: u64 = 0, // pages prompt passes borrowed from it (twins)
    live: u64 = 0, // live streams placed in it

    pub const help = .{
        .free = "Pages of this part nothing holds.",
        .live_pages = "Pages of this part live streams hold.",
        .twin_pages = "Pages of this part filling prompts borrowed (twins).",
        .live = "Live streams whose pages lie in this part.",
    };
};

pub const Stats = struct {
    parts: u64 = 0, // gauge: the parts (ranks)
    part_pages: u64 = 0, // gauge: pages a part
    page_tokens: u64 = 0, // gauge: positions a page
    owner_layers: u64 = 0,
    owner_rows: u64 = 0,
    owner_bytes: u64 = 0,
    owner_prompts: u64 = 0,
    twins_made: u64 = 0,
    twins_given: u64 = 0,
    twins_refused: u64 = 0,
    carries: u64 = 0,
    carry_bytes: u64 = 0,
    resumes_dropped: u64 = 0,
    checks: u64 = 0,
    part: [max_parts]Part = @splat(.{}),

    pub const gauges = [_][]const u8{ "parts", "part_pages", "page_tokens" };

    /// Each counter's /metrics help (a missing one fails the build).
    pub const help = .{
        .parts = "KV-ownership parts (one a rank).",
        .part_pages = "KV pages a part holds.",
        .page_tokens = "Positions a KV page holds.",
        .owner_layers = "Attention layers one rank ran for both ranks' heads (the other lacked the stream's rows).",
        .owner_rows = "Rows those layers exchanged queries and outputs for.",
        .owner_bytes = "Bytes rank 0 sent for owner-computed attention (queries and outputs).",
        .owner_prompts = "Prompt chunks that attended owner-computed (no twin).",
        .twins_made = "Prompt passes given a twin in the other part.",
        .twins_given = "Twins given back (a prompt pass ended).",
        .twins_refused = "Prompt passes the other part had no room to twin.",
        .carries = "Resumes whose kept latent rows went to the rank lacking them.",
        .carry_bytes = "Bytes those carries sent.",
        .resumes_dropped = "Resumes that started from 0: the stream could not go to its home part and carries are off.",
        .checks = "Layout digests rank 0 sent (the follower compared each; a difference ends the pair).",
    };
};

/// How the host reads a backend's stats (on its engine thread, the backend's own).
pub const Source = struct {
    ctx: *anyopaque,
    read: *const fn (ctx: *anyopaque, out: *Stats) void,
};

test "every kv_parts counter has its help" {
    inline for (@typeInfo(Stats).@"struct".field_names) |n| {
        if (comptime std.mem.eql(u8, n, "part")) continue;
        try std.testing.expect(@field(Stats.help, n).len > 0);
    }
    inline for (@typeInfo(Part).@"struct".field_names) |n| try std.testing.expect(@field(Part.help, n).len > 0);
}
