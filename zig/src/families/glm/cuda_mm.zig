//! GLM's bf16 linears on tensor cores (glm_mm): weights tiled once at load, any row count, row-stable K slices.
const std = @import("std");
const cuda = @import("cuda");

/// A linear's tiles: bf16 pairs (glm_mm_tile16), or `bits`-bit codes (glm_mm_tile4/6/8) with fp16 scales at `s`.
pub const Tiled = struct { w: u64, n: usize, k: usize, s: u64 = 0, bits: u8 = 16 };

/// The widths glm_mm packs to, by kernel set: bf16 first.
const widths = [_]u8{ 16, 4, 6, 8 };

fn widthIndex(bits: u8) !usize {
    return std.mem.indexOfScalar(u8, &widths, bits) orelse error.BadBits;
}

/// Bytes a packed linear's codes take (its scales follow, 256-aligned): what fits in place of its bf16 tiles.
pub fn packedBytes(n: usize, k: usize, bits: u8) [2]usize {
    return .{ std.mem.alignForward(usize, n * k * bits / 8, 256), n * (k / 64) * 2 };
}

/// The linears packed at load: each one's tiles pointer (sorted) and its bits; a linear not here is bf16.
pub const Packed = struct {
    ptrs: []const u64 = &.{},
    bits: []const u8 = &.{},

    /// The tiled weight `w` [n][k] as the matmul reads it: packed codes with their scales after them, or bf16.
    pub fn tiled(p: *const Packed, w: u64, n: usize, k: usize) Tiled {
        var lo: usize = 0;
        var hi: usize = p.ptrs.len;
        while (lo < hi) {
            const mid = (lo + hi) / 2;
            if (p.ptrs[mid] < w) lo = mid + 1 else hi = mid;
        }
        if (lo == p.ptrs.len or p.ptrs[lo] != w) return .{ .w = w, .n = n, .k = k };
        const b = p.bits[lo];
        return .{ .w = w, .n = n, .k = k, .s = w + packedBytes(n, k, b)[0], .bits = b };
    }
};

/// glm_mm's bf16 kernels by row bucket, the K-slice reduce and the tiling.
pub const Kernels = struct {
    by_rows: [4]cuda.Function, // bf16: 16, 32, 64, 128 rows a block
    packed_rows: [3][4]cuda.Function, // 4, 6 and 8 bits, as by_rows
    pair: [4]cuda.Function, // glm_mm{16,4,6,8}_16x2: two problems of one shape a launch, the 16-row blocks' sums
    packed_prompt: [3]cuda.Function, // glm_mm4/6/8_128s2: packed prompts in two stages, two blocks an SM, the same sums
    prompt: cuda.Function, // 128-row blocks, row blocks fastest: a prompt's blocks share each weight tile in L2
    reduce: cuda.Function,
    tile: cuda.Function,
    tile_packed: [3]cuda.Function, // glm_mm_tile4/6/8: packed rows and their scales into the tiles
    prompt_fixed_k: bool = false, // opt-in prompt experiment; never selected from the number of batched rows
    pdl: bool, // dependent launches: a matmul's weights stream in while the kernel before it finishes
    q8: cuda.Function, // glm_q8_rows: a prompt's bf16 rows to int8 with a scale per row and 64- or 32-group (TF_GLM_W8A8)
    mm8a8: [2][2]cuda.Function, // [64-, 32-groups][glm_mm8a8_64s3r (to 1024 rows), _128s3r]: int8 rows by the 8-bit codes
    a8: ?A8 = null, // TF_GLM_W8A8=1: the int8 rows' scratch; null: packed prompts stay on bf16 rows (the default)

    pub fn load(m: cuda.Module) !Kernels {
        var k: Kernels = .{ .by_rows = undefined, .packed_rows = undefined, .packed_prompt = undefined, .pair = undefined, .tile_packed = undefined, .prompt = try m.function("glm_mm16_rows"), .reduce = try m.function("glm_mm_reduce"), .tile = try m.function("glm_mm_tile16"), .pdl = true, .q8 = try m.function("glm_q8_rows"), .mm8a8 = .{
            .{ try m.function("glm_mm8a8_64s3r"), try m.function("glm_mm8a8_128s3r") },
            .{ try m.function("glm_mm8a8_64s3r_g32"), try m.function("glm_mm8a8_128s3r_g32") },
        } };
        for (k.mm8a8, [2]usize{ 64, 32 }) |fs, gs| for (fs, a8_bn) |f, bn| try f.allowDynamicShared(a8Shared(bn, gs));
        inline for (.{ "glm_mm16_16", "glm_mm16_32", "glm_mm16_64", "glm_mm16_128" }, 0..) |name, i| {
            k.by_rows[i] = try m.function(name);
            try k.by_rows[i].allowDynamicShared(smem(bucket(i)));
        }
        inline for (widths[1..], 0..) |b, wi| {
            inline for (.{ 16, 32, 64, 128 }, 0..) |bm, i| {
                k.packed_rows[wi][i] = try m.function(std.fmt.comptimePrint("glm_mm{d}_{d}", .{ b, bm }));
                try k.packed_rows[wi][i].allowDynamicShared(sharedBytes(b, bm));
            }
            k.tile_packed[wi] = try m.function(std.fmt.comptimePrint("glm_mm_tile{d}", .{b}));
            k.packed_prompt[wi] = try m.function(std.fmt.comptimePrint("glm_mm{d}_128s2", .{b}));
            try k.packed_prompt[wi].allowDynamicShared(sharedBytesStages(b, 128, 2));
        }
        inline for (widths, 0..) |b, wi| {
            k.pair[wi] = try m.function(std.fmt.comptimePrint("glm_mm{d}_16x2", .{b}));
            try k.pair[wi].allowDynamicShared(sharedBytes(b, 16));
        }
        try k.prompt.allowDynamicShared(smem(128));
        if (std.c.getenv("TF_GLM_PDL")) |v| k.pdl = !std.mem.eql(u8, std.mem.span(v), "0");
        return k;
    }
};

/// TF_GLM_W8A8's scratch for one prompt pass: int8 rows [rows][k] and their fp32 scales [k / gs][rows to 4] (+ 128);
/// `gs` the activations' group: 64 (TF_GLM_W8A8=1) or 32 (=32).
pub const A8 = struct {
    q: u64,
    xs: u64,
    rows: usize,
    k: usize,
    gs: usize = 64,
    skip_k: [4]usize = @splat(0), // TF_GLM_W8A8_SKIP_K (diagnostic): calls of these K widths keep bf16 rows

    fn takes(a: A8, rows: usize, k: usize) bool {
        if (rows > a.rows or k > a.k) return false;
        for (a.skip_k) |x| if (x == k) return false;
        return true;
    }

    pub fn bytes(rows: usize, k: usize, gs: usize) [2]usize {
        return .{ std.mem.alignForward(usize, rows * k, 256), ((k / gs) * std.mem.alignForward(usize, rows, 4) + 128) * 4 };
    }
};

const a8_bn = [2]usize{ 64, 128 };

/// glm_mm8a8's three stages: 128 int8 rows of 64 and their scales (64 / gs a row), `bn` columns of codes and theirs.
fn a8Shared(bn: usize, gs: usize) u32 {
    return @intCast(3 * (128 * 64 + 128 * 4 * (64 / gs) + bn / 8 * 512 + bn / 8 * 16));
}

/// A packed 8-bit prompt call on int8 rows (TF_GLM_W8A8): the rows quantized into the scratch, then the IMMA kernel.
fn matmulA8(k: *const Kernels, a: A8, s: cuda.Stream, x: u64, ldx: usize, w: Tiled, out: u64, m: usize, wide: bool) !void {
    const mp = std.mem.alignForward(usize, m, 4);
    const groups = m * (w.k / 64);
    var q: cuda.Args = .{};
    q.add(x);
    q.add(@as(c_longlong, @intCast(ldx)));
    q.add(a.q);
    q.add(a.xs);
    q.add(@as(c_int, @intCast(mp)));
    q.add(@as(c_int, @intCast(m)));
    q.add(@as(c_int, @intCast(w.k)));
    q.add(@as(c_int, @intCast(a.gs)));
    try cuda.launch.launch(k.q8, .{ .grid = .{ .x = @intCast((groups + 31) / 32) }, .block = .{ .x = 256 } }, s, &q);
    const wide_tile = m > 1024; // 128 columns a block for a whole pass, 64 for pieces (the faster of each, measured)
    const bn = a8_bn[@intFromBool(wide_tile)];
    var g: cuda.Args = .{};
    g.add(a.q);
    g.add(a.xs);
    g.add(@as(c_int, @intCast(mp)));
    g.add(w.w);
    g.add(w.s);
    g.add(out);
    g.add(@as(c_int, @intFromBool(wide)));
    g.add(@as(c_int, @intCast(m)));
    g.add(@as(c_int, @intCast(w.n)));
    g.add(@as(c_int, @intCast(w.k)));
    // row blocks fastest
    try cuda.launch.launch(k.mm8a8[@intFromBool(a.gs == 32)][@intFromBool(wide_tile)], .{ .grid = .{ .x = @intCast((m + 127) / 128), .y = @intCast((w.n + bn - 1) / bn) }, .block = .{ .x = 256 }, .shared = a8Shared(bn, a.gs), .pdl = k.pdl }, s, &g);
}

const split_target = 192; // column tiles times K slices a launch aims at (the GB10's SMs, a few blocks each)

fn bucket(i: usize) usize {
    return @as(usize, 16) << @intCast(i);
}

fn smem(bm: usize) u32 {
    return sharedBytes(16, bm);
}

/// A block's shared memory: three stages of `bm` rows of x and a 64-column tile of weights (`bits / 2` words a lane).
pub fn sharedBytes(bits: u8, bm: usize) u32 {
    return sharedBytesStages(bits, bm, 3);
}

pub fn sharedBytesStages(bits: u8, bm: usize, stages: usize) u32 {
    const wpc: usize = if (bits == 16) 8 else bits / 2;
    return @intCast(stages * (bm * 128 + 8 * 32 * wpc * 4 + 128));
}

/// K slices fixed by the shape: column tiles times slices reach the target, each slice at least 4 groups of 64.
pub fn splitK(n: usize, k: usize) usize {
    const tiles = (n + 63) / 64;
    const steps = k / 64;
    var sk: usize = 1;
    while (sk < 8 and tiles * sk < split_target and steps % (sk * 2) == 0 and steps / (sk * 2) >= 4) sk *= 2;
    return sk;
}

/// Bytes of fp32 partials a call of `m` rows needs (0: one slice; windows past 64 rows go 64 at a time).
pub fn partBytes(m: usize, w: Tiled) usize {
    const sk = splitK(w.n, w.k);
    return if (sk > 1) sk * @min(m, 64) * w.n * 4 else 0;
}

/// x [m][k] (`ldx` apart) by `w` -> out [m][n] (`wide`: fp32); `one`: prompt blocks at any rows, sums per row alike.
pub fn matmul(k: *const Kernels, s: cuda.Stream, x: u64, ldx: usize, w: Tiled, out: u64, part: u64, m: usize, wide: bool, one: bool) !void {
    if (w.k % 64 != 0 or w.n % 8 != 0 or x % 16 != 0 or ldx % 8 != 0) return error.BadShape;
    var done: usize = 0;
    const wi = try widthIndex(w.bits);
    while (done < m) { // windows to 64 rows a launch with K slices; prompts in one launch of 128-row blocks, one slice
        const big = if (one) !k.prompt_fixed_k else m - done >= 128;
        const rows = if (big) m - done else @min(m - done, 64);
        if (big and one and w.bits == 8) if (k.a8) |a| if (a.takes(rows, w.k)) {
            try matmulA8(k, a, s, x + done * ldx * 2, ldx, w, out + done * w.n * @as(usize, if (wide) 4 else 2), rows, wide);
            done += rows;
            continue;
        };
        const i: usize = if (big) 3 else if (rows <= 16) 0 else if (rows <= 32) 1 else 2;
        const bm = bucket(i);
        const sk = if (big) 1 else splitK(w.n, w.k); // a prompt's blocks share each weight tile in L2: no slices
        const o = out + done * w.n * @as(usize, if (wide) 4 else 2);
        var a: cuda.Args = .{};
        a.add(x + done * ldx * 2);
        a.add(@as(c_longlong, @intCast(ldx)));
        a.add(w.w);
        a.add(w.s);
        a.add(o);
        a.add(part);
        a.add(@as(c_int, @intFromBool(wide)));
        a.add(@as(c_int, @intCast(rows)));
        a.add(@as(c_int, @intCast(w.n)));
        a.add(@as(c_int, @intCast(w.k)));
        a.add(@as(c_int, @intCast(sk)));
        const cols: u32 = @intCast((w.n + 63) / 64);
        const row_blocks: u32 = @intCast((rows + bm - 1) / bm);
        if (big and wi == 0) {
            try cuda.launch.launch(k.prompt, .{ .grid = .{ .x = row_blocks, .y = cols }, .block = .{ .x = 256 }, .shared = smem(128), .pdl = k.pdl }, s, &a);
        } else if (big) { // packed prompts: the 128-row blocks in two stages, two an SM (the 3-stage kernel's sums)
            try cuda.launch.launch(k.packed_prompt[wi - 1], .{ .grid = .{ .x = cols, .y = row_blocks, .z = @intCast(sk) }, .block = .{ .x = 256 }, .shared = sharedBytesStages(w.bits, bm, 2), .pdl = k.pdl }, s, &a);
        } else {
            const f = if (wi == 0) k.by_rows[i] else k.packed_rows[wi - 1][i];
            try cuda.launch.launch(f, .{ .grid = .{ .x = cols, .y = row_blocks, .z = @intCast(sk) }, .block = .{ .x = if (bm <= 32) 128 else 256 }, .shared = sharedBytes(w.bits, bm), .pdl = k.pdl }, s, &a);
        }
        if (sk > 1) {
            const total = rows * w.n;
            var r: cuda.Args = .{};
            r.add(part);
            r.add(o);
            r.add(@as(c_longlong, @intCast(total)));
            r.add(@as(c_int, @intCast(sk)));
            r.add(@as(c_int, @intFromBool(wide)));
            try cuda.launch.launch(k.reduce, .{ .grid = .{ .x = @intCast((total + 255) / 256) }, .block = .{ .x = 256 } }, s, &r);
        }
        done += rows;
    }
}

/// Two decode linears of one shape and width in one launch (z: the problem), when each would take one K slice and the
/// 16-row blocks: the same blocks and sums as two `matmul` calls; false (nothing launched) when they would not.
pub fn matmulPair(k: *const Kernels, s: cuda.Stream, x0: u64, x1: u64, ldx: usize, w0: Tiled, w1: Tiled, out0: u64, out1: u64, m: usize) !bool {
    if (m == 0 or m > 16 or w0.n != w1.n or w0.k != w1.k or w0.bits != w1.bits) return false;
    if (w0.k % 64 != 0 or w0.n % 8 != 0 or x0 % 16 != 0 or x1 % 16 != 0 or ldx % 8 != 0) return false;
    if (splitK(w0.n, w0.k) != 1) return false;
    const wi = widthIndex(w0.bits) catch return false;
    var a: cuda.Args = .{};
    a.add(x0);
    a.add(x1);
    a.add(@as(c_longlong, @intCast(ldx)));
    a.add(w0.w);
    a.add(w1.w);
    a.add(w0.s);
    a.add(w1.s);
    a.add(out0);
    a.add(out1);
    a.add(@as(c_int, 0)); // bf16 out, as matmul's `wide` false
    a.add(@as(c_int, @intCast(m)));
    a.add(@as(c_int, @intCast(w0.n)));
    a.add(@as(c_int, @intCast(w0.k)));
    const cols: u32 = @intCast((w0.n + 63) / 64);
    try cuda.launch.launch(k.pair[wi], .{ .grid = .{ .x = cols, .y = 1, .z = 2 }, .block = .{ .x = 128 }, .shared = sharedBytes(w0.bits, 16), .pdl = k.pdl }, s, &a);
    return true;
}

/// Packed rows (cuda_qpack's) and fp16 scales [n][k / 64] at `src`, `ss` tiled into `dst`, `ds` (`bits` 4/6/8).
pub fn tilePacked(k: *const Kernels, s: cuda.Stream, bits: u8, src: u64, ss: u64, dst: u64, ds: u64, n: usize, kk: usize) !void {
    const wi = try widthIndex(bits);
    if (wi == 0) return error.BadBits;
    const chunks = n / 8 * (kk / 64) * 32;
    var a: cuda.Args = .{};
    a.add(src);
    a.add(ss);
    a.add(dst);
    a.add(ds);
    a.add(@as(c_int, @intCast(n)));
    a.add(@as(c_int, @intCast(kk)));
    try cuda.launch.launch(k.tile_packed[wi - 1], .{ .grid = .{ .x = @intCast((chunks + 255) / 256) }, .block = .{ .x = 256 } }, s, &a);
}

/// A row-major bf16 [n][k] linear at `src` tiled into `dst` (as many bytes).
pub fn tile(k: *const Kernels, s: cuda.Stream, src: u64, dst: u64, n: usize, kk: usize) !void {
    const chunks = n / 8 * (kk / 64) * 32;
    var a: cuda.Args = .{};
    a.add(src);
    a.add(dst);
    a.add(@as(c_int, @intCast(n)));
    a.add(@as(c_int, @intCast(kk)));
    try cuda.launch.launch(k.tile, .{ .grid = .{ .x = @intCast((chunks + 255) / 256) }, .block = .{ .x = 256 } }, s, &a);
}

test "K slices as the shapes take them" {
    try std.testing.expectEqual(@as(usize, 1), splitK(12288, 4096)); // 192 column tiles already
    try std.testing.expectEqual(@as(usize, 4), splitK(4096, 2048)); // 64 tiles, 32 groups: 4 slices of 8
}
