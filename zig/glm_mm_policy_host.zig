const std = @import("std");
const cuda = @import("cuda");
const mm = @import("mm");
const kernels: mm.Kernels = .{
    .by_rows = .{ .{ .id = 16 }, .{ .id = 32 }, .{ .id = 64 }, .{ .id = 128 } },
    .packed_rows = @splat(.{ .{ .id = 16 }, .{ .id = 32 }, .{ .id = 64 }, .{ .id = 128 } }),
    .prompt = .{ .id = 512 },
    .reduce = .{ .id = 1024 },
    .tile = .{},
    .tile_packed = @splat(.{}),
    .pair = @splat(.{}), // decode windows' two-problem launches (not this test's)
    .packed_prompt = @splat(.{ .id = 256 }),
    .pdl = false,
    .q8 = .{ .id = 2048 },
    .mm8a8 = .{ .{ .{ .id = 4096 }, .{ .id = 8192 } }, .{ .{ .id = 4097 }, .{ .id = 8193 } } },
};
test "verify and draft windows keep shape-selected split K at every width" {
    for ([_]u8{ 16, 4, 6, 8 }) |bits| for ([_]usize{ 1, 2, 3, 4, 8, 16, 17, 31, 32 }) |rows| {
        cuda.count = 0;
        try mm.matmul(&kernels, .{}, 4096, 2048, .{ .w = 8192, .s = 12288, .n = 4096, .k = 2048, .bits = bits }, 16384, 32768, rows, false, false);
        try std.testing.expectEqual(@as(usize, 2), cuda.count);
        try std.testing.expectEqual(@as(u32, 4), cuda.configs[0].grid.z);
        try std.testing.expectEqual(@as(u32, 1024), cuda.symbols[1]);
    };
}
test "prompt flag chooses one K slice even for one row, alone or many" {
    for ([_]u8{ 16, 4, 6, 8 }) |bits| for ([_]usize{ 1, 3, 16, 64, 127, 128, 129, 2048 }) |rows| {
        cuda.count = 0;
        try mm.matmul(&kernels, .{}, 4096, 2048, .{ .w = 8192, .s = 12288, .n = 4096, .k = 2048, .bits = bits }, 16384, 32768, rows, false, true);
        try std.testing.expectEqual(@as(usize, 1), cuda.count);
        try std.testing.expectEqual(@as(u32, 1), cuda.configs[0].grid.z);
        try std.testing.expectEqual(@as(u32, 256), cuda.configs[0].block.x);
    };
}

test "fixed K prompt experiment uses the same shape split across bucket, offset and tail boundaries" {
    var k = kernels;
    k.prompt_fixed_k = true;
    for ([_]u8{ 16, 4, 6, 8 }) |bits| for ([_]usize{ 1, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 2048 }) |rows| {
        cuda.count = 0;
        try mm.matmul(&k, .{}, 4096 + 7 * 2048 * 2, 2048, .{ .w = 8192, .s = 12288, .n = 4096, .k = 2048, .bits = bits }, 16384, 32768, rows, false, true);
        try std.testing.expectEqual(@as(usize, 2 * ((rows + 63) / 64)), cuda.count);
        var i: usize = 0;
        while (i < cuda.count) : (i += 2) {
            try std.testing.expectEqual(@as(u32, 4), cuda.configs[i].grid.z);
            try std.testing.expectEqual(@as(u32, 1024), cuda.symbols[i + 1]);
        }
    };
    try std.testing.expect(!kernels.prompt_fixed_k);
}

test "TF_GLM_W8A8: only packed 8-bit prompt calls inside the scratch quantize their rows, then one IMMA launch" {
    var k = kernels;
    k.a8 = .{ .q = 1 << 20, .xs = 1 << 24, .rows = 2048, .k = 4096 };
    for ([_]u8{ 16, 4, 6, 8 }) |bits| for ([_]usize{ 1, 127, 128, 768, 1024, 1025, 2048 }) |rows| for ([_]bool{ false, true }) |prompt| {
        cuda.count = 0;
        try mm.matmul(&k, .{}, 4096, 2048, .{ .w = 8192, .s = 12288, .n = 4096, .k = 2048, .bits = bits }, 16384, 32768, rows, false, prompt);
        if (bits == 8 and prompt) {
            try std.testing.expectEqual(@as(usize, 2), cuda.count);
            try std.testing.expectEqual(@as(u32, 2048), cuda.symbols[0]);
            try std.testing.expectEqual(@as(u32, if (rows > 1024) 8192 else 4096), cuda.symbols[1]);
            try std.testing.expectEqual(@as(u32, @intCast((rows + 127) / 128)), cuda.configs[1].grid.x); // row blocks fastest
        } else for (cuda.symbols[0..cuda.count]) |sym| try std.testing.expect(sym != 2048 and sym != 4096 and sym != 8192);
    };
    // TF_GLM_W8A8=32: the 32-group kernels
    k.a8.?.gs = 32;
    cuda.count = 0;
    try mm.matmul(&k, .{}, 4096, 2048, .{ .w = 8192, .s = 12288, .n = 4096, .k = 2048, .bits = 8 }, 16384, 32768, 2048, false, true);
    try std.testing.expectEqual(@as(u32, 8193), cuda.symbols[1]);
    k.a8.?.gs = 64;
    // a K width the diagnostic skips: the bf16-row kernels
    k.a8.?.skip_k[0] = 2048;
    cuda.count = 0;
    try mm.matmul(&k, .{}, 4096, 2048, .{ .w = 8192, .s = 12288, .n = 4096, .k = 2048, .bits = 8 }, 16384, 32768, 2048, false, true);
    try std.testing.expectEqual(@as(usize, 1), cuda.count);
    k.a8.?.skip_k[0] = 0;
    // past the scratch (rows or K): the bf16-row kernels
    for ([_][2]usize{ .{ 4096, 2048 }, .{ 2048, 8192 } }) |rk| {
        cuda.count = 0;
        try mm.matmul(&k, .{}, 4096, rk[1], .{ .w = 8192, .s = 12288, .n = 4096, .k = rk[1], .bits = 8 }, 16384, 32768, rk[0], false, true);
        try std.testing.expectEqual(@as(usize, 1), cuda.count);
        try std.testing.expectEqual(@as(u32, 256), cuda.symbols[0]);
    }
}
