// 背景模糊（毛玻璃）。
//
// 四趟：
//   1. downsample —— 把从表面上拷出来的那一块按 `step` 缩成一张小纹理（4×4 tent，缩的同时
//      就把高频抹掉一半）；
//   2. blur_h、3. blur_v —— 在小纹理上做可分离高斯，两轴各一趟；
//   4. glass —— 主 pass 以 Load 重开之后，把模糊结果按**屏幕坐标**映射回小纹理采样，乘上
//      圆角覆盖画回表面。
//
// 玻璃**只画模糊，不画颜色**：半透明的 tint 仍然由压在它上面那张普通背景 quad 提供。所以
// 「糊多少」与「透多少」是两个独立的数，而这一层不必重复 quad 已经做过的渐变与描边。
//
// 三处采样一律写死 `textureSampleLevel(..., 0)`：模糊图只有一级 mip，而玻璃那一片是按像素
// 分界判覆盖的，非一致控制流里隐式求导拿到的 LOD 是未定义的。
//
// 与 Rust 侧的对应：`Backdrop` ↔ `GpuBackdrop`（32 字节头 + 64 字节一条记录），
// `Blur` ↔ `GpuBlur`（24 字节）。错位不会报错，只会让每条玻璃读到隔壁那格的去向。

// 一条玻璃。64 字节：bounds 16 + content_mask 16 + corner_radii 16 + blur_radius 4 + 12 补白。
// 它**不在** uniform 里 —— naga 不收 uniform 结构体尾部的运行时长度数组（报
// 「The struct member[N] is not statically sized」，整条 shader module 验证失败），而 GLES
// 又没有 storage buffer。所以它走逐实例的顶点缓冲，数量不受布局约束。
struct GlassRecord {
    @location(0) bounds: vec4<f32>, // origin.xy, size.zw —— 设备像素
    @location(1) content_mask: vec4<f32>,
    @location(2) corner_radii: vec4<f32>, // top_left, top_right, bottom_right, bottom_left
    @location(3) blur_radius: f32,
}

struct Backdrop {
    region_origin: vec2<f32>, // 拷出来那一块在窗口里的左上角，设备像素
    region_size: vec2<f32>,   // 那一块的尺寸，设备像素
    window_size: vec2<f32>,   // 整窗尺寸，设备像素：换算裁剪坐标要用
    flags: vec2<u32>,         // x = 表面是不是预乘 alpha
}

@group(0) @binding(0) var<uniform> backdrop: Backdrop;
@group(0) @binding(1) var backdrop_map: texture_2d<f32>;
@group(0) @binding(2) var backdrop_sampler: sampler;

struct Blur {
    src_origin: vec2<f32>, // 采源纹理哪一块的左上角，像素
    src_size: vec2<f32>,   // 源纹理整张的尺寸，像素（不是区域尺寸）
    dst_size: vec2<f32>,
    step: f32,             // 降采样那一趟：一个目标纹素跨多少源像素
    radius: f32,           // 高斯那一趟：半径，目标纹素
}

@group(0) @binding(0) var<uniform> blur: Blur;
@group(0) @binding(1) var src_map: texture_2d<f32>;
@group(0) @binding(2) var src_sampler: sampler;

// 把整帧贴回表面：只有源纹理与采样器，尺寸用 textureDimensions 现取，不必带 uniform。
@group(0) @binding(0) var frame_map: texture_2d<f32>;
@group(0) @binding(1) var frame_sampler: sampler;

// 铺满目标的一张四边形：4 个顶点走 TriangleStrip，与主着色器那套 vs_quad 同形。
@vertex
fn vs_fullscreen(@builtin(vertex_index) vid: u32) -> @builtin(position) vec4<f32> {
    let unit = vec2<f32>(f32(vid & 1u), 0.5 * f32(vid & 2u));
    return vec4<f32>(unit.x * 2.0 - 1.0, 1.0 - unit.y * 2.0, 0.0, 1.0);
}

@fragment
fn fs_downsample(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let center = blur.src_origin + pos.xy * blur.step;
    var sum = vec4<f32>(0.0);
    for (var i: i32 = 0; i < 4; i = i + 1) {
        for (var j: i32 = 0; j < 4; j = j + 1) {
            let off = (vec2<f32>(f32(i), f32(j)) - vec2<f32>(1.5)) * blur.step * 0.5;
            sum = sum + textureSampleLevel(src_map, src_sampler, (center + off) / blur.src_size, 0.0);
        }
    }
    return sum * 0.0625;
}

// 17 抽的高斯：σ = radius/2，所以 ±8 抽正好覆盖到 ±4σ，尾巴截断带来的振铃看不出来。
fn gaussian(pos: vec2<f32>, dir: vec2<f32>) -> vec4<f32> {
    let sigma = max(blur.radius * 0.5, 0.5);
    let two_sigma_sq = 2.0 * sigma * sigma;
    var sum = vec4<f32>(0.0);
    var total = 0.0;
    for (var i: i32 = -8; i <= 8; i = i + 1) {
        let d = f32(i);
        let w = exp(-(d * d) / two_sigma_sq);
        let uv = (blur.src_origin + pos.xy + dir * d) / blur.src_size;
        sum = sum + textureSampleLevel(src_map, src_sampler, uv, 0.0) * w;
        total = total + w;
    }
    return sum / total;
}

@fragment
fn fs_blur_h(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    return gaussian(pos.xy, vec2<f32>(1.0, 0.0));
}

@fragment
fn fs_blur_v(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    return gaussian(pos.xy, vec2<f32>(0.0, 1.0));
}

// 把整帧从自有的中间纹理贴回表面。尺寸用 textureDimensions 现取，所以这趟不必带 uniform。
@fragment
fn fs_present(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let size = vec2<f32>(textureDimensions(frame_map));
    return textureSampleLevel(frame_map, frame_sampler, pos.xy / size, 0.0);
}

struct GlassVarying {
    @builtin(position) position: vec4<f32>,
    @location(0) bounds: vec4<f32>, // origin.xy, size.zw
    @location(1) radii: vec4<f32>,
    @location(2) clip: vec4<f32>,   // content_mask 的 origin.xy 与 size.zw
}

@vertex
fn vs_glass(
    @builtin(vertex_index) vid: u32,
    glass: GlassRecord,
) -> GlassVarying {
    let unit = vec2<f32>(f32(vid & 1u), 0.5 * f32(vid & 2u));
    let bounds = vec4<f32>(glass.bounds.xy, glass.bounds.zw);
    let device_pos = unit * bounds.zw + bounds.xy;
    let clipped = device_pos / backdrop.window_size * vec2<f32>(2.0, -2.0)
        + vec2<f32>(-1.0, 1.0);

    var out: GlassVarying;
    out.position = vec4<f32>(clipped.x, clipped.y, 0.0, 1.0);
    out.bounds = bounds;
    out.radii = glass.corner_radii;
    out.clip = glass.content_mask;
    return out;
}

// 离这一点最近那个角的半径。四角可以不同，所以不能先取个最大值。
fn corner_radius(local: vec2<f32>, size: vec2<f32>, radii: vec4<f32>) -> f32 {
    let left = local.x < size.x * 0.5;
    let top = local.y < size.y * 0.5;
    if (top) {
        if (left) {
            return radii.x;
        }
        return radii.y;
    }
    if (left) {
        return radii.w;
    }
    return radii.z;
}

// 圆角覆盖：只在角的那 1/4 方块里做圆弧裁剪，其余整片都是 1。半像素过渡做抗锯齿。
fn rounded_coverage(local: vec2<f32>, size: vec2<f32>, radii: vec4<f32>) -> f32 {
    let r = max(corner_radius(local, size, radii), 0.0);
    if (r <= 0.0) {
        return 1.0;
    }
    let in_corner_x = local.x < r || local.x > size.x - r;
    let in_corner_y = local.y < r || local.y > size.y - r;
    if (!in_corner_x || !in_corner_y) {
        return 1.0;
    }
    var center = vec2<f32>(r, r);
    if (local.x > size.x - r) {
        center.x = size.x - r;
    }
    if (local.y > size.y - r) {
        center.y = size.y - r;
    }
    let d = length(local - center);
    return clamp(r - d + 0.5, 0.0, 1.0);
}

@fragment
fn fs_glass(input: GlassVarying) -> @location(0) vec4<f32> {    let p = input.position.xy;
    // content_mask 是设备像素下的矩形，与 quad 那侧同一套判据。
    if (p.x < input.clip.x || p.x > input.clip.x + input.clip.z ||
        p.y < input.clip.y || p.y > input.clip.y + input.clip.w) {
        return vec4<f32>(0.0);
    }
    let local = p - input.bounds.xy;
    let coverage = rounded_coverage(local, input.bounds.zw, input.radii);
    if (coverage <= 0.0) {
        return vec4<f32>(0.0);
    }
    let uv = (p - backdrop.region_origin) / backdrop.region_size;
    let blurred = textureSampleLevel(backdrop_map, backdrop_sampler, uv, 0.0);
    // 表面按预乘解释时，rgb 必须先乘覆盖，否则圆角外沿会亮出一圈边。
    if (backdrop.flags.x == 0u) {
        return vec4<f32>(blurred.rgb, coverage);
    }
    return vec4<f32>(blurred.rgb * coverage, coverage);
}
