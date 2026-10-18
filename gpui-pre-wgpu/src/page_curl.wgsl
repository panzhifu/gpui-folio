// 页面卷曲的 WGSL：圆柱形变、正反面、页面纹理、阴影 map。
//
// 纸模型照 gmpagecurl-ios / XBPageCurl：
//   d <= 0            平摊
//   0 < d < πR        卷在圆柱上：u = R·sinθ, z = R·(1−cosθ)
//   d >= πR           背面平铺：u = πR − d, z = 2R
//
// 这里没有 compute pass：顶点数固定，顶点着色器直接算，省一读一写。
// shadow map 是一张 Rgba16Float 深度图，主 shader 用 3×3 PCF 采样。

struct GlobalParams {
    viewport_size: vec2<f32>,
    premultiplied_alpha: u32,
    pad: u32,
}

struct GammaParams {
    gamma_ratios: vec4<f32>,
    grayscale_enhanced_contrast: f32,
    subpixel_enhanced_contrast: f32,
    is_bgr: u32,
    pad: u32,
}

@group(0) @binding(0) var<uniform> globals: GlobalParams;
@group(0) @binding(1) var<uniform> gamma_params: GammaParams;

struct CurlUniforms {
    page_origin: vec2<f32>,
    page_size: vec2<f32>,
    window_size: vec2<f32>,
    axis_normal: vec2<f32>,
    axis_d: f32,
    radius: f32,
    camera_distance: f32,
    scale_factor: f32,
    back_color: vec4<f32>,
    light_dir: vec4<f32>,
    shadow_map_size: f32,
    light_view_proj: mat4x4<f32>,
}

@group(1) @binding(0) var<uniform> curl: CurlUniforms;

@group(2) @binding(0) var capture_texture: texture_2d<f32>;
@group(2) @binding(1) var capture_sampler: sampler;

@group(3) @binding(0) var shadow_texture: texture_2d<f32>;
@group(3) @binding(1) var shadow_sampler: sampler;

const PI: f32 = 3.141592653589793;
const SHADOW_BIAS: f32 = 0.01;

struct VertexInput {
    @location(0) position: vec2<f32>,
    @location(1) uv: vec2<f32>,
}

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) light_pos: vec4<f32>,
    @location(3) world: vec3<f32>,
    /// 到页面外框的像素距离；片元用它压一道纸边光。
    @location(4) edge: f32,
}

struct ShadowOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) depth: f32,
}

struct Deformed {
    position: vec3<f32>,
    normal: vec3<f32>,
}

fn to_device_position(position: vec2<f32>, depth: f32) -> vec4<f32> {
    let ndc = position / globals.viewport_size * vec2<f32>(2.0, -2.0) + vec2<f32>(-1.0, 1.0);
    return vec4<f32>(ndc, depth, 1.0);
}

fn deform(page_position: vec2<f32>) -> Deformed {
    let p = page_position * curl.scale_factor;
    let n = curl.axis_normal;
    let r = max(curl.radius, 0.001);
    let d = dot(n, p) - curl.axis_d;
    let center = curl.page_size * 0.5;
    let tangent = vec2<f32>(-n.y, n.x);

    var u: f32;
    var z: f32;
    var normal: vec3<f32>;

    if (d <= 0.0) {
        // 平摊：原样，法线朝上。
        u = d;
        z = 0.0;
        normal = vec3<f32>(0.0, 0.0, 1.0);
    } else if (d < PI * r) {
        // 卷曲段：弧长 = d，θ = d/R。
        let theta = d / r;
        u = r * sin(theta);
        z = r * (1.0 - cos(theta));
        normal = vec3<f32>(n * sin(theta), cos(theta));
    } else {
        // 绕过圆柱顶，落在 z = 2R 的背面平面；纹理仍是同一张纸。
        u = PI * r - d;
        z = 2.0 * r;
        normal = vec3<f32>(0.0, 0.0, -1.0);
    }

    // 弱透视：相机在页面上方，z 越大离读者越近，投影越大。
    let scale = curl.camera_distance / max(curl.camera_distance - z, 1.0);
    // 平摊时 `u = d`，这一串必须正好还原成 `p`：
    //   axis_foot + n*d = p
    // 卷曲时才把横截面坐标换成 `u = R·sinθ`。
    let axis_foot = p - n * d;
    let v_rel = dot(tangent, axis_foot - center);
    let axis_foot_proj = axis_foot - tangent * v_rel + tangent * (v_rel * scale);
    let world2 = curl.page_origin + axis_foot_proj + n * u * scale;

    return Deformed(vec3<f32>(world2, z), normal);
}

@vertex
fn vs_page_curl(input: VertexInput) -> VertexOutput {
    let def = deform(input.position);
    var out: VertexOutput;
    // 深度：z 越大越近 ⇒ 深度值越小，背面（z=2R）压住平摊页（z=0）。
    let depth = clamp(1.0 - def.position.z / curl.camera_distance, 0.0, 1.0);
    out.position = to_device_position(def.position.xy, depth);
    out.uv = (curl.page_origin + input.position * curl.scale_factor) / curl.window_size;
    out.normal = def.normal;
    out.world = def.position;
    out.light_pos = curl.light_view_proj * vec4<f32>(def.position, 1.0);
    let local = input.position * curl.scale_factor;
    out.edge = min(
        min(local.x, curl.page_size.x - local.x),
        min(local.y, curl.page_size.y - local.y),
    );
    return out;
}

fn sample_shadow(light_pos: vec4<f32>) -> f32 {
    let ndc = light_pos.xyz / light_pos.w;
    let uv = ndc.xy * vec2<f32>(0.5, -0.5) + vec2<f32>(0.5, 0.5);
    if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) {
        return 1.0;
    }

    let texel = 1.0 / max(curl.shadow_map_size, 1.0) * 2.0;
    var lit: f32 = 0.0;
    // 5×5 PCF：3×3 在 1024 的图上只有约 1px，真机上就是一条硬边黑带。
    for (var y: i32 = -2; y <= 2; y = y + 1) {
        for (var x: i32 = 2; x >= -2; x = x - 1) {
            let offset = vec2<f32>(f32(x), f32(y)) * texel;
            let stored = textureSample(shadow_texture, shadow_sampler, uv + offset).r;
            let shadowed = select(1.0, 0.0, ndc.z > stored + SHADOW_BIAS);
            lit = lit + shadowed;
        }
    }
    return lit / 25.0;
}

@fragment
fn fs_page_curl(
    in: VertexOutput,
    @builtin(front_facing) front_facing: bool,
) -> @location(0) vec4<f32> {
    let n = normalize(in.normal);
    let light = normalize(curl.light_dir.xyz);
    let diffuse = max(dot(n, light), 0.0);
    let light_z = max(light.z, 0.001);

    var base: vec3<f32>;
    var brightness: f32;
    // 正反面由法线判，不信 winding：平面的三角绕序在 NDC 的 y 翻转后
    // 可能被当成背面，那样平摊页会整片读成纸背。
    if (n.z >= 0.0) {
        base = textureSample(capture_texture, capture_sampler, in.uv).rgb;
        // 平摊页（n.z = 1）亮度正好 1；卷曲处按 N·L 压暗。
        brightness = clamp(0.55 + 0.45 * (diffuse / light_z), 0.35, 1.0);
    } else {
        // 背面：纸色 + 环境光。不采样正面纹理，避免把正文镜像到背面。
        base = curl.back_color.rgb;
        brightness = 0.92;
    }

    let shadow = sample_shadow(in.light_pos);
    // 纸边：最外 2px 抬一点白，读成纸的厚度，而不是一刀切的贴图边界。
    let edge = 1.0 - smoothstep(0.0, 2.0, in.edge);
    let lit = base * brightness * mix(0.78, 1.0, shadow);
    return vec4<f32>(lit + vec3<f32>(0.06 * edge), 1.0);
}

@vertex
fn vs_shadow(input: VertexInput) -> ShadowOutput {
    let def = deform(input.position);
    var out: ShadowOutput;
    out.position = curl.light_view_proj * vec4<f32>(def.position, 1.0);
    out.depth = out.position.z / out.position.w;
    return out;
}

@fragment
fn fs_shadow(in: ShadowOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(in.depth, 0.0, 0.0, 1.0);
}

// 新页上的投影：一张平铺在 z=0 的整页 quad，按 shadow map 输出黑色 alpha。
struct OverlayOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) light_pos: vec4<f32>,
}

@vertex
fn vs_overlay(input: VertexInput) -> OverlayOutput {
    let p = input.position * curl.scale_factor;
    let world = vec3<f32>(curl.page_origin + p, 0.0);
    var out: OverlayOutput;
    out.position = to_device_position(world.xy, 0.0);
    out.light_pos = curl.light_view_proj * vec4<f32>(world, 1.0);
    return out;
}

@fragment
fn fs_overlay(in: OverlayOutput) -> @location(0) vec4<f32> {
    let shadow = sample_shadow(in.light_pos);
    let alpha = (1.0 - shadow) * 0.20;
    return vec4<f32>(0.0, 0.0, 0.0, alpha);
}
