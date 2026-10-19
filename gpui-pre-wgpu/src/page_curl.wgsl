// 页面卷曲的 WGSL：一张纸绕着折痕卷成一段圆柱。
//
// 截面分三段，`past` = 那一点到折线的带符号距离（正 = 已经被卷走那一侧）：
//   past <= 0        平摊在原位
//   0 < past < πR    卷在半径 R 的圆柱上：θ = past/R，横移 u = R·sinθ，抬起 z = R·(1−cosθ)
//   past >= πR       绕过柱顶，平铺在 z = 2R 的背面，横坐标 u = πR − past（镜像回去）
//
// R 是**常数**（纸的挺度），不随手指拖多远变。前一版按「到锥顶的距离」把半径线性收成圆锥，
// 于是手指靠近书脊那一段锥顶跑远、半径随折线角度剧变，某一档角度上会跳一帧；圆柱在这一档
// 要的二维投影里与它看不出差别，就取圆柱。
//
// **没有 shadow map**，也**不在平摊纸上盖灰**。参照 harism/android-pagecurl 的做法，投影
// 是卷筒自己的几何落下的一道独立透明网格；把它乘进平摊纸面的颜色里，等于给整条纸边罩一层
// 灰，正是读者报的「纸上一层浅灰影」。平摊段只采样捕获到的页纸，不做任何明暗处理。
//
// 两处采样都用 `textureSampleLevel(..., 0)`：平摊与卷曲是按像素分界判的，非一致控制流里隐式
// 求导拿到的 LOD 是未定义的；而那张捕获纹理本来就只有一级 mip，写死 0 既不亏也不错。

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
    /// **烘那张纹理那一刻**页面左上角（设备像素）。取 uv 用它，摆几何用 `page_origin`：
    /// 阅读层在动（进出书的滑动、面板、尺寸变化）时这两个不是同一个数，而纹理里被写过的
    /// 只有烘的那一刻那一块 —— 混用就会采到未写区域（读成黑带）。
    capture_origin: vec2<f32>,
    /// 保持 uniform 结构体 16 字节对齐；没有语义。
    _padding: vec2<f32>,
}

@group(1) @binding(0) var<uniform> curl: CurlUniforms;

@group(2) @binding(0) var capture_texture: texture_2d<f32>;
@group(2) @binding(1) var capture_sampler: sampler;

const PI: f32 = 3.141592653589793;

/// 背面透墨的量：从这一页**正面**的字按几成透过来（0 = 死白的一张纸，1 = 正面原样）。
///
/// 书上的纸不是不透的：卷起来那一片的背面，看得见对面那一版字的淡影（左右反）。参照那一派
/// （`harism/android-pagecurl` 的 `CurlPatch`）给的是纯色纸 ×0.92，真机上读成一张死白 ——
/// 少的是这一格透墨，不是亮度。0.12 起：再高就分不清是纸脏了还是字透过来。
const BACK_INK: f32 = 0.12;

struct VertexInput {
    @location(0) position: vec2<f32>,
    @location(1) uv: vec2<f32>,
}

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) normal: vec3<f32>,
    /// 到折线的带符号距离（设备像素）：<= 0 平摊，> 0 已经被卷走。
    @location(2) past: f32,
    /// 到页面外框的像素距离；片元用它压一道纸边光。
    @location(3) edge: f32,
    /// 背面离折痕的归一化距离（0 = 贴着折痕，1 = 自由边），给背面压内侧渐变。
    @location(4) back_fade: f32,
}

struct Deformed {
    position: vec3<f32>,
    normal: vec3<f32>,
    past: f32,
    back_fade: f32,
}

fn to_device_position(position: vec2<f32>, depth: f32) -> vec4<f32> {
    let ndc = position / globals.viewport_size * vec2<f32>(2.0, -2.0) + vec2<f32>(-1.0, 1.0);
    return vec4<f32>(ndc, depth, 1.0);
}

fn deform(page_position: vec2<f32>) -> Deformed {
    let p = page_position * curl.scale_factor;
    let n = curl.axis_normal;
    let d = dot(n, p) - curl.axis_d;
    let r = max(curl.radius, 0.001);

    var u: f32;
    var z: f32;
    var normal: vec3<f32>;
    var back_fade: f32 = 0.0;

    if (d <= 0.0) {
        // 平摊：横坐标就是它自己，下面那串还原回 `p`，一格也不该挪。
        u = d;
        z = 0.0;
        normal = vec3<f32>(0.0, 0.0, 1.0);
    } else if (d < PI * r) {
        // 卷在柱面上：走过的弧长就是 `d`。
        let theta = d / r;
        u = r * sin(theta);
        z = r * (1.0 - cos(theta));
        normal = vec3<f32>(n * sin(theta), cos(theta));
    } else {
        // 绕过柱顶，落在 z = 2R 的背面平面；采的仍是同一张纸，所以看着是空白纸背。
        u = PI * r - d;
        z = 2.0 * r;
        normal = vec3<f32>(0.0, 0.0, -1.0);
        back_fade = clamp((d / r - PI) / 1.5, 0.0, 1.0);
    }

    // 弱透视：相机在页面上方，抬得越高离读者越近、投影越大。平摊时 z = 0 ⇒ scale = 1。
    let scale = curl.camera_distance / max(curl.camera_distance - z, 1.0);
    // `axis_foot + n*d = p`，而平摊段 `u = d`，于是 `scale = 1` 时下面这行正好还原成 `p`。
    let axis_foot = p - n * d;
    let center = curl.page_size * 0.5;
    let tangent = vec2<f32>(-n.y, n.x);
    let v_rel = dot(tangent, axis_foot - center);
    let world2 =
        curl.page_origin + axis_foot + tangent * (v_rel * (scale - 1.0)) + n * (u * scale);

    return Deformed(vec3<f32>(world2, z), normal, d, back_fade);
}

@vertex
fn vs_page_curl(input: VertexInput) -> VertexOutput {
    let def = deform(input.position);
    var out: VertexOutput;
    // 深度：抬得越高越靠近读者 ⇒ 深度值越小，背面（z = 2R）压住平摊页（z = 0）。
    let depth = clamp(1.0 - def.position.z / curl.camera_distance, 0.0, 1.0);
    out.position = to_device_position(def.position.xy, depth);
    // 捕获纹理是整窗大小、只有「烘那一刻的页框」那块被写过，所以按窗口归一取 uv，而且用的是
    // **烘时**的原点 `capture_origin`（摆几何用的是本帧的 `page_origin`，两者可以不同）。
    out.uv =
        (curl.capture_origin + input.position * curl.scale_factor) / curl.window_size;
    out.normal = def.normal;
    out.past = def.past;
    out.back_fade = def.back_fade;
    let local = input.position * curl.scale_factor;
    out.edge = min(
        min(local.x, curl.page_size.x - local.x),
        min(local.y, curl.page_size.y - local.y),
    );
    return out;
}

@fragment
fn fs_page_curl(in: VertexOutput) -> @location(0) vec4<f32> {
    // 捕获纹理是整窗大小，只有页面矩形那块被写过。取 uv 时往矩形内收半像素：即使页框
    // 原点有半像素取整误差，也不会采到清屏色或邻区。捕获纹理的清屏色在渲染器里与纸同色，
    // 这里再收一道只是不让边缘像素参与插值。
    let uv_min = (curl.capture_origin + vec2<f32>(0.5)) / curl.window_size;
    let uv_max = (curl.capture_origin + curl.page_size - vec2<f32>(0.5)) / curl.window_size;
    let capture_uv = clamp(in.uv, uv_min, uv_max);
    // 先在分支外采一次：这一趟是全场唯一一次读那张纹理，平摊与卷曲两段共用。
    let paper = textureSampleLevel(capture_texture, capture_sampler, capture_uv, 0.0).rgb;

    if (in.past <= 0.0) {
        // 平摊那一片就是页面本身。参照实现不在平摊纸上画投影；投影是卷筒在下面那张
        // 新页上的一道独立影。这里若乘一个 `1 - ink`，整条纸边都会罩一层浅灰。
        return vec4<f32>(paper, 1.0);
    }

    let n = normalize(in.normal);
    let light = normalize(curl.light_dir.xyz);
    let diffuse = max(dot(n, light), 0.0);
    let light_z = max(light.z, 0.001);
    // 卷曲段上走过的归一化弧长：0 = 贴着折线，0.5 = 柱顶，1 = 绕过柱顶落到背面。
    let roll = clamp(in.past / (PI * max(curl.radius, 0.001)), 0.0, 1.0);
    var base: vec3<f32>;
    var brightness: f32;
    // 正反面按法线判，不信 winding：平摊段的三角绕序在 NDC 翻 y 之后可能被当成背面，
    // 那样整页正文会读成空白纸背。
    if (n.z >= 0.0) {
        base = paper;
        // 柱顶那道**亮脊**是「纸有厚度」的主要线索，往两端收回到纸色，与平摊那一段接缝不跳。
        // 从前这里主项取 N·L，而 `light_dir` 主要沿视线 —— 柱顶那一点法向与视线垂直，亮度
        // 反而掉到 0.69，真机上量到的是卷曲带比纸面还暗（207→192 那个下凹）：受光装反了。
        // N·L 现在只留一点点调制。
        let crest = sin(roll * PI);
        brightness = 1.0 + 0.10 * crest + 0.05 * (diffuse / light_z - 1.0) * (1.0 - crest);
    } else {
        // 背面 = 接近纸白 + **对面那一版字透过来**，再只在贴着柱顶那一窄条压一道内侧影。参照里
        // 那句「整片 ×0.92、内侧再 ×0.82」在真机上量到 192 / 235 两档，摊开成一条三百多像素的
        // 灰带 —— 读者报的「黑条」就是它。这里底提到 0.94，让那道影在六分之一段内就走完，读出来
        // 是一条接触线而不是一条带。
        //
        // 透墨那一次采的是**同一张**捕获纹理，只是把材质坐标沿页的中线左右镜像（背面看到的字
        // 本来就是反的）：不新增 uniform、不新增 pass，只在背面那几片多一次采样。镜像后的坐标
        // 恒落在 `[0, 页宽]` 里 ⇒ 采不到捕获矩形的外圈 —— 那一圈没被写过，采到就是黑。
        let local = capture_uv * curl.window_size - curl.capture_origin;
        let ghost_uv =
            (curl.capture_origin + vec2<f32>(curl.page_size.x - local.x, local.y)) / curl.window_size;
        let ghost = textureSampleLevel(
            capture_texture,
            capture_sampler,
            clamp(ghost_uv, uv_min, uv_max),
            0.0,
        ).rgb;
        base = mix(curl.back_color.rgb, ghost, BACK_INK);
        brightness = mix(0.94, 1.0, min(in.back_fade * 6.0, 1.0));
    }
    // 纸边：最外 2px 抬一点白，读成纸有自己的厚度，而不是一刀切的贴图边界。
    let edge = 1.0 - smoothstep(0.0, 2.0, in.edge);
    return vec4<f32>(base * brightness + vec3<f32>(0.06 * edge), 1.0);
}
