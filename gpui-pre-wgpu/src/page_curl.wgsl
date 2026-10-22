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
    /// 圆锥：`[锥顶切向坐标, 自由边切向坐标, 权重, 最小半径]`。
    cone_a: vec4<f32>,
    /// `[最大半径, 0, 0, 0]`。
    cone_b: vec4<f32>,
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

/// 卷面根部的压暗深度：纸刚离开页面那一段最暗，往柱顶一路收回纸色原样。
///
/// 上一版把暗压在了折痕**外侧的平摊页**上，与 harism 恰好相反 —— 它的
/// `mPenumbraColor = v.mPosZ / radius` 与偏移 `(v.mPosZ / 2) * -curlDir` 都以抬起高度为
/// 唯一变量，而折痕那一格 `z = 0`，那里不该有影；给还躺着的纸蒙灰就是真机量到
/// 207→192 那个下凹的同一种错。把抬起那片真投到新页上（按 `(z/2)·(−n)` 偏移、alpha ∝ z）
/// 需要第二片网格 + 一条混合管线 + 压在新页之上卷面之下的层级，那是渲染器里的一趟。
const ROOT_SHADE: f32 = 0.18;

/// 投影最深那一格压掉几成。参照那一派（`CurlMesh` 的 `mPenumbraColor`）给的是随高度线性到
/// 满值再乘一个系数，这里取 0.35 —— 纸抬到最高（`z = 2R`）那一片底下最暗。
const SHADOW_PEAK: f32 = 0.35;

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
    /// **材质坐标** `0..1`，原点在页的左上角。`@location(0)` 那一份是「采样用的窗口归一坐标」
    /// `(capture_origin + 局部) / window`，两者只在「页铺满窗口、`capture_origin = 0`、缩放 1」
    /// 时数值相同 —— 页一内缩或 DPI 不为 1 就分开了。片元要把采样坐标映回页局部（现算折线、
    /// 背面透墨的镜像）必须读这一份，别拿那一份去反解。
    @location(5) material: vec2<f32>,
    /// 投影那一趟用的浓度（`0` = 这一趟不画投影）。正面那趟不读它。
    @location(6) shadow_alpha: f32,
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

/// 折痕切向坐标 `s` 处的卷径：`cone_weight = 0` 时就是 `radius`（圆柱），否则半径从
/// 书脊上的锥顶 0 线性长到自由边的 `radius`。CPU 那头的 `Cylinder::radius_at` 是同一条式子。
fn radius_at(p: vec2<f32>) -> f32 {
    let n = curl.axis_normal;
    let tangent = vec2<f32>(-n.y, n.x);
    let s = dot(tangent, p);
    let denom = curl.cone_a.y - curl.cone_a.x;
    let cone = clamp(
        curl.radius * (s - curl.cone_a.x) / select(1.0, denom, abs(denom) > 1.0),
        curl.cone_a.w,
        curl.cone_b.x,
    );
    return max(mix(curl.radius, cone, clamp(curl.cone_a.z, 0.0, 1.0)), 0.001);
}

fn deform(page_position: vec2<f32>) -> Deformed {
    let p = page_position * curl.scale_factor;
    let n = curl.axis_normal;
    let d = dot(n, p) - curl.axis_d;
    let r = radius_at(p);

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
    // 材质坐标原样带下去：片元要用它把「这一像素是页上的哪一点」问回来，而上面那份 `out.uv`
    // 是采样坐标（窗口归一 + 烘时原点），拿它反解页局部在页内缩或缩放不为 1 时就是错的。
    out.material = input.uv;
    let local = input.position * curl.scale_factor;
    out.edge = min(
        min(local.x, curl.page_size.x - local.x),
        min(local.y, curl.page_size.y - local.y),
    );
    return out;
}

/// 投影那一趟的顶点：把卷起来的网格沿 `-n` 按**抬起高度**平移，再钉在页面平面上。
///
/// 偏移与浓度都 `∝ z` —— 照 `harism/android-pagecurl` 的 `CurlMesh`：
/// `sv.mPenumbraX = (v.mPosZ / 2) * -curlDir.x`、`sv.mPenumbraColor = v.mPosZ / radius`
/// （背面那一段换成 `(v.mPosZ − radius) / (2 * radius)`）。折痕处 `z = 0` ⇒ 那里既不移也不暗，
/// 纸还贴着页，本来就没有影；越往柱顶抬，落点离折线越远、影越浓。
@vertex
fn vs_page_curl_shadow(input: VertexInput) -> VertexOutput {
    let def = deform(input.position);
    let r = radius_at(input.position * curl.scale_factor);
    let lift = def.position.z;
    var out: VertexOutput;
    out.position = to_device_position(
        def.position.xy - curl.axis_normal * (lift * 0.5), 1.0);
    out.uv =
        (curl.capture_origin + input.position * curl.scale_factor) / curl.window_size;
    out.normal = def.normal;
    out.past = def.past;
    out.back_fade = def.back_fade;
    out.material = input.uv;
    out.shadow_alpha = SHADOW_PEAK * clamp(lift / (2.0 * r), 0.0, 1.0);
    return out;
}

/// 投影那一趟的片元：一个预乘的黑，alpha 由顶点按高度给好。
@fragment
fn fs_page_curl_shadow(in: VertexOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(0.0, 0.0, 0.0, in.shadow_alpha);
}

@fragment
fn fs_page_curl(in: VertexOutput) -> @location(0) vec4<f32> {
    // 捕获纹理是整窗大小，只有页面矩形那块被写过。取 uv 时往矩形内收半像素：即使页框
    // 原点有半像素取整误差，也不会采到清屏色或邻区。捕获纹理的清屏色在渲染器里与纸同色，
    // 这里再收一道只是不让边缘像素参与插值。
    let uv_min = (curl.capture_origin + vec2<f32>(0.5)) / curl.window_size;
    let uv_max = (curl.capture_origin + curl.page_size - vec2<f32>(0.5)) / curl.window_size;
    let capture_uv = clamp(in.uv, uv_min, uv_max);
    // 全场唯一一次正面采样。
    let paper = textureSampleLevel(capture_texture, capture_sampler, capture_uv, 0.0).rgb;

    // **这一格是整段的关键**：`past` 不用顶点传进来的那一份，而是从 `uv` 现算。
    //
    // 顶点插值出来的 `past` 与 `normal`，等值线只能对齐网格顶点 —— 网格一格 = 页宽/96 ≈
    // 手机上 11 个物理像素，于是斜着的接缝在屏上是一格一格的台阶（真机量到过，读者报的
    // 「右下角跳动」里空间那一半就是它）。而 `uv` 是材质坐标，它在三角形内线性插值出来的
    // 就是那一点本身：`n·(材质·页尺寸) − d0` 逐像素精确，等值线是一条**解析直线**。
    // 顺带法向也不用插值了 —— 卷成柱面那一段的法向由 `θ = past / R` 完全决定。
    let local = in.material * curl.page_size;
    // 圆锥的半径沿折痕变化，必须像 `past` 一样逐像素现算；拿顶点插值会在斜折痕上出台阶。
    let r = radius_at(local);
    let past = dot(curl.axis_normal, local) - curl.axis_d;
    let theta = past / r;
    // 带宽 = 这一像素跨过的距离。硬边之所以难看不是因为它硬，是因为它走的是台阶；台阶换成
    // 直线之后，一到两个像素的过渡就够把剩下的锯齿吃掉。
    let w = max(fwidth(past), 1e-4);

    if (past <= -w) {
        // 平摊那一片 = 页面本身 + 卷起来那一片纸落在它上面的一道投影。
        //
        // 这**不是**「把投影乘进整张平摊页面」：那一条会在整条纸边罩出一层浅灰（真机量到
        // 207→192 那个下凹就是它，也是这一档从前被否掉的原因）。这里是一道**贴着折痕**的窄带：
        // 宽度跟着卷径走（`0.5·R`，页宽 390 那一档约 27 逻辑像素，与鸿蒙官方示例的
        // `maxShadowWidth = 30` 同一量级），折痕处最深、到带尾归零，带尾之外一个字都不调。
        // `past` 已经是逐像素解析距离，所以这条带不要第二片网格、不多一次采样、也不走 CPU 的
        // `Shades` 那一层（那条路上 gpui 的渐变只装两个色标，糊不出这道深浅）。
        let shadow_w = max(r * 0.5, 1.0);
        // 平摊那一片就是页面本身，一个字都不调：折痕处纸还贴在页上（`z = 0`），
        // 那里没有影可画。抬起那一段的根部压暗在下面的 `front` 里，真投影另说。
        return vec4<f32>(paper, 1.0);
    }

    // 卷面法向：圆柱那一条是 `(n·sinθ, cosθ)`；圆锥的半径沿折痕变化，法向还要沿切向
    // 偏一点 —— 偏量是 `r'(s)·(cosθ−1)`，θ=0 处不偏（贴着折痕），往背面一路增大。
    // 不做这一项的话，圆锥的几何在卷，受光却还是直筒，亮脊不会跟着锥面斜。
    let cone_denom = curl.cone_a.y - curl.cone_a.x;
    let r_prime = select(
        0.0,
        curl.cone_a.z * curl.radius / cone_denom,
        abs(cone_denom) > 1.0,
    );
    let tangent = vec2<f32>(-curl.axis_normal.y, curl.axis_normal.x);
    let n = normalize(
        vec3<f32>(curl.axis_normal * sin(theta), cos(theta))
            + vec3<f32>(tangent, 0.0) * (r_prime * (cos(theta) - 1.0)),
    );
    let light = normalize(curl.light_dir.xyz);
    let diffuse = max(dot(n, light), 0.0);
    let light_z = max(light.z, 0.001);
    // 卷曲段上走过的归一化弧长：0 = 贴着折线，0.5 = 柱顶，1 = 绕过柱顶落到背面。
    let roll = clamp(past / (PI * r), 0.0, 1.0);
    // 柱顶那道**亮脊**是「纸有厚度」的主要线索，往两端收回到纸色，与平摊那一段接缝不跳。
    // 从前这里主项取 N·L，而 `light_dir` 主要沿视线 —— 柱顶那一点法向与视线垂直，亮度
    // 反而掉到 0.69，真机上量到的是卷曲带比纸面还暗（207→192 那个下凹）：受光装反了。
    // N·L 现在只留一点点调制。
    let crest = sin(roll * PI);
    // 根部压暗：`height = 1 − cos θ` 归一的就是「纸离页多高」（`z = R·height`）。折痕处
    // `height = 0` ⇒ 一点不压，抬到柱面中段最深，再往柱顶收 —— 与 harism 那句
    // `mColorFactor = .1 + .9·sqrt(...)` 同一个形状：暗部跟着**抬起高度**走，不是贴在折痕上。
    let height = 1.0 - cos(theta);
    let root = ROOT_SHADE * height * clamp(1.0 - roll * 1.5, 0.0, 1.0);
    let front = paper * (1.0 + 0.10 * crest + 0.05 * (diffuse / light_z - 1.0) * (1.0 - crest)
        - root);

    // 背面 = 接近纸白 + **对面那一版字透过来**，再只在贴着柱顶那一窄条压一道内侧影。参照里
    // 那句「整片 ×0.92、内侧再 ×0.82」在真机上量到 192 / 235 两档，摊开成一条三百多像素的
    // 灰带 —— 读者报的「黑条」就是它。这里底提到 0.94，让那道影在六分之一段内就走完，读出来
    // 是一条接触线而不是一条带。
    //
    // 透墨那一次采的是**同一张**捕获纹理，只是把材质坐标沿页的中线左右镜像（背面看到的字
    // 本来就是反的）：不新增 uniform、不新增 pass，只在背面那几片多一次采样。镜像后的坐标
    // 恒落在 `[0, 页宽]` 里 ⇒ 采不到捕获矩形的外圈 —— 那一圈没被写过，采到就是黑。
    let material = in.material * curl.page_size;
    let ghost_uv =
        (curl.capture_origin + vec2<f32>(curl.page_size.x - material.x, material.y)) / curl.window_size;
    let ghost = textureSampleLevel(
        capture_texture,
        capture_sampler,
        clamp(ghost_uv, uv_min, uv_max),
        0.0,
    ).rgb;
    let back_fade = clamp((theta - PI) / 1.5, 0.0, 1.0);
    let back = mix(curl.back_color.rgb, ghost, BACK_INK)
        * mix(0.94, 1.0, min(back_fade * 6.0, 1.0));

    // 正面 ↔ 背面：纸绕过柱顶转过去的那一瞬（θ = π/2，即 `past = πR/2`）法向过零。从前这里
    // 是 `n.z >= 0` 一刀切 —— 判据本身没错，错在它读的是插值法向，于是那条线落在网格上就是
    // 台阶。读现算的 `past`，它就是一条解析直线，两像素的过渡把剩下的锯齿吃掉。
    let turn = PI * r * 0.5;   // 过这一点纸转到背面：`side` 随 past 增大而**减小**
    let side = clamp(0.5 + (turn - past) / (2.0 * w), 0.0, 1.0);
    let curled = mix(back, front, side);
    // 平摊 ↔ 卷起：位置在这里连续（`u` 两侧都等于 `past`），换的只有受光与内容那一档。
    let rim = vec3<f32>(0.06 * (1.0 - smoothstep(0.0, 2.0, in.edge)));
    return vec4<f32>(mix(paper, curled + rim, smoothstep(-w, w, past)), 1.0);
}
