//! 离屏渲染一张卷曲网格：验证管线、页面捕获、shadow pass 与 WGSL 都能在真设备上跑。
//!
//! ```sh
//! cargo run -p gpui-pre-wgpu --example page_curl_offscreen --features offscreen
//! ```
//!
//! 输出 `/tmp/page_curl_offscreen.png`。没有 GPU 的环境会在 `new()` 处报错。
//!
//! 这一份摆的是**生产那一档的数**，不是最省事的那一档：窗口比页大、页在窗口里内缩、
//! `scale_factor = 3`（鸿蒙模拟器 `densityDPI/160` 就是 3.0）。页铺满窗口、`scale = 1` 的
//! 时候着色器里「采样坐标映回页局部」那一步的两种读法恰好重合，量不出错 —— 夹具要是设在
//! 那个点上，它就永远不会红。
//!
//! 判据是**差分**的：同一张网格渲两次，一次把折线推到页外（全场平摊）、一次推到页中央，
//! 中间那一行上两个画面第一次不同的那一格就是屏上的折线。它应当落在解析解给的 `x` 上，
//! 容差 3 像素。色带本身是什么颜色不参与判断。

use std::sync::Arc;

use gpui::{
    Bounds, ContentMask, DevicePixels, Hsla, PageCurlMeshPrimitive, PageCurlUniforms,
    PageCurlVertex, Point, Quad, ScaledPixels, Scene, Size,
};
use gpui_wgpu::WgpuHeadlessRenderer;

/// 与 `folio_curl::GRID_X/Y` 同一档。
const GRID: usize = 96;

/// 窗口边长（设备像素）。
const WINDOW: f32 = 512.0;
/// 逻辑像素到设备像素的缩放 —— 手机上这一格是 2.x，鸿蒙模拟器是 3.0。
const SCALE: f32 = 3.0;
/// 页框在窗口里的左上角（设备像素）。
const ORIGIN: [f32; 2] = [48.0, 72.0];
/// 页框尺寸（设备像素）。
const PAGE: [f32; 2] = [416.0, 368.0];
/// 折线法向：这里取**竖直**的一条（`n = (1,0)`）。判据靠「屏上第一次不同的那一列」读折线，
/// 而折过柱顶那一段纸会**翻回折线左侧盖住平摊那一片** —— 斜的折线 + 折线在页中央 = 大半个页
/// 都进入折回区，那一列读出来的就不是折线。把折线推到贴着一侧的远缘，折线之外只剩薄薄一条，
/// 背面来不及盖回来，第一次不同才恰好是折线本身。斜的那一档留给肉眼看 PNG。
const NORMAL: [f32; 2] = [1.0, 0.0];
/// 卷径（设备像素），页宽的十分之一。
const RADIUS: f32 = PAGE[0] * 0.1;
/// 折线所在的 `n·p`（设备像素，页局部）。**离远缘留 100 像素**这一格是算出来的：折线外侧
/// 那点纸（100）必须小于半个卷弧 `πR ≈ 131`，否则绕柱顶那一段会翻回折线左侧盖住平摊区，
/// 「第一次不同」读到的就是那片近白的背面而不是折线。同时 100 > `πR/2 ≈ 65`，柱面最远那一格
/// 落在区间内部，右缘才跟着折线同速走。
const AXIS: f32 = PAGE[0] - 100.0;

fn mesh(axis_d: f32) -> PageCurlMeshPrimitive {
    let page_bounds = Bounds::new(
        Point {
            x: ScaledPixels(ORIGIN[0]),
            y: ScaledPixels(ORIGIN[1]),
        },
        Size {
            width: ScaledPixels(PAGE[0]),
            height: ScaledPixels(PAGE[1]),
        },
    );
    let window = Bounds::new(
        Point {
            x: ScaledPixels(0.0),
            y: ScaledPixels(0.0),
        },
        Size {
            width: ScaledPixels(WINDOW),
            height: ScaledPixels(WINDOW),
        },
    );
    PageCurlMeshPrimitive {
        order: 1,
        // 裁剪/剔除框给的是**页框**：越过书脊那一片本来就该画不出来（见 刀 2）。
        bounds: page_bounds,
        content_mask: ContentMask { bounds: window },
        page_bounds,
        // 与 `folio_curl::build_page_mesh` 同一张网格：顶点的 `position` 是**逻辑像素**，
        // 着色器自己乘 `scale_factor`。
        vertices: Arc::from({
            let logical = [PAGE[0] / SCALE, PAGE[1] / SCALE];
            let mut v = Vec::with_capacity((GRID + 1) * (GRID + 1));
            for y in 0..=GRID {
                for x in 0..=GRID {
                    let (u, w) = (x as f32 / GRID as f32, y as f32 / GRID as f32);
                    v.push(PageCurlVertex {
                        position: [u * logical[0], w * logical[1]],
                        uv: [u, w],
                    });
                }
            }
            v
        }),
        indices: Arc::from({
            let columns = (GRID + 1) as u32;
            let mut idx = Vec::with_capacity(GRID * GRID * 6);
            for y in 0..GRID as u32 {
                for x in 0..GRID as u32 {
                    let a = y * columns + x;
                    idx.extend_from_slice(&[
                        a,
                        a + columns,
                        a + 1,
                        a + 1,
                        a + columns,
                        a + columns + 1,
                    ]);
                }
            }
            idx
        }),
        uniforms: PageCurlUniforms {
            page_origin: ORIGIN,
            page_size: PAGE,
            window_size: [WINDOW, WINDOW],
            axis_normal: NORMAL,
            axis_d,
            radius: RADIUS,
            camera_distance: PAGE[0].max(PAGE[1]) * 3.0,
            scale_factor: SCALE,
            back_color: [0.92, 0.9, 0.86, 1.0],
            light_dir: [-0.35, -0.25, 1.0, 0.0],
            capture_origin: ORIGIN,
        },
    }
}

/// 页面内容：几条色带。捕获 pass 会把它烘进纹理，网格正面采样它；卷起来那段如果还能读到
/// 横向压缩的色带，说明 UV 与顶点变形都对了。
fn scene(axis_d: f32) -> Scene {
    let mut scene = Scene::default();
    // 横着切的四条带（**沿 x 是均匀的**），而且取中间亮度。两件事各治一个踩过的坑：
    //
    // - 竖着的色带会被柱面那一段的横向压缩（最多 `past − R·sin θ`，这里到 75 像素）搬到别的
    //   颜色上去，「卷曲 ÷ 平摊」的比值峰就落在**内容边界**而不是亮脊上（量到过：比值 1.8，
    //   而受光模型最多 1.10）。横带沿 x 均匀 ⇒ 比值里只剩受光。
    // - 一路顶到 255 的饱和色，4% 的受光差在 8 位图上逐位不变，差分什么都读不出来。
    let tones: [[u8; 4]; 4] = [
        [140, 60, 60, 255],
        [60, 140, 60, 255],
        [60, 60, 140, 255],
        [140, 140, 60, 255],
    ];
    let band = WINDOW / tones.len() as f32;
    for (index, color) in tones.into_iter().enumerate() {
        let [r, g, b, a] = color;
        let stripe = Bounds::new(
            Point {
                x: ScaledPixels(0.0),
                y: ScaledPixels(band * index as f32),
            },
            Size {
                width: ScaledPixels(WINDOW),
                height: ScaledPixels(band),
            },
        );
        scene.quads.push(Quad {
            order: 0,
            border_style: Default::default(),
            bounds: stripe,
            content_mask: ContentMask { bounds: stripe },
            background: gpui::Rgba {
                r: r as f32 / 255.0,
                g: g as f32 / 255.0,
                b: b as f32 / 255.0,
                a: a as f32 / 255.0,
            }
            .into(),
            border_color: Hsla::transparent_black(),
            corner_radii: Default::default(),
            border_widths: Default::default(),
        });
    }
    scene.page_capture = Some(Bounds::new(
        Point {
            x: ScaledPixels(ORIGIN[0]),
            y: ScaledPixels(ORIGIN[1]),
        },
        Size {
            width: ScaledPixels(PAGE[0]),
            height: ScaledPixels(PAGE[1]),
        },
    ));
    scene.page_curl_meshes.push(mesh(axis_d));
    scene
}

fn render(renderer: &mut WgpuHeadlessRenderer, axis_d: f32) -> image::RgbaImage {
    let size = Size {
        width: DevicePixels(WINDOW as i32),
        height: DevicePixels(WINDOW as i32),
    };
    renderer
        .render_to_image(&scene(axis_d), size)
        .expect("离屏渲得出来")

}

fn main() -> anyhow::Result<()> {
    let out = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "/tmp/page_curl_offscreen.png".into());
    let mut renderer = WgpuHeadlessRenderer::new()?;
    // 平摊参照：折线推到页外，整页都该与捕获逐字相同。
    let flat = render(&mut renderer, WINDOW * 4.0);
    // 同一个夹具、同一个卷径、同一个相机，只把折线沿法向推 `SHIFT` 个设备像素。
    const SHIFT: f32 = 40.0;
    let near = render(&mut renderer, AXIS);
    let far = render(&mut renderer, AXIS + SHIFT);
    near.save(&out)?;

    let row = (ORIGIN[1] + PAGE[1] * 0.5) as u32;
    let edge_left = |img: &image::RgbaImage| -> i32 {
        for x in (ORIGIN[0] as u32)..((ORIGIN[0] + PAGE[0]) as u32) {
            let a = flat.get_pixel(x, row).0;
            let b = img.get_pixel(x, row).0;
            if (0..3).any(|c| a[c].abs_diff(b[c]) > 2) {
                return x as i32;
            }
        }
        -1
    };
    let left_near = edge_left(&near);
    let left_far = edge_left(&far);
    // 加了贴折痕的投影之后，屏上第一条差异不在折线那一格了，而在**带尾**：`折线 − 0.5·R`。
    // 受光/投影都要爬到几个 LSB 才看得见，那一截固定延迟由下面第二格的差分抵消。
    let expected_left = ORIGIN[0] + AXIS - 0.5 * RADIUS;
    println!(
        "左缘（折线）：屏上 {left_near}，解析解 {expected_left:.1}，差 {:.1}",
        (left_near - expected_left as i32).abs()
    );
    if (left_near as f32 - expected_left).abs() > 12.0 {
        anyhow::bail!(
            "屏上的折线不在 `axis_d` 那一格（夹具：内缩 {}、缩放 {SCALE}）—— 片元把采样坐标映回页局部时读成了另一套尺度",
            ORIGIN[0]
        );
    }

    // 亮脊取「卷曲 ÷ 平摊」的**比值**峰：绝对亮度会被色带本身压过去（黄带比蓝带亮一半多，
    // 第一次就取到了内容边界上）。比值把内容除掉，剩下的只有受光那一项。
    // 第二格判据是**跟随**而不是绝对位置：把折线推 `SHIFT`，屏上那条边必须同速推 `SHIFT`。
    // 差分把「受光 ramps 起来才看得见」那几像素的固定延迟整个抵消掉，所以这一格不需要建模
    // 卷径、透视、亮脊在哪一柱角上 —— 前两档都在这里建模错过（比值峰其实是网格不再覆盖、
    // 露出底下原始 quad 的那条边，它跟的是覆盖宽度不是折线）。
    // 投影那道带的**深度剖面**（比值 = 卷曲 ÷ 平摊）：带尾之外该是 1.000，折痕处该掉到
    // `1 − SHADOW_PEAK ≈ 0.84`。这条不参与判据，是给人读数用的 —— 深不深、带宽不宽由眼睛判。
    let crease_x = (ORIGIN[0] + AXIS) as i32;
    for offset in [-30i32, -24, -18, -12, -6, -2, 2, 8] {
        let x = (crease_x + offset).clamp(0, (ORIGIN[0] + PAGE[0]) as i32 - 1) as u32;
        let a = flat.get_pixel(x, row).0;
        let b = near.get_pixel(x, row).0;
        let lo = (a[0] as u32) + (a[1] as u32) + (a[2] as u32);
        let hi = (b[0] as u32) + (b[1] as u32) + (b[2] as u32);
        println!(
            "  折痕{offset:+} x={x} 比值 {:.3}",
            hi as f32 / lo.max(1) as f32
        );
    }
    let moved = (left_far - left_near) as f32;
    println!("折线推 {SHIFT} 设备像素：屏上左缘从 {left_near} 到 {left_far}，走了 {moved:.1}");
    if (moved - SHIFT).abs() > 3.0 {
        anyhow::bail!(
            "屏上那条边只跟着折线走 {moved:.1}（该走 {SHIFT}）—— 片元现算 `past` 用的是另一套尺度"
        );
    }
    println!("屏上折线与 axis_d 同速 ✓");
    Ok(())
}
