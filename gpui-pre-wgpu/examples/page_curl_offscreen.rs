//! 离屏渲染一张卷曲网格：验证管线、页面捕获、shadow pass 与 WGSL 都能在真设备上跑。
//!
//! ```sh
//! cargo run -p gpui-pre-wgpu --example page_curl_offscreen --features offscreen
//! ```
//!
//! 输出 `/tmp/page_curl_offscreen.png`。没有 GPU 的环境会在 `new()` 处报错。

use std::sync::Arc;

use gpui::{
    Bounds, ContentMask, DevicePixels, Hsla, PageCurlMeshPrimitive, PageCurlUniforms,
    PageCurlVertex, Point, Quad, ScaledPixels, Scene, Size,
};
use gpui_wgpu::WgpuHeadlessRenderer;

/// 与 `folio_curl::GRID_X/Y` 同一档。
const GRID: usize = 96;

fn main() -> anyhow::Result<()> {
    // 512 而不是 256：网格一格 = 边长/96，边长太小台阶只有两三个像素，量不出差别。
    let side = 512;
    let out = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "/tmp/page_curl_offscreen.png".into());
    let size = Size {
        width: DevicePixels(side),
        height: DevicePixels(side),
    };
    let bounds = Bounds::new(
        Point {
            x: ScaledPixels(0.0),
            y: ScaledPixels(0.0),
        },
        Size {
            width: ScaledPixels(side as f32),
            height: ScaledPixels(side as f32),
        },
    );

    let mut scene = Scene::default();
    // 页面内容：几条色带。捕获 pass 会把它烘进纹理，网格正面采样它；
    // 如果卷起来那段还能读到色带，说明 UV 与捕获区域对上了。
    // 页面内容：几条色带。捕获 pass 会把它烘进纹理，网格正面采样它；
    // 卷曲段如果还能读到横向压缩的色带，说明 UV 与顶点变形都对了。
    let colors = [gpui::red(), gpui::green(), gpui::blue(), gpui::yellow()];
    for (index, color) in colors.into_iter().enumerate() {
        let left = side as f32 / colors.len() as f32 * index as f32;
        let stripe = Bounds::new(
            Point {
                x: ScaledPixels(left),
                y: ScaledPixels(0.0),
            },
            Size {
                width: ScaledPixels(side as f32 / colors.len() as f32),
                height: ScaledPixels(side as f32),
            },
        );
        scene.quads.push(Quad {
            order: 0,
            border_style: Default::default(),
            bounds: stripe,
            content_mask: ContentMask { bounds: stripe },
            background: color.into(),
            border_color: Hsla::transparent_black(),
            corner_radii: Default::default(),
            border_widths: Default::default(),
        });
    }
    scene.page_capture = Some(bounds);
    scene.page_curl_meshes.push(PageCurlMeshPrimitive {
        order: 1,
        bounds,
        content_mask: ContentMask { bounds },
        page_bounds: bounds,
        // 与 `folio_curl::build_page_mesh` 同一张网格：这一格的全部意义就是「折线落在三角形
        // 内部时，边界只能对齐顶点」，四角一张面片是量不出来的。折线取斜的（43°），因为竖直
        // 的折线恰好与网格对齐，台阶天生看不见。
        vertices: Arc::from({
            let mut v = Vec::with_capacity((GRID + 1) * (GRID + 1));
            for y in 0..=GRID {
                for x in 0..=GRID {
                    let (u, w) = (x as f32 / GRID as f32, y as f32 / GRID as f32);
                    v.push(PageCurlVertex {
                        position: [u * side as f32, w * side as f32],
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
            page_origin: [0.0, 0.0],
            page_size: [side as f32, side as f32],
            window_size: [side as f32, side as f32],
            // 43°：与点屏那一笔起手同一档（`MAX_TILT`）。轴过页面中心偏左一点，让柱面与
            // 背面那两片都落在画面里。
            axis_normal: [0.7282, 0.6854],
            axis_d: side as f32 * 0.62,
            radius: side as f32 * 0.1,
            camera_distance: side as f32 * 3.0,
            scale_factor: 1.0,
            back_color: [0.92, 0.9, 0.86, 1.0],
            light_dir: [-0.35, -0.25, 1.0, 0.0],
            capture_origin: [0.0, 0.0],
        },
    });

    let mut renderer = WgpuHeadlessRenderer::new()?;
    let image = renderer.render_to_image(&scene, size)?;
    image.save(&out)?;
    println!("wrote {out} ({}x{})", image.width(), image.height());
    Ok(())
}
