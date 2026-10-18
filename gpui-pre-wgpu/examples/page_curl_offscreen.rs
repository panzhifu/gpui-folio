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

fn main() -> anyhow::Result<()> {
    let side = 256;
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
        vertices: Arc::from(vec![
            PageCurlVertex {
                position: [0.0, 0.0],
                uv: [0.0, 0.0],
            },
            PageCurlVertex {
                position: [side as f32, 0.0],
                uv: [1.0, 0.0],
            },
            PageCurlVertex {
                position: [side as f32, side as f32],
                uv: [1.0, 1.0],
            },
            PageCurlVertex {
                position: [0.0, side as f32],
                uv: [0.0, 1.0],
            },
        ]),
        indices: Arc::from(vec![0u32, 1, 2, 0, 2, 3]),
        uniforms: PageCurlUniforms {
            page_origin: [0.0, 0.0],
            page_size: [side as f32, side as f32],
            window_size: [side as f32, side as f32],
            axis_normal: [1.0, 0.0],
            axis_d: side as f32 * 0.9,
            radius: side as f32 * 0.1,
            camera_distance: side as f32 * 3.0,
            scale_factor: 1.0,
            back_color: [0.92, 0.9, 0.86, 1.0],
            light_dir: [-0.35, -0.25, 1.0, 0.0],
            shadow_map_size: 1024.0,
        },
    });

    let mut renderer = WgpuHeadlessRenderer::new()?;
    let image = renderer.render_to_image(&scene, size)?;
    image.save("/tmp/page_curl_offscreen.png")?;
    println!(
        "wrote /tmp/page_curl_offscreen.png ({}x{})",
        image.width(),
        image.height()
    );
    Ok(())
}
