//! 离屏渲染一片毛玻璃：验证降采样、两趟高斯、玻璃管线与「贴回表面」这一串能在真设备上跑，
//! 也钉住玻璃管线必须与主 pass 的深度附件同格式 —— 0.3.14 起主 pass 带 Depth32Float，漏了
//! 就是每帧一次 `set_pipeline` 校验失败，整帧提交作废（屏上就是黑的）。
//!
//! ```sh
//! cargo run -p gpui-pre-wgpu --example backdrop_glass_offscreen --features offscreen
//! ```
//!
//! 输出 `/tmp/backdrop_glass_offscreen.png`。没有 GPU 的环境会在 `new()` 处报错。

use gpui::{
    BackdropGlassPrimitive, Bounds, ContentMask, Corners, DevicePixels, Hsla, Point, Quad,
    ScaledPixels, Scene, Size,
};
use gpui_wgpu::WgpuHeadlessRenderer;

fn main() -> anyhow::Result<()> {
    let side = 512;
    let out = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "/tmp/backdrop_glass_offscreen.png".into());
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
    // 玻璃底下要有东西可糊：四条色带铺满整张。
    let colors = [gpui::red(), gpui::green(), gpui::blue(), gpui::yellow()];
    let stripe_width = side as f32 / colors.len() as f32;
    for (index, color) in colors.into_iter().enumerate() {
        let stripe = Bounds::new(
            Point {
                x: ScaledPixels(stripe_width * index as f32),
                y: ScaledPixels(0.0),
            },
            Size {
                width: ScaledPixels(stripe_width),
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

    // 中间一片玻璃，半透明白色 tint 压在它上面 —— 与产品里「玻璃 + tint quad」同一结构。
    let glass_bounds = Bounds::new(
        Point {
            x: ScaledPixels(bounds.size.width.0 * 0.25),
            y: ScaledPixels(bounds.size.height.0 * 0.25),
        },
        Size {
            width: ScaledPixels(bounds.size.width.0 * 0.5),
            height: ScaledPixels(bounds.size.height.0 * 0.5),
        },
    );
    scene.backdrop_glasses.push(BackdropGlassPrimitive {
        order: 1,
        pad: 0,
        bounds: glass_bounds,
        content_mask: ContentMask {
            bounds: glass_bounds,
        },
        corner_radii: Corners {
            top_left: ScaledPixels(24.0),
            top_right: ScaledPixels(24.0),
            bottom_right: ScaledPixels(24.0),
            bottom_left: ScaledPixels(24.0),
        },
        blur_radius: ScaledPixels(20.0),
        _pad: [0.0; 5],
    });
    scene.quads.push(Quad {
        order: 2,
        border_style: Default::default(),
        bounds: glass_bounds,
        content_mask: ContentMask {
            bounds: glass_bounds,
        },
        background: gpui::white().opacity(0.15).into(),
        border_color: Hsla::transparent_black(),
        corner_radii: Corners {
            top_left: ScaledPixels(24.0),
            top_right: ScaledPixels(24.0),
            bottom_right: ScaledPixels(24.0),
            bottom_left: ScaledPixels(24.0),
        },
        border_widths: Default::default(),
    });

    let mut renderer = WgpuHeadlessRenderer::new()?;
    let image = renderer.render_to_image(&scene, size)?;
    image.save(&out)?;
    println!("wrote {out}");
    Ok(())
}
