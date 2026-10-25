//! 背景模糊（毛玻璃）的那一片表面。
//!
//! 玻璃本身**只画底下的模糊像**，不画颜色：半透明的 tint 仍然由普通的背景 quad 压在它
//! 上面。这样「模糊」与「 tint」两件事各自可调，而玻璃这一层不需要重复 quad 已经做过的
//! 渐变、描边与圆角裁剪。
//!
//! 与 [`crate::PageCurlMeshPrimitive`] 同一套排法：它是场景里独立的一列 primitive，
//! 有自己的 `order`，渲染器在轮到它的那一处把主 pass 关掉、补一趟模糊、再以 `Load` 重开。
//! 所以「谁在玻璃底下」由元素树的绘制顺序决定，而不是由渲染器猜。
//!
//! 只有 wgpu 那条渲染路径（Android 与 Linux 桌面）真做模糊。Metal 与 DirectX 的渲染器
//! 认得这个批次但跳过不画，那两端的玻璃表面因此只剩 tint —— 与加这个 primitive 之前
//! 逐像素相同。

use crate::{Bounds, ContentMask, Corners, DrawOrder, Pixels, ScaledPixels};

/// 调用方交给 [`crate::Window::paint_backdrop_glass`] 的那一片玻璃。长度是逻辑像素，
/// 缩放由 `paint_backdrop_glass` 统一做。
#[allow(missing_docs)]
#[derive(Clone, Copy, Debug, Default)]
pub struct BackdropGlass {
    pub bounds: Bounds<Pixels>,
    /// 与压在上面的那张 tint quad **同一个圆角**：两边不一致就会在角外露出一圈没被
    /// 裁掉的模糊矩形，在浅底上是一条脏边。
    pub corner_radii: Corners<Pixels>,
    /// 模糊半径（逻辑像素）。高斯 σ 取它的三分之一。
    pub blur_radius: Pixels,
}

/// 场景里的那一份：长度已缩放成设备像素。
///
/// 这一列 primitive 是**逐字节**送进 instance buffer 的（渲染器的 `write_instance_binding`
/// 把 `&[Self]` 直接转成 `&[u8]`），所以必须 `#[repr(C)]`、不许有编译器插的空洞，且尺寸
/// 要同时满足 WGSL 那边 `array<BackdropGlass>` 的 stride 与对齐。`pad` 与 `_pad` 就是
/// 为此存在的，不是占位好看。
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[allow(missing_docs)]
pub struct BackdropGlassPrimitive {
    pub order: DrawOrder,
    pub pad: u32, // align to 8 bytes
    pub bounds: Bounds<ScaledPixels>,
    pub content_mask: ContentMask<ScaledPixels>,
    pub corner_radii: Corners<ScaledPixels>,
    /// 模糊半径，设备像素。
    pub blur_radius: ScaledPixels,
    /// 补到 80 字节：`order`+`pad` 8、`bounds` 16、`content_mask` 16、`corner_radii` 16、
    /// `blur_radius` 4，到这里是 60。WGSL 那边这个结构含 `vec2<f32>`（`Bounds`），对齐是
    /// 8，于是 shader 侧的 stride 会向上取整到 64，而 Rust 侧是 60 —— 两边差 4 字节，
    /// 每一条玻璃都会从隔壁那格读起。补满 80（同时是 8 与 16 的倍数）把这件事钉死。
    pub _pad: [f32; 5],
}

// 布局错了不会在屏上看出来，只会让每一条玻璃读到隔壁那格的去向，所以钉在编译期。
const _: () = {
    use std::mem::size_of;
    assert!(size_of::<BackdropGlassPrimitive>() == 80);
};

impl From<BackdropGlassPrimitive> for crate::Primitive {
    fn from(glass: BackdropGlassPrimitive) -> Self {
        crate::Primitive::BackdropGlass(glass)
    }
}
