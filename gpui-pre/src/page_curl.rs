//! 页面卷曲网格的公共数据结构。
//!
//! 这里只描述「要画什么」，不含任何 wgpu 类型：`gpui-pre` 负责把网格与
//! uniform 带进场景，`gpui-pre-wgpu` 负责建管线和上传。这样 `gpui-pre`
//! 不依赖渲染后端，测试与无 GPU 构建照常。
//!
//! 顶点是**页面局部逻辑像素**，`uv` 是页面纹理坐标；Windows 在
//! [`paint_page_curl_mesh`](crate::Window::paint_page_curl_mesh) 里把
//! 长度字段统一乘 `scale_factor`，着色器不再猜 DPI。
//!
//! 捕获纹理铺的是**整窗大小**、只有请求那一块被写过，所以采样的 `uv` 是窗口归一的：
//! `(page_origin + 局部坐标) / window_size`。这条等式要求调用方给的 `page_origin` 与
//! `page_size` 就是 [`capture_page_region`](crate::Window::capture_page_region) 那块矩形
//! 本身 —— 差一个页边距，卷起来那一片就采到横移的页面，读成「纸上的字对不齐」。

use std::sync::Arc;

use bytemuck::{Pod, Zeroable};

use crate::geometry::Bounds;
use crate::scene::DrawOrder;
use crate::{ContentMask, Pixels, ScaledPixels};
/// 卷曲网格的顶点。布局与 WGSL 的 `VertexInput` 一一对应。
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct PageCurlVertex {
    /// 页面局部坐标（逻辑像素），原点在页面左上角。
    pub position: [f32; 2],
    /// 页面纹理坐标，0..1。
    pub uv: [f32; 2],
}

/// 一次卷曲的 GPU uniform。
///
/// 长度字段在进入场景前已经乘过 `scale_factor`，单位与 `globals.viewport_size`
/// 相同（设备像素）。顶点的 `position` 仍是逻辑像素，着色器用 `scale_factor`
/// 自己乘 —— 这样同一份顶点缓冲可以在不同 DPI 的窗口里复用。
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct PageCurlUniforms {
    /// 页面左上角在窗口里的位置（设备像素）。
    pub page_origin: [f32; 2],
    /// 页面尺寸（设备像素）。
    pub page_size: [f32; 2],
    /// 窗口尺寸（设备像素）。
    pub window_size: [f32; 2],
    /// 圆柱轴法向（页面局部坐标，单位向量）。
    pub axis_normal: [f32; 2],
    /// 圆柱轴在页面局部坐标下的有符号距离：`dot(axis_normal, p) = axis_d`。
    pub axis_d: f32,
    /// 圆柱半径（设备像素）。
    pub radius: f32,
    /// 透视相机距离（设备像素）。
    pub camera_distance: f32,
    /// 顶点位置从逻辑像素到设备像素的缩放。
    pub scale_factor: f32,
    /// 背面纸色（线性 RGBA）。正面永远采样页面捕获纹理。
    pub back_color: [f32; 4],
    /// 光照方向（页面/视线空间，指向光源）。
    pub light_dir: [f32; 4],
    /// 圆锥：`[锥顶切向坐标, 自由边切向坐标, 权重, 最小半径]`（设备像素，权重无量纲）。
    pub cone_a: [f32; 4],
    /// `[最大半径, 0, 0, 0]`。
    pub cone_b: [f32; 4],
    /// **烘那张纹理那一刻**页面左上角在窗口里的位置（设备像素）。
    ///
    /// 它与 [`Self::page_origin`] 分开是有原因的：那张纹理里被写过的只有烘的那一刻那一块矩形，
    /// 而网格取 uv 按「原点 + 局部坐标」归一。阅读层在动（进出书的 280 ms 视差滑动、面板位移、
    /// 尺寸变化）时两者就不是同一个数，差出来的那一截采到未写区域 —— 真机量到的是左缘一条
    /// 378 设备像素的纯黑带（烘的那一刻栏还偏着 134.4 逻辑像素）。几何照**本帧**的 `page_origin`
    /// 摆，采样照**烘时**的这里给，两类错位一起消失。
    pub capture_origin: [f32; 2],
    /// 背面捕获纹理（下一跨页的另一半）里，那张纸左上角在窗口里的位置。
    pub back_capture_origin: [f32; 2],
}

/// 调用方交给 [`Window::paint_page_curl_mesh`](crate::Window::paint_page_curl_mesh)
/// 的网格。`bounds` 是窗口坐标下的裁剪/剔除框（要盖住卷起来的高度与投影），
/// `page_bounds` 是页面本身在窗口里的位置。
#[allow(missing_docs)]
#[derive(Clone, Debug)]
pub struct PageCurlMesh {
    pub bounds: Bounds<Pixels>,
    pub page_bounds: Bounds<Pixels>,
    pub vertices: Arc<[PageCurlVertex]>,
    pub indices: Arc<[u32]>,
    pub uniforms: PageCurlUniforms,
}

/// 场景里的那一份：所有长度已经缩放成设备像素，`order` / `content_mask`
/// 与其他 primitive 同一套。
#[allow(missing_docs)]
#[derive(Clone, Debug)]
pub struct PageCurlMeshPrimitive {
    pub order: DrawOrder,
    pub bounds: Bounds<ScaledPixels>,
    pub content_mask: ContentMask<ScaledPixels>,
    pub page_bounds: Bounds<ScaledPixels>,
    pub vertices: Arc<[PageCurlVertex]>,
    pub indices: Arc<[u32]>,
    pub uniforms: PageCurlUniforms,
}

impl From<PageCurlMeshPrimitive> for crate::Primitive {
    fn from(mesh: PageCurlMeshPrimitive) -> Self {
        crate::Primitive::PageCurlMesh(mesh)
    }
}
