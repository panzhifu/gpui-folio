//! folio 的仿真翻页网格。
//!
//! 这里只放 CPU 侧能单测的东西：页面网格、圆柱参数、光矩阵的输入。
//! GPU 侧的管线与 WGSL 在 `gpui-pre-wgpu` 的 `folio_curl` 模块里。
//!
//! 坐标约定与 gmpagecurl-ios 的纸模型一致：
//!
//! - 页面局部坐标 `(x, y)`，原点在页面左上角；
//! - 圆柱轴是页面平面内的一条有向直线，`n` 是它的单位法向；
//! - `d = n·P - d0`：
//!   - `d <= 0` 平摊；
//!   - `0 < d < πR` 卷在圆柱上；
//!   - `d >= πR` 翻到 `z = 2R` 的背面平面。
//!
//! 这个 crate 不依赖 gpui，测试可以直接跑。

#![forbid(unsafe_code)]

use bytemuck::{Pod, Zeroable};

/// 网格分辨率。96×96 对手机屏幕的折痕已经够细；再高是白花顶点带宽。
pub const GRID_X: u32 = 96;
pub const GRID_Y: u32 = 96;

/// 一条页面的网格顶点。位置是**页面局部坐标**，`uv` 是页面纹理坐标。
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct Vertex {
    pub position: [f32; 2],
    pub uv: [f32; 2],
}

/// 一次翻页的圆柱参数，按帧上传给 GPU。
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct CurlUniforms {
    /// 页面在窗口里的左上角（设备像素无关的逻辑像素）。
    pub page_origin: [f32; 2],
    /// 页面尺寸。
    pub page_size: [f32; 2],
    /// 窗口尺寸（纹理采样用）。
    pub window_size: [f32; 2],
    /// 圆柱轴：页面局部坐标下 `n·P = d0`。
    pub axis_normal: [f32; 2],
    pub axis_d: f32,
    /// 圆柱半径。
    pub radius: f32,
    /// 透视相机距离。
    pub camera_distance: f32,
    /// 背面纸色（线性 RGBA）。正面永远采样捕获到的页面纹理。
    pub back_color: [f32; 4],
    /// 光照方向（视线空间，指向光源）。
    pub light_dir: [f32; 4],
    /// 圆锥：`[锥顶的切向坐标, 自由边的切向坐标, 圆锥权重, 最小半径]`。
    ///
    /// `cone_weight = 0` 时整条退化成圆柱；权重由阅读层按折痕角度/锥顶距离算好。
    pub cone_a: [f32; 4],
    /// `[最大半径, 0, 0, 0]`。单独一格是为了保持 16 字节对齐。
    pub cone_b: [f32; 4],
}

/// 生成 `GRID_X × GRID_Y` 的页面网格（三角形列表）。
///
/// 顶点只生成一次；每帧变的只有 [`CurlUniforms`]。
pub fn build_page_mesh(page_width: f32, page_height: f32) -> (Vec<Vertex>, Vec<u32>) {
    let columns = GRID_X + 1;
    let rows = GRID_Y + 1;
    let mut vertices = Vec::with_capacity((columns * rows) as usize);
    for y in 0..rows {
        let v = y as f32 / GRID_Y as f32;
        for x in 0..columns {
            let u = x as f32 / GRID_X as f32;
            vertices.push(Vertex {
                position: [u * page_width, v * page_height],
                uv: [u, v],
            });
        }
    }

    let mut indices = Vec::with_capacity((GRID_X * GRID_Y * 6) as usize);
    for y in 0..GRID_Y {
        for x in 0..GRID_X {
            let top_left = y * columns + x;
            let top_right = top_left + 1;
            let bottom_left = top_left + columns;
            let bottom_right = bottom_left + 1;
            indices.extend_from_slice(&[
                top_left,
                bottom_left,
                top_right,
                top_right,
                bottom_left,
                bottom_right,
            ]);
        }
    }

    (vertices, indices)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_page_curl_shader_parses_and_validates() {
        let source = include_str!("../../gpui-pre-wgpu/src/page_curl.wgsl");
        let module = naga::front::wgsl::parse_str(source).expect("WGSL 解析失败");
        let mut validator = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::default(),
        );
        validator.validate(&module).expect("WGSL 校验失败");
    }

    #[test]
    fn the_mesh_covers_the_whole_page_once() {
        let (vertices, indices) = build_page_mesh(390.0, 844.0);
        assert_eq!(vertices.len(), ((GRID_X + 1) * (GRID_Y + 1)) as usize);
        assert_eq!(indices.len(), (GRID_X * GRID_Y * 6) as usize);
        assert!(
            vertices
                .iter()
                .all(|v| v.position[0] >= 0.0 && v.position[0] <= 390.0)
        );
        assert!(
            vertices
                .iter()
                .all(|v| v.position[1] >= 0.0 && v.position[1] <= 844.0)
        );
        assert!(vertices.iter().all(|v| v.uv[0] >= 0.0 && v.uv[0] <= 1.0));
        assert!(vertices.iter().all(|v| v.uv[1] >= 0.0 && v.uv[1] <= 1.0));
        assert!(indices.iter().all(|&i| (i as usize) < vertices.len()));
    }
}
