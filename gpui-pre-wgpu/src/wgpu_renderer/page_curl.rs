//! 页面卷曲的 GPU 资源与绘制。
//!
//! 这个子模块挂在 `wgpu_renderer` 下，能拿到父模块的私有字段与类型；对外只
//! 暴露 `PageCurlResources` 的两个入口：
//!
//! - `prepare`：从场景里取第一张网格，建顶点/索引缓冲、上传这一帧的 uniform；
//! - `encode_mesh_pass`：单独开一趟带深度的 pass 画卷曲网格。
//!
//! 页面捕获本身由父模块的 `record_scene` 负责：这里只提供捕获纹理的 view 与
//! bind group。

use super::*;
use gpui::PageCurlUniforms;

const PAGE_CURL_SHADERS: &str = include_str!("../page_curl.wgsl");
/// 光源方向的正交 shadow map 边长。手机上一张 1024² 的 RGBA8 约 4 MB；再大换来的
/// 是 3×3 PCF 里一个纹素更细，肉眼分不出。
const SHADOW_MAP_SIZE: u32 = 1024;

/// 与 `page_curl.wgsl` 的 `CurlUniforms` 一一对上。尺寸与那三格关键偏移都要一致：错一位**不
/// 报错**，只是阴影矩阵与两张捕获纹理的采样原点各读到隔壁那格的去向。
const _: () = {
    use std::mem::{offset_of, size_of};
    assert!(size_of::<GpuCurlUniforms>() == 240);
    assert!(offset_of!(GpuCurlUniforms, shadow_map_size) == 128);
    assert!(offset_of!(GpuCurlUniforms, light_view_proj) == 160);
    assert!(offset_of!(GpuCurlUniforms, back_capture_origin) == 224);
};

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuCurlUniforms {
    page_origin: [f32; 2],
    page_size: [f32; 2],
    window_size: [f32; 2],
    axis_normal: [f32; 2],
    axis_d: f32,
    radius: f32,
    camera_distance: f32,
    scale_factor: f32,
    back_color: [f32; 4],
    light_dir: [f32; 4],
    cone_a: [f32; 4],
    cone_b: [f32; 4],
    capture_origin: [f32; 2],
    _padding: [f32; 2],
    shadow_map_size: f32,
    /// WGSL 那边 `vec3` 与 `mat4x4` 都按 **16** 字节对齐，而 Rust 这边全是 `f32` 数组（按 4
    /// 排）。这两格是把 shader 那侧的隐式空洞显式补回来：少了它们，`_pad1` 起往后每一格都
    /// 错位 12–16 字节，而 Vulkan 会直接以「shader 要 240、布局只给 224」拒掉整条
    /// `page_curl_mesh` 管线 —— 真机上读起来是应用卡在启动画面，一行渲染都出不去。
    _pad0: [f32; 3],
    _pad1: [f32; 3],
    _pad1a: f32,
    light_view_proj: [[f32; 4]; 4],
    back_capture_origin: [f32; 2],
    _pad2: [f32; 2],
}

impl GpuCurlUniforms {
    fn from_parts(uniforms: &PageCurlUniforms, light_view_proj: [[f32; 4]; 4]) -> Self {
        Self {
            page_origin: uniforms.page_origin,
            page_size: uniforms.page_size,
            window_size: uniforms.window_size,
            axis_normal: uniforms.axis_normal,
            axis_d: uniforms.axis_d,
            radius: uniforms.radius,
            camera_distance: uniforms.camera_distance,
            scale_factor: uniforms.scale_factor,
            back_color: uniforms.back_color,
            light_dir: uniforms.light_dir,
            cone_a: uniforms.cone_a,
            cone_b: uniforms.cone_b,
            capture_origin: uniforms.capture_origin,
            _padding: [0.0; 2],
            shadow_map_size: SHADOW_MAP_SIZE as f32,
            _pad0: [0.0; 3],
            _pad1: [0.0; 3],
            _pad1a: 0.0,
            light_view_proj,
            back_capture_origin: uniforms.back_capture_origin,
            _pad2: [0.0; 2],
        }
    }
}

pub(super) struct PageCurlPipelines {
    mesh: wgpu::RenderPipeline,
    /// 卷起那一片纸落在下面那页上的投影：同一套顶点与索引，另一条 pipeline。
    shadow: wgpu::RenderPipeline,
    /// 正面与背面两张捕获纹理共用的那一组布局：一组四个绑定（两张纹理各带一个采样器）。
    /// 单独存是因为 `WgpuBindGroupLayouts.texture` 只装一对纹理+采样器，装不下两张；
    /// 而合组是硬要求 —— 适配器只报 4 组时（Adreno 的 Vulkan 驱动实测），五组的
    /// 管线布局让 `create_shader_module` 当场验证失败，整个 Vulkan 设备被拒。
    capture_pair_layout: wgpu::BindGroupLayout,
}

pub(super) fn create_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("page_curl_uniforms_layout"),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: NonZeroU64::new(std::mem::size_of::<GpuCurlUniforms>() as u64),
            },
            count: None,
        }],
    })
}

pub(super) fn create_pipelines(
    device: &wgpu::Device,
    layouts: &WgpuBindGroupLayouts,
    target_format: wgpu::TextureFormat,
) -> PageCurlPipelines {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("page_curl_shaders"),
        source: wgpu::ShaderSource::Wgsl(PAGE_CURL_SHADERS.into()),
    });

    let vertex_buffers = [wgpu::VertexBufferLayout {
        array_stride: std::mem::size_of::<gpui::PageCurlVertex>() as u64,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &[
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32x2,
                offset: 0,
                shader_location: 0,
            },
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32x2,
                offset: 8,
                shader_location: 1,
            },
        ],
    }];

    let create = |label: &str,
                  vs: &str,
                  fs: &str,
                  groups: &[Option<&wgpu::BindGroupLayout>],
                  color_format: wgpu::TextureFormat,
                  depth: Option<wgpu::DepthStencilState>,
                  blend: Option<wgpu::BlendState>| {
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(label),
            bind_group_layouts: groups,
            immediate_size: 0,
        });
        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some(label),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some(vs),
                buffers: &vertex_buffers,
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some(fs),
                targets: &[Some(wgpu::ColorTargetState {
                    format: color_format,
                    blend,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: depth,
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview_mask: None,
            cache: None,
        })
    };

    let mesh_depth = wgpu::DepthStencilState {
        format: wgpu::TextureFormat::Depth32Float,
        depth_write_enabled: Some(true),
        depth_compare: Some(wgpu::CompareFunction::LessEqual),
        stencil: wgpu::StencilState::default(),
        bias: wgpu::DepthBiasState::default(),
    };

    // 正面捕获与背面捕获合用 group 2（四个绑定），shadow map 独占 group 3：
    // 连同 globals 与 uniforms 一共 4 组，贴着最抠门的适配器（max_bind_groups = 4，
    // Adreno 的 Vulkan 驱动实测）的天花板。原先 back 单独占 group 4，第五组一超，
    // 着色器模块验证失败，Vulkan 整个设备被拒、全应用退到 GL。
    let capture_pair_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("page_curl_capture_pair_layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
        ],
    });

    // 片元给回的 alpha 恒为 1，这一趟本来就是不透明的：开着混合只是让整页多一次读改写。
    let mesh = create(
        "page_curl_mesh",
        "vs_page_curl",
        "fs_page_curl",
        &[
            Some(&layouts.globals),
            Some(&layouts.page_curl_uniforms),
            // group 2：正面与背面两张捕获纹理并作一组；group 3：光源方向的 shadow map。
            Some(&capture_pair_layout),
            Some(&layouts.texture),
        ],
        target_format,
        Some(mesh_depth),
        None,
    );

    // 深度趟：从光源方向把同一张网格渲进 Rgba8 的 `.r`（与旧的 3ebf917 同一口径），
    // 主 pass 的片元再做 3×3 PCF。它不写颜色目标以外的任何东西，也不做混合。
    let shadow = create(
        "page_curl_shadow",
        "vs_shadow",
        "fs_shadow",
        &[Some(&layouts.globals), Some(&layouts.page_curl_uniforms)],
        wgpu::TextureFormat::Rgba8Unorm,
        Some(wgpu::DepthStencilState {
            format: wgpu::TextureFormat::Depth32Float,
            depth_write_enabled: Some(true),
            depth_compare: Some(wgpu::CompareFunction::LessEqual),
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        }),
        None,
    );

    PageCurlPipelines {
        mesh,
        shadow,
        capture_pair_layout,
    }
}

struct CaptureTexture {
    _texture: wgpu::Texture,
    view: wgpu::TextureView,
    size: Size<DevicePixels>,
}

struct DepthTexture {
    _texture: wgpu::Texture,
    view: wgpu::TextureView,
    size: Size<DevicePixels>,
}

struct ShadowTexture {
    _texture: wgpu::Texture,
    view: wgpu::TextureView,
    bind_group: wgpu::BindGroup,
    /// 深度测试用的那一张：shadow map 的颜色目标是 8 位，光栅顺序又不保证从近到远，
    /// 没有深度测试时后画的三角形会盖掉先画的那个深度 —— 图里存的是「最后一个」而不是
    /// 「最近的一个」。有了它，颜色目标里留下的才是离光源最近的表面。
    _depth: wgpu::Texture,
    depth_view: wgpu::TextureView,
}

struct FrameDraw {
    index_count: u32,
    /// 卷曲网格的页框（不是 canvas 的整块 content mask）：剪到版心，
    /// 避免卷起来那一截画到页眉/页脚上。
    page_bounds: Bounds<ScaledPixels>,
}

/// 页面网格的顶点/索引只由版心尺寸和网格分辨率决定，翻页进度只改 uniform。
/// 按 `(page_size, vertex_count, index_count)` 缓存，翻页期间不再每帧 create_buffer。
struct GeometryCache {
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    index_count: u32,
    vertex_count: usize,
    page_size: [f32; 2],
}

pub(super) struct PageCurlResources {
    pipelines: PageCurlPipelines,
    uniform_buffer: wgpu::Buffer,
    uniform_bind_group: wgpu::BindGroup,
    sampler: wgpu::Sampler,
    capture: Option<CaptureTexture>,
    back_capture: Option<CaptureTexture>,
    /// 正面与背面两张捕获纹理并成的**一张** bind group（group 2，四绑定）。两张纹理
    /// 谁（重）建了它都得跟着重建 —— 见 [`PageCurlResources::refresh_capture_pair`]。
    capture_pair_bind_group: Option<wgpu::BindGroup>,
    depth: Option<DepthTexture>,
    shadow: Option<ShadowTexture>,
    geometry: Option<GeometryCache>,
    frame: Option<FrameDraw>,
    target_format: wgpu::TextureFormat,
}

impl PageCurlResources {
    pub(super) fn new(
        device: &wgpu::Device,
        layouts: &WgpuBindGroupLayouts,
        pipelines: PageCurlPipelines,
        target_format: wgpu::TextureFormat,
    ) -> Self {
        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("page_curl_uniforms"),
            size: std::mem::size_of::<GpuCurlUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let uniform_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("page_curl_uniforms_bind_group"),
            layout: &layouts.page_curl_uniforms,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
        });

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("page_curl_sampler"),
            // 捕获纹理与卷曲网格是 1:1 的设备像素。最近邻既保住正文的锐度，也避免
            // 线性过滤在矩形外圈与清屏色之间揉出一条半透明灰边（参照实现同样用 NEAREST）。
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            ..Default::default()
        });

        Self {
            pipelines,
            uniform_buffer,
            uniform_bind_group,
            sampler,
            capture: None,
            back_capture: None,
            capture_pair_bind_group: None,
            depth: None,
            shadow: None,
            geometry: None,
            frame: None,
            target_format,
        }
    }

    pub(super) fn invalidate_size_dependent(&mut self) {
        self.capture = None;
        self.back_capture = None;
        self.capture_pair_bind_group = None;
        self.depth = None;
        self.geometry = None;
        self.frame = None;
    }

    /// 从场景里取第一张网格，准备这一帧的顶点/索引缓冲与 uniform。
    pub(super) fn prepare(
        &mut self,
        scene: &Scene,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        layouts: &WgpuBindGroupLayouts,
        size: Size<DevicePixels>,
    ) -> Result<()> {
        let Some(mesh) = scene.page_curl_meshes.first() else {
            self.frame = None;
            return Ok(());
        };
        let vertex_count = mesh.vertices.len();
        let index_count = mesh.indices.len() as u32;
        let cache_hit = self.geometry.as_ref().is_some_and(|geometry| {
            geometry.page_size == mesh.uniforms.page_size
                && geometry.vertex_count == vertex_count
                && geometry.index_count == index_count
        });
        if !cache_hit {
            let vertex_bytes = bytemuck::cast_slice(mesh.vertices.as_ref());
            let index_bytes = bytemuck::cast_slice(mesh.indices.as_ref());
            let vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("page_curl_vertices"),
                size: vertex_bytes.len().max(1) as u64,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            queue.write_buffer(&vertex_buffer, 0, vertex_bytes);

            let index_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("page_curl_indices"),
                size: index_bytes.len().max(4) as u64,
                usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            queue.write_buffer(&index_buffer, 0, index_bytes);

            self.geometry = Some(GeometryCache {
                vertex_buffer,
                index_buffer,
                index_count,
                vertex_count,
                page_size: mesh.uniforms.page_size,
            });
        }

        let light_view_proj = light_view_proj(&mesh.uniforms);
        let uniforms = GpuCurlUniforms::from_parts(&mesh.uniforms, light_view_proj);
        queue.write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&uniforms));

        self.frame = Some(FrameDraw {
            index_count,
            page_bounds: mesh.bounds,
        });

        self.ensure_shadow(device, layouts);
        self.ensure_depth(device, size);
        Ok(())
    }

    fn ensure_shadow(&mut self, device: &wgpu::Device, layouts: &WgpuBindGroupLayouts) {
        if self.shadow.is_some() {
            return;
        }
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("page_curl_shadow_map"),
            size: wgpu::Extent3d {
                width: SHADOW_MAP_SIZE,
                height: SHADOW_MAP_SIZE,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let depth = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("page_curl_shadow_depth"),
            size: wgpu::Extent3d {
                width: SHADOW_MAP_SIZE,
                height: SHADOW_MAP_SIZE,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let depth_view = depth.create_view(&wgpu::TextureViewDescriptor::default());
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("page_curl_shadow_bind_group"),
            layout: &layouts.texture,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
            ],
        });
        self.shadow = Some(ShadowTexture {
            _texture: texture,
            view,
            bind_group,
            _depth: depth,
            depth_view,
        });
    }

    /// 把网格从光源方向渲染进 shadow map。必须在主 pass 之前调用。
    pub(super) fn encode_shadow(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        globals_bind_group: &wgpu::BindGroup,
    ) {
        let (Some(_frame), Some(geometry), Some(shadow)) = (
            self.frame.as_ref(),
            self.geometry.as_ref(),
            self.shadow.as_ref(),
        ) else {
            return;
        };
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("page_curl_shadow_pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &shadow.view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::WHITE),
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &shadow.depth_view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
                    // 这一趟的深度只在它自己内部比先后，出去没人再读。
                    store: wgpu::StoreOp::Discard,
                }),
                stencil_ops: None,
            }),
            ..Default::default()
        });
        pass.set_pipeline(&self.pipelines.shadow);
        pass.set_bind_group(0, globals_bind_group, &[]);
        pass.set_bind_group(1, &self.uniform_bind_group, &[]);
        pass.set_vertex_buffer(0, geometry.vertex_buffer.slice(..));
        pass.set_index_buffer(geometry.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
        pass.draw_indexed(0..geometry.index_count, 0, 0..1);
    }

    pub(super) fn capture_view(
        &mut self,
        device: &wgpu::Device,
        _layouts: &WgpuBindGroupLayouts,
        size: Size<DevicePixels>,
    ) -> &wgpu::TextureView {
        let needs_new = self
            .capture
            .as_ref()
            .map(|capture| capture.size != size)
            .unwrap_or(true);
        if needs_new {
            let texture = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("page_capture"),
                size: wgpu::Extent3d {
                    width: size.width.0.max(1) as u32,
                    height: size.height.0.max(1) as u32,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: self.target_format,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::COPY_SRC,
                view_formats: &[],
            });
            let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
            self.capture = Some(CaptureTexture {
                _texture: texture,
                view,
                size,
            });
            self.refresh_capture_pair(device);
        }
        &self.capture.as_ref().unwrap().view
    }

    /// 背面那一页的捕获纹理，尺寸/格式与正面一致。
    pub(super) fn back_capture_view(
        &mut self,
        device: &wgpu::Device,
        _layouts: &WgpuBindGroupLayouts,
        size: Size<DevicePixels>,
    ) -> &wgpu::TextureView {
        let needs_new = self
            .back_capture
            .as_ref()
            .map(|capture| capture.size != size)
            .unwrap_or(true);
        if needs_new {
            let texture = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("page_capture_back"),
                size: wgpu::Extent3d {
                    width: size.width.0.max(1) as u32,
                    height: size.height.0.max(1) as u32,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: self.target_format,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::COPY_SRC,
                view_formats: &[],
            });
            let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
            self.back_capture = Some(CaptureTexture {
                _texture: texture,
                view,
                size,
            });
            self.refresh_capture_pair(device);
        }
        &self.back_capture.as_ref().unwrap().view
    }

    /// 正面与背面两张捕获纹理并成一张 bind group（group 2 的四个绑定）。
    ///
    /// 在任一张捕获纹理（重）建之后调：两张纹理都到位才建得出来，所以第一帧只建出
    /// 正面那一半时先欠着，背面到位的那次调用把它补上。谁重建纹理谁就得重建这张组 ——
    /// 旧的 view 一 drop，组里攥着的引用就悬了。尺寸没变时一个字节都不动。
    fn refresh_capture_pair(&mut self, device: &wgpu::Device) {
        let (Some(capture), Some(back)) = (&self.capture, &self.back_capture) else {
            return;
        };
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("page_capture_pair_bind_group"),
            layout: &self.pipelines.capture_pair_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&capture.view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&back.view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
            ],
        });
        self.capture_pair_bind_group = Some(bind_group);
    }

    fn ensure_depth(&mut self, device: &wgpu::Device, size: Size<DevicePixels>) {
        if self.depth.as_ref().is_some_and(|depth| depth.size == size) {
            return;
        }
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("page_curl_depth"),
            size: wgpu::Extent3d {
                width: size.width.0.max(1) as u32,
                height: size.height.0.max(1) as u32,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        self.depth = Some(DepthTexture {
            _texture: texture,
            view,
            size,
        });
    }

    /// 画卷曲网格。`target` 是主颜色附件。
    pub(super) fn encode_mesh_pass(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        target: &wgpu::TextureView,
        size: Size<DevicePixels>,
        globals_bind_group: &wgpu::BindGroup,
    ) {
        let Some(frame) = self.frame.as_ref() else {
            return;
        };
        let Some(geometry) = self.geometry.as_ref() else {
            return;
        };
        let Some(depth) = self.depth.as_ref() else {
            return;
        };
        // 没有捕获纹理就**不画这一趟**。从前这里绑一张 1×1 的透明黑兜底，理由是「别让管线因为
        // 缺绑定而跳过」，而它正是那条黑的来源：尺寸一变（`invalidate_size_dependent` 把纹理清了）
        // 或者翻页赶在烘纹理之前，整片纸就读成黑。跳过只是「这一趟没卷起来」—— 下面那张页本来
        // 就在那儿，读起来是一次普通的落定；绑黑读起来是屏幕上一条黑带。
        //
        // 合组那张 bind group 只在**两张**捕获纹理都到位时才建得出来（见
        // `refresh_capture_pair`），缺它就是某一半还没烘出来，与缺单张同一条退路。
        let Some(capture_pair) = self.capture_pair_bind_group.as_ref() else {
            log::debug!("page_curl: 缺捕获纹理，这一帧不画卷曲网格");
            return;
        };

        let b = frame.page_bounds;
        let x = b.origin.x.0.max(0.0) as u32;
        let y = b.origin.y.0.max(0.0) as u32;
        let right = (b.origin.x.0 + b.size.width.0)
            .min(size.width.0 as f32)
            .max(0.0) as u32;
        let bottom = (b.origin.y.0 + b.size.height.0)
            .min(size.height.0 as f32)
            .max(0.0) as u32;
        let width = right.saturating_sub(x);
        let height = bottom.saturating_sub(y);

        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("page_curl_mesh_pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &depth.view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
                    // 这一趟的深度只在网格自己之间分先后，出去没人再读。留着一个窗口大的
                    // 深度缓冲写回内存，换来的是一次从来没人读的读回。
                    store: wgpu::StoreOp::Discard,
                }),
                stencil_ops: None,
            }),
            ..Default::default()
        });
        if width > 0 && height > 0 {
            pass.set_scissor_rect(x, y, width, height);
        }

        // shadow map 在 `encode_shadow` 那一趟里已经烘好；这里只画卷面，片元自己采样
        // 投影（平摊段也按同一张图压暗），不再有第二张「假投影网格」。
        let Some(shadow) = self.shadow.as_ref() else {
            log::debug!("page_curl: 缺 shadow map，这一帧不画卷曲网格");
            return;
        };
        pass.set_pipeline(&self.pipelines.mesh);
        pass.set_bind_group(0, globals_bind_group, &[]);
        pass.set_bind_group(1, &self.uniform_bind_group, &[]);
        pass.set_bind_group(2, capture_pair, &[]);
        pass.set_bind_group(3, &shadow.bind_group, &[]);
        pass.set_vertex_buffer(0, geometry.vertex_buffer.slice(..));
        pass.set_index_buffer(geometry.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
        pass.draw_indexed(0..frame.index_count, 0, 0..1);
    }
}

fn normalize(v: [f32; 3]) -> [f32; 3] {
    let length = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt().max(1e-6);
    [v[0] / length, v[1] / length, v[2] / length]
}

fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

type Mat4 = [[f32; 4]; 4];

fn mul(a: Mat4, b: Mat4) -> Mat4 {
    let mut out = [[0.0f32; 4]; 4];
    for column in 0..4 {
        for row in 0..4 {
            out[column][row] = (0..4).map(|k| a[k][row] * b[column][k]).sum();
        }
    }
    out
}

fn ortho(left: f32, right: f32, bottom: f32, top: f32, near: f32, far: f32) -> Mat4 {
    [
        [2.0 / (right - left), 0.0, 0.0, 0.0],
        [0.0, 2.0 / (top - bottom), 0.0, 0.0],
        [0.0, 0.0, 1.0 / (far - near), 0.0],
        [
            -(right + left) / (right - left),
            -(top + bottom) / (top - bottom),
            -near / (far - near),
            1.0,
        ],
    ]
}

fn look_at(eye: [f32; 3], center: [f32; 3], up: [f32; 3]) -> Mat4 {
    let forward = normalize([center[0] - eye[0], center[1] - eye[1], center[2] - eye[2]]);
    let right = normalize(cross(forward, up));
    let up = cross(right, forward);
    [
        [right[0], up[0], -forward[0], 0.0],
        [right[1], up[1], -forward[1], 0.0],
        [right[2], up[2], -forward[2], 0.0],
        [-dot(right, eye), -dot(up, eye), dot(forward, eye), 1.0],
    ]
}

fn light_view_proj(uniforms: &PageCurlUniforms) -> Mat4 {
    let direction = normalize([
        uniforms.light_dir[0],
        uniforms.light_dir[1],
        uniforms.light_dir[2],
    ]);
    let center = [
        uniforms.page_origin[0] + uniforms.page_size[0] * 0.5,
        uniforms.page_origin[1] + uniforms.page_size[1] * 0.5,
        0.0,
    ];
    let diagonal = (uniforms.page_size[0] * uniforms.page_size[0]
        + uniforms.page_size[1] * uniforms.page_size[1])
        .sqrt()
        .max(1.0);
    let distance = diagonal * 2.0;
    let eye = [
        center[0] + direction[0] * distance,
        center[1] + direction[1] * distance,
        center[2] + direction[2] * distance,
    ];
    let up = if direction[1].abs() > 0.95 {
        [0.0, 0.0, 1.0]
    } else {
        [0.0, 1.0, 0.0]
    };
    let extent = diagonal * 0.65 + uniforms.radius * 2.5 + 8.0;
    // 视图空间的 z 在标准 look-at 下是**负的**（点在相机前方），而 `ortho` 把 `[near, far]`
    // 线性映到 `[0, 1]`。照正数代进去，整页 z ≈ −distance 落在近平面之外，shadow pass 的
    // 三角形全被裁掉、图里只剩清屏色，主片那边 `ndc.z > stored + bias` 永远为假 —— 真机上
    // 就是「翻页一点阴影都没有」。两端取负，`[-0.1, -(2d+4R)]` 才是这一段正确的深度区间。
    let projection = ortho(
        -extent,
        extent,
        -extent,
        extent,
        -0.1,
        -(distance * 2.0 + uniforms.radius * 4.0),
    );
    mul(projection, look_at(eye, center, up))
}
