//! 页面卷曲的 GPU 资源与绘制。
//!
//! 这个子模块挂在 `wgpu_renderer` 下，能拿到父模块的私有字段与类型；对外只
//! 暴露 `PageCurlResources` 的三个入口：
//!
//! - `prepare`：从场景里取第一张网格，建顶点/索引缓冲、上传 uniform、准备阴影图；
//! - `encode_shadow`：把网格从光源方向烘进 shadow map；
//! - `encode_mesh_pass`：先铺新页投影，再画卷曲网格（带深度）。
//!
//! 页面捕获本身由父模块的 `record_scene` 负责：这里只提供捕获纹理的 view 与
//! bind group。

use super::*;
use gpui::PageCurlUniforms;

const PAGE_CURL_SHADERS: &str = include_str!("../page_curl.wgsl");

const SHADOW_MAP_SIZE: u32 = 1024;

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
    shadow_map_size: f32,
    apex_v: f32,
    finger_v: f32,
    // `shadow_map_size + 3*f32` 之后 mat4 正好 96 字节对齐；补到 96。
    _pad0: f32,
    light_view_proj: [[f32; 4]; 4],
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
            shadow_map_size: uniforms.shadow_map_size,
            apex_v: uniforms.apex_v,
            finger_v: uniforms.finger_v,
            _pad0: 0.0,
            light_view_proj,
        }
    }
}

pub(super) struct PageCurlPipelines {
    mesh: wgpu::RenderPipeline,
    shadow: wgpu::RenderPipeline,
    overlay: wgpu::RenderPipeline,
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
    let overlay_depth = wgpu::DepthStencilState {
        depth_write_enabled: Some(false),
        depth_compare: Some(wgpu::CompareFunction::Always),
        ..mesh_depth.clone()
    };

    let mesh = create(
        "page_curl_mesh",
        "vs_page_curl",
        "fs_page_curl",
        &[
            Some(&layouts.globals),
            Some(&layouts.page_curl_uniforms),
            Some(&layouts.texture),
            Some(&layouts.texture),
        ],
        target_format,
        Some(mesh_depth),
        Some(wgpu::BlendState::ALPHA_BLENDING),
    );

    let shadow = create(
        "page_curl_shadow",
        "vs_shadow",
        "fs_shadow",
        &[Some(&layouts.globals), Some(&layouts.page_curl_uniforms)],
        wgpu::TextureFormat::Rgba8Unorm,
        None,
        None,
    );

    let overlay = create(
        "page_curl_overlay",
        "vs_overlay",
        "fs_overlay",
        &[
            Some(&layouts.globals),
            Some(&layouts.page_curl_uniforms),
            Some(&layouts.texture),
            Some(&layouts.texture),
        ],
        target_format,
        Some(overlay_depth),
        Some(wgpu::BlendState::ALPHA_BLENDING),
    );

    PageCurlPipelines {
        mesh,
        shadow,
        overlay,
    }
}

struct CaptureTexture {
    _texture: wgpu::Texture,
    view: wgpu::TextureView,
    bind_group: wgpu::BindGroup,
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
    fallback_bind_group: wgpu::BindGroup,
    depth: Option<DepthTexture>,
    shadow: Option<ShadowTexture>,
    geometry: Option<GeometryCache>,
    frame: Option<FrameDraw>,
    target_format: wgpu::TextureFormat,
}

impl PageCurlResources {
    pub(super) fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
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
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            ..Default::default()
        });

        // 还没捕获页面时绑一张 1×1 透明纹理，避免整条管线因为缺绑定而跳过。
        let fallback_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("page_curl_fallback"),
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: target_format,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &fallback_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &[0u8; 4],
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(4),
                rows_per_image: Some(1),
            },
            wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
        );
        let fallback_view = fallback_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let fallback_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("page_curl_fallback_bind_group"),
            layout: &layouts.texture,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&fallback_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
        });

        Self {
            pipelines,
            uniform_buffer,
            uniform_bind_group,
            sampler,
            capture: None,
            fallback_bind_group,
            depth: None,
            shadow: None,
            geometry: None,
            frame: None,
            target_format,
        }
    }

    pub(super) fn invalidate_size_dependent(&mut self) {
        self.capture = None;
        self.depth = None;
        self.shadow = None;
        self.geometry = None;
        self.frame = None;
    }

    /// 从场景里取第一张网格，准备这一帧的顶点/索引缓冲、uniform 与阴影图。
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
        });
    }

    pub(super) fn capture_view(
        &mut self,
        device: &wgpu::Device,
        layouts: &WgpuBindGroupLayouts,
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
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("page_capture_bind_group"),
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
            self.capture = Some(CaptureTexture {
                _texture: texture,
                view,
                bind_group,
                size,
            });
        }
        &self.capture.as_ref().unwrap().view
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

    /// 把网格从光源方向渲染进 shadow map。必须在主 pass 之前调用。
    pub(super) fn encode_shadow(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        globals_bind_group: &wgpu::BindGroup,
    ) {
        let (Some(frame), Some(geometry), Some(shadow)) = (
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
            depth_stencil_attachment: None,
            ..Default::default()
        });
        pass.set_pipeline(&self.pipelines.shadow);
        pass.set_bind_group(0, globals_bind_group, &[]);
        pass.set_bind_group(1, &self.uniform_bind_group, &[]);
        pass.set_vertex_buffer(0, geometry.vertex_buffer.slice(..));
        pass.set_index_buffer(geometry.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
        pass.draw_indexed(0..frame.index_count, 0, 0..1);
    }

    /// 在新页上铺投影，再画卷曲网格。`target` 是主颜色附件。
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
        let Some(shadow) = self.shadow.as_ref() else {
            return;
        };
        let capture_bind_group = self
            .capture
            .as_ref()
            .map(|capture| &capture.bind_group)
            .unwrap_or(&self.fallback_bind_group);

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
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            ..Default::default()
        });
        if width > 0 && height > 0 {
            pass.set_scissor_rect(x, y, width, height);
        }

        for pipeline in [&self.pipelines.overlay, &self.pipelines.mesh] {
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, globals_bind_group, &[]);
            pass.set_bind_group(1, &self.uniform_bind_group, &[]);
            pass.set_bind_group(2, capture_bind_group, &[]);
            pass.set_bind_group(3, &shadow.bind_group, &[]);
            pass.set_vertex_buffer(0, geometry.vertex_buffer.slice(..));
            pass.set_index_buffer(geometry.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
            pass.draw_indexed(0..frame.index_count, 0, 0..1);
        }
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

fn light_view_proj(uniforms: &PageCurlUniforms) -> [[f32; 4]; 4] {
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
    let projection = ortho(
        -extent,
        extent,
        -extent,
        extent,
        0.1,
        distance * 2.0 + uniforms.radius * 4.0,
    );
    mul(projection, look_at(eye, center, up))
}
