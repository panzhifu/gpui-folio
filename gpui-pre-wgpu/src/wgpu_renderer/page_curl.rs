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
    _pad1: [f32; 3],
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
            cone_a: uniforms.cone_a,
            cone_b: uniforms.cone_b,
            capture_origin: uniforms.capture_origin,
            _padding: [0.0; 2],
            shadow_map_size: SHADOW_MAP_SIZE as f32,
            _pad1: [0.0; 3],
            light_view_proj,
        }
    }
}

pub(super) struct PageCurlPipelines {
    mesh: wgpu::RenderPipeline,
    /// 卷起那一片纸落在下面那页上的投影：同一套顶点与索引，另一条 pipeline。
    shadow: wgpu::RenderPipeline,
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

    // 片元给回的 alpha 恒为 1，这一趟本来就是不透明的：开着混合只是让整页多一次读改写。
    let mesh = create(
        "page_curl_mesh",
        "vs_page_curl",
        "fs_page_curl",
        &[
            Some(&layouts.globals),
            Some(&layouts.page_curl_uniforms),
            Some(&layouts.texture),
            // group 3：光源方向的 shadow map。
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
        &[
            Some(&layouts.globals),
            Some(&layouts.page_curl_uniforms),
        ],
        wgpu::TextureFormat::Rgba8Unorm,
        None,
        None,
    );

    PageCurlPipelines { mesh, shadow }
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
            depth_stencil_attachment: None,
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
        let Some(capture) = self.capture.as_ref() else {
            log::debug!("page_curl: 缺捕获纹理，这一帧不画卷曲网格");
            return;
        };
        let capture_bind_group = &capture.bind_group;

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
        pass.set_bind_group(2, capture_bind_group, &[]);
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
