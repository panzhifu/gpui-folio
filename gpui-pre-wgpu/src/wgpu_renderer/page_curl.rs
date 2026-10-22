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
    capture_origin: [f32; 2],
    _padding: [f32; 2],
}

impl GpuCurlUniforms {
    fn from_parts(uniforms: &PageCurlUniforms) -> Self {
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
            capture_origin: uniforms.capture_origin,
            _padding: [0.0; 2],
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
        ],
        target_format,
        Some(mesh_depth),
        None,
    );

    // 投影那一趟：混合开着（黑压在新页的字上），深度**只测不写** —— 它钉在页面平面上，写下去
    // 会把后面卷面那些抬起来的像素（深度更小）挡在门外，正相反的顺序才是要的。
    let shadow = create(
        "page_curl_shadow",
        "vs_page_curl_shadow",
        "fs_page_curl_shadow",
        &[
            Some(&layouts.globals),
            Some(&layouts.page_curl_uniforms),
            Some(&layouts.texture),
        ],
        target_format,
        Some(wgpu::DepthStencilState {
            format: wgpu::TextureFormat::Depth32Float,
            depth_write_enabled: Some(false),
            depth_compare: Some(wgpu::CompareFunction::LessEqual),
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        }),
        Some(wgpu::BlendState {
            color: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::One,
                dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                operation: wgpu::BlendOperation::Add,
            },
            alpha: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::One,
                dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                operation: wgpu::BlendOperation::Add,
            },
        }),
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

        let uniforms = GpuCurlUniforms::from_parts(&mesh.uniforms);
        queue.write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&uniforms));

        self.frame = Some(FrameDraw {
            index_count,
            page_bounds: mesh.bounds,
        });

        self.ensure_depth(device, size);
        Ok(())
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

        // 先投影再卷面：`LoadOp::Load` 保留着已经画好的那一页字，投影压上去，卷面最后盖住
        // 它自己那一片 —— 顺序就是两层 draw 的顺序，不需要第二张深度图。
        pass.set_pipeline(&self.pipelines.shadow);
        pass.set_bind_group(0, globals_bind_group, &[]);
        pass.set_bind_group(1, &self.uniform_bind_group, &[]);
        pass.set_bind_group(2, capture_bind_group, &[]);
        pass.set_vertex_buffer(0, geometry.vertex_buffer.slice(..));
        pass.set_index_buffer(geometry.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
        pass.draw_indexed(0..frame.index_count, 0, 0..1);

        pass.set_pipeline(&self.pipelines.mesh);
        pass.set_bind_group(0, globals_bind_group, &[]);
        pass.set_bind_group(1, &self.uniform_bind_group, &[]);
        pass.set_bind_group(2, capture_bind_group, &[]);
        pass.set_vertex_buffer(0, geometry.vertex_buffer.slice(..));
        pass.set_index_buffer(geometry.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
        pass.draw_indexed(0..frame.index_count, 0, 0..1);
    }
}
