//! 背景模糊（毛玻璃）的 GPU 资源与编码。
//!
//! 这一列 primitive 逼出一个前提：**采不到屏幕上的像素**。表面的纹理在 Android 的 GLES
//! 通路上只有 `RENDER_ATTACHMENT` 一个用途位（这台 DBY-W09 实测：`copy_texture_to_texture`
//! 报 `Usage flags TextureUsages(RENDER_ATTACHMENT) of Texture with '<Surface Texture>'
//! label do not contain required usage flags TextureUsages(COPY_SRC)`），既不能拷出来也不
//! 能采样。所以这一帧必须先画进一张我们自己造的纹理，玻璃采它，最后整张贴回表面。
//!
//! 那张中间纹理**只在场景里真有玻璃时才建**：桌面端的嵌入整列不走玻璃，那条路径一个字节
//! 的额外开销都没有。
//!
//! 对外三个入口，按 `wgpu_renderer::record_frame` 的调用顺序：
//!
//! - [`BackdropResources::ensure_scene_target`]：这一帧要玻璃的话，备好整窗的中间纹理，
//!   回它的 view（调用方拿它当主 pass 的目标）；
//! - [`BackdropResources::encode`]：主 pass 关掉之后，把玻璃底下那一块降采样、两趟可分离
//!   高斯，备好最后画玻璃要用的绑定与顶点；
//! - [`BackdropResources::encode_present`]：整帧画完之后，把中间纹理贴回表面。
//!
//! [`BackdropResources::draw_glass`] 夹在中间，由 `encode_scene` 在重开主 pass 之后调。

use std::mem::size_of;

use super::*;

const BACKDROP_SHADERS: &str = include_str!("../backdrop.wgsl");

/// 降采样之后希望一个高斯半径占多少纹素。抽数固定 17（±8），所以这一格定的是**降采样倍数**：
/// 半径 60 设备像素、目标 6 纹素 ⇒ 倍数 10，小纹理上 ±8 抽正好盖到 ±48 像素，剩下的由
/// 降采样那一趟的 4×4 tent 补。
const TARGET_TEXEL_RADIUS: f32 = 6.0;
/// 降采样倍数的上下限。上限是怕矩形本来就小（一条 96 宽的侧栏降到 1/16 只剩几个纹素，
/// 采出来是一片糊），下限是怕半径太大而抽数不够。
const MIN_DOWNSCALE: u32 = 1;
const MAX_DOWNSCALE: u32 = 16;
/// 外扩多少才够：高斯的尾巴要读到玻璃之外的像素，不外扩就会在玻璃外沿读到一个硬边。
/// 半径的 2.5 倍 ≈ 5σ，再往外权重已经掉到 1e-6 以下。
const DILATE_FACTOR: f32 = 2.5;

/// 与 `backdrop.wgsl` 的 `GlassRecord` 一一对上：逐实例的顶点缓冲，不是 uniform。
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuGlass {
    /// origin.xy, size.zw —— 设备像素。
    bounds: [f32; 4],
    content_mask: [f32; 4],
    corner_radii: [f32; 4],
    blur_radius: f32,
    _pad: [f32; 3],
}

/// 与 `backdrop.wgsl` 的 `Backdrop` 一一对上。
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuBackdrop {
    region_origin: [f32; 2],
    region_size: [f32; 2],
    window_size: [f32; 2],
    /// x = 表面按预乘 alpha 解释。
    flags: [u32; 2],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuBlur {
    src_origin: [f32; 2],
    src_size: [f32; 2],
    dst_size: [f32; 2],
    step: f32,
    radius: f32,
}

const _: () = {
    assert!(size_of::<GpuGlass>() == 64);
    assert!(size_of::<GpuBackdrop>() == 32);
    assert!(size_of::<GpuBlur>() == 32);
};

struct BackdropPipelines {
    downsample: wgpu::RenderPipeline,
    blur_h: wgpu::RenderPipeline,
    blur_v: wgpu::RenderPipeline,
    glass: wgpu::RenderPipeline,
    present: wgpu::RenderPipeline,
    blur_layout: wgpu::BindGroupLayout,
    glass_layout: wgpu::BindGroupLayout,
    present_layout: wgpu::BindGroupLayout,
}

/// 一张按需重建的中间纹理。尺寸或格式对不上就整张换掉 —— 玻璃矩形与窗口都会变。
struct CachedTarget {
    /// 没人读它，但它就是这张纹理的持有者：留个 view 而把 texture 删掉，等于把所有权
    /// 交给「view 隐式扛着 texture」这条 wgpu 内部的约定。
    #[expect(dead_code, reason = "持有纹理本体，见上")]
    texture: wgpu::Texture,
    view: wgpu::TextureView,
    width: u32,
    height: u32,
}

/// `encode` 备好、`draw_glass` 消费的那一组绑定与顶点。
struct PendingGlass {
    bind_group: wgpu::BindGroup,
    /// 这一帧玻璃记录占的字节区间，喂给 `set_vertex_buffer`。
    vertices: std::ops::Range<u64>,
    count: u32,
}

pub(super) struct BackdropResources {
    pipelines: BackdropPipelines,
    sampler: wgpu::Sampler,
    format: wgpu::TextureFormat,
    /// 整帧先画进这里，玻璃采它，最后贴回表面。
    scene: Option<CachedTarget>,
    ping: Option<CachedTarget>,
    pong: Option<CachedTarget>,
    blur_uniform: Option<wgpu::Buffer>,
    glass_uniform: Option<wgpu::Buffer>,
    glass_vertices: Option<wgpu::Buffer>,
    pending: Option<PendingGlass>,
}

fn texture_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: true },
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    }
}

fn sampler_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
        count: None,
    }
}

fn uniform_entry(size: Option<NonZeroU64>) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding: 0,
        visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: size,
        },
        count: None,
    }
}

fn create_pipelines(
    device: &wgpu::Device,
    target_format: wgpu::TextureFormat,
    alpha_mode: wgpu::CompositeAlphaMode,
) -> BackdropPipelines {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("backdrop_shaders"),
        source: wgpu::ShaderSource::Wgsl(BACKDROP_SHADERS.into()),
    });

    let blur_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("backdrop_blur_layout"),
        entries: &[
            uniform_entry(NonZeroU64::new(size_of::<GpuBlur>() as u64)),
            texture_entry(1),
            sampler_entry(2),
        ],
    });
    let glass_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("backdrop_glass_layout"),
        entries: &[
            uniform_entry(NonZeroU64::new(size_of::<GpuBackdrop>() as u64)),
            texture_entry(1),
            sampler_entry(2),
        ],
    });
    let present_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("backdrop_present_layout"),
        entries: &[texture_entry(0), sampler_entry(1)],
    });

    let blur_target = wgpu::ColorTargetState {
        format: target_format,
        blend: None,
        write_mask: wgpu::ColorWrites::ALL,
    };
    // 玻璃是往表面上画，必须与表面同一套 alpha 约定：预乘的表面上画非预乘，圆角外沿会
    // 亮出一圈边；反过来会把边压暗。
    let glass_blend = match alpha_mode {
        wgpu::CompositeAlphaMode::PreMultiplied => {
            Some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING)
        }
        _ => Some(wgpu::BlendState::ALPHA_BLENDING),
    };
    let glass_target = wgpu::ColorTargetState {
        format: target_format,
        blend: glass_blend,
        write_mask: wgpu::ColorWrites::ALL,
    };

    let create = |label: &'static str,
                  vs: &str,
                  fs: &str,
                  layout: &wgpu::BindGroupLayout,
                  target: &wgpu::ColorTargetState,
                  vertex_buffers: &[wgpu::VertexBufferLayout<'_>]| {
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(label),
            bind_group_layouts: &[Some(layout)],
            immediate_size: 0,
        });
        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some(label),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some(vs),
                buffers: vertex_buffers,
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some(fs),
                targets: &[Some(target.clone())],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleStrip,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Cw,
                cull_mode: None,
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        })
    };

    // 一条玻璃 = 一个实例。步进是 Instance，四个顶点共用同一条记录。
    let glass_vertices = [wgpu::VertexBufferLayout {
        array_stride: size_of::<GpuGlass>() as u64,
        step_mode: wgpu::VertexStepMode::Instance,
        attributes: &[
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32x4,
                offset: 0,
                shader_location: 0,
            },
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32x4,
                offset: 16,
                shader_location: 1,
            },
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32x4,
                offset: 32,
                shader_location: 2,
            },
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32,
                offset: 48,
                shader_location: 3,
            },
        ],
    }];

    BackdropPipelines {
        downsample: create(
            "backdrop_downsample",
            "vs_fullscreen",
            "fs_downsample",
            &blur_layout,
            &blur_target,
            &[],
        ),
        blur_h: create(
            "backdrop_blur_h",
            "vs_fullscreen",
            "fs_blur_h",
            &blur_layout,
            &blur_target,
            &[],
        ),
        blur_v: create(
            "backdrop_blur_v",
            "vs_fullscreen",
            "fs_blur_v",
            &blur_layout,
            &blur_target,
            &[],
        ),
        glass: create(
            "backdrop_glass",
            "vs_glass",
            "fs_glass",
            &glass_layout,
            &glass_target,
            &glass_vertices,
        ),
        present: create(
            "backdrop_present",
            "vs_fullscreen",
            "fs_present",
            &present_layout,
            &blur_target,
            &[],
        ),
        blur_layout,
        glass_layout,
        present_layout,
    }
}

fn ensure_target<'a>(
    device: &wgpu::Device,
    slot: &'a mut Option<CachedTarget>,
    label: &str,
    width: u32,
    height: u32,
    format: wgpu::TextureFormat,
    usage: wgpu::TextureUsages,
) -> &'a CachedTarget {
    let stale = match slot.as_ref() {
        Some(target) => target.width != width || target.height != height,
        None => true,
    };
    if stale {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        *slot = Some(CachedTarget {
            texture,
            view,
            width,
            height,
        });
    }
    slot.as_ref().expect("just filled")
}

fn ensure_buffer<'a>(
    device: &wgpu::Device,
    slot: &'a mut Option<wgpu::Buffer>,
    label: &str,
    size: u64,
    usage: wgpu::BufferUsages,
) -> &'a wgpu::Buffer {
    let enough = slot
        .as_ref()
        .is_some_and(|buffer| buffer.size() >= size && buffer.usage() == usage);
    if !enough {
        *slot = Some(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size,
            usage,
            mapped_at_creation: false,
        }));
    }
    slot.as_ref().expect("just filled")
}

/// 一个铺满目标的小 pass：绑上源与参数，画四个顶点。
fn blit(
    encoder: &mut wgpu::CommandEncoder,
    pipeline: &wgpu::RenderPipeline,
    target: &wgpu::TextureView,
    bind_group: &wgpu::BindGroup,
    label: &str,
) {
    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some(label),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view: target,
            resolve_target: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                store: wgpu::StoreOp::Store,
            },
            depth_slice: None,
        })],
        depth_stencil_attachment: None,
        ..Default::default()
    });
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, bind_group, &[]);
    pass.draw(0..4, 0..1);
}

impl BackdropResources {
    pub(super) fn new(
        device: &wgpu::Device,
        target_format: wgpu::TextureFormat,
        alpha_mode: wgpu::CompositeAlphaMode,
    ) -> Self {
        Self {
            pipelines: create_pipelines(device, target_format, alpha_mode),
            sampler: device.create_sampler(&wgpu::SamplerDescriptor {
                label: Some("backdrop_sampler"),
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                address_mode_u: wgpu::AddressMode::ClampToEdge,
                address_mode_v: wgpu::AddressMode::ClampToEdge,
                ..Default::default()
            }),
            format: target_format,
            scene: None,
            ping: None,
            pong: None,
            blur_uniform: None,
            glass_uniform: None,
            glass_vertices: None,
            pending: None,
        }
    }

    /// 中间纹理按窗口尺寸活着：换尺寸、换设备都得整份丢掉。
    pub(super) fn invalidate_size_dependent(&mut self) {
        self.scene = None;
        self.ping = None;
        self.pong = None;
        self.pending = None;
    }

    /// 表面的 alpha 约定或格式变了（`update_transparency`）：重建这五条管线，因为玻璃的
    /// blend 状态必须跟着换。中间纹理与尺寸无关，不必丢。
    pub(super) fn set_alpha_mode(
        &mut self,
        device: &wgpu::Device,
        target_format: wgpu::TextureFormat,
        alpha_mode: wgpu::CompositeAlphaMode,
    ) {
        self.pipelines = create_pipelines(device, target_format, alpha_mode);
        self.format = target_format;
        self.pending = None;
    }

    /// 这一帧有玻璃的话，备好整窗的中间纹理并回它的 view —— 调用方要把主 pass 画进这里，
    /// 而不是画进表面。
    pub(super) fn ensure_scene_target(
        &mut self,
        device: &wgpu::Device,
        size: Size<DevicePixels>,
    ) -> wgpu::TextureView {
        let target = ensure_target(
            device,
            &mut self.scene,
            "backdrop_scene",
            size.width.0.max(1) as u32,
            size.height.0.max(1) as u32,
            self.format,
            wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        );
        target.view.clone()
    }

    /// 把玻璃底下那一块糊掉。调用方负责**已经关掉**主 pass，并在返回之后以 `LoadOp::Load`
    /// 重开、再调 [`Self::draw_glass`]。
    pub(super) fn encode(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        size: Size<DevicePixels>,
        glasses: &[gpui::BackdropGlassPrimitive],
        premultiplied_alpha: bool,
    ) {
        self.pending = None;

        let Some(scene) = &self.scene else {
            // 没有中间纹理就没有可采的背景：这一帧的玻璃只画 tint。
            return;
        };
        let scene_view = &scene.view;
        let Some(region) = backdrop_region(glasses, size) else {
            return;
        };
        let format = self.format;
        let factor = downscale_factor(glasses);
        let small_w = (region.width / factor).max(1);
        let small_h = (region.height / factor).max(1);

        let ping = ensure_target(
            device,
            &mut self.ping,
            "backdrop_ping",
            small_w,
            small_h,
            format,
            wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        );
        let ping_view = &ping.view;
        let pong = ensure_target(
            device,
            &mut self.pong,
            "backdrop_pong",
            small_w,
            small_h,
            format,
            wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        );
        let pong_view = &pong.view;

        let blur_uniform = ensure_buffer(
            device,
            &mut self.blur_uniform,
            "backdrop_blur_uniform",
            size_of::<GpuBlur>() as u64,
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );
        let blur_layout = &self.pipelines.blur_layout;
        let sampler = &self.sampler;
        let write_blur = |params: GpuBlur, label: &str, view: &wgpu::TextureView| {
            queue.write_buffer(blur_uniform, 0, bytemuck::bytes_of(&params));
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(label),
                layout: blur_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: blur_uniform.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Sampler(sampler),
                    },
                ],
            })
        };

        let window_size = [size.width.0 as f32, size.height.0 as f32];
        let region_size = [region.width as f32, region.height as f32];
        let origin = [region.origin.x as f32, region.origin.y as f32];
        let small_size = [small_w as f32, small_h as f32];
        let radius = small_radius(glasses, factor);

        // 三趟都是「读一张、写另一张」，所以每趟一个新绑定；同一个 blur_uniform 反复覆写
        // 是安全的 —— 队列写发生在提交之前，而这三趟按录制顺序执行。
        let group = write_blur(
            GpuBlur {
                src_origin: origin,
                src_size: window_size,
                dst_size: small_size,
                step: factor as f32,
                radius: 0.0,
            },
            "backdrop_downsample_group",
            scene_view,
        );
        blit(
            encoder,
            &self.pipelines.downsample,
            ping_view,
            &group,
            "backdrop_downsample_pass",
        );

        let straight = GpuBlur {
            src_origin: [0.0, 0.0],
            src_size: small_size,
            dst_size: small_size,
            step: 1.0,
            radius,
        };
        let group = write_blur(straight, "backdrop_blur_h_group", ping_view);
        blit(
            encoder,
            &self.pipelines.blur_h,
            pong_view,
            &group,
            "backdrop_blur_h_pass",
        );
        let group = write_blur(straight, "backdrop_blur_v_group", pong_view);
        blit(
            encoder,
            &self.pipelines.blur_v,
            ping_view,
            &group,
            "backdrop_blur_v_pass",
        );

        // 头部进 uniform，每条记录进逐实例的顶点缓冲 —— 数量因此不受 uniform 布局约束。
        let records: Vec<u8> = glasses
            .iter()
            .flat_map(|glass| bytemuck::bytes_of(&to_gpu_glass(glass)).to_vec())
            .collect();
        let glass_uniform = ensure_buffer(
            device,
            &mut self.glass_uniform,
            "backdrop_glass_uniform",
            size_of::<GpuBackdrop>() as u64,
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );
        queue.write_buffer(
            glass_uniform,
            0,
            bytemuck::bytes_of(&GpuBackdrop {
                region_origin: origin,
                region_size,
                window_size,
                flags: [premultiplied_alpha as u32, 0],
            }),
        );
        let vertex_bytes = records.len() as u64;
        let glass_vertices = ensure_buffer(
            device,
            &mut self.glass_vertices,
            "backdrop_glass_vertices",
            vertex_bytes,
            wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        );
        queue.write_buffer(glass_vertices, 0, &records);

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("backdrop_glass_group"),
            layout: &self.pipelines.glass_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: glass_uniform.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(ping_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
            ],
        });
        self.pending = Some(PendingGlass {
            bind_group,
            vertices: 0..vertex_bytes,
            count: glasses.len() as u32,
        });
    }

    /// 把上一趟 `encode` 备好的玻璃画进**当前**这个 pass。没编排出玻璃（区域空、或这一帧
    /// 根本没有玻璃）时什么都不做。
    pub(super) fn draw_glass(&self, pass: &mut wgpu::RenderPass<'_>) {
        let Some(pending) = &self.pending else {
            return;
        };
        let Some(vertices) = &self.glass_vertices else {
            return;
        };
        pass.set_pipeline(&self.pipelines.glass);
        pass.set_bind_group(0, &pending.bind_group, &[]);
        pass.set_vertex_buffer(0, vertices.slice(pending.vertices.clone()));
        pass.draw(0..4, 0..pending.count);
    }

    /// 整帧画完，把中间纹理贴回表面。这一趟不开混合、写满四个通道，所以像素与直接画在
    /// 表面上逐位相同。
    pub(super) fn encode_present(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        frame_view: &wgpu::TextureView,
    ) {
        let Some(scene) = &self.scene else {
            return;
        };
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("backdrop_present_group"),
            layout: &self.pipelines.present_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&scene.view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
            ],
        });
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("backdrop_present_pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: frame_view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: None,
            ..Default::default()
        });
        pass.set_pipeline(&self.pipelines.present);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.draw(0..4, 0..1);
    }
}

/// 一条玻璃矩形（设备像素，左上原点）。
struct Region {
    origin: wgpu::Origin3d,
    width: u32,
    height: u32,
}

/// `ScaledPixels` 到 `f32`：与 `scissor_for` 同一套取法。
fn dev(value: gpui::ScaledPixels) -> f32 {
    value.0
}

/// 全部玻璃格的 `bounds ∩ content_mask` 的并，再按最大半径外扩、裁进窗口。
/// 一条都落不进窗口时回 `None`（这一帧不必模糊任何东西）。
fn backdrop_region(
    glasses: &[gpui::BackdropGlassPrimitive],
    size: Size<DevicePixels>,
) -> Option<Region> {
    let mut left = f32::MAX;
    let mut top = f32::MAX;
    let mut right = f32::MIN;
    let mut bottom = f32::MIN;
    let mut radius = 0.0f32;
    for glass in glasses {
        let bounds = glass.bounds.intersect(&glass.content_mask.bounds);
        if bounds.is_empty() {
            continue;
        }
        left = left.min(dev(bounds.origin.x));
        top = top.min(dev(bounds.origin.y));
        right = right.max(dev(bounds.right()));
        bottom = bottom.max(dev(bounds.bottom()));
        radius = radius.max(dev(glass.blur_radius));
    }
    if left > right || top > bottom {
        return None;
    }

    let dilate = (radius * DILATE_FACTOR).ceil() as i32;
    let window_w = size.width.0.max(1) as u32;
    let window_h = size.height.0.max(1) as u32;
    let x = (left.floor() as i32 - dilate).clamp(0, window_w as i32) as u32;
    let y = (top.floor() as i32 - dilate).clamp(0, window_h as i32) as u32;
    let x_end = (right.ceil() as i32 + dilate).clamp(0, window_w as i32) as u32;
    let y_end = (bottom.ceil() as i32 + dilate).clamp(0, window_h as i32) as u32;
    if x_end <= x || y_end <= y {
        return None;
    }
    Some(Region {
        origin: wgpu::Origin3d { x, y, z: 0 },
        width: x_end - x,
        height: y_end - y,
    })
}

/// 最大的那一条玻璃半径（设备像素）。多条玻璃共用一张模糊图，所以按最大的来。
fn max_radius(glasses: &[gpui::BackdropGlassPrimitive]) -> f32 {
    glasses
        .iter()
        .map(|glass| dev(glass.blur_radius))
        .fold(0.0f32, f32::max)
}

/// 降采样倍数：按最大半径定，让小纹理上的半径落在 [`TARGET_TEXEL_RADIUS`] 附近。
fn downscale_factor(glasses: &[gpui::BackdropGlassPrimitive]) -> u32 {
    (max_radius(glasses) / TARGET_TEXEL_RADIUS)
        .round()
        .clamp(MIN_DOWNSCALE as f32, MAX_DOWNSCALE as f32) as u32
}

/// 小纹理上该用多高的半径：把设备像素的半径换算过去，至少留 1 纹素（否则 17 抽全落在
/// 同一个纹素上，等于没糊）。
fn small_radius(glasses: &[gpui::BackdropGlassPrimitive], factor: u32) -> f32 {
    (max_radius(glasses) / factor.max(1) as f32).max(1.0)
}

fn to_gpu_glass(glass: &gpui::BackdropGlassPrimitive) -> GpuGlass {
    let bounds = &glass.bounds;
    let mask = &glass.content_mask.bounds;
    let radii = &glass.corner_radii;
    GpuGlass {
        bounds: [
            dev(bounds.origin.x),
            dev(bounds.origin.y),
            dev(bounds.size.width),
            dev(bounds.size.height),
        ],
        content_mask: [
            dev(mask.origin.x),
            dev(mask.origin.y),
            dev(mask.size.width),
            dev(mask.size.height),
        ],
        corner_radii: [
            dev(radii.top_left),
            dev(radii.top_right),
            dev(radii.bottom_right),
            dev(radii.bottom_left),
        ],
        blur_radius: dev(glass.blur_radius),
        _pad: [0.0; 3],
    }
}
