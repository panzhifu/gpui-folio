//! 页面卷曲的 Metal 资源与绘制。母本是 `gpui-pre-wgpu/src/wgpu_renderer/page_curl.rs`：
//! `prepare`（顶点/索引/uniform 上传，几何按 (page_size, 顶点数, 索引数) 缓存）、
//! `encode_shadow`（光源方向的 shadow map）、`encode_mesh_pass`（画卷曲网格），
//! 以及正/背两张整窗捕获纹理的创建。口径差异只有两条：
//!   - uniform 不建常量缓冲，直接走 `set*Bytes`（240 字节，远在 4KB 限制内）——
//!     避开 managed 存储模式逐帧 memcpy + didModifyRange 的那一套。
//!   - 着色器在 `shaders.metal` 里（MSL 直译 WGSL 母本），随 metallib 一起编。

use gpui::{Bounds, DevicePixels, PageCurlUniforms, Scene, Size, ScaledPixels};

/// 与 `shaders.metal` 里 `CurlGpuUniforms` 一一对上（240 字节）。尺寸与三格关键
/// 偏移都要一致：错一位不报错，只是阴影矩阵与两张捕获纹理的采样原点各读到隔壁
/// 那格的去向。`[f32; 3]` 的补位格是 C 数组形状，与 MSL 侧保持同一对齐。
#[repr(C)]
#[derive(Clone, Copy)]
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
    incoming: f32,
    _padding: f32,
    shadow_map_size: f32,
    _pad0: [f32; 3],
    _pad1: [f32; 3],
    _pad1a: f32,
    light_view_proj: [[f32; 4]; 4],
    back_capture_origin: [f32; 2],
    _pad2: [f32; 2],
}

const _: () = {
    use std::mem::{offset_of, size_of};
    assert!(size_of::<GpuCurlUniforms>() == 240);
    assert!(offset_of!(GpuCurlUniforms, incoming) == 120);
    assert!(offset_of!(GpuCurlUniforms, shadow_map_size) == 128);
    assert!(offset_of!(GpuCurlUniforms, light_view_proj) == 160);
    assert!(offset_of!(GpuCurlUniforms, back_capture_origin) == 224);
};

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
            incoming: uniforms.incoming,
            _padding: 0.0,
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

/// 光源方向的正交 shadow map 边长。一张 1024² 的 RGBA8 约 4 MB；再大换来的
/// 是 3×3 PCF 里一个纹素更细，肉眼分不出。
const SHADOW_MAP_SIZE: u32 = 1024;

/// MSL 绑定表：与 shaders.metal 页面卷曲那段顶部的约定一致。
mod bind {
    pub const VERTEX: u64 = 0;
    pub const UNIFORMS: u64 = 1;
    pub const VIEWPORT: u64 = 2;
    pub const TEXTURE_CAPTURE: u64 = 0;
    pub const TEXTURE_SHADOW: u64 = 1;
    pub const TEXTURE_BACK: u64 = 2;
    pub const TEXTURE_INCOMING: u64 = 3;
    pub const SAMPLER_CAPTURE: u64 = 0;
    pub const SAMPLER_SHADOW: u64 = 1;
    pub const SAMPLER_BACK: u64 = 2;
    pub const SAMPLER_INCOMING: u64 = 3;
}

pub(super) struct PageCurlResources {
    mesh_pipeline: metal::RenderPipelineState,
    shadow_pipeline: metal::RenderPipelineState,
    sampler: metal::SamplerState,
    /// 本帧的 uniform（**每张网格一份**）：prepare 算好，两次 encode 各自用
    /// set*Bytes 递给着色器。一帧可以有两张网格（平移翻页的出场页 + 进场页），
    /// 从前只留一份 uniform、一次 draw，第二张网格整帧不存在。
    frame_uniforms: Vec<GpuCurlUniforms>,
    capture: Option<CaptureTexture>,
    back_capture: Option<CaptureTexture>,
    incoming_capture: Option<CaptureTexture>,
    depth: Option<DepthTexture>,
    shadow: Option<ShadowMap>,
    geometry: Option<GeometryCache>,
    frames: Vec<FrameDraw>,
}

struct CaptureTexture {
    texture: metal::Texture,
    size: Size<DevicePixels>,
}

struct DepthTexture {
    texture: metal::Texture,
    size: Size<DevicePixels>,
}

struct ShadowMap {
    color: metal::Texture,
    depth: metal::Texture,
}

/// 卷曲网格的顶点/索引只由版心尺寸和网格分辨率决定，翻页进度只改 uniform。
/// `key` 是拼接字节的内容指纹：指纹一样就复用缓冲（平移翻页两张平摊网格的
/// 顶点逐字节相同 —— 都是页局部坐标的四角；卷曲网格按形状变化重建）。
struct GeometryCache {
    vertex_buffer: metal::Buffer,
    index_buffer: metal::Buffer,
    key: u64,
}

struct FrameDraw {
    /// 这张网格在拼接后的索引缓冲里的起点（每张网格的索引是局部的，拼接时按
    /// 前面各张的顶点数做了重排）。
    index_start: u32,
    index_count: u32,
    /// 卷曲网格的页框（不是 canvas 的整块 content mask）：剪到版心，
    /// 避免卷起来那一截画到页眉/页脚上。
    page_bounds: Bounds<ScaledPixels>,
}

/// 顶点/索引拼接字节的内容指纹（FNV-1a）。
fn geometry_key(vertex_bytes: &[u8], index_bytes: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf29ce484222325;
    for byte in vertex_bytes.iter().chain(index_bytes) {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

/// 顶点/索引缓冲的存储模式：统一内存走 shared（CPU 写完 GPU 直见），
/// 离散内存走 managed（new_buffer_with_data 的初次上传自带同步）。
fn buffer_options(is_unified_memory: bool) -> metal::MTLResourceOptions {
    if is_unified_memory {
        metal::MTLResourceOptions::StorageModeShared
            | metal::MTLResourceOptions::CPUCacheModeWriteCombined
    } else {
        metal::MTLResourceOptions::StorageModeManaged
    }
}

impl PageCurlResources {
    /// 管线从 metallib 里按函数名加载；`target_format` 是主颜色附件的像素格式
    /// （捕获纹理与它一致）。`SHADERS_METALLIB_FALLBACK` 只在 metallib 缺函数时
    /// 给出可读的报错线索。
    pub(super) fn new(
        device: &metal::Device,
        library: &metal::LibraryRef,
        // 几何在 `prepare` 里建（那边才用得到统一/离散内存的判据），这里用不上。
        _is_unified_memory: bool,
        target_format: metal::MTLPixelFormat,
    ) -> Self {
        let mesh_pipeline = Self::build_mesh_pipeline(device, library, target_format);
        let shadow_pipeline = Self::build_shadow_pipeline(device, library);

        // 捕获纹理与卷曲网格是 1:1 的设备像素。最近邻既保住正文的锐度，也避免
        // 线性过滤在矩形外圈与清屏色之间揉出一条半透明灰边（母本同款 NEAREST）。
        let sampler_descriptor = metal::SamplerDescriptor::new();
        sampler_descriptor.set_min_filter(metal::MTLSamplerMinMagFilter::Nearest);
        sampler_descriptor.set_mag_filter(metal::MTLSamplerMinMagFilter::Nearest);
        sampler_descriptor.set_address_mode_s(metal::MTLSamplerAddressMode::ClampToEdge);
        sampler_descriptor.set_address_mode_t(metal::MTLSamplerAddressMode::ClampToEdge);
        let sampler = device.new_sampler(&sampler_descriptor);

        Self {
            mesh_pipeline,
            shadow_pipeline,
            sampler,
            frame_uniforms: Vec::new(),
            capture: None,
            back_capture: None,
            incoming_capture: None,
            depth: None,
            shadow: None,
            geometry: None,
            frames: Vec::new(),
        }
    }

    fn build_mesh_pipeline(
        device: &metal::Device,
        library: &metal::LibraryRef,
        target_format: metal::MTLPixelFormat,
    ) -> metal::RenderPipelineState {
        let vertex_fn = library
            .get_function("vs_page_curl", None)
            .expect("error locating vs_page_curl");
        let fragment_fn = library
            .get_function("fs_page_curl", None)
            .expect("error locating fs_page_curl");

        let descriptor = metal::RenderPipelineDescriptor::new();
        descriptor.set_label("page_curl_mesh");
        descriptor.set_vertex_function(Some(vertex_fn.as_ref()));
        descriptor.set_fragment_function(Some(fragment_fn.as_ref()));
        let color_attachment = descriptor.color_attachments().object_at(0).unwrap();
        color_attachment.set_pixel_format(target_format);
        // 片元给回的 alpha 恒为 1，这一趟本来就是不透明的：不开混合，
        // 片元直接替换 load 进来的颜色（与母本 blend: None 同口径）。
        color_attachment.set_blending_enabled(false);
        descriptor.set_depth_attachment_pixel_format(metal::MTLPixelFormat::Depth32Float);

        device
            .new_render_pipeline_state(&descriptor)
            .expect("could not create page_curl mesh pipeline state")
    }

    fn build_shadow_pipeline(device: &metal::Device, library: &metal::LibraryRef) -> metal::RenderPipelineState {
        let vertex_fn = library
            .get_function("vs_page_curl_shadow", None)
            .expect("error locating vs_page_curl_shadow");
        let fragment_fn = library
            .get_function("fs_page_curl_shadow", None)
            .expect("error locating fs_page_curl_shadow");

        let descriptor = metal::RenderPipelineDescriptor::new();
        descriptor.set_label("page_curl_shadow");
        descriptor.set_vertex_function(Some(vertex_fn.as_ref()));
        descriptor.set_fragment_function(Some(fragment_fn.as_ref()));
        let color_attachment = descriptor.color_attachments().object_at(0).unwrap();
        color_attachment.set_pixel_format(metal::MTLPixelFormat::RGBA8Unorm);
        color_attachment.set_blending_enabled(false);
        descriptor.set_depth_attachment_pixel_format(metal::MTLPixelFormat::Depth32Float);

        device
            .new_render_pipeline_state(&descriptor)
            .expect("could not create page_curl shadow pipeline state")
    }

    /// 窗口尺寸变了就换纹理：捕获/深度按窗口大小，几何按页框。
    pub(super) fn invalidate_size_dependent(&mut self) {
        self.capture = None;
        self.back_capture = None;
        self.incoming_capture = None;
        self.depth = None;
        self.geometry = None;
        self.frames.clear();
        self.frame_uniforms.clear();
    }

    /// 从场景里取**全部**网格，准备这一帧的顶点/索引缓冲与 uniform（每张一份）。
    pub(super) fn prepare(
        &mut self,
        scene: &Scene,
        device: &metal::Device,
        is_unified_memory: bool,
        size: Size<DevicePixels>,
    ) {
        let meshes = scene.page_curl_meshes.as_slice();
        if meshes.is_empty() {
            self.frames.clear();
            self.frame_uniforms.clear();
            return;
        }
        // 顶点/索引逐张拼接（第二张起的索引按前面的顶点数重排）；内容指纹没变就
        // 复用上一次的缓冲（卷曲网格逐帧形状不变，一个字节都不重传）。
        let mut vertex_bytes = Vec::with_capacity(
            meshes.iter().map(|m| std::mem::size_of_val(m.vertices.as_ref())).sum(),
        );
        let mut index_bytes = Vec::with_capacity(
            meshes.iter().map(|m| std::mem::size_of_val(m.indices.as_ref())).sum(),
        );
        let mut frame_uniforms = Vec::with_capacity(meshes.len());
        let mut frames = Vec::with_capacity(meshes.len());
        let (mut vertex_base, mut index_base) = (0u32, 0u32);
        for mesh in meshes {
            vertex_bytes.extend_from_slice(bytemuck_bytes(mesh.vertices.as_ref()));
            for index in mesh.indices.iter() {
                index_bytes.extend_from_slice(&(index + vertex_base).to_ne_bytes());
            }
            let light_view_proj = light_view_proj(&mesh.uniforms);
            frame_uniforms.push(GpuCurlUniforms::from_parts(&mesh.uniforms, light_view_proj));
            frames.push(FrameDraw {
                index_start: index_base,
                index_count: mesh.indices.len() as u32,
                page_bounds: mesh.bounds,
            });
            vertex_base += mesh.vertices.len() as u32;
            index_base += mesh.indices.len() as u32;
        }
        let key = geometry_key(&vertex_bytes, &index_bytes);
        if self.geometry.as_ref().is_none_or(|geometry| geometry.key != key) {
            let options = buffer_options(is_unified_memory);
            let vertex_buffer = device.new_buffer_with_data(
                vertex_bytes.as_ptr() as *const core::ffi::c_void,
                vertex_bytes.len() as u64,
                options,
            );
            let index_buffer = device.new_buffer_with_data(
                index_bytes.as_ptr() as *const core::ffi::c_void,
                index_bytes.len().max(4) as u64,
                options,
            );
            self.geometry = Some(GeometryCache {
                vertex_buffer,
                index_buffer,
                key,
            });
        }

        self.frame_uniforms = frame_uniforms;
        self.frames = frames;

        self.ensure_shadow(device);
        self.ensure_depth(device, size);
    }

    fn ensure_shadow(&mut self, device: &metal::Device) {
        if self.shadow.is_some() {
            return;
        }
        let color = new_texture(
            device,
            metal::MTLPixelFormat::RGBA8Unorm,
            SHADOW_MAP_SIZE,
            SHADOW_MAP_SIZE,
            metal::MTLTextureUsage::RenderTarget | metal::MTLTextureUsage::ShaderRead,
        );
        let depth = new_texture(
            device,
            metal::MTLPixelFormat::Depth32Float,
            SHADOW_MAP_SIZE,
            SHADOW_MAP_SIZE,
            metal::MTLTextureUsage::RenderTarget,
        );
        self.shadow = Some(ShadowMap { color, depth });
    }

    fn ensure_depth(&mut self, device: &metal::Device, size: Size<DevicePixels>) {
        if self.depth.as_ref().is_some_and(|depth| depth.size == size) {
            return;
        }
        let texture = new_texture(
            device,
            metal::MTLPixelFormat::Depth32Float,
            size.width.0.max(1) as u32,
            size.height.0.max(1) as u32,
            metal::MTLTextureUsage::RenderTarget,
        );
        self.depth = Some(DepthTexture { texture, size });
    }

    /// 正面捕获纹理，整窗大小、格式与主目标一致。
    pub(super) fn capture_view(
        &mut self,
        device: &metal::Device,
        target_format: metal::MTLPixelFormat,
        size: Size<DevicePixels>,
    ) -> metal::Texture {
        let needs_new = self
            .capture
            .as_ref()
            .map(|capture| capture.size != size)
            .unwrap_or(true);
        if needs_new {
            let texture = new_texture(
                device,
                target_format,
                size.width.0.max(1) as u32,
                size.height.0.max(1) as u32,
                metal::MTLTextureUsage::RenderTarget | metal::MTLTextureUsage::ShaderRead,
            );
            self.capture = Some(CaptureTexture { texture, size });
        }
        self.capture.as_ref().unwrap().texture.clone()
    }

    /// 背面捕获纹理，尺寸/格式与正面一致。
    pub(super) fn back_capture_view(
        &mut self,
        device: &metal::Device,
        target_format: metal::MTLPixelFormat,
        size: Size<DevicePixels>,
    ) -> metal::Texture {
        let needs_new = self
            .back_capture
            .as_ref()
            .map(|capture| capture.size != size)
            .unwrap_or(true);
        if needs_new {
            let texture = new_texture(
                device,
                target_format,
                size.width.0.max(1) as u32,
                size.height.0.max(1) as u32,
                metal::MTLTextureUsage::RenderTarget | metal::MTLTextureUsage::ShaderRead,
            );
            self.back_capture = Some(CaptureTexture { texture, size });
        }
        self.back_capture.as_ref().unwrap().texture.clone()
    }

    /// 进场页捕获纹理（第 3 层），尺寸/格式与正面、背面一致。
    pub(super) fn incoming_capture_view(
        &mut self,
        device: &metal::Device,
        target_format: metal::MTLPixelFormat,
        size: Size<DevicePixels>,
    ) -> metal::Texture {
        let needs_new = self
            .incoming_capture
            .as_ref()
            .map(|capture| capture.size != size)
            .unwrap_or(true);
        if needs_new {
            let texture = new_texture(
                device,
                target_format,
                size.width.0.max(1) as u32,
                size.height.0.max(1) as u32,
                metal::MTLTextureUsage::RenderTarget | metal::MTLTextureUsage::ShaderRead,
            );
            self.incoming_capture = Some(CaptureTexture { texture, size });
        }
        self.incoming_capture.as_ref().unwrap().texture.clone()
    }

    /// 把网格从光源方向渲染进 shadow map。必须在主 pass 之前调用。
    ///
    /// 多网格帧（平移翻页的两张平摊网格）里 shadow map 的内容其实没人读 ——
    /// 平摊分支在 shadow 采样之前就 return 了；这里把拼接后的整份几何画一遍，
    /// 只是让那一趟有东西可画。
    pub(super) fn encode_shadow(&self, command_buffer: &metal::CommandBufferRef) {
        let (Some(frame), Some(geometry), Some(shadow)) = (
            self.frames.first(),
            self.geometry.as_ref(),
            self.shadow.as_ref(),
        ) else {
            return;
        };
        let Some(uniforms) = self.frame_uniforms.first() else {
            return;
        };
        let render_pass_descriptor = metal::RenderPassDescriptor::new();
        let color_attachment = render_pass_descriptor
            .color_attachments()
            .object_at(0)
            .unwrap();
        color_attachment.set_texture(Some(&shadow.color));
        color_attachment.set_load_action(metal::MTLLoadAction::Clear);
        color_attachment.set_clear_color(metal::MTLClearColor::new(1.0, 1.0, 1.0, 1.0));
        color_attachment.set_store_action(metal::MTLStoreAction::Store);
        let depth_attachment = render_pass_descriptor.depth_attachment().unwrap();
        depth_attachment.set_texture(Some(&shadow.depth));
        depth_attachment.set_load_action(metal::MTLLoadAction::Clear);
        depth_attachment.set_clear_depth(1.0);
        // 这一趟的深度只在它自己内部比先后，出去没人再读。
        depth_attachment.set_store_action(metal::MTLStoreAction::DontCare);

        let encoder = command_buffer.new_render_command_encoder(render_pass_descriptor);
        encoder.set_render_pipeline_state(&self.shadow_pipeline);
        encoder.set_vertex_buffer(bind::VERTEX, Some(&geometry.vertex_buffer), 0);
        encoder.set_vertex_bytes(
            bind::UNIFORMS,
            std::mem::size_of::<GpuCurlUniforms>() as u64,
            std::ptr::from_ref(uniforms).cast(),
        );
        encoder.draw_indexed_primitives(
            metal::MTLPrimitiveType::Triangle,
            u64::from(frame.index_count),
            metal::MTLIndexType::UInt32,
            &geometry.index_buffer,
            0,
        );
        encoder.end_encoding();
    }

    /// 画卷曲网格。`target` 是主颜色附件。**逐网格绘制**：每张先剪自己的页框、
    /// 递自己的 uniform，再从拼接索引缓冲里取自己那一段（平移翻页的两张平摊
    /// 网格各盖半屏，从前只画先入树的那一张，另一张整帧不存在）。
    pub(super) fn encode_mesh_pass(
        &self,
        command_buffer: &metal::CommandBufferRef,
        target: &metal::TextureRef,
        size: Size<DevicePixels>,
    ) {
        if self.frames.is_empty() {
            return;
        }
        let Some(geometry) = self.geometry.as_ref() else {
            return;
        };
        let Some(depth) = self.depth.as_ref() else {
            return;
        };
        let Some(shadow) = self.shadow.as_ref() else {
            return;
        };
        // 没有捕获纹理就不画这一趟：绑一张兜底黑纹理正是「屏上一条黑带」的来源
        // （母本注释里量过的 378 设备像素）。跳过只是「这一趟没卷起来」。
        let Some(capture) = self.capture.as_ref() else {
            return;
        };
        let Some(_first_uniforms) = self.frame_uniforms.first() else {
            return;
        };
        // 背面纹理是懒创建的（层 2 只在折页里请求）：纯平移会话从头到尾没有背面，
        // 缺着它整趟跳过的话，平移翻页的两张平摊网格一张都画不出来（2026-11-02
        // 真机：左右平移整屏白）。先绑正面占位 —— 背面透墨分支只有卷曲网格走到，
        // 而卷曲网格落笔之前背面必已随层 2 的捕获就位。
        let back = self
            .back_capture
            .as_ref()
            .map_or(&capture.texture, |back| &back.texture);
        // 进场页纹理是懒创建的：还没请求过进场页捕获时先绑正面，着色器只在
        // `incoming = 1` 的平摊网格上采它，那种网格出现时这张纹理已经建好了。
        let incoming = self
            .incoming_capture
            .as_ref()
            .map_or(&capture.texture, |incoming| &incoming.texture);

        let render_pass_descriptor = metal::RenderPassDescriptor::new();
        let color_attachment = render_pass_descriptor
            .color_attachments()
            .object_at(0)
            .unwrap();
        color_attachment.set_texture(Some(target));
        color_attachment.set_load_action(metal::MTLLoadAction::Load);
        color_attachment.set_store_action(metal::MTLStoreAction::Store);
        let depth_attachment = render_pass_descriptor.depth_attachment().unwrap();
        depth_attachment.set_texture(Some(&depth.texture));
        depth_attachment.set_load_action(metal::MTLLoadAction::Clear);
        depth_attachment.set_clear_depth(1.0);
        // 这一趟的深度只在网格自己之间分先后，出去没人再读。
        depth_attachment.set_store_action(metal::MTLStoreAction::DontCare);

        let encoder = command_buffer.new_render_command_encoder(render_pass_descriptor);
        encoder.set_render_pipeline_state(&self.mesh_pipeline);
        encoder.set_vertex_buffer(bind::VERTEX, Some(&geometry.vertex_buffer), 0);
        encoder.set_fragment_texture(bind::TEXTURE_CAPTURE, Some(&capture.texture));
        encoder.set_fragment_texture(bind::TEXTURE_SHADOW, Some(&shadow.color));
        encoder.set_fragment_texture(bind::TEXTURE_BACK, Some(back));
        encoder.set_fragment_texture(bind::TEXTURE_INCOMING, Some(incoming));
        encoder.set_fragment_sampler_state(bind::SAMPLER_CAPTURE, Some(&self.sampler));
        encoder.set_fragment_sampler_state(bind::SAMPLER_SHADOW, Some(&self.sampler));
        encoder.set_fragment_sampler_state(bind::SAMPLER_BACK, Some(&self.sampler));
        encoder.set_fragment_sampler_state(bind::SAMPLER_INCOMING, Some(&self.sampler));
        for (frame, uniforms) in self.frames.iter().zip(&self.frame_uniforms) {
            // 剪到版心：卷起来那一截不该画到页眉/页脚上；两张平摊网格各剪各的范围。
            let scissor = scissor_rect(&frame.page_bounds, size);
            if scissor.width > 0 && scissor.height > 0 {
                encoder.set_scissor_rect(scissor);
            }
            encoder.set_vertex_bytes(
                bind::UNIFORMS,
                std::mem::size_of::<GpuCurlUniforms>() as u64,
                std::ptr::from_ref(uniforms).cast(),
            );
            encoder.set_vertex_bytes(
                bind::VIEWPORT,
                std::mem::size_of::<Size<DevicePixels>>() as u64,
                std::ptr::addr_of!(size).cast(),
            );
            encoder.set_fragment_bytes(
                bind::UNIFORMS,
                std::mem::size_of::<GpuCurlUniforms>() as u64,
                std::ptr::from_ref(uniforms).cast(),
            );
            encoder.draw_indexed_primitives(
                metal::MTLPrimitiveType::Triangle,
                u64::from(frame.index_count),
                metal::MTLIndexType::UInt32,
                &geometry.index_buffer,
                u64::from(frame.index_start) * 4,
            );
        }
        encoder.end_encoding();
    }
}

/// 本帧的 uniform 与它对应的网格帧。`Option` 拍平到资源结构体顶上，
/// 因为 `prepare` 与两次 encode 在 metal_renderer 里隔着捕获 pass。
pub(crate) fn scissor_rect(bounds: &Bounds<ScaledPixels>, size: Size<DevicePixels>) -> metal::MTLScissorRect {
    let x = bounds.origin.x.0.max(0.0).max(0.0) as u64;
    let y = bounds.origin.y.0.max(0.0) as u64;
    let right = ((bounds.origin.x.0 + bounds.size.width.0)
        .min(size.width.0 as f32)
        .max(0.0)) as u64;
    let bottom = ((bounds.origin.y.0 + bounds.size.height.0)
        .min(size.height.0 as f32)
        .max(0.0)) as u64;
    metal::MTLScissorRect {
        x,
        y,
        width: right.saturating_sub(x),
        height: bottom.saturating_sub(y),
    }
}

fn new_texture(
    device: &metal::Device,
    format: metal::MTLPixelFormat,
    width: u32,
    height: u32,
    usage: metal::MTLTextureUsage,
) -> metal::Texture {
    let descriptor = metal::TextureDescriptor::new();
    descriptor.set_pixel_format(format);
    descriptor.set_width(u64::from(width.max(1)));
    descriptor.set_height(u64::from(height.max(1)));
    descriptor.set_usage(usage);
    device.new_texture(&descriptor)
}

/// repr(C) POD 切片的字节视图。不引入 bytemuck：这里只有两个调用点，
/// 类型上已经钉死是 `PageCurlVertex` / `u32`。
fn bytemuck_bytes<T>(slice: &[T]) -> &[u8] {
    unsafe {
        std::slice::from_raw_parts(slice.as_ptr() as *const u8, std::mem::size_of_val(slice))
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
    // 视图空间的 z 在标准 look-at 下是负的（点在相机前方），而 ortho 把
    // [near, far] 线性映到 [0, 1]：两端取负，[-0.1, -(2d+4R)] 才是正确的深度
    // 区间（母本注释里量过「一点阴影都没有」的那个坑）。
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
