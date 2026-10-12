//! The compositing pipeline.
//!
//! Compositing is one compute pass per layer, dispatched over the dirty
//! rectangle only. Each pass reads the accumulated result (`dst`) and the layer
//! (`src`), writes back in place, and the next layer reads that result.
//!
//! Layers are therefore processed bottom-to-top, which matches Photoshop and
//! keeps every pass to a single buffer pair with no ping-pong.
//!
//! # Uniforms are written before the encoder opens
//!
//! This is the subtle part. A `queue.write_buffer` issued while a pass is
//! recording is **not** visible to that pass — the writes are queued separately
//! from the encoder's command stream. The symptom is not an error, it is a
//! composite that silently runs with stale uniforms. So every buffer write for
//! a dispatch happens before `begin_compute_pass`, and the per-layer `Blend`
//! values live in one slotted buffer addressed by dynamic offset.

use ultrablend::BlendMode;
use ultrapixel::{PixelFormat, Rect, Surface};

use crate::context::{workgroups_2d, GpuContext};
use crate::layer::LayerGpu;

/// The WGSL kernel. `include_str!` keeps it in one place and makes it obvious
/// in review that the shader and the Rust side are a single unit.
const COMPOSITE_WGSL: &str = include_str!("shaders/composite.wgsl");

/// Slot size for a layer's `Blend` uniform.
///
/// wgpu requires dynamic buffer offsets to be aligned to
/// `min_uniform_buffer_offset_alignment` — 256 bytes on every real driver. One
/// 16-byte `Blend` therefore costs a 256-byte slot, which is still far cheaper
/// than allocating a buffer per layer.
const BLEND_SLOT: u64 = 256;
/// Uniform block matching `Params` in the shader.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    rect: [i32; 4],
    size: [u32; 2],
    alpha_plane: u32,
    _pad: u32,
}

/// Uniform block matching `Blend` in the shader. 16 bytes.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Blend {
    mode: u32,
    non_separable: u32,
    opacity: f32,
    fill: f32,
}

/// One layer's contribution to a composite.
pub struct LayerInput<'a> {
    pub gpu: &'a LayerGpu,
    /// Blend mode as it appears in the UI.
    pub mode: BlendMode,
    pub opacity: f32,
    pub fill: f32,
    pub visible: bool,
}

impl<'a> LayerInput<'a> {
    pub fn new(gpu: &'a LayerGpu) -> Self {
        Self {
            gpu,
            mode: BlendMode::Normal,
            opacity: 1.0,
            fill: 1.0,
            visible: true,
        }
    }

    /// True when the layer cannot change a pixel, so the pass is skipped
    /// without dispatching at all.
    pub fn is_noop(&self) -> bool {
        !self.visible
            || self.gpu.width == 0
            || self.gpu.height == 0
            || self.opacity <= 0.0
            || self.fill <= 0.0
    }
}

/// Owns the compute pipeline and a bind-group cache.
pub struct Compositor {
    pipeline: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
    params_buf: wgpu::Buffer,
    /// Staging buffer holding one `Blend` per visible layer in 256-byte slots.
    blend_buf: wgpu::Buffer,
    /// Slots currently allocated in `blend_buf`.
    blend_slots: u64,
    /// (dst, src) buffer size pair -> bind group, since a group pinches those
    /// two buffers and can be reused for every layer sharing them.
    cache: std::collections::HashMap<(u64, u64, u64), wgpu::BindGroup>,
}

impl Compositor {
    pub fn new(ctx: &GpuContext) -> Self {
        let module = ctx
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("ultraphoto-composite"),
                source: wgpu::ShaderSource::Wgsl(COMPOSITE_WGSL.into()),
            });

        let storage = |read_only: bool| wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        };
        // Uniforms carry no dynamic offset here: the `Blend` for the layer being
        // dispatched is written into the buffer *before* the encoder opens, and
        // the buffer is bound whole. One uniform buffer per compositing pass,
        // rewritten per layer, is cheap and avoids the offset-alignment trap.
        let uniform = || wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        };
        let entry = |binding: u32, ty: wgpu::BindingType| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty,
            count: None,
        };

        let layout = ctx
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("composite-bgl"),
                entries: &[
                    entry(0, uniform()),
                    // dst is read *and* written, which is why the shader declares
                    // `read_write` storage.
                    entry(1, storage(false)),
                    entry(2, storage(true)),
                    entry(3, storage(true)),
                    entry(4, uniform()),
                ],
            });

        let pll = ctx
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("composite-pll"),
                bind_group_layouts: &[&layout],
                push_constant_ranges: &[],
            });

        let pipeline = ctx
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("composite"),
                layout: Some(&pll),
                module: &module,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });

        let buf = |label: &'static str, size: u64| {
            ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };

        Self {
            pipeline,
            layout,
            params_buf: buf("composite-params", std::mem::size_of::<Params>() as u64),
            blend_buf: buf("composite-blend", BLEND_SLOT),
            blend_slots: 1,
            cache: std::collections::HashMap::new(),
        }
    }

    /// Composites `layers` bottom-to-top into `dst`.
    ///
    /// `rect` is the dirty region; pass the same rect that was uploaded and the
    /// passes shrink to match.
    pub fn composite(
        &mut self,
        ctx: &GpuContext,
        dst: &LayerGpu,
        layers: &[LayerInput<'_>],
        rect: Rect,
    ) {
        let rect = rect.intersect(dst.full_rect());
        if rect.is_empty() {
            return;
        }
        let active: Vec<&LayerInput<'_>> = layers.iter().filter(|l| !l.is_noop()).collect();

        if active.is_empty() {
            return;
        }

        // --- All queue writes for this dispatch happen HERE, before the
        // encoder opens, because they would not be picked up by a pass that is
        // already recording.
        ctx.queue.write_buffer(
            &self.params_buf,
            0,
            bytemuck::bytes_of(&Params {
                rect: [rect.x0, rect.y0, rect.x1, rect.y1],
                size: [dst.width, dst.height],
                alpha_plane: alpha_plane_index(dst.channels),
                _pad: 0,
            }),
        );

        // All the writes for a dispatch happen before the encoder opens.
        //
        // The blend uniform is written once per layer, in order, into
        // consecutive 256-byte slots. Each dispatch needs its own slot, so the
        // buffer holds `active.len()` slots and a bind group is created per
        // (dst, src, slot) triple — the only way to make a non-dynamic offset
        // see the right value.
        if active.len() as u64 > self.blend_slots {
            self.ensure_slots(ctx, active.len() as u64);
        }
        let mut blends = Vec::with_capacity(active.len());
        for (i, layer) in active.iter().enumerate() {
            let value = Blend {
                mode: layer.mode as u32,
                non_separable: if layer.mode.is_non_separable() {
                    layer.mode as u32
                } else {
                    0
                },
                opacity: layer.opacity,
                fill: layer.fill,
            };
            ctx.queue.write_buffer(
                &self.blend_buf,
                i as u64 * BLEND_SLOT,
                bytemuck::bytes_of(&value),
            );
            blends.push(value);
        }

        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("composite"),
            });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("composite-layers"),
            timestamp_writes: None,
        });
        // One thread per pixel: the shader derives (x, y) from gid.xy, so the grid is
        // the pixel grid. Using `workgroups_for` here would under-cover the
        // rect, because a 2D workgroup of size 64 covers only 64 threads along
        // X, not 64 columns of the grid.
        let (w, h) = workgroups_2d(rect.width() as u32, rect.height() as u32);

        for (i, layer) in active.iter().enumerate() {
            if layer.gpu.width != dst.width || layer.gpu.height != dst.height {
                tracing::warn!(
                    w = layer.gpu.width,
                    h = layer.gpu.height,
                    "skipping layer with mismatched geometry"
                );
                continue;
            }
            let bg = self.bind_group(ctx, dst, layer.gpu, i as u64);
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(w, h, 1);
        }

        // wgpu 25 ends a pass when it is dropped; there is no explicit `end`.
        drop(pass);
        ctx.queue.submit([encoder.finish()]);
    }

    /// Grows the blend staging buffer until it holds `count` slots.
    fn ensure_slots(&mut self, ctx: &GpuContext, count: u64) {
        let needed = count.max(1);
        self.cache.clear();
        self.blend_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("composite-blend"),
            size: needed * BLEND_SLOT,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.blend_slots = needed;
    }

    /// Drops cached bind groups. Call after any buffer the cache captured is
    /// destroyed.
    pub fn invalidate(&mut self) {
        self.cache.clear();
    }

    /// Key that identifies a (dst, src, slot) triple.
    ///
    /// Buffer *size* is not a usable key: every layer of the same canvas has
    /// exactly the same size, so a size-based cache silently returns the bind
    /// group for the wrong pair of buffers. That bug is invisible — the
    /// composite runs, the shader executes, and the output is simply wrong.
    ///
    /// [`LayerGpu`] carries a unique id, which is what actually distinguishes
    /// the buffers.
    #[inline]
    fn cache_key(dst: &LayerGpu, src: &LayerGpu, slot: u64) -> (u64, u64, u64) {
        (dst.id, src.id, slot)
    }

    fn bind_group(
        &mut self,
        ctx: &GpuContext,
        dst: &LayerGpu,
        src: &LayerGpu,
        slot: u64,
    ) -> wgpu::BindGroup {
        let key = Self::cache_key(dst, src, slot);
        if let Some(bg) = self.cache.get(&key) {
            return bg.clone();
        }
        let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("composite-bg"),
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: dst.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: src.buffer().as_entire_binding(),
                },
                // The alpha plane reuses the source buffer; the shader reaches
                // it through `params.alpha_plane`.
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: src.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    // Exactly this layer's slot.
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &self.blend_buf,
                        offset: slot * BLEND_SLOT,
                        size: std::num::NonZeroU64::new(std::mem::size_of::<Blend>() as u64),
                    }),
                },
            ],
        });
        self.cache.insert(key, bg.clone());
        bg
    }
}

/// Index of the alpha plane for a channel count, or `u32::MAX` when the layout
/// carries no alpha. Mirrors `ChannelLayout::first_alpha`.
#[inline]
fn alpha_plane_index(channels: u32) -> u32 {
    match channels {
        0 | 1 | 3 => u32::MAX,
        2 => 1,
        4 => 3,
        // For wider stacks (CMYK+UCR+alpha) alpha is still last.
        n => n.saturating_sub(1),
    }
}

/// Allocates a transparent-black composite target.
pub fn new_target(ctx: &GpuContext, width: u32, height: u32) -> LayerGpu {
    LayerGpu::new(&ctx.device, 0, width, height, 4, "composite-target")
}

/// Pixel format a composite target must use.
pub const TARGET_FORMAT: PixelFormat = PixelFormat::RGBA;

/// Copies a composite target into a CPU RGBA surface.
pub fn copy_target_to_surface(
    ctx: &GpuContext,
    target: &LayerGpu,
    out: &mut Surface,
) -> Result<(), String> {
    if out.channels() < 4 {
        return Err("composite target needs an RGBA surface".into());
    }
    target
        .download_rect(&ctx.device, &ctx.queue, out, target.full_rect())
        .map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layer::upload_full;
    use ultrapixel::{Depth, Rgba};

    fn ctx() -> Option<GpuContext> {
        match pollster::block_on(GpuContext::new(true)) {
            Ok(c) => Some(c),
            Err(e) => {
                eprintln!("skipping GPU test: {e}");
                None
            }
        }
    }

    fn solid(w: usize, h: usize, c: [f32; 4]) -> Surface {
        let mut s = Surface::new(w, h, Depth::F32, PixelFormat::RGBA);
        for (i, v) in c.iter().enumerate() {
            s.plane_mut(i)
                .fill_rect(Rect::new(0, 0, w as i32, h as i32), *v);
        }
        s
    }

    fn at(s: &Surface, x: usize, y: usize) -> Rgba {
        Rgba {
            r: s.plane(0).sample_normalized(x, y),
            g: s.plane(1).sample_normalized(x, y),
            b: s.plane(2).sample_normalized(x, y),
            a: s.plane(3).sample_normalized(x, y),
        }
    }

    fn gpu_composite(
        ctx: &GpuContext,
        backdrop: &Surface,
        source: &Surface,
        mode: BlendMode,
        opacity: f32,
    ) -> Surface {
        let (w, h) = (backdrop.width(), backdrop.height());
        let mut compositor = Compositor::new(ctx);
        let target = new_target(ctx, w as u32, h as u32);
        target
            .upload_rect(&ctx.queue, backdrop, target.full_rect())
            .unwrap();

        let src_gpu = upload_full(&ctx.device, &ctx.queue, source, "src");
        let mut li = LayerInput::new(&src_gpu);
        li.mode = mode;
        li.opacity = opacity;
        compositor.composite(ctx, &target, &[li], target.full_rect());

        let mut out = Surface::new(w, h, Depth::F32, PixelFormat::RGBA);
        target
            .download_rect(&ctx.device, &ctx.queue, &mut out, target.full_rect())
            .unwrap();
        out
    }

    /// The whole point of this crate: the GPU must agree with the CPU reference.
    #[test]
    fn gpu_matches_the_cpu_reference_across_blend_modes() {
        let Some(ctx) = ctx() else { return };
        let (w, h) = (32usize, 32usize);
        let backdrop = solid(w, h, [0.5, 0.25, 0.75, 1.0]);
        let source = solid(w, h, [0.5, 0.5, 0.5, 1.0]);
        let bd = at(&backdrop, 10, 10);
        let sr = at(&source, 10, 10);

        // Every mode, separable and non-separable.
        for mode in BlendMode::ALL {
            let expected = ultrablend::composite(bd, sr, mode, 1.0, 1.0);
            let got = gpu_composite(&ctx, &backdrop, &source, mode, 1.0);
            let actual = at(&got, 10, 10);

            let worst = (actual.r - expected.r)
                .abs()
                .max((actual.g - expected.g).abs())
                .max((actual.b - expected.b).abs())
                .max((actual.a - expected.a).abs());
            // Non-separable modes round-trip through HSL, so they carry more
            // float error than the separable ones; allow a slightly wider band.
            let tol = if mode.is_non_separable() { 0.02 } else { 2e-3 };
            assert!(
                worst < tol,
                "{}: GPU {:?} vs CPU {:?} (worst delta {worst})",
                mode.label(),
                (actual.r, actual.g, actual.b, actual.a),
                (expected.r, expected.g, expected.b, expected.a),
            );
        }
    }

    #[test]
    fn opacity_is_honoured_and_matches_the_reference() {
        let Some(ctx) = ctx() else { return };
        let (w, h) = (16usize, 16usize);
        let backdrop = solid(w, h, [0.0, 0.0, 0.0, 1.0]);
        let source = solid(w, h, [1.0, 1.0, 1.0, 1.0]);
        let bd = at(&backdrop, 8, 8);
        let sr = at(&source, 8, 8);

        for opacity in [0.0f32, 0.25, 0.5, 0.75, 1.0] {
            let expected = ultrablend::composite(bd, sr, BlendMode::Normal, opacity, 1.0);
            let got = gpu_composite(&ctx, &backdrop, &source, BlendMode::Normal, opacity);
            let actual = at(&got, 8, 8);
            assert!(
                (actual.r - expected.r).abs() < 2e-3,
                "opacity {opacity}: {} vs {}",
                actual.r,
                expected.r
            );
        }
    }

    #[test]
    fn a_hidden_layer_never_touches_the_target() {
        let Some(ctx) = ctx() else { return };
        let (w, h) = (16usize, 16usize);
        let backdrop = solid(w, h, [0.2, 0.2, 0.2, 1.0]);
        let source = solid(w, h, [1.0, 0.0, 0.0, 1.0]);

        let mut compositor = Compositor::new(&ctx);
        let target = new_target(&ctx, w as u32, h as u32);
        target
            .upload_rect(&ctx.queue, &backdrop, target.full_rect())
            .unwrap();
        let src_gpu = upload_full(&ctx.device, &ctx.queue, &source, "hidden");
        let mut li = LayerInput::new(&src_gpu);
        li.visible = false;
        compositor.composite(&ctx, &target, &[li], target.full_rect());

        let mut got = Surface::new(w, h, Depth::F32, PixelFormat::RGBA);
        target
            .download_rect(&ctx.device, &ctx.queue, &mut got, target.full_rect())
            .unwrap();
        assert_eq!(
            got.plane(0).sample_normalized(8, 8),
            0.2,
            "hidden layer leaked"
        );
    }

    #[test]
    fn compositing_only_touches_the_dirty_rect() {
        let Some(ctx) = ctx() else { return };
        let (w, h) = (64usize, 64usize);
        let backdrop = solid(w, h, [0.1, 0.1, 0.1, 1.0]);
        let source = solid(w, h, [1.0, 1.0, 1.0, 1.0]);

        let mut compositor = Compositor::new(&ctx);
        let target = new_target(&ctx, w as u32, h as u32);
        target
            .upload_rect(&ctx.queue, &backdrop, target.full_rect())
            .unwrap();
        let src_gpu = upload_full(&ctx.device, &ctx.queue, &source, "src");
        let li = LayerInput::new(&src_gpu);
        let dirty = Rect::new(10, 10, 20, 20);
        compositor.composite(&ctx, &target, &[li], dirty);

        let mut got = Surface::new(w, h, Depth::F32, PixelFormat::RGBA);
        target
            .download_rect(&ctx.device, &ctx.queue, &mut got, target.full_rect())
            .unwrap();
        // Inside the dirty rect the white layer was composited.
        assert!(
            got.plane(0).sample_normalized(15, 15) > 0.5,
            "got {} inside the dirty rect",
            got.plane(0).sample_normalized(15, 15)
        );
        // Outside it the backdrop survives.
        assert_eq!(got.plane(0).sample_normalized(40, 40), 0.1);
    }

    #[test]
    fn alpha_plane_indexing_matches_the_channel_layouts() {
        assert_eq!(alpha_plane_index(3), u32::MAX, "RGB has no alpha");
        assert_eq!(alpha_plane_index(1), u32::MAX, "Gray has no alpha");
        assert_eq!(alpha_plane_index(2), 1, "GrayAlpha");
        assert_eq!(alpha_plane_index(4), 3, "RGBA");
        assert_eq!(alpha_plane_index(6), 5, "CMYK+UCR+alpha is last");
    }

    #[test]
    fn noop_layers_are_detected_before_dispatch() {
        let Some(ctx) = ctx() else { return };
        let layer = upload_full(
            &ctx.device,
            &ctx.queue,
            &solid(8, 8, [1.0, 1.0, 1.0, 1.0]),
            "noop",
        );
        assert!(!LayerInput::new(&layer).is_noop());

        let mut li = LayerInput::new(&layer);
        li.visible = false;
        assert!(li.is_noop());

        let mut li = LayerInput::new(&layer);
        li.opacity = 0.0;
        assert!(li.is_noop());
    }
}
