//! A minimal probe that isolates why the compositor's kernel does not write.
//!
//! It mirrors the real compositor's bind group *exactly*, in order:
//!
//! ```text
//! 0: uniform Params
//! 1: read_write storage  dst     <-- the one the kernel writes
//! 2: read        storage  src
//! 3: read        storage  alpha  (aliased to src)
//! 4: uniform Blend
//! ```
//!
//! The body reads the uniforms, honours the guards, and writes a constant into
//! every plane. If this fails, the fault is in the plumbing — bind group
//! layout, buffer usage flags, or dispatch size — and not in the blend maths.

use crate::context::GpuContext;
use crate::layer::{upload_full, LayerGpu};
use ultrapixel::{Depth, PixelFormat, Rect, Surface};

const W: u32 = 16;
const H: u32 = 16;
const PROBE: &str = r#"
struct Params {
    rect: vec4<i32>,
    size: vec2<u32>,
    alpha_plane: u32,
    _pad: u32,
};

struct Blend {
    mode: u32,
    non_separable: u32,
    opacity: f32,
    fill: f32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
@group(0) @binding(2) var<storage, read> src: array<f32>;
@group(0) @binding(3) var<storage, read> src_alpha: array<f32>;
@group(0) @binding(4) var<uniform> blend: Blend;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = i32(gid.x) + params.rect.x;
    let y = i32(gid.y) + params.rect.y;
    if (x >= params.rect.z || y >= params.rect.w) { return; }
    if (u32(x) >= params.size.x || u32(y) >= params.size.y) { return; }

    let idx = u32(y) * params.size.x + u32(x);
    let plane = params.size.x * params.size.y;
    dst[idx] = 1.0;
    dst[plane + idx] = 1.0;
    dst[plane * 2u + idx] = 1.0;
    dst[plane * 3u + idx] = 1.0;

    // Also exercise the real src/alpha read, so a silent zero in either is
    // visible here rather than only in the compositor's tests.
    let sb = src[idx];
    let sa = src[3u * plane + idx];
    if (sb < 0.0 || sa < 0.0) {
        dst[idx] = 0.5;
    }
}
"#;

/// The compositor's bind group layout, verbatim.
fn probe_bgl(ctx: &GpuContext) -> wgpu::BindGroupLayout {
    let uniform = || wgpu::BindingType::Buffer {
        ty: wgpu::BufferBindingType::Uniform,
        has_dynamic_offset: false,
        min_binding_size: None,
    };
    let storage = |ro: bool| wgpu::BindingType::Buffer {
        ty: wgpu::BufferBindingType::Storage { read_only: ro },
        has_dynamic_offset: false,
        min_binding_size: None,
    };
    let e = |b: u32, ty: wgpu::BindingType| wgpu::BindGroupLayoutEntry {
        binding: b,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty,
        count: None,
    };
    ctx.device
        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("probe-bgl"),
            entries: &[
                e(0, uniform()),
                e(1, storage(false)),
                e(2, storage(true)),
                e(3, storage(true)),
                e(4, uniform()),
            ],
        })
}

/// A read_write storage buffer built exactly as `LayerGpu::new` builds one.
fn probe_dst_buffer(ctx: &GpuContext) -> wgpu::Buffer {
    ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("probe-dst"),
        size: (W as u64 * H as u64 * 4 * 4).max(16),
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    })
}

/// Uniform bytes matching the shader's `Params`.
fn probe_params_bytes(rect: [i32; 4]) -> [u8; 32] {
    let mut b = [0u8; 32];
    b[0..16].copy_from_slice(bytemuck::bytes_of(&rect));
    b[16..20].copy_from_slice(&W.to_le_bytes());
    b[20..24].copy_from_slice(&H.to_le_bytes());
    b[24..28].copy_from_slice(&3u32.to_le_bytes());
    b
}

/// De-stride a channel-major planar buffer back into a `Surface`.
fn read_planes(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    src: &wgpu::Buffer,
    out: &mut Surface,
) -> Result<(), String> {
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("probe-readback"),
        size: src.size(),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    enc.copy_buffer_to_buffer(src, 0, &staging, 0, src.size());
    queue.submit([enc.finish()]);

    let slice = staging.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    let _ = device.poll(wgpu::PollType::Wait);
    rx.recv()
        .map_err(|e| e.to_string())?
        .map_err(|e| e.to_string())?;
    let words: Vec<f32> = bytemuck::cast_slice(&slice.get_mapped_range()).to_vec();
    staging.unmap();

    let w = W as usize;
    let h = H as usize;
    let full = w * h;
    for plane in 0..4usize {
        let mut buf = vec![0f32; full];
        for y in 0..h {
            let start = plane * full + y * w;
            buf[y * w..(y + 1) * w].copy_from_slice(&words[start..start + w]);
        }
        out.plane_mut(plane)
            .write_rect(Rect::new(0, 0, w as i32, h as i32), &buf);
    }
    Ok(())
}

#[test]
fn probe_writes_a_read_write_storage_buffer_at_binding_one() {
    let Ok(ctx) = pollster::block_on(GpuContext::new(true)) else {
        eprintln!("skipping: no GPU");
        return;
    };

    let dst_buffer = probe_dst_buffer(&ctx);

    // Same shape as the compositor's fixtures.
    let mut backdrop = Surface::new(W as usize, H as usize, Depth::F32, PixelFormat::RGBA);
    let mut source = Surface::new(W as usize, H as usize, Depth::F32, PixelFormat::RGBA);
    for (i, v) in [0.0f32, 0.0, 0.0, 1.0].into_iter().enumerate() {
        backdrop
            .plane_mut(i)
            .fill_rect(Rect::new(0, 0, W as i32, H as i32), v);
    }
    for (i, v) in [1.0f32, 1.0, 1.0, 1.0].into_iter().enumerate() {
        source
            .plane_mut(i)
            .fill_rect(Rect::new(0, 0, W as i32, H as i32), v);
    }
    // `target` exists so `upload_full`/`LayerGpu::new` paths are exercised; the
    // probe writes into `dst_buffer` instead.
    let target = LayerGpu::new(&ctx.device, 0, W, H, 4, "target");
    target
        .upload_rect(&ctx.queue, &backdrop, target.full_rect())
        .unwrap();
    let src_gpu = upload_full(&ctx.device, &ctx.queue, &source, "src");

    let module = ctx
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("probe"),
            source: wgpu::ShaderSource::Wgsl(PROBE.into()),
        });
    let bgl = probe_bgl(&ctx);
    let pll = ctx
        .device
        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });
    let pipeline = ctx
        .device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: Some(&pll),
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

    let params_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("probe-params"),
        size: 32,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let blend_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("probe-blend"),
        size: 16,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    ctx.queue.write_buffer(
        &params_buf,
        0,
        &probe_params_bytes([0, 0, W as i32, H as i32]),
    );
    ctx.queue.write_buffer(&blend_buf, 0, &[1u8; 16]);

    let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: params_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: dst_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: src_gpu.buffer().as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: src_gpu.buffer().as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: blend_buf.as_entire_binding(),
            },
        ],
    });

    let mut enc = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    {
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bg, &[]);
        // One thread per pixel. With workgroup_size(64), a 16x16 grid needs
        // 16*16/64 = 4 workgroups, dispatched as (16, 1) in 2D so that
        // gid.x runs 0..15 and gid.y runs 0..0 — which is wrong. The shader is
        // 2D, so the grid must be (W, H) workgroups of 64 threads each: that
        // covers every (x, y) pair, because gid = (x, y, 0) regardless of how
        // the threads are packed into workgroups.
        pass.dispatch_workgroups(W, H, 1);
    }
    ctx.queue.submit([enc.finish()]);
    let _ = ctx.device.poll(wgpu::PollType::Wait);

    let mut out = Surface::new(W as usize, H as usize, Depth::F32, PixelFormat::RGBA);
    read_planes(&ctx.device, &ctx.queue, &dst_buffer, &mut out).unwrap();

    let r = out.plane(0).sample_normalized(8, 8);
    assert!(
        (r - 1.0).abs() < 1e-5,
        "a read_write storage buffer at binding 1 was never written (r={r})."
    );
}
