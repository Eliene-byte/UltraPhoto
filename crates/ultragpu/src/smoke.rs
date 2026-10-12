//! Minimal kernel test: write a constant, verify it lands.
//!
//! Used to isolate "is the pipeline wired correctly at all" from the real
//! compositing logic. If this fails, the problem is bind groups, dispatch size
//! or buffer layout, not the blend maths.

use crate::context::GpuContext;
use crate::layer::LayerGpu;
use ultrapixel::{Depth, PixelFormat, Surface};

const KERNEL: &str = r#"
@group(0) @binding(0) var<storage, read_write> buf: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= arrayLength(&buf)) {
        return;
    }
    buf[i] = 42.0;
}
"#;

#[test]
fn a_trivial_kernel_can_write_a_storage_buffer() {
    let Ok(ctx) = pollster::block_on(GpuContext::new(true)) else {
        eprintln!("skipping: no GPU");
        return;
    };

    let layer = LayerGpu::new(&ctx.device, 0, 4, 4, 4, "trivial");

    let module = ctx
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("trivial"),
            source: wgpu::ShaderSource::Wgsl(KERNEL.into()),
        });
    let bgl = ctx
        .device
        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("trivial-bgl"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
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

    let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("trivial-bg"),
        layout: &bgl,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: layer.buffer().as_entire_binding(),
        }],
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
        // 4*4*1 = 16 floats; one workgroup of 64 threads covers it.
        pass.dispatch_workgroups(16, 1, 1);
    }
    ctx.queue.submit([enc.finish()]);

    let mut out = Surface::new(4, 4, Depth::F32, PixelFormat::GRAY);
    layer
        .download_rect(&ctx.device, &ctx.queue, &mut out, layer.full_rect())
        .unwrap();

    let got = out.plane(0).sample_normalized(0, 0);
    assert!(
        (got - 42.0).abs() < 1e-6,
        "kernel did not write through; got {got}, expected 42.0"
    );
}
