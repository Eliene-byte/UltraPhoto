//! # ultragpu
//!
//! GPU execution for UltraPhoto.
//!
//! The design goal is that **every filter and composite runs as a WGSL compute
//! shader**, with the CPU implementation in [`ultrablend`] kept as a reference
//! and as a fallback for machines without a usable GPU.
//!
//! Two decisions shape everything here:
//!
//! 1. **Storage buffers, not textures, for layer data.** A document can have six
//!    channels (CMYK + alpha + under-colour removal) and storage textures top
//!    out at four. A single channel-major `array<f32>` handles any layout, makes
//!    kernels trivial to write, and avoids the read-write storage texture format
//!    matrix. See [`layer`].
//! 2. **Dirty rectangles everywhere.** Upload, composite and readback all take a
//!    [`Rect`](ultrapixel::Rect), so a brush dab moves a few hundred KB instead
//!    of re-pushing the whole canvas.
//!
//! Kernels are required to match [`ultrablend`] exactly; a cross-check test
//! compares the two pixel-for-pixel so they cannot drift apart.

pub mod compositor;
pub mod context;
pub mod layer;
#[cfg(test)]
mod probe;
#[cfg(test)]
mod smoke;

pub use compositor::{copy_target_to_surface, new_target, Compositor, LayerInput, TARGET_FORMAT};
pub use context::{workgroups_for, GpuContext, GpuError, GpuInfo};
pub use layer::{upload_full, LayerGpu};

#[cfg(test)]
mod tests {
    use super::*;

    /// Builds a context, or returns `None` so a GPU-less CI machine skips the
    /// test instead of failing it.
    #[allow(dead_code)]
    fn ctx() -> Option<GpuContext> {
        match pollster::block_on(GpuContext::new(true)) {
            Ok(c) => Some(c),
            Err(e) => {
                eprintln!("skipping GPU test: {e}");
                None
            }
        }
    }
}
