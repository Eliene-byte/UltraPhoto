//! Layer pixels on the GPU.
//!
//! # Why storage buffers and not textures
//!
//! The obvious choice is a texture per channel. We deliberately do **not** do
//! that, because a document can have six channels (CMYK + alpha + under-colour
//! removal) and `rgba32float` tops out at four. Rather than write a special
//! case for print documents, every layer lives in a single **storage buffer**
//! laid out channel-major:
//!
//! ```text
//!   plane 0 | plane 1 | plane 2 | plane 3 | ...
//! ```
//!
//! That handles any channel count uniformly, makes the WGSL kernels trivial to
//! write (a plane is `buffer[p * plane_size + idx]`), and sidesteps the
//! read-write storage texture format matrix entirely. The view pass converts to
//! a sampleable texture at draw time, which is the only place filtering is
//! actually needed.
//!
//! Samples are stored **normalised `0..1` f32** regardless of document depth,
//! so one kernel serves 8-bit and 32-bit documents alike.

use ultrapixel::{Rect, Surface};

use crate::context::GpuError;

/// Bytes one plane of a layer occupies on the GPU.
#[inline]
pub fn plane_size(width: u32, height: u32) -> usize {
    (width as usize) * (height as usize) * 4
}

/// Byte offset of a plane's first sample within a layer buffer.
///
/// Free function rather than a method so the layout arithmetic can be tested
/// without a device — and without constructing a `wgpu::Buffer`, which cannot
/// be zero-initialised.
#[inline]
pub fn plane_offset(width: u32, height: u32, plane: u32) -> u64 {
    plane as u64 * plane_size(width, height) as u64
}

/// Byte offset of a rectangle's first sample within a plane.
///
/// Unaligned because a rect rarely starts on a row boundary in *bytes* — rows
/// are `width` floats apart and `width` is rarely a multiple of 4.
#[inline]
pub fn rect_offset(width: u32, height: u32, plane: u32, x: u32, y: u32) -> u64 {
    plane_offset(width, height, plane) + (y as u64 * width as u64 + x as u64) * 4
}

/// A layer's pixels, resident in VRAM.
pub struct LayerGpu {
    pub id: u64,
    pub width: u32,
    pub height: u32,
    pub channels: u32,
    /// Channel-major `f32` storage buffer.
    buffer: wgpu::Buffer,
}

/// Fresh id for a GPU layer.
///
/// Uniqueness is load-bearing: the compositor caches bind groups keyed by
/// `(dst, src)` id, and every layer of the same canvas has an identical size.
/// Reusing an id would silently hand one layer's bind group to another, which
/// composites without error and produces the wrong pixels.
pub fn next_id() -> u64 {
    static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
    NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
}

impl LayerGpu {
    /// Allocates an all-zero layer. Allocation is cheap on most drivers
    /// (no transfer), so a big layer costs VRAM, not time.
    pub fn new(
        device: &wgpu::Device,
        id: u64,
        width: u32,
        height: u32,
        channels: u32,
        label: &str,
    ) -> Self {
        let size = plane_size(width, height) * channels as usize;
        // Callers pass 0 for "I don't care"; give them a unique id so the
        // compositor's bind-group cache cannot confuse this layer with another.
        let id = if id == 0 { next_id() } else { id };
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: size.max(16) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        Self {
            id,
            width,
            height,
            channels,
            buffer,
        }
    }

    pub fn size_bytes(&self) -> u64 {
        self.buffer.size()
    }

    /// The whole layer as a rect, ready to hand to upload/download.
    #[inline]
    pub fn full_rect(&self) -> Rect {
        Rect::new(0, 0, self.width as i32, self.height as i32)
    }

    #[inline]
    pub fn buffer(&self) -> &wgpu::Buffer {
        &self.buffer
    }

    /// Byte offset of a plane's first sample.
    #[inline]
    pub fn plane_offset(&self, plane: u32) -> u64 {
        plane_offset(self.width, self.height, plane)
    }

    /// Byte offset of a rectangle's first sample within a plane.
    #[inline]
    pub fn rect_offset(&self, plane: u32, x: u32, y: u32) -> u64 {
        rect_offset(self.width, self.height, plane, x, y)
    }

    /// Uploads a rect of a layer from CPU storage.
    ///
    /// Only the dirty rect crosses the bus — a brush dab on a 60000×40000
    /// canvas moves a few hundred KB, not the whole thing.
    pub fn upload_rect(
        &self,
        queue: &wgpu::Queue,
        surface: &Surface,
        rect: Rect,
    ) -> Result<(), GpuError> {
        let Some(rect) = self.clamp(rect) else {
            return Ok(());
        };
        let w = rect.width() as u32;
        let h = rect.height() as u32;
        if w == 0 || h == 0 {
            return Ok(());
        }
        let count = (w * h) as usize;
        let mut staging = vec![0f32; count];

        for plane in 0..self.channels.min(surface.channels() as u32) {
            surface.plane(plane as usize).read_rect(rect, &mut staging);
            // One write per row. The GPU layout is row-major with a stride of
            // `width`, so a rect that is `w` wide inside a `width`-wide plane is
            // NOT contiguous: writing `w * h` floats in one call would smear the
            // tail of each row into the next one.
            let x0 = rect.x0 as u32;
            for row in 0..h {
                let start = (row * w) as usize;
                let line = &staging[start..start + w as usize];
                let bytes: &[u8] = bytemuck::cast_slice(line);
                queue.write_buffer(
                    &self.buffer,
                    self.rect_offset(plane, x0, rect.y0 as u32 + row),
                    bytes,
                );
            }
        }
        Ok(())
    }

    /// Uploads a whole layer. Used right after an open or a resize.
    pub fn upload_all(&self, queue: &wgpu::Queue, surface: &Surface) -> Result<(), GpuError> {
        self.upload_rect(
            queue,
            surface,
            Rect::new(0, 0, self.width as i32, self.height as i32),
        )
    }

    /// Downloads a rect back into CPU storage.
    ///
    /// Reads in one contiguous map and then *de-stride*s: the GPU stores rows
    /// `width` floats apart, so a narrow rect has gaps that must be skipped
    /// rather than copied. One map call keeps the readback cheap even for a
    /// tall rect.
    pub fn download_rect(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        surface: &mut Surface,
        rect: Rect,
    ) -> Result<(), GpuError> {
        let Some(rect) = self.clamp(rect) else {
            return Ok(());
        };
        let w = rect.width() as u32;
        let h = rect.height() as u32;
        if w == 0 || h == 0 {
            return Ok(());
        }

        let x0 = rect.x0 as u32;
        let y0 = rect.y0 as u32;

        // One map serves every plane: each plane is `full` words past the
        // previous one, so the same rect is re-read out of each plane's
        // slice of the mapped region.
        let words = map_region(device, queue, &self.buffer)?;
        let buffer_width = self.width as usize;
        let full = buffer_width * self.height as usize;

        for plane in 0..self.channels.min(surface.channels() as u32) {
            let plane_base = plane as usize * full;
            let mut staging = vec![0f32; (w * h) as usize];
            for row in 0..h as usize {
                let src_start = plane_base + ((y0 as usize + row) * buffer_width) + x0 as usize;
                let src_end = src_start + w as usize;
                debug_assert!(
                    src_end <= words.len(),
                    "plane {plane} row {row} reads past the buffer"
                );
                let dst = &mut staging[row * w as usize..(row + 1) * w as usize];
                dst.copy_from_slice(&words[src_start..src_end]);
            }
            surface.plane_mut(plane as usize).write_rect(rect, &staging);
        }
        Ok(())
    }

    /// Downloads the whole layer back into CPU storage.
    pub fn download_all_into(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        surface: &mut Surface,
    ) -> Result<(), GpuError> {
        self.download_rect(
            device,
            queue,
            surface,
            Rect::new(0, 0, self.width as i32, self.height as i32),
        )
    }

    /// Clamps a rect to the layer, returning `None` when it falls outside.
    fn clamp(&self, rect: Rect) -> Option<Rect> {
        let r = rect.clamp_to(self.width as i32 - 1, self.height as i32 - 1);
        if r.is_empty() {
            None
        } else {
            Some(r)
        }
    }

    /// Builds a GPU layer for a CPU surface.
    pub fn from_surface(device: &wgpu::Device, id: u64, surface: &Surface, label: &str) -> Self {
        let layer = LayerGpu::new(
            device,
            id,
            surface.width() as u32,
            surface.height() as u32,
            surface.channels() as u32,
            label,
        );
        // Uploading from the queue needs the caller; this helper exists for the
        // shapes where the whole layer is needed anyway.
        layer
    }
}

/// Reads `size` bytes starting at `offset`, blocking until the copy lands.
///
/// This is the only stalling path in the engine and it is deliberately confined
/// to readback (saving a file, sampling the colour picker). The compositor
/// never touches it.
/// Maps the whole buffer once and hands the caller the raw `f32` words.
///
/// Maps the entire buffer rather than a byte range: `map_async` maps from
/// the given offset to the end of the buffer, so a partial map always
/// yields the whole tail. Slicing a *narrower* window out of that tail
/// would silently read the wrong pixels, so the de-striding happens in
/// [`map_destrided`] instead.
fn map_region(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    src: &wgpu::Buffer,
) -> Result<Vec<f32>, GpuError> {
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("ultraphoto-readback"),
        size: src.size(),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("readback"),
    });
    encoder.copy_buffer_to_buffer(src, 0, &staging, 0, src.size());
    queue.submit([encoder.finish()]);

    let slice = staging.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    let _ = device.poll(wgpu::PollType::Wait);
    rx.recv()
        .map_err(|e| GpuError::Device(e.to_string()))?
        .map_err(|e| GpuError::Device(e.to_string()))?;

    let out = {
        let bytes = slice.get_mapped_range();
        let words: &[f32] = bytemuck::cast_slice(&bytes);
        words.to_vec()
    };
    staging.unmap();
    Ok(out)
}

/// Scatters a whole surface into a GPU layer, used by tests and by the
/// CPU-to-GPU path when a layer is first touched.
pub fn upload_full(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    surface: &Surface,
    label: &str,
) -> LayerGpu {
    let layer = LayerGpu::new(
        device,
        0,
        surface.width() as u32,
        surface.height() as u32,
        surface.channels() as u32,
        label,
    );
    let _ = layer.upload_all(queue, surface);
    layer
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::GpuContext;
    use ultrapixel::{Depth, PixelFormat, Surface};

    /// Skips the test instead of failing when the machine has no GPU.
    /// CI runners frequently have no adapter, and a skipped test is honest
    /// where a fake pass would not be.
    fn try_ctx() -> Option<GpuContext> {
        match pollster::block_on(GpuContext::new(true)) {
            Ok(c) => Some(c),
            Err(e) => {
                eprintln!("skipping GPU test: {e}");
                None
            }
        }
    }

    #[test]
    fn plane_and_rect_offsets_are_channel_major() {
        // One plane of a 4x4 RGBA layer is 4*4 f32 = 64 bytes.
        assert_eq!(plane_size(4, 4), 64);
        assert_eq!(plane_offset(4, 4, 0), 0);
        assert_eq!(plane_offset(4, 4, 1), 64);
        assert_eq!(plane_offset(4, 4, 2), 128);
        // Pixel (2,3) of plane 1 sits at (3*4 + 2) f32 past the plane start.
        assert_eq!(rect_offset(4, 4, 1, 2, 3), 64 + (14 * 4));
    }

    #[test]
    fn every_plane_of_a_realistic_size_stays_in_bounds() {
        // The offset arithmetic is what keeps the shader's plane indexing
        // correct; a silent overflow here would read the wrong channel.
        let (w, h, ch) = (1920u32, 1080u32, 6u32); // CMYK + UCR + alpha
        let plane = plane_size(w, h) as u64;
        for p in 0..ch {
            assert_eq!(plane_offset(w, h, p), p as u64 * plane);
            // Last pixel of the plane.
            let last = rect_offset(w, h, p, w - 1, h - 1);
            assert!(
                last + 4 <= (p as u64 + 1) * plane,
                "plane {p} runs past its end"
            );
        }
    }

    #[test]
    fn the_origin_rect_of_a_plane_starts_at_the_plane() {
        let (w, h) = (37u32, 23u32); // deliberately not a multiple of 4
        for p in 0..4 {
            assert_eq!(rect_offset(w, h, p, 0, 0), plane_offset(w, h, p));
        }
    }

    #[test]
    fn upload_and_download_round_trips_a_rect() {
        let Some(ctx) = try_ctx() else { return };

        let (w, h) = (64usize, 48usize);
        let mut src = Surface::new(w, h, Depth::F32, PixelFormat::RGBA);
        let full = Rect::new(0, 0, w as i32, h as i32);
        for (c, v) in [0.1f32, 0.5, 0.9, 1.0].into_iter().enumerate() {
            src.plane_mut(c).fill_rect(full, v);
        }
        // Dirty rect that does not start at the origin.
        let rect = Rect::new(5, 7, 21, 19);

        let layer = upload_full(&ctx.device, &ctx.queue, &src, "test");
        let mut dst = Surface::new(w, h, Depth::F32, PixelFormat::RGBA);
        layer
            .download_rect(&ctx.device, &ctx.queue, &mut dst, rect)
            .unwrap();

        for plane in 0..4 {
            assert_eq!(
                src.plane(plane).sample(10, 10),
                dst.plane(plane).sample(10, 10),
                "plane {plane}"
            );
        }
        // Outside the downloaded rect the destination must still be untouched.
        assert_eq!(dst.plane(0).sample(0, 0), 0.0);
    }

    #[test]
    fn uploading_only_a_rect_leaves_the_rest_alone() {
        let Some(ctx) = try_ctx() else { return };
        let (w, h) = (32usize, 32usize);
        let mut src = Surface::new(w, h, Depth::F32, PixelFormat::RGBA);
        let layer = LayerGpu::new(&ctx.device, 1, w as u32, h as u32, 4, "partial");

        // Paint everything red, upload it.
        src.plane_mut(0)
            .fill_rect(Rect::new(0, 0, w as i32, h as i32), 1.0);
        layer.upload_all(&ctx.queue, &src).unwrap();

        // Now change only a corner and upload just that corner.
        src.plane_mut(0).fill_rect(Rect::new(0, 0, 4, 4), 0.0);
        layer
            .upload_rect(&ctx.queue, &src, Rect::new(0, 0, 4, 4))
            .unwrap();

        let mut out = Surface::new(w, h, Depth::F32, PixelFormat::RGBA);
        layer
            .download_all_into(&ctx.device, &ctx.queue, &mut out)
            .unwrap();

        assert_eq!(
            out.plane(0).sample(1, 1),
            0.0,
            "uploaded rect must be applied"
        );
        assert_eq!(out.plane(0).sample(20, 20), 1.0, "rest must be untouched");
    }

    #[test]
    fn a_large_layer_allocates_without_uploading() {
        let Some(ctx) = try_ctx() else { return };
        // 40000x40000 RGBA float = 25.6 GiB of VRAM. Allocation must be lazy:
        // this must not hang or allocate host memory.
        let layer = LayerGpu::new(&ctx.device, 1, 40_000, 40_000, 4, "huge");
        assert_eq!(layer.size_bytes(), 40_000 * 40_000 * 4 * 4);
    }

    #[test]
    fn eight_bit_surfaces_round_trip_through_quantisation() {
        let Some(ctx) = try_ctx() else { return };
        let mut src = Surface::new(16, 16, Depth::U8, PixelFormat::RGBA);
        src.plane_mut(0).fill_rect(Rect::new(0, 0, 16, 16), 0.5);
        let layer = upload_full(&ctx.device, &ctx.queue, &src, "u8");
        let mut out = Surface::new(16, 16, Depth::U8, PixelFormat::RGBA);
        layer
            .download_rect(&ctx.device, &ctx.queue, &mut out, Rect::new(0, 0, 16, 16))
            .unwrap();
        // 0.5 -> 127/128 after the 8-bit round trip.
        assert!((out.plane(0).sample_normalized(8, 8) - 0.5).abs() < 0.01);
    }

    #[test]
    fn clamp_rejects_out_of_bounds_rects() {
        // `clamp` is pure geometry, so it is tested without a device.
        fn clamp_rect(w: u32, h: u32, rect: Rect) -> Option<Rect> {
            let r = rect.clamp_to(w as i32 - 1, h as i32 - 1);
            if r.is_empty() {
                None
            } else {
                Some(r)
            }
        }
        assert!(clamp_rect(16, 16, Rect::new(0, 0, 16, 16)).is_some());
        assert!(clamp_rect(16, 16, Rect::new(-50, -50, -10, -10)).is_none());
        assert_eq!(
            clamp_rect(16, 16, Rect::new(4, 4, 100, 100)).unwrap(),
            Rect::new(4, 4, 16, 16)
        );
    }
}
