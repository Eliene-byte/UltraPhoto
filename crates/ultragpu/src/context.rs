//! GPU device discovery and initialisation.
//!
//! UltraPhoto is GPU-first: if a usable adapter exists, filtering and
//! compositing run on compute shaders. If not, the caller falls back to the
//! CPU reference in [`ultrablend`] — correctness never depends on a GPU being
//! present, which is what lets CI run on machines without one.

use wgpu::util::DeviceExt;

/// Everything the status bar and the diagnostics overlay show.
#[derive(Debug, Clone, Default)]
pub struct GpuInfo {
    /// PCI vendor ID as reported by the backend.
    pub vendor_id: u32,
    pub device_id: u32,
    pub device: String,
    pub description: String,
    pub backend: String,
    pub is_fallback: bool,
    /// True when the adapter exposes enough features for the full kernel set.
    pub capable: bool,
    /// Why the adapter was rejected, when `capable` is false.
    pub rejection: Option<String>,
}

impl GpuInfo {
    /// One-line summary for the About box.
    pub fn summary(&self) -> String {
        if !self.capable {
            return format!(
                "No GPU acceleration ({}) - CPU fallback active",
                self.rejection.as_deref().unwrap_or("unknown")
            );
        }
        let dev = if self.device.is_empty() {
            self.description.clone()
        } else {
            self.device.clone()
        };
        format!("{} ({})", dev, self.backend)
    }
}

#[derive(Debug, thiserror::Error)]
pub enum GpuError {
    #[error("no compatible GPU adapter found")]
    NoAdapter,
    #[error("could not create a device: {0}")]
    Device(String),
    #[error("GPU feature is unavailable: {0}")]
    MissingFeature(&'static str),
}

/// Owns the wgpu instance, adapter, device and queue.
///
/// Kept as one unit on purpose: every pipeline in the app shares these
/// resources, and splitting them invites ordering bugs between the UI and the
/// compositor.
pub struct GpuContext {
    pub instance: wgpu::Instance,
    pub adapter: wgpu::Adapter,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub info: GpuInfo,
    /// Adapter limits, kept for the diagnostics overlay.
    pub limits: wgpu::Limits,
}

impl GpuContext {
    /// Picks the best available adapter.
    ///
    /// `prefer_fallback` is for headless CI and software rasterisers; on a real
    /// machine we always want the discrete GPU.
    pub async fn new(prefer_fallback: bool) -> Result<Self, GpuError> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::PRIMARY | wgpu::Backends::SECONDARY,
            ..Default::default()
        });

        let mut options = wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: None,
        };
        let adapter = match instance.request_adapter(&options).await {
            Ok(a) => a,
            Err(first_err) => {
                if !prefer_fallback {
                    return Err(GpuError::NoAdapter);
                }
                tracing::info!(error = %first_err, "no hardware adapter; trying software");
                options.force_fallback_adapter = true;
                options.power_preference = wgpu::PowerPreference::LowPower;
                instance
                    .request_adapter(&options)
                    .await
                    .map_err(|_| GpuError::NoAdapter)?
            }
        };

        let adapter_info = adapter.get_info();
        let is_fallback = adapter_info.device_type == wgpu::DeviceType::Cpu;

        // Ask for the adapter's limits so huge canvases do not silently fail
        // to allocate. A 60000x40000 RGBA32F layer is ~38 GiB, so this matters.
        let limits = adapter.limits();
        let features = adapter.features();

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("ultraphoto-device"),
                required_features: features,
                required_limits: limits.clone(),
                memory_hints: wgpu::MemoryHints::Performance,
                trace: wgpu::Trace::Off,
            })
            .await
            .map_err(|e| GpuError::Device(e.to_string()))?;

        // Surface an uncaptured-error hook instead of letting errors vanish.
        device.on_uncaptured_error(Box::new(|err| {
            tracing::error!(error = %err, "wgpu error");
        }));

        let backend = format!("{:?}", adapter_info.backend);
        let info = GpuInfo {
            vendor_id: adapter_info.vendor,
            device_id: adapter_info.device,
            device: adapter_info.name.clone(),
            description: adapter_info.backend.to_string(),
            backend,
            is_fallback,
            // Software adapters are "capable" but slow; we still use them so CI
            // exercises the real code path.
            capable: true,
            rejection: None,
        };

        Ok(Self {
            instance,
            adapter,
            device,
            queue,
            info,
            limits,
        })
    }

    /// True when a real hardware GPU was found (not a software rasteriser).
    #[inline]
    pub fn is_hardware(&self) -> bool {
        !self.info.is_fallback
    }

    /// Largest storage buffer a single layer may occupy.
    ///
    /// Reported in the diagnostics overlay so an "out of memory" failure on a
    /// huge canvas is explainable rather than mysterious.
    pub fn max_layer_pixels(&self) -> u64 {
        // 4 f32 planes is the widest layout we address in one buffer.
        let bytes = self.limits.max_storage_buffer_binding_size as u64;
        bytes / 16
    }
}

/// Creates a buffer initialised with `data`.
pub fn create_buffer_init(
    device: &wgpu::Device,
    label: &str,
    usage: wgpu::BufferUsages,
    data: &[u8],
) -> wgpu::Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents: data,
        usage,
    })
}

/// Workgroups needed to cover `count` threads with `workgroup_size(64)`.
///
/// NOTE: this counts *threads*, so a shader whose `global_invocation_id.x`
/// range is 0..count needs `div_ceil(count, 64)` workgroups. It is correct for
/// a 1D dispatch and for the X axis of a 2D dispatch where each workgroup's
/// threads run along X. It is NOT correct for a 2D shader that indexes `y`
/// by workgroup, which is what the compositor does — see
/// [`workgroups_2d`] for that case.
#[inline]
pub fn workgroups_for(count: u32) -> u32 {
    count.max(1).div_ceil(64)
}

/// Workgroups for a 2D shader that maps `gid.xy` to pixels directly.
///
/// With `@workgroup_size(64)` and a 2D dispatch of `(gw, gh, 1)`, each
/// workgroup contributes a 64-thread run along X, so a `gw * gh` grid only
/// covers a `(64 * gw, gh)` pixel rectangle. To cover `(w, h)` pixels with one
/// thread each the dispatch must be `(w, h, 1)`, and each dimension is its own
/// workgroup count regardless of the workgroup size — because a 2D workgroup's
/// `gid.xy` = `(workgroup_id.xy * size + local_id.xy)`.
#[inline]
pub fn workgroups_2d(width: u32, height: u32) -> (u32, u32) {
    (width.max(1), height.max(1))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn workgroup_math_rounds_up_and_never_returns_zero() {
        assert_eq!(workgroups_for(0), 1, "a zero dispatch must still be valid");
        assert_eq!(workgroups_for(1), 1);
        assert_eq!(workgroups_for(64), 1);
        assert_eq!(workgroups_for(65), 2);
        assert_eq!(workgroups_for(4096), 64);
    }
}
