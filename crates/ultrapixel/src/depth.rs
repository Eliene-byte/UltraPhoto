//! Bit depth of a document.

/// Bit depth of every plane in a document.
///
/// Photoshop uses the same three depths and we mirror the naming so muscle
/// memory transfers over. Note that 32-bit documents are stored as `f32`
/// (linear-light capable), *not* 32-bit integers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum Depth {
    /// 8 bits per sample, 0..=255. Fastest path, smallest memory.
    #[default]
    U8,
    /// 16 bits per sample, 0..=65535. The default for serious work.
    U16,
    /// 32-bit floating point. Unclamped headroom for compositing and filters.
    F32,
}

impl Depth {
    /// Bytes occupied by a single sample.
    #[inline]
    pub const fn sample_bytes(self) -> usize {
        match self {
            Depth::U8 => 1,
            Depth::U16 => 2,
            Depth::F32 => 4,
        }
    }

    #[inline]
    pub const fn bits(self) -> usize {
        self.sample_bytes() * 8
    }

    /// Highest representable integer value, or `f32::INFINITY` for float depth.
    #[inline]
    pub const fn max_value(self) -> f32 {
        match self {
            Depth::U8 => 255.0,
            Depth::U16 => 65535.0,
            Depth::F32 => f32::INFINITY,
        }
    }

    /// Maps a normalised `0.0..=1.0` value onto the native sample range.
    #[inline]
    pub fn from_normalized(self, v: f32) -> f32 {
        match self {
            Depth::U8 => (v.clamp(0.0, 1.0) * 255.0).round(),
            Depth::U16 => (v.clamp(0.0, 1.0) * 65535.0).round(),
            Depth::F32 => v,
        }
    }

    #[inline]
    pub fn to_normalized(self, v: f32) -> f32 {
        match self {
            Depth::U8 => v / 255.0,
            Depth::U16 => v / 65535.0,
            Depth::F32 => v,
        }
    }

    /// Quantises a float sample into the storage type for this depth.
    #[inline]
    pub fn quantize(self, v: f32) -> f32 {
        match self {
            Depth::F32 => v,
            Depth::U8 | Depth::U16 => {
                let m = self.max_value();
                if v.is_nan() {
                    0.0
                } else {
                    v.clamp(0.0, m)
                }
            }
        }
    }

    #[inline]
    pub const fn is_float(self) -> bool {
        matches!(self, Depth::F32)
    }
}
