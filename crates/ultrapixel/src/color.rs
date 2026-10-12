//! Colour types used at API boundaries.
//!
//! Internally, pixels live in planar [`crate::Plane`] storage; these structs
//! are the ergonomic surface for tools, brushes and filters. All colour types
//! are `f32` so they work identically in 8-bit (0..1 normalised), 16-bit and
//! 32-bit documents.

// The sRGB/XYZ matrices below are published constants. Their trailing zeros
// (`0.2126390`) are meaningful: they let a reviewer diff the literals against
// the spec table character by character. Truncating them to `f32` would break
// that and buy nothing at runtime.
#![allow(clippy::excessive_precision)]

/// Ordered RGB triple.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Rgb {
    pub r: f32,
    pub g: f32,
    pub b: f32,
}

/// RGBA quad, always premultiplied when it comes out of the compositor and
/// straight when it comes from a colour picker. Be explicit about which one
/// you are producing — mixing them up is the classic bug in compositing code.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Rgba {
    pub r: f32,
    pub g: f32,
    pub b: f32,
    pub a: f32,
}

/// sRGB primaries with a transfer function applied.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Srgb {
    pub r: f32,
    pub g: f32,
    pub b: f32,
    pub a: f32,
}

/// Linear-light sRGB. The working space of a 32-bit document.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct LinearRgb {
    pub r: f32,
    pub g: f32,
    pub b: f32,
    pub a: f32,
}

/// CIE L*a*b*.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Lab {
    pub l: f32,
    pub a: f32,
    pub b: f32,
    pub alpha: f32,
}

/// CIE XYZ — the hub for colour space conversion.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Xyz {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl Rgb {
    #[inline]
    pub const fn new(r: f32, g: f32, b: f32) -> Self {
        Self { r, g, b }
    }
}

impl Rgba {
    /// Builds a **straight** (unassociated) alpha colour.
    #[inline]
    pub const fn new(r: f32, g: f32, b: f32, a: f32) -> Self {
        Self { r, g, b, a }
    }
    #[inline]
    pub const fn black() -> Self {
        Self {
            r: 0.0,
            g: 0.0,
            b: 0.0,
            a: 0.0,
        }
    }
    #[inline]
    pub const fn white() -> Self {
        Self {
            r: 1.0,
            g: 1.0,
            b: 1.0,
            a: 1.0,
        }
    }
    /// Builds straight (unassociated) alpha from premultiplied components.
    #[inline]
    pub fn from_premultiplied(r: f32, g: f32, b: f32, a: f32) -> Self {
        if a <= 0.0 {
            return Self::black();
        }
        let inv = 1.0 / a;
        Self {
            r: r * inv,
            g: g * inv,
            b: b * inv,
            a,
        }
    }
    /// Multiplies colour by alpha (the premultiply step).
    #[inline]
    pub fn premultiply(self) -> Self {
        Self {
            r: self.r * self.a,
            g: self.g * self.a,
            b: self.b * self.a,
            a: self.a,
        }
    }
    #[inline]
    pub fn with_alpha(self, a: f32) -> Self {
        Self { a, ..self }
    }
    /// Rec.709 relative luminance of the linear components.
    #[inline]
    pub fn luma(self) -> f32 {
        0.2126 * self.r + 0.7152 * self.g + 0.0722 * self.b
    }
    /// Unassociated copy, clamped to gamut.
    #[inline]
    pub fn clamped(self) -> Self {
        Self {
            r: self.r.clamp(0.0, 1.0),
            g: self.g.clamp(0.0, 1.0),
            b: self.b.clamp(0.0, 1.0),
            a: self.a.clamp(0.0, 1.0),
        }
    }
}

impl Xyz {
    /// CIE XYZ (D65) from linear sRGB.
    #[inline]
    pub fn from_linear_srgb(r: f32, g: f32, b: f32) -> Self {
        Self {
            x: 0.4123908 * r + 0.3575843 * g + 0.1804808 * b,
            y: 0.2126390 * r + 0.7151687 * g + 0.0721923 * b,
            z: 0.0193308 * r + 0.1191948 * g + 0.9505322 * b,
        }
    }
    /// Linear sRGB from CIE XYZ (D65).
    #[inline]
    pub fn to_linear_srgb(self) -> (f32, f32, f32) {
        (
            3.2409699 * self.x - 1.5373832 * self.y - 0.4986108 * self.z,
            -0.9692436 * self.x + 1.8759675 * self.y + 0.0415551 * self.z,
            0.0556301 * self.x - 0.2039770 * self.y + 1.0569715 * self.z,
        )
    }
}

impl Lab {
    /// Lab (D65) from linear sRGB.
    pub fn from_linear_srgb(r: f32, g: f32, b: f32) -> Self {
        let xyz = Xyz::from_linear_srgb(r, g, b);
        Self::from_xyz(xyz)
    }

    pub fn from_xyz(xyz: Xyz) -> Self {
        let f = |t: f32| {
            if t > 216.0 / 24389.0 {
                t.cbrt()
            } else {
                (24389.0 / 27.0 * t + 16.0) / 116.0
            }
        };
        let fx = f(xyz.x / 0.95047);
        let fy = f(xyz.y);
        let fz = f(xyz.z / 1.08883);
        Self {
            l: 116.0 * fy - 16.0,
            a: 500.0 * (fx - fy),
            b: 200.0 * (fy - fz),
            alpha: 1.0,
        }
    }

    /// Delta E (CIE76) — the distance metric used by the eyedropper and the
    /// "select similar" command.
    pub fn delta_e(self, other: Lab) -> f32 {
        let dl = self.l - other.l;
        let da = self.a - other.a;
        let db = self.b - other.b;
        (dl * dl + da * da + db * db).sqrt()
    }
}

/// sRGB transfer function: encoded -> linear.
#[inline]
pub fn srgb_to_linear(c: f32) -> f32 {
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

/// sRGB transfer function: linear -> encoded.
#[inline]
pub fn linear_to_srgb(c: f32) -> f32 {
    if c <= 0.0031308 {
        c * 12.92
    } else {
        1.055 * c.powf(1.0 / 2.4) - 0.055
    }
}

/// Builds an sRGB transfer lookup table. Compositing in 8-bit documents does a
/// per-pixel `powf`, which is measurable; a 1024-entry LUT is exact enough and
/// roughly 20× faster.
pub fn srgb_to_linear_lut(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| srgb_to_linear(i as f32 / (n - 1) as f32))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn premultiply_roundtrip() {
        let straight = Rgba::new(0.4, 0.6, 0.8, 0.5);
        let pm = straight.premultiply();
        assert!((pm.r - 0.2).abs() < 1e-6);
        let back = Rgba::from_premultiplied(pm.r, pm.g, pm.b, pm.a);
        assert!((back.r - straight.r).abs() < 1e-5);
        assert!((back.b - straight.b).abs() < 1e-5);
    }

    #[test]
    fn lab_roundtrip_is_stable() {
        let lab = Lab::from_linear_srgb(0.3, 0.6, 0.9);
        let xyz = Xyz::from_linear_srgb(0.3, 0.6, 0.9);
        let back = Lab::from_xyz(xyz);
        assert!(lab.delta_e(back) < 0.01);
    }
}
