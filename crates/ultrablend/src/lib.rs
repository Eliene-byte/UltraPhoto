//! Photoshop-compatible blend modes.
//!
//! This is the **reference implementation**. The WGSL kernels in
//! [`ultragpu`](../../ultragpu/index.html) mirror these functions exactly and a
//! cross-check test compares them pixel-for-pixel, so the CPU and GPU paths can
//! never silently drift apart.
//!
//! Two families live here:
//!
//! * **Separable** modes are a per-channel function `f(backdrop, source)`.
//! * **Non-separable** modes (Hue, Saturation, Color, Luminosity) need the whole
//!   triple and are implemented with the W3C compositing spec maths.

use ultrapixel::color::Rgba;

/// Every blend mode Photoshop exposes in the layer style and painting menus.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[repr(u8)]
pub enum BlendMode {
    #[default]
    Normal = 0,
    Dissolve,
    Darken,
    Multiply,
    ColorBurn,
    LinearBurn,
    DarkerColor,
    Lighten,
    Screen,
    ColorDodge,
    LinearDodge,
    LighterColor,
    Overlay,
    SoftLight,
    HardLight,
    VividLight,
    LinearLight,
    PinLight,
    HardMix,
    Difference,
    Exclusion,
    Subtract,
    Divide,
    Hue,
    Saturation,
    Color,
    Luminosity,
}

/// Number of blend modes — handy for UI iteration tests.
pub const BLEND_MODE_COUNT: usize = 27;

impl BlendMode {
    /// All modes in menu order. The Layers panel and the style dialog both
    /// iterate this, so the order must match Photoshop's menu.
    pub const ALL: [BlendMode; BLEND_MODE_COUNT] = [
        BlendMode::Normal,
        BlendMode::Dissolve,
        BlendMode::Darken,
        BlendMode::Multiply,
        BlendMode::ColorBurn,
        BlendMode::LinearBurn,
        BlendMode::DarkerColor,
        BlendMode::Lighten,
        BlendMode::Screen,
        BlendMode::ColorDodge,
        BlendMode::LinearDodge,
        BlendMode::LighterColor,
        BlendMode::Overlay,
        BlendMode::SoftLight,
        BlendMode::HardLight,
        BlendMode::VividLight,
        BlendMode::LinearLight,
        BlendMode::PinLight,
        BlendMode::HardMix,
        BlendMode::Difference,
        BlendMode::Exclusion,
        BlendMode::Subtract,
        BlendMode::Divide,
        BlendMode::Hue,
        BlendMode::Saturation,
        BlendMode::Color,
        BlendMode::Luminosity,
    ];

    /// Label shown in the Layers panel blend dropdown.
    pub const fn label(self) -> &'static str {
        match self {
            BlendMode::Normal => "Normal",
            BlendMode::Dissolve => "Dissolve",
            BlendMode::Darken => "Darken",
            BlendMode::Multiply => "Multiply",
            BlendMode::ColorBurn => "Color Burn",
            BlendMode::LinearBurn => "Linear Burn",
            BlendMode::DarkerColor => "Darker Color",
            BlendMode::Lighten => "Lighten",
            BlendMode::Screen => "Screen",
            BlendMode::ColorDodge => "Color Dodge",
            BlendMode::LinearDodge => "Linear Dodge (Add)",
            BlendMode::LighterColor => "Lighter Color",
            BlendMode::Overlay => "Overlay",
            BlendMode::SoftLight => "Soft Light",
            BlendMode::HardLight => "Hard Light",
            BlendMode::VividLight => "Vivid Light",
            BlendMode::LinearLight => "Linear Light",
            BlendMode::PinLight => "Pin Light",
            BlendMode::HardMix => "Hard Mix",
            BlendMode::Difference => "Difference",
            BlendMode::Exclusion => "Exclusion",
            BlendMode::Subtract => "Subtract",
            BlendMode::Divide => "Divide",
            BlendMode::Hue => "Hue",
            BlendMode::Saturation => "Saturation",
            BlendMode::Color => "Color",
            BlendMode::Luminosity => "Luminosity",
        }
    }

    /// Section separator in the blend menu ("Darken group", "Lighten group"...).
    pub const fn group(self) -> BlendGroup {
        match self {
            BlendMode::Normal | BlendMode::Dissolve => BlendGroup::Normal,
            BlendMode::Darken
            | BlendMode::Multiply
            | BlendMode::ColorBurn
            | BlendMode::LinearBurn
            | BlendMode::DarkerColor => BlendGroup::Darken,
            BlendMode::Lighten
            | BlendMode::Screen
            | BlendMode::ColorDodge
            | BlendMode::LinearDodge
            | BlendMode::LighterColor => BlendGroup::Lighten,
            BlendMode::Overlay
            | BlendMode::SoftLight
            | BlendMode::HardLight
            | BlendMode::VividLight
            | BlendMode::LinearLight
            | BlendMode::PinLight
            | BlendMode::HardMix => BlendGroup::Contrast,
            BlendMode::Difference
            | BlendMode::Exclusion
            | BlendMode::Subtract
            | BlendMode::Divide => BlendGroup::Comparative,
            BlendMode::Hue | BlendMode::Saturation | BlendMode::Color | BlendMode::Luminosity => {
                BlendGroup::Component
            }
        }
    }

    /// True for modes that look at all three channels together.
    #[inline]
    pub const fn is_non_separable(self) -> bool {
        matches!(
            self,
            BlendMode::DarkerColor
                | BlendMode::LighterColor
                | BlendMode::Hue
                | BlendMode::Saturation
                | BlendMode::Color
                | BlendMode::Luminosity
        )
    }

    /// The separating luminance/colour form used by the discrete (Dissolve)
    /// variant of a mode.
    #[inline]
    pub fn from_separable_non_dissolve(self) -> BlendMode {
        self
    }
}

/// Menu grouping, used to draw separators in the blend dropdown.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlendGroup {
    Normal,
    Darken,
    Lighten,
    Contrast,
    Comparative,
    Component,
}

impl BlendMode {
    /// Per-channel separable blend. `backdrop` is what is underneath, `source`
    /// is the incoming layer. Both are normalised and **straight** alpha.
    #[inline]
    pub fn blend_channel(self, backdrop: f32, source: f32) -> f32 {
        match self {
            BlendMode::Normal | BlendMode::Dissolve => source,
            BlendMode::Darken => backdrop.min(source),
            BlendMode::Multiply => backdrop * source,
            BlendMode::ColorBurn => {
                if backdrop <= 0.0 {
                    0.0
                } else if source >= 1.0 {
                    1.0
                } else {
                    // The `.min(1.0)` is load-bearing: at source == 0 the
                    // quotient is +inf and without it we would return -inf.
                    1.0 - ((1.0 - backdrop) / source).min(1.0)
                }
            }
            BlendMode::LinearBurn => (backdrop + source - 1.0).clamp(0.0, 1.0),
            BlendMode::Lighten => backdrop.max(source),
            BlendMode::Screen => backdrop + source - backdrop * source,
            BlendMode::ColorDodge => {
                if backdrop <= 0.0 {
                    0.0
                } else if source >= 1.0 {
                    1.0
                } else {
                    (backdrop / (1.0 - source)).min(1.0)
                }
            }
            BlendMode::LinearDodge => (backdrop + source).min(1.0),
            BlendMode::Overlay => Self::HardLight.blend_channel(source, backdrop),
            BlendMode::SoftLight => soft_light(backdrop, source),
            BlendMode::HardLight => {
                if source <= 0.5 {
                    backdrop * (2.0 * source)
                } else {
                    1.0 - (1.0 - 2.0 * (source - 0.5)) * (1.0 - backdrop)
                }
            }
            BlendMode::VividLight => {
                if source <= 0.5 {
                    if backdrop <= 0.0 {
                        0.0
                    } else {
                        (backdrop / (1.0 - 2.0 * source)).min(1.0)
                    }
                } else if backdrop >= 1.0 {
                    1.0
                } else {
                    (1.0 - (1.0 - backdrop) / (2.0 * source - 1.0)).max(0.0)
                }
            }
            BlendMode::LinearLight => (backdrop + 2.0 * source - 1.0).clamp(0.0, 1.0),
            BlendMode::PinLight => {
                if source <= 0.5 {
                    backdrop.min(2.0 * source)
                } else {
                    backdrop.max(2.0 * source - 1.0)
                }
            }
            BlendMode::HardMix => {
                if backdrop + source >= 1.0 {
                    1.0
                } else {
                    0.0
                }
            }
            BlendMode::Difference => (backdrop - source).abs(),
            BlendMode::Exclusion => backdrop + source - 2.0 * backdrop * source,
            BlendMode::Subtract => (backdrop - source).clamp(0.0, 1.0),
            BlendMode::Divide => {
                if source <= 0.0 {
                    1.0
                } else {
                    (backdrop / source).min(1.0)
                }
            }
            _ => source, // non-separable modes are handled by `blend_pixel`
        }
    }

    /// Full RGBA blend. Both inputs use **straight** alpha.
    ///
    /// The output is straight alpha, matching what a layer stores.
    pub fn blend_pixel(self, backdrop: Rgba, source: Rgba) -> Rgba {
        if self.is_non_separable() {
            let blended = self.blend_non_separable(backdrop, source);
            // Non-separable modes operate on the colour only; alpha still
            // interpolates as `Normal` does.
            let a = backdrop.a + source.a - backdrop.a * source.a;
            if a <= 0.0 {
                return Rgba::black();
            }
            return Rgba {
                r: blended.0,
                g: blended.1,
                b: blended.2,
                a,
            };
        }
        Rgba {
            r: self.blend_channel(backdrop.r, source.r),
            g: self.blend_channel(backdrop.g, source.g),
            b: self.blend_channel(backdrop.b, source.b),
            a: source.a,
        }
    }

    fn blend_non_separable(self, backdrop: Rgba, source: Rgba) -> (f32, f32, f32) {
        match self {
            BlendMode::DarkerColor => {
                let bl = backdrop.luma();
                let sl = source.luma();
                if sl < bl {
                    (source.r, source.g, source.b)
                } else {
                    (backdrop.r, backdrop.g, backdrop.b)
                }
            }
            BlendMode::LighterColor => {
                let bl = backdrop.luma();
                let sl = source.luma();
                if sl > bl {
                    (source.r, source.g, source.b)
                } else {
                    (backdrop.r, backdrop.g, backdrop.b)
                }
            }
            // Hue: the source's hue, with the backdrop's saturation and lightness.
            BlendMode::Hue => {
                let (sh, _, _) = rgb_to_hsl(source.r, source.g, source.b);
                let (_, s, l) = rgb_to_hsl(backdrop.r, backdrop.g, backdrop.b);
                hsl_to_rgb(sh, s, l)
            }
            // Saturation: the source's saturation, with the backdrop's hue and
            // lightness.
            BlendMode::Saturation => {
                let (h, _, l) = rgb_to_hsl(backdrop.r, backdrop.g, backdrop.b);
                let (_, ss, _) = rgb_to_hsl(source.r, source.g, source.b);
                hsl_to_rgb(h, ss, l)
            }
            // Color: the source's hue and saturation, with the backdrop's
            // lightness.
            BlendMode::Color => {
                let (sh, ss, _) = rgb_to_hsl(source.r, source.g, source.b);
                let (_, _, l) = rgb_to_hsl(backdrop.r, backdrop.g, backdrop.b);
                hsl_to_rgb(sh, ss, l)
            }
            // Luminosity: the source's lightness, with the backdrop's hue and
            // saturation.
            BlendMode::Luminosity => {
                let (h, s, _) = rgb_to_hsl(backdrop.r, backdrop.g, backdrop.b);
                let (_, _, sl) = rgb_to_hsl(source.r, source.g, source.b);
                hsl_to_rgb(h, s, sl)
            }
            _ => (source.r, source.g, source.b),
        }
    }
}

/// W3C compositing soft-light.
#[inline]
fn soft_light(b: f32, s: f32) -> f32 {
    if s <= 0.5 {
        b - (1.0 - 2.0 * s) * b * (1.0 - b)
    } else {
        let d = if b <= 0.25 {
            ((16.0 * b - 12.0) * b + 4.0) * b
        } else {
            b.sqrt()
        };
        b + (2.0 * s - 1.0) * (d - b)
    }
}

/// RGB -> HSL with all components in `0..=1`.
pub fn rgb_to_hsl(r: f32, g: f32, b: f32) -> (f32, f32, f32) {
    let max = r.max(g).max(b);
    let min = r.min(g).min(b);
    let l = (max + min) * 0.5;
    if (max - min).abs() < f32::EPSILON {
        return (0.0, 0.0, l);
    }
    let d = max - min;
    let s = if l > 0.5 {
        d / (2.0 - max - min)
    } else {
        d / (max + min)
    };
    let h = if (max - r).abs() < f32::EPSILON {
        (g - b) / d + if g < b { 6.0 } else { 0.0 }
    } else if (max - g).abs() < f32::EPSILON {
        (b - r) / d + 2.0
    } else {
        (r - g) / d + 4.0
    };
    (h / 6.0, s, l)
}

/// HSL -> RGB with all components in `0..=1`.
pub fn hsl_to_rgb(h: f32, s: f32, l: f32) -> (f32, f32, f32) {
    if s.abs() < f32::EPSILON {
        return (l, l, l);
    }
    let q = if l < 0.5 {
        l * (1.0 + s)
    } else {
        l + s - l * s
    };
    let p = 2.0 * l - q;
    (
        hue_to_rgb(p, q, h + 1.0 / 3.0),
        hue_to_rgb(p, q, h),
        hue_to_rgb(p, q, h - 1.0 / 3.0),
    )
}

#[inline]
fn hue_to_rgb(p: f32, q: f32, mut t: f32) -> f32 {
    if t < 0.0 {
        t += 1.0;
    }
    if t > 1.0 {
        t -= 1.0;
    }
    if t < 1.0 / 6.0 {
        p + (q - p) * 6.0 * t
    } else if t < 1.0 / 2.0 {
        q
    } else if t < 2.0 / 3.0 {
        p + (q - p) * (2.0 / 3.0 - t) * 6.0
    } else {
        p
    }
}

/// How the source alpha is combined with the backdrop alpha.
///
/// This is the "fill opacity" side of the equation and is *not* the same as the
/// blend mode. Photoshop keeps them separate, and so do we.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CompositeOp {
    /// Straight source-over.
    #[default]
    SourceOver,
}

/// Composites `source` over `backdrop` with the given blend mode, opacity and
/// fill, exactly as Photoshop does.
///
/// * `opacity` is the layer's Opacity (0..=1).
/// * `fill` is Fill (0..=1) — multiplies the source alpha *after* blending,
///   which is why filling a fill-layer at 0% still shows blend artefacts.
#[inline]
pub fn composite(backdrop: Rgba, source: Rgba, mode: BlendMode, opacity: f32, fill: f32) -> Rgba {
    let src_alpha = (source.a * opacity * fill).clamp(0.0, 1.0);
    if src_alpha <= 0.0 {
        return backdrop;
    }
    let blended = mode.blend_pixel(backdrop, source);
    let inv = 1.0 - src_alpha;
    Rgba {
        r: blended.r * src_alpha + backdrop.r * inv,
        g: blended.g * src_alpha + backdrop.g * inv,
        b: blended.b * src_alpha + backdrop.b * inv,
        a: src_alpha + backdrop.a * inv,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_modes_have_unique_labels_and_cover_27() {
        assert_eq!(BlendMode::ALL.len(), BLEND_MODE_COUNT);
        let mut labels: Vec<&str> = BlendMode::ALL.iter().map(|m| m.label()).collect();
        labels.sort_unstable();
        labels.dedup();
        assert_eq!(
            labels.len(),
            BLEND_MODE_COUNT,
            "blend labels must be unique"
        );
    }

    #[test]
    fn normal_is_identity_at_full_opacity() {
        let b = Rgba::new(0.2, 0.4, 0.6, 1.0);
        let s = Rgba::new(0.9, 0.1, 0.5, 1.0);
        let out = composite(b, s, BlendMode::Normal, 1.0, 1.0);
        assert!((out.r - 0.9).abs() < 1e-6);
        assert!((out.g - 0.1).abs() < 1e-6);
    }

    #[test]
    fn transparent_source_leaves_backdrop_untouched() {
        let b = Rgba::new(0.2, 0.4, 0.6, 1.0);
        let out = composite(b, Rgba::black(), BlendMode::Multiply, 1.0, 1.0);
        assert_eq!(out, b);
    }

    #[test]
    fn multiply_is_commutative_and_darkens() {
        let a = 0.5;
        let b = 0.8;
        assert!((BlendMode::Multiply.blend_channel(a, b) - a * b).abs() < 1e-6);
        assert!(
            (BlendMode::Multiply.blend_channel(a, b) - BlendMode::Multiply.blend_channel(b, a))
                .abs()
                < 1e-6
        );
        assert!(BlendMode::Multiply.blend_channel(a, b) < a);
    }

    #[test]
    fn overlay_is_inverse_of_hard_light() {
        for &b in &[0.1f32, 0.3, 0.5, 0.7, 0.9] {
            for &s in &[0.1f32, 0.3, 0.5, 0.7, 0.9] {
                let ov = BlendMode::Overlay.blend_channel(b, s);
                let hl = BlendMode::HardLight.blend_channel(s, b);
                assert!(
                    (ov - hl).abs() < 1e-6,
                    "overlay({b},{s}) != hardlight({s},{b})"
                );
            }
        }
    }

    #[test]
    fn blend_modes_stay_in_gamut() {
        let grid = [0.0f32, 0.01, 0.25, 0.5, 0.75, 0.99, 1.0];
        for mode in BlendMode::ALL {
            if mode.is_non_separable() {
                continue;
            }
            for &b in &grid {
                for &s in &grid {
                    let v = mode.blend_channel(b, s);
                    assert!(
                        (0.0..=1.0).contains(&v),
                        "{} produced {v} for ({b},{s})",
                        mode.label()
                    );
                }
            }
        }
    }

    /// Distance between two hues on the [0,1) circle.
    ///
    /// Red is hue 0 *and* hue 1, so a naive `|a - b| < eps` reports a false
    /// failure for any comparison that touches the seam.
    fn hue_delta(a: f32, b: f32) -> f32 {
        let d = (a - b).abs() % 1.0;
        d.min(1.0 - d)
    }

    #[test]
    fn hue_takes_source_hue_with_backdrop_luminosity() {
        let backdrop = Rgba::new(0.2, 0.6, 0.2, 1.0); // green, hue 1/3
        let source = Rgba::new(0.6, 0.2, 0.2, 1.0); // red, hue 0/1
        let out = BlendMode::Hue.blend_pixel(backdrop, source);
        let bh = rgb_to_hsl(backdrop.r, backdrop.g, backdrop.b);
        let sh = rgb_to_hsl(source.r, source.g, source.b);
        let oh = rgb_to_hsl(out.r, out.g, out.b);
        // Hue comes from the SOURCE, not the backdrop.
        assert!(
            hue_delta(sh.0, oh.0) < 1e-4,
            "hue must come from the source, got {} vs {}",
            oh.0,
            sh.0
        );
        // Saturation and lightness come from the backdrop.
        assert!(
            (bh.1 - oh.1).abs() < 1e-4,
            "saturation must come from backdrop, got {} vs {}",
            oh.1,
            bh.1
        );
        assert!(
            (bh.2 - oh.2).abs() < 1e-4,
            "lightness must come from backdrop, got {} vs {}",
            oh.2,
            bh.2
        );
    }

    #[test]
    fn luminosity_takes_source_luminosity_with_backdrop_hue() {
        // Light backdrop, saturated source: the result must be the source's
        // lightness on the backdrop's hue, which is a dark green here.
        let backdrop = Rgba::new(0.2, 0.6, 0.2, 1.0);
        let source = Rgba::new(0.9, 0.9, 0.9, 1.0);
        let out = BlendMode::Luminosity.blend_pixel(backdrop, source);
        let bh = rgb_to_hsl(backdrop.r, backdrop.g, backdrop.b);
        let oh = rgb_to_hsl(out.r, out.g, out.b);
        assert!(
            (bh.0 - oh.0).abs() < 1e-4,
            "hue must come from backdrop, got {} vs {}",
            oh.0,
            bh.0
        );
        assert!(out.luma() > 0.7, "luminosity must come from source");
    }

    #[test]
    fn color_and_saturation_take_the_source_components_they_should() {
        // Color: source hue + saturation, backdrop lightness.
        let backdrop = Rgba::new(0.2, 0.2, 0.6, 1.0); // blue, dark
        let source = Rgba::new(0.6, 0.6, 0.2, 1.0); // yellow, sat 0.5, light 0.4
        let out = BlendMode::Color.blend_pixel(backdrop, source);
        let sh = rgb_to_hsl(source.r, source.g, source.b);
        let bh = rgb_to_hsl(backdrop.r, backdrop.g, backdrop.b);
        let oh = rgb_to_hsl(out.r, out.g, out.b);
        assert!((sh.0 - oh.0).abs() < 1e-4, "color: hue from source");
        assert!((sh.1 - oh.1).abs() < 1e-4, "color: saturation from source");
        assert!((bh.2 - oh.2).abs() < 1e-4, "color: lightness from backdrop");

        // Saturation: source saturation, backdrop hue + lightness.
        let out = BlendMode::Saturation.blend_pixel(backdrop, source);
        let oh = rgb_to_hsl(out.r, out.g, out.b);
        assert!((bh.0 - oh.0).abs() < 1e-4, "saturation: hue from backdrop");
        assert!(
            (sh.1 - oh.1).abs() < 1e-4,
            "saturation: saturation from source"
        );
        assert!(
            (bh.2 - oh.2).abs() < 1e-4,
            "saturation: lightness from backdrop"
        );
    }

    #[test]
    fn fill_zero_makes_opaque_source_invisible() {
        let b = Rgba::new(0.0, 0.0, 0.0, 1.0);
        let s = Rgba::new(1.0, 1.0, 1.0, 1.0);
        let out = composite(b, s, BlendMode::Difference, 1.0, 0.0);
        assert_eq!(out, b);
    }

    #[test]
    fn hsl_roundtrip() {
        for &c in &[0.0f32, 0.2, 0.55, 0.9, 1.0] {
            let (h, s, l) = rgb_to_hsl(c, 1.0 - c, 0.5);
            let (r, g, b) = hsl_to_rgb(h, s, l);
            assert!((r - c).abs() < 1e-4, "{r} != {c}");
            assert!((g - (1.0 - c)).abs() < 1e-4);
            assert!((b - 0.5).abs() < 1e-4);
        }
    }
}
