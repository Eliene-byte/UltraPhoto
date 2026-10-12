//! The layer node and its editable properties.

use serde::{Deserialize, Serialize};
pub use ultrablend::BlendMode;
use ultrapixel::{Depth, PixelFormat, Rect, Surface};

use crate::id::LayerId;

/// What a layer *is*, which decides which panel sections apply.
///
/// Pixel layers hold a [`Surface`]. Every other kind is procedural and is
/// evaluated by the compositor on demand — the same split Photoshop uses
/// between raster layers and adjustment/fill layers.
#[derive(Debug, Clone, PartialEq)]
pub enum LayerKind {
    /// Regular raster content.
    Pixel,
    /// Nested stack. Children render into an isolated group buffer when
    /// `isolate` is set, which is what "Group" does in the Layers panel.
    Group,
    /// Curves, levels, hue/saturation, ... evaluated at composite time.
    Adjustment(AdjustmentKind),
    /// Solid colour / gradient / pattern fill.
    Fill(FillKind),
    /// Vector or rasterised type.
    Text(TextSpec),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum AdjustmentKind {
    BrightnessContrast {
        brightness: f32,
        contrast: f32,
    },
    Levels {
        input_black: f32,
        input_white: f32,
        gamma: f32,
    },
    Curves {
        channel: u8,
        points: Vec<(f32, f32)>,
    },
    HueSaturation {
        hue: f32,
        saturation: f32,
        lightness: f32,
        colorize: bool,
    },
    ColorBalance {
        shadows: [f32; 3],
        midtones: [f32; 3],
        highlights: [f32; 3],
    },
    Invert,
    Posterize {
        levels: u8,
    },
    Threshold {
        level: u8,
    },
    SelectiveColor,
    ChannelMixer {
        matrix: [[f32; 4]; 4],
    },
    ColorLookup {
        lut_id: String,
        intensity: f32,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub enum FillKind {
    Solid {
        color: [f32; 4],
    },
    Gradient {
        stops: Vec<GradientStop>,
        angle: f32,
        scale: f32,
        dither: bool,
    },
    Pattern {
        pattern_id: String,
        scale: f32,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GradientStop {
    pub position: f32,
    pub color: [f32; 4],
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TextSpec {
    pub text: String,
    pub font_family: String,
    pub font_size: f32,
    pub tracking: f32,
    pub leading: f32,
    pub color: [f32; 4],
    pub align: u8,
}

/// Lock flags. Photoshop shows these as padlock icons in the Layers panel.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct LockState {
    pub position: bool,
    pub pixels: bool,
    pub transparent_pixels: bool,
    pub image_pixels: bool,
    pub all: bool,
}

impl LockState {
    /// Editing is blocked when any of these are set.
    #[inline]
    pub fn blocks_edit(&self) -> bool {
        self.all || self.pixels
    }
    /// Painting is blocked when either pixel lock is set.
    #[inline]
    pub fn blocks_paint(&self) -> bool {
        self.all || self.pixels || self.transparent_pixels
    }
}

/// Layer visibility/lock/enabled state shown as icons in the Layers panel.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LayerProps {
    pub opacity: f32,
    pub fill: f32,
    pub blend_mode: u8,
    pub visible: bool,
    pub locked: bool,
}

impl Default for LayerProps {
    fn default() -> Self {
        Self {
            opacity: 1.0,
            fill: 1.0,
            blend_mode: BlendMode::Normal as u8,
            visible: true,
            locked: false,
        }
    }
}

impl LayerProps {
    #[inline]
    pub fn blend(&self) -> BlendMode {
        // Round-trips safely: every `BlendMode` repr is < 27 and all 27 are
        // covered by `BlendMode::ALL`, so the fallback is unreachable.
        BlendMode::ALL
            .get(self.blend_mode as usize)
            .copied()
            .unwrap_or(BlendMode::Normal)
    }
}

/// A layer mask. `None` means the layer has no mask (the usual case).
#[derive(Debug, Clone)]
pub struct LayerMask {
    pub surface: Surface,
    /// When false the mask is inverted before use, matching the mask
    /// "Invert" checkbox.
    pub enabled: bool,
    pub linked: bool,
    pub density: f32,
    pub feather: f32,
}

/// A node in the layer tree.
#[derive(Debug, Clone)]
pub struct Layer {
    pub id: LayerId,
    pub name: String,
    pub kind: LayerKind,
    pub props: LayerProps,
    pub locks: LockState,

    /// Raster content. Empty for non-pixel layers.
    pub surface: Surface,

    /// Layer mask, if any.
    pub mask: Option<LayerMask>,

    /// Clipped to the alpha of the layer below (`Alt+click` the mask).
    pub clipped: bool,
    /// Excluded from composite output but visible in the panel.
    pub hidden: bool,

    /// Children, only meaningful for [`LayerKind::Group`]. Index 0 is the
    /// bottom-most child, matching Photoshop's panel order.
    pub children: Vec<LayerId>,
    pub parent: Option<LayerId>,
    /// Group isolation: render children into their own buffer first.
    pub isolate: bool,
    /// Folder expanded in the Layers panel.
    pub expanded: bool,
    /// Tint colour shown in the panel (None = no colour label).
    pub label_color: Option<u8>,

    /// Text layers keep an editable copy of their spec.
    pub depth: Depth,
    pub format: PixelFormat,
}

impl Layer {
    /// Creates a new pixel layer bound to a document's canvas geometry.
    pub fn pixel(
        name: impl Into<String>,
        width: usize,
        height: usize,
        depth: Depth,
        format: PixelFormat,
    ) -> Self {
        Self {
            id: LayerId::next(),
            name: name.into(),
            kind: LayerKind::Pixel,
            props: LayerProps::default(),
            locks: LockState::default(),
            surface: Surface::new(width, height, depth, format),
            mask: None,
            clipped: false,
            hidden: false,
            children: Vec::new(),
            parent: None,
            isolate: false,
            expanded: true,
            label_color: None,
            depth,
            format,
        }
    }

    pub fn group(name: impl Into<String>, depth: Depth, format: PixelFormat) -> Self {
        let mut l = Self::pixel(name, 0, 0, depth, format);
        l.kind = LayerKind::Group;
        l.isolate = true;
        l
    }

    pub fn adjustment(name: impl Into<String>, adj: AdjustmentKind) -> Self {
        Self {
            id: LayerId::next(),
            name: name.into(),
            kind: LayerKind::Adjustment(adj),
            props: LayerProps::default(),
            locks: LockState::default(),
            surface: Surface::new(0, 0, Depth::F32, PixelFormat::RGBA),
            mask: None,
            clipped: false,
            hidden: false,
            children: Vec::new(),
            parent: None,
            isolate: false,
            expanded: true,
            label_color: None,
            depth: Depth::F32,
            format: PixelFormat::RGBA,
        }
    }

    #[inline]
    pub fn is_group(&self) -> bool {
        matches!(self.kind, LayerKind::Group)
    }

    #[inline]
    pub fn is_pixel(&self) -> bool {
        matches!(self.kind, LayerKind::Pixel)
    }

    #[inline]
    pub fn is_procedural(&self) -> bool {
        !matches!(self.kind, LayerKind::Pixel | LayerKind::Group)
    }

    /// Effective bounds of this layer, including mask when present.
    pub fn effective_bounds(&self) -> Rect {
        let b = self.surface.content_bounds();
        match &self.mask {
            Some(m) => b.union(m.surface.content_bounds()),
            None => b,
        }
    }

    /// Total resident bytes — what the memory HUD attributes to this layer.
    pub fn resident_bytes(&self) -> usize {
        self.surface.resident_bytes() + self.mask.as_ref().map_or(0, |m| m.surface.resident_bytes())
    }
}
