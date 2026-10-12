//! Pixel formats: the arrangement of channels inside a layer.

/// Number of channels a layer's pixels are organised into.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum ChannelLayout {
    /// Single channel, no alpha (Bitmap mode).
    #[default]
    Gray,
    /// Grey + 1 alpha.
    GrayAlpha,
    /// Red, green, blue.
    Rgb,
    /// Red, green, blue + 1 alpha. The 99% case.
    Rgba,
    /// CIE L*a*b* + alpha.
    LabAlpha,
    /// Device-independent CMYK + alpha + 1 under-colour removal channel.
    CmykAlpha,
    /// Fully user-defined channel stack (spot channels, masks, extra alpha).
    Multi(u8),
}

impl ChannelLayout {
    /// Total channels including alpha.
    #[inline]
    pub const fn channel_count(self) -> usize {
        match self {
            ChannelLayout::Gray => 1,
            ChannelLayout::GrayAlpha => 2,
            ChannelLayout::Rgb => 3,
            ChannelLayout::Rgba => 4,
            ChannelLayout::LabAlpha => 4,
            ChannelLayout::CmykAlpha => 6,
            ChannelLayout::Multi(n) => n as usize,
        }
    }

    /// How many alpha channels the layout carries at the end of the stack.
    #[inline]
    pub const fn alpha_count(self) -> usize {
        match self {
            ChannelLayout::Gray | ChannelLayout::Rgb | ChannelLayout::Multi(_) => 0,
            ChannelLayout::GrayAlpha
            | ChannelLayout::Rgba
            | ChannelLayout::LabAlpha
            | ChannelLayout::CmykAlpha => 1,
        }
    }

    /// Index of the first alpha channel.
    #[inline]
    pub const fn first_alpha(self) -> Option<usize> {
        let n = self.channel_count();
        if self.alpha_count() > 0 {
            Some(n - self.alpha_count())
        } else {
            None
        }
    }

    /// Human-facing name shown in the Channels panel.
    #[inline]
    pub const fn name(self) -> &'static str {
        match self {
            ChannelLayout::Gray => "Grayscale",
            ChannelLayout::GrayAlpha => "Grayscale + Alpha",
            ChannelLayout::Rgb => "RGB",
            ChannelLayout::Rgba => "RGB + Alpha",
            ChannelLayout::LabAlpha => "Lab + Alpha",
            ChannelLayout::CmykAlpha => "CMYK + Alpha",
            ChannelLayout::Multi(_) => "Custom",
        }
    }

    /// Channel names in stack order, used by the Channels panel header.
    pub fn channel_names(self) -> Vec<String> {
        let mut names = Vec::with_capacity(self.channel_count());
        match self {
            ChannelLayout::Gray => names.push("Gray".into()),
            ChannelLayout::GrayAlpha => names.push("Gray".into()),
            ChannelLayout::Rgb | ChannelLayout::Rgba => {
                names.push("Red".into());
                names.push("Green".into());
                names.push("Blue".into());
            }
            ChannelLayout::LabAlpha => {
                names.push("Lightness".into());
                names.push("a".into());
                names.push("b".into());
            }
            ChannelLayout::CmykAlpha => {
                names.push("Cyan".into());
                names.push("Magenta".into());
                names.push("Yellow".into());
                names.push("Black".into());
                names.push("Undercolor".into());
            }
            ChannelLayout::Multi(n) => {
                for i in 0..n {
                    names.push(format!("Channel {}", i + 1));
                }
            }
        }
        for i in 0..self.alpha_count() {
            names.push(if i == 0 {
                "Alpha".into()
            } else {
                format!("Alpha {}", i + 1)
            });
        }
        names
    }
}

/// Colour model of a surface, independent of bit depth.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum ColorModel {
    Gray,
    #[default]
    Rgb,
    Cmyk,
    Lab,
    /// Indexed colour with a palette (the classic GIF-style mode).
    Indexed,
}

impl ColorModel {
    /// Colour primaries only. CMYK has four; the under-colour removal channel that
    /// follows it is coverage, not colour.
    #[inline]
    pub const fn base_channels(self) -> usize {
        match self {
            ColorModel::Gray | ColorModel::Indexed => 1,
            ColorModel::Rgb | ColorModel::Lab => 3,
            ColorModel::Cmyk => 4,
        }
    }

    #[inline]
    pub const fn is_photographic(self) -> bool {
        matches!(self, ColorModel::Rgb | ColorModel::Cmyk | ColorModel::Lab)
    }
}

/// A layout paired with a colour model — everything needed to interpret a
/// layer's planes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct PixelFormat {
    pub layout: ChannelLayout,
    pub color: ColorModel,
}

impl PixelFormat {
    pub const RGB: PixelFormat = PixelFormat {
        layout: ChannelLayout::Rgb,
        color: ColorModel::Rgb,
    };
    pub const RGBA: PixelFormat = PixelFormat {
        layout: ChannelLayout::Rgba,
        color: ColorModel::Rgb,
    };
    pub const GRAY: PixelFormat = PixelFormat {
        layout: ChannelLayout::Gray,
        color: ColorModel::Gray,
    };
    pub const GRAY_ALPHA: PixelFormat = PixelFormat {
        layout: ChannelLayout::GrayAlpha,
        color: ColorModel::Gray,
    };
    pub const CMYK: PixelFormat = PixelFormat {
        layout: ChannelLayout::CmykAlpha,
        color: ColorModel::Cmyk,
    };
    pub const LAB: PixelFormat = PixelFormat {
        layout: ChannelLayout::LabAlpha,
        color: ColorModel::Lab,
    };

    #[inline]
    pub const fn channel_count(self) -> usize {
        self.layout.channel_count()
    }

    #[inline]
    pub const fn with_alpha(self) -> PixelFormat {
        match self.layout {
            ChannelLayout::Rgb => PixelFormat {
                layout: ChannelLayout::Rgba,
                ..self
            },
            ChannelLayout::Gray => PixelFormat {
                layout: ChannelLayout::GrayAlpha,
                ..self
            },
            other => PixelFormat {
                layout: other,
                ..self
            },
        }
    }

    #[inline]
    pub const fn without_alpha(self) -> PixelFormat {
        match self.layout {
            ChannelLayout::Rgba => PixelFormat {
                layout: ChannelLayout::Rgb,
                ..self
            },
            ChannelLayout::GrayAlpha => PixelFormat {
                layout: ChannelLayout::Gray,
                ..self
            },
            other => PixelFormat {
                layout: other,
                ..self
            },
        }
    }

    /// True when the channel holds coverage data (alpha, mask, spot channel)
    /// rather than colour. Coverage channels are never gamma-encoded, so the
    /// compositor and the GPU kernels must treat them as linear even when the
    /// colour channels are gamma-encoded 8-bit values.
    #[inline]
    pub fn is_coverage_channel(self, index: usize) -> bool {
        if let Some(a) = self.layout.first_alpha() {
            if index >= a {
                return true;
            }
        }
        // Everything past the colour primaries of a multi-channel stack.
        index >= self.color.base_channels().min(self.channel_count())
    }
}
