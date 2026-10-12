//! Container/metadata detection and the pieces we surface to the UI.

use ultrapixel::Depth;

/// The file formats UltraPhoto can name. Not every format supports every
/// direction — see [`FileFormat::can_write`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FileFormat {
    Png,
    Jpeg,
    Tiff,
    WebP,
    Psd,
    Exr,
    /// Anything we can read but not name.
    Unknown,
}

impl FileFormat {
    /// Canonical lowercase extension, without the dot.
    pub const fn extension(self) -> &'static str {
        match self {
            FileFormat::Png => "png",
            FileFormat::Jpeg => "jpg",
            FileFormat::Tiff => "tiff",
            FileFormat::WebP => "webp",
            FileFormat::Psd => "psd",
            FileFormat::Exr => "exr",
            FileFormat::Unknown => "",
        }
    }

    /// Name shown in the file dialog's type filter.
    pub const fn display_name(self) -> &'static str {
        match self {
            FileFormat::Png => "PNG",
            FileFormat::Jpeg => "JPEG",
            FileFormat::Tiff => "TIFF",
            FileFormat::WebP => "WebP",
            FileFormat::Psd => "Photoshop PSD",
            FileFormat::Exr => "OpenEXR",
            FileFormat::Unknown => "Unknown",
        }
    }

    pub const fn can_write(self) -> bool {
        matches!(
            self,
            FileFormat::Png
                | FileFormat::Jpeg
                | FileFormat::Tiff
                | FileFormat::WebP
                | FileFormat::Psd
                | FileFormat::Exr
        )
    }

    /// True when the container can carry more than 8 bits per channel.
    pub const fn supports_high_depth(self) -> bool {
        matches!(self, FileFormat::Tiff | FileFormat::Exr | FileFormat::Psd)
    }

    /// True when the container can carry an alpha channel.
    pub const fn supports_alpha(self) -> bool {
        matches!(
            self,
            FileFormat::Png
                | FileFormat::Tiff
                | FileFormat::WebP
                | FileFormat::Psd
                | FileFormat::Exr
        )
    }
}

/// Everything the UI needs after a successful open.
#[derive(Debug, Clone, Default)]
pub struct ImageMetadata {
    pub format: Option<FileFormat>,
    pub width: usize,
    pub height: usize,
    /// Depth the *file* declared, before we map it to a document depth.
    pub file_depth: Depth,
    pub has_alpha: bool,
    pub is_grayscale: bool,
    pub has_icc: bool,
    /// Resolution in DPI, when the container records it.
    pub dpi: Option<(u32, u32)>,
    /// Bits per pixel as reported by the container, for the status bar.
    pub bits_per_pixel: Option<u32>,
    /// True when the file carried a gamma chunk we honoured.
    pub applied_gamma: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn container_capabilities_are_reported_honestly() {
        // JPEG cannot carry alpha; anything claiming otherwise is a bug.
        assert!(!FileFormat::Jpeg.supports_alpha());
        assert!(!FileFormat::Jpeg.supports_high_depth());
        assert!(FileFormat::Png.supports_alpha());
        assert!(FileFormat::Tiff.supports_high_depth());
        assert!(FileFormat::Unknown.extension().is_empty());
    }
}
