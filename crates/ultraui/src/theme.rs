//! Photoshop's dark theme, as data.
//!
//! These are not approximations. They are the sRGB values from Photoshop's
//! default dark workspace, and the test at the bottom pins the ones that
//! matter so a careless edit to "prettify" the palette fails the build.
//!
//! Photoshop's UI is built on a four-step neutral ramp with a single accent.
//! The whole design reads as one system because the ramp is tight: every
//! surface sits within 30 levels of its neighbour.

use serde::{Deserialize, Serialize};

/// Straight 8-bit sRGB. Values are the literal Photoshop ones.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Color {
    pub r: u8,
    pub g: u8,
    pub b: u8,
}

impl Color {
    pub const fn rgb(r: u8, g: u8, b: u8) -> Self {
        Self { r, g, b }
    }

    /// `#RRGGBB`, the form used in Photoshop's own export.
    pub fn hex(self) -> String {
        format!("#{:02X}{:02X}{:02X}", self.r, self.g, self.b)
    }

    /// Packed `0x00RRGGBB` for direct vertex-buffer upload.
    #[inline]
    pub const fn pack_rgb(self) -> u32 {
        ((self.r as u32) << 16) | ((self.g as u32) << 8) | (self.b as u32)
    }

    /// Straight alpha, non-premultiplied — what the UI shaders expect.
    #[inline]
    pub const fn with_alpha(self, a: f32) -> [f32; 4] {
        [
            self.r as f32 / 255.0,
            self.g as f32 / 255.0,
            self.b as f32 / 255.0,
            a,
        ]
    }

    /// Linear blend towards `other`, used for hover/pressed states.
    pub fn mix(self, other: Color, t: f32) -> Color {
        let t = t.clamp(0.0, 1.0);
        let f = |a: u8, b: u8| (a as f32 + (b as f32 - a as f32) * t).round() as u8;
        Color {
            r: f(self.r, other.r),
            g: f(self.g, other.g),
            b: f(self.b, other.b),
        }
    }

    /// WCAG relative luminance.
    ///
    /// The sRGB transfer function has to be undone first. Weighting the
    /// *encoded* bytes directly — the common shortcut — badly understates dark
    /// colours and made #D4D4D4 on #323232 look like 3.6:1 when it is really
    /// 8.6:1.
    pub fn luminance(self) -> f32 {
        fn lin(c: u8) -> f32 {
            let c = c as f32 / 255.0;
            if c <= 0.04045 {
                c / 12.92
            } else {
                ((c + 0.055) / 1.055).powf(2.4)
            }
        }
        0.2126 * lin(self.r) + 0.7152 * lin(self.g) + 0.0722 * lin(self.b)
    }

    /// WCAG contrast ratio against `other`, used to keep text legible.
    pub fn contrast_ratio(self, other: Color) -> f32 {
        let a = self.luminance();
        let b = other.luminance();
        let (hi, lo) = if a > b { (a, b) } else { (b, a) };
        (hi + 0.05) / (lo + 0.05)
    }
}

/// Spacing and sizing. Photoshop's UI is dense; these numbers are small.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Metrics {
    /// Menu bar height.
    pub menu_bar: f32,
    /// Options bar height (tool-specific settings under the menu).
    pub options_bar: f32,
    /// Document tab strip.
    pub tab_bar: f32,
    /// Title bar.
    pub title_bar: f32,
    /// Status bar.
    pub status_bar: f32,

    /// Standard panel header.
    pub panel_header: f32,
    /// Height of a row in a list (Layers panel entries).
    pub row: f32,
    /// Row with the layer thumbnail shown.
    pub row_with_thumb: f32,

    /// Width of a text input field.
    pub field_width: f32,
    /// Input field height.
    pub field_height: f32,
    /// Button height.
    pub button_height: f32,
    /// Small square icon button (panel header buttons, visibility eye).
    pub icon_button: f32,

    /// Docking splitter hit area. Wider than the visual line so it is grabbable.
    pub splitter: f32,
    /// Panel inner padding.
    pub panel_pad: f32,

    /// Font sizes.
    pub font_small: f32,
    pub font_normal: f32,
    pub font_large: f32,

    /// Corner radius. Photoshop is nearly square; only menus round slightly.
    pub corner_radius: f32,
    /// Menu popup corner radius.
    pub menu_corner_radius: f32,
}

impl Default for Metrics {
    fn default() -> Self {
        Self {
            menu_bar: 28.0,
            options_bar: 26.0,
            tab_bar: 26.0,
            title_bar: 30.0,
            status_bar: 22.0,

            panel_header: 24.0,
            row: 24.0,
            row_with_thumb: 40.0,

            field_width: 68.0,
            field_height: 20.0,
            button_height: 22.0,
            icon_button: 20.0,

            // The visual divider is 1px but the grab target is 6, which is what
            // makes docking feel right.
            splitter: 6.0,
            panel_pad: 8.0,

            font_small: 11.0,
            font_normal: 12.0,
            font_large: 13.0,

            corner_radius: 2.0,
            menu_corner_radius: 4.0,
        }
    }
}

/// The full colour set.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Palette {
    /// Window and panel background.
    pub panel_bg: Color,
    /// The area around the canvas.
    pub canvas_surround: Color,
    /// Menu bar background.
    pub menu_bar_bg: Color,
    /// Panel header strip.
    pub panel_header_bg: Color,
    /// Active tab background — matches `panel_bg` in Photoshop.
    pub tab_active_bg: Color,
    /// Inactive tab background.
    pub tab_inactive_bg: Color,

    /// Text input field background.
    pub field_bg: Color,
    /// Field / panel border.
    pub border: Color,
    /// Hover state for a clickable row.
    pub hover: Color,
    /// Pressed / active state.
    pub active: Color,
    /// Currently selected row (Layers panel).
    pub selection: Color,
    /// Focus ring around the focused control.
    pub focus: Color,

    /// Primary text.
    pub text: Color,
    /// Secondary text (units, hints).
    pub text_dim: Color,
    /// Disabled text.
    pub text_disabled: Color,
    /// Text on an accent-coloured surface.
    pub text_on_accent: Color,

    /// The accent, used for focus, selection highlights and sliders.
    pub accent: Color,
    /// Slider / scrubby track fill.
    pub fill: Color,
    /// Track background.
    pub track: Color,

    /// Panel divider lines.
    pub divider: Color,
    /// Scrollbar thumb.
    pub scrollbar: Color,
    /// Border around the whole canvas area.
    pub canvas_border: Color,
    /// Checkerboard light square for transparency.
    pub checker_light: Color,
    /// Checkerboard dark square for transparency.
    pub checker_dark: Color,
}

impl Default for Palette {
    /// Photoshop's default dark workspace.
    fn default() -> Self {
        Self {
            panel_bg: Color::rgb(0x32, 0x32, 0x32),
            canvas_surround: Color::rgb(0x1E, 0x1E, 0x1E),
            menu_bar_bg: Color::rgb(0x2C, 0x2C, 0x2C),
            panel_header_bg: Color::rgb(0x3C, 0x3C, 0x3C),
            tab_active_bg: Color::rgb(0x32, 0x32, 0x32),
            tab_inactive_bg: Color::rgb(0x28, 0x28, 0x28),

            field_bg: Color::rgb(0x1E, 0x1E, 0x1E),
            border: Color::rgb(0x4A, 0x4A, 0x4A),
            hover: Color::rgb(0x3F, 0x3F, 0x3F),
            active: Color::rgb(0x46, 0x46, 0x46),
            selection: Color::rgb(0x2D, 0x5F, 0x8B),
            focus: Color::rgb(0x14, 0x73, 0xE6),

            text: Color::rgb(0xD4, 0xD4, 0xD4),
            text_dim: Color::rgb(0x9A, 0x9A, 0x9A),
            text_disabled: Color::rgb(0x6E, 0x6E, 0x6E),
            text_on_accent: Color::rgb(0xFF, 0xFF, 0xFF),

            accent: Color::rgb(0x14, 0x73, 0xE6),
            fill: Color::rgb(0xB5, 0xB5, 0xB5),
            track: Color::rgb(0x4D, 0x4D, 0x4D),

            divider: Color::rgb(0x3C, 0x3C, 0x3C),
            scrollbar: Color::rgb(0x4D, 0x4D, 0x4D),
            canvas_border: Color::rgb(0x1A, 0x1A, 0x1A),
            checker_light: Color::rgb(0xCC, 0xCC, 0xCC),
            checker_dark: Color::rgb(0x99, 0x99, 0x99),
        }
    }
}

/// A named theme, so a future light workspace is data, not a fork.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Default)]
pub struct Theme {
    pub palette: Palette,
    pub metrics: Metrics,
}

/// Photoshop's dark workspace.
pub const PS_DARK: Theme = Theme {
    palette: Palette {
        panel_bg: Color::rgb(0x32, 0x32, 0x32),
        canvas_surround: Color::rgb(0x1E, 0x1E, 0x1E),
        menu_bar_bg: Color::rgb(0x2C, 0x2C, 0x2C),
        panel_header_bg: Color::rgb(0x3C, 0x3C, 0x3C),
        tab_active_bg: Color::rgb(0x32, 0x32, 0x32),
        tab_inactive_bg: Color::rgb(0x28, 0x28, 0x28),
        field_bg: Color::rgb(0x1E, 0x1E, 0x1E),
        border: Color::rgb(0x4A, 0x4A, 0x4A),
        hover: Color::rgb(0x3F, 0x3F, 0x3F),
        active: Color::rgb(0x46, 0x46, 0x46),
        selection: Color::rgb(0x2D, 0x5F, 0x8B),
        focus: Color::rgb(0x14, 0x73, 0xE6),
        text: Color::rgb(0xD4, 0xD4, 0xD4),
        text_dim: Color::rgb(0x9A, 0x9A, 0x9A),
        text_disabled: Color::rgb(0x6E, 0x6E, 0x6E),
        text_on_accent: Color::rgb(0xFF, 0xFF, 0xFF),
        accent: Color::rgb(0x14, 0x73, 0xE6),
        fill: Color::rgb(0xB5, 0xB5, 0xB5),
        track: Color::rgb(0x4D, 0x4D, 0x4D),
        divider: Color::rgb(0x3C, 0x3C, 0x3C),
        scrollbar: Color::rgb(0x4D, 0x4D, 0x4D),
        canvas_border: Color::rgb(0x1A, 0x1A, 0x1A),
        checker_light: Color::rgb(0xCC, 0xCC, 0xCC),
        checker_dark: Color::rgb(0x99, 0x99, 0x99),
    },
    metrics: Metrics {
        menu_bar: 28.0,
        options_bar: 26.0,
        tab_bar: 26.0,
        title_bar: 30.0,
        status_bar: 22.0,
        panel_header: 24.0,
        row: 24.0,
        row_with_thumb: 40.0,
        field_width: 68.0,
        field_height: 20.0,
        button_height: 22.0,
        icon_button: 20.0,
        splitter: 6.0,
        panel_pad: 8.0,
        font_small: 11.0,
        font_normal: 12.0,
        font_large: 13.0,
        corner_radius: 2.0,
        menu_corner_radius: 4.0,
    },
};

impl Palette {
    /// Background of a control in a given interaction state.
    pub fn control_bg(&self, state: ControlState) -> Color {
        match state {
            ControlState::Normal => self.field_bg,
            ControlState::Hover => self.hover,
            ControlState::Pressed => self.active,
            ControlState::Focused => self.field_bg,
            ControlState::Disabled => self.field_bg,
        }
    }

    /// Foreground of a control in a given interaction state.
    pub fn control_fg(&self, state: ControlState) -> Color {
        match state {
            ControlState::Disabled => self.text_disabled,
            _ => self.text,
        }
    }

    /// Colour of the transparency checkerboard at `(x, y)`.
    ///
    /// Photoshop's checker is 8×8 px with the light square on the top-left.
    pub fn checker(&self, x: i32, y: i32) -> Color {
        let cell = ((x / 8) + (y / 8)) & 1;
        if cell == 0 {
            self.checker_light
        } else {
            self.checker_dark
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ControlState {
    Normal,
    Hover,
    Pressed,
    Focused,
    Disabled,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_default_theme_matches_photoshops_palette() {
        // Pinned so a "prettify the palette" commit cannot silently change the
        // look. These are Photoshop's values.
        assert_eq!(
            Theme::default().palette.panel_bg,
            Color::rgb(0x32, 0x32, 0x32)
        );
        assert_eq!(
            Theme::default().palette.canvas_surround,
            Color::rgb(0x1E, 0x1E, 0x1E)
        );
        assert_eq!(Theme::default().palette.text, Color::rgb(0xD4, 0xD4, 0xD4));
        assert_eq!(Theme::default().palette.focus, Color::rgb(0x14, 0x73, 0xE6));
    }

    #[test]
    fn the_explicit_ps_dark_constant_equals_the_default() {
        assert_eq!(PS_DARK, Theme::default());
    }

    #[test]
    fn the_neutral_ramp_is_tight() {
        // The whole design reads as one system because no two adjacent
        // surfaces are more than 30 levels apart.
        let p = PS_DARK.palette;
        let ramp = [
            p.field_bg,
            p.canvas_surround,
            p.tab_inactive_bg,
            p.menu_bar_bg,
            p.panel_bg,
            p.hover,
            p.panel_header_bg,
            p.border,
        ];
        for w in ramp.windows(2) {
            let delta = (w[0].r as i32 - w[1].r as i32).abs();
            assert!(delta <= 30, "ramp step {delta} is too wide: {w:?}");
        }
    }

    #[test]
    fn body_text_meets_wcag_aa_on_every_surface() {
        let p = PS_DARK.palette;
        for bg in [
            p.panel_bg,
            p.canvas_surround,
            p.menu_bar_bg,
            p.panel_header_bg,
            p.field_bg,
        ] {
            let ratio = p.text.contrast_ratio(bg);
            assert!(
                ratio >= 4.5,
                "contrast {ratio:.2} against {bg:?} is below WCAG AA"
            );
        }
    }

    #[test]
    fn white_on_the_accent_meets_wcag_aa() {
        let p = PS_DARK.palette;
        let ratio = p.text_on_accent.contrast_ratio(p.accent);
        assert!(
            ratio >= 4.5,
            "white on {} is only {ratio:.2}:1",
            p.accent.hex()
        );
    }

    #[test]
    fn the_focus_colour_is_distinguishable_from_selection() {
        // They must not be confusable, or keyboard focus would look like a
        // selection change.
        let p = PS_DARK.palette;
        assert!(
            p.focus.contrast_ratio(p.panel_bg) > p.selection.contrast_ratio(p.panel_bg),
            "focus should read more strongly than selection"
        );
    }

    #[test]
    fn disabled_text_is_visibly_dimmer_than_normal_text() {
        let p = PS_DARK.palette;
        assert!(p.text_disabled.luminance() < p.text.luminance());
        assert!(p.text_dim.luminance() < p.text.luminance());
    }

    #[test]
    fn checkerboard_alternates_and_starts_light() {
        let p = PS_DARK.palette;
        assert_eq!(p.checker(0, 0), p.checker_light);
        assert_eq!(p.checker(8, 0), p.checker_dark);
        assert_eq!(p.checker(0, 8), p.checker_dark);
        assert_eq!(p.checker(8, 8), p.checker_light);
        // Stable across the cell, not per pixel.
        assert_eq!(p.checker(3, 3), p.checker(7, 7));
    }

    #[test]
    fn hex_and_packing_round_trip() {
        let c = Color::rgb(0x32, 0x5F, 0x8B);
        assert_eq!(c.hex(), "#325F8B");
        assert_eq!(c.pack_rgb(), 0x325F8B);
    }

    #[test]
    fn splitter_is_grabbable_while_looking_thin() {
        let m = PS_DARK.metrics;
        assert!(m.splitter >= 4.0, "too thin to grab reliably");
        assert!(m.row >= 22.0, "rows too short for a 100% DPI hit target");
    }
}
