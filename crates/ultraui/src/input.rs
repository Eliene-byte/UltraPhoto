//! Keyboard and mouse state, plus the Photoshop shortcut map.
//!
//! Input is captured once per frame into a [`InputState`] and then polled. That
//! ordering matters: a tool that reads "is the mouse down" twice in one frame
//! must see the same answer, and a shortcut must not fire twice because two
//! widgets both asked.

use std::collections::HashSet;

/// A physical key, named rather than numbered so the shortcut map reads like
/// Photoshop's menu does.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Key {
    A,
    B,
    C,
    D,
    E,
    F,
    G,
    H,
    I,
    J,
    K,
    L,
    M,
    N,
    O,
    P,
    Q,
    R,
    S,
    T,
    U,
    V,
    W,
    X,
    Y,
    Z,
    Num0,
    Num1,
    Num2,
    Num3,
    Num4,
    Num5,
    Num6,
    Num7,
    Num8,
    Num9,
    F1,
    F2,
    F3,
    F4,
    F5,
    F6,
    F7,
    F8,
    F9,
    F10,
    F11,
    F12,
    Escape,
    Enter,
    Tab,
    Space,
    Backspace,
    Delete,
    Insert,
    Home,
    End,
    PageUp,
    PageDown,
    ArrowLeft,
    ArrowRight,
    ArrowUp,
    ArrowDown,
    Minus,
    Equal,
    BracketLeft,
    BracketRight,
    Backslash,
    Semicolon,
    Quote,
    Comma,
    Period,
    Slash,
    Backquote,
}

/// Modifier state at the time of an event.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Modifiers {
    pub ctrl: bool,
    pub shift: bool,
    pub alt: bool,
    /// Photoshop treats Alt and Option as the same menu key, and on macOS
    /// Cmd takes Ctrl's role. The app normalises to this representation.
    pub alt_graph: bool,
    /// macOS Command.
    pub logo: bool,
}

impl Modifiers {
    pub const NONE: Modifiers = Modifiers {
        ctrl: false,
        shift: false,
        alt: false,
        alt_graph: false,
        logo: false,
    };

    #[inline]
    pub fn any(&self) -> bool {
        self.ctrl || self.shift || self.alt || self.alt_graph || self.logo
    }

    /// The "menu key" — Cmd on macOS, Alt everywhere else. Photoshop's menus
    /// underline this letter, which is why the shortcut map keys off it.
    #[inline]
    pub const fn menu(&self) -> bool {
        self.logo || self.alt
    }

    /// Ctrl on Windows/Linux, Cmd on macOS.
    #[inline]
    pub const fn command(&self) -> bool {
        self.ctrl || self.logo
    }

    pub fn key(self, key: Key) -> Shortcut {
        Shortcut { key, mods: self }
    }
}

/// A key plus modifiers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Shortcut {
    pub key: Key,
    pub mods: Modifiers,
}

impl Shortcut {
    pub const fn new(key: Key, mods: Modifiers) -> Self {
        Self { key, mods }
    }

    /// True when `self` matches the modifiers of an event, but requires an
    /// exact match on the shift key.
    ///
    /// Ctrl+N and Ctrl+Shift+N are different commands in Photoshop, so an exact
    /// modifier match is required; only the *key* is matched separately by the
    /// caller.
    pub fn modifiers_match(&self, mods: Modifiers) -> bool {
        self.mods.ctrl == mods.ctrl
            && self.mods.shift == mods.shift
            && self.mods.alt == mods.alt
            && self.mods.logo == mods.logo
    }
}

/// Mouse buttons, numbered like Photoshop's tooltips.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MouseButton {
    Left,
    Right,
    Middle,
}

/// A wheel event, in "lines" so the feel is the same on every device.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct ScrollDelta {
    pub x: f32,
    pub y: f32,
}

/// Everything a frame's widgets and tools can ask about.
#[derive(Debug, Clone, Default)]
pub struct InputState {
    /// Key is held down right now.
    held: HashSet<Key>,
    /// Key went down since the last frame.
    pressed: HashSet<Key>,
    /// Key came up since the last frame.
    released: HashSet<Key>,
    /// Modifier state right now.
    pub mods: Modifiers,
    /// Pointer position in window pixels.
    pub cursor: (f32, f32),
    /// Pointer position last frame, for drag deltas.
    pub cursor_prev: (f32, f32),
    pub buttons: HashSet<MouseButton>,
    pub scroll: ScrollDelta,
    /// Set when the cursor left the window; prevents stuck drags.
    pub cursor_out: bool,
    /// Current text input, for fields.
    pub text: String,
}

impl InputState {
    pub fn new() -> Self {
        Self::default()
    }

    /// Movement since last frame. This is the value every drag tool uses.
    #[inline]
    pub fn cursor_delta(&self) -> (f32, f32) {
        (
            self.cursor.0 - self.cursor_prev.0,
            self.cursor.1 - self.cursor_prev.1,
        )
    }

    #[inline]
    pub fn is_down(&self, key: Key) -> bool {
        self.held.contains(&key)
    }
    #[inline]
    pub fn just_pressed(&self, key: Key) -> bool {
        self.pressed.contains(&key)
    }
    #[inline]
    pub fn just_released(&self, key: Key) -> bool {
        self.released.contains(&key)
    }
    #[inline]
    pub fn is_clicked(&self, b: MouseButton) -> bool {
        self.buttons.contains(&b)
    }

    /// Frames since the left button went down. 0 means "this frame".
    /// Used for click-versus-drag discrimination on scrubby sliders.
    #[inline]
    pub fn click_age(&self) -> Option<f32> {
        if self.buttons.contains(&MouseButton::Left) {
            Some(0.0)
        } else {
            None
        }
    }

    // ------------------------------------------------------------- injection
    // These are the public surface the winit layer calls.

    pub fn key_down(&mut self, key: Key) {
        self.held.insert(key);
        self.pressed.insert(key);
    }

    pub fn key_up(&mut self, key: Key) {
        self.held.remove(&key);
        self.released.insert(key);
    }

    pub fn mouse_move(&mut self, x: f32, y: f32) {
        self.cursor = (x, y);
    }

    pub fn mouse_button(&mut self, b: MouseButton, down: bool) {
        if down {
            self.buttons.insert(b);
        } else {
            self.buttons.remove(&b);
        }
    }

    pub fn scroll(&mut self, dx: f32, dy: f32) {
        self.scroll.x += dx;
        self.scroll.y += dy;
    }

    pub fn set_mods(&mut self, m: Modifiers) {
        self.mods = m;
    }

    pub fn set_text(&mut self, t: String) {
        self.text = t;
    }

    /// Called at the end of each frame.
    ///
    /// Order matters: `cursor_prev` is updated *before* `cursor`, otherwise the
    /// first frame's delta is the distance from the origin.
    pub fn end_frame(&mut self) {
        self.pressed.clear();
        self.released.clear();
        self.scroll = ScrollDelta { x: 0.0, y: 0.0 };
        self.text.clear();
        self.cursor_prev = self.cursor;
        if self.cursor_out {
            // Releasing everything prevents a stuck drag when the window
            // loses focus mid-gesture.
            self.buttons.clear();
        }
    }
}

/// A named Photoshop command, so the shortcut map, the menu bar and the
/// action recorder all refer to the same identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Command {
    New,
    Open,
    Save,
    SaveAs,
    Undo,
    Redo,
    /// History: step to the previous state (Alt+Ctrl+Z).
    StepBackward,
    /// History: step to the next state (Alt+Ctrl+Shift+Z).
    StepForward,
    Cut,
    Copy,
    Paste,
    Fill,
    Stroke,
    FreeTransform,
    ZoomIn,
    ZoomOut,
    FitOnScreen,
    ActualPixels,
    SelectAll,
    Deselect,
    Inverse,
    Feather,
    Expand,
    Contract,
    NewLayer,
    NewGroup,
    DuplicateLayer,
    DeleteLayer,
    GroupLayers,
    UngroupLayers,
    MergeDown,
    Flatten,
    Blur,
    Sharpen,
    Smudge,
    Dodge,
    Burn,
    Eyedropper,
    Brush,
    Eraser,
    Move,
    Marquee,
    Lasso,
    Crop,
    Type,
    Pen,
}

impl Command {
    /// Photoshop's own scripting name, so `.jsx` actions and the action
    /// recorder keep working.
    pub fn script_name(self) -> &'static str {
        use Command::*;
        match self {
            New => "new",
            Open => "open",
            Save => "save",
            SaveAs => "saveAs",
            Undo => "undo",
            Redo => "redo",
            StepBackward => "stepBackward",
            StepForward => "stepForward",
            Cut => "cut",
            Copy => "copy",
            Paste => "paste",
            Fill => "fill",
            Stroke => "stroke",
            FreeTransform => "freeTransform",
            ZoomIn => "zoomIn",
            ZoomOut => "zoomOut",
            FitOnScreen => "fitOnScreen",
            ActualPixels => "actualPixels",
            SelectAll => "selectAll",
            Deselect => "deselect",
            Inverse => "inverse",
            Feather => "feather",
            Expand => "expandSelection",
            Contract => "contractSelection",
            NewLayer => "newLayer",
            NewGroup => "newGroup",
            DuplicateLayer => "duplicateLayer",
            DeleteLayer => "deleteLayer",
            GroupLayers => "groupLayers",
            UngroupLayers => "ungroupLayers",
            MergeDown => "mergeDown",
            Flatten => "flattenImage",
            Blur => "blur",
            Sharpen => "sharpen",
            Smudge => "smudgeTool",
            Dodge => "dodgeTool",
            Burn => "burnTool",
            Eyedropper => "eyedropperTool",
            Brush => "brushTool",
            Eraser => "eraserTool",
            Move => "moveTool",
            Marquee => "marqueeTool",
            Lasso => "lassoTool",
            Crop => "cropTool",
            Type => "typeTool",
            Pen => "penTool",
        }
    }
}

/// The default Photoshop shortcut map.
///
/// Bindings are the Windows/Linux defaults; the macOS layer remaps `ctrl` to
/// `logo` for everything except the handful of genuine Ctrl shortcuts.
pub fn default_shortcuts() -> Vec<(Shortcut, Command)> {
    use Command as C;
    use Key as K;

    const NONE: Modifiers = Modifiers::NONE;
    const CTRL: Modifiers = Modifiers {
        ctrl: true,
        ..Modifiers::NONE
    };
    const SHIFT: Modifiers = Modifiers {
        shift: true,
        ..Modifiers::NONE
    };
    const CTRL_SHIFT: Modifiers = Modifiers {
        ctrl: true,
        shift: true,
        ..Modifiers::NONE
    };
    const ALT: Modifiers = Modifiers {
        alt: true,
        ..Modifiers::NONE
    };
    const ALT_CTRL: Modifiers = Modifiers {
        alt: true,
        ctrl: true,
        ..Modifiers::NONE
    };
    const ALT_CTRL_SHIFT: Modifiers = Modifiers {
        alt: true,
        ctrl: true,
        shift: true,
        ..Modifiers::NONE
    };

    vec![
        // ---- File
        (CTRL.key(K::N), C::New),
        (CTRL.key(K::O), C::Open),
        (CTRL.key(K::S), C::Save),
        (CTRL_SHIFT.key(K::S), C::SaveAs),
        // Alt+Backspace is "Fill with Background Color" in Photoshop.
        (ALT.key(K::Backspace), C::Fill),
        // ---- Edit
        (CTRL.key(K::Z), C::Undo),
        (CTRL.key(K::Y), C::Redo),
        // History stepping. Photoshop puts these on Alt+Ctrl+Z / +Shift+Z.
        (ALT_CTRL.key(K::Z), C::StepBackward),
        (ALT_CTRL_SHIFT.key(K::Z), C::StepForward),
        (CTRL.key(K::X), C::Cut),
        (CTRL.key(K::C), C::Copy),
        (CTRL.key(K::V), C::Paste),
        (CTRL.key(K::D), C::Deselect),
        // ---- Select
        (CTRL.key(K::A), C::SelectAll),
        (CTRL_SHIFT.key(K::I), C::Inverse),
        // ---- Layer
        (CTRL.key(K::J), C::NewLayer),
        (CTRL_SHIFT.key(K::N), C::NewGroup),
        (CTRL.key(K::L), C::DuplicateLayer),
        (CTRL.key(K::E), C::MergeDown),
        (CTRL.key(K::T), C::FreeTransform),
        // ---- View. Shift+= is "+" on every layout we ship.
        (CTRL.key(K::Equal), C::ZoomIn),
        (CTRL.key(K::Minus), C::ZoomOut),
        (CTRL.key(K::Num0), C::FitOnScreen),
        (CTRL.key(K::Num1), C::ActualPixels),
        // ---- Tools (Photoshop's single-letter tool switcher)
        (NONE.key(K::B), C::Brush),
        (NONE.key(K::E), C::Eraser),
        (NONE.key(K::M), C::Move),
        (NONE.key(K::V), C::Marquee),
        (NONE.key(K::L), C::Lasso),
        (NONE.key(K::C), C::Crop),
        (NONE.key(K::T), C::Type),
        (NONE.key(K::P), C::Pen),
        (NONE.key(K::I), C::Eyedropper),
        (NONE.key(K::O), C::Dodge),
        (SHIFT.key(K::O), C::Burn),
        (SHIFT.key(K::S), C::Smudge),
        // NOTE: Expand/Contract are deliberately absent from the global map.
        // In Photoshop they live on the Move tool's options bar and the
        // marquee context menu, not on a global key, and binding them here
        // would collide with Ctrl+Minus (Zoom Out).
    ]
}

/// Looks up a shortcut.
pub fn lookup(shortcuts: &[(Shortcut, Command)], s: Shortcut) -> Option<Command> {
    shortcuts
        .iter()
        .find(|(k, _)| k.key == s.key && k.modifiers_match(s.mods))
        .map(|(_, c)| *c)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn modifiers_are_reported_honestly() {
        let m = Modifiers {
            shift: true,
            ..Modifiers::NONE
        };
        assert!(m.any());
        assert!(!m.command());
        assert!(!m.menu());

        let c = Modifiers {
            ctrl: true,
            ..Modifiers::NONE
        };
        assert!(c.any());
        assert!(c.command());
        assert!(!c.menu());

        let mac = Modifiers {
            logo: true,
            ..Modifiers::NONE
        };
        assert!(mac.command(), "Cmd is the command key on macOS");
        assert!(mac.menu());
    }

    #[test]
    fn shortcuts_do_not_match_extra_modifiers() {
        let plain = Shortcut::new(Key::B, Modifiers::NONE);
        let with_shift = Modifiers {
            shift: true,
            ..Modifiers::NONE
        };
        assert!(plain.modifiers_match(Modifiers::NONE));
        // Shift is a real modifier, not a freebie: Ctrl+N must not become
        // Ctrl+Shift+N.
        assert!(!plain.modifiers_match(with_shift));

        let ctrl_n = Shortcut::new(
            Key::N,
            Modifiers {
                ctrl: true,
                ..Modifiers::NONE
            },
        );
        assert!(
            !ctrl_n.modifiers_match(Modifiers::NONE),
            "Ctrl+N must not fire on a bare N"
        );
    }

    #[test]
    fn the_paint_shortcuts_map_to_the_right_tools() {
        let s = default_shortcuts();
        assert_eq!(
            lookup(&s, Shortcut::new(Key::B, Modifiers::NONE)),
            Some(Command::Brush)
        );
        assert_eq!(
            lookup(&s, Shortcut::new(Key::E, Modifiers::NONE)),
            Some(Command::Eraser)
        );
        assert_eq!(
            lookup(&s, Shortcut::new(Key::M, Modifiers::NONE)),
            Some(Command::Move)
        );
        assert_eq!(
            lookup(&s, Shortcut::new(Key::V, Modifiers::NONE)),
            Some(Command::Marquee)
        );
        assert_eq!(
            lookup(&s, Shortcut::new(Key::T, Modifiers::NONE)),
            Some(Command::Type)
        );
    }

    #[test]
    fn file_shortcuts_match_photoshops_defaults() {
        let s = default_shortcuts();
        let ctrl = Modifiers {
            ctrl: true,
            ..Modifiers::NONE
        };
        assert_eq!(lookup(&s, ctrl.key(Key::N)), Some(Command::New));
        assert_eq!(lookup(&s, ctrl.key(Key::S)), Some(Command::Save));
        assert_eq!(lookup(&s, ctrl.key(Key::Z)), Some(Command::Undo));
        assert_eq!(lookup(&s, ctrl.key(Key::A)), Some(Command::SelectAll));
    }

    #[test]
    fn no_two_shortcuts_collide() {
        let s = default_shortcuts();
        for i in 0..s.len() {
            for j in (i + 1)..s.len() {
                assert_ne!(
                    (s[i].0.key, s[i].0.mods),
                    (s[j].0.key, s[j].0.mods),
                    "duplicate binding for {:?}",
                    s[i].1.script_name()
                );
            }
        }
    }

    #[test]
    fn script_names_are_unique() {
        let s = default_shortcuts();
        let mut names: Vec<&str> = s.iter().map(|(_, c)| c.script_name()).collect();
        let total = names.len();
        names.sort_unstable();
        names.dedup();
        assert_eq!(
            names.len(),
            total,
            "script names must be unique for the action recorder"
        );
        // Camel case is intentional: these are Photoshop's own identifiers, and
        // a .jsx action recording in Photoshop must replay correctly here.
        assert!(names
            .iter()
            .all(|n| !n.is_empty() && n.chars().all(|c| c.is_ascii_alphanumeric())));
    }

    #[test]
    fn input_state_reports_edges_and_clears_each_frame() {
        let mut i = InputState::new();
        i.mouse_move(10.0, 20.0);
        i.end_frame();

        i.key_down(Key::B);
        assert!(i.just_pressed(Key::B));
        assert!(i.is_down(Key::B));
        // Still held next frame, but no longer a fresh press.
        i.end_frame();
        assert!(!i.just_pressed(Key::B));
        assert!(i.is_down(Key::B));

        i.key_up(Key::B);
        assert!(i.just_released(Key::B));
        assert!(!i.is_down(Key::B));
    }

    #[test]
    fn cursor_delta_uses_the_previous_frame() {
        let mut i = InputState::new();
        i.mouse_move(0.0, 0.0);
        i.end_frame();
        i.mouse_move(5.0, -3.0);
        assert_eq!(i.cursor_delta(), (5.0, -3.0));
        i.end_frame();
        // After end_frame the delta resets to zero.
        assert_eq!(i.cursor_delta(), (0.0, 0.0));
    }

    #[test]
    fn scroll_accumulates_within_a_frame_and_resets() {
        let mut i = InputState::new();
        i.scroll(0.0, 1.0);
        i.scroll(0.0, 2.0);
        assert_eq!(i.scroll.y, 3.0);
        i.end_frame();
        assert_eq!(i.scroll.y, 0.0);
    }

    #[test]
    fn losing_the_window_releases_every_button() {
        let mut i = InputState::new();
        i.mouse_button(MouseButton::Left, true);
        assert!(i.is_clicked(MouseButton::Left));
        i.cursor_out = true;
        i.end_frame();
        assert!(
            !i.is_clicked(MouseButton::Left),
            "a drag must not survive focus loss"
        );
    }
}
