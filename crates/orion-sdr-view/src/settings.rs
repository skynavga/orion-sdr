use eframe::egui;

const OVERLAY_W: f32 = 480.0;
const OVERLAY_H: f32 = 320.0;
const ROW_H: f32 = 26.0;
const INDENT: f32 = 24.0;

// ── Tab index constants ────────────────────────────────────────────────────
const TAB_DISPLAY: usize = 0;
#[allow(dead_code)]
const TAB_SOURCE: usize = 1;
const N_TABS: usize = 2;
const TAB_NAMES: [&str; N_TABS] = ["Display", "Source"];

// ── Field descriptors ──────────────────────────────────────────────────────

/// A single editable numeric field.
struct Field {
    label: &'static str,
    /// Current value (pointer into ViewApp state passed at draw time).
    value: f32,
    default: f32,
    step: f32,
    min: f32,
    max: f32,
    /// Format string suffix (e.g. " dB", " Hz").
    unit: &'static str,
}

impl Field {
    fn nudge(&mut self, delta: f32) {
        self.value = (self.value + delta * self.step).clamp(self.min, self.max);
    }
    fn reset(&mut self) {
        self.value = self.default;
    }
}

// ── SettingsState ──────────────────────────────────────────────────────────

/// All mutable state for the settings popover.
pub struct SettingsState {
    pub visible: bool,
    active_tab: usize,
    /// Which field row is focused within the active tab (None = tab bar focused).
    focused_row: Option<usize>,

    // Display tab fields (indices 0..N_DISPLAY)
    display_fields: Vec<Field>,

    // Source tab fields (indices 0..N_SOURCE)
    source_fields: Vec<Field>,
}

impl SettingsState {
    pub fn new(
        db_min: f32,
        db_max: f32,
        freq_hz: f32,
        noise_amp: f32,
        amp_max: f32,
        ramp_secs: f32,
        pause_secs: f32,
    ) -> Self {
        Self {
            visible: false,
            active_tab: TAB_DISPLAY,
            focused_row: None,
            display_fields: vec![
                Field {
                    label: "dB min",
                    value: db_min,
                    default: -80.0,
                    step: 1.0,
                    min: -160.0,
                    max: -1.0,
                    unit: " dB",
                },
                Field {
                    label: "dB max",
                    value: db_max,
                    default: -20.0,
                    step: 1.0,
                    min: -159.0,
                    max: 0.0,
                    unit: " dB",
                },
            ],
            source_fields: vec![
                Field {
                    label: "Frequency",
                    value: freq_hz,
                    default: 3000.0,
                    step: 100.0,
                    min: 100.0,
                    max: 23_900.0,
                    unit: " Hz",
                },
                Field {
                    label: "Noise amp",
                    value: noise_amp,
                    default: 0.05,
                    step: 0.01,
                    min: 0.0,
                    max: 1.0,
                    unit: "",
                },
                Field {
                    label: "Tone amp max",
                    value: amp_max,
                    default: 0.65,
                    step: 0.05,
                    min: 0.0,
                    max: 1.0,
                    unit: "",
                },
                Field {
                    label: "Ramp secs",
                    value: ramp_secs,
                    default: 3.0,
                    step: 0.5,
                    min: 0.5,
                    max: 30.0,
                    unit: " s",
                },
                Field {
                    label: "Pause secs",
                    value: pause_secs,
                    default: 7.0,
                    step: 0.5,
                    min: 0.5,
                    max: 60.0,
                    unit: " s",
                },
            ],
        }
    }

    fn active_fields(&self) -> &[Field] {
        match self.active_tab {
            TAB_DISPLAY => &self.display_fields,
            _ => &self.source_fields,
        }
    }

    fn active_fields_mut(&mut self) -> &mut Vec<Field> {
        match self.active_tab {
            TAB_DISPLAY => &mut self.display_fields,
            _ => &mut self.source_fields,
        }
    }

    fn n_rows(&self) -> usize {
        self.active_fields().len()
    }

    /// Handle keyboard input. Returns true if a key was consumed.
    pub fn handle_keys(&mut self, ctx: &egui::Context) {
        if !self.visible {
            return;
        }
        ctx.input(|i| {
            // S or Escape: close (Escape deselects field first if one is focused)
            if i.key_pressed(egui::Key::S) {
                self.visible = false;
                self.focused_row = None;
                return;
            }
            if i.key_pressed(egui::Key::Escape) {
                if self.focused_row.is_some() {
                    self.focused_row = None;
                } else {
                    self.visible = false;
                }
                return;
            }

            // Tab / Shift-Tab: switch tabs, clear field focus
            if i.key_pressed(egui::Key::Tab) {
                if i.modifiers.shift {
                    self.active_tab = (self.active_tab + N_TABS - 1) % N_TABS;
                } else {
                    self.active_tab = (self.active_tab + 1) % N_TABS;
                }
                self.focused_row = None;
                return;
            }

            // Up/Down: navigate between field rows
            if i.key_pressed(egui::Key::ArrowUp) {
                self.focused_row = Some(match self.focused_row {
                    None => self.n_rows().saturating_sub(1),
                    Some(r) => r.saturating_sub(1),
                });
                return;
            }
            if i.key_pressed(egui::Key::ArrowDown) {
                let max = self.n_rows().saturating_sub(1);
                self.focused_row = Some(match self.focused_row {
                    None => 0,
                    Some(r) => (r + 1).min(max),
                });
                return;
            }

            // Left/Right: nudge focused field, or switch tabs if none focused
            if i.key_pressed(egui::Key::ArrowLeft) {
                if let Some(row) = self.focused_row {
                    let step = self.active_fields()[row].step;
                    self.active_fields_mut()[row].nudge(-1.0 * step.signum());
                } else {
                    self.active_tab = (self.active_tab + N_TABS - 1) % N_TABS;
                    self.focused_row = None;
                }
                return;
            }
            if i.key_pressed(egui::Key::ArrowRight) {
                if let Some(row) = self.focused_row {
                    let step = self.active_fields()[row].step;
                    self.active_fields_mut()[row].nudge(1.0 * step.signum());
                } else {
                    self.active_tab = (self.active_tab + 1) % N_TABS;
                    self.focused_row = None;
                }
                return;
            }

            // R: reset focused field (or all fields in tab if none focused)
            if i.key_pressed(egui::Key::R) {
                if let Some(row) = self.focused_row {
                    self.active_fields_mut()[row].reset();
                } else {
                    for f in self.active_fields_mut() {
                        f.reset();
                    }
                }
            }
        });
    }

    // ── Accessors for ViewApp to sync state back ───────────────────────────

    pub fn db_min(&self) -> f32 { self.display_fields[0].value }
    pub fn db_max(&self) -> f32 { self.display_fields[1].value }
    pub fn freq_hz(&self) -> f32 { self.source_fields[0].value }
    pub fn noise_amp(&self) -> f32 { self.source_fields[1].value }
    pub fn amp_max(&self) -> f32 { self.source_fields[2].value }
    pub fn ramp_secs(&self) -> f32 { self.source_fields[3].value }
    pub fn pause_secs(&self) -> f32 { self.source_fields[4].value }

    // ── Drawing ────────────────────────────────────────────────────────────

    pub fn draw(&self, ui: &mut egui::Ui, mono: &egui::FontId) {
        if !self.visible {
            return;
        }

        let screen = ui.ctx().content_rect();
        let rect = egui::Rect::from_center_size(
            screen.center(),
            egui::vec2(OVERLAY_W, OVERLAY_H),
        );

        let painter = ui.painter();

        // Background + border
        painter.rect_filled(
            rect,
            6.0,
            egui::Color32::from_rgba_premultiplied(15, 15, 30, 240),
        );
        painter.rect_stroke(
            rect,
            6.0,
            egui::Stroke::new(1.0, egui::Color32::from_gray(80)),
            egui::StrokeKind::Outside,
        );

        let small = egui::FontId::new(12.0, egui::FontFamily::Monospace);
        let med = egui::FontId::new(13.0, egui::FontFamily::Monospace);
        let mut y = rect.top() + 10.0;

        // ── Tab bar ────────────────────────────────────────────────────────
        let tab_w = (OVERLAY_W - 24.0) / N_TABS as f32;
        for (t, name) in TAB_NAMES.iter().enumerate() {
            let tab_rect = egui::Rect::from_min_size(
                egui::pos2(rect.left() + 12.0 + t as f32 * tab_w, y),
                egui::vec2(tab_w - 4.0, 22.0),
            );
            let active = t == self.active_tab && self.focused_row.is_none();
            let selected = t == self.active_tab;
            let bg = if selected {
                egui::Color32::from_rgb(40, 60, 100)
            } else {
                egui::Color32::from_gray(30)
            };
            painter.rect_filled(tab_rect, 4.0, bg);
            if active {
                painter.rect_stroke(
                    tab_rect,
                    4.0,
                    egui::Stroke::new(1.5, egui::Color32::from_rgb(100, 160, 255)),
                    egui::StrokeKind::Outside,
                );
            }
            painter.text(
                tab_rect.center(),
                egui::Align2::CENTER_CENTER,
                *name,
                mono.clone(),
                if selected { egui::Color32::WHITE } else { egui::Color32::from_gray(140) },
            );
        }
        y += 28.0;

        // Divider
        painter.hline(
            (rect.left() + 8.0)..=(rect.right() - 8.0),
            y,
            egui::Stroke::new(0.5, egui::Color32::from_gray(60)),
        );
        y += 8.0;

        // ── Fields ────────────────────────────────────────────────────────
        let fields = self.active_fields();
        for (row, field) in fields.iter().enumerate() {
            let focused = self.focused_row == Some(row);
            let row_rect = egui::Rect::from_min_size(
                egui::pos2(rect.left() + 8.0, y),
                egui::vec2(OVERLAY_W - 16.0, ROW_H),
            );

            // Highlight focused row
            if focused {
                painter.rect_filled(
                    row_rect,
                    3.0,
                    egui::Color32::from_rgba_premultiplied(60, 100, 180, 80),
                );
                painter.rect_stroke(
                    row_rect,
                    3.0,
                    egui::Stroke::new(1.0, egui::Color32::from_rgb(100, 160, 255)),
                    egui::StrokeKind::Outside,
                );
            }

            // Label
            painter.text(
                egui::pos2(rect.left() + INDENT, y + ROW_H / 2.0),
                egui::Align2::LEFT_CENTER,
                field.label,
                med.clone(),
                if focused { egui::Color32::WHITE } else { egui::Color32::from_gray(180) },
            );

            // Value
            let val_str = if field.step < 0.1 {
                format!("{:.2}{}", field.value, field.unit)
            } else if field.step < 1.0 {
                format!("{:.1}{}", field.value, field.unit)
            } else {
                format!("{:.0}{}", field.value, field.unit)
            };
            painter.text(
                egui::pos2(rect.right() - 120.0, y + ROW_H / 2.0),
                egui::Align2::LEFT_CENTER,
                val_str,
                med.clone(),
                egui::Color32::from_rgb(100, 220, 180),
            );

            // Nudge hint (only when focused)
            if focused {
                painter.text(
                    egui::pos2(rect.right() - 14.0, y + ROW_H / 2.0),
                    egui::Align2::RIGHT_CENTER,
                    "◀ ▶",
                    small.clone(),
                    egui::Color32::from_gray(140),
                );
            }

            y += ROW_H;
        }

        // ── Footer ────────────────────────────────────────────────────────
        y = rect.bottom() - 22.0;
        painter.hline(
            (rect.left() + 8.0)..=(rect.right() - 8.0),
            y,
            egui::Stroke::new(0.5, egui::Color32::from_gray(60)),
        );
        y += 6.0;
        let hint = if self.focused_row.is_some() {
            "↑↓ navigate   ◀▶ adjust   R reset field   Esc deselect"
        } else {
            "↑↓ select field   Tab switch tab   R reset all   Esc close"
        };
        painter.text(
            egui::pos2(rect.left() + 12.0, y),
            egui::Align2::LEFT_TOP,
            hint,
            small,
            egui::Color32::from_gray(110),
        );
    }
}
