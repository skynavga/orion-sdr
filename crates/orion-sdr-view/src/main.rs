use eframe::egui;

const PANE_NAMES: [&str; 3] = ["Spectrum", "Persistence", "Waterfall"];
const PANE_COLORS: [egui::Color32; 3] = [
    egui::Color32::from_rgb(30, 40, 60),
    egui::Color32::from_rgb(20, 50, 40),
    egui::Color32::from_rgb(40, 30, 60),
];

struct ViewApp {
    pane_visible: [bool; 3],
    // Fractional height per pane — stored even when hidden so proportions are
    // remembered when re-shown. Future resize handles will mutate these values.
    pane_frac: [f32; 3],
    show_help: bool,
    mono_font_id: egui::FontId,
}

impl ViewApp {
    fn new(cc: &eframe::CreationContext<'_>) -> Self {
        let font_bytes = include_bytes!("../assets/fonts/DejaVuSansMono.ttf");
        let mut fonts = egui::FontDefinitions::default();
        fonts.font_data.insert(
            "DejaVuSansMono".to_owned(),
            egui::FontData::from_static(font_bytes).into(),
        );
        fonts
            .families
            .entry(egui::FontFamily::Monospace)
            .or_default()
            .insert(0, "DejaVuSansMono".to_owned());
        cc.egui_ctx.set_fonts(fonts);

        Self {
            pane_visible: [true; 3],
            pane_frac: [1.0 / 3.0; 3],
            show_help: false,
            mono_font_id: egui::FontId::new(14.0, egui::FontFamily::Monospace),
        }
    }

    fn handle_keys(&mut self, ctx: &egui::Context) {
        ctx.input(|i| {
            if i.key_pressed(egui::Key::Num1) {
                self.pane_visible[0] ^= true;
            }
            if i.key_pressed(egui::Key::Num2) {
                self.pane_visible[1] ^= true;
            }
            if i.key_pressed(egui::Key::Num3) {
                self.pane_visible[2] ^= true;
            }
            if i.key_pressed(egui::Key::H) {
                self.show_help ^= true;
            }
            // '?' — match via Event::Text for cross-layout reliability
            for e in &i.events {
                if let egui::Event::Text(s) = e {
                    if s == "?" {
                        self.show_help ^= true;
                    }
                }
            }
            if i.key_pressed(egui::Key::Escape) {
                self.show_help = false;
            }
            if i.key_pressed(egui::Key::Q) {
                ctx.send_viewport_cmd(egui::ViewportCommand::Close);
            }
        });
    }

    fn draw_hud(&self, ctx: &egui::Context) {
        egui::TopBottomPanel::top("hud").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.label(
                    egui::RichText::new("orion-sdr-view")
                        .font(self.mono_font_id.clone())
                        .strong(),
                );
                ui.separator();
                for (i, name) in PANE_NAMES.iter().enumerate() {
                    let active = self.pane_visible[i];
                    ui.label(
                        egui::RichText::new(format!("[{}] {}", i + 1, name))
                            .font(self.mono_font_id.clone())
                            .color(if active {
                                egui::Color32::WHITE
                            } else {
                                egui::Color32::DARK_GRAY
                            }),
                    );
                }
                ui.separator();
                ui.label(
                    egui::RichText::new("? help  Q quit")
                        .font(self.mono_font_id.clone())
                        .color(egui::Color32::GRAY),
                );
            });
        });
    }

    fn draw_panes(&self, ui: &mut egui::Ui) {
        let visible_count = self.pane_visible.iter().filter(|&&v| v).count();
        if visible_count == 0 {
            return;
        }

        let avail = ui.available_rect_before_wrap();
        let total_h = avail.height();

        let total_frac: f32 = self
            .pane_visible
            .iter()
            .zip(self.pane_frac.iter())
            .map(|(&vis, &f)| if vis { f } else { 0.0 })
            .sum();

        let mut y = avail.top();
        for i in 0..3 {
            if !self.pane_visible[i] {
                continue;
            }
            let h = (self.pane_frac[i] / total_frac) * total_h;
            let rect = egui::Rect::from_min_size(
                egui::pos2(avail.left(), y),
                egui::vec2(avail.width(), h),
            );
            let painter = ui.painter_at(rect);
            painter.rect_filled(rect, 0.0, PANE_COLORS[i]);
            painter.text(
                rect.center(),
                egui::Align2::CENTER_CENTER,
                PANE_NAMES[i],
                self.mono_font_id.clone(),
                egui::Color32::WHITE,
            );
            y += h;
        }
    }

    fn draw_help_overlay(&self, ui: &mut egui::Ui) {
        let screen = ui.ctx().content_rect();
        let overlay_rect = egui::Rect::from_center_size(
            screen.center(),
            egui::vec2(520.0, 220.0),
        );
        let painter = ui.painter();
        painter.rect_filled(
            overlay_rect,
            8.0,
            egui::Color32::from_rgba_premultiplied(0, 0, 0, 220),
        );
        painter.rect_stroke(
            overlay_rect,
            8.0,
            egui::Stroke::new(1.0, egui::Color32::GRAY),
            egui::StrokeKind::Outside,
        );

        let lines: &[(&str, bool)] = &[
            ("Keyboard shortcuts", true),
            ("1 / 2 / 3   toggle Spectrum / Persistence / Waterfall panes", false),
            ("? or H      toggle this help overlay", false),
            ("Escape      dismiss overlays", false),
            ("Q           quit", false),
        ];
        let mut y = overlay_rect.top() + 18.0;
        for (text, is_header) in lines {
            let size = if *is_header { 16.0 } else { 13.0 };
            painter.text(
                egui::pos2(overlay_rect.left() + 20.0, y),
                egui::Align2::LEFT_TOP,
                *text,
                egui::FontId::new(size, egui::FontFamily::Monospace),
                egui::Color32::WHITE,
            );
            y += if *is_header { 28.0 } else { 22.0 };
        }
    }
}

impl eframe::App for ViewApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.handle_keys(ctx);
        self.draw_hud(ctx);
        egui::CentralPanel::default().show(ctx, |ui| {
            self.draw_panes(ui);
            if self.show_help {
                self.draw_help_overlay(ui);
            }
        });
    }
}

fn main() -> eframe::Result<()> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("orion-sdr-view")
            .with_inner_size([1200.0, 800.0]),
        ..Default::default()
    };
    eframe::run_native(
        "orion-sdr-view",
        options,
        Box::new(|cc| Ok(Box::new(ViewApp::new(cc)))),
    )
}
