mod persistence;
mod signal;
mod spectrum;
mod waterfall;

use eframe::egui;
use persistence::PersistenceRenderer;
use signal::TestSignalGen;
use spectrum::{RingBuffer, SpectrumProcessor};
use waterfall::WaterfallDisplay;

const PANE_NAMES: [&str; 3] = ["Spectrum", "Persistence", "Waterfall"];
const PANE_BG: [egui::Color32; 3] = [
    egui::Color32::from_rgb(10, 10, 20),
    egui::Color32::from_rgb(20, 50, 40),
    egui::Color32::from_rgb(40, 30, 60),
];

const FFT_SIZE: usize = 1024;
const SAMPLE_RATE: f32 = 48_000.0;
// Number of new samples fed per frame, targeting ~60 fps.
const SAMPLES_PER_FRAME: usize = (SAMPLE_RATE / 60.0) as usize;

struct ViewApp {
    pane_visible: [bool; 3],
    // Fractional height per pane — stored even when hidden so proportions are
    // remembered when re-shown. Future resize handles will mutate these values.
    pane_frac: [f32; 3],
    show_help: bool,
    mono_font_id: egui::FontId,

    // Signal source and spectrum processing
    signal_gen: TestSignalGen,
    ring_buf: RingBuffer,
    spectrum: SpectrumProcessor,
    db_min: f32,
    db_max: f32,

    // Pane 2: persistence density
    persistence: PersistenceRenderer,
    envelope_visible: bool,

    // Pane 3: waterfall
    waterfall: WaterfallDisplay,
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

            signal_gen: TestSignalGen::new(3_000.0, SAMPLE_RATE),
            ring_buf: RingBuffer::new(FFT_SIZE),
            spectrum: SpectrumProcessor::new(FFT_SIZE),
            db_min: -80.0,
            db_max: -20.0,

            persistence: PersistenceRenderer::new(FFT_SIZE / 2 + 1, 100),
            envelope_visible: true,

            waterfall: WaterfallDisplay::new(FFT_SIZE / 2 + 1, 512, -80.0, -20.0),
        }
    }

    fn handle_keys(&mut self, ctx: &egui::Context) {
        // Capture quit intent outside the closure to avoid deadlock:
        // ctx.input() holds a read lock; send_viewport_cmd() needs a write lock.
        let mut quit = false;
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
            if i.key_pressed(egui::Key::C) {
                if self.signal_gen.cycling {
                    self.signal_gen.stop_cycling();
                } else {
                    self.signal_gen.start_cycling();
                }
            }
            if i.key_pressed(egui::Key::E) {
                self.envelope_visible ^= true;
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
                quit = true;
            }
        });
        if quit {
            ctx.send_viewport_cmd(egui::ViewportCommand::Close);
        }
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
            painter.rect_filled(rect, 0.0, PANE_BG[i]);
            match i {
                0 => self.draw_spectrum(&painter, rect),
                1 => self.persistence.draw(&painter, rect, self.envelope_visible),
                _ => self.waterfall.draw(&painter, rect),
            }
            y += h;
        }
    }

    fn draw_spectrum(&self, painter: &egui::Painter, rect: egui::Rect) {
        let bins = &self.spectrum.fft_out_db;
        let n = bins.len();
        if n < 2 {
            return;
        }

        let x_for_bin =
            |b: usize| rect.left() + (b as f32 / (n - 1) as f32) * rect.width();
        let y_for_db = |db: f32| {
            let t = (db - self.db_min) / (self.db_max - self.db_min);
            rect.bottom() - t.clamp(0.0, 1.0) * rect.height()
        };

        // Horizontal dB grid lines
        let grid_stroke = egui::Stroke::new(0.5, egui::Color32::from_gray(45));
        let label_font = egui::FontId::new(10.0, egui::FontFamily::Monospace);
        let mut db = (self.db_min / 10.0).ceil() * 10.0;
        while db <= self.db_max {
            let y = y_for_db(db);
            painter.hline(rect.x_range(), y, grid_stroke);
            painter.text(
                egui::pos2(rect.left() + 4.0, y - 2.0),
                egui::Align2::LEFT_BOTTOM,
                format!("{:.0}dB", db),
                label_font.clone(),
                egui::Color32::from_gray(110),
            );
            db += 10.0;
        }

        // Vertical frequency grid lines + labels
        let nyquist = SAMPLE_RATE / 2.0;
        for frac in [0.0f32, 0.25, 0.5, 0.75, 1.0] {
            let hz = frac * nyquist;
            let b = (frac * (n - 1) as f32) as usize;
            let x = x_for_bin(b);
            painter.vline(x, rect.y_range(), grid_stroke);
            let label = if hz >= 1000.0 {
                format!("{:.1}kHz", hz / 1000.0)
            } else {
                format!("{:.0}Hz", hz)
            };
            painter.text(
                egui::pos2(x + 3.0, rect.bottom() - 2.0),
                egui::Align2::LEFT_BOTTOM,
                label,
                label_font.clone(),
                egui::Color32::from_gray(110),
            );
        }

        // Spectrum line
        let points: Vec<egui::Pos2> = (0..n)
            .map(|b| egui::pos2(x_for_bin(b), y_for_db(bins[b])))
            .collect();
        painter.line(points, egui::Stroke::new(1.5, egui::Color32::from_rgb(0, 220, 180)));
    }

    fn draw_help_overlay(&self, ui: &mut egui::Ui) {
        let screen = ui.ctx().content_rect();
        let overlay_rect = egui::Rect::from_center_size(
            screen.center(),
            egui::vec2(520.0, 264.0),
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
            ("C           toggle signal amplitude cycling (ramp up/down)", false),
            ("E           toggle persistence envelope overlay", false),
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
        // Feed new samples and process spectrum before drawing.
        for _ in 0..SAMPLES_PER_FRAME {
            let s = self.signal_gen.next_sample();
            self.ring_buf.push(s);
        }
        self.spectrum.process(&self.ring_buf);
        self.persistence.map.accumulate(&self.spectrum.fft_out_db, self.db_min, self.db_max);
        self.persistence.map.decay();
        self.persistence.update_texture(ctx);
        self.waterfall.push_row(&self.spectrum.fft_out_db);
        self.waterfall.update_texture(ctx);

        self.handle_keys(ctx);
        self.draw_hud(ctx);
        egui::CentralPanel::default().show(ctx, |ui| {
            self.draw_panes(ui);
            if self.show_help {
                self.draw_help_overlay(ui);
            }
        });

        // Request continuous repaints for live animation.
        ctx.request_repaint();
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
