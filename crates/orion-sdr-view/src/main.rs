mod persistence;
mod settings;
mod signal;
mod source;
mod spectrum;
mod waterfall;

use eframe::egui;
use persistence::PersistenceRenderer;
use settings::SettingsState;
use signal::TestSignalGen;
use source::{AmDsbSource, BuiltinAudio, SignalSource, TestToneSource, load_builtin};
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

// ── Source mode ───────────────────────────────────────────────────────────────

#[derive(Clone, Copy, PartialEq, Eq)]
enum SourceMode {
    TestTone,
    AmDsb,
}

impl SourceMode {
    const ALL: &'static [SourceMode] = &[SourceMode::TestTone, SourceMode::AmDsb];

    fn label(self) -> &'static str {
        match self {
            SourceMode::TestTone => "Test Tone",
            SourceMode::AmDsb => "AM DSB",
        }
    }

    fn index(self) -> usize {
        Self::ALL.iter().position(|&m| m == self).unwrap_or(0)
    }

    fn next(self) -> Self {
        let idx = (self.index() + 1) % Self::ALL.len();
        Self::ALL[idx]
    }
}

// ── ViewApp ───────────────────────────────────────────────────────────────────

struct ViewApp {
    pane_visible: [bool; 3],
    // Fractional height per pane — stored even when hidden so proportions are
    // remembered when re-shown. Future resize handles will mutate these values.
    pane_frac: [f32; 3],
    show_help: bool,
    mono_font_id: egui::FontId,

    // Active signal source (Box<dyn SignalSource> for easy future extension)
    source_mode: SourceMode,
    source: Box<dyn SignalSource>,

    // Test tone generator — kept alive so its state (cycling, settings) persists
    // across source switches. TestToneSource borrows it when active.
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

    // Settings popover
    settings: SettingsState,
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

        let signal_gen = TestSignalGen::new(3_000.0, SAMPLE_RATE);
        // The active source starts as TestTone; we clone the gen's initial state.
        let source: Box<dyn SignalSource> =
            Box::new(TestToneSource::new(TestSignalGen::new(3_000.0, SAMPLE_RATE)));

        Self {
            pane_visible: [true; 3],
            pane_frac: [1.0 / 3.0; 3],
            show_help: false,
            mono_font_id: egui::FontId::new(14.0, egui::FontFamily::Monospace),

            source_mode: SourceMode::TestTone,
            source,
            signal_gen,

            ring_buf: RingBuffer::new(FFT_SIZE),
            spectrum: SpectrumProcessor::new(FFT_SIZE),
            db_min: -80.0,
            db_max: -20.0,

            persistence: PersistenceRenderer::new(FFT_SIZE / 2 + 1, 100),
            envelope_visible: true,

            waterfall: WaterfallDisplay::new(FFT_SIZE / 2 + 1, 512, -80.0, -20.0),

            settings: SettingsState::new(
                -80.0, -20.0,           // db_min, db_max
                3_000.0,                // freq_hz
                0.05,                   // noise_amp
                0.65,                   // amp_max
                3.0,                    // ramp_secs
                7.0,                    // pause_secs
            ),
        }
    }

    /// Build a fresh AmDsbSource from current settings values.
    fn make_am_source(&self) -> AmDsbSource {
        let (audio, audio_rate) = if self.settings.am_audio_is_custom() {
            // Custom with no path yet — start silent; audio loaded on WAV entry
            (Vec::new(), SAMPLE_RATE)
        } else {
            let builtin = BuiltinAudio::ALL[self.settings.am_audio_idx().min(BuiltinAudio::ALL.len() - 1)];
            load_builtin(builtin)
        };
        AmDsbSource::new(
            audio,
            audio_rate,
            self.settings.am_carrier_hz(),
            self.settings.am_mod_index(),
            self.settings.am_loop_gap_secs(),
            self.settings.am_noise_amp(),
            SAMPLE_RATE,
        )
    }

    /// Switch the active source to `mode`, constructing a new source box.
    fn switch_source(&mut self, mode: SourceMode) {
        self.source_mode = mode;
        self.source = match mode {
            SourceMode::TestTone => {
                // Re-create from signal_gen's current settings
                Box::new(TestToneSource::new(TestSignalGen::new(
                    self.signal_gen.freq_hz,
                    SAMPLE_RATE,
                )))
            }
            SourceMode::AmDsb => Box::new(self.make_am_source()),
        };
        self.settings.set_source_mode(mode as usize);
    }

    fn handle_keys(&mut self, ctx: &egui::Context) {
        // Settings popover consumes arrow/tab/escape/R keys when visible.
        if self.settings.visible {
            let result = self.settings.handle_keys(ctx);
            if result.source_switched {
                let idx = self.settings.source_mode_idx().min(SourceMode::ALL.len() - 1);
                let new_mode = SourceMode::ALL[idx];
                if new_mode != self.source_mode {
                    self.switch_source(new_mode);
                }
            }
            if result.am_audio_changed {
                self.reload_builtin_audio();
            }
            if result.wav_load_requested {
                self.try_load_wav();
            }
            self.sync_settings();
            return;
        }

        let mut quit = false;
        let mut toggle_source = false;
        ctx.input(|i| {
            if i.key_pressed(egui::Key::Num1) { self.pane_visible[0] ^= true; }
            if i.key_pressed(egui::Key::Num2) { self.pane_visible[1] ^= true; }
            if i.key_pressed(egui::Key::Num3) { self.pane_visible[2] ^= true; }
            if i.key_pressed(egui::Key::I) { toggle_source = true; }
            if i.key_pressed(egui::Key::C) {
                if self.signal_gen.cycling {
                    self.signal_gen.stop_cycling();
                } else {
                    self.signal_gen.start_cycling();
                }
                // Propagate to active TestToneSource if applicable
                if let Some(tts) = self.source.as_any_mut().downcast_mut::<TestToneSource>() {
                    if tts.signal_gen.cycling {
                        tts.signal_gen.stop_cycling();
                    } else {
                        tts.signal_gen.start_cycling();
                    }
                }
            }
            if i.key_pressed(egui::Key::E) { self.envelope_visible ^= true; }
            if i.key_pressed(egui::Key::S) { self.settings.visible ^= true; }
            if i.key_pressed(egui::Key::H) { self.show_help ^= true; }
            for e in &i.events {
                if let egui::Event::Text(s) = e {
                    if s == "?" { self.show_help ^= true; }
                }
            }
            if i.key_pressed(egui::Key::Escape) {
                self.show_help = false;
                self.settings.visible = false;
            }
            if i.key_pressed(egui::Key::Q) { quit = true; }
        });

        if toggle_source {
            self.switch_source(self.source_mode.next());
        }
        if quit {
            ctx.send_viewport_cmd(egui::ViewportCommand::Close);
        }
    }

    /// Reload the built-in audio buffer into the active AmDsbSource after the
    /// AM audio toggle changes (Morse ↔ Voice). No-op if source is not AM DSB
    /// or if Custom is selected (user WAV takes precedence).
    fn reload_builtin_audio(&mut self) {
        if self.source_mode != SourceMode::AmDsb {
            return;
        }
        if self.settings.am_audio_is_custom() {
            return;
        }
        let builtin = BuiltinAudio::ALL[self.settings.am_audio_idx()];
        let (audio, rate) = load_builtin(builtin);
        if let Some(am) = self.source.as_any_mut().downcast_mut::<AmDsbSource>() {
            am.set_audio(audio, rate);
        }
    }

    /// Attempt to load the WAV path from settings into the AM DSB source.
    fn try_load_wav(&mut self) {
        let path_str = self.settings.wav_path().to_owned();
        if path_str.is_empty() {
            self.settings.set_wav_status(false);
            return;
        }
        match source::load_wav_file(std::path::Path::new(&path_str)) {
            Ok((audio, rate)) => {
                if self.source_mode == SourceMode::AmDsb {
                    if let Some(am) = self.source.as_any_mut().downcast_mut::<AmDsbSource>() {
                        am.set_audio(audio, rate);
                    }
                }
                self.settings.set_wav_status(true);
            }
            Err(_) => {
                self.settings.set_wav_status(false);
            }
        }
    }

    /// Push current settings values into live signal/display state.
    fn sync_settings(&mut self) {
        self.db_min = self.settings.db_min();
        self.db_max = self.settings.db_max();
        self.waterfall.db_min = self.settings.db_min();
        self.waterfall.db_max = self.settings.db_max();
        self.signal_gen.freq_hz = self.settings.freq_hz();
        self.signal_gen.noise_amp = self.settings.noise_amp();
        self.signal_gen.amp_max = self.settings.amp_max();
        self.signal_gen.ramp_secs = self.settings.ramp_secs();
        self.signal_gen.pause_secs = self.settings.pause_secs();

        // Propagate test-tone settings into the active source if applicable
        if let Some(tts) = self.source.as_any_mut().downcast_mut::<TestToneSource>() {
            tts.signal_gen.freq_hz = self.settings.freq_hz();
            tts.signal_gen.noise_amp = self.settings.noise_amp();
            tts.signal_gen.amp_max = self.settings.amp_max();
            tts.signal_gen.ramp_secs = self.settings.ramp_secs();
            tts.signal_gen.pause_secs = self.settings.pause_secs();
        }

        // Propagate AM DSB settings into the active source if applicable
        if let Some(am) = self.source.as_any_mut().downcast_mut::<AmDsbSource>() {
            let carrier_changed = (am.carrier_hz - self.settings.am_carrier_hz()).abs() > 0.5;
            let index_changed = (am.mod_index - self.settings.am_mod_index()).abs() > 0.001;
            am.carrier_hz = self.settings.am_carrier_hz();
            am.mod_index = self.settings.am_mod_index();
            if carrier_changed || index_changed {
                am.rebuild_mod();
            }
            let gap_changed = (am.loop_gap_secs - self.settings.am_loop_gap_secs()).abs() > 0.01;
            if gap_changed {
                am.loop_gap_secs = self.settings.am_loop_gap_secs();
                am.update_loop_gap();
            }
            am.noise_amp = self.settings.am_noise_amp();
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
                    egui::RichText::new(format!(
                        "SRC: {}  I input  S settings  ? help  Q quit",
                        self.source_mode.label()
                    ))
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
            egui::vec2(540.0, 308.0),
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
            ("I           select next input source", false),
            ("C           toggle signal amplitude cycling (Test Tone only)", false),
            ("E           toggle persistence envelope overlay", false),
            ("S           open/close settings popover", false),
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

// ── eframe::App ───────────────────────────────────────────────────────────────

impl eframe::App for ViewApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Feed new samples and process spectrum before drawing.
        let samples = self.source.next_samples(SAMPLES_PER_FRAME);
        for s in samples {
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
            let mono = self.mono_font_id.clone();
            self.settings.draw(ui, &mono);
        });

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
