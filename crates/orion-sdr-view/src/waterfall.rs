use eframe::egui;

/// Maps a dB value to a waterfall color (thermal palette).
/// db_min → black/deep blue, db_max → yellow/white.
fn db_to_color(db: f32, db_min: f32, db_max: f32) -> egui::Color32 {
    let t = ((db - db_min) / (db_max - db_min)).clamp(0.0, 1.0);
    let (r, g, b) = if t < 0.33 {
        let s = t / 0.33;
        (0, 0, (s * 255.0) as u8)
    } else if t < 0.66 {
        let s = (t - 0.33) / 0.33;
        (0, (s * 255.0) as u8, (255.0 * (1.0 - s)) as u8)
    } else {
        let s = (t - 0.66) / 0.34;
        ((s * 255.0) as u8, 255, 0)
    };
    egui::Color32::from_rgb(r, g, b)
}

/// Scrolling 2D spectrogram display.
///
/// New rows are prepended at the top; older rows shift down.
/// `max_rows` controls how much time history is kept.
pub struct WaterfallDisplay {
    pub freq_bins: usize,
    pub max_rows: usize,
    /// Pixel buffer: `max_rows × freq_bins`, row 0 = newest.
    pixels: Vec<egui::Color32>,
    current_rows: usize,
    texture: Option<egui::TextureHandle>,
    pub db_min: f32,
    pub db_max: f32,
}

impl WaterfallDisplay {
    pub fn new(freq_bins: usize, max_rows: usize, db_min: f32, db_max: f32) -> Self {
        Self {
            freq_bins,
            max_rows,
            pixels: vec![egui::Color32::BLACK; freq_bins * max_rows],
            current_rows: 0,
            texture: None,
            db_min,
            db_max,
        }
    }

    /// Prepend one new row (newest at top), shifting existing rows down by one.
    pub fn push_row(&mut self, spectrum_db: &[f32]) {
        let n = spectrum_db.len().min(self.freq_bins);
        let filled = self.current_rows.min(self.max_rows - 1);

        // Shift existing rows down by one to make room at index 0.
        self.pixels.copy_within(0..filled * self.freq_bins, self.freq_bins);

        // Write new row at the top.
        for i in 0..n {
            self.pixels[i] = db_to_color(spectrum_db[i], self.db_min, self.db_max);
        }
        for i in n..self.freq_bins {
            self.pixels[i] = egui::Color32::BLACK;
        }

        self.current_rows = (self.current_rows + 1).min(self.max_rows);
    }

    /// Upload the current pixel buffer to a GPU texture. Call once per frame.
    pub fn update_texture(&mut self, ctx: &egui::Context) {
        let rows = self.current_rows;
        if rows == 0 {
            return;
        }
        let rgba: Vec<u8> = self.pixels[..rows * self.freq_bins]
            .iter()
            .flat_map(|c| [c.r(), c.g(), c.b(), 255])
            .collect();
        let image = egui::ColorImage::from_rgba_unmultiplied([self.freq_bins, rows], &rgba);
        match &mut self.texture {
            Some(h) => h.set(image, egui::TextureOptions::NEAREST),
            None => {
                self.texture =
                    Some(ctx.load_texture("waterfall", image, egui::TextureOptions::NEAREST));
            }
        }
    }

    /// Expose the texture handle for UV-cropped rendering by the caller.
    pub fn texture_handle(&self) -> Option<&egui::TextureHandle> {
        self.texture.as_ref()
    }

    /// Draw the waterfall into `rect` (full UV, no zoom).
    pub fn draw(&self, painter: &egui::Painter, rect: egui::Rect) {
        if let Some(tex) = &self.texture {
            let uv = egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0));
            painter.image(tex.id(), rect, uv, egui::Color32::WHITE);
        }
    }
}
