use eframe::egui;

/// Maps a normalized density value [0, 1] to a color.
/// Gradient: black → dark blue → cyan → green → yellow → white.
fn density_color(t: f32) -> egui::Color32 {
    let t = t.clamp(0.0, 1.0);
    let (r, g, b) = if t < 0.25 {
        let s = t / 0.25;
        (0, 0, (s * 180.0) as u8)
    } else if t < 0.5 {
        let s = (t - 0.25) / 0.25;
        (0, (s * 200.0) as u8, 180)
    } else if t < 0.75 {
        let s = (t - 0.5) / 0.25;
        (0, 200, (180.0 * (1.0 - s)) as u8)
    } else {
        let s = (t - 0.75) / 0.25;
        ((s * 255.0) as u8, 200, 0)
    };
    egui::Color32::from_rgb(r, g, b)
}

/// 2D histogram accumulating spectral density over time.
///
/// Layout: `counts[power_bin * freq_bins + freq_bin]`
/// Power bin 0 = db_min, power_bin (power_bins-1) = db_max.
pub struct PersistenceMap {
    pub freq_bins: usize,
    pub power_bins: usize,
    counts: Vec<u32>,
    max_count: u32,
    /// Fraction subtracted from each count per frame. ~0.005 → ~3 s persistence at 60 fps.
    pub decay_rate: f32,
}

impl PersistenceMap {
    pub fn new(freq_bins: usize, power_bins: usize) -> Self {
        Self {
            freq_bins,
            power_bins,
            counts: vec![0; freq_bins * power_bins],
            max_count: 1,
            decay_rate: 0.005,
        }
    }

    /// Accumulate one spectrum frame into the histogram.
    pub fn accumulate(&mut self, spectrum_db: &[f32], db_min: f32, db_max: f32) {
        let db_range = db_max - db_min;
        let n = spectrum_db.len().min(self.freq_bins);
        for fb in 0..n {
            let t = (spectrum_db[fb] - db_min) / db_range;
            let pb = ((t * (self.power_bins - 1) as f32).round() as usize)
                .clamp(0, self.power_bins - 1);
            let idx = pb * self.freq_bins + fb;
            self.counts[idx] = self.counts[idx].saturating_add(1);
            if self.counts[idx] > self.max_count {
                self.max_count = self.counts[idx];
            }
        }
    }

    /// Apply exponential decay to all counts.
    pub fn decay(&mut self) {
        let scale = 1.0 - self.decay_rate;
        let mut new_max = 1u32;
        for c in &mut self.counts {
            *c = (*c as f32 * scale) as u32;
            if *c > new_max {
                new_max = *c;
            }
        }
        self.max_count = new_max;
    }

    /// Build a ColorImage from the current histogram.
    /// Row 0 = highest power (db_max), last row = lowest power (db_min).
    pub fn to_color_image(&self) -> egui::ColorImage {
        let mut pixels = Vec::with_capacity(self.freq_bins * self.power_bins);
        for pb in (0..self.power_bins).rev() {
            for fb in 0..self.freq_bins {
                let count = self.counts[pb * self.freq_bins + fb];
                let t = count as f32 / self.max_count as f32;
                pixels.push(density_color(t));
            }
        }
        egui::ColorImage::from_rgba_unmultiplied(
            [self.freq_bins, self.power_bins],
            &pixels
                .iter()
                .flat_map(|c| [c.r(), c.g(), c.b(), 255])
                .collect::<Vec<u8>>(),
        )
    }
}

/// Wraps PersistenceMap and manages the GPU texture handle.
pub struct PersistenceRenderer {
    pub map: PersistenceMap,
    texture: Option<egui::TextureHandle>,
}

impl PersistenceRenderer {
    pub fn new(freq_bins: usize, power_bins: usize) -> Self {
        Self {
            map: PersistenceMap::new(freq_bins, power_bins),
            texture: None,
        }
    }

    /// Rebuild the texture from the current map state. Call once per frame.
    pub fn update_texture(&mut self, ctx: &egui::Context) {
        let image = self.map.to_color_image();
        match &mut self.texture {
            Some(h) => h.set(image, egui::TextureOptions::NEAREST),
            None => {
                self.texture =
                    Some(ctx.load_texture("persistence", image, egui::TextureOptions::NEAREST));
            }
        }
    }

    /// Draw the persistence heatmap into `rect`.
    pub fn draw(&self, painter: &egui::Painter, rect: egui::Rect) {
        if let Some(tex) = &self.texture {
            let uv = egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0));
            painter.image(tex.id(), rect, uv, egui::Color32::WHITE);
        }
    }
}
