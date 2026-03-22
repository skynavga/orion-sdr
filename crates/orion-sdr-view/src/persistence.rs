use eframe::egui;

/// Maps a normalized density value [0, 1] to a color.
/// t=0 (no hits) → pane background (dark navy).
/// t>0: dark blue → cyan → green → yellow → white.
fn density_color(t: f32, count: u32) -> egui::Color32 {
    if count == 0 {
        return egui::Color32::from_rgb(10, 10, 20); // matches PANE_BG[1] background
    }
    let t = t.clamp(0.0, 1.0);
    let (r, g, b) = if t < 0.25 {
        let s = t / 0.25;
        (0, 0, (80.0 + s * 175.0) as u8)
    } else if t < 0.5 {
        let s = (t - 0.25) / 0.25;
        (0, (s * 220.0) as u8, 255)
    } else if t < 0.75 {
        let s = (t - 0.5) / 0.25;
        (0, 220, (255.0 * (1.0 - s)) as u8)
    } else {
        let s = (t - 0.75) / 0.25;
        ((s * 255.0) as u8, 220, 0)
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
    /// Fraction multiplied into each count on each decay step (e.g. 0.85 → 15% drop per step).
    pub decay_factor: f32,
    /// Decay is applied once every this many frames to avoid wiping counts faster than
    /// accumulation can build them up.
    pub decay_every_n_frames: u32,
    frame_counter: u32,
}

impl PersistenceMap {
    pub fn new(freq_bins: usize, power_bins: usize) -> Self {
        Self {
            freq_bins,
            power_bins,
            counts: vec![0; freq_bins * power_bins],
            max_count: 1,
            decay_factor: 0.75,
            decay_every_n_frames: 30,
            frame_counter: 0,
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

    /// Apply decay once every `decay_every_n_frames` frames.
    /// Call this once per frame; it handles its own rate limiting.
    pub fn decay(&mut self) {
        self.frame_counter += 1;
        if self.frame_counter < self.decay_every_n_frames {
            return;
        }
        self.frame_counter = 0;
        let mut new_max = 1u32;
        for c in &mut self.counts {
            *c = (*c as f32 * self.decay_factor) as u32;
            if *c > new_max {
                new_max = *c;
            }
        }
        self.max_count = new_max;
    }

    /// For each frequency column, return the density-weighted mean power bin,
    /// or `None` if the column is empty. This follows the current amplitude
    /// as it cycles rather than being pinned to the historical peak.
    pub fn mean_power_bins(&self) -> Vec<Option<usize>> {
        (0..self.freq_bins)
            .map(|fb| {
                let mut weight_sum = 0u64;
                let mut bin_sum = 0u64;
                for pb in 0..self.power_bins {
                    let c = self.counts[pb * self.freq_bins + fb] as u64;
                    weight_sum += c;
                    bin_sum += c * pb as u64;
                }
                if weight_sum == 0 {
                    None
                } else {
                    Some((bin_sum / weight_sum) as usize)
                }
            })
            .collect()
    }

    /// Build a ColorImage from the current histogram.
    /// Row 0 = highest power (db_max), last row = lowest power (db_min).
    pub fn to_color_image(&self) -> egui::ColorImage {
        let mut pixels = Vec::with_capacity(self.freq_bins * self.power_bins);
        for pb in (0..self.power_bins).rev() {
            for fb in 0..self.freq_bins {
                let count = self.counts[pb * self.freq_bins + fb];
                let t = count as f32 / self.max_count as f32;
                pixels.push(density_color(t, count));
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

    /// Expose the texture handle for UV-cropped rendering by the caller.
    pub fn texture_handle(&self) -> Option<&egui::TextureHandle> {
        self.texture.as_ref()
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

    /// Draw the envelope cropped to UV range [lo_uv, hi_uv] (frequency zoom).
    pub fn draw_envelope_cropped(&self, painter: &egui::Painter, rect: egui::Rect, lo_uv: f32, hi_uv: f32) {
        let peaks = self.map.mean_power_bins();
        let n = peaks.len();
        if n < 2 { return; }

        let lo_bin = (lo_uv * (n - 1) as f32) as usize;
        let hi_bin = ((hi_uv * (n - 1) as f32) as usize).min(n - 1);
        if hi_bin <= lo_bin { return; }
        let vis_bins = hi_bin - lo_bin + 1;

        let y_for_pb = |pb: usize| {
            let t = pb as f32 / (self.map.power_bins - 1) as f32;
            rect.bottom() - t * rect.height()
        };
        let x_for_vis = |vi: usize| rect.left() + (vi as f32 / (vis_bins - 1) as f32) * rect.width();

        let stroke = egui::Stroke::new(1.0, egui::Color32::from_rgba_premultiplied(255, 255, 255, 160));
        let mut prev: Option<egui::Pos2> = None;
        for vi in 0..vis_bins {
            let fb = lo_bin + vi;
            let pt = peaks[fb].map(|pb| egui::pos2(x_for_vis(vi), y_for_pb(pb)));
            match (prev, pt) {
                (Some(a), Some(b)) => { painter.line_segment([a, b], stroke); prev = Some(b); }
                (_, Some(b)) => prev = Some(b),
                (_, None) => prev = None,
            }
        }
    }

    /// Draw the persistence heatmap into `rect`, optionally with an envelope outline on top.
    pub fn draw(&self, painter: &egui::Painter, rect: egui::Rect, show_envelope: bool) {
        if let Some(tex) = &self.texture {
            let uv = egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0));
            painter.image(tex.id(), rect, uv, egui::Color32::WHITE);
        }

        if !show_envelope {
            return;
        }

        // Envelope: connect the density-weighted mean power bin of each frequency column.
        let peaks = self.map.mean_power_bins();
        let n = peaks.len();
        if n < 2 {
            return;
        }

        // power_bin → Y pixel: bin (power_bins-1) = db_max = rect.top(),
        //                       bin 0             = db_min = rect.bottom().
        let y_for_pb = |pb: usize| {
            let t = pb as f32 / (self.map.power_bins - 1) as f32;
            rect.bottom() - t * rect.height()
        };
        let x_for_fb =
            |fb: usize| rect.left() + (fb as f32 / (n - 1) as f32) * rect.width();

        let stroke = egui::Stroke::new(1.0, egui::Color32::from_rgba_premultiplied(255, 255, 255, 160));

        // Walk columns, emitting line segments between consecutive non-empty columns.
        let mut prev: Option<egui::Pos2> = None;
        for fb in 0..n {
            let pt = peaks[fb].map(|pb| egui::pos2(x_for_fb(fb), y_for_pb(pb)));
            match (prev, pt) {
                (Some(a), Some(b)) => {
                    painter.line_segment([a, b], stroke);
                    prev = Some(b);
                }
                (_, Some(b)) => prev = Some(b),
                (_, None) => prev = None,
            }
        }
    }
}
