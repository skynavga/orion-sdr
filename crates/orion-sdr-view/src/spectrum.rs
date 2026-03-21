use num_complex::Complex32;
use rustfft::Fft;
use std::sync::Arc;

/// Circular buffer of f32 samples.
pub struct RingBuffer {
    buf: Vec<f32>,
    head: usize,
}

impl RingBuffer {
    pub fn new(capacity: usize) -> Self {
        Self {
            buf: vec![0.0; capacity],
            head: 0,
        }
    }

    pub fn push(&mut self, sample: f32) {
        self.buf[self.head] = sample;
        self.head = (self.head + 1) % self.buf.len();
    }

    /// Copy samples into `out` in oldest-to-newest order, filling at most `out.len()` samples.
    pub fn fill_linear(&self, out: &mut Vec<f32>) {
        let cap = self.buf.len();
        let n = out.len().min(cap);
        // `head` is the next write position, so `head` is the oldest sample.
        let start = (self.head + cap - n) % cap;
        for i in 0..n {
            out[i] = self.buf[(start + i) % cap];
        }
    }
}

/// Computes a magnitude spectrum (dBFS) from the latest samples in a RingBuffer.
pub struct SpectrumProcessor {
    pub fft_size: usize,
    window: Vec<f32>,
    fft: Arc<dyn Fft<f32>>,
    scratch_real: Vec<f32>,
    scratch_complex: Vec<Complex32>,
    /// Positive-frequency magnitude bins in dBFS; length = fft_size / 2 + 1.
    pub fft_out_db: Vec<f32>,
}

impl SpectrumProcessor {
    pub fn new(fft_size: usize) -> Self {
        let mut planner = rustfft::FftPlanner::new();
        let fft = planner.plan_fft_forward(fft_size);
        let window = hann_window(fft_size);
        let bins = fft_size / 2 + 1;
        Self {
            fft_size,
            window,
            fft,
            scratch_real: vec![0.0; fft_size],
            scratch_complex: vec![Complex32::new(0.0, 0.0); fft_size],
            fft_out_db: vec![-120.0; bins],
        }
    }

    /// Fill from the ring buffer, apply a Hann window, run the FFT, and update `fft_out_db`.
    pub fn process(&mut self, ring: &RingBuffer) {
        ring.fill_linear(&mut self.scratch_real);

        // Apply Hann window and load into complex scratch buffer.
        for (i, (c, w)) in self
            .scratch_complex
            .iter_mut()
            .zip(self.window.iter())
            .enumerate()
        {
            c.re = self.scratch_real[i] * w;
            c.im = 0.0;
        }

        self.fft.process(&mut self.scratch_complex);

        // Compute dBFS for positive-frequency bins only.
        let scale = 1.0 / self.fft_size as f32;
        for (i, c) in self.scratch_complex[..self.fft_out_db.len()]
            .iter()
            .enumerate()
        {
            let mag_sq = (c.re * c.re + c.im * c.im) * scale * scale;
            self.fft_out_db[i] = 10.0 * (mag_sq + 1e-12).log10();
        }
    }
}

fn hann_window(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| {
            0.5 * (1.0 - (2.0 * std::f32::consts::PI * i as f32 / (n - 1) as f32).cos())
        })
        .collect()
}
