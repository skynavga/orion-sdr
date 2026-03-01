use num_complex::Complex32 as C32;
use core::f32::consts::TAU;

/// Lightweight NCO used for RF/IF mixing in modulators.
/// Uses a phasor-recurrence oscillator (one complex multiply per sample)
/// instead of per-sample cos/sin.
#[derive(Debug, Clone)]
pub struct Nco {
    fs: f32,
    freq_hz: f32,
    z: C32,    // current phasor (cos + j*sin)
    w: C32,    // per-sample step: e^{j 2π f/fs}
    renorm_ctr: u32,
}

impl Nco {
    pub fn new(freq_hz: f32, fs: f32) -> Self {
        let dphi = TAU * freq_hz / fs;
        let (s, c) = dphi.sin_cos();
        Self {
            fs,
            freq_hz,
            z: C32::new(1.0, 0.0),
            w: C32::new(c, s),
            renorm_ctr: 0,
        }
    }

    #[inline]
    pub fn set_freq(&mut self, freq_hz: f32) {
        self.freq_hz = freq_hz;
        let dphi = TAU * freq_hz / self.fs;
        let (s, c) = dphi.sin_cos();
        self.w = C32::new(c, s);
    }

    /// Return (cos, sin) for the current phase and advance by one sample.
    #[inline(always)]
    pub fn next_cs(&mut self) -> (f32, f32) {
        // Complex multiply: z *= w
        let zr = self.z.re.mul_add(self.w.re, -self.z.im * self.w.im);
        let zi = self.z.im.mul_add(self.w.re,  self.z.re * self.w.im);
        self.z = C32::new(zr, zi);

        // Periodic renormalization to keep |z| ~ 1.0
        self.renorm_ctr = self.renorm_ctr.wrapping_add(1);
        if (self.renorm_ctr & 0x3FF) == 0 {
            let inv = (self.z.re * self.z.re + self.z.im * self.z.im).sqrt().recip();
            self.z.re *= inv;
            self.z.im *= inv;
        }
        (self.z.re, self.z.im)
    }
}

// Helper to multiply complex sample by e^{jθ} from an NCO
#[inline]
pub fn mix_with_nco(x: C32, nco: &mut Nco) -> C32 {
    let (c, s) = nco.next_cs();
    C32::new(x.re * c - x.im * s, x.re * s + x.im * c)
}
