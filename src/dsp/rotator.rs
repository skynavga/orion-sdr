use num_complex::Complex32 as C32;

/// Complex oscillator / frequency translator without per-sample trig.
#[derive(Clone, Copy, Debug)]
pub struct Rotator {
    z: C32,      // current phasor
    w: C32,      // per-sample step: e^{j 2π f/fs}
    renorm_ctr: u32,
}

impl Rotator {
    #[inline]
    pub fn new(freq_hz: f32, fs: f32) -> Self {
        let phi = core::f32::consts::TAU * freq_hz / fs;
        let (s, c) = phi.sin_cos(); // one call, both results
        Self {
            z: C32::new(1.0, 0.0),
            w: C32::new(c, s),
            renorm_ctr: 0,
        }
    }

    /// Reset phase to 0 (phasor = 1 + j0).
    #[inline]
    pub fn reset_phase(&mut self) {
        self.z = C32::new(1.0, 0.0);
        self.renorm_ctr = 0;
    }

    /// Set frequency; recomputes step phasor.
    #[inline]
    pub fn set_freq(&mut self, freq_hz: f32, fs: f32) {
        let phi = core::f32::consts::TAU * freq_hz / fs;
        let (s, c) = phi.sin_cos();
        self.w = C32::new(c, s);
    }

    /// Advance and return next phasor (cos, sin) as complex.
    #[inline(always)]
    pub fn next(&mut self) -> C32 {
        // Complex multiply with FMAs:
        let zr = self.z.re.mul_add(self.w.re, -self.z.im * self.w.im);
        let zi = self.z.im.mul_add(self.w.re,  self.z.re * self.w.im);
        self.z = C32::new(zr, zi);

        // Periodic renormalization to keep |z| ~ 1.0
        self.renorm_ctr = self.renorm_ctr.wrapping_add(1);
        if (self.renorm_ctr & 0x3FF) == 0 { // every 1024 steps
            let r2 = self.z.re * self.z.re + self.z.im * self.z.im;
            // One-step Newton for 1/sqrt(r2) is overkill for f32; do the simple thing:
            let inv = r2.sqrt().recip();
            self.z.re *= inv;
            self.z.im *= inv;
        }
        self.z
    }

    /// Convenience: advance and return (cos, sin).
    #[inline]
    pub fn next_cs(&mut self) -> (f32, f32) {
        let p = self.next();
        (p.re, p.im)
    }

    // -------- Optional high-level helpers (handy in hot loops) --------

    /// Mix (rotate) a complex block by current NCO; writes into `out`.
    #[inline]
    pub fn rotate_block(&mut self, input: &[C32], out: &mut [C32]) {
        let n = input.len().min(out.len());
        for i in 0..n {
            let p = self.next();
            // (a+jb)*(c+jd) = (ac - bd) + j(ad + bc)
            let a = input[i].re; let b = input[i].im;
            out[i].re = a.mul_add(p.re, -b * p.im);
            out[i].im = b.mul_add(p.re,  a * p.im);
        }
    }

    /// USB product detector primitive: y = I*cos + Q*sin
    #[inline]
    pub fn mix_usb_block(&mut self, input: &[C32], out: &mut [f32]) {
        let n = input.len().min(out.len());
        for i in 0..n {
            let p = self.next();
            out[i] = input[i].re.mul_add(p.re, input[i].im * p.im);
        }
    }
}
