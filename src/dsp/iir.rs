
#[derive(Clone, Copy, Debug, Default)]
pub struct Biquad {
    b0: f32, b1: f32, b2: f32,
    a1: f32, a2: f32,
    z1: f32, z2: f32,
}

impl Biquad {
    #[inline] pub fn new(b0:f32,b1:f32,b2:f32,a1:f32,a2:f32)->Self{
        Self{b0,b1,b2,a1,a2,z1:0.0,z2:0.0}
    }
    #[inline] pub fn reset(&mut self){ self.z1=0.0; self.z2=0.0; }
    #[inline(always)] pub fn process(&mut self, x: f32) -> f32 {
        // Transposed Direct Form II, with FMAs
        let y = x.mul_add(self.b0, self.z1);
        self.z1 = x.mul_add(self.b1, self.z2) - self.a1 * y;
        self.z2 = x * self.b2 - self.a2 * y;
        y
    }
}

#[derive(Clone, Debug)]
pub struct LpCascade {
    s: [Biquad; 2], // 4th-order Linkwitz-Riley (two Butterworth biquads)
}

impl LpCascade {
    pub fn design(fs: f32, fc: f32) -> Self {
        // RBJ cookbook Butterworth biquad (Q = 1/√2), LR4 = two cascaded
        let w0 = core::f32::consts::TAU * fc / fs;
        let (sin, cos) = w0.sin_cos();
        let alpha = sin / (2.0 * (0.5_f32).sqrt()); // Q=√2/2

        let b0 = (1.0 - cos) * 0.5;
        let b1 =  1.0 - cos;
        let b2 = (1.0 - cos) * 0.5;
        let a0 =  1.0 + alpha;
        let a1 = -2.0 * cos;
        let a2 =  1.0 - alpha;

        let norm = 1.0 / a0;
        let b0n = b0 * norm; let b1n = b1 * norm; let b2n = b2 * norm;
        let a1n = a1 * norm; let a2n = a2 * norm;

        let stage = Biquad::new(b0n,b1n,b2n,a1n,a2n);
        Self { s: [stage, stage] }
    }

    #[inline] pub fn reset(&mut self) { self.s[0].reset(); self.s[1].reset(); }
    #[inline(always)] pub fn process(&mut self, mut x: f32) -> f32 {
        x = self.s[0].process(x);
        x = self.s[1].process(x);
        x
    }
}

/// Fused 4th-order Butterworth LP cascade + 1st-order DC blocker.
/// All state is held inline; process() runs as a single recurrence
/// with no intermediate function-call boundaries.
#[derive(Clone, Debug)]
pub struct LpDcCascade {
    // Biquad 0 (TDF-II state)
    z0_1: f32, z0_2: f32,
    // Biquad 1 (TDF-II state)
    z1_1: f32, z1_2: f32,
    // DC blocker state
    dc_x1: f32, dc_y1: f32,
    // Coefficients (both biquads share the same set)
    b0: f32, b1: f32, b2: f32,
    a1: f32, a2: f32,
    // DC blocker pole
    r: f32,
}

impl LpDcCascade {
    pub fn design(fs: f32, lp_fc: f32, dc_cut_hz: f32) -> Self {
        let w0 = core::f32::consts::TAU * lp_fc / fs;
        let (sin, cos) = w0.sin_cos();
        let alpha = sin / (2.0 * (0.5_f32).sqrt()); // Q=√2/2
        let b0_raw = (1.0 - cos) * 0.5;
        let b1_raw =  1.0 - cos;
        let b2_raw = (1.0 - cos) * 0.5;
        let a0 = 1.0 + alpha;
        let a1_raw = -2.0 * cos;
        let a2_raw =  1.0 - alpha;
        let norm = 1.0 / a0;
        let r = (1.0 - 2.0 * core::f32::consts::PI * (dc_cut_hz.max(0.1) / fs)).clamp(0.0, 0.9999);
        Self {
            z0_1: 0.0, z0_2: 0.0,
            z1_1: 0.0, z1_2: 0.0,
            dc_x1: 0.0, dc_y1: 0.0,
            b0: b0_raw * norm, b1: b1_raw * norm, b2: b2_raw * norm,
            a1: a1_raw * norm, a2: a2_raw * norm,
            r,
        }
    }

    #[inline] pub fn reset(&mut self) {
        self.z0_1 = 0.0; self.z0_2 = 0.0;
        self.z1_1 = 0.0; self.z1_2 = 0.0;
        self.dc_x1 = 0.0; self.dc_y1 = 0.0;
    }

    /// Process one sample through the full fused LP + DC chain.
    #[inline(always)]
    pub fn process(&mut self, x: f32) -> f32 {
        // Biquad 0 (TDF-II)
        let y0 = x.mul_add(self.b0, self.z0_1);
        self.z0_1 = x.mul_add(self.b1, self.z0_2) - self.a1 * y0;
        self.z0_2 = x * self.b2 - self.a2 * y0;
        // Biquad 1 (TDF-II)
        let y1 = y0.mul_add(self.b0, self.z1_1);
        self.z1_1 = y0.mul_add(self.b1, self.z1_2) - self.a1 * y1;
        self.z1_2 = y0 * self.b2 - self.a2 * y1;
        // DC blocker: y = x - x1 + r*y1
        let y = y1 - self.dc_x1 + self.r * self.dc_y1;
        self.dc_x1 = y1;
        self.dc_y1 = y;
        y
    }

    /// Process one sample through LP only, apply f, then through DC blocker.
    /// Used by AM-PowerSqrt: process_mapped(power, f32::sqrt)
    #[inline(always)]
    pub fn process_mapped(&mut self, x: f32, f: impl Fn(f32) -> f32) -> f32 {
        // Biquad 0
        let y0 = x.mul_add(self.b0, self.z0_1);
        self.z0_1 = x.mul_add(self.b1, self.z0_2) - self.a1 * y0;
        self.z0_2 = x * self.b2 - self.a2 * y0;
        // Biquad 1
        let y1 = y0.mul_add(self.b0, self.z1_1);
        self.z1_1 = y0.mul_add(self.b1, self.z1_2) - self.a1 * y1;
        self.z1_2 = y0 * self.b2 - self.a2 * y1;
        // Apply mapping (e.g. sqrt)
        let mapped = f(y1);
        // DC blocker
        let y = mapped - self.dc_x1 + self.r * self.dc_y1;
        self.dc_x1 = mapped;
        self.dc_y1 = y;
        y
    }
}
