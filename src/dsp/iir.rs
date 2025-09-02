
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
    #[inline] pub fn process(&mut self, x: f32) -> f32 {
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
    #[inline] pub fn process(&mut self, mut x: f32) -> f32 {
        x = self.s[0].process(x);
        x = self.s[1].process(x);
        x
    }
}
