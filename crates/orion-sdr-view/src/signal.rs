/// Simple test signal generator: sine tone + AWGN.
///
/// Uses xorshift64 for the PRNG and a 12-sample CLT sum for Gaussian noise,
/// matching the convention used in orion_sdr::util test helpers.
pub struct TestSignalGen {
    phase: f32,
    pub freq_hz: f32,
    pub sample_rate: f32,
    pub tone_amp: f32,
    pub noise_amp: f32,
    rng: u64,
}

impl TestSignalGen {
    pub fn new(freq_hz: f32, sample_rate: f32) -> Self {
        Self {
            phase: 0.0,
            freq_hz,
            sample_rate,
            tone_amp: 0.5,
            noise_amp: 0.05,
            rng: 0x853c_49e6_748f_ea9b,
        }
    }

    pub fn next_sample(&mut self) -> f32 {
        let tone = self.tone_amp * self.phase.sin();
        let noise = self.noise_amp * self.awgn();
        self.phase += 2.0 * std::f32::consts::PI * self.freq_hz / self.sample_rate;
        if self.phase > std::f32::consts::PI {
            self.phase -= 2.0 * std::f32::consts::PI;
        }
        tone + noise
    }

    /// Approximate Gaussian sample via 12-uniform CLT sum (zero mean, unit variance).
    fn awgn(&mut self) -> f32 {
        let mut sum = 0.0f32;
        for _ in 0..12 {
            sum += self.xorshift_f32();
        }
        sum - 6.0
    }

    fn xorshift_f32(&mut self) -> f32 {
        self.rng ^= self.rng << 13;
        self.rng ^= self.rng >> 7;
        self.rng ^= self.rng << 17;
        // Map to [0, 1)
        (self.rng >> 11) as f32 * (1.0 / (1u64 << 53) as f32)
    }
}
