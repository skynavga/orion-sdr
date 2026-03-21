/// Simple test signal generator: sine tone + AWGN.
///
/// Uses xorshift64 for the PRNG and a 12-sample CLT sum for Gaussian noise,
/// matching the convention used in orion_sdr::util test helpers.
///
/// When `cycling` is true the tone amplitude ramps up and down between
/// `amp_min` and `amp_max` at `cycle_hz` cycles per second, making the
/// signal rise and fall visibly in the spectrum and persistence panes.
pub struct TestSignalGen {
    phase: f32,
    pub freq_hz: f32,
    pub sample_rate: f32,
    pub tone_amp: f32,
    pub noise_amp: f32,
    rng: u64,

    // Amplitude cycling
    pub cycling: bool,
    pub amp_min: f32,
    pub amp_max: f32,
    /// How many complete ramp cycles per second.
    pub cycle_hz: f32,
    /// Internal phase of the amplitude cycle [0, 1).
    cycle_phase: f32,
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
            cycling: false,
            amp_min: 0.02,
            amp_max: 0.8,
            cycle_hz: 0.2,   // one full ramp cycle every 5 seconds
            cycle_phase: 0.0,
        }
    }

    pub fn next_sample(&mut self) -> f32 {
        // Update amplitude cycling before generating the sample.
        if self.cycling {
            self.cycle_phase += self.cycle_hz / self.sample_rate;
            if self.cycle_phase >= 1.0 {
                self.cycle_phase -= 1.0;
            }
            // Triangle wave: ramps up for first half, down for second half.
            let t = if self.cycle_phase < 0.5 {
                self.cycle_phase * 2.0
            } else {
                (1.0 - self.cycle_phase) * 2.0
            };
            self.tone_amp = self.amp_min + t * (self.amp_max - self.amp_min);
        }

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
