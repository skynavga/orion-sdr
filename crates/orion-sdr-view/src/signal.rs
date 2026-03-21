/// Amplitude cycling state machine.
///
/// Sequence: RampUp → PauseHigh → RampDown → PauseLow → RampUp → …
/// Each state counts down `samples_remaining` to zero then transitions.
#[derive(Clone, Copy)]
enum CycleState {
    RampUp,
    PauseHigh,
    RampDown,
    PauseLow,
}

/// Simple test signal generator: sine tone + AWGN.
///
/// Uses xorshift64 for the PRNG and a 12-sample CLT sum for Gaussian noise,
/// matching the convention used in orion_sdr::util test helpers.
///
/// When `cycling` is true the tone amplitude follows a 4-phase sequence:
/// ramp 0.0 → 0.65, pause, ramp 0.65 → 0.0, pause. Each ramp takes
/// `ramp_secs` seconds; each pause lasts `pause_secs` seconds.
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
    /// Duration of each ramp (up or down) in seconds.
    pub ramp_secs: f32,
    /// Duration of each pause (at top or bottom) in seconds.
    pub pause_secs: f32,
    cycle_state: CycleState,
    samples_remaining: u32,
}

impl TestSignalGen {
    pub fn new(freq_hz: f32, sample_rate: f32) -> Self {
        let ramp_secs = 3.0f32;
        let pause_secs = 7.0f32; // ~2.3× ramp duration
        let pause_samples = (pause_secs * sample_rate) as u32;
        Self {
            phase: 0.0,
            freq_hz,
            sample_rate,
            tone_amp: 0.65,              // start at maximum, visible immediately
            noise_amp: 0.05,
            rng: 0x853c_49e6_748f_ea9b,
            cycling: false,
            amp_min: 0.0,
            amp_max: 0.65,
            ramp_secs,
            pause_secs,
            cycle_state: CycleState::PauseHigh,  // FSM starts mid-sequence at peak
            samples_remaining: pause_samples,
        }
    }

    pub fn next_sample(&mut self) -> f32 {
        if self.cycling {
            self.advance_cycle();
        }

        let tone = self.tone_amp * self.phase.sin();
        let noise = self.noise_amp * self.awgn();
        self.phase += 2.0 * std::f32::consts::PI * self.freq_hz / self.sample_rate;
        if self.phase > std::f32::consts::PI {
            self.phase -= 2.0 * std::f32::consts::PI;
        }
        tone + noise
    }

    /// Begin cycling: start a ramp-down from the current amplitude peak.
    pub fn start_cycling(&mut self) {
        if self.cycling {
            return;
        }
        self.tone_amp = self.amp_max;
        self.cycle_state = CycleState::RampDown;
        self.samples_remaining = (self.ramp_secs * self.sample_rate) as u32;
        self.cycling = true;
    }

    /// Stop cycling: snap immediately to full amplitude.
    pub fn stop_cycling(&mut self) {
        if !self.cycling {
            return;
        }
        self.cycling = false;
        self.tone_amp = self.amp_max;
        // Reset FSM so next start_cycling begins with a ramp-down again.
        self.cycle_state = CycleState::PauseHigh;
        self.samples_remaining = (self.pause_secs * self.sample_rate) as u32;
    }

    fn advance_cycle(&mut self) {
        let ramp_samples = (self.ramp_secs * self.sample_rate) as u32;
        let pause_samples = (self.pause_secs * self.sample_rate) as u32;

        match self.cycle_state {
            CycleState::RampUp => {
                // Interpolate amp_min → amp_max over ramp_samples.
                let t = 1.0 - (self.samples_remaining as f32 / ramp_samples as f32);
                self.tone_amp = self.amp_min + t * (self.amp_max - self.amp_min);
                if self.samples_remaining == 0 {
                    self.tone_amp = self.amp_max;
                    self.cycle_state = CycleState::PauseHigh;
                    self.samples_remaining = pause_samples;
                } else {
                    self.samples_remaining -= 1;
                }
            }
            CycleState::PauseHigh => {
                self.tone_amp = self.amp_max;
                if self.samples_remaining == 0 {
                    self.cycle_state = CycleState::RampDown;
                    self.samples_remaining = ramp_samples;
                } else {
                    self.samples_remaining -= 1;
                }
            }
            CycleState::RampDown => {
                // Interpolate amp_max → amp_min over ramp_samples.
                let t = 1.0 - (self.samples_remaining as f32 / ramp_samples as f32);
                self.tone_amp = self.amp_max - t * (self.amp_max - self.amp_min);
                if self.samples_remaining == 0 {
                    self.tone_amp = self.amp_min;
                    self.cycle_state = CycleState::PauseLow;
                    self.samples_remaining = pause_samples;
                } else {
                    self.samples_remaining -= 1;
                }
            }
            CycleState::PauseLow => {
                self.tone_amp = self.amp_min;
                if self.samples_remaining == 0 {
                    self.cycle_state = CycleState::RampUp;
                    self.samples_remaining = ramp_samples;
                } else {
                    self.samples_remaining -= 1;
                }
            }
        }
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
