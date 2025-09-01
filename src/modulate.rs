use crate::core::*;
use crate::dsp::*;
use num_complex::Complex32 as C32;

// Helper to multiply complex sample by e^{jθ} from an NCO
#[inline]
pub fn mix_with_nco(x: C32, nco: &mut Nco) -> C32 {
    let (c, s) = nco.next_cs();
    C32::new(x.re * c - x.im * s, x.re * s + x.im * c)
}

/// ---------------------------------------------------------------------------
/// CW (keyed carrier) – envelope-shaped keyed NCO
pub struct CwKeyedMod {
    nco: Nco,
    env: f32,
    alpha_rise: f32,
    alpha_fall: f32,
    gain: f32,
}

impl CwKeyedMod {
    /// `tone_hz`: CW tone frequency (baseband or RF, depending on usage)
    /// `rise_ms`/`fall_ms`: envelope time constants (to avoid key clicks)
    pub fn new(sample_rate: f32, tone_hz: f32, rise_ms: f32, fall_ms: f32) -> Self {
        let tau_r = (rise_ms.max(0.1) * 1e-3) * sample_rate;
        let tau_f = (fall_ms.max(0.1) * 1e-3) * sample_rate;
        let alpha_r = (-1.0 / tau_r).exp();
        let alpha_f = (-1.0 / tau_f).exp();
        Self {
            nco: Nco::new(tone_hz, sample_rate),
            env: 0.0,
            alpha_rise: alpha_r,
            alpha_fall: alpha_f,
            gain: 1.0,
        }
    }
    pub fn set_gain(&mut self, g: f32) { self.gain = g; }
}

impl Block for CwKeyedMod {
    type In = f32;   // keying envelope 0..1 (you can derive this from audio or key events)
    type Out = C32;  // IQ

    fn process(&mut self, input: &[f32], output: &mut [C32]) -> WorkReport {
        let n = input.len().min(output.len());
        for i in 0..n {
            let tgt = input[i].clamp(0.0, 1.0);
            // asymmetric slew for rise/fall shaping
            self.env = if tgt >= self.env {
                self.alpha_rise * self.env + (1.0 - self.alpha_rise) * tgt
            } else {
                self.alpha_fall * self.env + (1.0 - self.alpha_fall) * tgt
            };
            let base = C32::new(self.env * self.gain, 0.0);
            output[i] = mix_with_nco(base, &mut self.nco);
        }
        WorkReport { in_read: n, out_written: n }
    }
}

/// ---------------------------------------------------------------------------
/// AM (DSB) modulator (baseband or RF-shift via NCO).
/// Output is complex IQ: s[n] = (carrier_level + m * x[n]) * e^{jφ[n]}
/// - `carrier_level` = 0.0 → DSB-SC; >0 adds a carrier (conventional AM)
/// - `modulation_index` m: recommended 0.0..1.0 to avoid overmodulation
pub struct AmDsbMod {
    nco: Nco,
    modulation_index: f32,
    carrier_level: f32,
    clamp: bool,
    gain: f32,
}

impl AmDsbMod {
    /// `carrier_hz` may be 0.0 for complex baseband.
    pub fn new(sample_rate: f32, carrier_hz: f32, modulation_index: f32, carrier_level: f32) -> Self {
        Self {
            nco: Nco::new(carrier_hz, sample_rate),
            modulation_index,
            carrier_level,
            clamp: true,
            gain: 1.0,
        }
    }
    pub fn set_gain(&mut self, g: f32) { self.gain = g; }
    pub fn set_modulation_index(&mut self, m: f32) { self.modulation_index = m; }
    pub fn set_carrier_level(&mut self, c: f32) { self.carrier_level = c; }
    pub fn set_limiter(&mut self, on: bool) { self.clamp = on; }
}

impl Block for AmDsbMod {
    type In  = f32;  // mono audio (−1..+1 recommended)
    type Out = C32;  // complex IQ

    fn process(&mut self, input: &[Self::In], output: &mut [Self::Out]) -> WorkReport {
        let n = input.len().min(output.len());
        for i in 0..n {
            let m = self.carrier_level + self.modulation_index * input[i];
            let a = if self.clamp { m.clamp(-1.0, 1.0) } else { m };
            // Treat amplitude as real part, then frequency shift with NCO
            let base = C32::new(a * self.gain, 0.0);
            output[i] = mix_with_nco(base, &mut self.nco);
        }
        WorkReport { in_read: n, out_written: n }
    }
}

/// ---------------------------------------------------------------------------
/// SSB (Weaver/phasing method) – audio → analytic (via sin/cos mix + LPF), optional RF upconvert.
/// Produces USB by default; set `sideband = -1.0` for LSB (flips Q sign).
pub struct SsbPhasingMod {
    fs: f32,
    // Audio IF for Weaver mixing (typically ~1–2 kHz); controls tone placement.
    audio_if_hz: f32,
    // Lowpass to constrain audio BW around 0 after the Weaver mixers
    pre_lp: FirLowpass,
    // Optional upconversion to RF (0 Hz => stay at complex baseband)
    rf_nco: Nco,
    // +1.0 => USB, -1.0 => LSB
    sideband: f32,
    gain: f32,
    // scratch buffers (audio domain)
    i_buf: Vec<f32>,
    q_buf: Vec<f32>,
    tmp: Vec<f32>,
    cos_nco: Nco,
    sin_nco: Nco,
}

impl SsbPhasingMod {
    /// `audio_bw_hz`: desired SSB audio bandwidth
    /// `audio_if_hz`: Weaver mixing frequency inside audio band (e.g., 1500.0)
    /// `rf_hz`: final RF translation (0 for baseband IQ)
    /// `usb`: true for USB, false for LSB
    pub fn new(sample_rate: f32, audio_bw_hz: f32, audio_if_hz: f32, rf_hz: f32, usb: bool) -> Self {
        // create pre-emphasis lowpass for audio before phasing
        let trans_hz = (audio_bw_hz * 0.15).clamp(200.0, 600.0);
        let pre_lp = FirLowpass::design(sample_rate, audio_bw_hz, trans_hz);
        let side = if usb { 1.0 } else { -1.0 };
        Self {
            fs: sample_rate,
            audio_if_hz,
            pre_lp,
            rf_nco: Nco::new(rf_hz, sample_rate),
            sideband: side,
            gain: 1.0,
            i_buf: Vec::new(),
            q_buf: Vec::new(),
            tmp: Vec::new(),
            cos_nco: Nco::new(audio_if_hz, sample_rate),
            sin_nco: Nco::new(audio_if_hz, sample_rate),
        }
    }
    pub fn set_gain(&mut self, g: f32) { self.gain = g; }
    pub fn set_audio_if(&mut self, f: f32) {
        self.audio_if_hz = f;
        self.cos_nco = Nco::new(f, self.fs);
        self.sin_nco = Nco::new(f, self.fs);
    }
    pub fn set_usb(&mut self, usb: bool) { self.sideband = if usb { 1.0 } else { -1.0 }; }
}

impl Block for SsbPhasingMod {
    type In = f32;   // audio
    type Out = C32;  // IQ

    fn process(&mut self, input: &[f32], output: &mut [C32]) -> WorkReport {
        let n = input.len().min(output.len());
        if self.i_buf.len() < n { self.i_buf.resize(n, 0.0); }
        if self.q_buf.len() < n { self.q_buf.resize(n, 0.0); }
        let (i_buf, q_buf) = (&mut self.i_buf[..n], &mut self.q_buf[..n]);

        // Weaver: multiply audio by cos and sin at audio_if_hz, then lowpass both
        for k in 0..n {
            let (c, s) = self.cos_nco.next_cs(); // returns (cos, sin) of running phase
            let x = input[k];
            i_buf[k] = x * c;
            q_buf[k] = x * s;
        }
        if self.tmp.len() != n { self.tmp.resize(n, 0.0); }
        self.tmp.copy_from_slice(&i_buf[..n]);
        self.pre_lp.process_block(&self.tmp[..], &mut i_buf[..n]);
        self.tmp.copy_from_slice(&q_buf[..n]);
        self.pre_lp.process_block(&self.tmp[..], &mut q_buf[..n]);

        // Build analytic audio: I = LP(x·cos), Q = ± LP(x·sin)
        for k in 0..n {
            let z = C32::new(i_buf[k], self.sideband * q_buf[k]) * self.gain;
            // Optional RF translation
            output[k] = mix_with_nco(z, &mut self.rf_nco);
        }

        WorkReport { in_read: n, out_written: n }
    }
}

/// ---------------------------------------------------------------------------
/// FM (direct) – phase accumulator with deviation scaling (Hz per unit input).
pub struct FmPhaseAccumMod {
    fs: f32,
    kf_hz_per_unit: f32, // peak deviation per |x|=1
    phase: f32,          // radians
    rf_nco: Nco,         // optional RF translation (0 Hz for baseband)
    gain: f32,           // overall output gain (post-carrier)
}

impl FmPhaseAccumMod {
    pub fn new(sample_rate: f32, deviation_hz: f32, rf_hz: f32) -> Self {
        Self {
            fs: sample_rate,
            kf_hz_per_unit: deviation_hz,
            phase: 0.0,
            rf_nco: Nco::new(rf_hz, sample_rate),
            gain: 1.0,
        }
    }
    pub fn set_deviation(&mut self, deviation_hz: f32) { self.kf_hz_per_unit = deviation_hz; }
    pub fn set_gain(&mut self, g: f32) { self.gain = g; }
}

impl Block for FmPhaseAccumMod {
    type In = f32;
    type Out = C32;

    fn process(&mut self, input: &[f32], output: &mut [C32]) -> WorkReport {
        let n = input.len().min(output.len());
        let two_pi = std::f32::consts::TAU;
        for i in 0..n {
            // Δφ = 2π * kf * x / fs
            self.phase = (self.phase + two_pi * self.kf_hz_per_unit * input[i] / self.fs).rem_euclid(two_pi);
            let base = C32::new(self.phase.cos(), self.phase.sin()) * self.gain;
            output[i] = mix_with_nco(base, &mut self.rf_nco);
        }
        WorkReport { in_read: n, out_written: n }
    }
}

/// ---------------------------------------------------------------------------
/// PM (direct) – instantaneous phase φ = kp * x[n], optional RF upconversion.
pub struct PmDirectPhaseMod {
    kp_rad_per_unit: f32,
    rf_nco: Nco,
    gain: f32,
}

impl PmDirectPhaseMod {
    pub fn new(sample_rate: f32, kp_rad_per_unit: f32, rf_hz: f32) -> Self {
        Self {
            kp_rad_per_unit,
            rf_nco: Nco::new(rf_hz, sample_rate),
            gain: 1.0,
        }
    }
    pub fn set_gain(&mut self, g: f32) { self.gain = g; }
    pub fn set_sensitivity(&mut self, kp_rad_per_unit: f32) { self.kp_rad_per_unit = kp_rad_per_unit; }
}

impl Block for PmDirectPhaseMod {
    type In = f32;
    type Out = C32;

    fn process(&mut self, input: &[f32], output: &mut [C32]) -> WorkReport {
        let n = input.len().min(output.len());
        for i in 0..n {
            let phi = self.kp_rad_per_unit * input[i];
            let base = C32::new(phi.cos(), phi.sin()) * self.gain;
            output[i] = mix_with_nco(base, &mut self.rf_nco);
        }
        WorkReport { in_read: n, out_written: n }
    }
}
