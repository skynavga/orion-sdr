use num_complex::Complex32 as C32;
use crate::core::{Block, WorkReport};
use crate::dsp::{FirLowpass, Nco, mix_with_nco};

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
    // scratch buffers
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
        self.pre_lp.process(&self.tmp[..], &mut i_buf[..n]);
        self.tmp.copy_from_slice(&q_buf[..n]);
        self.pre_lp.process(&self.tmp[..], &mut q_buf[..n]);

        // Build analytic audio: I = LP(x·cos), Q = ± LP(x·sin)
        for k in 0..n {
            let z = C32::new(i_buf[k], self.sideband * q_buf[k]) * self.gain;
            // Optional RF translation
            output[k] = mix_with_nco(z, &mut self.rf_nco);
        }

        WorkReport { in_read: n, out_written: n }
    }
}
