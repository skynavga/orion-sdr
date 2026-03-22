use std::io::Cursor;
use std::path::Path;

use orion_sdr::core::AudioToIqChain;
use orion_sdr::modulate::AmDsbMod;

use crate::signal::TestSignalGen;

// ── SignalSource trait ────────────────────────────────────────────────────────

/// Common interface for all signal sources.
///
/// Implementations produce real-valued (f32) samples ready to push into the
/// existing `RingBuffer` and spectrum display pipeline.
///
/// `as_any_mut` enables downcasting a `Box<dyn SignalSource>` to a concrete type:
///   ```
///   if let Some(am) = source.as_any_mut().downcast_mut::<AmDsbSource>() { ... }
///   ```
pub trait SignalSource {
    fn next_samples(&mut self, n: usize) -> Vec<f32>;
    #[allow(dead_code)]
    fn sample_rate(&self) -> f32;
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any;
}

// ── TestToneSource ────────────────────────────────────────────────────────────

/// Adapts the existing `TestSignalGen` to the `SignalSource` trait.
/// All cycling/settings on the inner generator remain accessible via `.gen`.
pub struct TestToneSource {
    pub signal_gen: TestSignalGen,
}

impl TestToneSource {
    pub fn new(signal_gen: TestSignalGen) -> Self {
        Self { signal_gen }
    }
}

impl SignalSource for TestToneSource {
    fn next_samples(&mut self, n: usize) -> Vec<f32> {
        (0..n).map(|_| self.signal_gen.next_sample()).collect()
    }
    fn sample_rate(&self) -> f32 {
        self.signal_gen.sample_rate
    }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any { self }
}

// ── BuiltinAudio ─────────────────────────────────────────────────────────────

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum BuiltinAudio {
    Morse,
    Voice,
}

impl BuiltinAudio {
    pub const ALL: [BuiltinAudio; 2] = [BuiltinAudio::Morse, BuiltinAudio::Voice];
    #[allow(dead_code)]
    pub fn label(self) -> &'static str {
        match self {
            BuiltinAudio::Morse => "Morse",
            BuiltinAudio::Voice => "Voice",
        }
    }
}

static CQ_MORSE_WAV: &[u8] = include_bytes!("../assets/audio/cq_morse.wav");
static CQ_VOICE_WAV: &[u8] = include_bytes!("../assets/audio/cq_voice.wav");

fn decode_wav_bytes(bytes: &[u8]) -> (Vec<f32>, f32) {
    let cursor = Cursor::new(bytes);
    let mut reader = hound::WavReader::new(cursor).expect("decode built-in wav");
    let spec = reader.spec();
    let fs = spec.sample_rate as f32;
    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader
            .samples::<f32>()
            .map(|s| s.expect("read f32 sample"))
            .collect(),
        hound::SampleFormat::Int => {
            let max = (1i64 << (spec.bits_per_sample - 1)) as f32;
            reader
                .samples::<i32>()
                .map(|s| s.expect("read i32 sample") as f32 / max)
                .collect()
        }
    };
    (samples, fs)
}

pub fn load_builtin(kind: BuiltinAudio) -> (Vec<f32>, f32) {
    match kind {
        BuiltinAudio::Morse => decode_wav_bytes(CQ_MORSE_WAV),
        BuiltinAudio::Voice => decode_wav_bytes(CQ_VOICE_WAV),
    }
}

pub fn load_wav_file(path: &Path) -> Result<(Vec<f32>, f32), String> {
    let mut reader = hound::WavReader::open(path)
        .map_err(|e| format!("{e}"))?;
    let spec = reader.spec();
    let fs = spec.sample_rate as f32;
    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader
            .samples::<f32>()
            .map(|s| s.map_err(|e| format!("{e}")))
            .collect::<Result<Vec<_>, _>>()?,
        hound::SampleFormat::Int => {
            let max = (1i64 << (spec.bits_per_sample - 1)) as f32;
            reader
                .samples::<i32>()
                .map(|s| s.map_err(|e| format!("{e}")).map(|v| v as f32 / max))
                .collect::<Result<Vec<_>, _>>()?
        }
    };
    Ok((samples, fs))
}

// ── AmDsbSource ───────────────────────────────────────────────────────────────

/// AM DSB signal source driven by a looped audio buffer.
///
/// PTT model: while audio is playing the modulator runs with `carrier_level = 1.0`
/// (carrier always present when keyed). During the inter-loop gap `C32::ZERO` is
/// emitted directly — the modulator is bypassed — mirroring PTT release (no RF).
pub struct AmDsbSource {
    // Audio buffer and playback state
    audio: Vec<f32>,
    audio_rate: f32,
    /// Fractional read position into `audio` (in audio-rate samples).
    audio_pos: f32,
    /// How many output-rate samples remain in the current PTT gap.
    gap_remaining: usize,
    /// Gap length in output-rate samples (recomputed when loop_gap_secs changes).
    loop_gap_samples: usize,
    pub loop_gap_secs: f32,

    // Modulator
    chain: AudioToIqChain<AmDsbMod>,
    mod_rate: f32,

    // Exposed parameters (write → call rebuild_mod() to apply)
    pub carrier_hz: f32,
    pub mod_index: f32,
    pub noise_amp: f32,

    // PRNG for AWGN
    rng: u64,
}

impl AmDsbSource {
    pub fn new(
        audio: Vec<f32>,
        audio_rate: f32,
        carrier_hz: f32,
        mod_index: f32,
        loop_gap_secs: f32,
        noise_amp: f32,
        mod_rate: f32,
    ) -> Self {
        let loop_gap_samples = (loop_gap_secs * mod_rate) as usize;
        let block = AmDsbMod::new(mod_rate, carrier_hz, 1.0, mod_index);
        Self {
            audio,
            audio_rate,
            audio_pos: 0.0,
            gap_remaining: 0,
            loop_gap_samples,
            loop_gap_secs,
            chain: AudioToIqChain::new(block),
            mod_rate,
            carrier_hz,
            mod_index,
            noise_amp,
            rng: 0x853c_49e6_748f_ea9b,
        }
    }

    /// Replace audio buffer (e.g. after loading a user WAV file).
    pub fn set_audio(&mut self, audio: Vec<f32>, audio_rate: f32) {
        self.audio = audio;
        self.audio_rate = audio_rate;
        self.audio_pos = 0.0;
        self.gap_remaining = 0;
    }

    fn xorshift(&mut self) -> f32 {
        self.rng ^= self.rng << 13;
        self.rng ^= self.rng >> 7;
        self.rng ^= self.rng << 17;
        (self.rng >> 11) as f32 * (1.0 / (1u64 << 53) as f32) * 2.0 - 1.0
    }

    /// Rebuild the modulator after carrier_hz or mod_index changes.
    pub fn rebuild_mod(&mut self) {
        let block = AmDsbMod::new(self.mod_rate, self.carrier_hz, 1.0, self.mod_index);
        self.chain = AudioToIqChain::new(block);
    }

    /// Update loop gap length after loop_gap_secs changes.
    pub fn update_loop_gap(&mut self) {
        self.loop_gap_samples = (self.loop_gap_secs * self.mod_rate) as usize;
    }

    /// Interpolate one audio sample at the current fractional position,
    /// then advance the position. Returns (sample, wrapped) where `wrapped`
    /// is true when the position just looped back to zero.
    fn read_audio_sample(&mut self) -> (f32, bool) {
        let len = self.audio.len();
        if len == 0 {
            return (0.0, false);
        }
        let idx = self.audio_pos as usize;
        let frac = self.audio_pos - idx as f32;
        let s0 = self.audio[idx % len];
        let s1 = self.audio[(idx + 1) % len];
        let sample = s0 + frac * (s1 - s0);

        let ratio = self.audio_rate / self.mod_rate;
        self.audio_pos += ratio;

        let wrapped = self.audio_pos >= len as f32;
        if wrapped {
            self.audio_pos -= len as f32;
        }
        (sample, wrapped)
    }
}

impl SignalSource for AmDsbSource {
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any { self }

    fn next_samples(&mut self, n: usize) -> Vec<f32> {
        let mut out = Vec::with_capacity(n);
        let mut audio_chunk: Vec<f32> = Vec::with_capacity(n);
        let mut i = 0;

        while i < n {
            if self.gap_remaining > 0 {
                // PTT released — no carrier, but AWGN is always present
                let gap_now = self.gap_remaining.min(n - i);
                for _ in 0..gap_now {
                    let noise = if self.noise_amp > 0.0 {
                        self.noise_amp * self.xorshift()
                    } else {
                        0.0
                    };
                    out.push(noise);
                }
                self.gap_remaining -= gap_now;
                i += gap_now;
            } else {
                // PTT keyed — accumulate audio samples until gap or end of n
                audio_chunk.clear();
                while i < n && self.gap_remaining == 0 {
                    let (s, wrapped) = self.read_audio_sample();
                    audio_chunk.push(s);
                    i += 1;
                    if wrapped {
                        self.gap_remaining = self.loop_gap_samples;
                    }
                }
                // Modulate the keyed chunk; use real part to preserve carrier position
                if !audio_chunk.is_empty() {
                    let iq = self.chain.process_ref(&audio_chunk);
                    for c in iq {
                        let noise = if self.noise_amp > 0.0 {
                            self.noise_amp * self.xorshift()
                        } else {
                            0.0
                        };
                        out.push(c.re + noise);
                    }
                }
            }
        }

        out
    }

    fn sample_rate(&self) -> f32 {
        self.mod_rate
    }
}
