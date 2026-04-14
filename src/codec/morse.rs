// Copyright (c) 2026 G & R Associates LLC
// SPDX-License-Identifier: MIT OR Apache-2.0

// src/codec/morse.rs
//
// ITU-R M.1677 International Morse Code encoder.
//
// Converts ASCII text into a keying envelope (0.0 / 1.0 sample buffer)
// suitable for feeding into `CwKeyedMod`.  Supports configurable WPM,
// element-duration jitter, dash weighting, and inter-element spacing
// to simulate human operator characteristics.

/// Standard Morse timing: 1 unit = 1200 / wpm milliseconds (PARIS standard).
fn unit_samples(sample_rate: f32, wpm: f32) -> f32 {
    let unit_ms = 1200.0 / wpm.max(1.0);
    (unit_ms * 1e-3) * sample_rate
}

// ── ITU Morse table ──────────────────────────────────────────────────────────

/// Morse patterns: each character maps to a string of '.' (dit) and '-' (dah).
/// Source: ITU-R M.1677-1 (10/2009).
const MORSE_TABLE: &[(char, &str)] = &[
    // Letters
    ('A', ".-"),
    ('B', "-..."),
    ('C', "-.-."),
    ('D', "-.."),
    ('E', "."),
    ('F', "..-."),
    ('G', "--."),
    ('H', "...."),
    ('I', ".."),
    ('J', ".---"),
    ('K', "-.-"),
    ('L', ".-.."),
    ('M', "--"),
    ('N', "-."),
    ('O', "---"),
    ('P', ".--."),
    ('Q', "--.-"),
    ('R', ".-."),
    ('S', "..."),
    ('T', "-"),
    ('U', "..-"),
    ('V', "...-"),
    ('W', ".--"),
    ('X', "-..-"),
    ('Y', "-.--"),
    ('Z', "--.."),
    // Digits
    ('0', "-----"),
    ('1', ".----"),
    ('2', "..---"),
    ('3', "...--"),
    ('4', "....-"),
    ('5', "....."),
    ('6', "-...."),
    ('7', "--..."),
    ('8', "---.."),
    ('9', "----."),
    // Punctuation
    ('.', ".-.-.-"),
    (',', "--..--"),
    ('?', "..--.."),
    ('\'', ".----."),
    ('!', "-.-.--"),
    ('/', "-..-."),
    ('(', "-.--."),
    (')', "-.--.-"),
    ('&', ".-..."),
    (':', "---..."),
    (';', "-.-.-."),
    ('=', "-...-"),
    ('+', ".-.-."),
    ('-', "-....-"),
    ('_', "..--.-"),
    ('"', ".-..-."),
    ('$', "...-..-"),
    ('@', ".--.-."),
];

/// Look up the Morse pattern for a character (case-insensitive).
fn char_to_morse(c: char) -> Option<&'static str> {
    let upper = c.to_ascii_uppercase();
    MORSE_TABLE
        .iter()
        .find(|(ch, _)| *ch == upper)
        .map(|(_, pat)| *pat)
}

// ── MorseEncoder ─────────────────────────────────────────────────────────────

/// Encodes ASCII text into a keying envelope (0.0 / 1.0 sample buffer).
///
/// # Timing model
///
/// Standard Morse (PARIS standard): 1 unit = 1200 / wpm ms.
///
/// | Element           | Duration (units) |
/// |-------------------|------------------|
/// | Dot               | 1                |
/// | Dash              | `dash_weight`    |
/// | Intra-char gap    | 1                |
/// | Inter-char gap    | `char_space`     |
/// | Word gap          | `word_space`     |
///
/// # Human-operator parameters
///
/// - `jitter_pct` — per-element duration jitter (±% of 1 unit)
/// - `dash_weight` — dash-to-dot ratio (3.0 = perfect ITU)
/// - `char_space` — inter-character gap in units (3.0 = ITU)
/// - `word_space` — word gap in units (7.0 = ITU)
pub struct MorseEncoder {
    sample_rate: f32,
    wpm: f32,
    jitter_pct: f32,
    dash_weight: f32,
    char_space: f32,
    word_space: f32,
    /// xorshift64 PRNG state (public for test seeding).
    pub rng: u64,
}

impl MorseEncoder {
    /// Create a new encoder with default (perfect) timing.
    pub fn new(sample_rate: f32, wpm: f32) -> Self {
        Self {
            sample_rate,
            wpm,
            jitter_pct: 0.0,
            dash_weight: 3.0,
            char_space: 3.0,
            word_space: 7.0,
            rng: 0x853c_49e6_748f_ea9b,
        }
    }

    /// Set element-duration jitter (±% of 1 unit).  Clamped to 0–30.
    pub fn with_jitter(mut self, pct: f32) -> Self {
        self.jitter_pct = pct.clamp(0.0, 30.0);
        self
    }

    /// Set dash-to-dot ratio.  Clamped to 2.5–3.5.
    pub fn with_dash_weight(mut self, w: f32) -> Self {
        self.dash_weight = w.clamp(2.5, 3.5);
        self
    }

    /// Set inter-character gap in units.  Clamped to 2.5–4.0.
    pub fn with_char_space(mut self, s: f32) -> Self {
        self.char_space = s.clamp(2.5, 4.0);
        self
    }

    /// Set word gap in units.  Clamped to 6.0–9.0.
    pub fn with_word_space(mut self, s: f32) -> Self {
        self.word_space = s.clamp(6.0, 9.0);
        self
    }

    /// Encode ASCII text into a keying envelope (`0.0` = key up, `1.0` = key down).
    ///
    /// Unknown characters are silently skipped.  Whitespace is treated as
    /// a word boundary (word gap emitted).  Multiple consecutive spaces
    /// produce a single word gap.
    pub fn encode_text(&mut self, text: &str) -> Vec<f32> {
        let unit = unit_samples(self.sample_rate, self.wpm);
        let mut out = Vec::new();
        // Pending gap to emit before the next character.  Deferred so that
        // a word boundary can upgrade a char gap to a word gap.
        let mut pending_gap: Option<f32> = None;

        for c in text.chars() {
            if c.is_ascii_whitespace() {
                // Upgrade any pending char gap to a word gap, or start one.
                if pending_gap.is_some() || !out.is_empty() {
                    pending_gap = Some(self.word_space);
                }
                continue;
            }

            let pattern = match char_to_morse(c) {
                Some(p) => p,
                None => continue, // skip unknown
            };

            // Emit the pending gap (char or word) before this character.
            if let Some(gap_units) = pending_gap.take() {
                let n = self.jittered_samples(unit, gap_units);
                out.extend(std::iter::repeat_n(0.0f32, n));
            }

            for (i, elem) in pattern.chars().enumerate() {
                // Intra-character gap (between elements within a character).
                if i > 0 {
                    let n = self.jittered_samples(unit, 1.0);
                    out.extend(std::iter::repeat_n(0.0f32, n));
                }

                let units = match elem {
                    '.' => 1.0,
                    '-' => self.dash_weight,
                    _ => continue,
                };
                let n = self.jittered_samples(unit, units);
                out.extend(std::iter::repeat_n(1.0f32, n));
            }

            // Queue a char gap for the next character.
            pending_gap = Some(self.char_space);
        }

        out
    }

    /// Compute jittered sample count for a given number of units.
    fn jittered_samples(&mut self, unit: f32, units: f32) -> usize {
        let nominal = unit * units;
        if self.jitter_pct <= 0.0 {
            return nominal.round() as usize;
        }
        // Uniform jitter in ±jitter_pct% of one unit.
        let max_delta = unit * (self.jitter_pct / 100.0);
        let r = self.xorshift_uniform(); // in [-1, 1]
        let jittered = nominal + r * max_delta;
        jittered.round().max(1.0) as usize
    }

    /// xorshift64 → uniform float in [-1, 1].
    fn xorshift_uniform(&mut self) -> f32 {
        self.rng ^= self.rng << 13;
        self.rng ^= self.rng >> 7;
        self.rng ^= self.rng << 17;
        (self.rng >> 11) as f32 * (1.0 / (1u64 << 53) as f32) * 2.0 - 1.0
    }
}
