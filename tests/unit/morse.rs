// Copyright (c) 2026 G & R Associates LLC
// SPDX-License-Identifier: MIT OR Apache-2.0

use orion_sdr::codec::MorseEncoder;

const SR: f32 = 48_000.0;

/// At 20 WPM, 1 unit = 1200/20 = 60 ms = 2880 samples at 48 kHz.
const WPM_20: f32 = 20.0;
const UNIT_20: usize = 2880;

// ── Basic timing ─────────────────────────────────────────────────────────────

#[test]
fn single_dot_duration() {
    // 'E' = "." → 1 dot (1 unit).
    let samples = MorseEncoder::new(SR, WPM_20).encode_text("E");
    assert_eq!(samples.len(), UNIT_20);
    assert!(samples.iter().all(|&s| s == 1.0));
}

#[test]
fn single_dash_duration() {
    // 'T' = "-" → 1 dash (3 units, default dash_weight).
    let samples = MorseEncoder::new(SR, WPM_20).encode_text("T");
    assert_eq!(samples.len(), UNIT_20 * 3);
    assert!(samples.iter().all(|&s| s == 1.0));
}

#[test]
fn letter_a_structure() {
    // 'A' = ".-" → dot(1) + intra-gap(1) + dash(3) = 5 units.
    let samples = MorseEncoder::new(SR, WPM_20).encode_text("A");
    assert_eq!(samples.len(), UNIT_20 * 5);

    // First UNIT_20 samples should be 1.0 (dot).
    assert!(samples[..UNIT_20].iter().all(|&s| s == 1.0));
    // Next UNIT_20 samples should be 0.0 (intra-char gap).
    assert!(samples[UNIT_20..UNIT_20 * 2].iter().all(|&s| s == 0.0));
    // Last 3*UNIT_20 samples should be 1.0 (dash).
    assert!(samples[UNIT_20 * 2..].iter().all(|&s| s == 1.0));
}

// ── SOS timing ───────────────────────────────────────────────────────────────

#[test]
fn sos_envelope_length() {
    // "SOS" = "..." + char_gap + "---" + char_gap + "..."
    //
    // S = 3 dots + 2 intra gaps = 3 + 2 = 5 units
    // O = 3 dashes + 2 intra gaps = 9 + 2 = 11 units
    // char_gap = 3 units (×2, between S-O and O-S)
    // Total = 5 + 3 + 11 + 3 + 5 = 27 units
    let samples = MorseEncoder::new(SR, WPM_20).encode_text("SOS");
    assert_eq!(samples.len(), UNIT_20 * 27);
}

// ── Word spacing ─────────────────────────────────────────────────────────────

#[test]
fn word_gap() {
    // "E E" = dot + char_space(3) + word_extra(7-3=4) + dot
    // = 1 + 3 + 4 + 1 = 9 units
    let samples = MorseEncoder::new(SR, WPM_20).encode_text("E E");
    assert_eq!(samples.len(), UNIT_20 * 9);
}

#[test]
fn multiple_spaces_single_word_gap() {
    // "E   E" should produce the same output as "E E".
    let single = MorseEncoder::new(SR, WPM_20).encode_text("E E");
    let multi = MorseEncoder::new(SR, WPM_20).encode_text("E   E");
    assert_eq!(single.len(), multi.len());
}

// ── Dash weight ──────────────────────────────────────────────────────────────

#[test]
fn dash_weight_affects_length() {
    // 'T' = "-" → dash length = dash_weight * unit.
    let heavy = MorseEncoder::new(SR, WPM_20)
        .with_dash_weight(3.5)
        .encode_text("T");
    let expected = (UNIT_20 as f32 * 3.5).round() as usize;
    assert_eq!(heavy.len(), expected);

    let light = MorseEncoder::new(SR, WPM_20)
        .with_dash_weight(2.5)
        .encode_text("T");
    let expected = (UNIT_20 as f32 * 2.5).round() as usize;
    assert_eq!(light.len(), expected);
}

// ── Jitter bounds ────────────────────────────────────────────────────────────

#[test]
fn jitter_stays_within_bounds() {
    // Encode the same text many times with 10% jitter; total length should
    // stay within ±10% of nominal for each run.
    let nominal = MorseEncoder::new(SR, WPM_20).encode_text("SOS").len();
    let tolerance = (nominal as f32 * 0.10) as usize;

    for seed_offset in 0u64..20 {
        let mut enc = MorseEncoder::new(SR, WPM_20).with_jitter(10.0);
        // Perturb RNG slightly for each run so we test different sequences.
        enc.rng = enc.rng.wrapping_add(seed_offset * 1000);
        let len = enc.encode_text("SOS").len();
        assert!(
            len.abs_diff(nominal) <= tolerance,
            "jittered len {} outside ±10% of nominal {} (seed_offset={})",
            len,
            nominal,
            seed_offset,
        );
    }
}

// ── Char / word spacing parameters ───────────────────────────────────────────

#[test]
fn custom_char_space() {
    // "EE" with char_space=4.0: dot + 4-unit gap + dot = 6 units.
    let samples = MorseEncoder::new(SR, WPM_20)
        .with_char_space(4.0)
        .encode_text("EE");
    assert_eq!(samples.len(), UNIT_20 * 6);
}

#[test]
fn custom_word_space() {
    // "E E" with word_space=9.0: dot(1) + word_gap(9) + dot(1) = 11 units.
    let samples = MorseEncoder::new(SR, WPM_20)
        .with_word_space(9.0)
        .encode_text("E E");
    assert_eq!(samples.len(), UNIT_20 * 11);
}

// ── Unknown characters ───────────────────────────────────────────────────────

#[test]
fn unknown_chars_skipped() {
    // '{' and '}' are not in the Morse table; they should be silently skipped.
    let with_unknown = MorseEncoder::new(SR, WPM_20).encode_text("E{E");
    // Should be equivalent to "EE" (no inter-char gap for skipped chars — the
    // '{' is simply absent, so we get E + char_gap + E = 1 + 3 + 1 = 5 units).
    let plain = MorseEncoder::new(SR, WPM_20).encode_text("EE");
    assert_eq!(with_unknown.len(), plain.len());
}

#[test]
fn empty_text_produces_empty_envelope() {
    let samples = MorseEncoder::new(SR, WPM_20).encode_text("");
    assert!(samples.is_empty());
}

// ── Case insensitivity ───────────────────────────────────────────────────────

#[test]
fn case_insensitive() {
    let upper = MorseEncoder::new(SR, WPM_20).encode_text("SOS");
    let lower = MorseEncoder::new(SR, WPM_20).encode_text("sos");
    assert_eq!(upper.len(), lower.len());
    assert_eq!(upper, lower);
}

// ── WPM range ────────────────────────────────────────────────────────────────

#[test]
fn slow_wpm() {
    // At 3 WPM, 1 unit = 1200/3 = 400 ms = 19200 samples.
    let samples = MorseEncoder::new(SR, 3.0).encode_text("E");
    assert_eq!(samples.len(), 19200);
}

#[test]
fn fast_wpm() {
    // At 30 WPM, 1 unit = 1200/30 = 40 ms = 1920 samples.
    let samples = MorseEncoder::new(SR, 30.0).encode_text("E");
    assert_eq!(samples.len(), 1920);
}

// ── Envelope values ──────────────────────────────────────────────────────────

#[test]
fn envelope_only_zero_and_one() {
    let samples = MorseEncoder::new(SR, WPM_20).encode_text("CQ CQ DE N0GNR");
    for (i, &s) in samples.iter().enumerate() {
        assert!(
            s == 0.0 || s == 1.0,
            "sample[{}] = {} (expected 0.0 or 1.0)",
            i,
            s
        );
    }
}
