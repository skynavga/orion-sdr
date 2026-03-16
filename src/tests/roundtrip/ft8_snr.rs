// SNR sensitivity regression tests for FT8 and FT4.
//
// These are fixed-threshold CI tests: each asserts that the full pipeline
// (codec encode → mod → add_awgn → ft8_sync → decode_soft) succeeds at a
// specific SNR level.  SNR is expressed in the WSJT-X convention: signal
// power relative to noise in a 2500 Hz reference bandwidth.
//
// SNR definition
// ──────────────
// P_sig ≈ 0.5 for a unit-amplitude complex CPFSK waveform.
// noise_power is the total per-sample noise variance across the full Nyquist
// bandwidth (fs/2).  Noise in the 2500 Hz reference BW:
//
//   P_noise_ref = noise_power × (2500 / fs)
//
//   SNR_2500 = 10·log10(P_sig / P_noise_ref)
//            = 10·log10(P_sig · fs / (noise_power · 2500))
//
// Inverting:
//   noise_power = P_sig · fs / (2500 · 10^(SNR_2500/10))
//               = 0.5 × 12000 / (2500 × 10^(SNR/10))
//               = 2.4 / 10^(SNR/10)
//
// Measured sensitivity (50-trial Monte Carlo, debug build, see performance/snr/ft8.rs)
// ──────────────────────────────────────────────────────────────────────────────
//   FT8: 50% decode at ≈−19 dB, 100% at −15 dB SNR/2500 Hz
//   FT4: 50% decode at ≈−13 dB, 100% at −11 dB SNR/2500 Hz
//
// These are ~6 dB above the WSJT-X published limits (−21 dB for FT8, −17 dB
// for FT4).  The gap is expected: WSJT-X runs many frames averaged over a
// 15-second window with Doppler tracking and iterative decoding, whereas this
// decoder processes a single frame with no iterative refinement.
//
// CI thresholds are anchored at the observed 100% success level so they pass
// reliably on every platform without being noise-sensitive.

use crate::sync::{ft8_sync, ft4_sync};
use crate::codec::ft8::Ft8Codec;
use crate::codec::ft4::Ft4Codec;
use super::{make_ft8_test_buffer, make_ft4_test_buffer};

const FS: f32 = 12_000.0;
const REF_BW_HZ: f32 = 2_500.0;
const SIG_POWER: f32 = 0.5;

/// Convert an SNR target (dB, 2500 Hz reference BW) to noise_power_per_sample.
pub fn snr_to_noise_power(snr_db: f32) -> f32 {
    SIG_POWER * FS / (REF_BW_HZ * 10.0_f32.powf(snr_db / 10.0))
}

// ── FT8 ──────────────────────────────────────────────────────────────────────

/// FT8 must decode at −15 dB SNR/2500 Hz (measured 100% success level).
#[test]
fn ft8_decodes_at_minus_15db_snr_2500hz() {
    let base_hz = 1_000.0_f32;
    let noise_power = snr_to_noise_power(-15.0);
    let (buf, payload) = make_ft8_test_buffer(0, base_hz, noise_power);

    let results = ft8_sync(
        &buf, FS,
        base_hz - 6.25, base_hz + 50.0 + 6.25,
        0, 0, 5,
    );

    assert!(
        !results.is_empty(),
        "FT8 sync returned no candidates at -15 dB SNR/2500 Hz \
         (noise_power={:.5})",
        noise_power
    );
    let decoded = Ft8Codec::decode_soft(&results[0].llr);
    assert!(
        decoded == Some(payload),
        "FT8 decode_soft failed at -15 dB SNR/2500 Hz \
         (noise_power={:.5}, got={:?})",
        noise_power, decoded
    );
}

// ── FT4 ──────────────────────────────────────────────────────────────────────

/// FT4 must decode at −11 dB SNR/2500 Hz (measured 100% success level).
#[test]
fn ft4_decodes_at_minus_11db_snr_2500hz() {
    let base_hz = 1_000.0_f32;
    let noise_power = snr_to_noise_power(-11.0);
    let (buf, payload) = make_ft4_test_buffer(0, base_hz, noise_power);

    let results = ft4_sync(
        &buf, FS,
        base_hz - 20.833, base_hz + 100.0 + 20.833,
        0, 0, 5,
    );

    assert!(
        !results.is_empty(),
        "FT4 sync returned no candidates at -11 dB SNR/2500 Hz \
         (noise_power={:.5})",
        noise_power
    );
    let decoded = Ft4Codec::decode_soft(&results[0].llr);
    assert!(
        decoded == Some(payload),
        "FT4 decode_soft failed at -11 dB SNR/2500 Hz \
         (noise_power={:.5}, got={:?})",
        noise_power, decoded
    );
}
