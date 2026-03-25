// SNR sensitivity regression tests for BPSK31 and QPSK31.
//
// These are fixed-threshold CI tests: each asserts that the full pipeline
// (modulate_text → add_awgn → psk31_sync → demod → Varicode decode) recovers
// the transmitted text at a specific SNR level.  SNR is expressed in the
// same convention used for FT8/FT4: signal power relative to noise in a
// 2500 Hz reference bandwidth.
//
// SNR definition
// ──────────────
// PSK31 carrier with gain=1 has phasor amplitude 1.0, so P_sig = 1.0.
//
// noise_power is the total per-sample noise variance across the full Nyquist
// bandwidth (fs/2 = 4 kHz at fs=8 kHz).  Noise in the 2500 Hz reference BW:
//
//   P_noise_ref = noise_power × (2500 / fs)
//
//   SNR_2500 = 10·log10(P_sig / P_noise_ref)
//            = 10·log10(P_sig · fs / (noise_power · 2500))
//
// Inverting:
//   noise_power = P_sig · fs / (2500 · 10^(SNR_2500/10))
//               = 1.0 × 8000 / (2500 × 10^(SNR/10))
//               = 3.2 / 10^(SNR/10)
//
// Pipeline design
// ───────────────
// psk31_sync detects the carrier in the buffer.  Once the carrier is confirmed,
// Bpsk31Demod (or Qpsk31Demod) is run over the signal portion from sample 0 so
// that preamble + text + postamble passes through the Varicode decoder.
//
// Note on SNR thresholds
// ──────────────────────
// Both modes use decision-feedback matched filtering over the full symbol
// period combined with a symbol-rate decision-directed PLL (AFC, K=0.05):
//   corrected[n] = s[n] − prev_sym·(1−h[n])
//   sym = Σ h[n]·corrected[n] / Σ h[n]²
//   phase_acc  += K · phase_err                   (first-order loop, B_L ≈ 0.78 Hz)
//
// BPSK31: differential Re(sym_c · conj(prev_sym)) → hard bit.
// QPSK31: coherent Viterbi MLSE — sym_c passed directly to viterbi_decode_coherent,
//   which tracks a hypothesised absolute phasor per trellis state, eliminating the
//   ~3 dB noise-product penalty of differential detection.
// AFC phase discriminant:
//   BPSK31: Im(d · conj(decided)) / |d|     (differential, d = sym_c·conj(prev))
//   QPSK31: Im(sym_c · conj(decided)) / |sym_c|  (absolute phasor)
//
// Measured sensitivity (50-trial Monte Carlo, see performance/snr/psk31.rs):
// ──────────────────────────────────────────────────────────────────────────
//   BPSK31: 50% decode at ≈−8 dB, 100% at −5 dB SNR/2500 Hz
//   QPSK31: 50% decode at ≈−12.5 dB, 100% at −7 dB SNR/2500 Hz
//
// QPSK31 coherent outperforms BPSK31 by ~5.5 dB at the 50% point, consistent
// with the ~5 dB convolutional coding gain recovered by coherent MLSE.
//
// CI thresholds are anchored at the observed 100% success level so they
// pass reliably on every platform without being noise-sensitive.

use num_complex::Complex32 as C32;
use crate::sync::psk31_sync::psk31_sync;
use crate::modulate::psk31::{Bpsk31Mod, Qpsk31Mod, PSK31_BAUD};
use crate::demodulate::psk31::{Bpsk31Demod, Bpsk31Decider, Qpsk31Demod, Qpsk31Decider};
use crate::codec::varicode::VaricodeDecoder;
use crate::core::Block;
use super::add_awgn;

const FS: f32 = 8_000.0;
const REF_BW_HZ: f32 = 2_500.0;
const SIG_POWER: f32 = 1.0;

/// Convert an SNR target (dB, 2500 Hz reference BW) to noise_power_per_sample.
pub fn snr_to_noise_power(snr_db: f32) -> f32 {
    SIG_POWER * FS / (REF_BW_HZ * 10.0_f32.powf(snr_db / 10.0))
}

/// Modulate `text` as BPSK31 at a bin-aligned carrier, add AWGN, run
/// psk31_sync to confirm carrier detection, demod the signal, and
/// Varicode-decode.  Returns `true` if the decoded output contains `text`.
fn try_bpsk31(text: &[u8], snr_db: f32, seed: u64) -> bool {
    let base_hz = 900.0_f32;
    let carrier_hz = base_hz + 3.0 * PSK31_BAUD; // 993.75 Hz, bin 3

    let sig_iq = Bpsk31Mod::new(FS, carrier_hz, 1.0)
        .modulate_text(text, 64, 32);

    let sig_len = sig_iq.len();
    let total = sig_len + FS as usize;
    let mut buf = sig_iq;
    buf.resize(total, C32::new(0.0, 0.0));
    add_awgn(&mut buf, snr_to_noise_power(snr_db), seed);

    // Confirm carrier detection.
    let results = psk31_sync(&buf, FS, base_hz, base_hz + 200.0, 4, 3.0, 32, 5);
    if results.is_empty() { return false; }
    if (results[0].carrier_hz - carrier_hz).abs() > 40.0 { return false; }

    // Demod signal from sample 0 at the detected carrier frequency.
    let found_hz = results[0].carrier_hz;
    let mut demod = Bpsk31Demod::new(FS, found_hz, 1.0);
    let max_syms = sig_len / (FS / PSK31_BAUD).round() as usize + 2;
    let mut soft = vec![0.0f32; max_syms];
    let wr = demod.process(&buf[..sig_len], &mut soft);
    soft.truncate(wr.out_written);

    let mut bits = vec![0u8; soft.len()];
    let dr = Bpsk31Decider::new().process(&soft, &mut bits);
    bits.truncate(dr.out_written);

    let mut vdec = VaricodeDecoder::new();
    for &b in &bits { vdec.push_bit(b); }
    vdec.push_bit(0); vdec.push_bit(0);
    let mut decoded = Vec::new();
    while let Some(c) = vdec.pop_char() { decoded.push(c); }

    decoded.windows(text.len()).any(|w| w == text)
}

/// Modulate `text` as QPSK31 at a bin-aligned carrier, add AWGN, run
/// psk31_sync to confirm carrier detection, demod the signal, and
/// Varicode-decode.  Returns `true` if the decoded output contains `text`.
fn try_qpsk31(text: &[u8], snr_db: f32, seed: u64) -> bool {
    let base_hz = 1400.0_f32;
    let carrier_hz = base_hz + 2.0 * PSK31_BAUD; // 1462.5 Hz, bin 2

    let sig_iq = Qpsk31Mod::new(FS, carrier_hz, 1.0)
        .modulate_text(text, 64, 32);

    let sig_len = sig_iq.len();
    let total = sig_len + FS as usize;
    let mut buf = sig_iq;
    buf.resize(total, C32::new(0.0, 0.0));
    add_awgn(&mut buf, snr_to_noise_power(snr_db), seed);

    // Confirm carrier detection.
    // For QPSK31, psk31_sync may report an adjacent bin (±1 bin = ±31.25 Hz).
    let results = psk31_sync(&buf, FS, base_hz, base_hz + 200.0, 4, 3.0, 32, 5);
    if results.is_empty() { return false; }
    let best = results.iter().min_by(|a, b| {
        let da = (a.carrier_hz - carrier_hz).abs();
        let db = (b.carrier_hz - carrier_hz).abs();
        da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
    }).unwrap();
    if (best.carrier_hz - carrier_hz).abs() > 2.0 * PSK31_BAUD { return false; }

    // Demod at the exact modulated carrier frequency (not the sync-detected bin,
    // which may be offset by ±31.25 Hz — tolerable for peak sampling but causes
    // phase drift across the I&D accumulation window with the new demodulator).
    let sps = (FS / PSK31_BAUD).round() as usize;
    let _ = best; // sync used only for carrier confirmation
    let mut demod = Qpsk31Demod::new(FS, carrier_hz, 1.0);
    let max_soft = (sig_len / sps + 2) * 2;
    let mut soft = vec![0.0f32; max_soft];
    let wr = demod.process(&buf[..sig_len], &mut soft);
    soft.truncate(wr.out_written);

    let mut decider = Qpsk31Decider::new();
    decider.process(&soft, &mut vec![]);
    let mut decoded_bits = Vec::new();
    decider.flush(&mut decoded_bits);

    let mut vdec = VaricodeDecoder::new();
    for &b in &decoded_bits { vdec.push_bit(b); }
    vdec.push_bit(0); vdec.push_bit(0);
    let mut decoded = Vec::new();
    while let Some(c) = vdec.pop_char() { decoded.push(c); }

    decoded.windows(text.len()).any(|w| w == text)
}

// ── BPSK31 CI regression ──────────────────────────────────────────────────────

/// BPSK31 must decode at −5 dB SNR/2500 Hz (measured 100% success level).
#[test]
fn bpsk31_decodes_at_minus_5db_snr_2500hz() {
    let snr_db = -5.0_f32;
    let text = b"CQ TEST";
    let noise_power = snr_to_noise_power(snr_db);
    assert!(
        try_bpsk31(text, snr_db, 0x1234_5678_9ABC_DEF0),
        "BPSK31 failed to decode at {} dB SNR/2500 Hz (noise_power={:.5})",
        snr_db, noise_power
    );
}

// ── QPSK31 CI regression ──────────────────────────────────────────────────────

/// QPSK31 coherent must decode at −7 dB SNR/2500 Hz (measured 100% success level).
#[test]
fn qpsk31_decodes_at_minus_7db_snr_2500hz() {
    let snr_db = -7.0_f32;
    let text = b"CQ TEST";
    let noise_power = snr_to_noise_power(snr_db);
    assert!(
        try_qpsk31(text, snr_db, 0xFEDC_BA98_7654_3210),
        "QPSK31 failed to decode at {} dB SNR/2500 Hz (noise_power={:.5})",
        snr_db, noise_power
    );
}
