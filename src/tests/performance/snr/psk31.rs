// SNR sensitivity sweep for BPSK31 and QPSK31.
//
// Feature-gated (`--features throughput`).  Tests always pass — they are
// measurement / characterisation runs, not assertions.  Run with --nocapture
// to see the printed table.
//
// SNR convention: signal power relative to noise in a 2500 Hz reference
// bandwidth (same as FT8/FT4).  See roundtrip/psk31_snr.rs for the full
// derivation.
//
// Usage:
//   cargo test --lib --features throughput "performance::snr" -- --nocapture --test-threads=1
//
// Each SNR point runs TRIALS independent trials with different AWGN seeds.
// The table printed at the end shows:
//   SNR(dB) | trials | decoded | success%
//
// Pipeline: modulate_text → add_awgn → psk31_sync (carrier detection) →
//   demod signal portion from sample 0 → Varicode decode → text search.
//
// Note: Both modes use peak sampling (not integrate-and-dump), so the demod
// sees the full noise bandwidth.  Combined with differential detection this
// means sensitivity thresholds are higher than in-band SNR alone suggests.
// See roundtrip/psk31_snr.rs for details.

use num_complex::Complex32 as C32;
use crate::modulate::psk31::{Bpsk31Mod, Qpsk31Mod, PSK31_BAUD};
use crate::demodulate::psk31::{Bpsk31Demod, Bpsk31Decider, Qpsk31Demod, Qpsk31Decider};
use crate::sync::psk31_sync::psk31_sync;
use crate::codec::varicode::VaricodeDecoder;
use crate::core::Block;
use crate::tests::roundtrip::add_awgn;

const FS: f32 = 8_000.0;
const REF_BW_HZ: f32 = 2_500.0;
const SIG_POWER: f32 = 1.0;
const TRIALS: usize = 50;
const TEXT: &[u8] = b"CQ TEST";

fn snr_to_noise_power(snr_db: f32) -> f32 {
    SIG_POWER * FS / (REF_BW_HZ * 10.0_f32.powf(snr_db / 10.0))
}

// ── helpers ───────────────────────────────────────────────────────────────────

/// Run one BPSK31 decode trial. Returns true on success.
fn try_bpsk31(noise_power: f32, seed: u64) -> bool {
    let base_hz = 900.0_f32;
    let carrier_hz = base_hz + 3.0 * PSK31_BAUD; // 993.75 Hz, bin 3

    let sig_iq = Bpsk31Mod::new(FS, carrier_hz, 1.0)
        .modulate_text(TEXT, 64, 32);
    let sig_len = sig_iq.len();
    let total = sig_len + FS as usize;
    let mut buf = sig_iq;
    buf.resize(total, C32::new(0.0, 0.0));
    add_awgn(&mut buf, noise_power, seed);

    let results = psk31_sync(&buf, FS, base_hz, base_hz + 200.0, 4, 3.0, 32, 5);
    if results.is_empty() { return false; }
    if (results[0].carrier_hz - carrier_hz).abs() > 40.0 { return false; }

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
    decoded.windows(TEXT.len()).any(|w| w == TEXT)
}

/// Run one QPSK31 decode trial. Returns true on success.
fn try_qpsk31(noise_power: f32, seed: u64) -> bool {
    let base_hz = 1400.0_f32;
    let carrier_hz = base_hz + 2.0 * PSK31_BAUD; // 1462.5 Hz, bin 2

    let sig_iq = Qpsk31Mod::new(FS, carrier_hz, 1.0)
        .modulate_text(TEXT, 64, 32);
    let sig_len = sig_iq.len();
    let total = sig_len + FS as usize;
    let mut buf = sig_iq;
    buf.resize(total, C32::new(0.0, 0.0));
    add_awgn(&mut buf, noise_power, seed);

    let results = psk31_sync(&buf, FS, base_hz, base_hz + 200.0, 4, 3.0, 32, 5);
    if results.is_empty() { return false; }
    let best = results.iter().min_by(|a, b| {
        let da = (a.carrier_hz - carrier_hz).abs();
        let db = (b.carrier_hz - carrier_hz).abs();
        da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
    }).unwrap();
    if (best.carrier_hz - carrier_hz).abs() > 2.0 * PSK31_BAUD { return false; }

    let sps = (FS / PSK31_BAUD).round() as usize;
    let mut demod = Qpsk31Demod::new(FS, best.carrier_hz, 1.0);
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
    decoded.windows(TEXT.len()).any(|w| w == TEXT)
}

// ── BPSK31 sweep ─────────────────────────────────────────────────────────────

#[test]
fn snr_sweep_bpsk31() {
    let snr_levels: &[f32] = &[
        0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
    ];

    println!("\n[BPSK31 SNR sweep, ref BW = {:.0} Hz, {} trials/point, text={:?}]",
             REF_BW_HZ, TRIALS, std::str::from_utf8(TEXT).unwrap_or("?"));
    println!("{:>10} {:>8} {:>8} {:>9}", "SNR(dB)", "trials", "decoded", "success%");
    println!("{}", "-".repeat(40));

    for &snr_db in snr_levels {
        let noise_power = snr_to_noise_power(snr_db);
        let decoded: usize = (0..TRIALS)
            .filter(|&i| {
                let seed = 0x1234_0000_0000_0000_u64
                    .wrapping_add(i as u64)
                    .wrapping_add((snr_db * 100.0) as u64);
                try_bpsk31(noise_power, seed)
            })
            .count();
        let pct = 100.0 * decoded as f32 / TRIALS as f32;
        println!("{:>10.1} {:>8} {:>8} {:>8.1}%", snr_db, TRIALS, decoded, pct);
    }
    println!();
    // Always passes — measurement run, not an assertion.
}

// ── QPSK31 sweep ─────────────────────────────────────────────────────────────

#[test]
fn snr_sweep_qpsk31() {
    let snr_levels: &[f32] = &[
        16.0, 18.0, 20.0, 22.0, 24.0, 26.0, 28.0, 30.0, 32.0, 34.0,
    ];

    println!("\n[QPSK31 SNR sweep, ref BW = {:.0} Hz, {} trials/point, text={:?}]",
             REF_BW_HZ, TRIALS, std::str::from_utf8(TEXT).unwrap_or("?"));
    println!("{:>10} {:>8} {:>8} {:>9}", "SNR(dB)", "trials", "decoded", "success%");
    println!("{}", "-".repeat(40));

    for &snr_db in snr_levels {
        let noise_power = snr_to_noise_power(snr_db);
        let decoded: usize = (0..TRIALS)
            .filter(|&i| {
                let seed = 0xABCD_0000_0000_0000_u64
                    .wrapping_add(i as u64)
                    .wrapping_add((snr_db * 100.0) as u64);
                try_qpsk31(noise_power, seed)
            })
            .count();
        let pct = 100.0 * decoded as f32 / TRIALS as f32;
        println!("{:>10.1} {:>8} {:>8} {:>8.1}%", snr_db, TRIALS, decoded, pct);
    }
    println!();
}
