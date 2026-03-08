use std::time::Instant;
use std::hint::black_box;
use num_complex::Complex32 as C32;

// --- small helpers ------------------------------------------------------

fn real_tone(fs: f32, f_hz: f32, n: usize, amp: f32) -> Vec<f32> {
    (0..n)
        .map(|k| amp * (std::f32::consts::TAU * f_hz * (k as f32) / fs).sin())
        .collect()
}

fn key_envelope_square(fs: f32, key_hz: f32, n: usize) -> Vec<f32> {
    (0..n)
        .map(|k| ((k as f32 * key_hz / fs).fract() < 0.5) as i32 as f32)
        .collect()
}

fn minsps_from_env(default_msps: f32) -> f32 {
    std::env::var("ORION_SDR_THROUGHPUT_MINSPS")
        .ok()
        .and_then(|s| s.parse::<f32>().ok())
        .unwrap_or(default_msps)
}

/// Run `repeats` passes, return (msamples/sec, elapsed_secs).
fn measure_throughput(mut f: impl FnMut() -> usize, samples_per_pass: usize, repeats: usize) -> (f32, f64) {
    let start = Instant::now();
    let mut sink = 0.0f64;
    for _ in 0..repeats {
        // black_box to avoid optimizing away work
        let n = f();
        sink = black_box(sink + (n as f64) * 1e-12);
    }
    let dt = start.elapsed().as_secs_f64();
    let total_samples = samples_per_pass as f64 * repeats as f64;
    let msps = (total_samples / dt) / 1.0e6;
    // prevent "unused" warning
    black_box(sink);
    (msps as f32, dt)
}

// --- CW -----------------------------------------------------------------

#[test]
fn throughput_cw_roundtrip() {
    use crate::core::{AudioToIqChain, IqToAudioChain};
    use crate::modulate::CwKeyedMod;
    use crate::demodulate::CwEnvelopeDemod;

    let fs = 48_000.0;
    let n = 65_536;
    let repeats = 30;

    let key = key_envelope_square(fs, 5.0, n);
    let mut tx = AudioToIqChain::new(CwKeyedMod::new(fs, 700.0, 3.0, 3.0));
    let mut rx = IqToAudioChain::new(CwEnvelopeDemod::new(fs, 700.0, 300.0));

    let (msps, dt) = measure_throughput(
        || {
            let iq: Vec<C32> = tx.process(key.clone());
            let _audio: Vec<f32> = rx.process(iq);
            _audio.len()
        },
        n,
        repeats,
    );

    println!("[CW] {:.2} Msps in {:.3}s", msps, dt);
    let min_msps = minsps_from_env(0.2);
    assert!(msps >= min_msps, "CW throughput {:.2} Msps < min {:.2} Msps", msps, min_msps);
}

// --- AM-PowerSqrt -------------------------------------------------------

#[test]
fn throughput_am_powersqrt_roundtrip() {
    use crate::core::{AudioToIqChain, IqToAudioChain};
    use crate::modulate::AmDsbMod;
    use crate::demodulate::AmEnvelopeDemod;

    let fs = 48_000.0;
    let n = 65_536;
    let repeats = 30;

    let audio = real_tone(fs, 1000.0, n, 0.5);
    let mut tx = AudioToIqChain::new(AmDsbMod::new(fs, 0.0, 0.8, 0.5));
    let mut rx = IqToAudioChain::new(AmEnvelopeDemod::new(fs, 5_000.0));

    let (msps, dt) = measure_throughput(
        || {
            let iq = tx.process(audio.clone());
            let out = rx.process(iq);
            out.len()
        },
        n,
        repeats,
    );

    println!("[AM](PowerSqrt) {:.2} Msps in {:.3}s", msps, dt);
    let min_msps = minsps_from_env(0.2);
    assert!(msps >= min_msps, "AM-PowerSqrt throughput {:.2} Msps < min {:.2} Msps", msps, min_msps);
}

// --- AM-AbsApprox -------------------------------------------------------

#[test]
fn throughput_am_absapprox_roundtrip() {
    use crate::core::{AudioToIqChain, IqToAudioChain};
    use crate::modulate::AmDsbMod;
    use crate::demodulate::AmEnvelopeDemod;

    let fs = 48_000.0;
    let n = 65_536;
    let repeats = 30;

    // Same signal shape & size as the PowerSqrt test for apples-to-apples comparison
    let audio = real_tone(fs, 1000.0, n, 0.5);

    // Keep the same modulator config you use in the PowerSqrt test
    let mut tx = AudioToIqChain::new(AmDsbMod::new(fs, 0.0, 0.8, 0.5));

    // AbsApprox envelope
    let mut rx = IqToAudioChain::new(
        AmEnvelopeDemod::new(fs, 5_000.0).with_abs_approx(0.9475, 0.3925)
    );

    let (msps, dt) = measure_throughput(
        || {
            // Clone the tone each pass for fairness (matches existing style)
            let iq = tx.process(audio.clone());
            let out = rx.process(iq);
            out.len()
        },
        n,
        repeats,
    );

    println!("[AM](AbsApprox) {:.2} Msps in {:.3}s", msps, dt);
    let min_msps = minsps_from_env(0.2);
    assert!(msps >= min_msps,"AM-AbsApprox throughput {:.2} Msps < min {:.2} Msps", msps, min_msps);
}

// --- SSB (USB) ----------------------------------------------------------

#[test]
fn throughput_ssb_usb_roundtrip() {
    use crate::core::{AudioToIqChain, IqToAudioChain};
    use crate::modulate::SsbPhasingMod;
    use crate::demodulate::SsbProductDemod;

    let fs = 48_000.0;
    let n = 65_536;
    let repeats = 20; // SSB path has more math per sample

    let audio = real_tone(fs, 1200.0, n, 0.4);
    let mut tx = AudioToIqChain::new(SsbPhasingMod::new(fs, 2_800.0, 1_500.0, 0.0, true));
    let mut rx = IqToAudioChain::new(SsbProductDemod::new(fs, 0.0, 2_800.0));

    let (msps, dt) = measure_throughput(
        || {
            let iq = tx.process(audio.clone());
            let out = rx.process(iq);
            out.len()
        },
        n,
        repeats,
    );

    println!("[SSB-USB] {:.2} Msps in {:.3}s", msps, dt);
    let min_msps = minsps_from_env(0.15);
    assert!(msps >= min_msps, "SSB throughput {:.2} Msps < min {:.2} Msps", msps, min_msps);
}

// --- FM -----------------------------------------------------------------

#[test]
fn throughput_fm_roundtrip() {
    use crate::core::{AudioToIqChain, IqToAudioChain};
    use crate::modulate::FmPhaseAccumMod;
    use crate::demodulate::FmQuadratureDemod;

    let fs = 48_000.0;
    let n = 65_536;
    let repeats = 30;

    let audio = real_tone(fs, 1000.0, n, 0.5);
    let mut tx = AudioToIqChain::new(FmPhaseAccumMod::new(fs, 2_500.0, 0.0));
    let mut rx = IqToAudioChain::new(FmQuadratureDemod::new(fs, 2_500.0, 5_000.0));

    let (msps, dt) = measure_throughput(
        || {
            let iq = tx.process(audio.clone());
            let out = rx.process(iq);
            out.len()
        },
        n,
        repeats,
    );

    println!("[FM] {:.2} Msps in {:.3}s", msps, dt);
    let min_msps = minsps_from_env(0.25);
    assert!(msps >= min_msps, "FM throughput {:.2} Msps < min {:.2} Msps", msps, min_msps);
}

// --- QAM (mapper → mod → demod → decider, full symbol pipeline) ---------

fn throughput_qam_roundtrip<const BITS: usize>(label: &str) {
    use crate::core::{Block, IqToIqChain};
    use crate::modulate::{QamMapper, QamMod};
    use crate::demodulate::{QamDemod, QamDecider};

    let n_syms = 65_536;
    let n_bits = n_syms * BITS;
    let repeats = 30;

    let bits_in: Vec<u8> = (0..n_bits).map(|i| ((i / BITS + i % BITS) & 1) as u8).collect();
    let mut syms     = vec![C32::default(); n_syms];
    let mut soft     = vec![C32::default(); n_syms];
    let mut bits_out = vec![0u8; n_bits];

    let mut mapper   = QamMapper::<BITS>::new();
    let mut modstage = IqToIqChain::new(QamMod::new(1.0, 0.0, 1.0));
    let mut demod    = IqToIqChain::new(QamDemod::new(1.0));
    let mut decider  = QamDecider::<BITS>::new();

    let (msps, dt) = measure_throughput(
        || {
            mapper.process(&bits_in, &mut syms);
            let iq = modstage.process_ref(&syms);
            demod.process_into(&iq, &mut soft);
            decider.process(&soft, &mut bits_out);
            black_box(bits_out[0]);
            n_syms
        },
        n_syms,
        repeats,
    );

    println!("[{}] {:.2} Msps in {:.3}s", label, msps, dt);
    let min_msps = minsps_from_env(0.2);
    assert!(msps >= min_msps, "{} throughput {:.2} Msps < min {:.2} Msps", label, msps, min_msps);
}

#[test]
fn throughput_qam16_roundtrip()  { throughput_qam_roundtrip::<4>("QAM-16");  }

#[test]
fn throughput_qam64_roundtrip()  { throughput_qam_roundtrip::<6>("QAM-64");  }

#[test]
fn throughput_qam256_roundtrip() { throughput_qam_roundtrip::<8>("QAM-256"); }

// --- QPSK (mapper → mod → demod → decider, full symbol pipeline) --------

#[test]
fn throughput_qpsk_roundtrip() {
    use crate::core::{Block, IqToIqChain};
    use crate::modulate::{QpskMapper, QpskMod};
    use crate::demodulate::{QpskDemod, QpskDecider};

    let n_syms = 65_536;
    let n_bits = n_syms * 2;
    let repeats = 30;

    // Cycle through all four dibits
    let bits_in: Vec<u8> = (0..n_bits).map(|i| ((i / 2 + i) & 1) as u8).collect();
    let mut syms     = vec![C32::default(); n_syms];
    let mut soft     = vec![C32::default(); n_syms];
    let mut bits_out = vec![0u8; n_bits];

    let mut mapper   = QpskMapper::new();
    let mut modstage = IqToIqChain::new(QpskMod::new(1.0, 0.0, 1.0));
    let mut demod    = IqToIqChain::new(QpskDemod::new(1.0));
    let mut decider  = QpskDecider::new();

    let (msps, dt) = measure_throughput(
        || {
            mapper.process(&bits_in, &mut syms);
            let iq = modstage.process_ref(&syms);
            demod.process_into(&iq, &mut soft);
            decider.process(&soft, &mut bits_out);
            black_box(bits_out[0]);
            n_syms
        },
        n_syms,
        repeats,
    );

    println!("[QPSK] {:.2} Msps in {:.3}s", msps, dt);
    let min_msps = minsps_from_env(0.2);
    assert!(msps >= min_msps, "QPSK throughput {:.2} Msps < min {:.2} Msps", msps, min_msps);
}

// --- BPSK (mapper → mod → demod → decider, full symbol pipeline) --------

#[test]
fn throughput_bpsk_roundtrip() {
    use crate::core::{Block, IqToIqChain};
    use crate::modulate::{BpskMapper, BpskMod};
    use crate::demodulate::{BpskDemod, BpskDecider};

    let n = 65_536;
    let repeats = 30;

    let bits_in: Vec<u8> = (0..n).map(|i| (i & 1) as u8).collect();
    let mut syms   = vec![C32::default(); n];
    let mut soft   = vec![C32::default(); n];
    let mut bits_out = vec![0u8; n];

    let mut mapper  = BpskMapper::new();
    let mut modstage = IqToIqChain::new(BpskMod::new(1.0, 0.0, 1.0));
    let mut demod   = IqToIqChain::new(BpskDemod::new(1.0));
    let mut decider = BpskDecider::new();

    let (msps, dt) = measure_throughput(
        || {
            mapper.process(&bits_in, &mut syms);
            let iq = modstage.process_ref(&syms);
            demod.process_into(&iq, &mut soft);
            decider.process(&soft, &mut bits_out);
            black_box(bits_out[0]);
            n
        },
        n,
        repeats,
    );

    println!("[BPSK] {:.2} Msps in {:.3}s", msps, dt);
    let min_msps = minsps_from_env(0.2);
    assert!(msps >= min_msps, "BPSK throughput {:.2} Msps < min {:.2} Msps", msps, min_msps);
}

// --- PM -----------------------------------------------------------------

#[test]
fn throughput_pm_roundtrip() {
    use crate::core::{AudioToIqChain, IqToAudioChain};
    use crate::modulate::PmDirectPhaseMod;
    use crate::demodulate::PmQuadratureDemod;

    let fs = 48_000.0;
    let n = 65_536;
    let repeats = 30;

    let audio = real_tone(fs, 900.0, n, 0.5);
    let mut tx = AudioToIqChain::new(PmDirectPhaseMod::new(fs, 0.9, 0.0));
    let mut rx = IqToAudioChain::new(PmQuadratureDemod::new(fs, 0.9, 5_000.0));

    let (msps, dt) = measure_throughput(
        || {
            let iq = tx.process(audio.clone());
            let out = rx.process(iq);
            out.len()
        },
        n,
        repeats,
    );

    println!("[PM] {:.2} Msps in {:.3}s", msps, dt);
    let min_msps = minsps_from_env(0.25);
    assert!(msps >= min_msps, "PM throughput {:.2} Msps < min {:.2} Msps", msps, min_msps);
}
