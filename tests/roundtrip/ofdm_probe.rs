// Copyright (c) 2026 G & R Associates LLC
// SPDX-License-Identifier: MIT OR Apache-2.0

// The COFDM receive probe: `OfdmFrameStreamDemod::feed_probed` exposing the
// equalizer's output symbols and a per-coded-bit correction map.
//
// **These sweeps run with noise on purpose.** A correction map is a picture of
// errors, and a test with no errors in it cannot tell a working map from an
// all-`Clean` stub — the same way every defect the COFDM RX-metrics work turned
// up (sync-candidate selection, CFO blindness, burst-boundary accounting)
// passed at exactly zero noise. Where a sweep's discriminating power depends on
// errors actually occurring, that is asserted too.

use crate::common::add_awgn;
use num_complex::Complex32 as C32;
use orion_sdr::core::Block;
use orion_sdr::demodulate::ofdm_frame::decode_chain;
use orion_sdr::demodulate::{
    BitOutcome, OfdmFrameStreamDemod, OfdmRxProbe, OfdmSoftDemod, RxFrame,
};
use orion_sdr::fec::{
    ConvCode, FrameMetadata, FramePacket, InnerFec, InterleaverKind, LdpcCode, OuterFec,
    PunctureRate, RxError, ScramblerKind, ScramblerPos, SeedMode,
};
use orion_sdr::modulate::ofdm_frame::{BlockPlan, block_plan, encode_chain_stages, symbol_config};
use orion_sdr::modulate::{
    CodecCache, ConstellationOrder, Mcs, McsTable, OfdmConfig, OfdmFrameMod,
};
use orion_sdr::multicarrier::CarrierPlan;
use orion_sdr::sync::OfdmPreamble;

const N_FFT: usize = 64;
const CP_LEN: usize = 8;
/// `McsTable::default_ladder` entry 1: QPSK + LDPC(N512R12) + BCH(t = 8) — the
/// concatenation the viewer's link runs, and the one whose *systematic* inner
/// code is what makes the estimate half of the map checkable at all.
const MCS: u8 = 1;

fn probe_config() -> OfdmConfig {
    let half = (N_FFT / 2) as i32;
    let data: Vec<i32> = (1..half).chain(-(half - 1)..0).collect();
    let plan = CarrierPlan::new(N_FFT, CP_LEN).with_data_carriers(data);
    OfdmConfig::new(plan, 48_000.0, 0.0, 1.0, ConstellationOrder::Bpsk)
}

fn probe_preamble(cfg: &OfdmConfig) -> OfdmPreamble {
    OfdmPreamble::new(4, 16)
        .with_training_symbol(cfg.carrier_plan.n_fft(), cfg.carrier_plan.cp_len())
}

fn sample_payload(len: usize) -> Vec<u8> {
    (0..len).map(|i| ((i * 37 + 11) & 0xff) as u8).collect()
}

/// One modulated frame plus everything needed to receive and re-derive it.
struct Link {
    cfg: OfdmConfig,
    pre: OfdmPreamble,
    table: McsTable,
    /// Index into `table` — a field rather than the [`MCS`] constant, so the
    /// same harness drives the convolutional inner arm as well as the LDPC one.
    mcs: u8,
    payload: Vec<u8>,
    /// Lead-in silence, the frame, trailing silence — what a real stream looks
    /// like to the sync search.
    iq: Vec<C32>,
    /// Offset of the frame's first sample in `iq`.
    frame_start: usize,
}

impl Link {
    fn new(payload_len: usize, seq: u32) -> Self {
        Self::with_chain(
            probe_config(),
            McsTable::default_ladder(),
            MCS,
            payload_len,
            seq,
        )
    }

    /// A link over an arbitrary config and MCS table — the entry point for the
    /// coding chains `Link::new`'s default does not reach: the convolutional
    /// inner arm, and the interleaved/scrambled configurations whose re-encode
    /// takes different branches.
    fn with_chain(cfg: OfdmConfig, table: McsTable, mcs: u8, payload_len: usize, seq: u32) -> Self {
        let pre = probe_preamble(&cfg);
        let modu = OfdmFrameMod::new(cfg.clone(), table.clone(), pre);
        let payload = sample_payload(payload_len);
        let frame = FramePacket::new(FrameMetadata::new(seq, mcs), payload.clone());

        let frame_start = 24;
        let mut iq = vec![C32::default(); frame_start];
        iq.extend_from_slice(&modu.modulate_frame(&frame, 0));
        iq.extend(vec![C32::default(); 64]);
        Self {
            cfg,
            pre,
            table,
            mcs,
            payload,
            iq,
            frame_start,
        }
    }

    /// A receiver over this link, error rates on so the correction map can be
    /// checked against the scalar rate it is an expansion of.
    fn receiver(&self) -> OfdmFrameStreamDemod {
        OfdmFrameStreamDemod::new(self.cfg.clone(), self.table.clone(), self.pre)
            .with_error_rates(true)
    }

    /// The frame with AWGN at `noise_scale` times the **payload's** power,
    /// applied to the frame body only.
    ///
    /// Two methodology points, both load-bearing when reading the sweeps below:
    ///
    /// - `noise_scale` is referenced to the payload's power rather than the
    ///   whole buffer's mean, as `snr::cofdm_fer` does, so a figure here means
    ///   what it means there instead of being hostage to the preamble's
    ///   amplitude.
    /// - The **preamble is left clean**. These tests characterize the map, not
    ///   acquisition, and noising the preamble folds sync failures and channel
    ///   estimation error into the same knob — measured here, that pulls the
    ///   usable band down to `noise_scale <= 0.2` with frames dropping out
    ///   seed by seed, well before the payload FEC is under any real pressure.
    ///   `snr::ofdm_sync` covers acquisition separately.
    fn noisy(&self, noise_scale: f32, seed: u64) -> Vec<C32> {
        let mut iq = self.iq.clone();
        if noise_scale <= 0.0 {
            return iq;
        }
        let start = self.body_start();
        let body = &self.iq[start..];
        let power: f32 = body.iter().map(|c| c.norm_sqr()).sum::<f32>() / body.len() as f32;
        add_awgn(&mut iq[start..], power * noise_scale, seed);
        iq
    }

    /// Offset in `iq` of the first sample after the preamble.
    fn body_start(&self) -> usize {
        self.frame_start + self.pre.total_len()
    }

    fn sps(&self) -> usize {
        self.cfg.samples_per_ofdm_symbol()
    }

    /// This link's MCS entry.
    fn mcs(&self) -> Mcs {
        self.table.get(self.mcs).expect("the link's MCS index")
    }

    /// This link's payload block plan — the bit counts every index computation
    /// below is expressed in.
    fn plan(&self) -> BlockPlan {
        let mcs = self.mcs();
        block_plan(
            self.payload.len(),
            self.cfg.payload_crc,
            mcs.outer_fec,
            mcs.inner_fec,
            self.cfg.outer_interleaver,
            self.cfg.inner_interleaver,
            &CodecCache::new(),
        )
    }

    /// What the transmitter sent, in the coded-bit domain: the re-encode of the
    /// (known) payload. The receiver derives the same stream from the payload it
    /// recovered — that is what makes the map work over the air — so computing
    /// it from the known payload here is an independent path to the same bits.
    fn truth(&self) -> Vec<u8> {
        let mcs = self.mcs();
        encode_chain_stages(
            &self.payload,
            self.cfg.payload_crc,
            mcs.outer_fec,
            mcs.inner_fec,
            self.cfg.outer_interleaver,
            self.cfg.inner_interleaver,
            self.cfg.scrambler,
            self.cfg.scrambler_pos,
            0,
            &CodecCache::new(),
        )
        .coded
    }
}

/// Receives `iq` through a probed `feed`, returning the probe and the frames.
fn feed_probed(rx: &mut OfdmFrameStreamDemod, iq: &[C32]) -> (OfdmRxProbe, Vec<RxFrame>) {
    let mut probe = OfdmRxProbe::new();
    let frames: Vec<RxFrame> = rx
        .feed_probed(iq, &mut probe)
        .into_iter()
        .filter_map(Result::ok)
        .collect();
    (probe, frames)
}

// ── The map against the rate it expands ────────────────────────────────────

#[test]
fn corrections_track_the_channel_ber() {
    // THE load-bearing test. The correction map is `received XOR truth` kept per
    // bit rather than collapsed to a scalar, so counting the bits it says
    // arrived wrong and dividing by the coded-bit count must reproduce the
    // reported `channel_ber` EXACTLY — not approximately.
    //
    // That ties the map to a quantity which has already survived a noise sweep
    // and a cross-check against a regenerated payload. If the map ever disagrees
    // with the rate it expands, everything downstream of it is guessing.
    let link = Link::new(40, 0x5EED);
    let coded_bits = link.plan().coded_bits;
    let mut any_errors = false;

    for (i, &noise) in [0.0f32, 0.2, 0.4, 0.6].iter().enumerate() {
        let mut rx = link.receiver();
        let (probe, frames) = feed_probed(&mut rx, &link.noisy(noise, 0xA11CE + i as u64));
        assert_eq!(frames.len(), 1, "noise {noise}: the frame must decode");
        assert_eq!(probe.frames().len(), 1, "noise {noise}: one probe frame");

        let f = probe.iter().next().expect("one probe frame");
        assert!(
            f.meta.decoded,
            "noise {noise}: a decoded frame has ground truth"
        );
        let map = f.correction;
        assert_eq!(
            map.len(),
            coded_bits,
            "noise {noise}: the map spans the whole coded block"
        );
        let cber = frames[0]
            .diagnostics
            .channel_ber
            .expect("with_error_rates(true)");

        let wrong = map.iter().filter(|o| o.arrived_wrong()).count();
        assert_eq!(
            wrong as f32 / map.len() as f32,
            cber,
            "noise {noise}: the map must be the exact per-bit expansion of \
             channel_ber ({wrong} of {coded_bits} bits vs {cber})"
        );
        any_errors |= wrong > 0;
    }

    // A sweep in which nothing ever went wrong would pass against an
    // all-`Clean` stub, so require the sweep to have carried real errors.
    assert!(
        any_errors,
        "the sweep must reach a noise level that actually corrupts bits"
    );
}

#[test]
fn uncorrected_bits_track_the_inner_ber() {
    // The re-encode is deterministic and LDPC(N512R12) is systematic, so the
    // decoder's estimate differs from the truth in the first `k` bits of each
    // codeword exactly where the inner decoder's output differs from what the
    // transmitter fed the inner encoder. Counting over those systematic
    // positions — excluding the final codeword's zero-padding tail, which
    // `inner_ber` is not measured over — must reproduce
    // `inner_ber * outer_il_bits`.
    let link = Link::new(40, 0x1234);
    let plan = link.plan();
    let (n, k) = (LdpcCode::N512R12.n(), LdpcCode::N512R12.k());
    let mut checked = 0usize;
    let mut with_errors = 0usize;

    // Deep enough into the cliff that the inner decoder regularly fails to clean
    // a codeword up, shallow enough that most frames still verify and so still
    // have ground truth. Per-seed, not per-noise-level: whether a given frame's
    // inner decoder leaves anything behind is a coin flip at this SNR.
    for &noise in &[0.6f32, 0.7] {
        for seed in 0..8u64 {
            let mut rx = link.receiver();
            let (probe, frames) = feed_probed(&mut rx, &link.noisy(noise, 0xBEE5 + seed));
            if frames.is_empty() {
                continue; // past the cliff; no ground truth to compare against
            }
            let f = probe.iter().next().expect("one probe frame");
            let map = f.correction;
            assert_eq!(
                f.meta.codeword_bits, n,
                "the LDPC codeword length is reported"
            );
            assert_eq!(
                f.meta.codeword_info_bits, k,
                "the LDPC info length is reported"
            );

            let inner_ber = frames[0].diagnostics.inner_ber.expect("with_error_rates");
            let expected = (inner_ber as f64 * plan.outer_il_bits as f64).round() as usize;

            let got = systematic_disagreements(map, n, k, plan.outer_il_bits);
            assert_eq!(
                got, expected,
                "noise {noise} seed {seed}: systematic bits the inner decoder \
                 got wrong ({got}) must equal inner_ber * outer_il_bits ({expected})"
            );
            checked += 1;
            with_errors += usize::from(got > 0);
        }
    }

    assert!(
        checked >= 8,
        "the sweep must decode enough frames to mean something, got {checked}"
    );
    assert!(
        with_errors > 0,
        "{checked} frames checked but none had a bit the inner decoder failed to \
         clean up — the sweep cannot tell this apart from an all-Clean stub"
    );
}

/// Counts the map positions where the decoder's estimate disagrees with the
/// truth — `Uncorrected` and `Introduced` are exactly those two states — over
/// each codeword's first `k` (systematic) bits, stopping at `info_bits`.
fn systematic_disagreements(map: &[BitOutcome], n: usize, k: usize, info_bits: usize) -> usize {
    map.chunks(n)
        .enumerate()
        .flat_map(|(c, cw)| {
            cw.iter()
                .take(k)
                .enumerate()
                .map(move |(b, o)| (c * k + b, *o))
        })
        .filter(|&(idx, o)| idx < info_bits && o.decoder_disagreed())
        .count()
}

#[test]
fn the_estimate_covers_the_final_codewords_padding_tail() {
    // The truncation hazard, asserted where it actually bites.
    //
    // `decode_chain` trims the inner decoder's output to the block plan's
    // `outer_il_bits` before the outer decoder sees it — here 552 bits of the
    // 768 the decoder produced, so 216 bits of the last codeword's zero padding
    // are dropped. Re-encoding the *trimmed* output zero-pads that tail back to
    // zero, which silently asserts the decoder got the padding right and paints
    // any error there `Clean`. It is one region of a few hundred bits: invisible
    // in a screenshot, wrong in a test.
    //
    // `uncorrected_bits_track_the_inner_ber` cannot catch this — it counts only
    // `idx < outer_il_bits`, where trimmed and untrimmed agree by construction.
    // So reconstruct the estimate from `truth XOR map` and require its
    // systematic bits to equal the inner decoder's own output over the WHOLE
    // codeword, padding included.
    let link = Link::new(40, 0xC0DE);
    let plan = link.plan();
    let truth = link.truth();
    let (n, k) = (LdpcCode::N512R12.n(), LdpcCode::N512R12.k());
    let mut tails_exercised = 0usize;

    for &noise in &[0.6f32, 0.7, 0.8] {
        for seed in 0..8u64 {
            let iq = link.noisy(noise, 0xD00D + seed);
            let mut rx = link.receiver();
            let (probe, frames) = feed_probed(&mut rx, &iq);
            if frames.is_empty() {
                continue;
            }
            let f = probe.iter().next().expect("one probe frame");
            let map = f.correction;

            // The estimate, recovered from the two streams the map is the XOR of.
            let estimate: Vec<u8> = map
                .iter()
                .zip(truth.iter())
                .map(|(&o, &t)| t ^ u8::from(o.decoder_disagreed()))
                .collect();

            // The inner decoder's own output, re-derived by decoding the same
            // symbols the probe handed back — an independent path to the vector
            // `ChainOutcome::inner_out_bits` carries, now untrimmed.
            let inner_out = decode_probe_symbols(&link, f.symbols, &plan);
            assert_eq!(
                inner_out.len(),
                plan.inner_coded_bits / n * k,
                "the inner decoder emits whole codewords' worth of info bits"
            );

            for (c, cw) in estimate.chunks(n).enumerate() {
                for b in 0..k.min(cw.len()) {
                    assert_eq!(
                        cw[b],
                        inner_out[c * k + b],
                        "noise {noise} seed {seed}, codeword {c} bit {b}: the \
                         estimate's systematic bits must be the inner decoder's \
                         own output"
                    );
                }
            }
            // The assertion above only discriminates where the decoder's padding
            // tail is nonzero — a trimmed re-encode agrees everywhere else.
            if inner_out[plan.outer_il_bits..].iter().any(|&b| b != 0) {
                tails_exercised += 1;
            }
        }
    }

    assert!(
        tails_exercised > 0,
        "the sweep never produced a nonzero decoder padding tail, so it could \
         not have told a trimmed re-encode from an untrimmed one"
    );
}

/// Re-demaps the probe's equalized symbols and runs the payload decode chain
/// over them, returning the inner decoder's untrimmed output.
///
/// The probe hands back the symbols at exactly the point the receiver's LLRs
/// come from — post-equalizer, post-common-phase-error removal, pre-demapper —
/// so this reproduces the receiver's own decode rather than approximating it.
fn decode_probe_symbols(link: &Link, symbols: &[C32], plan: &BlockPlan) -> Vec<u8> {
    let mcs = link.mcs();
    let cfg = symbol_config(&link.cfg, mcs.constellation);
    let n_data = cfg.carrier_plan.data_carriers().len();
    let bps = cfg.bits_per_ofdm_symbol();
    let n_sym = symbols.len() / n_data;

    let mut soft = OfdmSoftDemod::new(&cfg);
    let mut llrs = vec![0.0f32; n_sym * bps];
    for s in 0..n_sym {
        let wr = soft.process(
            &symbols[s * n_data..(s + 1) * n_data],
            &mut llrs[s * bps..(s + 1) * bps],
        );
        assert_eq!(wr.out_written, bps, "symbol {s} demaps");
    }

    decode_chain(
        &llrs,
        plan,
        link.cfg.payload_crc,
        mcs.outer_fec,
        mcs.inner_fec,
        link.cfg.outer_interleaver,
        link.cfg.inner_interleaver,
        link.cfg.scrambler,
        link.cfg.scrambler_pos,
        0,
        &CodecCache::new(),
        link.cfg.ldpc_decode_rule,
    )
    .expect("the probe's symbols decode the same way the receiver's did")
    .inner_out_bits
}

// ── The probe as an observation, not a code path ───────────────────────────

#[test]
fn probing_does_not_change_what_decodes() {
    // The probe must be an observation. Same input through `feed` and
    // `feed_probed` has to yield identical packets AND identical diagnostics —
    // if probing moved a single LLR, every number the map is checked against
    // would be describing a different decode.
    let link = Link::new(48, 0xF00D);
    for (i, &noise) in [0.0f32, 0.4, 0.7, 1.0].iter().enumerate() {
        let iq = link.noisy(noise, 0xC0FFEE + i as u64);

        let plain: Vec<Result<RxFrame, RxError>> = link.receiver().feed(&iq);
        let mut probe = OfdmRxProbe::new();
        let probed: Vec<Result<RxFrame, RxError>> = link.receiver().feed_probed(&iq, &mut probe);

        assert_eq!(
            plain.len(),
            probed.len(),
            "noise {noise}: same number of results"
        );
        for (a, b) in plain.iter().zip(probed.iter()) {
            match (a, b) {
                (Ok(x), Ok(y)) => {
                    assert_eq!(x.packet, y.packet, "noise {noise}: same packet");
                    assert_eq!(
                        x.diagnostics, y.diagnostics,
                        "noise {noise}: same diagnostics"
                    );
                }
                (Err(x), Err(y)) => assert_eq!(x, y, "noise {noise}: same error"),
                _ => panic!("noise {noise}: probing changed success/failure"),
            }
        }
    }
}

#[test]
fn the_probe_reuses_its_buffers() {
    // The direct test of the outer-cache property: without it, "no per-frame
    // allocation" is a comment. Points per frame is `coded_bits /
    // bits_per_symbol` rounded up to whole OFDM symbols, and neither term
    // depends on the carrier occupancy — so a buffer sized once never has to
    // grow again, in steady state rather than merely usually.
    //
    // `Vec::capacity` is not on the public surface, but a `Vec` that does not
    // reallocate keeps its backing pointer and one that grows moves it, so the
    // property is observable through the slices the probe already exposes.
    let link = Link::new(40, 0x2222);
    let iq = link.noisy(0.3, 0x7777);
    let mut probe = OfdmRxProbe::new();

    for _ in 0..2 {
        link.receiver().feed_probed(&iq, &mut probe);
    }
    assert!(
        !probe.symbols().is_empty() && !probe.correction().is_empty(),
        "the warm-up must actually fill the buffers"
    );
    let base = (
        probe.symbols().as_ptr() as usize,
        probe.correction().as_ptr() as usize,
    );
    let lens = (probe.symbols().len(), probe.correction().len());

    for i in 0..8 {
        link.receiver().feed_probed(&iq, &mut probe);
        assert_eq!(
            (probe.symbols().len(), probe.correction().len()),
            lens,
            "call {i}: the same frame must fill the same amount"
        );
        assert_eq!(
            (
                probe.symbols().as_ptr() as usize,
                probe.correction().as_ptr() as usize
            ),
            base,
            "call {i}: a warm probe must not reallocate its buffers"
        );
    }
}

// ── The symbols, against the point in the chain they claim to come from ────

#[test]
fn probe_symbols_are_what_evm_measured() {
    // Ties the exposed symbols to the point in the chain they are claimed to
    // come from — post-equalizer, post-CPE-removal, pre-demapper — rather than
    // to a plausible-looking cloud. Recomputing EVM from the probe's symbols
    // against their own hard decisions must reproduce the reported `evm_db`.
    let link = Link::new(40, 0x3333);
    for (i, &noise) in [0.0f32, 0.3, 0.6].iter().enumerate() {
        let mut rx = link.receiver();
        let (probe, frames) = feed_probed(&mut rx, &link.noisy(noise, 0x4444 + i as u64));
        assert_eq!(frames.len(), 1, "noise {noise}: the frame must decode");

        let symbols = probe.iter().next().expect("one probe frame").symbols;
        let reported = frames[0]
            .diagnostics
            .evm_db
            .expect("EVM is always measured");
        let recomputed = qpsk_evm_db(symbols);
        assert!(
            (recomputed - reported).abs() < 0.01,
            "noise {noise}: EVM recomputed from the probe's symbols \
             ({recomputed:.4} dB) must match the reported {reported:.4} dB"
        );
    }
}

/// EVM in dB against each symbol's own nearest QPSK point — the same
/// decision-directed reference the receiver uses.
fn qpsk_evm_db(symbols: &[C32]) -> f32 {
    let a = 1.0 / 2.0f32.sqrt();
    let mut err = 0.0f64;
    let mut reference = 0.0f64;
    for s in symbols {
        let ideal = C32::new(a * s.re.signum(), a * s.im.signum());
        let e = s - ideal;
        err += e.norm_sqr() as f64;
        reference += ideal.norm_sqr() as f64;
    }
    (10.0 * (err / reference).log10()) as f32
}

#[test]
fn the_equalized_cloud_sits_at_unit_energy() {
    // QPSK maps to ±1/√2, and the equalizer divides out the channel *including*
    // any uniform scalar — so the cloud lands on the unit circle whatever the
    // transmit amplitude was. That is what lets a display fix its plot extent as
    // a constant instead of auto-scaling, so assert it across four decades of
    // gain.
    let mean_radius = |gain: f32| -> f32 {
        let link = Link::new(40, 0x5555);
        let mut iq = link.noisy(0.0, 0);
        for s in iq.iter_mut() {
            *s *= gain;
        }
        let mut rx = link.receiver();
        let (probe, frames) = feed_probed(&mut rx, &iq);
        assert_eq!(frames.len(), 1, "gain {gain}: the frame must decode");
        let symbols = probe.iter().next().expect("one probe frame").symbols;
        symbols.iter().map(|s| s.norm()).sum::<f32>() / symbols.len() as f32
    };

    for gain in [0.01f32, 1.0, 10.0, 100.0] {
        let r = mean_radius(gain);
        assert!(
            (r - 1.0).abs() < 0.02,
            "gain {gain}: the equalized cloud must sit at unit energy, \
             got mean |s| = {r:.4}"
        );
    }
}

// ── The states, at the ends of the range ───────────────────────────────────

#[test]
fn a_clean_link_shows_no_corrections() {
    // Every state `Clean` at high SNR — and in particular NO `Introduced`. One
    // of those on a noiseless link means the re-encode or the state derivation
    // is wrong, not that the link is.
    let link = Link::new(64, 0x6666);
    let mut rx = link.receiver();
    let (probe, frames) = feed_probed(&mut rx, &link.noisy(0.0, 0));
    assert_eq!(frames.len(), 1);
    assert_eq!(frames[0].packet.payload, link.payload);

    let f = probe.iter().next().expect("one probe frame");
    let map = f.correction;
    assert!(!map.is_empty(), "a decoded frame carries a map");
    assert_eq!(
        map.iter().filter(|&&o| o != BitOutcome::Clean).count(),
        0,
        "a noiseless link must be entirely Clean; got {} Corrected, \
         {} Uncorrected, {} Introduced",
        count(map, BitOutcome::Corrected),
        count(map, BitOutcome::Uncorrected),
        count(map, BitOutcome::Introduced),
    );
}

fn count(map: &[BitOutcome], want: BitOutcome) -> usize {
    map.iter().filter(|&&o| o == want).count()
}

#[test]
fn a_failed_frame_yields_symbols_but_no_map() {
    // A payload that fails its CRC has no ground truth, so no map — but
    // `soft_demap` ran, so the symbols exist, and the constellation is precisely
    // where an operator looks when frames stop decoding.
    let link = Link::new(40, 0x7777);
    // Corrupt deep in the payload, past the header, beyond the FEC's reach.
    let mut iq = link.noisy(0.0, 0);
    for s in iq
        .iter_mut()
        .skip(link.body_start() + 12 * link.sps())
        .take(400)
    {
        *s = C32::new(-s.re * 3.0, -s.im * 3.0);
    }

    let mut rx = link.receiver();
    let mut probe = OfdmRxProbe::new();
    let results = rx.feed_probed(&iq, &mut probe);
    assert!(
        !results.is_empty() && results.iter().all(|r| r.is_err()),
        "the corrupted payload must be reported as an error"
    );
    assert_eq!(probe.frames().len(), 1, "the failed frame is still probed");

    let f = probe.iter().next().expect("one probe frame");
    assert!(!f.meta.decoded, "no ground truth, so not `decoded`");
    assert!(
        f.correction.is_empty(),
        "a frame without ground truth carries no map"
    );
    assert!(
        !f.symbols.is_empty(),
        "the constellation survives a failed payload"
    );
    assert_eq!(
        f.meta.constellation,
        ConstellationOrder::Qpsk,
        "the header decoded, so the payload constellation is known"
    );
    assert_eq!(
        f.meta.sequence_num,
        Some(0x7777),
        "and so is the sequence number"
    );
}

#[test]
fn a_failed_header_yields_nothing() {
    // A frame whose header fails never reaches the payload demapper, so there is
    // nothing to probe — not even a constellation.
    let link = Link::new(40, 0x8888);
    let mut iq = link.noisy(0.0, 0);
    for s in iq.iter_mut().skip(link.body_start()).take(6 * link.sps()) {
        *s = C32::new(-s.re * 4.0, s.im * 4.0);
    }

    let mut rx = link.receiver();
    let mut probe = OfdmRxProbe::new();
    let results = rx.feed_probed(&iq, &mut probe);
    assert!(
        !results.iter().any(|r| r.is_ok()),
        "a corrupted header must not decode"
    );
    assert!(
        probe.is_empty(),
        "a header failure yields no probe frame at all, got {}",
        probe.frames().len()
    );
    assert!(probe.symbols().is_empty(), "and no symbols");
}

/// Every symbol and every outcome in the flat buffers must belong to exactly one
/// frame.
///
/// This is what "a probe frame is committed as one unit" means from the outside:
/// a partial attempt that appended and then failed would leave symbols no frame
/// accounts for, and a caller iterating `iter()` would never see them while
/// `symbols()` still counted them.
///
/// Stated as coverage rather than as span arithmetic, because the spans are
/// private now — `iter()` hands out resolved slices so a stale record cannot
/// index the wrong frame's data. Contiguity is guaranteed by construction (the
/// buffers are only ever appended to); coverage is the property that can fail.
fn assert_spans_cover(probe: &OfdmRxProbe) {
    let (mut sym, mut corr) = (0usize, 0usize);
    for f in probe.iter() {
        sym += f.symbols.len();
        corr += f.correction.len();
    }
    assert_eq!(
        sym,
        probe.symbols().len(),
        "{} orphan symbol(s) belong to no frame",
        probe.symbols().len() - sym
    );
    assert_eq!(
        corr,
        probe.correction().len(),
        "{} orphan outcome(s) belong to no frame",
        probe.correction().len() - corr
    );
}

#[test]
fn an_incomplete_frame_is_not_reported_twice() {
    // `BodyError::Incomplete` consumes no buffer and the next `feed` re-runs the
    // frame from its header. A probe that appended as it went would emit the
    // same frame twice — so a probe frame is committed as one unit, and a
    // partial attempt is rolled back.
    //
    // Today `soft_demap` returns `None` from its symbol-extraction loop, before
    // the sink loop runs, so a short buffer cannot append anything and the
    // rollback in `try_one_frame` never has work to do. It is a guard against a
    // future rearrangement of that function, not a live code path — which is why
    // this asserts the *observable* invariant (nothing appended, spans cover the
    // buffers) rather than the rollback itself.
    let link = Link::new(64, 0x9999);
    let iq = link.noisy(0.0, 0);
    let split = link.body_start() + 15 * link.sps(); // mid-payload

    let mut rx = link.receiver();
    let mut probe = OfdmRxProbe::new();

    let first = rx.feed_probed(&iq[..split], &mut probe);
    assert!(first.is_empty(), "the frame has not fully arrived");
    assert!(
        probe.is_empty() && probe.symbols().is_empty(),
        "a partial attempt must leave nothing behind, got {} frame(s) / {} symbols",
        probe.frames().len(),
        probe.symbols().len()
    );
    assert_spans_cover(&probe);

    let second: Vec<RxFrame> = rx
        .feed_probed(&iq[split..], &mut probe)
        .into_iter()
        .filter_map(Result::ok)
        .collect();
    assert_eq!(second.len(), 1, "the frame completes on the second feed");
    assert_eq!(second[0].packet.payload, link.payload);
    assert_eq!(probe.frames().len(), 1, "exactly one probe frame, not two");
    assert_spans_cover(&probe);
}

#[test]
fn several_frames_in_one_call_get_separate_spans() {
    // `feed` drains as many frames as the buffer holds, so the flat buffers have
    // to carry several frames end to end with non-overlapping spans — which is
    // also why a borrowed buffer could not have lived on `RxFrame`.
    //
    // The stream ends mid-frame on purpose: the drain loop then exits on
    // `Incomplete` *after* having produced records, which is the arrangement in
    // which a half-committed fourth frame would show up as symbols no span
    // covers.
    let cfg = probe_config();
    let pre = probe_preamble(&cfg);
    let table = McsTable::default_ladder();
    let modu = OfdmFrameMod::new(cfg.clone(), table.clone(), pre);

    let mut iq = vec![C32::default(); 24];
    let payloads: Vec<Vec<u8>> = (0..3).map(|i| sample_payload(32 + i * 8)).collect();
    for (i, p) in payloads.iter().enumerate() {
        let frame = FramePacket::new(FrameMetadata::new(100 + i as u32, MCS), p.clone());
        iq.extend_from_slice(&modu.modulate_frame(&frame, 0));
    }
    // A fourth frame, truncated part-way through its payload.
    let partial = FramePacket::new(FrameMetadata::new(103, MCS), sample_payload(56));
    let partial_iq = modu.modulate_frame(&partial, 0);
    let keep = pre.total_len() + 15 * cfg.samples_per_ofdm_symbol();
    iq.extend_from_slice(&partial_iq[..keep]);

    let mut rx = OfdmFrameStreamDemod::new(cfg, table, pre).with_error_rates(true);
    let mut probe = OfdmRxProbe::new();
    let frames: Vec<RxFrame> = rx
        .feed_probed(&iq, &mut probe)
        .into_iter()
        .filter_map(Result::ok)
        .collect();

    assert_eq!(
        frames.len(),
        3,
        "the three complete frames drain in one call"
    );
    assert_eq!(
        probe.frames().len(),
        3,
        "and the truncated fourth contributes nothing"
    );
    for (i, f) in probe.iter().enumerate() {
        assert_eq!(f.meta.sequence_num, Some(100 + i as u32));
        assert!(!f.symbols.is_empty());
        assert!(!f.correction.is_empty());
    }
    assert_spans_cover(&probe);
}

#[test]
fn flush_probed_clears_the_probe() {
    // **`flush_probed` cannot produce a frame that `feed_probed` did not.**
    // `feed` extends the buffer and then drains it to exhaustion, and nothing
    // else fills the buffer, so by the time `flush` runs there is never a
    // complete frame left. An earlier version of this test asserted that a
    // frame "completing at end of stream" still got a probe record; measured,
    // that reduced to `0 == 0` and asserted nothing, because the state it
    // described is unreachable.
    //
    // What is real — and is a trap for a caller — is the per-call clear: a
    // `flush_probed` after a `feed_probed` wipes the probe the feed just
    // filled. Assert that, because a caller who drains `frames()` after
    // flushing rather than after feeding silently gets nothing.
    let link = Link::new(40, 0xAAAA);
    let iq = link.noisy(0.0, 0);
    let mut rx = link.receiver();
    let mut probe = OfdmRxProbe::new();

    let held = rx.feed_probed(&iq[..iq.len() - 64], &mut probe);
    assert_eq!(
        held.iter().filter(|r| r.is_ok()).count(),
        1,
        "the feed drains the frame; there is nothing left for flush to find"
    );
    assert_eq!(probe.frames().len(), 1, "and the feed filled the probe");

    let flushed: Vec<RxFrame> = rx
        .flush_probed(&mut probe)
        .into_iter()
        .filter_map(Result::ok)
        .collect();
    assert!(flushed.is_empty(), "nothing is left to flush");
    assert!(
        probe.is_empty() && probe.symbols().is_empty(),
        "flush_probed clears the probe, so the feed's records are gone"
    );
}

// ── The other coding chains ────────────────────────────────────────────────

/// What a noise sweep turned up, in frames rather than bits.
#[derive(Debug, Default)]
struct SweepStats {
    decoded: usize,
    /// Frames carrying at least one bit in the named state.
    corrected: usize,
    uncorrected: usize,
    introduced: usize,
    /// Frames where the decoder's estimate disagreed with the truth somewhere,
    /// i.e. `Uncorrected + Introduced > 0`.
    disagreed: usize,
}

/// Sweeps `noises` x `seeds` over `link`, holding every decoded frame's map to
/// both halves of what defines it.
///
/// **Two assertions, because one of them is blind.** `arrived_wrong()` is
/// `Corrected | Uncorrected` — both of which mean "received != truth" — so
/// counting them recovers the channel-error count *whatever the estimate stream
/// says*. Checking the map against `channel_ber` therefore validates only the
/// received-versus-truth half, and a completely broken re-encode would sail
/// through it. So the estimate half is tied to `inner_ber` as well.
///
/// `strict_inner` asks for the converse — no disagreement at all when
/// `inner_ber` is zero. That holds only where the inner decoder's output has no
/// bits outside `inner_ber`'s comparison window: true for the convolutional arm,
/// whose Viterbi returns exactly `outer_il_bits`, and false for LDPC, which
/// emits whole codewords and so carries a padding tail that `inner_ber` does not
/// see — the tail `the_estimate_covers_the_final_codewords_padding_tail` exists
/// for.
fn sweep_map_against_cber(
    link: &Link,
    noises: &[f32],
    seeds: u64,
    strict_inner: bool,
) -> SweepStats {
    let coded_bits = link.plan().coded_bits;
    let mut stats = SweepStats::default();
    for &noise in noises {
        for seed in 0..seeds {
            let mut rx = link.receiver();
            let (probe, frames) = feed_probed(&mut rx, &link.noisy(noise, 0x5A17 + seed * 7919));
            if frames.is_empty() || probe.frames().is_empty() {
                continue; // past the cliff
            }
            let f = probe.iter().next().expect("one probe frame");
            if !f.meta.decoded {
                continue; // no ground truth
            }
            stats.decoded += 1;
            let map = f.correction;
            assert_eq!(
                map.len(),
                coded_bits,
                "noise {noise} seed {seed}: the map spans the whole coded block"
            );
            let d = &frames[0].diagnostics;
            let cber = d.channel_ber.expect("with_error_rates");
            let wrong = map.iter().filter(|o| o.arrived_wrong()).count();
            assert_eq!(
                wrong as f32 / map.len() as f32,
                cber,
                "noise {noise} seed {seed}: the map must be the exact per-bit \
                 expansion of channel_ber ({wrong} of {coded_bits} vs {cber})"
            );

            // **The zero-noise rung, and it is not redundant.** Nothing arrived
            // wrong and the decoder had nothing to do, so every cell must be
            // `Clean` — which makes this the one assertion that bounds the
            // estimate stream from *above* on any chain. Without it a re-encode
            // that skipped a stage (the inner interleave, the after-inner
            // scramble) would disagree with the truth almost everywhere and no
            // other check here would notice: `channel_ber` is blind to the
            // estimate, and the `inner_ber > 0` implication only ever asks for
            // *more* disagreement, never less. Verified by mutation — deleting
            // the scramble from `reencode_inner_output` passes every other test
            // in this file.
            if noise == 0.0 {
                assert_eq!(
                    count(map, BitOutcome::Clean),
                    map.len(),
                    "noise 0 seed {seed}: a noiseless link must be entirely \
                     Clean; got {} Corrected, {} Uncorrected, {} Introduced",
                    count(map, BitOutcome::Corrected),
                    count(map, BitOutcome::Uncorrected),
                    count(map, BitOutcome::Introduced),
                );
            }

            // The estimate half. `Uncorrected` and `Introduced` are exactly the
            // states where the decoder's re-encoded output disagrees with the
            // truth, so they must be present precisely when the inner decoder
            // left an error behind.
            let inner_ber = d.inner_ber.expect("with_error_rates");
            let disagreed = map.iter().filter(|o| o.decoder_disagreed()).count();
            if inner_ber > 0.0 {
                assert!(
                    disagreed > 0,
                    "noise {noise} seed {seed}: inner_ber is {inner_ber} but the \
                     map says the decoder agreed with the truth everywhere"
                );
            }
            if strict_inner && inner_ber == 0.0 {
                assert_eq!(
                    disagreed, 0,
                    "noise {noise} seed {seed}: inner_ber is zero and this arm \
                     has no padding tail outside its window, so the re-encoded \
                     estimate must match the truth everywhere"
                );
            }

            stats.corrected += usize::from(count(map, BitOutcome::Corrected) > 0);
            stats.uncorrected += usize::from(count(map, BitOutcome::Uncorrected) > 0);
            stats.introduced += usize::from(count(map, BitOutcome::Introduced) > 0);
            stats.disagreed += usize::from(disagreed > 0);
        }
    }
    stats
}

/// A non-scattered link whose inner code is convolutional — the *other* arm of
/// `reencode_inner_output`, reachable today on any link that selects it.
fn conv_table() -> McsTable {
    McsTable::new(vec![Mcs::new(
        ConstellationOrder::Qpsk,
        InnerFec::Convolutional {
            rate: PunctureRate::R1_2,
            code: ConvCode::K5,
        },
        OuterFec::Bch { t: 8 },
    )])
}

#[test]
fn the_convolutional_arm_maps_the_same_way() {
    // The map is defined by re-encode-and-compare precisely so that it renders
    // identically on both inner arms — that is what decides it against exposing
    // the LDPC decoder's internal codeword, which the convolutional arm has no
    // counterpart for. Every other test here runs on LDPC, so without this the
    // argument that chose the design is the one thing unverified.
    //
    // **This arm is reachable now, not only via DVB-T.** Any link whose MCS
    // selects a convolutional inner code takes it; `throughput_frame_chain_conv_rs`
    // already builds exactly such a table for the batch path.
    //
    // A *clean* convolutional link would prove nothing: with received == truth
    // == estimate everywhere the map is all-`Clean` however wrong the re-encode
    // is. Only noise discriminates.
    // `strict_inner`: Viterbi returns exactly `outer_il_bits`, so unlike LDPC
    // there is no padding tail outside `inner_ber`'s window and the estimate
    // must match the truth *everywhere* on a frame the inner decoder cleaned up.
    // That is the assertion a broken convolutional re-encode fails; the
    // channel_ber one cannot, being blind to the estimate stream.
    let link = Link::with_chain(probe_config(), conv_table(), 0, 40, 0xC047);
    let stats = sweep_map_against_cber(&link, &[0.0, 0.3, 0.5, 0.7], 8, true);

    assert!(
        stats.decoded >= 12,
        "the sweep must decode enough frames to mean something, got {}",
        stats.decoded
    );
    assert!(
        stats.corrected > 0,
        "{} frames checked but none had a bit the convolutional decoder fixed \
         — the sweep cannot tell this apart from an all-Clean stub",
        stats.decoded
    );
    assert!(
        stats.disagreed > 0,
        "{} frames checked but the decoder's estimate never disagreed with the \
         truth — the sweep never exercised the estimate half of the map",
        stats.decoded
    );

    // A convolutional code terminates once per frame, so there are no codeword
    // boundaries for a display to draw — reported as zero rather than guessed.
    let mut rx = link.receiver();
    let (probe, frames) = feed_probed(&mut rx, &link.noisy(0.0, 0));
    assert_eq!(frames.len(), 1);
    let f = probe.iter().next().expect("one probe frame");
    assert_eq!(f.meta.codeword_bits, 0, "no block structure to report");
    assert_eq!(f.meta.codeword_info_bits, 0);
    assert_eq!(f.meta.constellation, ConstellationOrder::Qpsk);
}

#[test]
fn a_diverged_decoder_introduces_errors_on_both_arms() {
    // `Introduced` — arrived correct, and the decoder broke it — is the state
    // the design argues for at length, and until now nothing asserted it is
    // ever *produced*. `a_clean_link_shows_no_corrections` asserts its absence;
    // `uncorrected_bits_track_the_inner_ber` folds it in with `Uncorrected`. So
    // a `classify` that never returned it would have gone unnoticed on the
    // strength of those two.
    //
    // Asserted over a sweep, never per frame: measured across 16 seeds, 6 to 11
    // of them carry an `Introduced` bit at these noise levels, so one frame is
    // a coin flip and a sweep is not.
    //
    // Measured alongside it, and worth knowing before reading the pane: the two
    // states **co-occur exactly** — every frame with an `Uncorrected` bit also
    // had an `Introduced` one, on both arms, at every level. They are two faces
    // of one event, a block the decoder got wrong; the bits that happened to
    // arrive right are `Introduced` and the ones that did not are
    // `Uncorrected`. Their *counts* differ by orders of magnitude, though:
    // belief propagation scatters a failed codeword across all 512 positions
    // (hundreds of bits), while Viterbi errors stay localised (tens).
    for (name, link) in [
        ("LDPC", Link::new(40, 0x1D0C)),
        (
            "convolutional",
            Link::with_chain(probe_config(), conv_table(), 0, 40, 0xC0F0),
        ),
    ] {
        let stats = sweep_map_against_cber(&link, &[0.6, 0.7, 0.8], 12, false);
        assert!(
            stats.decoded >= 8,
            "{name}: the sweep must decode enough frames to mean something, got {}",
            stats.decoded
        );
        assert!(
            stats.introduced > 0,
            "{name}: {} frames decoded, {} carried an Uncorrected bit, but none \
             carried an Introduced one — either the sweep never pushed a decoder \
             into divergence or `classify` never returns that state",
            stats.decoded,
            stats.uncorrected
        );
    }
}

#[test]
fn a_non_default_coding_chain_maps_the_same_way() {
    // `reencode_inner_output` re-runs encode steps 4 and 5 — inner interleave
    // and the after-inner scramble — on the decoder's output. With the default
    // config both are no-ops, so every other test here exercises only the
    // trivial branch of each.
    //
    // The risk is the same shape as the truncation hazard: an asymmetry between
    // the forward and reverse paths that shows up in one region and looks fine
    // everywhere else. Reasoning says there is none — `interleave_bits` pads the
    // final partial block and `block_plan` mirrors the padded length, and both
    // scramble call sites run over the *same* pre-truncation buffer so the PN
    // lines up — but that is exactly the reasoning the truncation hazard
    // defeated, so assert it instead.
    let cfg = probe_config()
        .with_inner_interleaver(InterleaverKind::Block { rows: 8, cols: 16 })
        .with_outer_interleaver(InterleaverKind::Block { rows: 4, cols: 8 })
        .with_scrambler(ScramblerKind::Additive {
            poly: 0b1001,
            width: 7,
            seed: SeedMode::Fixed(0x7F),
        })
        .with_scrambler_pos(ScramblerPos::AfterInnerFec);
    let link = Link::with_chain(cfg, McsTable::default_ladder(), MCS, 40, 0x171E);

    let stats = sweep_map_against_cber(&link, &[0.0, 0.3, 0.5, 0.7], 6, false);
    assert!(
        stats.decoded >= 8,
        "the sweep must decode enough frames to mean something, got {}",
        stats.decoded
    );
    assert!(
        stats.corrected > 0,
        "{} frames checked but none had a corrected bit — an interleaved, \
         scrambled chain whose map is always empty would pass this otherwise",
        stats.decoded
    );
}

#[test]
fn probing_without_error_rates_still_maps() {
    // The probe computes its own `encode_chain_stages` rather than requiring
    // `with_error_rates`, and the two share the one encode. A hidden second
    // flag whose absence silently emptied the map would be worse than the
    // duplicated entry point — but the flag still has to gate the BER *fields*,
    // or `feed_probed` would report numbers `feed` does not.
    //
    // `probing_does_not_change_what_decodes` cannot catch that: it only ever
    // runs with error rates on, where both paths report them.
    let link = Link::new(40, 0xBE12);
    let iq = link.noisy(0.4, 0x9001);

    let mut with = link.receiver(); // with_error_rates(true)
    let (probe_with, frames_with) = feed_probed(&mut with, &iq);
    let mut without = OfdmFrameStreamDemod::new(link.cfg.clone(), link.table.clone(), link.pre);
    let mut probe_without = OfdmRxProbe::new();
    let frames_without: Vec<RxFrame> = without
        .feed_probed(&iq, &mut probe_without)
        .into_iter()
        .filter_map(Result::ok)
        .collect();

    assert_eq!(frames_with.len(), 1);
    assert_eq!(frames_without.len(), 1);
    assert!(
        frames_with[0].diagnostics.channel_ber.is_some()
            && frames_with[0].diagnostics.inner_ber.is_some(),
        "with_error_rates(true) reports both rungs"
    );
    assert_eq!(
        (
            frames_without[0].diagnostics.channel_ber,
            frames_without[0].diagnostics.inner_ber
        ),
        (None, None),
        "probing must not switch the BER rungs on behind the caller's back"
    );

    // And the map is identical either way: it does not depend on the flag.
    let a = probe_with.iter().next().expect("one probe frame");
    let b = probe_without.iter().next().expect("one probe frame");
    assert!(
        !b.correction.is_empty(),
        "the map is populated without error rates"
    );
    assert_eq!(
        b.correction, a.correction,
        "the correction map must not depend on with_error_rates"
    );
    assert_eq!(b.symbols, a.symbols);
}

// ── The reading surface ────────────────────────────────────────────────────

#[test]
fn the_two_predicates_are_independent_axes() {
    // `arrived_wrong` is the channel's half of the map, `decoder_disagreed` the
    // decoder's. They are not complements and neither implies the other — a bit
    // can be both (`Uncorrected`), either, or neither (`Clean`). Pinned as a
    // table because these two predicates *are* the map's meaning to a consumer,
    // and every use of them here and downstream reads through this pair rather
    // than re-deriving it from the variants.
    for (o, wrong, disagreed) in [
        (BitOutcome::Clean, false, false),
        (BitOutcome::Corrected, true, false),
        (BitOutcome::Uncorrected, true, true),
        (BitOutcome::Introduced, false, true),
    ] {
        assert_eq!(o.arrived_wrong(), wrong, "{o:?}.arrived_wrong()");
        assert_eq!(
            o.decoder_disagreed(),
            disagreed,
            "{o:?}.decoder_disagreed()"
        );
    }
}

#[test]
fn symbols_carry_more_bit_slots_than_the_map_covers() {
    // **The symbols and the map do not index 1:1**, and a display correlating
    // them has to know it. A payload occupies a whole number of OFDM symbols,
    // so the last one is padded past the end of the coded block: walking the
    // symbols and slicing `bps` outcomes per symbol runs off the end of the map.
    //
    // Documented on `ProbedFrame`; asserted here so a change to the padding
    // cannot quietly invalidate that note.
    let link = Link::new(40, 0x5107);
    let mut rx = link.receiver();
    let (probe, frames) = feed_probed(&mut rx, &link.noisy(0.0, 0));
    assert_eq!(frames.len(), 1);

    let f = probe.iter().next().expect("one probe frame");
    let bps = f.meta.constellation.bits_per_symbol();
    let n_data = link.cfg.carrier_plan.data_carriers().len();
    let slots = f.symbols.len() * bps;

    assert_eq!(
        f.correction.len(),
        link.plan().coded_bits,
        "the map spans exactly the coded block"
    );
    assert!(
        slots >= f.correction.len(),
        "{slots} bit-slots must cover the {}-bit map",
        f.correction.len()
    );
    assert!(
        slots - f.correction.len() < n_data * bps,
        "the padding is the tail of one OFDM symbol, not more: {} slack over \
         {} slots per symbol",
        slots - f.correction.len(),
        n_data * bps
    );
    // The direction that is always safe: every map bit has a symbol.
    assert!(f.correction.len().div_ceil(bps) <= f.symbols.len());
}

// ── The known gap, asserted ────────────────────────────────────────────────

#[test]
fn the_scattered_path_reports_no_probe_frames() {
    // `soft_demap_scattered` takes no symbol sink: DVB-T frames are decoded by
    // `waveform::dvb_t_frame`, which has its own diagnostics. So a scattered
    // link decodes normally and reports ZERO probe frames.
    //
    // Asserted rather than left implicit, so it is a known gap instead of a
    // silent one. The gap is about the *pilot* structure, not the inner code:
    // the convolutional re-encode and `Introduced` are both already exercised
    // on the static-grid path (`the_convolutional_arm_maps_the_same_way`,
    // `a_diverged_decoder_introduces_errors_on_both_arms`), so wiring the
    // scattered path up is a matter of giving its demap a symbol sink, not of
    // new decode machinery.
    use orion_sdr::waveform::dvb_t::{GuardInterval, dvb_t_mcs_table, dvb_t_scattered_config};

    let cfg = dvb_t_scattered_config(GuardInterval::G1_32, 1_000_000.0);
    let pre = probe_preamble(&cfg);
    let table = dvb_t_mcs_table();
    let modu = OfdmFrameMod::new(cfg.clone(), table.clone(), pre);

    let payload = sample_payload(184);
    let frame = FramePacket::new(FrameMetadata::new(11, 0), payload.clone());
    let mut iq = vec![C32::default(); 24];
    iq.extend_from_slice(&modu.modulate_frame(&frame, 0));
    iq.extend(vec![C32::default(); 64]);

    let mut rx = OfdmFrameStreamDemod::new(cfg, table, pre).with_error_rates(true);
    let mut probe = OfdmRxProbe::new();
    let frames: Vec<RxFrame> = rx
        .feed_probed(&iq, &mut probe)
        .into_iter()
        .filter_map(Result::ok)
        .collect();

    assert_eq!(frames.len(), 1, "the scattered frame still decodes");
    assert_eq!(frames[0].packet.payload, payload);
    assert!(
        probe.is_empty() && probe.symbols().is_empty(),
        "the scattered path is not probed — known gap"
    );
}
