// Copyright (c) 2025-2026 G & R Associates LLC
// SPDX-License-Identifier: MIT OR Apache-2.0

// src/demodulate/ofdm.rs
use super::bpsk::BpskDecider;
use super::qam::{Qam16Decider, Qam64Decider, Qam256Decider, QamDecider};
use super::qpsk::QpskDecider;
use crate::core::{Block, WorkReport};
use crate::modulate::qam::{axis_scale, build_axis_table};
use crate::modulate::{ConstellationOrder, OfdmConfig};
use crate::multicarrier::{CarrierGrid, GridExtract, SymbolFft};
use crate::sync::ofdm_sync::training_symbol_freq_pattern;
use num_complex::Complex32 as C32;

/// OFDM receiver: `C32` IQ → `C32` soft symbols.
///
/// Pipeline: [`CyclicPrefixRemove`] → [`FftBlock`] → [`GridExtract`], the
/// exact inverse of `OfdmMod`'s TX chain, plus an optional scalar gain
/// correction (mirroring `BpskDemod`'s `gain`/`set_gain()`).
///
/// Explicitly scoped to this release: **known packet start, no CFO,
/// AWGN/flat channel only** — no acquisition, no equalization. Consumes
/// whole `samples_per_ofdm_symbol()`-sized IQ chunks, produces whole
/// `num_data_carriers()`-sized soft-symbol chunks; a partial trailing chunk
/// is a no-op, with no cross-call buffering.
pub struct OfdmDemod {
    samples_per_symbol: usize,
    num_data_carriers: usize,
    gain: f32,
    symbol_fft: SymbolFft,
    grid_extract: GridExtract,
}

impl OfdmDemod {
    pub fn new(cfg: &OfdmConfig) -> Self {
        let grid = CarrierGrid::from_plan(&cfg.carrier_plan);
        let n_fft = cfg.carrier_plan.n_fft();
        let cp_len = cfg.carrier_plan.cp_len();
        let num_data_carriers = grid.num_data_carriers();

        Self {
            samples_per_symbol: cfg.samples_per_ofdm_symbol(),
            num_data_carriers,
            gain: 1.0,
            symbol_fft: SymbolFft::new(n_fft, cp_len).with_window_backoff(cfg.rx_window_backoff),
            grid_extract: GridExtract::new(grid),
        }
    }

    pub fn set_gain(&mut self, g: f32) {
        self.gain = g;
    }

    pub fn num_data_carriers(&self) -> usize {
        self.num_data_carriers
    }

    pub fn samples_per_symbol(&self) -> usize {
        self.samples_per_symbol
    }
}

impl Block for OfdmDemod {
    type In = C32;
    type Out = C32;

    fn process(&mut self, input: &[C32], output: &mut [C32]) -> WorkReport {
        if input.len() < self.samples_per_symbol || output.len() < self.num_data_carriers {
            return WorkReport::default();
        }

        let freq = match self
            .symbol_fft
            .demod_symbol(&input[..self.samples_per_symbol])
        {
            Some(f) => f,
            None => return WorkReport::default(),
        };
        let grid_wr = self.grid_extract.process(freq, output);

        debug_assert_eq!(grid_wr.out_written, self.num_data_carriers);

        let g = self.gain;
        if (g - 1.0).abs() > f32::EPSILON {
            for s in output[..self.num_data_carriers].iter_mut() {
                *s = C32::new(g * s.re, g * s.im);
            }
        }

        WorkReport {
            in_read: self.samples_per_symbol,
            out_written: self.num_data_carriers,
        }
    }
}

/// Dispatches to the existing per-order hard deciders (reused verbatim, not
/// reimplemented) via a plain `match` — the receive-side mirror of
/// `OfdmMod`'s internal `MapperKind`.
enum DeciderKind {
    Bpsk(BpskDecider),
    Qpsk(QpskDecider),
    Qam16(Qam16Decider),
    Qam64(Qam64Decider),
    Qam256(Qam256Decider),
}

impl DeciderKind {
    fn new(order: ConstellationOrder) -> Self {
        match order {
            ConstellationOrder::Bpsk => DeciderKind::Bpsk(BpskDecider::new()),
            ConstellationOrder::Qpsk => DeciderKind::Qpsk(QpskDecider::new()),
            ConstellationOrder::Qam16 => DeciderKind::Qam16(QamDecider::new()),
            ConstellationOrder::Qam64 => DeciderKind::Qam64(QamDecider::new()),
            ConstellationOrder::Qam256 => DeciderKind::Qam256(QamDecider::new()),
        }
    }

    #[inline(always)]
    fn process(&mut self, input: &[C32], output: &mut [u8]) -> WorkReport {
        match self {
            DeciderKind::Bpsk(d) => d.process(input, output),
            DeciderKind::Qpsk(d) => d.process(input, output),
            DeciderKind::Qam16(d) => d.process(input, output),
            DeciderKind::Qam64(d) => d.process(input, output),
            DeciderKind::Qam256(d) => d.process(input, output),
        }
    }
}

/// OFDM hard-decision decider: `C32` soft symbol → `u8` bits, dispatching to
/// `BpskDecider`/`QpskDecider`/`QamDecider<BITS>` by `ConstellationOrder`.
///
/// Same whole-symbol-chunk-per-call contract as the other stages: consumes
/// whole `num_data_carriers()`-sized soft-symbol chunks, produces whole
/// `bits_per_ofdm_symbol()`-sized bit chunks.
pub struct OfdmDecider {
    num_data_carriers: usize,
    bits_per_ofdm_symbol: usize,
    decider: DeciderKind,
}

impl OfdmDecider {
    pub fn new(cfg: &OfdmConfig) -> Self {
        Self {
            num_data_carriers: cfg.carrier_plan.data_carriers().len(),
            bits_per_ofdm_symbol: cfg.bits_per_ofdm_symbol(),
            decider: DeciderKind::new(cfg.constellation),
        }
    }
}

impl Block for OfdmDecider {
    type In = C32;
    type Out = u8;

    fn process(&mut self, input: &[C32], output: &mut [u8]) -> WorkReport {
        if input.len() < self.num_data_carriers || output.len() < self.bits_per_ofdm_symbol {
            return WorkReport::default();
        }
        self.decider.process(
            &input[..self.num_data_carriers],
            &mut output[..self.bits_per_ofdm_symbol],
        )
    }
}

/// Per-pipeline-stage diagnostics for one demodulated OFDM packet.
///
/// `Option<f32>`/`Option<i32>` (not sentinel values) make "not yet measured
/// at this pipeline stage" explicit: fields are populated incrementally as
/// later releases add acquisition (`cfo_hz`, `timing_offset_samples`) and
/// channel estimation (`channel_mse`).
#[derive(Debug, Clone, PartialEq)]
pub struct OfdmRxFrame {
    pub bits: Vec<u8>,
    pub num_symbols: usize,
    pub evm_db: Option<f32>,
    pub cfo_hz: Option<f32>,
    pub timing_offset_samples: Option<i32>,
    pub channel_mse: Option<f32>,
}

/// Builds an [`OfdmRxFrame`] from demodulated soft symbols and their
/// corresponding hard-decided bits.
///
/// `soft_symbols` and `bits` must together span `num_symbols` OFDM symbols:
/// `soft_symbols.len() == num_symbols * num_data_carriers`, `bits.len() ==
/// num_symbols * bits_per_ofdm_symbol`. EVM is computed by re-mapping the
/// hard-decided bits back to their ideal constellation points (via the same
/// per-order mapper `OfdmMod` uses) and comparing against the soft symbols —
/// it needs only this soft/hard pair, no CFO/timing/channel machinery, so
/// it's available starting this release.
pub fn build_ofdm_rx_frame(cfg: &OfdmConfig, soft_symbols: &[C32], bits: Vec<u8>) -> OfdmRxFrame {
    let num_data_carriers = cfg.carrier_plan.data_carriers().len();
    let num_symbols = soft_symbols
        .len()
        .checked_div(num_data_carriers)
        .unwrap_or(0);

    let evm_db = evm_db(cfg, soft_symbols, &bits, num_symbols);

    OfdmRxFrame {
        bits,
        num_symbols,
        evm_db,
        cfo_hz: None,
        timing_offset_samples: None,
        channel_mse: None,
    }
}

fn evm_db(cfg: &OfdmConfig, soft_symbols: &[C32], bits: &[u8], num_symbols: usize) -> Option<f32> {
    if num_symbols == 0 || soft_symbols.is_empty() {
        return None;
    }

    let mut mapper = crate::modulate::ofdm::ideal_symbol_mapper(cfg.constellation);
    let mut ideal = vec![C32::default(); soft_symbols.len()];
    let wr = mapper.process(bits, &mut ideal);
    if wr.out_written != soft_symbols.len() {
        return None;
    }

    let mut err_energy = 0.0f64;
    let mut ref_energy = 0.0f64;
    for (s, r) in soft_symbols.iter().zip(ideal.iter()) {
        let e = s - r;
        err_energy += (e.re * e.re + e.im * e.im) as f64;
        ref_energy += (r.re * r.re + r.im * r.im) as f64;
    }

    if ref_energy <= 0.0 {
        return None;
    }

    Some((10.0 * (err_energy / ref_energy).log10()) as f32)
}

/// Selects how [`OfdmEqualizer`] derives its per-carrier channel estimate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum EqualizerMethod {
    /// Estimate once from a training symbol (via
    /// [`OfdmEqualizer::estimate_from_training_symbol`]) and hold it
    /// constant for the rest of the packet.
    ///
    /// The **default**, and for this feature's target bands (VHF–EHF,
    /// L–Ka — predominantly line-of-sight terrestrial-microwave or
    /// satellite links) not merely the simplest choice but the correct one:
    /// the channel is dominated by static or slowly-varying
    /// frequency-selective multipath, so an estimate taken once per packet
    /// stays valid for the packet.
    #[default]
    TrainingSymbolHold,
    /// Re-estimate every data symbol via frequency-domain linear
    /// interpolation between [`CarrierGrid`]'s in-band pilot bins. Data bins
    /// beyond the outermost pilots (the band edges) hold the nearest pilot's
    /// estimate rather than wrapping; with zero pilots this method is a
    /// no-op and the held (identity) estimate passes the symbol through
    /// unchanged.
    ///
    /// The explicit opt-in for genuinely time-varying channels — fast-moving
    /// aeronautical or LEO geometries with meaningful intra-packet Doppler
    /// spread — where a held estimate would go stale.
    PerSymbolPilotInterp,
}

/// Frequency-domain channel equalizer: `C32` → `C32`, operating on full
/// `n_fft`-bin vectors. Sits between [`FftBlock`] and [`GridExtract`] as its
/// own composable stage (not fused into [`OfdmDemod`]), so it can be
/// swapped or disabled independently. Divides each bin by its channel
/// estimate, with a small floor guard against near-zero estimates.
///
/// Scoped to this release: delay spread up to `cp_len` (the cyclic prefix
/// absorbs the channel's impulse response — a longer delay spread causes
/// inter-symbol interference this simple per-bin division does not model).
pub struct OfdmEqualizer {
    method: EqualizerMethod,
    n_fft: usize,
    /// Per-bin channel estimate; `1.0 + 0j` (no correction) until an
    /// estimate is available.
    estimate: Vec<C32>,
    /// Pilot bins with known TX values, kept **sorted by bin** so the per-symbol
    /// pilot interpolation can binary-search for a data bin's bracketing pilots
    /// instead of scanning the whole set.
    pilot_bins: Vec<(usize, C32)>,
    data_bins: Vec<usize>,
    /// Reused scratch for the per-symbol pilot ratios (`received/known`, in the
    /// sorted `pilot_bins` order), so `interpolate_from_pilots` allocates nothing
    /// per `process()` call.
    pilot_ratios: Vec<(usize, C32)>,
}

/// Floor on `|estimate|²` before division, guarding against near-zero
/// channel nulls blowing up the corrected magnitude.
const EQUALIZER_FLOOR: f32 = 1e-6;

impl OfdmEqualizer {
    pub fn new(cfg: &OfdmConfig, method: EqualizerMethod) -> Self {
        let grid = CarrierGrid::from_plan(&cfg.carrier_plan);
        let n_fft = cfg.carrier_plan.n_fft();
        let mut pilot_bins = grid.pilot_bins().to_vec();
        pilot_bins.sort_by_key(|&(bin, _)| bin);
        let n_pilots = pilot_bins.len();
        Self {
            method,
            n_fft,
            estimate: vec![C32::new(1.0, 0.0); n_fft],
            pilot_bins,
            data_bins: grid.data_bins().to_vec(),
            pilot_ratios: Vec::with_capacity(n_pilots),
        }
    }

    pub fn method(&self) -> EqualizerMethod {
        self.method
    }

    /// Replaces the pilot bins (and the data bins interpolated between them) for
    /// the next `process()` call, without changing `process()` itself. Used by
    /// DVB-T's scattered-pilot receiver, whose pilot/data layout rotates every
    /// symbol (`l mod 4`): the caller installs symbol `l`'s pilot set here, then
    /// runs `process()` under [`EqualizerMethod::PerSymbolPilotInterp`], which
    /// re-interpolates the channel from exactly those pilots. `pilots` are
    /// `(rustfft bin, known TX value)` pairs; `data_bins` are the bins to
    /// interpolate an estimate for (a symbol's data-carrier bins). Bins covered
    /// by neither keep their previous estimate — harmless, since the surrounding
    /// grid extractor reads only the data bins.
    ///
    /// A mirror of [`estimate_from_training_symbol`](Self::estimate_from_training_symbol):
    /// a separate pre-`process` call that sets up the estimate, not a change to
    /// the per-symbol `Block` contract.
    pub fn set_pilot_bins(&mut self, pilots: &[(usize, C32)], data_bins: &[usize]) {
        self.pilot_bins.clear();
        self.pilot_bins.extend_from_slice(pilots);
        self.pilot_bins.sort_by_key(|&(bin, _)| bin);
        self.data_bins.clear();
        self.data_bins.extend_from_slice(data_bins);
    }

    /// Computes and holds the channel estimate from a received training
    /// symbol's FFT output (`n_fft` bins), dividing by the training
    /// symbol's known frequency-domain pattern per bin. Only meaningful for
    /// [`EqualizerMethod::TrainingSymbolHold`]; a no-op under
    /// [`EqualizerMethod::PerSymbolPilotInterp`], which re-estimates from
    /// pilots on every `process()` call instead.
    pub fn estimate_from_training_symbol(&mut self, received_freq: &[C32]) {
        if self.method != EqualizerMethod::TrainingSymbolHold || received_freq.len() < self.n_fft {
            return;
        }
        let known = training_symbol_freq_pattern(self.n_fft);
        for bin in 0..self.n_fft {
            self.estimate[bin] = received_freq[bin] / known[bin];
        }
    }

    /// Re-estimates every carrier's channel by linearly interpolating (in
    /// the complex frequency domain) between the pilot bins' known-vs-
    /// received ratios. Requires at least one pilot; with zero pilots the
    /// held estimate (`1.0 + 0j` if never set) is left unchanged.
    ///
    /// `pilot_bins` is kept sorted by bin (at construction and in
    /// `set_pilot_bins`), so the per-data-bin bracketing pilots are found by
    /// binary search rather than a full scan, and the ratio buffer is reused
    /// across calls — no per-symbol allocation.
    fn interpolate_from_pilots(&mut self, received_freq: &[C32]) {
        if self.pilot_bins.is_empty() {
            return;
        }

        // Pilot ratios in the (already sorted) pilot-bin order — reused scratch.
        self.pilot_ratios.clear();
        self.pilot_ratios.extend(
            self.pilot_bins
                .iter()
                .map(|&(bin, known)| (bin, received_freq[bin] / known)),
        );

        for &bin in &self.data_bins {
            self.estimate[bin] = interpolate_at(&self.pilot_ratios, bin);
        }
        for &(bin, ratio) in &self.pilot_ratios {
            self.estimate[bin] = ratio;
        }
    }
}

/// Estimates the channel ratio at `bin` from the **bin-sorted** `pilots`:
/// linear interpolation between the two pilots bracketing `bin`, or a hold of
/// the nearest pilot when `bin` lies outside the pilot span on one side (no
/// circular wrap across bin 0 — the band edges hold their nearest pilot).
/// `pilots` must be non-empty and sorted ascending by bin.
///
/// The bracketing pilots are located by binary search (`partition_point`), so
/// this is O(log P) per data bin — keeping the equalizer's per-symbol interpolation
/// at O(data·log pilots), which matters for the DVB-T RX (1512 data × ~193 pilots
/// per symbol).
fn interpolate_at(pilots: &[(usize, C32)], bin: usize) -> C32 {
    if pilots.len() == 1 {
        return pilots[0].1;
    }

    // `hi` = index of the first pilot with bin >= `bin` (the "upper" bracket).
    let hi = pilots.partition_point(|&(pbin, _)| pbin < bin);
    if hi == 0 {
        // Every pilot is above `bin`: hold the nearest (lowest) pilot.
        return pilots[0].1;
    }
    if hi == pilots.len() {
        // Every pilot is below `bin`: hold the nearest (highest) pilot.
        return pilots[pilots.len() - 1].1;
    }
    let (ub, ur) = pilots[hi];
    if ub == bin {
        // Bin sits exactly on a pilot.
        return ur;
    }
    let (lb, lr) = pilots[hi - 1];
    let t = (bin - lb) as f32 / (ub - lb) as f32;
    lr + (ur - lr) * t
}

impl Block for OfdmEqualizer {
    type In = C32;
    type Out = C32;

    fn process(&mut self, input: &[C32], output: &mut [C32]) -> WorkReport {
        if input.len() < self.n_fft || output.len() < self.n_fft {
            return WorkReport::default();
        }

        if self.method == EqualizerMethod::PerSymbolPilotInterp {
            self.interpolate_from_pilots(&input[..self.n_fft]);
        }

        for bin in 0..self.n_fft {
            let h = self.estimate[bin];
            let mag_sq = h.norm_sqr().max(EQUALIZER_FLOOR);
            // Divide by h: multiply by conj(h) / |h|^2.
            output[bin] = input[bin] * h.conj() / mag_sq;
        }

        WorkReport {
            in_read: self.n_fft,
            out_written: self.n_fft,
        }
    }
}

// ── Soft (LLR) demapping ─────────────────────────────────────────────────────
//
// Max-log LLR extraction per constellation order: `LLR(bit) = d0² - d1²`
// where `d0`/`d1` are the distances from the received soft value to the
// nearest constellation point with that bit equal to 0/1 respectively.
// Positive LLR ⇒ bit more likely 0, matching the crate-wide LLR convention
// (see the Acronym Glossary in docs/design.md). No mandatory FEC ships in
// this release — soft LLRs are the deliverable, directly usable by an
// external/user-supplied FEC layer.

/// BPSK soft LLR for one axis value.
///
/// `BpskMapper` convention: bit 0 → (+1, 0), bit 1 → (−1, 0), so the raw
/// in-phase value directly is the max-log LLR up to a constant scale (both
/// candidate points are equidistant from any `v.re` in one dimension, so
/// `d0² - d1² = 4·v.re`).
#[inline]
pub fn bpsk_soft_llr(v: C32) -> f32 {
    4.0 * v.re
}

/// QPSK soft LLR for one symbol → `[b0_llr, b1_llr]`.
///
/// `QpskMapper` convention: b0 from the in-phase axis, b1 from quadrature,
/// each an independent BPSK-style axis scaled by `1/√2`.
#[inline]
pub fn qpsk_soft_llr(v: C32) -> [f32; 2] {
    let scale = 4.0 * std::f32::consts::SQRT_2;
    [scale * v.re, scale * v.im]
}

/// Square-QAM soft LLR for one axis value, `K = BITS/2` bits (MSB-first),
/// matching `QamMapper<BITS>`/`QamDecider<BITS>`'s Gray coding and bit
/// order. Reuses the exact same Gray-coded amplitude table those types
/// build internally (`build_axis_table`/`axis_scale` in `modulate::qam`).
pub fn qam_axis_soft_llr<const BITS: usize>(v: f32, out: &mut [f32]) {
    let k = BITS / 2;
    let m = 1usize << k;
    let table = build_axis_table(BITS, axis_scale(BITS));

    for (b, slot) in out.iter_mut().enumerate().take(k) {
        let bit_shift = k - 1 - b;
        let mut d0_sq = f32::INFINITY;
        let mut d1_sq = f32::INFINITY;
        for (gray, &level) in table.iter().enumerate().take(m) {
            let d_sq = (v - level) * (v - level);
            if (gray >> bit_shift) & 1 == 0 {
                d0_sq = d0_sq.min(d_sq);
            } else {
                d1_sq = d1_sq.min(d_sq);
            }
        }
        // Positive LLR <=> bit more likely 0 <=> closer to a bit=0 point
        // (smaller d0_sq) than any bit=1 point.
        *slot = d1_sq - d0_sq;
    }
}

/// One QAM symbol's soft LLRs: `BITS` values, `K = BITS/2` from the
/// in-phase axis then `K` from quadrature, matching `QamMapper<BITS>`'s
/// input layout.
pub fn qam_soft_llr<const BITS: usize>(v: C32) -> [f32; 8] {
    let k = BITS / 2;
    let mut out = [0.0f32; 8];
    qam_axis_soft_llr::<BITS>(v.re, &mut out[..k]);
    qam_axis_soft_llr::<BITS>(v.im, &mut out[k..2 * k]);
    out
}

/// Dispatches soft-LLR extraction by `ConstellationOrder` — the soft-output
/// mirror of `DeciderKind`'s hard-decision dispatch.
enum SoftKind {
    Bpsk,
    Qpsk,
    Qam16,
    Qam64,
    Qam256,
}

impl SoftKind {
    fn new(order: ConstellationOrder) -> Self {
        match order {
            ConstellationOrder::Bpsk => SoftKind::Bpsk,
            ConstellationOrder::Qpsk => SoftKind::Qpsk,
            ConstellationOrder::Qam16 => SoftKind::Qam16,
            ConstellationOrder::Qam64 => SoftKind::Qam64,
            ConstellationOrder::Qam256 => SoftKind::Qam256,
        }
    }

    #[inline]
    fn llrs_per_symbol(&self) -> usize {
        match self {
            SoftKind::Bpsk => 1,
            SoftKind::Qpsk => 2,
            SoftKind::Qam16 => 4,
            SoftKind::Qam64 => 6,
            SoftKind::Qam256 => 8,
        }
    }

    #[inline]
    fn extract(&self, v: C32, out: &mut [f32]) {
        match self {
            SoftKind::Bpsk => out[0] = bpsk_soft_llr(v),
            SoftKind::Qpsk => out[..2].copy_from_slice(&qpsk_soft_llr(v)),
            SoftKind::Qam16 => out[..4].copy_from_slice(&qam_soft_llr::<4>(v)[..4]),
            SoftKind::Qam64 => out[..6].copy_from_slice(&qam_soft_llr::<6>(v)[..6]),
            SoftKind::Qam256 => out[..8].copy_from_slice(&qam_soft_llr::<8>(v)[..8]),
        }
    }
}

/// OFDM soft demapper: `C32` soft symbol → `f32` LLRs, dispatching by
/// `ConstellationOrder`. A separate type from [`OfdmDecider`] (not a mode
/// flag), mirroring the crate's existing preference for distinct types per
/// distinct output contract (e.g. `Ft8Demod` vs `Ft8Codec::decode_soft`).
///
/// Same whole-symbol-chunk-per-call contract as the other stages: consumes
/// whole `num_data_carriers()`-sized soft-symbol chunks, produces whole
/// `bits_per_ofdm_symbol()`-sized LLR chunks (one `f32` per bit, matching
/// [`OfdmDecider`]'s bit-for-bit layout).
pub struct OfdmSoftDemod {
    num_data_carriers: usize,
    bits_per_ofdm_symbol: usize,
    kind: SoftKind,
}

impl OfdmSoftDemod {
    pub fn new(cfg: &OfdmConfig) -> Self {
        Self {
            num_data_carriers: cfg.carrier_plan.data_carriers().len(),
            bits_per_ofdm_symbol: cfg.bits_per_ofdm_symbol(),
            kind: SoftKind::new(cfg.constellation),
        }
    }
}

impl Block for OfdmSoftDemod {
    type In = C32;
    type Out = f32;

    fn process(&mut self, input: &[C32], output: &mut [f32]) -> WorkReport {
        if input.len() < self.num_data_carriers || output.len() < self.bits_per_ofdm_symbol {
            return WorkReport::default();
        }

        let llrs_per_symbol = self.kind.llrs_per_symbol();
        for (k, &v) in input[..self.num_data_carriers].iter().enumerate() {
            self.kind.extract(
                v,
                &mut output[k * llrs_per_symbol..(k + 1) * llrs_per_symbol],
            );
        }

        WorkReport {
            in_read: self.num_data_carriers,
            out_written: self.bits_per_ofdm_symbol,
        }
    }
}
