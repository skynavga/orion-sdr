// Copyright (c) 2025-2026 G & R Associates LLC
// SPDX-License-Identifier: MIT OR Apache-2.0

// src/demodulate/ofdm.rs
use super::bpsk::BpskDecider;
use super::qam::{Qam16Decider, Qam64Decider, Qam256Decider, QamDecider};
use super::qpsk::QpskDecider;
use crate::core::{Block, WorkReport};
use crate::modulate::{ConstellationOrder, OfdmConfig};
use crate::multicarrier::{CarrierGrid, CyclicPrefixRemove, FftBlock, GridExtract};
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
    cp_remove: CyclicPrefixRemove,
    fft: FftBlock,
    grid_extract: GridExtract,
    // scratch, sized once in new()
    time_scratch: Vec<C32>,
    freq_scratch: Vec<C32>,
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
            cp_remove: CyclicPrefixRemove::new(n_fft, cp_len),
            fft: FftBlock::new(n_fft),
            grid_extract: GridExtract::new(grid),
            time_scratch: vec![C32::default(); n_fft],
            freq_scratch: vec![C32::default(); n_fft],
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

        let cp_wr = self
            .cp_remove
            .process(&input[..self.samples_per_symbol], &mut self.time_scratch);
        let fft_wr = self.fft.process(&self.time_scratch, &mut self.freq_scratch);
        let grid_wr = self.grid_extract.process(&self.freq_scratch, output);

        debug_assert_eq!(cp_wr.out_written, self.fft.n_fft());
        debug_assert_eq!(fft_wr.out_written, self.fft.n_fft());
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
    /// interpolation between [`CarrierGrid`]'s in-band pilot bins.
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
    pilot_bins: Vec<(usize, C32)>,
    data_bins: Vec<usize>,
}

/// Floor on `|estimate|²` before division, guarding against near-zero
/// channel nulls blowing up the corrected magnitude.
const EQUALIZER_FLOOR: f32 = 1e-6;

impl OfdmEqualizer {
    pub fn new(cfg: &OfdmConfig, method: EqualizerMethod) -> Self {
        let grid = CarrierGrid::from_plan(&cfg.carrier_plan);
        let n_fft = cfg.carrier_plan.n_fft();
        Self {
            method,
            n_fft,
            estimate: vec![C32::new(1.0, 0.0); n_fft],
            pilot_bins: grid.pilot_bins().to_vec(),
            data_bins: grid.data_bins().to_vec(),
        }
    }

    pub fn method(&self) -> EqualizerMethod {
        self.method
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
    fn interpolate_from_pilots(&mut self, received_freq: &[C32]) {
        if self.pilot_bins.is_empty() {
            return;
        }

        let mut pilots: Vec<(usize, C32)> = self
            .pilot_bins
            .iter()
            .map(|&(bin, known)| (bin, received_freq[bin] / known))
            .collect();
        pilots.sort_by_key(|&(bin, _)| bin);

        for &bin in &self.data_bins {
            self.estimate[bin] = interpolate_at(&pilots, bin, self.n_fft);
        }
        for &(bin, ratio) in &pilots {
            self.estimate[bin] = ratio;
        }
    }
}

/// Linearly interpolates the channel ratio at `bin` between the nearest
/// pilot bins on either side (circularly, wrapping across bin 0). Falls
/// back to the nearest single pilot if `bin` is outside the pilot span on
/// one side.
fn interpolate_at(pilots: &[(usize, C32)], bin: usize, n_fft: usize) -> C32 {
    if pilots.len() == 1 {
        return pilots[0].1;
    }

    // Find surrounding pilots in linear (non-circular) bin order first.
    let mut lower: Option<(usize, C32)> = None;
    let mut upper: Option<(usize, C32)> = None;
    for &(pbin, ratio) in pilots {
        if pbin <= bin {
            lower = Some((pbin, ratio));
        }
        if pbin >= bin && upper.is_none() {
            upper = Some((pbin, ratio));
        }
    }

    match (lower, upper) {
        (Some((lb, lr)), Some((ub, ur))) if lb != ub => {
            let t = (bin - lb) as f32 / (ub - lb) as f32;
            lr + (ur - lr) * t
        }
        (Some((_, r)), _) => r,
        (None, Some((_, r))) => r,
        (None, None) => {
            // No pilot below `bin`: wrap circularly to the last pilot,
            // treating it as if it were `n_fft` bins before `bin`.
            let (first_bin, first_ratio) = pilots[0];
            let (last_bin, last_ratio) = pilots[pilots.len() - 1];
            let span = (first_bin + n_fft) - last_bin;
            let t = (bin + n_fft - last_bin) as f32 / span as f32;
            last_ratio + (first_ratio - last_ratio) * t
        }
    }
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
