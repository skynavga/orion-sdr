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
/// at this pipeline stage" explicit: a field is `None` where the stage that
/// would produce it did not run, not where the measurement was zero.
#[derive(Debug, Clone, PartialEq)]
pub struct OfdmRxFrame {
    pub bits: Vec<u8>,
    pub num_symbols: usize,
    pub evm_db: Option<f32>,
    pub cfo_hz: Option<f32>,
    pub timing_offset_samples: Option<i32>,
    pub channel_mse: Option<f32>,
    /// Normalized Schmidl & Cox timing-metric score in `[0, 1]` for the
    /// candidate this frame was acquired from — the streaming receiver's
    /// confidence that it found a real preamble rather than noise.
    ///
    /// `None` on the batch path, which is handed a frame body that has already
    /// been located and so never runs acquisition.
    pub sync_score: Option<f32>,
    /// Per-bin channel estimate `H[k] = received[k] / known[k]`, in natural
    /// FFT bin order, measured from the training symbol at the same window
    /// back-off the data symbols use.
    ///
    /// Populated only when the receiver was built with
    /// [`OfdmFrameStreamDemod::with_channel_estimate`], since it costs an
    /// `n_fft`-sized allocation per frame and most callers do not want it.
    ///
    /// This is the *channel*, not the raw received training bins: the known
    /// pattern is crate-internal, so a caller could not divide it out. A power
    /// delay profile — and from it delay spread and whether echoes fall inside
    /// the guard — is the inverse FFT of this.
    pub channel_estimate: Option<Vec<C32>>,
    /// Whether every **inner**-FEC block of the payload converged, reported
    /// separately from the outer stage.
    ///
    /// `false` here with a frame that still decoded is a link running hot but
    /// delivering — errors the inner code corrected never reached the outer
    /// one. That distinction is the whole point of separating the stages; a
    /// folded flag cannot express it.
    pub inner_fec_ok: Option<bool>,
    /// Whether every **outer**-FEC block of the payload decoded.
    pub outer_fec_ok: Option<bool>,
    /// Bit error rate at the **channel's output** — the inner decoder's input.
    /// The classic "pre-FEC BER" / `CBER`.
    ///
    /// Measured against a re-encode of the recovered frame, so it needs no
    /// prior knowledge of the payload and works over the air. `None` unless
    /// the receiver was built with
    /// [`OfdmFrameStreamDemod::with_error_rates`], and only ever present on
    /// frames that decoded — there is no ground truth for one that did not.
    pub channel_ber: Option<f32>,
    /// Bit error rate at the **inner decoder's output**, before the outer
    /// decoder — `IBER`, the rung the inner code's coding gain shows up in.
    ///
    /// Same provenance and same conditions as
    /// [`channel_ber`](Self::channel_ber).
    pub inner_ber: Option<f32>,
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
        sync_score: None,
        channel_estimate: None,
        inner_fec_ok: None,
        outer_fec_ok: None,
        channel_ber: None,
        inner_ber: None,
    }
}

pub(crate) fn evm_db(
    cfg: &OfdmConfig,
    soft_symbols: &[C32],
    bits: &[u8],
    num_symbols: usize,
) -> Option<f32> {
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
/// estimate, erasing (zeroing) any bin whose estimate falls under a small
/// floor rather than dividing by a null.
///
/// Scoped to this release: delay spread up to `cp_len` (the cyclic prefix
/// absorbs the channel's impulse response — a longer delay spread causes
/// inter-symbol interference this simple per-bin division does not model).
pub struct OfdmEqualizer {
    method: EqualizerMethod,
    n_fft: usize,
    /// Per-bin channel estimate, written by
    /// [`interpolate_from_pilots`](OfdmEqualizer::interpolate_from_pilots) and
    /// consumed by `process` under [`EqualizerMethod::PerSymbolPilotInterp`].
    /// `1.0 + 0j` (no correction) until an estimate is available.
    estimate: Vec<C32>,
    /// Per-bin equalizer coefficient — `conj(h) / |h|²`, or `0` where the bin is
    /// erased — under [`EqualizerMethod::TrainingSymbolHold`], where it is the
    /// state `process` actually reads. `1.0 + 0j` (pass through) until
    /// [`set_channel`](OfdmEqualizer::set_channel) writes one.
    ///
    /// **The coefficient is cached exactly where it is reused.** `TrainingSymbolHold`
    /// measures once per *packet* and applies the result to every symbol, so
    /// folding the magnitude, the divide and the erasure compare out of the
    /// per-symbol loop leaves it a bare complex multiply: measured at
    /// `n_fft = 2048`, 3337 Msps against 2691 for the old clamp-and-divide.
    /// That also makes the erasure rule free — written inside the loop it cost
    /// 6% as a compare-and-select (2521 Msps) and 65% as a branch (919), the
    /// latter by defeating auto-vectorization outright.
    ///
    /// `PerSymbolPilotInterp` re-estimates every occupied bin on every symbol,
    /// so there is nothing to reuse and a cache is pure overhead — caching
    /// there measured 144 Msps against 158 for computing inline. It reads
    /// `estimate` and applies the same rule in the loop body instead.
    weight: Vec<C32>,
    /// The layout built from the config's own grid at construction time —
    /// what [`interpolate_from_pilots`](Self::interpolate_from_pilots) reads
    /// when `active_phase` is `None`, i.e. no caller has ever called
    /// [`set_pilot_bins`](Self::set_pilot_bins). The only layout a
    /// non-rotating (generic) caller ever needs.
    default_layout: PilotLayout,
    /// Cached pilot/data layouts for [`EqualizerMethod::PerSymbolPilotInterp`],
    /// one per distinct `phase` a caller has passed to
    /// [`set_pilot_bins`](Self::set_pilot_bins). A caller whose pilot/data bin
    /// *positions* repeat (DVB-T's four-phase scattered-pilot rotation) installs
    /// a phase once and gets an O(1)-per-carrier lookup on every subsequent
    /// repeat, instead of rediscovering the bracket structure by search on every
    /// symbol. Small and scanned linearly: nothing in this crate uses more than
    /// 4 distinct phases, and `OfdmEqualizer` itself does not need to know that
    /// number.
    layouts: Vec<(usize, PilotLayout)>,
    /// Which layout `process` reads under
    /// [`EqualizerMethod::PerSymbolPilotInterp`]: `None` means `default_layout`
    /// (no [`set_pilot_bins`](Self::set_pilot_bins) call yet), `Some(phase)`
    /// means the matching entry in `layouts` — the phase most recently
    /// installed.
    active_phase: Option<usize>,
}

/// A cached bracket structure for one pilot/data-bin layout: the layout's pilot
/// bins (sorted, with known TX values) and, for every data bin, which two
/// pilots bracket it and the lerp weight between them — precomputed once, in a
/// single linear sweep over the sorted pilot/data bins, rather than
/// rediscovered by search on every symbol. Has no notion of *which* phase (or
/// whether there even is one) it belongs to — that's the caller's bookkeeping
/// (`OfdmEqualizer::layouts`/`default_layout`), kept separate so this type
/// can't itself be asked for a nonexistent phase.
struct PilotLayout {
    /// This layout's pilot bins, sorted ascending, with their known TX values.
    pilot_bins: Vec<(usize, C32)>,
    /// `(data_bin, bracket)` for every data bin this phase covers, in the order
    /// they were built (ascending by bin) — not necessarily the caller's
    /// `data_bins` order, since `interpolate_from_pilots` only ever indexes
    /// `estimate` by bin, never walks this in order.
    brackets: Vec<(usize, Bracket)>,
}

/// The two pilots bracketing one data bin, and the lerp weight between them.
/// `left_bin == right_bin` (weight irrelevant) encodes both edge-hold cases — a
/// data bin outside the pilot span, or a single-pilot layout — as well as a bin
/// that lands exactly on a pilot: `estimate[left] + (estimate[right] -
/// estimate[left]) * t` collapses to `estimate[left]` whenever the two bins
/// coincide, regardless of `t`.
#[derive(Clone, Copy)]
struct Bracket {
    left_bin: usize,
    right_bin: usize,
    t: f32,
}

impl PilotLayout {
    /// Builds one phase's layout: sorts `pilots` and (separately) `data_bins`
    /// by bin — `O((pilots + data)·log(pilots + data))`, since neither
    /// arrives pre-sorted — then walks both alongside each other in a single
    /// two-pointer merge pass, `O(pilots + data)`, to bracket every data bin.
    /// The sort dominates the one-time build cost; what the merge avoids is a
    /// **search** (binary or otherwise) *per data bin*, which is what ran on
    /// every symbol before this cache existed.
    fn build(pilots: &[(usize, C32)], data_bins: &[usize]) -> Self {
        let mut pilot_bins = pilots.to_vec();
        pilot_bins.sort_by_key(|&(bin, _)| bin);

        let mut sorted_data = data_bins.to_vec();
        sorted_data.sort_unstable();

        let mut brackets = Vec::with_capacity(sorted_data.len());
        if !pilot_bins.is_empty() {
            // `pi` only ever advances across the whole `sorted_data` walk below,
            // so the `while`'s total work over the loop is O(pilot_bins.len()),
            // not O(pilot_bins.len()) per data bin.
            let mut pi = 0usize;
            for db in sorted_data {
                while pi + 1 < pilot_bins.len() && pilot_bins[pi + 1].0 <= db {
                    pi += 1;
                }
                let lb = pilot_bins[pi].0;
                let bracket = if lb >= db || pi + 1 == pilot_bins.len() {
                    // Exact hit, or `db` lies outside the pilot span on one
                    // side: hold the nearest pilot (`lb`).
                    Bracket {
                        left_bin: lb,
                        right_bin: lb,
                        t: 0.0,
                    }
                } else {
                    let ub = pilot_bins[pi + 1].0;
                    let t = (db - lb) as f32 / (ub - lb) as f32;
                    Bracket {
                        left_bin: lb,
                        right_bin: ub,
                        t,
                    }
                };
                brackets.push((db, bracket));
            }
        }

        Self {
            pilot_bins,
            brackets,
        }
    }
}

/// Threshold on `|estimate|²` below which a bin is treated as **erased** rather
/// than equalized: the output is zeroed instead of divided through.
///
/// A null is not a small channel, it is an absent one. Clamping `|h|²` up to
/// this value and dividing anyway — which is what this constant used to do —
/// turns a null into a gain of up to `1e6`, so the one bin carrying no
/// information becomes the loudest thing in the symbol and the demapper reads a
/// large random value as a *confident* decision. Zero is the honest answer: it
/// lands on the constellation's centroid, every LLR it produces is ~0, and the
/// FEC gets an erasure, which is the impairment it is best at absorbing.
///
/// The scale is absolute because the training pattern is unit-magnitude, so
/// `h ≈ gain · H` and `1e-3` in magnitude is 60 dB below a unit channel — a
/// genuine null, not a quiet link. A receiver running its channel estimate that
/// far below unity should scale its input up rather than rely on this.
const EQUALIZER_FLOOR: f32 = 1e-6;

impl OfdmEqualizer {
    pub fn new(cfg: &OfdmConfig, method: EqualizerMethod) -> Self {
        let grid = CarrierGrid::from_plan(&cfg.carrier_plan);
        let n_fft = cfg.carrier_plan.n_fft();
        Self {
            method,
            n_fft,
            // Identity channel → identity coefficient: pass every bin through
            // unchanged until an estimator says otherwise.
            estimate: vec![C32::new(1.0, 0.0); n_fft],
            weight: vec![C32::new(1.0, 0.0); n_fft],
            // The config's own grid pilots — the only layout a non-rotating
            // (generic) caller ever needs, since it never calls
            // `set_pilot_bins`. A rotating caller (DVB-T) installs its own
            // phase-0 layout before its first `process()` call, which — with
            // `active_phase` still `None` at that point — is a genuine cache
            // miss, so it builds fresh rather than reusing this one.
            default_layout: PilotLayout::build(grid.pilot_bins(), grid.data_bins()),
            layouts: Vec::new(),
            active_phase: None,
        }
    }

    pub fn method(&self) -> EqualizerMethod {
        self.method
    }

    /// Installs the pilot bins (and the data bins interpolated between them) for
    /// the next `process()` call under [`EqualizerMethod::PerSymbolPilotInterp`],
    /// keyed by a caller-supplied `phase`. `pilots` are `(rustfft bin, known TX
    /// value)` pairs; `data_bins` are the bins to interpolate an estimate for.
    /// Bins covered by neither keep their previous estimate — harmless, since the
    /// surrounding grid extractor reads only the data bins.
    ///
    /// The bracket structure (which two pilots bound each data bin, and the lerp
    /// weight between them) is built once per distinct `phase` and cached: a
    /// second call with a `phase` already seen reuses it, ignoring `pilots`/
    /// `data_bins` entirely, since a repeated phase implies the same bin
    /// *positions* — only the pilots' *values*, read fresh from `received_freq`
    /// in [`interpolate_from_pilots`](Self::interpolate_from_pilots), can change
    /// between calls. DVB-T's scattered-pilot receiver installs symbol `l`'s
    /// `phase = l mod 4` here before every `process()` call: the same four
    /// layouts get built once per frame decode, then reused for that frame's
    /// remaining symbols.
    ///
    /// A mirror of [`estimate_from_training_symbol`](Self::estimate_from_training_symbol):
    /// a separate pre-`process` call that sets up the estimate, not a change to
    /// the per-symbol `Block` contract.
    pub fn set_pilot_bins(&mut self, phase: usize, pilots: &[(usize, C32)], data_bins: &[usize]) {
        if !self.layouts.iter().any(|&(p, _)| p == phase) {
            self.layouts
                .push((phase, PilotLayout::build(pilots, data_bins)));
        }
        self.active_phase = Some(phase);
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
            self.set_channel(bin, received_freq[bin] / known[bin]);
        }
    }

    /// Records the channel estimate `h` for one bin as the equalizer
    /// coefficient `conj(h) / |h|²` — or zero where `|h|²` falls under
    /// [`EQUALIZER_FLOOR`] and the bin is erased.
    #[inline]
    fn set_channel(&mut self, bin: usize, h: C32) {
        self.weight[bin] = equalizer_weight(h);
    }

    /// Re-estimates every data carrier's channel from the active layout's
    /// cached bracket structure — `default_layout` if
    /// [`set_pilot_bins`](Self::set_pilot_bins) was never called, otherwise
    /// its cached entry for `active_phase` (always present:
    /// `set_pilot_bins` inserts before activating a phase, so that lookup
    /// failing is unreachable and only guarded defensively): each pilot's
    /// ratio (`received/known`) is written directly into `estimate` at its
    /// own bin, then every data bin's estimate is a single lerp between its
    /// two cached bracketing pilots — an O(1) lookup per carrier, since the
    /// bracket relationship was already resolved when the layout was built.
    /// A no-op when the active layout has no pilots (the held estimate —
    /// `1.0 + 0j` if never set — is left unchanged).
    fn interpolate_from_pilots(&mut self, received_freq: &[C32]) {
        let layout = match self.active_phase {
            Some(phase) => match self.layouts.iter().find(|&&(p, _)| p == phase) {
                Some((_, layout)) => layout,
                None => return,
            },
            None => &self.default_layout,
        };
        if layout.pilot_bins.is_empty() {
            return;
        }

        for &(bin, known) in &layout.pilot_bins {
            self.estimate[bin] = received_freq[bin] / known;
        }
        for &(data_bin, br) in &layout.brackets {
            let l = self.estimate[br.left_bin];
            let r = self.estimate[br.right_bin];
            self.estimate[data_bin] = l + (r - l) * br.t;
        }
    }
}

/// The equalizer coefficient for a channel estimate: `conj(h) / |h|²`, or zero
/// where `|h|²` falls under [`EQUALIZER_FLOOR`] and the bin is erased.
///
/// Free-standing so the cached (`TrainingSymbolHold`) and inline
/// (`PerSymbolPilotInterp`) paths state the erasure rule once between them.
#[inline(always)]
fn equalizer_weight(h: C32) -> C32 {
    let mag_sq = h.norm_sqr();
    if mag_sq >= EQUALIZER_FLOOR {
        h.conj() / mag_sq
    } else {
        C32::default()
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
            // Every bin was just re-estimated, so there is no cached
            // coefficient to reuse — derive it in the loop (see `weight`).
            for bin in 0..self.n_fft {
                output[bin] = input[bin] * equalizer_weight(self.estimate[bin]);
            }
        } else {
            // One complex multiply per bin. The magnitude, the divide and the
            // erasure compare were all folded into `weight` when the packet's
            // estimate was taken.
            for bin in 0..self.n_fft {
                output[bin] = input[bin] * self.weight[bin];
            }
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
