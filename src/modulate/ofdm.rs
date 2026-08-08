// Copyright (c) 2025-2026 G & R Associates LLC
// SPDX-License-Identifier: MIT OR Apache-2.0

// src/modulate/ofdm.rs
use super::bpsk::BpskMapper;
use super::qam::{Qam16Mapper, Qam64Mapper, Qam256Mapper, QamMapper};
use super::qpsk::QpskMapper;
use crate::core::{Block, WorkReport};
use crate::dsp::Rotator;
use crate::fec::{
    CrcKind, DecodeRule, HeaderFormat, InnerFec, InterleaverKind, OuterFec, ScramblerKind,
    ScramblerPos, SeedMode,
};
use crate::multicarrier::{
    CarrierGrid, CarrierPlan, CyclicPrefixInsert, GridMap, IfftBlock, TxLowpass,
};
use num_complex::Complex32 as C32;

/// Constellation order used by an OFDM data carrier's symbol mapper.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConstellationOrder {
    Bpsk,
    Qpsk,
    Qam16,
    Qam64,
    Qam256,
}

impl ConstellationOrder {
    pub fn bits_per_symbol(self) -> usize {
        match self {
            ConstellationOrder::Bpsk => 1,
            ConstellationOrder::Qpsk => 2,
            ConstellationOrder::Qam16 => 4,
            ConstellationOrder::Qam64 => 6,
            ConstellationOrder::Qam256 => 8,
        }
    }
}

/// OFDM waveform configuration: the resource grid ([`CarrierPlan`]) plus the
/// sample rate, RF/IF carrier, output gain, and data-carrier constellation
/// order shared by the transmitter ([`OfdmMod`]) and receiver.
///
/// Numerology (`n_fft`, `cp_len`, carrier layout) is caller-owned and lives
/// in `carrier_plan`; the library bakes in no standard's spacing or CP length
/// (see the numerology guidance in `docs/design.md`). `rf_hz == 0.0` selects
/// baseband output; any nonzero value upconverts via a `Rotator`.
///
/// The frame-layer fields (`outer_fec`, `inner_fec`, the two interleavers,
/// `header_format`, the two CRCs, `scrambler`, `scrambler_pos`) default to
/// "absent" and are set with the `with_*` builder methods, so the positional
/// [`OfdmConfig::new`] and the bare `OfdmMod`/`OfdmDemod` symbol pipeline are
/// unaffected by them. They configure the concatenated COFDM coding chain used
/// by the OFDM frame modulator/demodulator (see `modulate::ofdm_frame`).
#[derive(Debug, Clone, PartialEq)]
pub struct OfdmConfig {
    pub carrier_plan: CarrierPlan,
    pub fs: f32,
    pub rf_hz: f32,
    pub gain: f32,
    pub constellation: ConstellationOrder,
    // ── Frame-layer (COFDM) configuration ──
    pub outer_fec: OuterFec,
    pub inner_fec: InnerFec,
    pub outer_interleaver: InterleaverKind,
    pub inner_interleaver: InterleaverKind,
    pub header_format: HeaderFormat,
    pub payload_crc: CrcKind,
    pub header_crc: CrcKind,
    pub scrambler: ScramblerKind,
    pub scrambler_pos: ScramblerPos,
    /// Check-node rule the receiver's LDPC inner decoder uses.
    /// [`DecodeRule::SumProduct`] (the default) is exact belief propagation;
    /// [`DecodeRule::ScaledMinSum`] trades ≲0.3 dB of coding gain for ~2×
    /// decode throughput (see the R8a investigation in `docs/performance.md`).
    /// TX-only paths ignore this.
    pub ldpc_decode_rule: DecodeRule,
    /// When set, the frame layer maps/demaps payload symbols through DVB-T's
    /// four-phase **scattered-pilot** grid rotation instead of the single static
    /// grid in `carrier_plan` (see `waveform::dvb_t`). The `carrier_plan` still
    /// describes the representative phase-0 grid (its 1512 data carriers drive
    /// all the count-based bookkeeping); the physical pilot/data bins rotate per
    /// symbol underneath. Only valid for a 2K DVB-T plan. Defaults to `false`,
    /// so every non-DVB-T link is unaffected.
    pub dvb_t_scattered: bool,
    /// Receiver FFT-window back-off in samples: how far the demodulator pulls
    /// its `n_fft`-sample window *earlier* from the cyclic-prefix boundary into
    /// the guard interval (clamped to `cp_len` at use). `0` (the default) is the
    /// standard CP-boundary window. A positive value leaves guard on both sides
    /// of the useful part — receiver practice for multipath/pre-echo robustness,
    /// and the enabler for RX-transparent TX symbol windowing. **RX-only:**
    /// TX paths ignore this, so on-air output is unaffected.
    ///
    /// **Requires an equalizer.** Sliding the window by `b` multiplies every
    /// subcarrier by a linear phase ramp `exp(-j2πkb/n_fft)` (FFT shift
    /// theorem). This is transparent only on the *equalized* path (the streaming
    /// demod, or the DVB-T scattered path), where the training/pilot estimate is
    /// measured at the same back-off and divides the ramp back out. On a bare,
    /// unequalized demod (`OfdmDemod` / batch `OfdmFrameDemod` with no channel
    /// estimate) a nonzero back-off leaves the ramp uncorrected and corrupts the
    /// decode — leave it `0` there.
    pub rx_window_backoff: usize,
    /// Optional TX baseband low-pass (spectral mask) applied by the frame
    /// modulator across the **assembled** stream, after CP insertion and any
    /// symbol windowing. `None` (the default) leaves the on-air output
    /// unchanged.
    ///
    /// **TX-only field**, but not RX-indifferent: the filter is a linear
    /// channel the pilot/training equalizer absorbs, so no *decoding* change is
    /// needed, yet its group delay must land in guard the receiver discards.
    /// Pair it with [`rx_window_backoff`](Self::rx_window_backoff) — the same
    /// knob symbol windowing uses — and keep
    /// `roll_off + group_delay ≤ min(cp_len − backoff, backoff)`
    /// ([`TxLowpass::fits_guard`]).
    pub tx_lowpass: Option<TxLowpass>,
}

/// Rejects an [`OfdmConfig`] whose frame-layer settings are mutually
/// inconsistent.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum FrameConfigError {
    /// A per-frame-random scrambler seed has no way to reach the receiver when
    /// there is no in-band header to carry it.
    #[error("per-frame-random scrambler seed requires a header (header_format != NoHeader)")]
    PerFrameSeedNeedsHeader,
    /// A block interleaver was requested with a zero dimension.
    #[error("block interleaver dimensions must be nonzero")]
    ZeroInterleaverDim,
    /// A BCH outer code was requested with t = 0 (no correction).
    #[error("BCH outer code requires t >= 1")]
    ZeroBchT,
    /// A Reed–Solomon outer code has invalid dimensions.
    #[error("Reed–Solomon requires 0 < n_parity < n <= 255 with n_parity even")]
    BadRsConfig,
}

impl OfdmConfig {
    pub fn new(
        carrier_plan: CarrierPlan,
        fs: f32,
        rf_hz: f32,
        gain: f32,
        constellation: ConstellationOrder,
    ) -> Self {
        Self {
            carrier_plan,
            fs,
            rf_hz,
            gain,
            constellation,
            outer_fec: OuterFec::None,
            inner_fec: InnerFec::None,
            outer_interleaver: InterleaverKind::None,
            inner_interleaver: InterleaverKind::None,
            header_format: HeaderFormat::OrionSdr,
            payload_crc: CrcKind::Crc32,
            header_crc: CrcKind::Crc16,
            scrambler: ScramblerKind::None,
            scrambler_pos: ScramblerPos::BeforeOuterFec,
            ldpc_decode_rule: DecodeRule::SumProduct,
            dvb_t_scattered: false,
            rx_window_backoff: 0,
            tx_lowpass: None,
        }
    }

    /// Sets the sample rate (S/s). A generic builder — e.g. a DVB-T caller
    /// selects a narrowband bandwidth mode with
    /// `cfg.with_fs(NbBandwidth::Bw1MHz.fs())`.
    pub fn with_fs(mut self, fs: f32) -> Self {
        self.fs = fs;
        self
    }

    pub fn with_outer_fec(mut self, outer_fec: OuterFec) -> Self {
        self.outer_fec = outer_fec;
        self
    }

    pub fn with_inner_fec(mut self, inner_fec: InnerFec) -> Self {
        self.inner_fec = inner_fec;
        self
    }

    pub fn with_outer_interleaver(mut self, il: InterleaverKind) -> Self {
        self.outer_interleaver = il;
        self
    }

    pub fn with_inner_interleaver(mut self, il: InterleaverKind) -> Self {
        self.inner_interleaver = il;
        self
    }

    pub fn with_header_format(mut self, header_format: HeaderFormat) -> Self {
        self.header_format = header_format;
        self
    }

    pub fn with_payload_crc(mut self, crc: CrcKind) -> Self {
        self.payload_crc = crc;
        self
    }

    pub fn with_header_crc(mut self, crc: CrcKind) -> Self {
        self.header_crc = crc;
        self
    }

    pub fn with_scrambler(mut self, scrambler: ScramblerKind) -> Self {
        self.scrambler = scrambler;
        self
    }

    pub fn with_scrambler_pos(mut self, pos: ScramblerPos) -> Self {
        self.scrambler_pos = pos;
        self
    }

    /// Selects the LDPC inner-decoder check-node rule (receiver side). Defaults
    /// to [`DecodeRule::SumProduct`]; pass [`DecodeRule::ScaledMinSum`] (α ≈
    /// 0.75) for ~2× decode throughput at a ≲0.3 dB coding-gain cost.
    pub fn with_ldpc_decode_rule(mut self, rule: DecodeRule) -> Self {
        self.ldpc_decode_rule = rule;
        self
    }

    /// Enables DVB-T four-phase scattered-pilot grid rotation in the frame layer
    /// (see [`dvb_t_scattered`](Self::dvb_t_scattered)). The `carrier_plan` must
    /// be a 2K DVB-T phase-0 plan (1512 data carriers).
    pub fn with_dvb_t_scattered(mut self, scattered: bool) -> Self {
        self.dvb_t_scattered = scattered;
        self
    }

    /// Sets the receiver FFT-window back-off in samples (see
    /// [`rx_window_backoff`](Self::rx_window_backoff)). RX-only; TX output is
    /// unaffected. Clamped to `cp_len` where the window is selected.
    pub fn with_rx_window_backoff(mut self, backoff: usize) -> Self {
        self.rx_window_backoff = backoff;
        self
    }

    /// Enables TX symbol windowing with a `roll_off`-sample raised-cosine taper
    /// per symbol edge (see
    /// [`CarrierPlan::with_window_roll_off`](crate::multicarrier::CarrierPlan::with_window_roll_off)).
    /// `0` disables it (the default). The taper reduces out-of-band emission but
    /// is only RX-transparent when paired with a compatible
    /// [`rx_window_backoff`](Self::rx_window_backoff) (`roll_off ≤ cp_len/2` with
    /// back-off `cp_len/2` is the transparent operating point).
    ///
    /// See [`with_symbol_window_beta_guard`](Self::with_symbol_window_beta_guard)
    /// and [`with_symbol_window_beta_tu`](Self::with_symbol_window_beta_tu) to
    /// specify the roll-off as a fraction instead of raw samples.
    pub fn with_symbol_window(mut self, roll_off: usize) -> Self {
        self.carrier_plan = self.carrier_plan.with_window_roll_off(roll_off);
        self
    }

    /// Enables TX symbol windowing with a roll-off given as a fraction of the
    /// **guard** (cyclic prefix): `roll_off = round(beta * cp_len)`. `beta` in
    /// `0.0..=0.5`; `beta = 0.5` is the maximum RX-transparent taper
    /// (`roll_off = cp_len/2`, paired with `rx_window_backoff = cp_len/2`). This
    /// convention makes the transparency budget explicit, since the taper is
    /// bounded by half the guard. Clamped to `[0, 0.5]`.
    pub fn with_symbol_window_beta_guard(self, beta: f32) -> Self {
        let cp_len = self.carrier_plan.cp_len();
        let roll_off = (beta.clamp(0.0, 0.5) * cp_len as f32).round() as usize;
        self.with_symbol_window(roll_off)
    }

    /// Enables TX symbol windowing with a roll-off given as a fraction of the
    /// **useful symbol** `Tu` (`n_fft`): `roll_off = round(beta * n_fft)` — the
    /// convention used by DVB-family windowing tables (which express roll-offs
    /// relative to `Tu`). Note the resulting `roll_off` must still satisfy the
    /// transparency bound `roll_off ≤ cp_len/2` for a matched back-off to keep
    /// the decode transparent; a larger `beta` shapes the spectrum more but is
    /// only transparent if the guard is long enough. Clamped so `2*roll_off` does
    /// not exceed the symbol length.
    pub fn with_symbol_window_beta_tu(self, beta: f32) -> Self {
        let n_fft = self.carrier_plan.n_fft();
        let roll_off = (beta.max(0.0) * n_fft as f32).round() as usize;
        self.with_symbol_window(roll_off)
    }

    /// Enables the TX baseband low-pass (spectral mask) applied by
    /// [`OfdmFrameMod::modulate_frame`](crate::modulate::OfdmFrameMod::modulate_frame)
    /// across the assembled stream (see [`tx_lowpass`](Self::tx_lowpass)).
    /// Off by default.
    ///
    /// Unlike symbol windowing this is *not* bounded by the windowing ceiling —
    /// it attenuates the skirt directly in the frequency domain, so its gain
    /// stacks on top. It changes nothing about how the receiver *decodes*, but
    /// its group delay shares the guard budget with any symbol taper and needs
    /// a nonzero [`rx_window_backoff`](Self::rx_window_backoff) to land in:
    /// check [`TxLowpass::fits_guard`].
    pub fn with_tx_lowpass(mut self, lowpass: TxLowpass) -> Self {
        self.tx_lowpass = Some(lowpass);
        self
    }

    /// Convenience form of [`with_tx_lowpass`](Self::with_tx_lowpass) that reads
    /// the occupied band edge straight off the carrier plan and centres the
    /// filter's transition in the unoccupied band above it
    /// ([`TxLowpass::for_null_band`]). `num_taps` stays the caller's choice
    /// because it is what the cyclic-prefix budget constrains;
    /// [`TxLowpass::taps_for_null_band`] suggests a length.
    pub fn with_tx_lowpass_null_band(self, num_taps: usize, stopband_db: f32) -> Self {
        let lowpass = TxLowpass::for_null_band(
            self.carrier_plan.n_fft(),
            self.carrier_plan.occupied_half_carriers(),
            num_taps,
            stopband_db,
        );
        self.with_tx_lowpass(lowpass)
    }

    /// Validates the frame-layer configuration. Returns `Ok(())` for the bare
    /// (no-FEC, no-frame) defaults.
    pub fn validate(&self) -> Result<(), FrameConfigError> {
        // A per-frame-random seed needs a header block to carry it to the RX.
        // Only OrionSdr has one; NoHeader and DvbTps do not (DvbTps signals via
        // TPS, which does not carry a scrambler seed).
        if let ScramblerKind::Additive {
            seed: SeedMode::PerFrameRandom,
            ..
        } = self.scrambler
            && !self.header_format.has_header_block()
        {
            return Err(FrameConfigError::PerFrameSeedNeedsHeader);
        }
        for il in [self.outer_interleaver, self.inner_interleaver] {
            match il {
                InterleaverKind::Block { rows, cols } if rows == 0 || cols == 0 => {
                    return Err(FrameConfigError::ZeroInterleaverDim);
                }
                InterleaverKind::Convolutional { branches, depth }
                    if branches == 0 || depth == 0 =>
                {
                    return Err(FrameConfigError::ZeroInterleaverDim);
                }
                _ => {}
            }
        }
        if let OuterFec::Bch { t } = self.outer_fec
            && t == 0
        {
            return Err(FrameConfigError::ZeroBchT);
        }
        if let OuterFec::ReedSolomon { n, n_parity } = self.outer_fec
            && (n == 0 || n > 255 || n_parity == 0 || n_parity >= n || n_parity % 2 != 0)
        {
            return Err(FrameConfigError::BadRsConfig);
        }
        Ok(())
    }

    pub fn bits_per_ofdm_symbol(&self) -> usize {
        self.carrier_plan.data_carriers().len() * self.constellation.bits_per_symbol()
    }

    pub fn samples_per_ofdm_symbol(&self) -> usize {
        self.carrier_plan.n_fft() + self.carrier_plan.cp_len()
    }
}

/// Dispatches to the existing per-order symbol mappers (reused verbatim, not
/// reimplemented) via a plain `match` — no `dyn` dispatch in the hot loop.
///
/// `pub(crate)` so `demodulate::ofdm` can reuse it to compute EVM (mapping
/// hard-decided bits back to their ideal constellation points) without
/// duplicating the per-order dispatch.
pub(crate) enum MapperKind {
    Bpsk(BpskMapper),
    Qpsk(QpskMapper),
    Qam16(Qam16Mapper),
    Qam64(Qam64Mapper),
    Qam256(Qam256Mapper),
}

impl MapperKind {
    fn new(order: ConstellationOrder) -> Self {
        match order {
            ConstellationOrder::Bpsk => MapperKind::Bpsk(BpskMapper::new()),
            ConstellationOrder::Qpsk => MapperKind::Qpsk(QpskMapper::new()),
            ConstellationOrder::Qam16 => MapperKind::Qam16(QamMapper::new()),
            ConstellationOrder::Qam64 => MapperKind::Qam64(QamMapper::new()),
            ConstellationOrder::Qam256 => MapperKind::Qam256(QamMapper::new()),
        }
    }

    #[inline(always)]
    pub(crate) fn process(&mut self, input: &[u8], output: &mut [C32]) -> WorkReport {
        match self {
            MapperKind::Bpsk(m) => m.process(input, output),
            MapperKind::Qpsk(m) => m.process(input, output),
            MapperKind::Qam16(m) => m.process(input, output),
            MapperKind::Qam64(m) => m.process(input, output),
            MapperKind::Qam256(m) => m.process(input, output),
        }
    }
}

/// Constructs the ideal-symbol mapper for `order`, for crate-internal reuse
/// (e.g. EVM computation in `demodulate::ofdm`).
pub(crate) fn ideal_symbol_mapper(order: ConstellationOrder) -> MapperKind {
    MapperKind::new(order)
}

/// OFDM transmitter: `u8` bits → `C32` IQ.
///
/// Pipeline: bits → symbol mapper (BPSK/QPSK/QAM, order given by
/// `OfdmConfig::constellation`) → [`GridMap`] → [`IfftBlock`] →
/// [`CyclicPrefixInsert`] → optional [`Rotator`] (`rf_hz == 0.0` ⇒ baseband
/// passthrough, exactly like `BpskMod`).
///
/// Consumes whole `bits_per_ofdm_symbol()`-sized bit chunks, produces whole
/// `samples_per_ofdm_symbol()`-sized IQ chunks; a partial trailing chunk is
/// a no-op, with no cross-call buffering. All intermediate buffers are
/// struct fields sized once in `new()`.
pub struct OfdmMod {
    bits_per_symbol: usize,
    samples_per_symbol: usize,
    gain: f32,
    rf_hz: f32,
    mapper: MapperKind,
    grid_map: GridMap,
    ifft: IfftBlock,
    cp_insert: CyclicPrefixInsert,
    rot: Rotator,
    // scratch, sized once in new()
    syms_scratch: Vec<C32>,
    freq_scratch: Vec<C32>,
    time_scratch: Vec<C32>,
    cp_scratch: Vec<C32>,
}

impl OfdmMod {
    pub fn new(cfg: &OfdmConfig) -> Self {
        let grid = CarrierGrid::from_plan(&cfg.carrier_plan);
        let n_fft = cfg.carrier_plan.n_fft();
        let cp_len = cfg.carrier_plan.cp_len();
        let num_data = grid.num_data_carriers();

        Self {
            bits_per_symbol: cfg.bits_per_ofdm_symbol(),
            samples_per_symbol: cfg.samples_per_ofdm_symbol(),
            gain: cfg.gain,
            rf_hz: cfg.rf_hz,
            mapper: MapperKind::new(cfg.constellation),
            grid_map: GridMap::new(grid),
            ifft: IfftBlock::new(n_fft),
            cp_insert: CyclicPrefixInsert::new(n_fft, cp_len),
            rot: Rotator::new(cfg.rf_hz, cfg.fs),
            syms_scratch: vec![C32::default(); num_data],
            freq_scratch: vec![C32::default(); n_fft],
            time_scratch: vec![C32::default(); n_fft],
            cp_scratch: vec![C32::default(); n_fft + cp_len],
        }
    }

    pub fn set_gain(&mut self, g: f32) {
        self.gain = g;
    }

    /// Convenience wrapper mirroring `Ft8Mod::modulate()`: modulates all of
    /// `bits`, zero-padding a final partial symbol.
    pub fn modulate(&mut self, bits: &[u8]) -> Vec<C32> {
        let bps = self.bits_per_symbol;
        if bps == 0 {
            return Vec::new();
        }
        let n_symbols = bits.len().div_ceil(bps);
        let mut padded = bits.to_vec();
        padded.resize(n_symbols * bps, 0);

        let mut out = vec![C32::default(); n_symbols * self.samples_per_symbol];
        let mut bits_read = 0usize;
        let mut samples_written = 0usize;
        while bits_read < padded.len() {
            let wr = self.process(
                &padded[bits_read..],
                &mut out[samples_written..samples_written + self.samples_per_symbol],
            );
            if wr.in_read == 0 {
                break;
            }
            bits_read += wr.in_read;
            samples_written += wr.out_written;
        }
        out
    }
}

impl Block for OfdmMod {
    type In = u8;
    type Out = C32;

    fn process(&mut self, input: &[u8], output: &mut [C32]) -> WorkReport {
        if input.len() < self.bits_per_symbol || output.len() < self.samples_per_symbol {
            return WorkReport::default();
        }

        let map_wr = self
            .mapper
            .process(&input[..self.bits_per_symbol], &mut self.syms_scratch);
        let grid_wr = self
            .grid_map
            .process(&self.syms_scratch, &mut self.freq_scratch);
        let ifft_wr = self
            .ifft
            .process(&self.freq_scratch, &mut self.time_scratch);
        let cp_wr = self
            .cp_insert
            .process(&self.time_scratch, &mut self.cp_scratch);

        debug_assert_eq!(map_wr.in_read, self.bits_per_symbol);
        debug_assert_eq!(grid_wr.out_written, self.ifft.n_fft());
        debug_assert_eq!(ifft_wr.out_written, self.ifft.n_fft());
        debug_assert_eq!(cp_wr.out_written, self.samples_per_symbol);

        let g = self.gain;
        let n = self.samples_per_symbol;
        if self.rf_hz != 0.0 {
            for (out, &s) in output[..n].iter_mut().zip(self.cp_scratch[..n].iter()) {
                let r = self.rot.next();
                *out = C32::new(
                    g * s.re.mul_add(r.re, -s.im * r.im),
                    g * s.im.mul_add(r.re, s.re * r.im),
                );
            }
        } else {
            for (out, &s) in output[..n].iter_mut().zip(self.cp_scratch[..n].iter()) {
                *out = C32::new(g * s.re, g * s.im);
            }
        }

        WorkReport {
            in_read: self.bits_per_symbol,
            out_written: n,
        }
    }
}
