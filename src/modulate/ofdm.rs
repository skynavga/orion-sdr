// Copyright (c) 2025-2026 G & R Associates LLC
// SPDX-License-Identifier: MIT OR Apache-2.0

// src/modulate/ofdm.rs
use super::bpsk::BpskMapper;
use super::qam::{Qam16Mapper, Qam64Mapper, Qam256Mapper, QamMapper};
use super::qpsk::QpskMapper;
use crate::core::{Block, WorkReport};
use crate::dsp::Rotator;
use crate::multicarrier::{CarrierGrid, CarrierPlan, CyclicPrefixInsert, GridMap, IfftBlock};
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
#[derive(Debug, Clone, PartialEq)]
pub struct OfdmConfig {
    pub carrier_plan: CarrierPlan,
    pub fs: f32,
    pub rf_hz: f32,
    pub gain: f32,
    pub constellation: ConstellationOrder,
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
        }
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
