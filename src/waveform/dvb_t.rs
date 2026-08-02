// Copyright (c) 2026 G & R Associates LLC
// SPDX-License-Identifier: MIT OR Apache-2.0

// src/waveform/dvb_t.rs
//
// DVB-T (and narrowband DVB-T for amateur DATV) waveform assembly. This module
// holds the DVB-T-specific parameters and coding pieces that the generic
// `fec/`/`multicarrier/`/`modulate/` primitives are configured with. It grows
// across the NB-DVB-T phases; this first piece is the energy-dispersal whitener.
//
// ENERGY DISPERSAL (ETSI EN 300 744 §4.3.1). DVB-T whitens the transport stream
// with a PRBS from the polynomial 1 + X^14 + X^15, the 15-bit register
// initialized to `100101010000000`. Two conventions distinguish it from the
// crate's generic `PnScrambler`/`PnScramblerStream` (which are LSB-first with a
// low-bit output tap):
//
//   • the PRBS output is taken from the last register stage and MSB-first per
//     byte — the first PRBS bit XORs the MSB of the first payload byte;
//   • the first 8 PRBS output bits are `0000 0011` (= 0x03), a known-answer
//     anchor from the standard.
//
// SCOPE (Phase 1): this is the *whitener algorithm* — the continuous PRBS applied
// to the payload byte stream. The full DVB-T energy dispersal also re-inits the
// PRBS every 8 transport packets and inverts the first MPEG-2 sync byte
// (0x47 -> 0xB8, sync bytes skipped-but-clocked); that 188-byte-TS-packet framing
// lands with the TS-packet payload layer in a later phase. Applied continuously,
// this whitener is self-inverse: running a byte stream through twice from the
// same init recovers it.

/// DVB-T energy-dispersal PRBS register initialization: `100101010000000`
/// (stage 1 = MSB), i.e. `0b100_1010_1000_0000` over the low 15 bits.
pub const DVB_T_PRBS_INIT: u16 = 0b100_1010_1000_0000;

/// A DVB-T energy-dispersal whitener: the standard's 1 + X^14 + X^15 PRBS,
/// MSB-first, carried as streaming state across [`feed`](Self::feed) calls.
/// Self-inverse (XOR of a data-independent PRBS), so the same whitener recovers a
/// stream it scrambled once it is [`reset`](Self::reset).
///
/// This is the whitener algorithm only (see the module header): the 8-packet
/// re-init and sync-byte inversion are part of the later TS-packet framing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DvbTEnergyDispersal {
    /// Live 15-bit PRBS register (low 15 bits used).
    reg: u16,
}

impl Default for DvbTEnergyDispersal {
    fn default() -> Self {
        Self::new()
    }
}

impl DvbTEnergyDispersal {
    /// A whitener starting from the standard init `100101010000000`.
    pub fn new() -> Self {
        Self {
            reg: DVB_T_PRBS_INIT,
        }
    }

    /// Reloads the standard init sequence (frame/packet-group re-init).
    pub fn reset(&mut self) {
        self.reg = DVB_T_PRBS_INIT;
    }

    /// Advances the PRBS one step and returns the output bit. The register holds
    /// stages `x1..x15` with `x1` at bit 14 (the init's MSB) and `x15` at bit 0.
    /// Feedback is `x15 XOR x14` (bits 0 and 1, the taps of 1 + X^14 + X^15); the
    /// register shifts toward `x15` (right) with the feedback inserted at `x1`
    /// (bit 14), and the output bit is that feedback. Verified against the
    /// standard's `0000 0011` first-8-bits known answer.
    #[inline]
    fn next_bit(&mut self) -> u8 {
        let fb = (self.reg ^ (self.reg >> 1)) & 1;
        self.reg = (self.reg >> 1) | (fb << 14);
        fb as u8
    }

    /// XORs the PRBS over `data` in place, MSB-first per byte, carrying register
    /// state across calls. Run once to scramble, and again from the same init to
    /// descramble.
    pub fn feed_in_place(&mut self, data: &mut [u8]) {
        for byte in data.iter_mut() {
            let mut out = 0u8;
            for bit in (0..8).rev() {
                let pn = self.next_bit();
                out |= ((((*byte >> bit) & 1) ^ pn) & 1) << bit;
            }
            *byte = out;
        }
    }

    /// XORs the PRBS over `data`, MSB-first, returning the whitened bytes.
    pub fn feed(&mut self, data: &[u8]) -> Vec<u8> {
        let mut out = data.to_vec();
        self.feed_in_place(&mut out);
        out
    }
}

// ── DVB-T constellation mapping (ETSI EN 300 744 §4.3.5, Figure 9a) ─────────
//
// DVB-T maps the v bits y0..y(v-1) of a symbol so that the **even** bits
// (y0,y2,y4) select the in-phase axis and the **odd** bits (y1,y3,y5) select the
// quadrature axis (v = 2 for QPSK, 4 for 16-QAM, 6 for 64-QAM; non-hierarchical
// uses the α = 1 uniform constellation). Each axis is a Gray map with the
// lowest-index bit as the MSB. The per-axis index→level tables below are read
// directly off Figure 9a (V1.6.2) — the axis MSB is effectively a sign bit, and
// the levels are the odd integers ±1, ±3, … , ±(M−1) with `M = 2^(v/2)` levels
// per axis.
//
// This differs from `modulate::qam`'s generic mapper in both the axis-bit
// assignment (DVB-T interleaves even/odd bits across the axes; the generic
// mapper groups the first half of the bits on I and the second half on Q) and
// the exact Gray labelling, so it is a distinct DVB-T-specific mapping selected
// for DVB-T configs. Amplitudes are normalized to unit average symbol energy
// (matching `modulate::qam::axis_scale`), so the constellation power is the same
// as the generic mapper's.
//
// LLR/soft demap is not provided here: the frame layer's soft demapper consumes
// the generic mapper's convention, so DVB-T soft-decision RX is wired when the
// DVB-T config path is assembled; this module supplies the mapping tables and a
// hard nearest-point demap for round-trip/known-answer verification.

/// Per-axis DVB-T level tables (index = the axis's bits, lowest-y-index as MSB),
/// from Figure 9a. Unnormalized odd-integer levels.
const DVB_T_AXIS_QPSK: [i32; 2] = [1, -1];
const DVB_T_AXIS_16QAM: [i32; 4] = [3, 1, -3, -1];
const DVB_T_AXIS_64QAM: [i32; 8] = [7, 5, 1, 3, -7, -5, -1, -3];

/// Returns the unnormalized per-axis level table for `v` bits per symbol
/// (`v ∈ {2, 4, 6}` → QPSK/16-QAM/64-QAM), or `None` for an unsupported order.
fn dvb_t_axis_table(v: usize) -> Option<&'static [i32]> {
    match v {
        2 => Some(&DVB_T_AXIS_QPSK),
        4 => Some(&DVB_T_AXIS_16QAM),
        6 => Some(&DVB_T_AXIS_64QAM),
        _ => None,
    }
}

/// Packs the axis bits (MSB-first) into a table index.
#[inline]
fn axis_index(bits: &[u8]) -> usize {
    bits.iter()
        .fold(0usize, |acc, &b| (acc << 1) | (b & 1) as usize)
}

/// Maps one DVB-T symbol's `v` bits `y0..y(v-1)` to a normalized constellation
/// point. Even bits (`y0,y2,…`) form the I axis, odd bits (`y1,y3,…`) the Q axis,
/// each Gray-mapped per Figure 9a. Returns `None` if `bits.len()` is not a
/// supported `v ∈ {2,4,6}`.
pub fn dvb_t_map_symbol(bits: &[u8]) -> Option<num_complex::Complex32> {
    let v = bits.len();
    let table = dvb_t_axis_table(v)?;
    let scale = crate::modulate::qam::axis_scale(v);
    // De-interleave: even indices → I, odd indices → Q (both MSB-first in y-order).
    let i_bits: Vec<u8> = bits.iter().step_by(2).copied().collect();
    let q_bits: Vec<u8> = bits.iter().skip(1).step_by(2).copied().collect();
    let i = table[axis_index(&i_bits)] as f32 * scale;
    let q = table[axis_index(&q_bits)] as f32 * scale;
    Some(num_complex::Complex32::new(i, q))
}

/// Hard nearest-point demap: inverse of [`dvb_t_map_symbol`] for `v` bits. Finds
/// the nearest per-axis level and recovers the `v` bits `y0..y(v-1)`. Returns
/// `None` for an unsupported order.
pub fn dvb_t_demap_symbol(sym: num_complex::Complex32, v: usize) -> Option<Vec<u8>> {
    let table = dvb_t_axis_table(v)?;
    let scale = crate::modulate::qam::axis_scale(v);
    let k = v / 2; // bits per axis
    // Nearest table index on one axis for a received (normalized) coordinate.
    let nearest = |coord: f32| -> usize {
        let mut best = 0usize;
        let mut best_d = f32::INFINITY;
        for (idx, &lvl) in table.iter().enumerate() {
            let d = (coord - lvl as f32 * scale).abs();
            if d < best_d {
                best_d = d;
                best = idx;
            }
        }
        best
    };
    let i_idx = nearest(sym.re);
    let q_idx = nearest(sym.im);
    // Unpack each axis index back to k MSB-first bits, then re-interleave.
    let unpack = |idx: usize| -> Vec<u8> { (0..k).rev().map(|b| ((idx >> b) & 1) as u8).collect() };
    let ib = unpack(i_idx);
    let qb = unpack(q_idx);
    let mut out = vec![0u8; v];
    for j in 0..k {
        out[2 * j] = ib[j]; // even → I
        out[2 * j + 1] = qb[j]; // odd → Q
    }
    Some(out)
}

// ── 2K-mode numerology and carrier map (ETSI EN 300 744 §4.4–4.5) ───────────
//
// DVB-T 2K mode: an n_fft = 2048 IFFT with Kmax = 1704 active carriers (indices
// 0..=1704, so 1705 carriers), of which 1512 carry data, 45 are continual
// pilots, plus scattered pilots and 17 TPS carriers. Narrowband DVB-T scales
// only the sample rate `fs` (occupied_BW = fs·1705/2048); the carrier map is
// unchanged. This phase places the 45 continual pilots and treats the scattered/
// TPS positions as data (tightened to the conformant 1512 when scattered pilots
// and TPS land); channel estimation uses the preamble training symbol.

use crate::fec::{
    ConvCode, InnerFec, InterleaverKind, OuterFec, PunctureRate, ScramblerKind, ScramblerPos,
};
use crate::modulate::{ConstellationOrder, Mcs, McsTable, OfdmConfig};
use crate::multicarrier::CarrierPlan;
use num_complex::Complex32 as C32;

/// DVB-T 2K-mode FFT size.
pub const DVB_T_N_FFT: usize = 2048;
/// Highest active carrier index (Kmax) in 2K mode; carriers span `0..=DVB_T_KMAX`.
pub const DVB_T_KMAX: usize = 1704;
/// Number of active (used) carriers in 2K mode, `DVB_T_KMAX + 1`.
pub const DVB_T_ACTIVE_CARRIERS: usize = DVB_T_KMAX + 1; // 1705
/// Number of data-carrying carriers per symbol in 2K mode (constant).
pub const DVB_T_DATA_CARRIERS: usize = 1512;
/// DC-centering offset: DVB active index `a` maps to signed carrier `a − OFFSET`.
const DVB_T_CENTER: i32 = (DVB_T_KMAX / 2) as i32; // 852

/// The 45 continual-pilot carrier indices for 2K mode (EN 300 744 Table 7,
/// 2K column). "Continual" = present on every symbol.
pub const DVB_T_CONTINUAL_PILOTS_2K: [usize; 45] = [
    0, 48, 54, 87, 141, 156, 192, 201, 255, 279, 282, 333, 432, 450, 483, 525, 531, 618, 636, 714,
    759, 765, 780, 804, 873, 888, 918, 939, 942, 969, 984, 1050, 1101, 1107, 1110, 1137, 1140,
    1146, 1206, 1269, 1323, 1377, 1491, 1683, 1704,
];

/// DVB-T guard interval as a fraction of the useful symbol part `Tu`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GuardInterval {
    G1_32,
    G1_16,
    G1_8,
    G1_4,
}

impl GuardInterval {
    /// Cyclic-prefix length in samples for 2K mode (`Tu = 2048`).
    pub const fn cp_len_2k(self) -> usize {
        match self {
            GuardInterval::G1_32 => DVB_T_N_FFT / 32, // 64
            GuardInterval::G1_16 => DVB_T_N_FFT / 16, // 128
            GuardInterval::G1_8 => DVB_T_N_FFT / 8,   // 256
            GuardInterval::G1_4 => DVB_T_N_FFT / 4,   // 512
        }
    }
}

/// Maps a DVB active-carrier index `a` (0..=Kmax) to the crate's signed,
/// DC-centered carrier index (`a − Kmax/2`), so the band spans
/// `−852..=+852` around DC.
#[inline]
pub const fn active_to_signed(a: usize) -> i32 {
    a as i32 - DVB_T_CENTER
}

/// The reference-sequence PRBS `w_k` (EN 300 744 §4.5.2): polynomial
/// X^11 + X^2 + 1, register all-ones, one bit per active carrier. Returns the
/// first `len` bits; the sequence begins `1111111111100…`.
pub fn wk_prbs(len: usize) -> Vec<u8> {
    // 11-bit register x1..x11 in bits 0..10; output = x11 (bit 10), feedback =
    // x11 XOR x2 (bits 10 and 1), shifted into x1 (bit 0).
    let mut reg: u16 = 0x7FF; // 11 ones
    let mut out = Vec::with_capacity(len);
    for _ in 0..len {
        let bit = ((reg >> 10) & 1) as u8;
        out.push(bit);
        let fb = ((reg >> 10) ^ (reg >> 1)) & 1;
        reg = ((reg << 1) | fb) & 0x7FF;
    }
    out
}

/// The boosted continual/scattered-pilot value for a PRBS bit `w_k`:
/// `Re = 4/3·2·(1/2 − w_k)` (= ±4/3), `Im = 0` (EN 300 744 §4.5.3/4.5.4). The
/// 4/3 factor is the boosted amplitude giving `E[c·c*] = 16/9`.
#[inline]
pub fn boosted_pilot_value(wk: u8) -> C32 {
    C32::new((4.0 / 3.0) * 2.0 * (0.5 - wk as f32), 0.0)
}

/// Builds the 2K-mode carrier plan for the given guard interval: n_fft = 2048,
/// the 45 continual pilots (boosted, PRBS-valued) as pilot carriers, and every
/// other active carrier as data. (Phase 1: scattered-pilot and TPS positions are
/// still data — 1660 data carriers — tightened to 1512 when those land.)
pub fn dvb_t_2k_plan(guard: GuardInterval) -> CarrierPlan {
    let wk = wk_prbs(DVB_T_ACTIVE_CARRIERS);
    let pilots: Vec<(i32, C32)> = DVB_T_CONTINUAL_PILOTS_2K
        .iter()
        .map(|&a| (active_to_signed(a), boosted_pilot_value(wk[a])))
        .collect();
    let pilot_set: std::collections::HashSet<usize> =
        DVB_T_CONTINUAL_PILOTS_2K.iter().copied().collect();
    let data: Vec<i32> = (0..=DVB_T_KMAX)
        .filter(|a| !pilot_set.contains(a))
        .map(active_to_signed)
        .collect();
    CarrierPlan::new(DVB_T_N_FFT, guard.cp_len_2k())
        .with_data_carriers(data)
        .with_pilot_carriers(pilots)
}

// ── Bandwidth / sample-rate scaling (narrowband DVB-T) ──────────────────────
//
// occupied_BW = fs · (active/n_fft) = fs · 1705/2048, so fs = BW · 2048/1705.

/// Sample rate (S/s) for a target occupied RF bandwidth (Hz):
/// `fs = occupied_hz · 2048 / 1705`.
pub fn dvb_t_fs_for_bandwidth(occupied_hz: f32) -> f32 {
    occupied_hz * DVB_T_N_FFT as f32 / DVB_T_ACTIVE_CARRIERS as f32
}

/// Occupied RF bandwidth (Hz) for a sample rate: the inverse of
/// [`dvb_t_fs_for_bandwidth`].
pub fn dvb_t_occupied_bw(fs: f32) -> f32 {
    fs * DVB_T_ACTIVE_CARRIERS as f32 / DVB_T_N_FFT as f32
}

/// fs for the ~333 kHz narrowband mode (robust 70 cm config). Below PlutoSDR's
/// ~521 kS/s continuous-TX floor — valid for the library, not for continuous
/// Pluto TX.
pub const DVB_T_FS_333KHZ: f32 = 333_000.0 * DVB_T_N_FFT as f32 / DVB_T_ACTIVE_CARRIERS as f32;
/// fs for the ~1 MHz narrowband mode (common general-purpose amateur DATV).
pub const DVB_T_FS_1MHZ: f32 = 1_000_000.0 * DVB_T_N_FFT as f32 / DVB_T_ACTIVE_CARRIERS as f32;
/// fs for the ~2 MHz narrowband mode (wider repeater config).
pub const DVB_T_FS_2MHZ: f32 = 2_000_000.0 * DVB_T_N_FFT as f32 / DVB_T_ACTIVE_CARRIERS as f32;

// ── DVB-T MCS table (concatenated FEC) ──────────────────────────────────────

/// A DVB-T MCS table: QPSK and 16-QAM, each with the K=7 punctured convolutional
/// inner code (rates 1/2, 2/3, 3/4) and the RS(204,188) outer code — the DVB-T
/// concatenation. The scrambler (energy dispersal) and the Forney outer
/// interleaver are link-wide settings on the config, not per-MCS.
pub fn dvb_t_mcs_table() -> McsTable {
    let rs = OuterFec::ReedSolomon {
        n: 204,
        n_parity: 16,
    };
    let conv = |rate| InnerFec::Convolutional {
        rate,
        code: ConvCode::DvbK7,
    };
    McsTable::new(vec![
        // Robust: QPSK rate 1/2 (333 kHz-class).
        Mcs::new(ConstellationOrder::Qpsk, conv(PunctureRate::R1_2), rs),
        // General-purpose: QPSK rate 2/3 (1 MHz-class).
        Mcs::new(ConstellationOrder::Qpsk, conv(PunctureRate::R2_3), rs),
        // Wider: 16-QAM rate 3/4 (2 MHz-class).
        Mcs::new(ConstellationOrder::Qam16, conv(PunctureRate::R3_4), rs),
    ])
}

// ── DVB-T link config assembly ──────────────────────────────────────────────

/// Assembles a DVB-T 2K-mode [`OfdmConfig`] for a target occupied bandwidth:
/// the 2K carrier plan (with continual pilots) at `fs = occupied_hz·2048/1705`,
/// DVB-T energy dispersal (before the outer FEC), and the Forney I=12/M=17 outer
/// interleaver. Pair with [`dvb_t_mcs_table`] and drive it through the COFDM
/// frame layer (`OfdmFrameMod`/`demodulate_frame`).
///
/// The payload FEC is DVB-T-conformant (K=7 conv + Forney + RS(204,188) + exact
/// energy dispersal); the symbol mapping through the frame layer is the generic
/// QAM mapper for now (Phase 1) — the DVB-T-exact constellation
/// (`dvb_t_map_symbol`) is wired into the shared soft-decision core in a later
/// phase.
pub fn dvb_t_config(guard: GuardInterval, occupied_hz: f32) -> OfdmConfig {
    let plan = dvb_t_2k_plan(guard);
    let fs = dvb_t_fs_for_bandwidth(occupied_hz);
    // The base constellation is overridden per-frame by the MCS table; QPSK is a
    // sensible default for the positional constructor.
    OfdmConfig::new(plan, fs, 0.0, 1.0, ConstellationOrder::Qpsk)
        .with_scrambler(ScramblerKind::DvbTEnergyDispersal)
        .with_scrambler_pos(ScramblerPos::BeforeOuterFec)
        .with_outer_interleaver(InterleaverKind::Convolutional {
            branches: 12,
            depth: 17,
        })
}
