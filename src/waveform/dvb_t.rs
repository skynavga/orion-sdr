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

    /// Advances the PRBS by one byte (8 steps) WITHOUT applying its output — the
    /// "PRBS generation continues but output disabled" behaviour the standard
    /// specifies over the MPEG-2 sync bytes of the seven trailing packets in a
    /// group (§4.3.1). Keeps the register phase aligned with a run that skips
    /// those bytes for output but still clocks the generator.
    pub fn advance_byte(&mut self) {
        for _ in 0..8 {
            self.next_bit();
        }
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

/// Max-log soft LLRs for one received DVB-T symbol `sym` at order `v` bits,
/// returned in the same `y0..y(v-1)` order as [`dvb_t_map_symbol`] (even bits on
/// the I axis, odd on Q). `LLR > 0 ⇒ bit more likely 0`, matching the crate-wide
/// convention (see `demodulate::ofdm`). Returns `None` for an unsupported order.
///
/// This is the DVB-T counterpart to the generic `qam_soft_llr`: the generic
/// mapper groups the first half of the bits on I and the second on Q with its own
/// Gray labels, whereas DVB-T interleaves even/odd bits across the axes and uses
/// the Figure-9a per-axis tables — so DVB-T soft-decision needs this dedicated
/// LLR extraction rather than the generic one.
pub fn dvb_t_soft_llr(sym: num_complex::Complex32, v: usize) -> Option<Vec<f32>> {
    let table = dvb_t_axis_table(v)?;
    let scale = crate::modulate::qam::axis_scale(v);
    let k = v / 2; // bits per axis

    // Per-axis LLRs for the k axis bits (MSB-first, matching `axis_index`).
    let axis_llrs = |coord: f32| -> Vec<f32> {
        let mut out = vec![0.0f32; k];
        for (b, slot) in out.iter_mut().enumerate() {
            // Axis index bit `b` is at shift (k-1-b): MSB is bit 0 of the label.
            let shift = k - 1 - b;
            let mut d0 = f32::INFINITY;
            let mut d1 = f32::INFINITY;
            for (idx, &lvl) in table.iter().enumerate() {
                let d = coord - lvl as f32 * scale;
                let d_sq = d * d;
                if (idx >> shift) & 1 == 0 {
                    d0 = d0.min(d_sq);
                } else {
                    d1 = d1.min(d_sq);
                }
            }
            // Positive ⇒ bit 0 (closer to a 0-labelled level than any 1-labelled).
            *slot = d1 - d0;
        }
        out
    };

    let il = axis_llrs(sym.re);
    let ql = axis_llrs(sym.im);
    // Re-interleave to y-order: even bits from I, odd from Q.
    let mut out = vec![0.0f32; v];
    for j in 0..k {
        out[2 * j] = il[j];
        out[2 * j + 1] = ql[j];
    }
    Some(out)
}

/// Whether `order` is one of the three DVB-T constellations (QPSK/16-QAM/64-QAM)
/// that [`dvb_t_map_symbol`]/[`dvb_t_soft_llr`] handle. BPSK/256-QAM are crate
/// extensions outside DVB-T, so a link carrying them (e.g. an `OrionSdr`-header
/// BPSK block) must fall back to the generic mapper for those symbols.
pub fn is_dvb_t_constellation(order: ConstellationOrder) -> bool {
    matches!(
        order,
        ConstellationOrder::Qpsk | ConstellationOrder::Qam16 | ConstellationOrder::Qam64
    )
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
use crate::multicarrier::{CarrierGrid, CarrierPlan};
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

/// The 17 TPS (Transmission Parameter Signalling) carrier indices for 2K mode
/// (EN 300 744 Table 8, 2K column). These carriers convey the DBPSK-encoded TPS
/// word (added in a later phase); here they are reserved as non-data so the
/// data-carrier count is the conformant 1512 (they carry a `w_k`-derived
/// placeholder pilot until TPS signalling is wired).
pub const DVB_T_TPS_CARRIERS_2K: [usize; 17] = [
    34, 50, 209, 346, 413, 569, 595, 688, 790, 901, 1073, 1219, 1262, 1286, 1469, 1594, 1687,
];

/// Number of distinct scattered-pilot symbol phases: the pilot pattern repeats
/// every 4 OFDM symbols (`l mod 4`).
pub const DVB_T_SCATTERED_PHASES: usize = 4;

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

    /// Recovers the guard interval from a 2K cyclic-prefix length (the inverse of
    /// [`cp_len_2k`](Self::cp_len_2k)). Returns `None` for a non-2K CP length. Used
    /// by the frame layer to rebuild the scattered-pilot orchestrators from a
    /// config's plan alone.
    pub const fn from_cp_len_2k(cp_len: usize) -> Option<Self> {
        match cp_len {
            64 => Some(GuardInterval::G1_32),
            128 => Some(GuardInterval::G1_16),
            256 => Some(GuardInterval::G1_8),
            512 => Some(GuardInterval::G1_4),
            _ => None,
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

/// The scattered-pilot active-carrier indices for symbol phase `phase`
/// (`l mod 4`, EN 300 744 §4.5.3): `{ k = 3·(phase mod 4) + 12p | p ≥ 0,
/// 0 ≤ k ≤ Kmax }`. With `Kmin = 0` in 2K mode this is exactly
/// `k mod 12 == 3·(phase mod 4)`. Present on every symbol but stepping through
/// four positions, so the union over four consecutive symbols samples every
/// third carrier.
pub fn scattered_pilot_indices(phase: usize) -> Vec<usize> {
    let start = 3 * (phase % DVB_T_SCATTERED_PHASES);
    (start..=DVB_T_KMAX).step_by(12).collect()
}

/// The 17 TPS active-carrier indices for 2K mode. A thin wrapper over
/// [`DVB_T_TPS_CARRIERS_2K`] mirroring [`scattered_pilot_indices`]'s shape.
pub fn tps_carrier_indices() -> &'static [usize] {
    &DVB_T_TPS_CARRIERS_2K
}

/// The rustfft bin index for each TPS carrier in the 2K plan, in the order of
/// [`DVB_T_TPS_CARRIERS_2K`]: `bin = (active − 852).rem_euclid(2048)`. The DVB-T
/// frame assembler overwrites these bins with the DBPSK TPS cells after the
/// scattered-grid mapper has placed data + pilots.
pub fn tps_carrier_bins() -> [usize; DVB_T_TPS_CARRIERS_2K.len()] {
    core::array::from_fn(|i| {
        active_to_signed(DVB_T_TPS_CARRIERS_2K[i]).rem_euclid(DVB_T_N_FFT as i32) as usize
    })
}

/// The rustfft bin index for each of the 45 continual pilots in the 2K plan, in
/// the order of [`DVB_T_CONTINUAL_PILOTS_2K`]: `bin = (active − 852).rem_euclid(
/// 2048)`. These sit at fixed positions on every symbol (unlike the scattered
/// pilots), so they anchor integer-CFO estimation: an integer offset of `k`
/// subcarriers slides the whole spectrum, landing the continual pilots at
/// `bin + k` (see `sync::dvb_t_integer_cfo`).
pub fn continual_pilot_bins() -> [usize; DVB_T_CONTINUAL_PILOTS_2K.len()] {
    core::array::from_fn(|i| {
        active_to_signed(DVB_T_CONTINUAL_PILOTS_2K[i]).rem_euclid(DVB_T_N_FFT as i32) as usize
    })
}

/// Builds the four symbol-phase carrier plans for the conformant 2K frame
/// structure (EN 300 744 §4.5). Plan `p` (for symbols with `l mod 4 == p`)
/// reserves the following as boosted, `w_k`-valued pilots:
///
/// - the 45 **continual** pilots (same on every symbol),
/// - the phase-`p` **scattered** pilots (`k mod 12 == 3·p`),
/// - the 17 **TPS** carriers (placeholder `w_k` value until TPS signalling is
///   added; reserved now so the data layout is conformant),
///
/// with every remaining active carrier as data. By construction each plan
/// carries **exactly [`DVB_T_DATA_CARRIERS`] = 1512 data carriers** — the
/// standard fixes the scattered-pilot spacing so this count is constant across
/// all four phases (see the note in §4.5). This invariant is what lets the
/// frame layer's count-based bookkeeping (`bits_per_ofdm_symbol`, `block_plan`)
/// stay valid while the physical pilot/data bins rotate underneath.
///
/// # Panics
///
/// Panics if any plan does not carry exactly 1512 data carriers — a transcription
/// error in the pilot/TPS tables would otherwise silently desync TX and RX.
pub fn dvb_t_2k_plans(guard: GuardInterval) -> [CarrierPlan; DVB_T_SCATTERED_PHASES] {
    let wk = wk_prbs(DVB_T_ACTIVE_CARRIERS);
    core::array::from_fn(|phase| {
        // Reserve continual + phase-p scattered + TPS as pilots (deduped: a
        // scattered pilot may coincide with a continual one every fourth symbol).
        let mut reserved: std::collections::BTreeSet<usize> =
            DVB_T_CONTINUAL_PILOTS_2K.iter().copied().collect();
        reserved.extend(scattered_pilot_indices(phase));
        reserved.extend(DVB_T_TPS_CARRIERS_2K.iter().copied());

        let pilots: Vec<(i32, C32)> = reserved
            .iter()
            .map(|&a| (active_to_signed(a), boosted_pilot_value(wk[a])))
            .collect();
        let data: Vec<i32> = (0..=DVB_T_KMAX)
            .filter(|a| !reserved.contains(a))
            .map(active_to_signed)
            .collect();
        assert_eq!(
            data.len(),
            DVB_T_DATA_CARRIERS,
            "scattered plan phase {phase} must carry exactly {DVB_T_DATA_CARRIERS} data carriers"
        );
        CarrierPlan::new(DVB_T_N_FFT, guard.cp_len_2k())
            .with_data_carriers(data)
            .with_pilot_carriers(pilots)
    })
}

// ── Scattered-pilot symbol-phase orchestration ──────────────────────────────
//
// DVB-T scattered pilots move every symbol, cycling through four phases
// (`l mod 4`). The generic OFDM grid (`CarrierGrid`) is built once and applies
// the same bin→role map to every symbol, so a single grid cannot express the
// rotating pilot pattern. These orchestrators own the four pre-built grids (one
// per phase) and a symbol counter, selecting `grids[phase]` for each symbol and
// advancing the phase — a frame-layer concern (cross-symbol state), so they are
// deliberately NOT `Block`s (which are single-symbol and stateless-across-calls).
// The per-symbol bin count is the same 1512 in every phase (see
// [`dvb_t_2k_plans`]), so the surrounding count-based bookkeeping is unchanged;
// only which physical bins are data vs. pilot rotates.

/// The four symbol-phase grids plus a running phase counter, shared by the TX
/// mapper and RX extractor. `l = 0` is defined at the first frame symbol (the
/// caller [`reset`](Self::reset)s per frame), matching the scattered-pilot phase
/// `l mod 4`.
///
/// Each grid reserves the 17 TPS carriers as pilots (so they are non-data and the
/// 1512-data invariant holds), but the TPS bins are **not** channel-estimation
/// references: the modulator overwrites them with data-power DBPSK cells, not the
/// boosted `w_k` pilot value the grid records. `ref_pilots` therefore holds each
/// phase's channel-reference pilots — the continual + scattered pilots **only**,
/// with the 17 TPS bins removed — for the RX equalizer to interpolate the channel
/// from. Feeding TPS bins to the estimator instead would divide the transmitted
/// ±1.0 DBPSK cell by the grid's ±4/3 known value, yielding a bogus `h = ∓0.75`
/// that smears onto the data carriers straddling each TPS carrier.
#[derive(Debug, Clone)]
struct ScatteredGridCycle {
    grids: [CarrierGrid; DVB_T_SCATTERED_PHASES],
    /// Per-phase channel-reference pilots: grid pilots minus the TPS bins.
    ref_pilots: [Vec<(usize, C32)>; DVB_T_SCATTERED_PHASES],
    phase: usize,
}

impl ScatteredGridCycle {
    fn new(guard: GuardInterval) -> Self {
        let plans = dvb_t_2k_plans(guard);
        let grids = plans.each_ref().map(CarrierGrid::from_plan);
        let tps: std::collections::HashSet<usize> = tps_carrier_bins().into_iter().collect();
        let ref_pilots = grids.each_ref().map(|g| {
            g.pilot_bins()
                .iter()
                .copied()
                .filter(|&(bin, _)| !tps.contains(&bin))
                .collect::<Vec<_>>()
        });
        Self {
            grids,
            ref_pilots,
            phase: 0,
        }
    }

    /// The grid for the current symbol phase.
    fn current(&self) -> &CarrierGrid {
        &self.grids[self.phase]
    }

    /// The current phase's channel-reference pilots (continual + scattered,
    /// **excluding** the TPS carriers).
    fn current_ref_pilots(&self) -> &[(usize, C32)] {
        &self.ref_pilots[self.phase]
    }

    /// Advances to the next symbol phase (`l mod 4`).
    fn advance(&mut self) {
        self.phase = (self.phase + 1) % DVB_T_SCATTERED_PHASES;
    }

    /// Restarts at phase 0 (call once per frame, so `l = 0` is the first symbol).
    fn reset(&mut self) {
        self.phase = 0;
    }
}

/// TX-side scattered-pilot grid mapper: scatters each symbol's dense data
/// symbols onto the phase-appropriate 2K grid (continual, phase-`l` scattered,
/// and TPS pilots inserted; all other active carriers = data), producing one
/// `n_fft`-bin frequency-domain vector per symbol and advancing the symbol
/// phase. The caller supplies already-mapped constellation symbols
/// (`num_data_carriers()` of them) — the constellation mapping itself is
/// grid-agnostic and stays with the generic mappers.
#[derive(Debug, Clone)]
pub struct ScatteredPilotMapper {
    cycle: ScatteredGridCycle,
}

impl ScatteredPilotMapper {
    /// Builds the mapper for the given guard interval (n_fft = 2048).
    pub fn new(guard: GuardInterval) -> Self {
        Self {
            cycle: ScatteredGridCycle::new(guard),
        }
    }

    /// Data carriers per symbol — the constant [`DVB_T_DATA_CARRIERS`] (1512).
    pub fn num_data_carriers(&self) -> usize {
        DVB_T_DATA_CARRIERS
    }

    /// FFT size (2048).
    pub fn n_fft(&self) -> usize {
        DVB_T_N_FFT
    }

    /// Restarts the symbol-phase counter (call once at the start of each frame).
    pub fn reset(&mut self) {
        self.cycle.reset();
    }

    /// Maps one symbol's `num_data_carriers()` constellation symbols into an
    /// `n_fft`-bin frequency vector using the current phase's grid (pilots
    /// inserted from their known boosted values), then advances the phase.
    /// `freq_out` must be at least `n_fft` long; extra bins are zeroed.
    pub fn map_symbol(&mut self, data: &[C32], freq_out: &mut [C32]) {
        let grid = self.cycle.current();
        let n_fft = grid.n_fft();
        debug_assert!(data.len() >= grid.num_data_carriers());
        debug_assert!(freq_out.len() >= n_fft);
        for bin in freq_out[..n_fft].iter_mut() {
            *bin = C32::default();
        }
        for (k, &bin) in grid.data_bins().iter().enumerate() {
            freq_out[bin] = data[k];
        }
        for &(bin, value) in grid.pilot_bins() {
            freq_out[bin] = value;
        }
        self.cycle.advance();
    }
}

/// RX-side scattered-pilot orchestrator: for each symbol it exposes the current
/// phase's data bins (to gather the dense data stream from an equalized
/// frequency vector) and its pilot bins (to drive the channel estimator), then
/// advances the phase. Mirrors [`ScatteredPilotMapper`].
#[derive(Debug, Clone)]
pub struct ScatteredPilotExtractor {
    cycle: ScatteredGridCycle,
}

impl ScatteredPilotExtractor {
    /// Builds the extractor for the given guard interval (n_fft = 2048).
    pub fn new(guard: GuardInterval) -> Self {
        Self {
            cycle: ScatteredGridCycle::new(guard),
        }
    }

    /// Data carriers per symbol — the constant [`DVB_T_DATA_CARRIERS`] (1512).
    pub fn num_data_carriers(&self) -> usize {
        DVB_T_DATA_CARRIERS
    }

    /// FFT size (2048).
    pub fn n_fft(&self) -> usize {
        DVB_T_N_FFT
    }

    /// Restarts the symbol-phase counter (call once at the start of each frame).
    pub fn reset(&mut self) {
        self.cycle.reset();
    }

    /// The current phase's **channel-reference** pilot bins (rustfft bin index +
    /// known boosted TX value), for installing on the equalizer before this
    /// symbol. These are the continual + scattered pilots **only** — the 17 TPS
    /// carriers are deliberately excluded, because the modulator transmits
    /// data-power DBPSK on them rather than the boosted `w_k` value the grid
    /// records, so they are not valid channel references (see
    /// [`ScatteredGridCycle`]). The equalizer interpolates its estimate for the
    /// TPS bins from the surrounding real pilots instead.
    pub fn current_pilot_bins(&self) -> &[(usize, C32)] {
        self.cycle.current_ref_pilots()
    }

    /// The current phase's data-carrier bins (rustfft bin indices), for the
    /// equalizer to interpolate a channel estimate across before extraction.
    pub fn data_bins(&self) -> &[usize] {
        self.cycle.current().data_bins()
    }

    /// Gathers the current phase's data bins from an equalized `n_fft`-bin
    /// frequency vector into `data_out` (`num_data_carriers()` symbols), then
    /// advances the phase.
    pub fn extract_symbol(&mut self, freq: &[C32], data_out: &mut [C32]) {
        let grid = self.cycle.current();
        debug_assert!(freq.len() >= grid.n_fft());
        debug_assert!(data_out.len() >= grid.num_data_carriers());
        for (k, &bin) in grid.data_bins().iter().enumerate() {
            data_out[k] = freq[bin];
        }
        self.cycle.advance();
    }
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

/// The standard narrowband DVB-T (amateur DATV) channel bandwidths. NB-DVB-T is
/// a pure fs-scaling of the fixed 2K structure, so a mode is just a target
/// occupied bandwidth; this enum is a convenience so callers pick a named mode
/// instead of a raw `occupied_hz`. Compose with the config builders, e.g.
/// `dvb_t_scattered_config(guard, NbBandwidth::Bw1MHz.occupied_hz())` or
/// `cfg.with_fs(NbBandwidth::Bw1MHz.fs())`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NbBandwidth {
    /// ~333 kHz — robust 70 cm config. Below PlutoSDR's ~521 kS/s continuous-TX
    /// floor (a hardware limit, not a waveform one); valid as a library mode.
    Bw333kHz,
    /// ~1 MHz — the common general-purpose amateur DATV channel.
    Bw1MHz,
    /// ~2 MHz — a wider repeater/high-rate config.
    Bw2MHz,
}

impl NbBandwidth {
    /// The nominal occupied RF bandwidth (Hz) for this mode.
    pub const fn occupied_hz(self) -> f32 {
        match self {
            NbBandwidth::Bw333kHz => 333_000.0,
            NbBandwidth::Bw1MHz => 1_000_000.0,
            NbBandwidth::Bw2MHz => 2_000_000.0,
        }
    }

    /// The sample rate (S/s) for this mode: `occupied_hz · 2048/1705`.
    pub fn fs(self) -> f32 {
        dvb_t_fs_for_bandwidth(self.occupied_hz())
    }

    /// Whether this mode's `fs` is representable for continuous PlutoSDR TX
    /// (≥ ~521 kS/s). `Bw333kHz` is below the floor (a valid library/test mode).
    pub fn is_pluto_continuous_tx(self) -> bool {
        self.fs() >= 521_000.0
    }
}

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
    dvb_t_config_with_plan(dvb_t_2k_plan(guard), occupied_hz)
}

/// Like [`dvb_t_config`], but with the conformant **scattered-pilot** structure:
/// the payload symbols are mapped/demapped through the four-phase grid rotation
/// (continual + scattered + TPS pilots reserved, exactly 1512 data carriers per
/// symbol — EN 300 744 §4.5). The config's `carrier_plan` is the representative
/// phase-0 plan (1512 data), and the frame layer rotates the physical bins
/// underneath (`OfdmConfig::dvb_t_scattered` is set). Channel estimation uses the
/// scattered + continual pilots per symbol, which tracks a frequency-selective
/// channel far better than the 45 continual pilots alone.
pub fn dvb_t_scattered_config(guard: GuardInterval, occupied_hz: f32) -> OfdmConfig {
    // Representative plan = phase 0 (1512 data). The other three phases live in
    // the orchestrators the frame layer builds from this config.
    let plan = dvb_t_2k_plans(guard)[0].clone();
    dvb_t_config_with_plan(plan, occupied_hz).with_dvb_t_scattered(true)
}

/// Shared assembly for the DVB-T link config over an explicit representative
/// `plan`: DVB-T energy dispersal (before the outer FEC) and the Forney
/// I=12/M=17 outer interleaver, at `fs = occupied_hz·2048/1705`.
///
/// The payload FEC is DVB-T-conformant (K=7 conv + Forney + RS(204,188) + exact
/// energy dispersal); the symbol mapping through the frame layer is the generic
/// QAM mapper for now — the DVB-T-exact constellation (`dvb_t_map_symbol`) is
/// wired into the shared soft-decision core in a later phase.
fn dvb_t_config_with_plan(plan: CarrierPlan, occupied_hz: f32) -> OfdmConfig {
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

// ── Conformant DVB-T frame: shared parameters (TX/RX common) ────────────────
//
// `DvbTFrameParams` and the FEC constants below are shared by the conformant
// DVB-T frame modulator (`modulate::dvb_t_frame`) and demodulator
// (`demodulate::dvb_t_frame`) — the direction-split assemblers that build/parse
// the preamble-less, TPS-signalled on-air frame. They live here, with the rest
// of the DVB-T waveform definitions, the same way `Mcs`/`BlockPlan` are shared
// frame-layer types (not duplicated per direction).

/// The DVB-T outer FEC: RS(204,188), t = 8 (§4.3.2).
pub const DVB_T_FRAME_OUTER: OuterFec = OuterFec::ReedSolomon {
    n: 204,
    n_parity: 16,
};
/// The DVB-T outer interleaver: Forney convolutional, I = 12 branches, M = 17.
pub const DVB_T_FRAME_OUTER_IL: InterleaverKind = InterleaverKind::Convolutional {
    branches: 12,
    depth: 17,
};

/// The modulation-and-coding parameters shared by every DVB-T frame on a link:
/// guard interval, constellation, and inner code rate. These are constant across a
/// (super-)frame; [`DvbTFrameParams`] and [`DvbTSuperFrameParams`] each embed one,
/// so the shared set is defined once rather than duplicated.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DvbTLinkParams {
    pub guard: GuardInterval,
    pub constellation: ConstellationOrder,
    pub code_rate: PunctureRate,
}

/// The transmission parameters of one conformant DVB-T frame — the shared link
/// parameters ([`DvbTLinkParams`]) plus this frame's TPS-signalled number and
/// cell-id byte. A cold receiver is given this (real receivers acquire on
/// assumptions) and TPS verifies it. Shared by the modulator and demodulator.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DvbTFrameParams {
    /// Guard interval, constellation, and code rate (constant across the link).
    pub link: DvbTLinkParams,
    /// This frame's number within the super-frame (0..=3).
    pub frame_number: u8,
    /// The cell-identifier **byte** this frame carries on its TPS carriers (the
    /// full 16-bit id is a super-frame property — see [`DvbTSuperFrameParams`]).
    pub cell_id: u8,
}

impl DvbTFrameParams {
    /// The guard interval (from the link parameters).
    pub fn guard(self) -> GuardInterval {
        self.link.guard
    }

    /// The constellation (from the link parameters).
    pub fn constellation(self) -> ConstellationOrder {
        self.link.constellation
    }

    /// The inner code rate (from the link parameters).
    pub fn code_rate(self) -> PunctureRate {
        self.link.code_rate
    }

    /// The inner FEC (K=7 punctured convolutional at the frame's code rate).
    pub fn inner(self) -> InnerFec {
        InnerFec::Convolutional {
            rate: self.link.code_rate,
            code: ConvCode::DvbK7,
        }
    }

    /// The TPS word this frame signals on the 17 TPS carriers.
    pub fn tps_word(self) -> crate::waveform::dvb_t_tps::TpsWord {
        crate::waveform::dvb_t_tps::TpsWord {
            frame_number: self.frame_number,
            constellation: self.link.constellation,
            code_rate_hp: self.link.code_rate,
            guard: self.link.guard,
            cell_id: self.cell_id,
        }
    }

    /// The representative (phase-0) [`OfdmConfig`] for this frame: the 1512-data
    /// phase-0 plan at the frame's constellation, scattered-pilot mode enabled.
    /// `fs` is set from the 1 MHz scaling — it only affects timing/CFO units in
    /// the batch assemblers (no RF upconversion), so the exact value is not
    /// load-bearing for a baseband frame.
    pub fn config(self) -> OfdmConfig {
        let plan0 = dvb_t_2k_plans(self.link.guard)[0].clone();
        let fs = dvb_t_fs_for_bandwidth(1_000_000.0);
        OfdmConfig::new(plan0, fs, 0.0, 1.0, self.link.constellation).with_dvb_t_scattered(true)
    }
}
