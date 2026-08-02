// Copyright (c) 2026 G & R Associates LLC
// SPDX-License-Identifier: MIT OR Apache-2.0

// src/waveform/dvb_t_tps.rs
//
// DVB-T Transmission Parameter Signalling (TPS), ETSI EN 300 744 §4.6. TPS is
// the DVB-T on-air signalling scheme: 17 TPS carriers (2K mode) each carry the
// SAME differentially-BPSK-encoded bit per symbol, spelling out a 68-bit word
// over one 68-symbol OFDM frame (four frames = one super-frame). The word
// carries the constellation, code rate(s), guard interval, transmission mode,
// frame number, and cell id, protected by a shortened BCH(67,53) t=2 code.
//
// This module holds the standard-exact, waveform-layer pieces:
//   • the standalone GF(2^7) BCH(67,53) code (deliberately NOT a parameter of
//     the hot GF(2^8) `Gf256`/`Bch` — it runs once per 68-symbol frame);
//   • the `TpsWord` (pack/unpack of the 68-bit block, incl. the sync word and
//     BCH parity);
//   • the DBPSK-along-the-symbol-axis encoder/decoder (`TpsEncoder`/
//     `TpsDecoder`), a frame-layer orchestrator with per-frame state.
//
// The 17 TPS carrier indices live in `waveform::dvb_t` (`DVB_T_TPS_CARRIERS_2K`,
// reserved as pilots in the four scattered-pilot plans since Phase 2); this
// module fills them with signalling.

// ── Standalone GF(2^7) BCH(67,53) t=2 (EN 300 744 §4.6.2.11) ────────────────
//
// The TPS parity is a shortened BCH(67,53) derived from the primitive
// BCH(127,113) t=2 code over GF(2^7) with primitive polynomial
// p(x) = x^7 + x^3 + 1 (0x89 — the choice for which g(x) = m_1(x)·m_3(x) equals
// the standard's generator). Shortening prepends 60 implicit zero information
// bits to the 53 real ones (53 + 60 = 113 = k of the mother code), encodes with
// the mother generator, then transmits only the 53 info + 14 parity bits.
//
// Code generator polynomial (given explicitly by the standard):
//   h(x) = x^14 + x^9 + x^8 + x^6 + x^5 + x^4 + x^2 + x + 1   (= 0x4377)
//
// This is a self-contained ~t=2 syndrome / Berlekamp–Massey / Chien decoder over
// the small 127-element field. It is cold (once per frame), so its own speed is
// irrelevant and the hot GF(2^8) tables stay monomorphic and untouched.

/// GF(2^7) primitive polynomial x^7 + x^3 + 1 (low 8 bits).
const GF128_PRIM: u16 = 0x89;
/// Order of the multiplicative group of GF(2^7): 2^7 − 1.
const GF128_ORDER: usize = 127;

/// BCH(67,53) generator h(x) = x^14 + x^9 + x^8 + x^6 + x^5 + x^4 + x^2 + x + 1.
const TPS_BCH_GEN: u32 = 0x4377;
/// TPS codeword length in bits.
pub const TPS_CODEWORD_BITS: usize = 67;
/// TPS information (sync + payload) length in bits: s1..s53.
pub const TPS_INFO_BITS: usize = 53;
/// TPS BCH parity length in bits: s54..s67.
pub const TPS_PARITY_BITS: usize = 14;

/// Precomputed GF(2^7) exp/log tables (127-element multiplicative group).
struct Gf128 {
    /// `exp[i] = α^i` for `i` in 0..2·127 (doubled so `exp[log a + log b]` needs
    /// no modulo — the log sum is at most 126 + 126 = 252).
    exp: [u8; 2 * GF128_ORDER],
    /// `log[a]` such that `α^log[a] == a`, for `a` in 1..=127; `log[0]` unused.
    log: [u8; GF128_ORDER + 1],
}

impl Gf128 {
    fn new() -> Self {
        let mut exp = [0u8; 2 * GF128_ORDER];
        let mut log = [0u8; GF128_ORDER + 1];
        let mut x: u16 = 1;
        for (i, slot) in exp.iter_mut().enumerate().take(GF128_ORDER) {
            *slot = x as u8;
            log[x as usize] = i as u8;
            x <<= 1;
            if x & 0x80 != 0 {
                x ^= GF128_PRIM;
            }
        }
        // Second half repeats the first so index sums up to 252 are in range.
        for i in GF128_ORDER..2 * GF128_ORDER {
            exp[i] = exp[i - GF128_ORDER];
        }
        Self { exp, log }
    }

    #[inline]
    fn mul(&self, a: u8, b: u8) -> u8 {
        if a == 0 || b == 0 {
            0
        } else {
            self.exp[self.log[a as usize] as usize + self.log[b as usize] as usize]
        }
    }

    /// `α^i` for any `i ≥ 0` (reduced mod the group order).
    #[inline]
    fn pow_alpha(&self, i: usize) -> u8 {
        self.exp[i % GF128_ORDER]
    }
}

/// Systematic BCH(67,53) encode: appends the 14 parity bits to the 53 info bits.
/// `info` is exactly [`TPS_INFO_BITS`] bits (MSB-first, s1 first); the result is
/// [`TPS_CODEWORD_BITS`] bits `[info | parity]`. Parity = (info·x^14) mod h(x)
/// over GF(2), which for a shortened code is identical to running the mother
/// BCH(127,113) encoder on 60 prepended zeros + info and dropping the zeros.
///
/// # Panics
///
/// Panics if `info.len() != TPS_INFO_BITS`.
pub fn tps_bch_encode(info: &[u8]) -> [u8; TPS_CODEWORD_BITS] {
    assert_eq!(info.len(), TPS_INFO_BITS, "TPS info must be 53 bits");
    let parity = tps_bch_parity(info);
    let mut out = [0u8; TPS_CODEWORD_BITS];
    out[..TPS_INFO_BITS].copy_from_slice(info);
    for (i, slot) in out[TPS_INFO_BITS..].iter_mut().enumerate() {
        *slot = ((parity >> (TPS_PARITY_BITS - 1 - i)) & 1) as u8;
    }
    out
}

/// The 14 parity bits (as an integer, MSB = highest degree) of `info` under the
/// generator [`TPS_BCH_GEN`]: polynomial remainder of `info·x^14 mod h(x)`.
///
/// Computed with a 14-bit LFSR so the intermediate never needs more than the
/// parity width (the full `info·x^14` polynomial spans 67 bits — too wide for a
/// single machine register, so a shift-register division is used instead).
fn tps_bch_parity(info: &[u8]) -> u32 {
    // `reg` holds the running remainder (degree < 14). Feed the info bits
    // MSB-first followed by 14 zero bits (the `·x^14` systematic multiply): for
    // each input bit shift the remainder up, OR the bit in at the bottom, and if
    // the term at position 14 became set, reduce by the generator.
    let mut reg: u32 = 0;
    let top = 1u32 << TPS_PARITY_BITS; // x^14 term of the generator
    let feed = info
        .iter()
        .copied()
        .chain(std::iter::repeat_n(0u8, TPS_PARITY_BITS));
    for b in feed {
        reg = (reg << 1) | (b & 1) as u32;
        if reg & top != 0 {
            reg ^= TPS_BCH_GEN;
        }
    }
    reg & ((1 << TPS_PARITY_BITS) - 1)
}

/// Decodes a received 67-bit BCH(67,53) codeword, correcting up to 2 bit errors,
/// and returns the 53 information bits. Returns `None` if the errors are
/// uncorrectable (more than `t = 2`).
///
/// Uses the classic syndrome → Berlekamp–Massey → Chien pipeline over GF(2^7).
/// The received word occupies the low `n0 = 127` positions of the mother code
/// with the shortened (implicit-zero) prefix accounted for in the error-locator
/// exponents.
pub fn tps_bch_decode(codeword: &[u8]) -> Option<[u8; TPS_INFO_BITS]> {
    if codeword.len() != TPS_CODEWORD_BITS {
        return None;
    }
    let gf = Gf128::new();

    // Map the 67 received bits to positions in the length-127 mother code. The
    // shortening prepended 60 zeros before the info bits, so codeword bit j
    // (0 = s1, the highest info position) sits at mother-code position
    // (126 − (60 + j)) counting the MSB as position n0−1. Equivalently the bit
    // at codeword index j has locator exponent:
    //   power(j) = (TPS_CODEWORD_BITS - 1 - j)   for the parity/info LSBs,
    // taken as the exponent of α in the Chien search below. We index errors by
    // the codeword bit position directly and use α^(shift + pos).
    let n_shift = GF128_ORDER - TPS_CODEWORD_BITS; // 60 implicit-zero prefix

    // Syndromes S1, S2, S3, S4 = r(α^i), i = 1..4.
    let mut synd = [0u8; 4];
    for (s, syn) in synd.iter_mut().enumerate() {
        let i = s + 1;
        let mut acc = 0u8;
        for (pos, &bit) in codeword.iter().enumerate() {
            if bit & 1 != 0 {
                // Highest-degree term first: codeword[0] is x^(n0-1-n_shift).
                let deg = TPS_CODEWORD_BITS - 1 - pos + n_shift;
                acc ^= gf.pow_alpha(i * deg);
            }
        }
        *syn = acc;
    }

    // No errors.
    if synd.iter().all(|&s| s == 0) {
        return Some(info_of(codeword));
    }

    // Berlekamp–Massey specialized for t = 2 (binary BCH: S2 = S1^2, S4 = S2^2).
    // σ(x) = 1 + σ1 x + σ2 x^2 with:
    //   σ1 = S1
    //   σ2 = (S3 + S1^3) / S1        (if S1 ≠ 0)
    let s1 = synd[0];
    let s3 = synd[2];
    let (sig1, sig2) = if s1 == 0 {
        // S1 = 0 with nonzero syndrome ⇒ more than 2 errors (uncorrectable for
        // this t = 2 code).
        return None;
    } else {
        let s1_3 = gf.mul(gf.mul(s1, s1), s1);
        let num = s3 ^ s1_3;
        let sig2 = if num == 0 {
            0
        } else {
            // divide num / s1 = num · s1^(order-1-log)
            gf.exp[(gf.log[num as usize] as usize + GF128_ORDER - gf.log[s1 as usize] as usize)
                % GF128_ORDER]
        };
        (s1, sig2)
    };

    // Chien search: find roots of σ(x); an error at codeword position `pos`
    // corresponds to a root α^(-deg). We test every codeword position.
    let mut err = [0u8; TPS_CODEWORD_BITS];
    let mut found = 0usize;
    for (pos, e) in err.iter_mut().enumerate() {
        let deg = TPS_CODEWORD_BITS - 1 - pos + n_shift;
        // Evaluate σ at x = α^(-deg): σ(x) = 1 + σ1·x + σ2·x^2.
        let x = gf.pow_alpha((GF128_ORDER - (deg % GF128_ORDER)) % GF128_ORDER);
        let x2 = gf.mul(x, x);
        let val = 1u8 ^ gf.mul(sig1, x) ^ gf.mul(sig2, x2);
        if val == 0 {
            *e = 1;
            found += 1;
        }
    }

    // The number of distinct roots must equal the error-locator degree (1 if
    // σ2 = 0, else 2); otherwise the pattern is uncorrectable.
    let expected = if sig2 == 0 { 1 } else { 2 };
    if found != expected {
        return None;
    }

    let mut fixed = [0u8; TPS_CODEWORD_BITS];
    for i in 0..TPS_CODEWORD_BITS {
        fixed[i] = codeword[i] ^ err[i];
    }
    // Re-check: the corrected word must satisfy the parity (guards against a
    // false 2-root pattern from >2 errors that happens to have 2 roots).
    let re = tps_bch_encode(&info_of(&fixed));
    if re != fixed {
        return None;
    }
    Some(info_of(&fixed))
}

/// Extracts the 53 information bits (the systematic prefix) from a 67-bit word.
fn info_of(codeword: &[u8]) -> [u8; TPS_INFO_BITS] {
    let mut info = [0u8; TPS_INFO_BITS];
    info.copy_from_slice(&codeword[..TPS_INFO_BITS]);
    info
}

// ── TPS word (EN 300 744 §4.6.2, Table 9) ───────────────────────────────────
//
// The 68-bit TPS block spans one OFDM frame (68 symbols), bit s_l on symbol l:
//   s0            initialization (DBPSK phase reference, not part of the BCH)
//   s1..s16       synchronization word (frame-parity dependent)
//   s17..s22      length indicator
//   s23,s24       frame number (0..3 → "frame 1..4")
//   s25,s26       constellation
//   s27..s29      hierarchy (α)
//   s30..s32      code rate, HP stream
//   s33..s35      code rate, LP stream
//   s36,s37       guard interval
//   s38,s39       transmission mode
//   s40..s47      cell identifier byte
//   s48..s53      reserved, all zero
//   s54..s67      BCH(67,53) parity over s1..s53
//
// "The left most bit is sent first" — s1 is the MSB of the BCH info block.

use crate::fec::PunctureRate;
use crate::modulate::ConstellationOrder;
use crate::waveform::dvb_t::{
    DVB_T_ACTIVE_CARRIERS, DVB_T_TPS_CARRIERS_2K, GuardInterval, wk_prbs,
};
use num_complex::Complex32 as C32;

/// Synchronization word for TPS frames 1 & 3 in a super-frame (`s1..s16`).
pub const TPS_SYNC_WORD_13: u16 = 0b0011_0101_1110_1110;
/// Synchronization word for TPS frames 2 & 4 in a super-frame (`s1..s16`).
pub const TPS_SYNC_WORD_24: u16 = 0b1100_1010_0001_0001;
/// TPS length indicator when cell id is present (31 bits in use, §4.6.2.3).
const TPS_LENGTH_WITH_CELL_ID: u8 = 0b011111;

/// The decoded TPS parameters carried once per 68-symbol frame. Narrowband and
/// broadcast DVB-T share this exact word (only `fs` scales), so it is named for
/// the signalling scheme, not the bandwidth.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TpsWord {
    /// Frame number within the super-frame, 0..=3 (signalled as "frame 1..4").
    pub frame_number: u8,
    /// Payload constellation.
    pub constellation: ConstellationOrder,
    /// Inner (convolutional) code rate for the high-priority stream. In
    /// non-hierarchical transmission this is the single code rate.
    pub code_rate_hp: PunctureRate,
    /// Guard interval.
    pub guard: GuardInterval,
    /// Cell identifier byte for this frame (b15..b8 in frames 1/3, b7..b0 in
    /// frames 2/4; 0 if not provided).
    pub cell_id: u8,
}

impl TpsWord {
    /// Maps a constellation to its 2-bit code (Table 11). QPSK/16-QAM/64-QAM are
    /// the DVB-T set; BPSK/256-QAM (crate extensions) are not DVB-T and map to
    /// QPSK's code — DVB-T links only ever carry the three standard orders.
    fn constellation_code(self) -> u8 {
        match self.constellation {
            ConstellationOrder::Qam16 => 0b01,
            ConstellationOrder::Qam64 => 0b10,
            _ => 0b00, // QPSK (and any non-DVB-T order, which won't occur here)
        }
    }

    fn constellation_from_code(code: u8) -> Option<ConstellationOrder> {
        match code {
            0b00 => Some(ConstellationOrder::Qpsk),
            0b01 => Some(ConstellationOrder::Qam16),
            0b10 => Some(ConstellationOrder::Qam64),
            _ => None, // 0b11 reserved
        }
    }

    /// Maps a code rate to its 3-bit code (Table 13).
    fn rate_code(rate: PunctureRate) -> u8 {
        match rate {
            PunctureRate::R1_2 => 0b000,
            PunctureRate::R2_3 => 0b001,
            PunctureRate::R3_4 => 0b010,
            PunctureRate::R5_6 => 0b011,
            PunctureRate::R7_8 => 0b100,
        }
    }

    fn rate_from_code(code: u8) -> Option<PunctureRate> {
        match code {
            0b000 => Some(PunctureRate::R1_2),
            0b001 => Some(PunctureRate::R2_3),
            0b010 => Some(PunctureRate::R3_4),
            0b011 => Some(PunctureRate::R5_6),
            0b100 => Some(PunctureRate::R7_8),
            _ => None, // reserved
        }
    }

    fn guard_code(guard: GuardInterval) -> u8 {
        match guard {
            GuardInterval::G1_32 => 0b00,
            GuardInterval::G1_16 => 0b01,
            GuardInterval::G1_8 => 0b10,
            GuardInterval::G1_4 => 0b11,
        }
    }

    fn guard_from_code(code: u8) -> GuardInterval {
        match code & 0b11 {
            0b00 => GuardInterval::G1_32,
            0b01 => GuardInterval::G1_16,
            0b10 => GuardInterval::G1_8,
            _ => GuardInterval::G1_4,
        }
    }

    /// The synchronization word for this frame number (frames 1&3 vs 2&4).
    fn sync_word(self) -> u16 {
        if self.frame_number.is_multiple_of(2) {
            TPS_SYNC_WORD_13 // frame numbers 0,2 → "frame 1,3"
        } else {
            TPS_SYNC_WORD_24 // frame numbers 1,3 → "frame 2,4"
        }
    }

    /// Packs the word into the 68 TPS bits `s0..s67` (MSB-first per field). `s0`
    /// is the DBPSK initialization slot, left 0 here (the modulator derives its
    /// absolute reference from `w_k`); `s1..s53` is the BCH info and `s54..s67`
    /// its parity. Non-hierarchical: hierarchy = 000, LP rate = HP rate's code
    /// per §4.6.2.7's "followed by another three bits of value 000" — here we
    /// mirror the HP rate into the LP field as many encoders do; decoders read HP.
    pub fn pack(self) -> [u8; 68] {
        let mut info = [0u8; TPS_INFO_BITS]; // s1..s53 (index 0 = s1)
        let mut set = |range: std::ops::Range<usize>, value: u32| {
            let width = range.len();
            for (j, idx) in range.enumerate() {
                info[idx] = ((value >> (width - 1 - j)) & 1) as u8;
            }
        };
        // s1..s16 sync (info indices 0..16)
        set(0..16, self.sync_word() as u32);
        // s17..s22 length indicator (indices 16..22)
        set(16..22, TPS_LENGTH_WITH_CELL_ID as u32);
        // s23,s24 frame number (indices 22..24)
        set(22..24, (self.frame_number & 0b11) as u32);
        // s25,s26 constellation (indices 24..26)
        set(24..26, self.constellation_code() as u32);
        // s27..s29 hierarchy = non-hierarchical 000 (indices 26..29)
        set(26..29, 0);
        // s30..s32 HP code rate (indices 29..32)
        set(29..32, Self::rate_code(self.code_rate_hp) as u32);
        // s33..s35 LP code rate — non-hierarchical: mirror HP (indices 32..35)
        set(32..35, Self::rate_code(self.code_rate_hp) as u32);
        // s36,s37 guard interval (indices 35..37)
        set(35..37, Self::guard_code(self.guard) as u32);
        // s38,s39 transmission mode = 2K = 00 (indices 37..39)
        set(37..39, 0);
        // s40..s47 cell id (indices 39..47)
        set(39..47, self.cell_id as u32);
        // s48..s53 reserved zero (indices 47..53) — already zero.

        let cw = tps_bch_encode(&info);
        let mut out = [0u8; 68];
        // s0 init left 0; s1..s67 = codeword (info || parity).
        out[1..].copy_from_slice(&cw);
        out
    }

    /// Recovers a [`TpsWord`] from the 68 received TPS bits `s0..s67`,
    /// BCH-correcting up to 2 errors over `s1..s53`. Returns `None` if the BCH is
    /// uncorrectable or a field holds a reserved code. `s0` is ignored (the DBPSK
    /// reference, consumed by the demodulator).
    pub fn unpack(bits: &[u8]) -> Option<Self> {
        if bits.len() != 68 {
            return None;
        }
        let info = tps_bch_decode(&bits[1..])?;
        let get = |range: std::ops::Range<usize>| -> u32 {
            let mut v = 0u32;
            for idx in range {
                v = (v << 1) | (info[idx] & 1) as u32;
            }
            v
        };
        let frame_number = get(22..24) as u8;
        let constellation = Self::constellation_from_code(get(24..26) as u8)?;
        let code_rate_hp = Self::rate_from_code(get(29..32) as u8)?;
        let guard = Self::guard_from_code(get(35..37) as u8);
        let cell_id = get(39..47) as u8;
        Some(Self {
            frame_number,
            constellation,
            code_rate_hp,
            guard,
            cell_id,
        })
    }
}

// ── DBPSK modulation along the symbol axis (EN 300 744 §4.6.3) ──────────────
//
// Every TPS carrier is DBPSK-modulated and conveys the SAME 68-bit message; the
// modulation is differential *along the time (symbol) axis*, re-initialized at
// the start of each frame. TPS cells are at "normal" power (E[c·c*] = 1), i.e.
// ±1 (not the ±4/3 pilot boost).
//
//   symbol l = 0:  Re{c_k} = 2(1/2 − w_k)        (absolute reference, ±1)
//   symbol l > 0:  s_l = 0 → Re{c_k} = Re{c_{l-1,k}}   (keep sign)
//                  s_l = 1 → Re{c_k} = −Re{c_{l-1,k}}  (flip)
//   Im{c_k} = 0 throughout.
//
// The 68 bits are s0..s67 (s0 the initialization bit); the reference sequence
// w_k is the same PRBS used for the pilots (§4.5.2), indexed by absolute carrier
// index k. This lives in the frame-layer orchestrator (cross-symbol per-frame
// state), not a per-symbol `Block`.

/// Number of TPS carriers in 2K mode.
pub const TPS_CARRIER_COUNT: usize = DVB_T_TPS_CARRIERS_2K.len();
/// Number of OFDM symbols spanned by one TPS block (one OFDM frame).
pub const TPS_SYMBOLS_PER_FRAME: usize = 68;

/// The per-carrier DBPSK reference signs for the first symbol of a frame:
/// `sign_k = 2(1/2 − w_k) ∈ {+1, −1}` at each of the 17 TPS carriers, from the
/// `w_k` PRBS at the carriers' absolute indices.
fn tps_reference_signs() -> [f32; TPS_CARRIER_COUNT] {
    let wk = wk_prbs(DVB_T_ACTIVE_CARRIERS);
    let mut signs = [0.0f32; TPS_CARRIER_COUNT];
    for (s, &k) in signs.iter_mut().zip(DVB_T_TPS_CARRIERS_2K.iter()) {
        *s = 2.0 * (0.5 - wk[k] as f32); // w_k=0 → +1, w_k=1 → −1
    }
    signs
}

/// TX-side TPS DBPSK encoder: turns a 68-bit TPS block into the per-symbol,
/// per-carrier TPS cell values, differentially along the symbol axis with the
/// `w_k`-derived reference on symbol 0. Frame-layer state (the running sign per
/// carrier), reset per frame.
#[derive(Debug, Clone)]
pub struct TpsEncoder {
    /// Running DBPSK sign (±1) per TPS carrier.
    signs: [f32; TPS_CARRIER_COUNT],
    /// Current symbol index within the frame (0..=67).
    symbol: usize,
}

impl Default for TpsEncoder {
    fn default() -> Self {
        Self::new()
    }
}

impl TpsEncoder {
    /// A fresh encoder positioned at symbol 0 with the `w_k` reference signs.
    pub fn new() -> Self {
        Self {
            signs: tps_reference_signs(),
            symbol: 0,
        }
    }

    /// Restarts at the start of a frame (symbol 0, `w_k` reference).
    pub fn reset(&mut self) {
        self.signs = tps_reference_signs();
        self.symbol = 0;
    }

    /// Emits the TPS cell values for one symbol: for symbol 0 the absolute
    /// reference (`bit` is `s0`, unused for the phase but part of the block);
    /// for later symbols the differential update by `bit` (`s_l`). Returns the
    /// `TPS_CARRIER_COUNT` complex cell values (all real, ±1), in the carrier
    /// order of [`DVB_T_TPS_CARRIERS_2K`]. Advances the symbol counter.
    pub fn next_symbol(&mut self, bit: u8) -> [C32; TPS_CARRIER_COUNT] {
        if self.symbol > 0 && (bit & 1) == 1 {
            for s in self.signs.iter_mut() {
                *s = -*s;
            }
        }
        // symbol 0 keeps the absolute reference regardless of s0.
        let mut out = [C32::default(); TPS_CARRIER_COUNT];
        for (o, &s) in out.iter_mut().zip(self.signs.iter()) {
            *o = C32::new(s, 0.0);
        }
        self.symbol += 1;
        out
    }
}

/// RX-side TPS DBPSK decoder: recovers the 68-bit block from the received TPS
/// carrier values across a frame. Differential detection compares each symbol's
/// TPS cells against the previous symbol's, averaged over the 17 carriers, so it
/// needs no absolute phase reference (robust to a residual channel phase).
/// Frame-layer state (the previous symbol's cell values), reset per frame.
#[derive(Debug, Clone)]
pub struct TpsDecoder {
    prev: [C32; TPS_CARRIER_COUNT],
    symbol: usize,
    bits: Vec<u8>,
}

impl Default for TpsDecoder {
    fn default() -> Self {
        Self::new()
    }
}

impl TpsDecoder {
    /// A fresh decoder positioned at symbol 0.
    pub fn new() -> Self {
        Self {
            prev: [C32::default(); TPS_CARRIER_COUNT],
            symbol: 0,
            bits: Vec::with_capacity(TPS_SYMBOLS_PER_FRAME),
        }
    }

    /// Restarts at the start of a frame.
    pub fn reset(&mut self) {
        self.prev = [C32::default(); TPS_CARRIER_COUNT];
        self.symbol = 0;
        self.bits.clear();
    }

    /// Feeds one symbol's `TPS_CARRIER_COUNT` received TPS cell values (in the
    /// order of [`DVB_T_TPS_CARRIERS_2K`]). Symbol 0 sets the differential
    /// reference and records `s0 = 0`; each later symbol decides `s_l` from the
    /// sign of the mean per-carrier correlation with the previous symbol.
    pub fn feed_symbol(&mut self, cells: &[C32]) {
        debug_assert!(cells.len() >= TPS_CARRIER_COUNT);
        if self.symbol == 0 {
            self.bits.push(0); // s0 initialization slot
        } else {
            // Correlate each carrier with its previous value: a positive real
            // part means "same sign" (s=0), negative means "flipped" (s=1).
            let mut acc = 0.0f32;
            for (cell, prev) in cells[..TPS_CARRIER_COUNT].iter().zip(self.prev.iter()) {
                acc += (cell * prev.conj()).re;
            }
            self.bits.push(u8::from(acc < 0.0));
        }
        self.prev[..TPS_CARRIER_COUNT].copy_from_slice(&cells[..TPS_CARRIER_COUNT]);
        self.symbol += 1;
    }

    /// Whether a full 68-symbol frame has been fed.
    pub fn is_complete(&self) -> bool {
        self.bits.len() >= TPS_SYMBOLS_PER_FRAME
    }

    /// The recovered TPS bits `s0..s(len-1)` so far (68 once complete).
    pub fn bits(&self) -> &[u8] {
        &self.bits
    }

    /// Decodes the accumulated bits into a [`TpsWord`] once a full frame is
    /// present, `None` otherwise (or if the BCH is uncorrectable).
    pub fn word(&self) -> Option<TpsWord> {
        if !self.is_complete() {
            return None;
        }
        TpsWord::unpack(&self.bits[..TPS_SYMBOLS_PER_FRAME])
    }
}
