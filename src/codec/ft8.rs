// Copyright (c) 2026 G & R Associates LLC
// SPDX-License-Identifier: MIT OR Apache-2.0

// FT8 channel codec: ties together CRC-14, LDPC(174,91), and Gray code.
//
// Encode path:  77-bit payload → CRC → LDPC → Gray → Ft8Frame (58 tone indices)
// Decode path:  Ft8Frame → inverse Gray → LLRs → LDPC → CRC check → 77-bit payload

use num_complex::Complex32 as C32;
use crate::codec::crc::{ft8_add_crc, ft8_extract_crc, ft8_crc14};
use crate::codec::gray::{gray8_encode, gray8_decode};
use crate::codec::ldpc::{self, ldpc_encode, ldpc_decode_soft};
use crate::message::{CallsignHashTable, Ft8Message, unpack77};
use crate::modulate::Ft8Frame;
use crate::modulate::ft8::{FT8_DATA_SYMS, FT8_FRAME_LEN, FT8_TONE_SPACING_HZ};
use crate::modulate::ft4::{FT4_FRAME_LEN, FT4_TONE_SPACING_HZ};
use crate::sync::{ft8_sync, ft4_sync};

/// 77-bit FT8 payload packed into 10 bytes (MSB first; bits 77..79 of byte 9 are zero).
pub type Ft8Bits = [u8; 10];

/// FT8 channel encoder/decoder.
pub struct Ft8Codec;

impl Ft8Codec {
    /// Encode a 77-bit payload into an `Ft8Frame` of 58 Gray-coded tone indices.
    ///
    /// Steps: payload → CRC-14 appended → LDPC(174,91) → Gray code → 58 tones.
    pub fn encode(payload: &Ft8Bits) -> Ft8Frame {
        // 1. Append CRC-14 to get 91-bit a91
        let mut a91 = [0u8; ldpc::K_BYTES];
        ft8_add_crc(payload, &mut a91);

        // 2. LDPC encode: 91 bits → 174-bit codeword
        let mut codeword = [0u8; ldpc::N_BYTES];
        ldpc_encode(&a91, &mut codeword);

        // 3. Extract 174 bits, group into 58 × 3-bit words, Gray-encode each
        let mut tones = [0u8; FT8_DATA_SYMS];
        let mut mask: u8 = 0x80;
        let mut byte_idx = 0usize;

        for tone in tones.iter_mut() {
            let mut bits3: u8 = 0;
            for bit_pos in (0u8..3).rev() {
                if codeword[byte_idx] & mask != 0 {
                    bits3 |= 1 << bit_pos;
                }
                mask >>= 1;
                if mask == 0 {
                    mask = 0x80;
                    byte_idx += 1;
                }
            }
            *tone = gray8_encode(bits3);
        }

        Ft8Frame::new(tones)
    }

    /// Decode an `Ft8Frame` using hard decisions.
    ///
    /// Applies inverse Gray code, treats each bit as a ±10 LLR, runs LDPC,
    /// then verifies the CRC.  Returns the 77-bit payload on success.
    pub fn decode_hard(frame: &Ft8Frame) -> Option<Ft8Bits> {
        // Build ±10 LLRs from hard tone decisions
        let llr = Self::frame_to_llr_hard(frame);
        Self::decode_llr(&llr)
    }

    /// Decode using soft LLR values produced by a sync/correlator stage.
    ///
    /// `llrs` — 174 floats, LLR = log(P(bit=0)/P(bit=1)), positive ⇒ likely 0.
    pub fn decode_soft(llr: &[f32; ldpc::N]) -> Option<Ft8Bits> {
        Self::decode_llr(llr)
    }

    /// Convert an `Ft8Frame` (hard tone decisions) into 174 LLRs (±10.0).
    pub fn frame_to_llr_hard(frame: &Ft8Frame) -> [f32; ldpc::N] {
        let mut llr = [0.0f32; ldpc::N];
        for (sym_idx, &tone) in frame.0.iter().enumerate() {
            let bin = gray8_decode(tone);
            for bit_pos in 0..3usize {
                let bit = (bin >> (2 - bit_pos)) & 1;
                llr[sym_idx * 3 + bit_pos] = if bit == 0 { 10.0 } else { -10.0 };
            }
        }
        llr
    }

    fn decode_llr(llr: &[f32; ldpc::N]) -> Option<Ft8Bits> {
        let mut plain = [0u8; ldpc::N];
        let errors = ldpc_decode_soft(llr, 20, &mut plain);
        if errors != 0 {
            return None;
        }

        // Pack the first K bits back into bytes
        let mut a91 = [0u8; ldpc::K_BYTES];
        for i in 0..ldpc::K {
            if plain[i] == 1 {
                a91[i / 8] |= 0x80 >> (i % 8);
            }
        }

        // Verify CRC.
        //
        // The CRC covers only the 77-bit payload zero-extended to 82 bits; the
        // 14 CRC bits themselves (bits 77-90 of a91) must NOT be included.
        // We therefore zero out the CRC area (bits 77-95 of the buffer) before
        // calling ft8_crc14 with num_bits=82.  Running ft8_crc14(&a91, 82)
        // without zeroing would include 5 CRC bits in the computation and
        // produce a wrong answer.
        let extracted = ft8_extract_crc(&a91);
        let mut buf = a91;
        buf[9]  &= 0xF8; // zero bits 77-79 (slack bits, also start of CRC)
        buf[10]  = 0;    // zero bits 80-87 (CRC bits 3-10)
        buf[11]  = 0;    // zero bits 88-95 (CRC bits 11-13 + unused)
        let computed = ft8_crc14(&buf, 82);
        if extracted != computed {
            return None;
        }

        // Return the 77 payload bits.  Bits 77-79 of byte 9 are slack; mask
        // them to zero so callers get a canonical representation.
        let mut payload = [0u8; 10];
        payload.copy_from_slice(&a91[..10]);
        payload[9] &= 0xF8;
        Some(payload)
    }
}

// ── Ft8StreamDecoder ──────────────────────────────────────────────────────────

/// Result of one successfully decoded FT8 or FT4 frame.
pub struct Ft8DecodeResult {
    /// Decoded message content.
    pub message:    Ft8Message,
    /// Tone-0 frequency in Hz (carrier of the detected frame).
    pub carrier_hz: f32,
    /// SNR estimate in dB (Costas score, arbitrary but monotone with true SNR).
    pub snr_db:     f32,
}

/// Accumulates IQ samples at 12 kHz and decodes FT8 or FT4 frames.
///
/// Feed samples incrementally with [`feed`].  When the internal buffer reaches
/// `frame_len` samples, a decode attempt is triggered automatically and the
/// results are returned.  Call [`flush`] to attempt a decode on whatever is
/// currently buffered (useful at the end of a session or after a gap).  Call
/// [`clear`] to discard the buffer without decoding.
///
/// The decoder operates at the FT8/FT4 native sample rate of **12 000 Hz**.
/// Callers receiving samples at a higher rate (e.g. 48 kHz) must decimate
/// before feeding.
///
/// A single [`CallsignHashTable`] is maintained across frames so nonstandard
/// callsigns hashed in earlier frames can be resolved in later ones.
pub struct Ft8StreamDecoder {
    buf:        Vec<C32>,
    fs:         f32,
    base_hz:    f32,
    max_hz:     f32,
    frame_len:  usize,
    is_ft8:     bool,
    max_cand:   usize,
    hash_table: CallsignHashTable,
}

impl Ft8StreamDecoder {
    /// Create a decoder for FT8 frames.
    ///
    /// - `fs`       — sample rate (should be 12 000 Hz)
    /// - `base_hz`  — lowest tone-0 frequency to search (Hz)
    /// - `max_hz`   — highest tone-0 frequency to search (Hz)
    /// - `max_cand` — maximum sync candidates to score per decode attempt
    pub fn new_ft8(fs: f32, base_hz: f32, max_hz: f32, max_cand: usize) -> Self {
        Self {
            buf:        Vec::new(),
            fs,
            base_hz,
            max_hz,
            frame_len:  FT8_FRAME_LEN,
            is_ft8:     true,
            max_cand:   max_cand.max(1),
            hash_table: CallsignHashTable::new(),
        }
    }

    /// Create a decoder for FT4 frames.
    pub fn new_ft4(fs: f32, base_hz: f32, max_hz: f32, max_cand: usize) -> Self {
        Self {
            buf:        Vec::new(),
            fs,
            base_hz,
            max_hz,
            frame_len:  FT4_FRAME_LEN,
            is_ft8:     false,
            max_cand:   max_cand.max(1),
            hash_table: CallsignHashTable::new(),
        }
    }

    /// Feed IQ samples into the accumulation buffer.
    ///
    /// If the buffer reaches `frame_len` samples after appending, a decode
    /// attempt is triggered and the results are returned.  Otherwise returns
    /// an empty `Vec`.
    pub fn feed(&mut self, iq: &[C32]) -> Vec<Ft8DecodeResult> {
        self.buf.extend_from_slice(iq);
        if self.buf.len() >= self.frame_len {
            self.decode_buf()
        } else {
            Vec::new()
        }
    }

    /// Attempt a decode on whatever is currently in the buffer.
    ///
    /// Useful at a gap edge when the caller knows a frame has ended but the
    /// buffer may be shorter than `frame_len` (e.g. due to signal dropout).
    /// Returns decoded results; does NOT clear the buffer.
    pub fn flush(&mut self) -> Vec<Ft8DecodeResult> {
        if self.buf.is_empty() {
            return Vec::new();
        }
        self.decode_buf()
    }

    /// Discard all accumulated samples without decoding.
    pub fn clear(&mut self) {
        self.buf.clear();
    }

    /// Number of accumulated samples.
    pub fn len(&self) -> usize {
        self.buf.len()
    }

    /// True if no samples have been accumulated.
    pub fn is_empty(&self) -> bool {
        self.buf.is_empty()
    }

    /// Read-only view of the accumulated IQ buffer.
    pub fn view_buf(&self) -> &[C32] {
        &self.buf
    }

    // ── Internal ──────────────────────────────────────────────────────────

    fn decode_buf(&mut self) -> Vec<Ft8DecodeResult> {
        let mut results = Vec::new();

        let tone_spacing = if self.is_ft8 {
            FT8_TONE_SPACING_HZ
        } else {
            FT4_TONE_SPACING_HZ
        };

        // Clamp search range so base_hz is valid.
        let search_min = self.base_hz;
        let search_max = (self.max_hz + tone_spacing).max(search_min + tone_spacing);

        let candidates = if self.is_ft8 {
            let raw = ft8_sync(
                &self.buf,
                self.fs,
                search_min,
                search_max,
                0, 0,
                self.max_cand,
            );
            // Convert to unified representation for processing below.
            raw.into_iter().map(|r| SyncCandidate {
                freq_bin: r.freq_bin,
                score:    r.score,
                llr:      r.llr,
            }).collect::<Vec<_>>()
        } else {
            let raw = ft4_sync(
                &self.buf,
                self.fs,
                search_min,
                search_max,
                0, 0,
                self.max_cand,
            );
            raw.into_iter().map(|r| SyncCandidate {
                freq_bin: r.freq_bin,
                score:    r.score,
                llr:      r.llr,
            }).collect::<Vec<_>>()
        };

        for cand in candidates {
            let payload = if self.is_ft8 {
                Ft8Codec::decode_soft(&cand.llr)
            } else {
                crate::codec::ft4::Ft4Codec::decode_soft(&cand.llr)
            };

            if let Some(bits) = payload {
                let message = unpack77(&bits, &self.hash_table);
                let carrier_hz = self.base_hz + cand.freq_bin as f32 * tone_spacing;
                results.push(Ft8DecodeResult {
                    message,
                    carrier_hz,
                    snr_db: cand.score,
                });
                // Stop after the first CRC-passing candidate.
                break;
            }
        }

        results
    }
}

/// Internal unified sync result (avoids duplicating processing logic for FT8/FT4).
struct SyncCandidate {
    freq_bin: usize,
    score:    f32,
    llr:      [f32; ldpc::N],
}
