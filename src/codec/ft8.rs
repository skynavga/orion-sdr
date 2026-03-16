// FT8 channel codec: ties together CRC-14, LDPC(174,91), and Gray code.
//
// Encode path:  77-bit payload → CRC → LDPC → Gray → Ft8Frame (58 tone indices)
// Decode path:  Ft8Frame → inverse Gray → LLRs → LDPC → CRC check → 77-bit payload

use crate::codec::crc::{ft8_add_crc, ft8_extract_crc, ft8_crc14};
use crate::codec::gray::{gray8_encode, gray8_decode};
use crate::codec::ldpc::{self, ldpc_encode, ldpc_decode_soft};
use crate::modulate::Ft8Frame;
use crate::modulate::ft8::FT8_DATA_SYMS;

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
