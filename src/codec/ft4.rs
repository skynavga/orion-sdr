// FT4 channel codec: same LDPC(174,91) as FT8 but 4-FSK (2 bits/symbol).
//
// FT4 difference from FT8:
//   - 77-bit payload is XOR'd with a 10-byte pseudorandom sequence before CRC+LDPC.
//   - 174 codeword bits are split into 87 × 2-bit groups (not 58 × 3-bit).
//   - Gray code is the 4-symbol table [0,1,3,2] (not the 8-symbol FT8 table).

use crate::codec::crc::{ft8_add_crc, ft8_extract_crc, ft8_crc14};
use crate::codec::gray::{gray4_encode, gray4_decode};
use crate::codec::ldpc::{self, ldpc_encode, ldpc_decode_soft};
use crate::modulate::Ft4Frame;
use crate::modulate::ft4::FT4_DATA_SYMS;

/// 77-bit FT4 payload packed into 10 bytes (MSB first).
pub type Ft4Bits = [u8; 10];

// FT4 XOR scramble sequence — applied to payload before CRC+LDPC.
// Source: ft8_lib `kFT4_XOR_sequence`.
const FT4_XOR: [u8; 10] = [0x4A, 0x5E, 0x89, 0xB4, 0xB0, 0x8A, 0x79, 0x55, 0xBE, 0x28];

/// FT4 channel encoder/decoder.
pub struct Ft4Codec;

impl Ft4Codec {
    /// Encode a 77-bit payload into an `Ft4Frame` of 87 Gray-coded tone indices.
    pub fn encode(payload: &Ft4Bits) -> Ft4Frame {
        // 1. XOR payload with scramble sequence
        let mut scrambled = [0u8; 10];
        for i in 0..10 {
            scrambled[i] = payload[i] ^ FT4_XOR[i];
        }

        // 2. Append CRC-14
        let mut a91 = [0u8; ldpc::K_BYTES];
        ft8_add_crc(&scrambled, &mut a91);

        // 3. LDPC encode
        let mut codeword = [0u8; ldpc::N_BYTES];
        ldpc_encode(&a91, &mut codeword);

        // 4. Extract 174 bits, group into 87 × 2-bit words, Gray-encode each
        let mut tones = [0u8; FT4_DATA_SYMS];
        let mut mask: u8 = 0x80;
        let mut byte_idx = 0usize;

        for tone in tones.iter_mut() {
            let mut bits2: u8 = 0;
            for bit_pos in (0u8..2).rev() {
                if codeword[byte_idx] & mask != 0 {
                    bits2 |= 1 << bit_pos;
                }
                mask >>= 1;
                if mask == 0 {
                    mask = 0x80;
                    byte_idx += 1;
                }
            }
            *tone = gray4_encode(bits2);
        }

        Ft4Frame::new(tones)
    }

    /// Decode an `Ft4Frame` using hard decisions.
    pub fn decode_hard(frame: &Ft4Frame) -> Option<Ft4Bits> {
        let llr = Self::frame_to_llr_hard(frame);
        Self::decode_llr(&llr)
    }

    /// Decode using soft LLR values.
    pub fn decode_soft(llr: &[f32; ldpc::N]) -> Option<Ft4Bits> {
        Self::decode_llr(llr)
    }

    /// Convert an `Ft4Frame` (hard tone decisions) into 174 LLRs (±10.0).
    pub fn frame_to_llr_hard(frame: &Ft4Frame) -> [f32; ldpc::N] {
        let mut llr = [0.0f32; ldpc::N];
        for (sym_idx, &tone) in frame.0.iter().enumerate() {
            let bin = gray4_decode(tone);
            for bit_pos in 0..2usize {
                let bit = (bin >> (1 - bit_pos)) & 1;
                llr[sym_idx * 2 + bit_pos] = if bit == 0 { 10.0 } else { -10.0 };
            }
        }
        llr
    }

    fn decode_llr(llr: &[f32; ldpc::N]) -> Option<Ft4Bits> {
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

        // Verify CRC — same logic as FT8: zero the CRC area before recomputing
        // so the 14 CRC bits don't corrupt the computation.  See ft8.rs for a
        // detailed explanation.
        let extracted = ft8_extract_crc(&a91);
        let mut buf = a91;
        buf[9]  &= 0xF8;
        buf[10]  = 0;
        buf[11]  = 0;
        let computed = ft8_crc14(&buf, 82);
        if extracted != computed {
            return None;
        }

        // Un-XOR to recover the original (pre-scramble) payload.
        // Note: a91 holds the *scrambled* payload; XOR again to undo it.
        // Also zero the 3 slack bits of byte 9 for a canonical representation.
        let mut payload = [0u8; 10];
        for i in 0..10 {
            payload[i] = a91[i] ^ FT4_XOR[i];
        }
        payload[9] &= 0xF8;
        Some(payload)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ft4_codec_roundtrip_zeros() {
        let payload: Ft4Bits = [0u8; 10];
        let frame = Ft4Codec::encode(&payload);
        let decoded = Ft4Codec::decode_hard(&frame).expect("FT4 hard decode failed (zeros)");
        assert_eq!(payload, decoded);
    }

    #[test]
    fn ft4_codec_roundtrip_pattern() {
        let payload: Ft4Bits = [0xDE, 0xAD, 0xBE, 0xEF, 0xCA, 0xFE, 0x01, 0x23, 0x45, 0x60];
        let frame = Ft4Codec::encode(&payload);
        let decoded = Ft4Codec::decode_hard(&frame).expect("FT4 hard decode failed (pattern)");
        assert_eq!(payload, decoded);
    }

    #[test]
    fn ft4_codec_frame_has_valid_tones() {
        let payload: Ft4Bits = [0xAA; 10];
        let frame = Ft4Codec::encode(&payload);
        for &t in frame.0.iter() {
            assert!(t < 4, "FT4 tone index out of range: {t}");
        }
    }

    #[test]
    fn ft4_xor_scramble_changes_frame() {
        // Two different payloads that differ only in the XOR sequence should produce different frames
        let payload_a: Ft4Bits = [0u8; 10];
        let payload_b: Ft4Bits = [0x4A, 0x5E, 0x89, 0xB4, 0xB0, 0x8A, 0x79, 0x55, 0xBE, 0x28];
        let frame_a = Ft4Codec::encode(&payload_a);
        let frame_b = Ft4Codec::encode(&payload_b);
        assert_ne!(frame_a, frame_b, "XOR scramble should change the encoded frame");
    }
}
