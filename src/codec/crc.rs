// CRC-14 for FT8/FT4, polynomial 0x2757.
//
// Algorithm: straight-line bit-by-bit shift register, MSB first.
// Ported from ft8_lib (kgoba, MIT licence).
//
// ── Polynomial note ────────────────────────────────────────────────────────
// Several secondary sources quote 0x6757 — that is wrong.  The correct value
// from the ft8_lib reference implementation (constants.h) is 0x2757.
//
// ── CRC domain ─────────────────────────────────────────────────────────────
// The CRC is computed over the 77-bit payload zero-extended to 82 bits (i.e.
// 5 zero bits appended), NOT over the full 91-bit a91 block.  In other words
// the 14-bit CRC field itself is never fed back into the CRC calculation.
// Concretely: call ft8_crc14(buf, 82) where buf has the 77 payload bits in
// positions 0-76 and zeros in positions 77-81.  The CRC result occupies
// positions 77-90 of the a91 layout.
//
// ── Byte 9 slack bits ──────────────────────────────────────────────────────
// The 77-bit payload is packed MSB-first into 10 bytes.  Bit 76 (last payload
// bit) lands at byte 9 bit 3 (counting from MSB = bit 7).  Bits 77-79 (byte 9
// bits 2-0) are unused slack and must be zero.  Mask: payload[9] &= 0xF8.
// For a "77 bits all-ones" payload, byte 9 = 0xF8 (not 0xFF or 0xFE).
//
// ── a91 layout (12 bytes = 96 bits) ────────────────────────────────────────
//   bits  0-76:  77-bit payload
//   bits 77-90:  14-bit CRC  (stored as: a91[9] bits 2-0, a91[10], a91[11] bits 7-5)
//   bits 91-95:  unused (zero)

const POLY: u16 = 0x2757;
const WIDTH: u32 = 14;
const TOPBIT: u16 = 1 << (WIDTH - 1);

/// Compute CRC-14 over the first `num_bits` bits of `message` (MSB first).
pub fn ft8_crc14(message: &[u8], num_bits: usize) -> u16 {
    let mut remainder: u16 = 0;
    let mut idx_byte = 0usize;

    for idx_bit in 0..num_bits {
        if idx_bit % 8 == 0 {
            remainder ^= (message[idx_byte] as u16) << (WIDTH - 8);
            idx_byte += 1;
        }
        if remainder & TOPBIT != 0 {
            remainder = (remainder << 1) ^ POLY;
        } else {
            remainder <<= 1;
        }
    }

    remainder & ((TOPBIT << 1) - 1)
}

/// Append a 14-bit CRC to 77 bits of payload, producing a 91-bit `a91` block.
///
/// Layout: bits 0–76 = payload, bits 77–90 = CRC (MSB first).
/// The CRC is computed over the 77-bit payload zero-padded to 82 bits
/// (i.e., 96 - 14 = 82 bits processed).
pub fn ft8_add_crc(payload: &[u8; 10], a91: &mut [u8; 12]) {
    // Copy 10 bytes (covers 77 payload bits + 3 slack bits at byte boundary)
    for i in 0..10 {
        a91[i] = payload[i];
    }
    // Clear the 3 low bits of byte 9 and all of byte 10 (slack + CRC area)
    a91[9] &= 0xF8;
    a91[10] = 0;
    a91[11] = 0;

    // CRC is computed over bits 0..81 (77 payload + 5 appended zeros)
    let checksum = ft8_crc14(a91, 96 - 14);

    // Pack the 14-bit CRC into bits 77..90 (a91[9] bits 2..0, a91[10], a91[11] bits 7..5)
    a91[9]  |= (checksum >> 11) as u8;
    a91[10]  = (checksum >> 3) as u8;
    a91[11]  = (checksum << 5) as u8;
}

/// Extract the 14-bit CRC from bits 77..90 of a packed 91-bit `a91` block.
pub fn ft8_extract_crc(a91: &[u8; 12]) -> u16 {
    ((a91[9] & 0x07) as u16) << 11
        | (a91[10] as u16) << 3
        | (a91[11] >> 5) as u16
}

#[cfg(test)]
mod tests {
    use super::*;

    // Helper: compute CRC the same way ft8_add_crc does — over the 77-bit payload
    // zero-extended to 82 bits (i.e. clear the bottom 5 bits of a 10-byte window).
    fn recompute_crc(payload: &[u8; 10]) -> u16 {
        let mut buf = [0u8; 12];
        for i in 0..10 { buf[i] = payload[i]; }
        buf[9] &= 0xF8; // zero bits 77-79 (the 3 slack bits at the end of byte 9)
        ft8_crc14(&buf, 82)
    }

    #[test]
    fn crc_roundtrip_all_zeros() {
        let payload = [0u8; 10];
        let mut a91 = [0u8; 12];
        ft8_add_crc(&payload, &mut a91);
        let extracted = ft8_extract_crc(&a91);
        let computed = recompute_crc(&payload);
        assert_eq!(extracted, computed);
    }

    #[test]
    fn crc_roundtrip_all_ones() {
        // 77 bits all-ones: bits 0-76 set; bits 77-79 must be zero.
        // Bit 76 = byte 9 bit 4 (MSB=bit 0), so byte 9 = 0b11111000 = 0xF8.
        let mut payload = [0xFFu8; 10];
        payload[9] = 0xF8;
        let mut a91 = [0u8; 12];
        ft8_add_crc(&payload, &mut a91);
        let extracted = ft8_extract_crc(&a91);
        let computed = recompute_crc(&payload);
        assert_eq!(extracted, computed);
    }

    #[test]
    fn crc_is_nonzero_for_nonzero_payload() {
        let mut payload = [0u8; 10];
        payload[0] = 0xAB;
        let mut a91 = [0u8; 12];
        ft8_add_crc(&payload, &mut a91);
        let extracted = ft8_extract_crc(&a91);
        assert_ne!(extracted, 0, "CRC should be non-zero for non-zero payload");
    }

    #[test]
    fn crc_changes_with_payload() {
        let mut payload_a = [0u8; 10];
        payload_a[0] = 0x01;
        let mut payload_b = [0u8; 10];
        payload_b[0] = 0x02;
        let mut a91_a = [0u8; 12];
        let mut a91_b = [0u8; 12];
        ft8_add_crc(&payload_a, &mut a91_a);
        ft8_add_crc(&payload_b, &mut a91_b);
        assert_ne!(ft8_extract_crc(&a91_a), ft8_extract_crc(&a91_b));
    }
}
