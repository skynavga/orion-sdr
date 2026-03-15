// Gray code tables for FT8 (8-FSK, 3 bits/symbol) and FT4 (4-FSK, 2 bits/symbol).
//
// FT8: 3-bit index → tone index.  Table from ft8_lib `kFT8_Gray_map`.
// FT4: 2-bit index → tone index.  Table from ft8_lib `kFT4_Gray_map`.

/// FT8 Gray code: binary index (0–7) → tone index (0–7).
const FT8_GRAY: [u8; 8] = [0, 1, 3, 2, 5, 6, 4, 7];

/// FT8 inverse Gray: tone index (0–7) → binary index (0–7).
/// Precomputed inverse of FT8_GRAY.
const FT8_GRAY_INV: [u8; 8] = {
    let mut inv = [0u8; 8];
    let mut i = 0u8;
    loop {
        inv[FT8_GRAY[i as usize] as usize] = i;
        i += 1;
        if i == 8 { break; }
    }
    inv
};

/// FT4 Gray code: binary index (0–3) → tone index (0–3).
const FT4_GRAY: [u8; 4] = [0, 1, 3, 2];

/// FT4 inverse Gray: tone index (0–3) → binary index (0–3).
const FT4_GRAY_INV: [u8; 4] = {
    let mut inv = [0u8; 4];
    let mut i = 0u8;
    loop {
        inv[FT4_GRAY[i as usize] as usize] = i;
        i += 1;
        if i == 4 { break; }
    }
    inv
};

/// Map a 3-bit binary index to an FT8 tone index via Gray code.
#[inline]
pub fn gray8_encode(bin_idx: u8) -> u8 {
    FT8_GRAY[(bin_idx & 0x7) as usize]
}

/// Map an FT8 tone index back to a 3-bit binary index (inverse Gray).
#[inline]
pub fn gray8_decode(tone: u8) -> u8 {
    FT8_GRAY_INV[(tone & 0x7) as usize]
}

/// Map a 2-bit binary index to an FT4 tone index via Gray code.
#[inline]
pub fn gray4_encode(bin_idx: u8) -> u8 {
    FT4_GRAY[(bin_idx & 0x3) as usize]
}

/// Map an FT4 tone index back to a 2-bit binary index (inverse Gray).
#[inline]
pub fn gray4_decode(tone: u8) -> u8 {
    FT4_GRAY_INV[(tone & 0x3) as usize]
}
