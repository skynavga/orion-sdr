use super::tables::{nchar, charn, Table};

/// Encode a free-text string (up to 13 chars, base-42 alphabet) into 9 bytes (71 bits).
/// Returns None if any character is not in the FULL table or the string is too long.
pub fn encode_free_text(text: &str) -> Option<[u8; 9]> {
    if text.len() > 13 { return None; }

    let mut b71 = [0u8; 9];

    // Pad text to 13 characters with spaces
    let mut chars = Vec::with_capacity(13);
    for c in text.chars() {
        chars.push(c);
    }
    while chars.len() < 13 {
        chars.push(' ');
    }

    for &c in &chars {
        let cid = nchar(c, Table::Full)? as u16;
        // b71 = b71 * 42 + cid  (big-endian multiply-and-add across 9 bytes)
        let mut rem = cid;
        for i in (0..9usize).rev() {
            rem += b71[i] as u16 * 42;
            b71[i] = (rem & 0xFF) as u8;
            rem >>= 8;
        }
    }

    Some(b71)
}

/// Decode 9 bytes (71 bits big-endian) back into a free-text string (up to 13 chars).
pub fn decode_free_text(b71: &[u8; 9]) -> String {
    let mut b = *b71;
    let mut chars = [' '; 13];

    for i in (0..13usize).rev() {
        // Divide the big-endian 9-byte integer by 42, get remainder
        let mut rem: u16 = 0;
        for j in 0..9usize {
            rem = (rem << 8) | b[j] as u16;
            b[j] = (rem / 42) as u8;
            rem %= 42;
        }
        chars[i] = charn(rem as u8, Table::Full);
    }

    let s: String = chars.iter().collect();
    s.trim_end_matches(' ').to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn free_text_roundtrip() {
        let cases = &[
            "CQ DX",
            "HELLO WORLD",
            "TNX 73 GL",
            "73",
            " ",
            "",
        ];
        for &text in cases {
            let encoded = encode_free_text(text)
                .expect(&format!("encode failed for '{}'", text));
            let decoded = decode_free_text(&encoded);
            assert_eq!(decoded, text.trim_end(), "Roundtrip failed for '{}'", text);
        }
    }

    #[test]
    fn free_text_too_long() {
        assert!(encode_free_text("12345678901234").is_none());
    }

    #[test]
    fn free_text_invalid_char() {
        // '#' is not in the FULL table
        assert!(encode_free_text("HELLO#WORLD").is_none());
    }
}
