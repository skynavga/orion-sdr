use std::collections::HashMap;
use super::tables::{nchar, charn, Table};

pub const NTOKENS: u32 = 2_063_592;
pub const MAX22: u32 = 4_194_304; // 2^22

/// Callsign hash table keyed by 22-bit hash, storing original callsign strings.
pub struct CallsignHashTable {
    inner: HashMap<u32, String>,
}

impl CallsignHashTable {
    pub fn new() -> Self {
        Self { inner: HashMap::new() }
    }

    /// Compute the 22-bit hash of a callsign and store it.
    /// Returns (n22, n12, n10).
    pub fn save(&mut self, call: &str) -> (u32, u16, u16) {
        let n22 = hash22(call);
        let n12 = (n22 >> 10) as u16;
        let n10 = (n22 >> 12) as u16;
        self.inner.insert(n22, call.to_string());
        (n22, n12, n10)
    }

    /// Look up a callsign by its n12 (12-bit) hash.
    pub fn lookup_n12(&self, n12: u16) -> Option<&str> {
        // n12 = n22 >> 10, so we need n22 in [n12<<10 .. (n12+1)<<10)
        let lo = (n12 as u32) << 10;
        let hi = lo + 1024;
        for (&k, v) in &self.inner {
            if k >= lo && k < hi {
                return Some(v.as_str());
            }
        }
        None
    }

    /// Look up by full n22.
    pub fn lookup_n22(&self, n22: u32) -> Option<&str> {
        self.inner.get(&n22).map(|s| s.as_str())
    }
}

impl Default for CallsignHashTable {
    fn default() -> Self { Self::new() }
}

/// Compute the 22-bit hash of a callsign (base-38 encode then multiply-shift).
pub fn hash22(call: &str) -> u32 {
    let mut n58: u64 = 0;
    let mut i = 0;
    for c in call.chars() {
        if i >= 11 { break; }
        let j = nchar(c, Table::AlphanumSpaceSlash).unwrap_or(0) as u64;
        n58 = 38 * n58 + j;
        i += 1;
    }
    // Pad with spaces (index 0 in AlphanumSpaceSlash)
    while i < 11 {
        n58 = 38 * n58;
        i += 1;
    }
    let n22 = ((47_055_833_459_u64.wrapping_mul(n58)) >> (64 - 22)) & 0x3F_FFFF;
    n22 as u32
}

/// Try to pack a callsign as a standard 6-char basecall.
/// Returns `None` if the callsign is not a valid standard basecall.
/// Returns `Some(n28)` in [0, MAX22) on success.
pub fn pack_basecall(call: &str) -> Option<u32> {
    let len = call.len();
    if len <= 2 { return None; }

    let bytes = call.as_bytes();
    let mut c6 = [b' '; 6];

    if call.starts_with("3DA0") && len > 4 && len <= 7 {
        // Swaziland: 3DA0XYZ -> 3D0XYZ
        c6[..3].copy_from_slice(b"3D0");
        let rest = &bytes[4..len];
        c6[3..3 + rest.len()].copy_from_slice(rest);
    } else if call.starts_with("3X") && len >= 3 && bytes[2].is_ascii_uppercase() && len <= 7 {
        // Guinea: 3XA0XYZ -> QA0XYZ
        c6[0] = b'Q';
        let rest = &bytes[2..len];
        c6[1..1 + rest.len()].copy_from_slice(rest);
    } else if len >= 3 && bytes[2].is_ascii_digit() && len <= 6 {
        // AB0XYZ — digit at position 2
        c6[..len].copy_from_slice(bytes);
    } else if len >= 2 && bytes[1].is_ascii_digit() && len <= 5 {
        // A0XYZ — digit at position 1, right-align with leading space
        c6[1..1 + len].copy_from_slice(bytes);
    } else {
        return None;
    }

    let i0 = nchar(c6[0] as char, Table::AlphanumSpace)?;
    let i1 = nchar(c6[1] as char, Table::Alphanum)?;
    let i2 = nchar(c6[2] as char, Table::Numeric)?;
    let i3 = nchar(c6[3] as char, Table::LettersSpace)?;
    let i4 = nchar(c6[4] as char, Table::LettersSpace)?;
    let i5 = nchar(c6[5] as char, Table::LettersSpace)?;

    let mut n = i0 as u32;
    n = n * 36 + i1 as u32;
    n = n * 10 + i2 as u32;
    n = n * 27 + i3 as u32;
    n = n * 27 + i4 as u32;
    n = n * 27 + i5 as u32;
    Some(n)
}

/// Unpack a standard basecall from its packed integer (result of `pack_basecall`).
fn unpack_basecall(n: u32) -> Option<String> {
    let mut n = n;
    let c5 = charn((n % 27) as u8, Table::LettersSpace); n /= 27;
    let c4 = charn((n % 27) as u8, Table::LettersSpace); n /= 27;
    let c3 = charn((n % 27) as u8, Table::LettersSpace); n /= 27;
    let c2 = charn((n % 10) as u8, Table::Numeric);       n /= 10;
    let c1 = charn((n % 36) as u8, Table::Alphanum);      n /= 36;
    let c0 = charn((n % 37) as u8, Table::AlphanumSpace);

    // Build raw 6-char string, trim spaces
    let raw: String = [c0, c1, c2, c3, c4, c5].iter().collect();
    let trimmed = raw.trim_matches(' ');

    // Apply reverse work-arounds
    if trimmed.starts_with("3D0") && trimmed.len() > 3 && !trimmed.chars().nth(3).map(|c| c == ' ').unwrap_or(true) {
        // 3D0XYZ -> 3DA0XYZ
        let mut s = String::from("3DA0");
        s.push_str(&trimmed[3..]);
        Some(s)
    } else if trimmed.starts_with('Q') && trimmed.len() > 1 && trimmed.chars().nth(1).map(|c| c.is_ascii_uppercase()).unwrap_or(false) {
        // QA0XYZ -> 3XA0XYZ
        let mut s = String::from("3X");
        s.push_str(&trimmed[1..]);
        Some(s)
    } else {
        if trimmed.len() < 3 { return None; }
        Some(trimmed.to_string())
    }
}

/// Pack a callsign into a 28-bit token value.
/// Sets `ip` to true if the callsign has /R or /P suffix.
/// Returns None on failure.
pub fn pack28(call: &str, ht: &mut CallsignHashTable, ip: &mut bool) -> Option<u32> {
    *ip = false;

    // Special tokens
    if call == "DE"  { return Some(0); }
    if call == "QRZ" { return Some(1); }
    if call == "CQ"  { return Some(2); }

    let len = call.len();

    // CQ NNN or CQ ABCD
    if call.starts_with("CQ ") && len < 8 {
        let v = parse_cq_modifier(call)?;
        return Some(3 + v);
    }

    // Strip /R or /P suffix
    let (base, has_suffix) = if call.ends_with("/R") || call.ends_with("/P") {
        (&call[..len - 2], true)
    } else {
        (call, false)
    };

    if has_suffix { *ip = true; }

    // Try standard basecall
    if let Some(n28) = pack_basecall(base) {
        ht.save(call);
        return Some(NTOKENS + MAX22 + n28);
    }

    // Non-standard: 3–11 chars, ALPHANUM_SPACE_SLASH
    if len >= 3 && len <= 11 {
        let all_valid = call.chars().all(|c| nchar(c, Table::AlphanumSpaceSlash).is_some());
        if all_valid {
            *ip = false;
            let (n22, _, _) = ht.save(call);
            return Some(NTOKENS + n22);
        }
    }

    None
}

/// Unpack a callsign from a 28-bit value. `ip` is the suffix flag, `i3` is the message type.
pub fn unpack28(n28: u32, ip: bool, i3: u8, ht: &CallsignHashTable) -> Option<String> {
    if n28 < NTOKENS {
        if n28 <= 2 {
            return Some(match n28 { 0 => "DE", 1 => "QRZ", _ => "CQ" }.to_string());
        }
        if n28 <= 1002 {
            // CQ NNN
            return Some(format!("CQ {:03}", n28 - 3));
        }
        if n28 <= 532_443 {
            // CQ ABCD (base-27 1-4 letters)
            let mut n = n28 - 1003;
            let mut aaaa = [' '; 4];
            for i in (0..4).rev() {
                aaaa[i] = charn((n % 27) as u8, Table::LettersSpace);
                n /= 27;
            }
            let s: String = aaaa.iter().collect();
            let s = s.trim_matches(' ');
            return Some(format!("CQ {}", s));
        }
        return None; // Unspecified range
    }

    let n28 = n28 - NTOKENS;
    if n28 < MAX22 {
        // 22-bit hash lookup
        let found = ht.lookup_n22(n28);
        return Some(found.map(|s| format!("<{}>", s)).unwrap_or_else(|| "<...>".to_string()));
    }

    // Standard callsign
    let n = n28 - MAX22;
    let mut call = unpack_basecall(n)?;

    if ip {
        match i3 {
            1 => call.push_str("/R"),
            2 => call.push_str("/P"),
            _ => return None,
        }
    }

    Some(call)
}

/// Parse CQ modifier: "CQ NNN" -> Some(nnn), "CQ ABCD" -> Some(1000+m).
fn parse_cq_modifier(s: &str) -> Option<u32> {
    // s starts with "CQ "
    let rest = &s[3..];
    let mut nnum = 0usize;
    let mut nlet = 0usize;
    let mut m: u32 = 0;

    for c in rest.chars() {
        if c == ' ' { break; }
        if c.is_ascii_digit() {
            nnum += 1;
        } else if c.is_ascii_uppercase() {
            nlet += 1;
            m = 27 * m + (c as u32 - 'A' as u32 + 1);
        } else {
            return None;
        }
    }

    if nnum == 3 && nlet == 0 {
        rest[..3].parse::<u32>().ok()
    } else if nnum == 0 && nlet >= 1 && nlet <= 4 {
        Some(1000 + m)
    } else {
        None
    }
}

/// Pack a full callsign (up to 11 chars) into 58 bits using base-38.
/// Returns None if any character is not in ALPHANUM_SPACE_SLASH.
pub fn pack58(call: &str, ht: &mut CallsignHashTable) -> Option<u64> {
    // Strip surrounding angle brackets if present
    let src = if call.starts_with('<') && call.ends_with('>') {
        &call[1..call.len() - 1]
    } else {
        call
    };

    let mut n58: u64 = 0;
    for c in src.chars() {
        let j = nchar(c, Table::AlphanumSpaceSlash)?;
        n58 = n58 * 38 + j as u64;
    }
    ht.save(src);
    Some(n58)
}

/// Unpack a 58-bit value into a callsign string using base-38.
pub fn unpack58(n58: u64, ht: &mut CallsignHashTable) -> String {
    let mut n = n58;
    let mut chars = [' '; 11];
    for i in (0..11).rev() {
        chars[i] = charn((n % 38) as u8, Table::AlphanumSpaceSlash);
        n /= 38;
    }
    let s: String = chars.iter().collect();
    let s = s.trim_matches(' ').to_string();
    if s.len() >= 3 {
        ht.save(&s);
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pack_basecall_w9xyz() {
        // "W9XYZ" has digit at position 1 -> right-align to " W9XYZ"
        // i0=' '=0, i1='W'=32, i2='9'=9, i3='X'=24, i4='Y'=25, i5='Z'=26
        let n = pack_basecall("W9XYZ").expect("W9XYZ should be valid");
        // n = 0*36 + 32 = 32; *10+9=329; *27+24=8907; *27+25=240514; *27+26=6493904
        assert_eq!(n, 6_493_904);
    }

    #[test]
    fn pack_basecall_roundtrip() {
        for call in &["W9XYZ", "KD9ABC", "VE3XYZ", "G0ABC"] {
            let n = pack_basecall(call).expect(call);
            let back = unpack_basecall(n).expect("unpack");
            assert_eq!(&back, call, "Roundtrip failed for {}", call);
        }
    }
}
