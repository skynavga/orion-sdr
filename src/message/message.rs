use super::callsign::{CallsignHashTable, pack28, unpack28, pack58};
use super::grid::{GridField, unpackgrid, gridfield_to_pack};
use super::free_text::{encode_free_text, decode_free_text};

/// A 77-bit FT8/FT4 payload packed into 10 bytes (MSB-first, bits 77–79 of byte 9 are zero).
pub type Payload77 = [u8; 10];

/// The "extra" field in a Type-4 (nonstandard) message.
#[derive(Debug, Clone, PartialEq)]
pub enum NonstdExtra {
    RRR,
    RR73,
    Seventy3,
    None,
}

/// A decoded FT8/FT4 message.
#[derive(Debug, Clone, PartialEq)]
pub enum Ft8Message {
    /// i3=1 or i3=2: two callsigns plus a grid/report/token.
    Standard {
        call_to: String,
        call_de: String,
        extra: GridField,
    },
    /// i3=0, n3=0: free text, up to 13 chars, base-42 alphabet.
    FreeText(String),
    /// i3=4: one 58-bit nonstandard callsign + one 12-bit hashed callsign.
    NonStd {
        call_to: String,
        call_de: String,
        extra: NonstdExtra,
    },
    /// i3=0, n3=5: telemetry — 71 arbitrary bits.
    Telemetry([u8; 9]),
    /// Unknown or unimplemented type.
    Unknown(Payload77),
}

/// Encode a message into a 77-bit payload.
/// Returns `None` if the message cannot be encoded (e.g. invalid callsign).
pub fn pack77(msg: &Ft8Message, ht: &mut CallsignHashTable) -> Option<Payload77> {
    match msg {
        Ft8Message::Standard { call_to, call_de, extra } => {
            pack77_standard(call_to, call_de, extra, ht)
        }
        Ft8Message::FreeText(text) => {
            pack77_free_text(text)
        }
        Ft8Message::NonStd { call_to, call_de, extra } => {
            pack77_nonstd(call_to, call_de, extra, ht)
        }
        Ft8Message::Telemetry(data) => {
            Some(pack77_telemetry(data))
        }
        Ft8Message::Unknown(p) => Some(*p),
    }
}

/// Decode a 77-bit payload into a message.
pub fn unpack77(payload: &Payload77, ht: &CallsignHashTable) -> Ft8Message {
    let i3 = (payload[9] >> 3) & 0x07;
    let n3 = ((payload[8] << 2) | (payload[9] >> 6)) & 0x07;

    match i3 {
        0 => match n3 {
            0 => {
                // Free text: shift payload bits right by 1 to recover b71
                let b71 = payload_to_b71(payload);
                Ft8Message::FreeText(decode_free_text(&b71))
            }
            5 => {
                let b71 = payload_to_b71(payload);
                Ft8Message::Telemetry(b71)
            }
            _ => Ft8Message::Unknown(*payload),
        },
        1 | 2 => unpack77_standard(payload, i3, ht),
        4 => unpack77_nonstd(payload, ht),
        _ => Ft8Message::Unknown(*payload),
    }
}

// ---------------------------------------------------------------------------
// Pack helpers
// ---------------------------------------------------------------------------

fn pack77_standard(
    call_to: &str,
    call_de: &str,
    extra: &GridField,
    ht: &mut CallsignHashTable,
) -> Option<Payload77> {
    let mut ipa = false;
    let mut ipb = false;

    let n28a = pack28(call_to, ht, &mut ipa)?;
    let n28b = pack28(call_de, ht, &mut ipb)?;

    // Determine i3
    let mut i3: u8 = 1;
    if call_to.ends_with("/P") || call_de.ends_with("/P") {
        i3 = 2;
    }

    // Build n29a, n29b (28-bit value shifted left by 1, with ip in LSB)
    let mut n29a = (n28a << 1) | ipa as u32;
    let n29b = (n28b << 1) | ipb as u32;

    // Override ipa if suffix detected in call_to
    if call_to.ends_with("/R") {
        n29a = (n28a << 1) | 1;
    } else if call_to.ends_with("/P") {
        n29a = (n28a << 1) | 1;
        i3 = 2;
    }

    let (igrid4_15bits, ir) = gridfield_to_pack(extra);
    let igrid4_packed = igrid4_15bits | (if ir { 0x8000 } else { 0 });

    // Pack into 10 bytes: 29 + 29 + 1[ir] + 15[grid] + 3[i3] = 77 bits
    // The ft8_lib packs ir separately as a bit within the grid word area,
    // but in the payload layout it goes: n29a(29) n29b(29) ir(1) igrid4(15) i3(3)
    // That's 77 bits total. Let's follow ft8_lib exactly.
    //
    // payload[7] = (n29b << 6) | (ir << 5) | (igrid4 >> 10)
    // where igrid4 is the 15-bit value

    let ir_bit = if ir { 1u8 } else { 0u8 };
    let igrid4 = igrid4_15bits; // 15-bit grid value (no ir bit)

    let mut p = [0u8; 10];
    p[0] = (n29a >> 21) as u8;
    p[1] = (n29a >> 13) as u8;
    p[2] = (n29a >> 5) as u8;
    p[3] = ((n29a << 3) as u8) | (n29b >> 26) as u8;
    p[4] = (n29b >> 18) as u8;
    p[5] = (n29b >> 10) as u8;
    p[6] = (n29b >> 2) as u8;
    p[7] = ((n29b << 6) as u8) | (ir_bit << 5) | ((igrid4 >> 10) as u8);
    p[8] = (igrid4 >> 2) as u8;
    p[9] = ((igrid4 << 6) as u8) | ((i3 << 3) as u8);

    // Suppress unused warning for igrid4_packed
    let _ = igrid4_packed;

    Some(p)
}

fn pack77_free_text(text: &str) -> Option<Payload77> {
    let b71 = encode_free_text(text)?;
    let mut p = [0u8; 10];
    // Left-shift b71 by 1 bit (ft8_lib ftx_message_encode_telemetry, loop i=8..0)
    let mut carry: u8 = 0;
    for i in (0..9usize).rev() {
        p[i] = (b71[i] << 1) | (carry >> 7);
        carry = b71[i] & 0x80;
    }
    p[9] = 0; // i3=0, n3=0
    Some(p)
}

fn pack77_nonstd(
    call_to: &str,
    call_de: &str,
    extra: &NonstdExtra,
    ht: &mut CallsignHashTable,
) -> Option<Payload77> {
    let i3: u8 = 4;

    let icq: u8 = if call_to == "CQ" || call_to.starts_with("CQ ") { 1 } else { 0 };

    let (iflip, n12, call58_str) = if icq == 0 {
        // Non-CQ: call_de is the 58-bit full callsign, call_to is the 12-bit hash
        let iflip: u8 = 0;
        let n12 = {
            let (_, n12_val, _) = ht.save(call_to);
            n12_val
        };
        (iflip, n12, call_de.to_string())
    } else {
        // CQ: call_de is the 58-bit callsign, n12=0
        (0u8, 0u16, call_de.to_string())
    };

    let n58 = pack58(&call58_str, ht)?;

    let nrpt: u8 = if icq != 0 {
        0
    } else {
        match extra {
            NonstdExtra::RRR      => 1,
            NonstdExtra::RR73     => 2,
            NonstdExtra::Seventy3 => 3,
            NonstdExtra::None     => 0,
        }
    };

    let mut p = [0u8; 10];
    p[0] = (n12 >> 4) as u8;
    p[1] = ((n12 << 4) as u8) | ((n58 >> 54) as u8);
    p[2] = (n58 >> 46) as u8;
    p[3] = (n58 >> 38) as u8;
    p[4] = (n58 >> 30) as u8;
    p[5] = (n58 >> 22) as u8;
    p[6] = (n58 >> 14) as u8;
    p[7] = (n58 >> 6) as u8;
    p[8] = ((n58 << 2) as u8) | (iflip << 1) | (nrpt >> 1);
    p[9] = (nrpt << 7) | (icq << 6) | (i3 << 3);

    Some(p)
}

fn pack77_telemetry(data: &[u8; 9]) -> Payload77 {
    let mut p = [0u8; 10];
    // Left-shift data by 1 bit (same as free text)
    let mut carry: u8 = 0;
    for i in (0..9usize).rev() {
        p[i] = (data[i] << 1) | (carry >> 7);
        carry = data[i] & 0x80;
    }
    // Encode n3=5, i3=0 in payload[8] bit 0 and payload[9] bits 7-6
    // n3=5=0b101: bit2 -> payload[8] bit 0; bits 1-0 -> payload[9] bits 7-6
    p[8] |= 0x01;     // n3 bit 2
    p[9] = 0b01 << 6; // n3 bits 1-0 = 0b01; i3 = 0
    p
}

// ---------------------------------------------------------------------------
// Unpack helpers
// ---------------------------------------------------------------------------

fn payload_to_b71(payload: &Payload77) -> [u8; 9] {
    // Right-shift payload[0..8] by 1 bit (ft8_lib ftx_message_decode_telemetry)
    let mut b71 = [0u8; 9];
    let mut carry: u8 = 0;
    for i in 0..9usize {
        b71[i] = (carry << 7) | (payload[i] >> 1);
        carry = payload[i] & 0x01;
    }
    b71
}

fn unpack77_standard(payload: &Payload77, i3: u8, ht: &CallsignHashTable) -> Ft8Message {
    let n29a: u32 = ((payload[0] as u32) << 21)
        | ((payload[1] as u32) << 13)
        | ((payload[2] as u32) << 5)
        | ((payload[3] as u32) >> 3);
    let n29b: u32 = (((payload[3] & 0x07) as u32) << 26)
        | ((payload[4] as u32) << 18)
        | ((payload[5] as u32) << 10)
        | ((payload[6] as u32) << 2)
        | ((payload[7] as u32) >> 6);
    let ir = (payload[7] & 0x20) != 0;
    let igrid4: u16 = (((payload[7] & 0x1F) as u16) << 10)
        | ((payload[8] as u16) << 2)
        | ((payload[9] as u16) >> 6);

    let call_to = unpack28(n29a >> 1, (n29a & 1) != 0, i3, ht)
        .unwrap_or_else(|| "<?>".to_string());
    let call_de = unpack28(n29b >> 1, (n29b & 1) != 0, i3, ht)
        .unwrap_or_else(|| "<?>".to_string());
    let extra = unpackgrid(igrid4, ir);

    Ft8Message::Standard { call_to, call_de, extra }
}

fn unpack77_nonstd(payload: &Payload77, ht: &CallsignHashTable) -> Ft8Message {
    let n12 = ((payload[0] as u16) << 4) | ((payload[1] as u16) >> 4);
    let n58: u64 = (((payload[1] & 0x0F) as u64) << 54)
        | ((payload[2] as u64) << 46)
        | ((payload[3] as u64) << 38)
        | ((payload[4] as u64) << 30)
        | ((payload[5] as u64) << 22)
        | ((payload[6] as u64) << 14)
        | ((payload[7] as u64) << 6)
        | ((payload[8] as u64) >> 2);

    let iflip = (payload[8] >> 1) & 0x01;
    let nrpt = ((payload[8] & 0x01) << 1) | (payload[9] >> 7);
    let icq = (payload[9] >> 6) & 0x01;

    // Use a temporary mutable ht for unpack58 (it saves the callsign)
    // We need a mutable reference, but we only have an immutable one here.
    // We'll decode the callsign without saving, which is fine for display.
    let call_decoded = unpack58_readonly(n58);
    let call_hashed = ht.lookup_n12(n12)
        .map(|s| format!("<{}>", s))
        .unwrap_or_else(|| "<...>".to_string());

    // Flip: iflip=0 means call_de is the 58-bit one
    let (call_to_str, call_de_str) = if iflip == 0 {
        (call_hashed, call_decoded)
    } else {
        (call_decoded, call_hashed)
    };

    let (call_to_final, call_de_final) = if icq == 1 {
        ("CQ".to_string(), call_de_str)
    } else {
        (call_to_str, call_de_str)
    };

    let extra = match nrpt {
        1 => NonstdExtra::RRR,
        2 => NonstdExtra::RR73,
        3 => NonstdExtra::Seventy3,
        _ => NonstdExtra::None,
    };

    let extra = if icq == 1 { NonstdExtra::None } else { extra };

    Ft8Message::NonStd {
        call_to: call_to_final,
        call_de: call_de_final,
        extra,
    }
}

/// Decode base-38 without saving to hash table (read-only context).
fn unpack58_readonly(n58: u64) -> String {
    use super::tables::{charn, Table};
    let mut n = n58;
    let mut chars = [' '; 11];
    for i in (0..11).rev() {
        chars[i] = charn((n % 38) as u8, Table::AlphanumSpaceSlash);
        n /= 38;
    }
    let s: String = chars.iter().collect();
    s.trim_matches(' ').to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pack77_type1_roundtrip() {
        let mut ht = CallsignHashTable::new();
        let msg = Ft8Message::Standard {
            call_to: "KD9ABC".to_string(),
            call_de: "W9XYZ".to_string(),
            extra: GridField::Grid("FN31".to_string()),
        };
        let payload = pack77(&msg, &mut ht).expect("pack77 failed");
        let decoded = unpack77(&payload, &ht);
        assert_eq!(decoded, msg, "Type 1 roundtrip failed");
    }

    #[test]
    fn pack77_free_text_roundtrip() {
        let mut ht = CallsignHashTable::new();
        let msg = Ft8Message::FreeText("CQ DX".to_string());
        let payload = pack77(&msg, &mut ht).expect("pack77 free text");
        let decoded = unpack77(&payload, &ht);
        assert_eq!(decoded, msg);
    }

    #[test]
    fn pack77_telemetry_roundtrip() {
        let mut ht = CallsignHashTable::new();
        let data = [0x12u8, 0x34, 0x56, 0x78, 0x9A, 0xBC, 0xDE, 0xF0, 0x11];
        let msg = Ft8Message::Telemetry(data);
        let payload = pack77(&msg, &mut ht).expect("pack77 telemetry");
        let decoded = unpack77(&payload, &ht);
        assert_eq!(decoded, msg);
    }
}
