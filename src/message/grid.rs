const MAXGRID4: u16 = 32_400;

/// The "extra" field in a standard FT8 message.
#[derive(Debug, Clone, PartialEq)]
pub enum GridField {
    Grid(String),   // 4-char Maidenhead, e.g. "FN31"
    Report(i8),     // signal report in dB, e.g. +7 or -12
    RReport(i8),    // same with "R" prefix
    RRR,
    RR73,
    Seventy3,       // "73"
    None,
}

/// Pack the extra/grid field into a u16.
/// The high bit (bit 15) is the `ir` flag (set for R-prefixed reports).
/// The low 15 bits are `igrid4` sent in the payload.
pub fn packgrid(extra: &str) -> u16 {
    if extra.is_empty() {
        return MAXGRID4 + 1; // No extra field
    }
    match extra {
        "RRR"  => return MAXGRID4 + 2,
        "RR73" => return MAXGRID4 + 3,
        "73"   => return MAXGRID4 + 4,
        _ => {}
    }

    let bytes = extra.as_bytes();

    // 4-char Maidenhead grid
    if bytes.len() == 4
        && bytes[0] >= b'A' && bytes[0] <= b'R'
        && bytes[1] >= b'A' && bytes[1] <= b'R'
        && bytes[2].is_ascii_digit()
        && bytes[3].is_ascii_digit()
    {
        let igrid4 = (bytes[0] - b'A') as u16 * 18 * 10 * 10
            + (bytes[1] - b'A') as u16 * 10 * 10
            + (bytes[2] - b'0') as u16 * 10
            + (bytes[3] - b'0') as u16;
        return igrid4;
    }

    // Signal report: optional 'R' prefix, then +/- and digits
    if bytes[0] == b'R' && bytes.len() >= 2 {
        let dd = dd_to_int(&extra[1..]);
        let irpt = (35i32 + dd as i32) as u16;
        return (MAXGRID4 + irpt) | 0x8000; // ir = 1
    } else {
        let dd = dd_to_int(extra);
        let irpt = (35i32 + dd as i32) as u16;
        return MAXGRID4 + irpt; // ir = 0
    }
}

fn dd_to_int(s: &str) -> i8 {
    let bytes = s.as_bytes();
    if bytes.is_empty() { return 0; }
    let (neg, start) = match bytes[0] {
        b'-' => (true, 1),
        b'+' => (false, 1),
        _    => (false, 0),
    };
    let mut val: i8 = 0;
    for &b in &bytes[start..] {
        if b.is_ascii_digit() {
            val = val.wrapping_mul(10).wrapping_add((b - b'0') as i8);
        } else {
            break;
        }
    }
    if neg { -val } else { val }
}

/// Unpack the grid/extra field.
/// `igrid4` is the 15-bit value from the payload; `ir` is the high bit (bit 15).
pub fn unpackgrid(igrid4: u16, ir: bool) -> GridField {
    if igrid4 <= MAXGRID4 {
        // 4-char Maidenhead grid
        let mut n = igrid4;
        let d3 = (n % 10) as u8; n /= 10;
        let d2 = (n % 10) as u8; n /= 10;
        let c1 = (n % 18) as u8; n /= 18;
        let c0 = (n % 18) as u8;
        let grid = format!(
            "{}{}{}{}",
            (b'A' + c0) as char,
            (b'A' + c1) as char,
            (b'0' + d2) as char,
            (b'0' + d3) as char,
        );
        if ir {
            // Grid with R prefix in unpack corresponds to R-prefixed report, but
            // per ft8_lib, ir=1 on a grid means "R " prepended
            return GridField::Grid(format!("R {}", grid));
        }
        return GridField::Grid(grid);
    }

    let irpt = (igrid4 - MAXGRID4) as i32;
    match irpt {
        1 => GridField::None,
        2 => GridField::RRR,
        3 => GridField::RR73,
        4 => GridField::Seventy3,
        _ => {
            let dd = (irpt - 35) as i8;
            if ir {
                GridField::RReport(dd)
            } else {
                GridField::Report(dd)
            }
        }
    }
}

/// Convert a `GridField` back to its string representation for display.
pub fn gridfield_to_str(gf: &GridField) -> String {
    match gf {
        GridField::Grid(s)    => s.clone(),
        GridField::Report(n)  => format!("{:+03}", n),
        GridField::RReport(n) => format!("R{:+03}", n),
        GridField::RRR        => "RRR".to_string(),
        GridField::RR73       => "RR73".to_string(),
        GridField::Seventy3   => "73".to_string(),
        GridField::None       => String::new(),
    }
}

/// Convert a `GridField` to (igrid4_15bits, ir) for packing.
pub fn gridfield_to_pack(gf: &GridField) -> (u16, bool) {
    let raw = match gf {
        GridField::Grid(s)    => packgrid(s),
        GridField::Report(n)  => packgrid(&format!("{:+03}", n)),
        GridField::RReport(n) => packgrid(&format!("R{:+03}", n)),
        GridField::RRR        => packgrid("RRR"),
        GridField::RR73       => packgrid("RR73"),
        GridField::Seventy3   => packgrid("73"),
        GridField::None       => packgrid(""),
    };
    let ir = (raw & 0x8000) != 0;
    let igrid4 = raw & 0x7FFF;
    (igrid4, ir)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packgrid_grid_roundtrip() {
        for g in &["FN31", "AA00", "RR99"] {
            let raw = packgrid(g);
            let gf = unpackgrid(raw & 0x7FFF, (raw & 0x8000) != 0);
            assert_eq!(gf, GridField::Grid(g.to_string()), "Roundtrip failed for {}", g);
        }
    }

    #[test]
    fn packgrid_report_roundtrip() {
        let cases: &[(&str, GridField)] = &[
            ("+07",  GridField::Report(7)),
            ("-12",  GridField::Report(-12)),
            ("R-05", GridField::RReport(-5)),
        ];
        for (s, expected) in cases {
            let raw = packgrid(s);
            let gf = unpackgrid(raw & 0x7FFF, (raw & 0x8000) != 0);
            assert_eq!(&gf, expected, "Roundtrip failed for {}", s);
        }
    }

    #[test]
    fn packgrid_tokens() {
        let raw = packgrid("RRR");
        assert_eq!(unpackgrid(raw & 0x7FFF, false), GridField::RRR);
        let raw = packgrid("RR73");
        assert_eq!(unpackgrid(raw & 0x7FFF, false), GridField::RR73);
        let raw = packgrid("73");
        assert_eq!(unpackgrid(raw & 0x7FFF, false), GridField::Seventy3);
        let raw = packgrid("");
        assert_eq!(unpackgrid(raw & 0x7FFF, false), GridField::None);
    }
}
