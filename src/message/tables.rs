/// Character table selector — matches ft8_lib's `ft8_char_table_e`.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Table {
    Full,               // 42 chars: " 0-9A-Z+-./?"
    AlphanumSpaceSlash, // 38 chars: " 0-9A-Z/"
    AlphanumSpace,      // 37 chars: " 0-9A-Z"
    LettersSpace,       // 27 chars: " A-Z"
    Alphanum,           // 36 chars: "0-9A-Z"
    Numeric,            // 10 chars: "0-9"
}

/// Return the index of `c` in `table`, or `None` if `c` is not in the table.
pub fn nchar(c: char, table: Table) -> Option<u8> {
    let mut n: i32 = 0;
    // Space is index 0 in all tables that include it
    if table != Table::Alphanum && table != Table::Numeric {
        if c == ' ' {
            return Some(n as u8);
        }
        n += 1;
    }
    // Digits 0-9
    if table != Table::LettersSpace {
        if c >= '0' && c <= '9' {
            return Some((n + (c as i32 - '0' as i32)) as u8);
        }
        n += 10;
    }
    // Letters A-Z
    if table != Table::Numeric {
        if c >= 'A' && c <= 'Z' {
            return Some((n + (c as i32 - 'A' as i32)) as u8);
        }
        n += 26;
    }
    // Extra characters
    match table {
        Table::Full => {
            match c {
                '+' => Some((n + 0) as u8),
                '-' => Some((n + 1) as u8),
                '.' => Some((n + 2) as u8),
                '/' => Some((n + 3) as u8),
                '?' => Some((n + 4) as u8),
                _ => None,
            }
        }
        Table::AlphanumSpaceSlash => {
            if c == '/' { Some((n + 0) as u8) } else { None }
        }
        _ => None,
    }
}

/// Return the character at index `n` in `table`.
pub fn charn(n: u8, table: Table) -> char {
    let mut n = n as i32;
    if table != Table::Alphanum && table != Table::Numeric {
        if n == 0 {
            return ' ';
        }
        n -= 1;
    }
    if table != Table::LettersSpace {
        if n < 10 {
            return char::from_u32('0' as u32 + n as u32).unwrap();
        }
        n -= 10;
    }
    if table != Table::Numeric {
        if n < 26 {
            return char::from_u32('A' as u32 + n as u32).unwrap();
        }
        n -= 26;
    }
    match table {
        Table::Full => {
            "+-./?"
                .chars()
                .nth(n as usize)
                .unwrap_or('_')
        }
        Table::AlphanumSpaceSlash => {
            if n == 0 { '/' } else { '_' }
        }
        _ => '_',
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nchar_charn_roundtrip_full() {
        let chars: &str = " 0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ+-./? ";
        for c in " 0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ+-./?".chars() {
            let idx = nchar(c, Table::Full).expect(&format!("nchar failed for '{}'", c));
            assert_eq!(charn(idx, Table::Full), c);
        }
        // Make sure ' ' is not repeated by FULL test
        let _ = chars;
    }

    #[test]
    fn nchar_alphanum_space() {
        assert_eq!(nchar(' ', Table::AlphanumSpace), Some(0));
        assert_eq!(nchar('0', Table::AlphanumSpace), Some(1));
        assert_eq!(nchar('Z', Table::AlphanumSpace), Some(36));
        assert_eq!(nchar('/', Table::AlphanumSpace), None);
    }
}
