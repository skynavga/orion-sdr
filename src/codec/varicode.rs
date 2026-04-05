// src/codec/varicode.rs
//
// IZ8BLY / G3PLX Varicode encoding / decoding for PSK31.
//
// Each ASCII character (0–127) maps to a codeword of 1–11 bits (MSB-first).
// Characters are separated by two 0-bits ("00") when transmitted.
// This matches the convention used by fldigi and the original PSK31 spec.
//
// Table source: Peter Martinez G3PLX, "PSK31: A New Radio-Teletype Mode" (1998).
// The table is also given verbatim in the fldigi source (varicode.cxx).

use std::collections::VecDeque;

/// Maximum codeword length in the table (10 bits).  The decoder's shift
/// register must accommodate one extra bit for the leading zero of the "00"
/// separator, so the effective capacity is VARICODE_MAX_BITS + 1 = 11.
pub const VARICODE_MAX_BITS: usize = 10;

/// Varicode table: index = ASCII value (0–127).
/// Each entry is `(codeword: u16, len: u8)` where the codeword is stored
/// MSB-first so bit `(len-1)` is the first bit transmitted.
///
/// Canonical IZ8BLY / fldigi table (pskvaricode.cxx).
const VARICODE: [(u16, u8); 128] = [
    (0b1010101011, 10), //   0  NUL
    (0b1011011011, 10), //   1  SOH
    (0b1011101101, 10), //   2  STX
    (0b1101110111, 10), //   3  ETX
    (0b1011101011, 10), //   4  EOT
    (0b1101011111, 10), //   5  ENQ
    (0b1011101111, 10), //   6  ACK
    (0b1011111101, 10), //   7  BEL
    (0b1011111111, 10), //   8  BS
    (0b11101111,    8), //   9  HT
    (0b11101,       5), //  10  LF
    (0b1101101111, 10), //  11  VT
    (0b1011011101, 10), //  12  FF
    (0b11111,       5), //  13  CR
    (0b1101110101, 10), //  14  SO
    (0b1110101011, 10), //  15  SI
    (0b1011110111, 10), //  16  DLE
    (0b1011110101, 10), //  17  DC1
    (0b1110101101, 10), //  18  DC2
    (0b1110101111, 10), //  19  DC3
    (0b1101011011, 10), //  20  DC4
    (0b1101101011, 10), //  21  NAK
    (0b1101101101, 10), //  22  SYN
    (0b1101010111, 10), //  23  ETB
    (0b1101111011, 10), //  24  CAN
    (0b1101111101, 10), //  25  EM
    (0b1110110111, 10), //  26  SUB
    (0b1101010101, 10), //  27  ESC
    (0b1101011101, 10), //  28  FS
    (0b1110111011, 10), //  29  GS
    (0b1011111011, 10), //  30  RS
    (0b1101111111, 10), //  31  US
    (0b1,           1), //  32  SP
    (0b111111111,   9), //  33  !
    (0b101011111,   9), //  34  "
    (0b111110101,   9), //  35  #
    (0b111011011,   9), //  36  $
    (0b1011010101, 10), //  37  %
    (0b1010111011, 10), //  38  &
    (0b101111111,   9), //  39  '
    (0b11111011,    8), //  40  (
    (0b11110111,    8), //  41  )
    (0b101101111,   9), //  42  *
    (0b111011111,   9), //  43  +
    (0b1110101,     7), //  44  ,
    (0b110101,      6), //  45  -
    (0b1010111,     7), //  46  .
    (0b110101111,   9), //  47  /
    (0b10110111,    8), //  48  0
    (0b10111101,    8), //  49  1
    (0b11101101,    8), //  50  2
    (0b11111111,    8), //  51  3
    (0b101110111,   9), //  52  4
    (0b101011011,   9), //  53  5
    (0b101101011,   9), //  54  6
    (0b110101101,   9), //  55  7
    (0b110101011,   9), //  56  8
    (0b110110111,   9), //  57  9
    (0b11110101,    8), //  58  :
    (0b110111101,   9), //  59  ;
    (0b111101101,   9), //  60  <
    (0b1010101,     7), //  61  =
    (0b111010111,   9), //  62  >
    (0b1010101111, 10), //  63  ?
    (0b1010111101, 10), //  64  @
    (0b1111101,     7), //  65  A
    (0b11101011,    8), //  66  B
    (0b10101101,    8), //  67  C
    (0b10110101,    8), //  68  D
    (0b1110111,     7), //  69  E
    (0b11011011,    8), //  70  F
    (0b11111101,    8), //  71  G
    (0b101010101,   9), //  72  H
    (0b1111111,     7), //  73  I
    (0b111111101,   9), //  74  J
    (0b101111101,   9), //  75  K
    (0b11010111,    8), //  76  L
    (0b10111011,    8), //  77  M
    (0b11011101,    8), //  78  N
    (0b10101011,    8), //  79  O
    (0b11010101,    8), //  80  P
    (0b111011101,   9), //  81  Q
    (0b10101111,    8), //  82  R
    (0b1101111,     7), //  83  S
    (0b1101101,     7), //  84  T
    (0b101010111,   9), //  85  U
    (0b110110101,   9), //  86  V
    (0b101011101,   9), //  87  W
    (0b101110101,   9), //  88  X
    (0b101111011,   9), //  89  Y
    (0b1010101101, 10), //  90  Z
    (0b111110111,   9), //  91  [
    (0b111101111,   9), //  92  backslash
    (0b111111011,   9), //  93  ]
    (0b1010111111, 10), //  94  ^
    (0b101101101,   9), //  95  _
    (0b1011011111, 10), //  96  `
    (0b1011,        4), //  97  a
    (0b1011111,     7), //  98  b
    (0b101111,      6), //  99  c
    (0b101101,      6), // 100  d
    (0b11,          2), // 101  e
    (0b111101,      6), // 102  f
    (0b1011011,     7), // 103  g
    (0b101011,      6), // 104  h
    (0b1101,        4), // 105  i
    (0b111101011,   9), // 106  j
    (0b10111111,    8), // 107  k
    (0b11011,       5), // 108  l
    (0b111011,      6), // 109  m
    (0b1111,        4), // 110  n
    (0b111,         3), // 111  o
    (0b111111,      6), // 112  p
    (0b110111111,   9), // 113  q
    (0b10101,       5), // 114  r
    (0b10111,       5), // 115  s
    (0b101,         3), // 116  t
    (0b110111,      6), // 117  u
    (0b1111011,     7), // 118  v
    (0b1101011,     7), // 119  w
    (0b11011111,    8), // 120  x
    (0b1011101,     7), // 121  y
    (0b111010101,   9), // 122  z
    (0b1010110111, 10), // 123  {
    (0b110111011,   9), // 124  |
    (0b1010110101, 10), // 125  }
    (0b1011010111, 10), // 126  ~
    (0b1110110101, 10), // 127  DEL
];

/// Encode one ASCII byte → `(codeword: u16, len: u8)`.
///
/// `codeword` is MSB-first: bit `len-1` is the first bit transmitted.
/// Values ≥ 128 are mapped to the NUL entry (index 0).
#[inline]
pub fn varicode_encode(byte: u8) -> (u16, u8) {
    if byte >= 128 { VARICODE[0] } else { VARICODE[byte as usize] }
}

/// Decode a Varicode codeword back to an ASCII byte.
///
/// Returns `None` if the codeword is not in the table.
/// Linear scan over 128 entries (fast enough for 31.25 baud).
#[inline]
pub fn varicode_decode(bits: u16, len: u8) -> Option<u8> {
    for (i, &(cw, cw_len)) in VARICODE.iter().enumerate() {
        if cw_len == len && cw == bits {
            return Some(i as u8);
        }
    }
    None
}

// ── VaricodeEncoder ───────────────────────────────────────────────────────────

/// Stateful Varicode bit-stream encoder.
///
/// Encodes ASCII bytes to a bit stream, inserting a 2-bit "00" gap between
/// characters.  The leading gap before the first character is suppressed.
#[derive(Debug, Clone)]
pub struct VaricodeEncoder {
    pending: VecDeque<u8>,
    first: bool,
}

impl Default for VaricodeEncoder {
    fn default() -> Self { Self::new() }
}

impl VaricodeEncoder {
    pub fn new() -> Self {
        Self { pending: VecDeque::new(), first: true }
    }

    /// Push `n` 0-bits (preamble — continuous phase reversals for AFC lock).
    ///
    /// The preamble ends with zeros so the first character's leading "00" gap is
    /// naturally provided.  We leave `first = true` so that `push_byte` after a
    /// preamble does NOT insert an additional "00" before the codeword.
    pub fn push_preamble(&mut self, n_bits: usize) {
        for _ in 0..n_bits {
            self.pending.push_back(0);
        }
        // Do NOT change self.first — the next push_byte will skip the "00" prefix,
        // which is the right behaviour since the preamble zeros serve as the gap.
        self.first = true;
    }

    /// Push one ASCII byte.
    /// Inserts a "00" inter-character gap before the codeword (except the first).
    pub fn push_byte(&mut self, b: u8) {
        if !self.first {
            self.pending.push_back(0);
            self.pending.push_back(0);
        }
        self.first = false;
        let (cw, len) = varicode_encode(b);
        for i in (0..len).rev() {
            self.pending.push_back(((cw >> i) & 1) as u8);
        }
    }

    /// Push `n` 1-bits (postamble — carrier hold / idle).
    ///
    /// First inserts a "00" inter-character gap so the Varicode decoder can
    /// flush the last encoded character before the 1-bit idle sequence begins.
    pub fn push_postamble(&mut self, n_bits: usize) {
        // "00" gap to flush the last character through the decoder.
        if !self.first {
            self.pending.push_back(0);
            self.pending.push_back(0);
        }
        for _ in 0..n_bits {
            self.pending.push_back(1);
        }
    }

    /// Pop the next bit.  Returns `None` when the queue is empty.
    pub fn next_bit(&mut self) -> Option<u8> {
        self.pending.pop_front()
    }

    pub fn is_empty(&self) -> bool {
        self.pending.is_empty()
    }

    /// Drain all pending bits into a `Vec<u8>`.
    pub fn drain_bits(&mut self) -> Vec<u8> {
        self.pending.drain(..).collect()
    }
}

// ── VaricodeDecoder ───────────────────────────────────────────────────────────

/// Stateful Varicode bit-stream decoder.
///
/// Feed bits one at a time with `push_bit`.  Characters are emitted whenever
/// two consecutive 0-bits are detected (the inter-character separator).
/// Retrieve decoded characters with `pop_char`.
#[derive(Debug, Clone, Default)]
pub struct VaricodeDecoder {
    shift: u16,
    len: u8,
    prev_zero: bool,
    chars: VecDeque<u8>,
}

impl VaricodeDecoder {
    pub fn new() -> Self { Self::default() }

    /// Push one received bit (0 or 1).
    pub fn push_bit(&mut self, bit: u8) {
        let is_zero = bit == 0;

        if is_zero && self.prev_zero {
            // "00" boundary detected.
            // The previous zero was already shifted into `self.shift` in the last call.
            // Remove it before decoding: the codeword is `shift >> 1` with `len - 1`.
            let cw     = if self.len > 0 { self.shift >> 1 } else { 0 };
            let cw_len = self.len.saturating_sub(1);
            if cw_len > 0 {
                if let Some(ch) = varicode_decode(cw, cw_len) {
                    self.chars.push_back(ch);
                }
            }
            self.shift = 0;
            self.len = 0;
            self.prev_zero = false;
        } else {
            self.shift = (self.shift << 1) | (bit & 1) as u16;
            if self.len < (VARICODE_MAX_BITS as u8 + 1) {
                self.len += 1;
            }
            self.prev_zero = is_zero;
        }
    }

    /// Pop the next decoded character.  Returns `None` when the queue is empty.
    pub fn pop_char(&mut self) -> Option<u8> {
        self.chars.pop_front()
    }
}
