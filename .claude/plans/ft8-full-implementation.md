# FT8/FT4 Full Implementation Plan

## Protocol Background

FT8 is an 8-FSK weak-signal digital mode designed for HF amateur radio.
A complete encode/decode pipeline has four distinct layers:

1. **Waveform** — CPFSK mod/demod with Costas sync arrays
2. **Channel coding** — CRC-14 + LDPC(174,91) + Gray code bit-to-tone mapping
3. **Frame synchronization** — Costas array detection, timing and frequency offset estimation
4. **Message packing** — 77-bit structured payloads (callsigns, grid squares, reports)

Each phase below adds one layer, building on the previous.

---

## Phase 1 — Rectangular CPFSK waveform with frame mod/demod [DONE]

**Branch:** `feature/ft8-mod-demod`

### What was built

- `Ft8Frame([u8; 58])` / `Ft4Frame([u8; 87])` — raw tone-index containers
- `Ft8Mod` / `Ft4Mod` — CPFSK modulator
  - Phase-continuous phasor recurrence (no per-sample sin/cos)
  - 4-sample inner-loop unroll; periodic phasor renormalisation
  - Costas sync arrays inserted at fixed symbol positions
  - Optional RF upconversion via `Rotator`
- `Ft8Demod` / `Ft4Demod` — dot-product correlator demodulator
  - Goertzel-style energy detection per tone per symbol
  - 4-sample inner-loop unroll
  - Costas symbols stripped; returns 58/87 data tone indices
- Unit tests: frame length, symbol count, Costas positions, IQ power
- Roundtrip tests: noiseless and all-tones

### Post-Phase-1 correction (applied before Phase 3)

The initial FT4 frame structure did not match the reference (ft8_lib):

| | Phase 1 (wrong) | Corrected |
|---|---|---|
| Total symbols | 103 | **105** |
| Frame length | 59 328 samples | **60 480 samples** |
| Costas positions | (0,4),(29,33),(60,64),(99,103) | **(1,5),(34,38),(67,71),(100,104)** |
| Costas row 2 | `[2,3,0,1]` | **`[2,3,1,0]`** |
| Costas row 3 | `[3,2,1,0]` | **`[3,2,0,1]`** |
| Ramp symbols | none | **tone 0 at positions 0 and 104** |

Frame layout: `R S4_1 D29 S4_2 D29 S4_3 D29 S4_4 R`
(2 ramps + 4×4 Costas + 87 data = 105 symbols)

### What is explicitly NOT included

- No channel coding (CRC, LDPC, Gray mapping)
- No frame/timing synchronisation
- No message structure
- No Python bindings for FT8/FT4

---

## Phase 2 — Channel coding: CRC-14, LDPC(174,91), Gray code [DONE]

This phase adds the full bit-level codec between raw 77-bit messages and
the 58 tone indices consumed/produced by Phase 1.

### FT8 channel coding stack (encode direction)

```
77-bit message
  → append 14-bit CRC (poly 0x2757, no reflection, init 0)
  → 91-bit augmented message
  → LDPC(174,91) encoding → 174-bit codeword
  → split into 58 × 3-bit groups
  → Gray code each group: [0,1,3,2,5,6,4,7]
  → 58 tone indices  →  Ft8Frame  →  Ft8Mod
```

Decode direction is the reverse, with soft decisions feeding the LDPC decoder.

### Tasks

- [x] `src/codec/crc.rs` — `ft8_crc14`, `ft8_add_crc`, `ft8_extract_crc`
  - Generator: 0x2757 (14-bit); CRC computed over 77-bit payload zero-padded to 82 bits
- [x] `src/codec/ldpc.rs` — LDPC(174,91) encoder and sum-product soft-decision decoder
  - Encoder: systematic; parity-check matrix Nm embedded as `const` (from ft8_lib MIT)
  - Decoder: belief propagation, 20 iterations, fast tanh/atanh approximation
  - `ldpc_count_errors` syndrome check for hard codewords
- [x] `src/codec/gray.rs` — `gray8_encode/decode` (FT8) and `gray4_encode/decode` (FT4)
- [x] `src/codec/ft8.rs` — `Ft8Codec { encode, decode_hard, decode_soft, frame_to_llr_hard }`
- [x] `src/codec/ft4.rs` — `Ft4Codec { encode, decode_hard, decode_soft, frame_to_llr_hard }`
  - FT4-specific XOR scramble applied before CRC+LDPC
- [x] Unit tests in `src/tests/unit.rs`:
  - Gray encode/decode roundtrip (FT8 and FT4)
  - CRC known-answer test
  - LDPC encode→syndrome-check passes
  - Codec encode produces valid tone ranges
- [x] Roundtrip tests in `src/tests/roundtrip.rs`:
  - `roundtrip_ft8_codec_noiseless` and `roundtrip_ft8_codec_zeros`
  - `roundtrip_ft4_codec_noiseless` and `roundtrip_ft4_codec_zeros`
  - Full stack: Ft8Codec::encode → Ft8Mod → Ft8Demod → Ft8Codec::decode_hard

### Implementation notes and gotchas

**CRC polynomial is 0x2757, not 0x6757.**
Multiple secondary sources (QEX paper, online summaries) cite 0x6757.
The ft8_lib reference implementation (`constants.h`) uses `0x2757u`.
Trust the code, not the docs.

**CRC domain is 82 bits, not 91.**
The CRC is computed over the 77-bit payload zero-extended to 82 bits (5
appended zero bits), corresponding to `ft8_crc14(buf, 82)`.  The 14-bit CRC
result is then stored in bits 77-90 of the 91-bit `a91` block.  Critically,
the CRC bits themselves are NOT fed into the CRC computation.  When verifying
during decode, zero out `buf[9] &= 0xF8; buf[10] = 0; buf[11] = 0;` before
calling `ft8_crc14(buf, 82)`.  Forgetting this causes a wrong answer because
5 of the 14 CRC bits land in the first 82-bit window.

**Byte 9 slack bits — mask is 0xF8, not 0xFE.**
The 77-bit payload is packed MSB-first into 10 bytes (bits 0-76).  Bit 76 is
byte 9 bit 3 (counting from MSB=bit 7); bits 77-79 (byte 9 bits 2-0) are
slack.  For a "77 bits all-ones" test payload, byte 9 = 0xF8.  Using 0xFE
(only one slack bit) is wrong.  The mask `& 0xF8` must be applied consistently
in three places: `ft8_add_crc`, `decode_llr` output, and FT4 un-XOR output.

**LDPC codeword byte 11 is a split byte.**
K=91 bits → K_BYTES=12, but 91 % 8 = 3, so byte 11 is shared:
  - bits 7-5 (top 3): message bits from a91[11]
  - bits 4-0 (bottom 5): first 5 parity bits
A test asserting `codeword[..12] == a91[..12]` will fail on byte 11.
The correct test checks `codeword[..11]` exactly and `codeword[11] & 0xE0 == a91[11] & 0xE0`.

**NM array has exactly 83 rows.**
The ft8_lib `kFTX_LDPC_Nm` array has 83 rows, but when transcribing from the
source it is easy to stop 2 rows early (rows 81-82, 0-indexed), as they appear
after a visual gap near the end of the array.  If the Rust const is declared
as `[[u8;7]; M]` with M=83, the compiler will catch the count mismatch.

- Represent bit arrays as `[u8; N]` with one bit per byte for clarity;
  pack to `u64`/`u128` only if benchmarks show it matters.
- `Ft8Frame` currently holds tone indices 0–7; after Phase 2 these are
  always Gray-coded LDPC output — the type is unchanged but semantics shift.

---

## Phase 3 — Frame synchronisation: timing and frequency offset [DONE]

This phase enables decoding a raw IQ stream captured from the air, rather
than a pre-aligned synthetic block.

### What was built

- `src/sync/waterfall.rs` — symbol-rate magnitude spectrogram
  - `compute_waterfall(iq, fs, base_hz, tone_spacing_hz, samples_per_sym, num_syms, num_tones, time_offset) -> Waterfall`
  - Uses same Goertzel/dot-product correlator as Phase 1 demodulator (4-sample inner-loop unroll)
  - Stores `ln(energy + 1e-12)` per (symbol, tone) cell
- `src/sync/costas.rs` — Costas difference-metric scorer and candidate search
  - `costas_score(wf, costas, sync_pos, time_sym, freq_bin) -> f32`
  - Difference metric: signal energy minus max(frequency neighbour, time neighbour) per Costas symbol
  - `find_candidates(wf, costas, sync_pos, ...) -> Vec<Candidate>` — fixed-size min-heap, top-N by score
- `src/sync/ft8_sync.rs` — FT8 sync pipeline
  - `ft8_sync(iq, fs, base_hz, max_hz, t_min, t_max, max_cand) -> Vec<Ft8SyncResult>`
  - Computes waterfall once, searches for top-N Costas-matching candidates
  - Extracts 174 soft LLRs per candidate via max-log over Gray-indexed energies
  - Normalises by sqrt(24/variance) before returning
- `src/sync/ft4_sync.rs` — FT4 sync pipeline
  - `ft4_sync(iq, fs, ...) -> Vec<Ft4SyncResult>` — same structure, 4 different Costas blocks
  - FT4 data symbol offsets: k<29: sym=k+5; k<58: sym=k+9; k<87: sym=k+13
- `src/sync/mod.rs` — module root

### Integration tests in `src/tests/roundtrip.rs`

- `sync_ft8_noiseless_aligned` — frame at t=0, exact frequency
- `sync_ft8_noiseless_time_offset` — frame starts 3 symbols into search window
- `sync_ft8_noisy_high_snr` — frame + AWGN (noise_power=0.005, ~20 dB SNR)
- `sync_ft4_noiseless_aligned`
- `sync_ft4_noiseless_time_offset`

### Implementation notes and gotchas

**Waterfall starts at user-supplied `base_hz`.**
The candidate `freq_bin` is relative to the waterfall grid.  When the signal
tone-0 is at `base_hz`, its `freq_bin` = 0.  When searching from `base_hz - Δf`,
the signal lands at `freq_bin = Δf / tone_spacing`.

**FT8 Gray8 table is `[0,1,3,2,5,6,4,7]` (same as the codec).**
An earlier version had `[0,1,3,2,7,6,4,5]` — wrong values at indices 4-7.
The correct table is `FT8_GRAY = [0,1,3,2,5,6,4,7]` from ft8_lib `kFT8_Gray_map`.

**LLR sign convention: negate ft8_lib output before feeding our LDPC decoder.**
ft8_lib uses `LLR > 0 ↔ bit likely 1`; our decoder uses `LLR > 0 ↔ bit likely 0`.

**LDPC soft decoder divergence fix (in `src/codec/ldpc.rs`).**
With soft LLRs of moderate magnitude (~5, vs ±10 for hard decode), the belief-
propagation can diverge — successive iterations increase syndrome errors.
Two fixes applied:
1. Early exit: check syndrome before any BP; if initial hard decisions already
   satisfy all parity checks (0 errors), return immediately.
2. Save best: track the minimum-error `plain[]` snapshot across all iterations
   and return it, rather than the last iteration's `plain[]`.

**FT4 data symbol positions in the 105-symbol frame:**
Slots occupied by ramps (0, 104) and Costas (1-4, 34-37, 67-70, 100-103) are
skipped.  The data symbols at frame-relative positions are:
  k in [0,29): sym = k + 5   (positions 5..33)
  k in [29,58): sym = k + 9  (positions 38..66)
  k in [58,87): sym = k + 13 (positions 71..99)

### What is explicitly NOT included

- Sub-symbol (oversampled) timing / frequency refinement
- Frequency offset search beyond the explicit waterfall bin grid
  (the caller controls the search range via base_hz / max_hz)

---

## Phase 4 — Message packing: 77-bit structured payloads [DONE]

This phase adds the application layer: encoding human-readable QSO
messages into the 77-bit payload and decoding them back.

### Message types supported

| i3 | n3 | Description |
|----|----|-------------|
| 1/2 | — | Standard QSO: callsign + callsign + grid/report |
| 0  | 0  | Free text (up to 13 chars, base-42 alphabet) |
| 0  | 5  | Telemetry (71 arbitrary bits) |
| 4  | —  | Nonstandard callsign (58-bit plain + 12-bit hash) |

### What was built

- `src/message/tables.rs` — `Table` enum + `nchar`/`charn` matching ft8_lib's
  six character tables (FULL 42, ALPHANUM_SPACE_SLASH 38, ALPHANUM_SPACE 37,
  LETTERS_SPACE 27, ALPHANUM 36, NUMERIC 10)
- `src/message/callsign.rs` — `pack_basecall`, `pack28`/`unpack28`,
  `pack58`/`unpack58`, `CallsignHashTable` (in-memory `HashMap<u32, String>`
  keyed by 22-bit multiply-shift hash)
- `src/message/grid.rs` — `packgrid`/`unpackgrid`, `GridField` enum
  (Grid, Report, RReport, RRR, RR73, Seventy3, None)
- `src/message/free_text.rs` — `encode_free_text`/`decode_free_text`
  (big-endian base-42, 9-byte 71-bit integer)
- `src/message/message.rs` — `Ft8Message` enum, `pack77`/`unpack77`
- `src/message/mod.rs` — public re-exports

### Bit layout

**Type 1/2** (29 + 29 + 1 + 15 + 3 = 77 bits):
```
n29a[28:0] | n29b[28:0] | ir[0] | igrid4[14:0] | i3[2:0]
```
Packed MSB-first across `payload[0..9]`; `i3=1` (no suffix), `i3=2` (/P suffix).
The `ir` bit is bit 5 of `payload[7]`; `igrid4` spans bits 4:0 of `payload[7]`
through bits 7:6 of `payload[9]`.

**Type 0/free-text** (71 bits + 6 zero bits):
The 9-byte `b71` big-endian integer is left-shifted 1 bit into `payload[0..8]`;
`payload[9] = 0x00` (i3=0, n3=0).

**Type 0/telemetry** (n3=5): same left-shift; `payload[8] |= 0x01` (n3 bit 2),
`payload[9] = 0x40` (n3 bits 1:0 = 0b01, i3=0).

**Type 4** (12 + 58 + 1 + 2 + 1 + 3 = 77 bits):
```
n12[11:0] | n58[57:0] | iflip[0] | nrpt[1:0] | icq[0] | i3[2:0]
```

### Implementation notes and gotchas

**Telemetry type encoding spans two bytes.**
`n3` is extracted as `((payload[8] << 2) & 0x04) | ((payload[9] >> 6) & 0x03)`.
For n3=5=0b101: bit 2 lives in `payload[8]` bit 0 (which the left-shift leaves
as zero), and bits 1:0 live in `payload[9]` bits 7:6.  Setting `payload[9] = 5<<3`
is wrong — that sets i3=5, not n3=5.

**Free-text left-shift direction.**
ft8_lib iterates the shift loop `for i in (0..9).rev()` (i=8 down to 0), carrying
the MSB of each byte into the LSB of the byte above.  The decode loop runs
forward (i=0..9), shifting right and carrying LSBs upward.

**CallsignHashTable n12 lookup.**
The hash table is keyed by the full 22-bit hash n22.  When decoding a Type 4
message the payload carries only n12 = n22 >> 10.  The lookup scans the map for
any key in `[n12<<10, (n12+1)<<10)` — this is correct and cheap for the small
hash tables used in practice.

### Tests

- 26 unit tests in `src/tests/unit/message.rs`
- 3 full-stack roundtrip tests in `src/tests/roundtrip/message.rs`:
  `full_stack_ft8_type1`, `full_stack_ft8_free_text`, `full_stack_ft4_type1`
  (each: `pack77` → `Ft8Codec::encode` → `Ft8Mod` → `Ft8Demod` →
  `Ft8Codec::decode_hard` → `unpack77`)

---

## Phase 4 addendum — Throughput tests [DONE]

Added alongside Phase 4 completion:

- `src/tests/throughput/ft8.rs` — 5 tests: mod, demod, codec encode, codec decode,
  full roundtrip (codec encode → mod → demod → codec decode)
- `src/tests/throughput/ft4.rs` — 5 tests: same structure for FT4

Results on Apple M2 Pro (release build, 20 passes × 1 frame):

| Stage | FT8 Msps | FT4 Msps |
|---|---:|---:|
| mod | 266 | 222 |
| demod | 29 | 45 |
| roundtrip | 27 | 44 |

Demod is the bottleneck in both cases (8 Goertzel correlators × 79 symbols for
FT8; 4 × 105 for FT4). FT4 demod is faster per-sample because the frame is 2.5×
shorter and uses only 4 tones. Codec encode/decode numbers are elided by the
optimizer at constant input in release mode — this is expected for a floor test.

---

## Phase 5 — Python bindings and documentation [DONE]

This phase exposes the complete Rust FT8/FT4 stack to Python via PyO3,
following the same patterns as the existing 16-class analog/digital bindings.

### What was built

- `src/python/ft8.rs` — all PyO3 bindings (new file, ~370 lines):
  - `PyFt8Mod` / `PyFt4Mod` — waveform modulators (`modulate(tones) → complex64[N]`)
  - `PyFt8Demod` / `PyFt4Demod` — Goertzel demodulators (`demodulate(iq) → uint8[N]`)
  - `PyFt8Codec` / `PyFt4Codec` — stateless codec classes with three static methods:
    `encode`, `decode_hard`, `decode_soft`
  - `ft8_sync` / `ft4_sync` — sync functions returning `list[dict]`
    (each dict: `time_sym`, `freq_bin`, `score`, `llr: float32[174]`)
  - `ft8_pack_standard`, `ft8_pack_free_text`, `ft8_pack_telemetry` — message packing
  - `ft8_unpack` — payload → typed dict

- `src/python/mod.rs` — added `mod ft8;` and registered 6 classes + 6 functions
- `python/orion_sdr/__init__.py` — added 12 new exports
- `python/orion_sdr/__init__.pyi` — added type stubs for all new classes/functions
- `python/tests/test_ft8.py` — 22 pytest tests (all pass):
  waveform shape/roundtrip, codec encode/decode, sync detection, message roundtrips,
  full-stack (pack → encode → mod → sync → decode_soft → unpack)

### PyO3 0.28 notes

`PyObject` was removed from the root namespace in PyO3 0.28. For methods that
return `bytes | None`, use `Bound<'py, PyAny>` as the return type and
`py.None().into_bound(py)` for the None branch.

---

## Implementation order (actual)

```
Phase 1 (waveform) → Phase 2 (codec) → Phase 3 (sync) → Phase 4 (messages) → Phase 5 (bindings)
```

All five phases are complete.

---

## References

Primary sources consulted during implementation:

- **ft8_lib** (Karlis Goba, MIT licence) — C reference implementation of the
  complete FT8/FT4 encode/decode stack. The definitive source for all numeric
  constants: CRC polynomial, LDPC matrices, Gray code tables, Costas arrays,
  XOR scramble sequence, and frame structure.
  https://github.com/kgoba/ft8_lib

- **WSJT-X** (Joe Taylor K1JT et al., GPLv3) — original Fortran/C++ reference
  implementation. Authoritative for protocol semantics and message packing.
  https://sourceforge.net/p/wsjt/wsjtx/

- **"The FT4 and FT8 Communication Protocols"** (Steven Franke K9AN and
  Joe Taylor K1JT, QEX Jul/Aug 2020) — the protocol specification paper.
  Describes frame structure, LDPC code, CRC, Gray mapping, and message types.
  https://wsjt.sourceforge.io/FT4_FT8_QEX.pdf

- **"The Coding Process for FT8"** (Andy Talbot G4JNT) — concise walkthrough
  of the full encoding pipeline from message bits to transmitted tones.
  http://www.g4jnt.com/WSJT-X_LdpcModesCodingProcess.pdf
