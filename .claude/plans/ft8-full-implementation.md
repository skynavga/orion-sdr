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

## Phase 3 — Frame synchronisation: timing and frequency offset [TODO]

This phase enables decoding a raw IQ stream captured from the air, rather
than a pre-aligned synthetic block.

### Problem

An over-the-air FT8 signal arrives:
- At an unknown time offset within the 15-second window (up to ±2 s)
- At an unknown frequency offset (receiver LO error, Doppler, etc.)
- With unknown amplitude

The demodulator must find the frame, correct offsets, and produce per-symbol
soft metrics (LLRs) for the LDPC decoder.

### Tasks

- [ ] `src/sync/costas.rs` — 2-D Costas correlator
  - Slide a time×frequency grid over the IQ block
  - For each candidate (t_offset, f_offset): score = dot product of
    received spectrum vs expected Costas template at all three positions
  - Return top-N candidates ranked by score
  - Search range: ±2 s in time (≈ ±15 symbol widths), ±4 kHz in freq
- [ ] `src/sync/ft8_sync.rs` — `Ft8Sync` pipeline
  - Input: arbitrary-length IQ slice at 12 kHz
  - Stage 1: coarse Costas search → list of (t, f, score) candidates
  - Stage 2: per-candidate fine refinement (sub-symbol timing, sub-bin freq)
  - Output: `Ft8SyncResult { t_samples: i64, f_hz: f32, score: f32 }`
- [ ] Soft-symbol extractor
  - Given a sync result, extract per-symbol log-energy vectors `[f32; 8]`
    (one energy per tone per symbol)
  - Compute LLRs via: for each bit position, LLR = log(P(b=0)/P(b=1))
    approximated by max-log over tones consistent with that bit value
  - Output: `[f32; 174]` ready for `Ft8Decoder::decode_soft`
- [ ] Integration test: synthesise frame with known message + AWGN at
  several SNR levels (0 dB, −5 dB, −10 dB, −15 dB); verify decode success
- [ ] FT4 equivalents

### Notes

- The coarse Costas search is the computationally expensive step.
  Consider a waterfall (time×frequency power grid) implementation using
  the existing Goertzel correlator to avoid full FFT dependency.
- Sync may be factored as a standalone `Block` or as a free function —
  decide based on whether stateful streaming is needed.

---

## Phase 4 — Message packing: 77-bit structured payloads [TODO]

This phase adds the application layer: encoding human-readable QSO
messages into the 77-bit payload and decoding them back.

### Message types (FT8 standard)

| Type | Bits | Description |
|------|------|-------------|
| 0    | 77   | Free text / telemetry (13 chars, base-42 alphabet) |
| 1    | 77   | Standard QSO: callsign1 + callsign2 + grid/report |
| 2    | 77   | EU VHF contest |
| 3    | 77   | ARRL field day |
| 4    | 77   | Nonstandard callsign (hashed) |
| 5    | 77   | Telemetry (71 bits arbitrary) |

### Phase 4a — Type 1 (standard QSO) [primary target]

- [ ] `src/message/callsign.rs` — standard callsign encoder/decoder
  - 28-bit encoding for callsigns matching `[A-Z0-9]{1,3}[0-9][A-Z]{1,4}`
  - Special codes: CQ (0), DE, QRZ, etc.
- [ ] `src/message/grid.rs` — Maidenhead grid square encoder/decoder
  - 15-bit encoding for 4-character grid (e.g. "FN31")
- [ ] `src/message/report.rs` — signal report / RRR / 73 encoder/decoder
- [ ] `src/message/type1.rs` — `Ft8MessageType1 { from, to, grid_or_report }`
  - `pack(&self) -> [u8; 77]`
  - `unpack(bits: &[u8; 77]) -> Option<Self>`
- [ ] Unit tests: known pack/unpack roundtrips matching WSJT-X output

### Phase 4b — Type 0 (free text)

- [ ] `src/message/freetext.rs` — base-42 encode/decode
  - Alphabet: `0-9A-Z /?.,` (42 chars)
  - Up to 13 characters → 71 bits

### Phase 4c — remaining types (if needed)

Types 2, 3, 4, 5 can be added incrementally based on need.

---

## Phase 5 — Python bindings and documentation [TODO]

- [ ] PyO3 bindings for `Ft8Encoder`, `Ft8Decoder`, `Ft8Sync`, `Ft8MessageType1`
  - Expose `encode(msg_str) -> bytes` and `decode(iq: np.ndarray) -> str | None`
- [ ] Pytest integration tests mirroring Rust roundtrip tests
- [ ] Update `docs/modulate.md`, `docs/demodulate.md`, `docs/api.md`
- [ ] Throughput benchmark: encode+decode cycles/sec at target SNR

---

## Implementation order recommendation

```
Phase 2 (codec) → Phase 3 (sync) → Phase 4a (Type 1 messages) → Phase 5 (bindings)
```

Phase 2 is self-contained and testable without real RF. Phase 3 depends on
Phase 2 (needs LLRs → soft decode). Phase 4 is independent of 2 and 3 and
could be parallelised. Phase 5 wraps everything at the end.

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
