<!--
  Copyright (c) 2026 G & R Associates LLC
  SPDX-License-Identifier: MIT OR Apache-2.0
-->

# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.0.52] - 2026-08-03

NB-DVB-T Phase 5: replaces the DVB-T batch frame and super-frame free functions
with stateful modulator/demodulator objects, making DVB-T consistent with every
other waveform in the crate, and folds the integer-CFO correction into the demod
as a construction-time flag. This is an API-shape change (the object bodies are
the former free-function bodies verbatim), so decoded output is byte-identical.

### Changed

- **Breaking:** the DVB-T frame and super-frame paths are now objects, not free
  functions. `dvb_t_frame_modulate` / `dvb_t_frame_demodulate` become
  `DvbTFrameMod::new(params).modulate(payload)` /
  `DvbTFrameDemod::new(params).decode(iq, n_symbols, payload_len)`; likewise
  `dvb_t_super_frame_modulate` / `dvb_t_super_frame_demodulate` become
  `DvbTSuperFrameMod` / `DvbTSuperFrameDemod`. These were the crate's last
  mod/demod free-function outliers; every other waveform (and COFDM's own
  modulator and streaming receiver) was already an object.
- Integer-CFO correction is now a construction-time builder flag on the
  demodulator, `DvbTFrameDemod::new(params).with_integer_cfo_correction(true)`
  (and the same on `DvbTSuperFrameDemod` and, via a constructor argument on the
  Python `DvbTFrameStreamDemod`). When enabled, the demod estimates the
  whole-subcarrier offset from the continual pilots after its own guard-interval
  acquisition and rotates it out internally, replacing the previous
  caller-assembled pre-pass. The estimator `sync::dvb_t_integer_cfo` remains
  public for standalone use. Always-on, the correction costs a few percent of the
  decode (~4–6%, `throughput_dvb_t_integer_cfo`).
- **Breaking:** the shared modulation-and-coding fields (guard interval,
  constellation, code rate) are extracted into a `DvbTLinkParams` that both
  `DvbTFrameParams` and `DvbTSuperFrameParams` embed as a `link` field.
  `DvbTFrameParams` keeps its `frame_number` and an 8-bit `cell_id` (the byte a
  frame's TPS carriers actually transmit); `DvbTSuperFrameParams` keeps the full
  16-bit `cell_id` (split across the four frames). Delegating
  `guard()` / `constellation()` / `code_rate()` accessors are provided on both.
  The Python string constructors are unchanged (they build the link set
  internally).

## [0.0.51] - 2026-08-03

NB-DVB-T Phase 4b: adds the capture-readiness pieces on top of the conformant
single frame — a four-frame super-frame driver, a streaming receiver, and an
integer-CFO estimator — and exposes the super-frame and streaming paths to
Python. All three are self-verified in-repo; the external-IQ capture harness
(the fourth audited gap) remains deferred pending a published capture. Folding
the integer-CFO correction into the receivers as a construction-time flag is
left to a follow-up that makes the DVB-T frame/super-frame paths stateful
objects.

### Added

- DVB-T super-frame orchestration (`modulate::dvb_t_super_frame` +
  `demodulate::dvb_t_super_frame`): the multi-frame driver over the single-frame
  conformant path (EN 300 744 §4.4/§4.6). It emits and recovers a four-frame
  super-frame (frame numbers 0..3) with the standard's super-frame structure —
  the TPS synchronization word alternates every frame (frames 1&3 use
  `TPS_SYNC_WORD_13`, frames 2&4 use `TPS_SYNC_WORD_24`, §4.6.2.2) so a receiver
  can find the super-frame boundary, and the 16-bit cell identifier is split
  across the super-frame (b15..b8 in frames 1&3, b7..b0 in frames 2&4,
  §4.6.2.10) and reassembled on RX. `DvbTSuperFrameParams` carries the full
  16-bit cell id; the payload is split into four contiguous parts (zero-padded to
  a common length) and the RX trims each back to its recorded length. Each frame
  codes its own part independently; the standard's byte-continuous stream (an RS
  packet straddling a frame boundary for fractional-per-frame rates, e.g. QPSK
  r3/4 = 94.5 pkts/frame) is left to a future streaming path.
- Streaming DVB-T receiver (`demodulate::dvb_t_stream::DvbTFrameStreamDemod`):
  `feed` / `flush` / `clear` accumulate-and-drain over the batch conformant
  decoder, chunk-boundary-invariant, mirroring `OfdmFrameStreamDemod` — the
  conformant path is no longer batch-only.
- DVB-T integer-CFO estimator (`sync::dvb_t_integer_cfo`): the guard-interval
  acquisition resolves the CFO only within ±½ a subcarrier, so a capture with a
  larger front-end offset is shifted by whole subcarriers and will not demap. The
  45 continual pilots (fixed positions, boosted 16/9) anchor that integer offset —
  given an aligned symbol's FFT bins, the estimator searches trial shifts and
  returns the one maximizing the continual-pilot energy (`IntegerCfoResult
  { bins, confidence }`). This is the DVB-T-native counterpart to the OFDM
  preamble path's training-symbol integer-CFO recovery, which a preamble-less
  frame cannot use. Correction is caller-driven for now (rotate by −k·fs/n_fft
  before decoding); adds a `waveform::dvb_t::continual_pilot_bins` helper. Because
  the pilot peak is modest (45 of 1705 carriers, ~1.7× over the all-shifts mean),
  accumulate `|X|²` per bin over several symbols under noise.
- Python bindings for the super-frame and streaming paths
  (`src/python/dvb_t_frame.rs`): `DvbTSuperFrameParams`,
  `dvb_t_super_frame_modulate` → `DvbTSuperFrame`, `dvb_t_super_frame_demodulate`
  → `DvbTRxSuperFrame` (`.payload` / `.cell_id`), and `DvbTFrameStreamDemod`
  (`feed` / `feed_with_errors` / `flush` / `clear` / `buffered`), with `.pyi`
  stubs and worked examples in `docs/python.md`.

### Changed

- Reorganized the throughput benches by waveform layer: new
  `tests/performance/throughput/cofdm.rs` (COFDM frame-chain benches + the
  `frame_chain` driver) and `dvbt.rs` (DVB-T bandwidth sweep, conformant frame,
  super-frame, and integer-CFO benches); `fec.rs` is now pure FEC-block kernels +
  construction costs. `docs/performance.md` gains super-frame and integer-CFO
  subsections and per-waveform run commands.
- Documentation: `docs/modulate.md` and `docs/demodulate.md` gain super-frame and
  streaming RX sections (replacing the stale "not yet provided" note),
  `docs/dvb.md` documents the three frame-transport layers and the integer-CFO
  acquisition step, and `docs/demodulate.md` carries a worked integer-CFO wiring
  example.

## [0.0.50] - 2026-08-02

NB-DVB-T Phase 4a: fixes the conformant DVB-T frame's channel estimation, packs
the frame with real coded data, exposes the frame path to Python, and optimizes
the receiver. A diagnostic re-audit found the decode-vs-SNR anomaly documented in
0.0.49 was a channel-estimation bug — the receiver fed the 17 TPS carriers to the
equalizer as boosted pilots, but the modulator transmits data-power DBPSK on
them — not the short-frame/band-edge effect previously suspected. With that fixed
the decode-vs-SNR waterfall is clean and ordered by nominal robustness, and both
0.0.49 known limitations (the anomaly and the Rust-only frame path) are resolved.

### Fixed

- DVB-T conformant-frame channel estimation: the RX scattered-pilot equalizer no
  longer uses the 17 TPS carriers as channel references. The modulator transmits
  data-power DBPSK on them (EN 300 744 §4.6), not the boosted `w_k` pilot value
  the grid records, so dividing the received cell by that known value produced a
  bogus channel estimate that the interpolator smeared onto the data carriers
  straddling each TPS carrier — a deterministic, SNR-invariant pre-FEC error
  floor. `ScatteredGridCycle` now precomputes a per-phase reference-pilot set
  (continual + scattered only) that `ScatteredPilotExtractor::current_pilot_bins`
  returns; the equalizer interpolates across the TPS bins instead. The
  decode-vs-SNR curves are now robustness-ordered (QPSK r1/2 locks by ~4 dB,
  16-QAM r3/4 by ~12–15 dB). Guarded by a new noiseless regression asserting
  every equalized data carrier equals the transmitted one.

### Added

- Python bindings for the conformant DVB-T frame (`src/python/dvb_t_frame.rs`):
  `DvbTFrameParams`, `dvb_t_frame_modulate`, `dvb_t_frame_demodulate` (returning
  the payload and the recovered `TpsWord`), and the `nb_bandwidth_fs` /
  `nb_bandwidth_occupied_hz` helpers, with a worked round-trip example in
  `docs/python.md`. The conformant frame path is no longer Rust-only.
- Null-packet stuffing (`waveform::dvb_t_ts::{ts_null_packet, ts_stuff_null_packets}`):
  the modulator fills a short payload with MPEG-2 null packets (PID `0x1FFF`) so
  every data carrier carries real coded data, as EN 300 744 §4.4/§4.3.1 require
  (a compliant signal never leaves carriers zeroed). The receiver trims to the
  original payload length, so the stuffing is transparent.

### Changed

- DVB-T conformant-frame receiver throughput (~4.0 → ~13 Msps): the per-symbol
  pilot interpolation now binary-searches the sorted pilot set (O(data·log
  pilots)), sorts the pilot bins once, and reuses a ratio scratch buffer, so it
  allocates nothing per symbol. Same decoded bits; benefits every
  `PerSymbolPilotInterp` user. (The conformant modulate figure drops ~90 → ~33
  Msps as a consequence of frame-filling — it now runs the full payload FEC over
  a whole frame of RS packets.)
- `docs/performance.md`: refreshed all DVB-T figures, added roundtrip columns to
  the bandwidth-sweep and conformant-frame tables, and replaced the retracted
  band-edge narrative.

## [0.0.49] - 2026-08-02

NB-DVB-T (narrowband DVB-T for amateur DATV) Phase 3: the fully conformant,
preamble-less DVB-T on-air frame (ETSI EN 300 744), closing the last fidelity
gaps. A frame now carries an MPEG-2 transport-stream payload with TPS signalling
on the 17 reserved carriers, four-phase scattered pilots, DVB-T-exact
soft-decision, and guard-interval acquisition — no Schmidl & Cox preamble and no
`OrionSdr` header. Every stage is bit-exact to the standard where a known answer
exists, decoded end-to-end by orion-sdr's own RX. Also adds the narrowband
bandwidth-mode API and DVB-T benchmarks/documentation.

### Added

- TPS signalling (`waveform::dvb_t_tps`): a standalone GF(2^7) BCH(67,53) t=2
  code (generator `h(x)=0x4377`, primitive poly `x^7+x^3+1`; kept separate from
  the hot GF(2^8) `Gf256`/`Bch`), the 68-bit `TpsWord` pack/unpack with the
  standard signalling tables and sync words, and `TpsEncoder`/`TpsDecoder` for
  DBPSK along the symbol axis (the decoder averages the 17 carriers, needing no
  absolute phase reference).
- Guard-interval acquisition (`sync::dvb_t_gi_sync`, `GiSyncConfig`): the van de
  Beek ML cyclic-prefix timing/CFO estimator (`|γ| − ρ·Φ`) for preamble-less
  OFDM, tunable via the energy weight `ρ` and a coherent-accumulation bound.
- DVB-T soft-decision: `dvb_t_soft_llr` (Figure-9a max-log LLRs) wired into the
  scattered frame path, giving the DVB-T link real soft-decision coding gain.
- MPEG-TS payload layer (`waveform::dvb_t_ts`): 188-byte packet adaptation +
  energy dispersal (8-packet PRBS re-init, first-sync-byte inversion
  0x47→0xB8, sync-bytes skipped-but-clocked; first randomized byte 0x03).
- `HeaderFormat::DvbTps` + `HeaderFormat::has_header_block()`; the two hardcoded
  `== OrionSdr` dispatch sites become a semantic check. `OfdmConfig::validate`
  and the Python `with_header_format` gain the variant.
- Conformant frame assemblers, direction-split: `modulate::dvb_t_frame`
  (`dvb_t_frame_modulate`, `DvbTFrame`) and `demodulate::dvb_t_frame`
  (`dvb_t_frame_demodulate`, `DvbTRxFrame`, `DvbTRxError`), sharing
  `DvbTFrameParams` from `waveform::dvb_t`. `decode_chain` is now `pub` for reuse.
- Narrowband bandwidth-mode API: `NbBandwidth {Bw333kHz, Bw1MHz, Bw2MHz}` with
  `occupied_hz()`/`fs()`/`is_pluto_continuous_tx()`, composed via a new generic
  `OfdmConfig::with_fs()` builder.
- Benchmarks: DVB-T 2K throughput swept over the three bandwidth modes, a
  conformant-frame end-to-end throughput bench, and a `snr/dvb_t.rs` BER /
  frame-decode-vs-SNR sweep. New `docs/dvb.md`; DVB-T results and examples added
  to `docs/performance.md` / `modulate.md` / `demodulate.md` / `python.md`.

### Changed

- The frame layer routes payload symbols through the DVB-T-exact constellation
  (map + soft-LLR) on a DVB-T constellation; BPSK header blocks keep the generic
  path. Non-DVB-T links are unaffected.

### Known limitations

- The conformant frame's decode-vs-SNR is erratic and not ordered by nominal
  robustness: a small payload is padded to a full 68-symbol frame, so the real
  coded data occupies only ~5–13 of 68 symbols and the per-symbol scattered-pilot
  equalizer's band-edge residual dominates. Acquisition and TPS are unaffected
  (verified). Candidate fixes (multi-codeword frame packing, MMSE equalizer) are
  deferred to a later optimization pass. The conformant `dvb_t_frame` path is not
  yet exposed through the Python bindings.

## [0.0.48] - 2026-08-01

NB-DVB-T (narrowband DVB-T for amateur DATV) Phase 2: DVB-T's four-phase
scattered-pilot channel estimation (ETSI EN 300 744 §4.5). Pilots at
`k mod 12 == 3·(l mod 4)` cycle over four symbols, giving far denser per-symbol
channel tracking than the 45 continual pilots alone — enough to decode a
frequency-selective (multipath) channel that the continual-pilots-only path
cannot. Acquisition still uses the Schmidl & Cox preamble + `OrionSdr` header,
and TPS carriers hold a placeholder value (DBPSK signalling, guard-interval
acquisition, and full soft-decision are Phase 3).

### Added

- `dvb_t_2k_plans(guard) -> [CarrierPlan; 4]`: the four symbol-phase 2K carrier
  plans, each reserving the 45 continual + phase-`p` scattered + 17 TPS carriers
  as boosted `w_k`-valued pilots, with **exactly 1512 data carriers** in every
  phase (the standard fixes this count constant). New `scattered_pilot_indices`,
  `tps_carrier_indices` / `DVB_T_TPS_CARRIERS_2K` (verified vs. EN 300 744
  Table 8), `DVB_T_SCATTERED_PHASES`, and `GuardInterval::from_cp_len_2k`.
- `ScatteredPilotMapper` / `ScatteredPilotExtractor` (in `src/waveform/dvb_t.rs`):
  non-`Block` frame-layer orchestrators owning the four `CarrierGrid`s plus a
  symbol-phase counter, selecting the phase-appropriate grid per symbol.
- `OfdmEqualizer::set_pilot_bins`: an additive method to install each symbol's
  pilot set before `process` (under `PerSymbolPilotInterp`), leaving the
  per-symbol `Block` contract unchanged.
- `OfdmConfig::dvb_t_scattered` (+ `with_dvb_t_scattered`) and
  `dvb_t_scattered_config(guard, occupied_hz)`, which assemble a scattered-pilot
  DVB-T link over a representative phase-0 plan.
- Scattered-pilot tests: all four plans (× four guard intervals) expose exactly
  1512 data carriers; the scattered index formula and TPS table; a bit-exact
  four-phase noiseless roundtrip; and the load-bearing pair
  (`dvb_t_scattered_multipath_decodes` / `dvb_t_scattered_needed_for_multipath`)
  showing a 2-tap channel the scattered per-symbol estimate recovers but the
  continual-pilots-only path cannot.

### Changed

- The frame layer maps/demaps payload symbols through the four-phase grid
  rotation when `dvb_t_scattered` is set: one orchestrator spans a frame's
  header-then-payload symbols so `l = 0` is the first header symbol and TX/RX
  stay phase-aligned. Non-DVB-T links are unaffected (the flag defaults `false`).

### Fixed

- Dropped a redundant `.clone()` on the `Copy`-type `OfdmPreamble` in the
  throughput tests (a pre-existing Clippy warning).

## [0.0.47] - 2026-08-01

NB-DVB-T (narrowband DVB-T for amateur DATV) Phase 1: a conformant DVB-T payload
FEC chain and 2K-mode carrier map, assembled into an end-to-end fs-scaled link.
The payload coding is bit-exact to ETSI EN 300 744 V1.6.2 where verified;
acquisition still uses the crate's Schmidl & Cox preamble + `OrionSdr` header
(DVB-T's guard-interval/TPS acquisition, full soft-decision, and the 188-byte
TS-packet layer are later phases).

### Added

- DVB-T K=7 punctured convolutional inner code (G0=0o171, G1=0o133, 64-state
  soft Viterbi), selected via `ConvCode::DvbK7` on `InnerFec::Convolutional`; the
  K=5 code (shared with PSK31) is unchanged.
- `src/waveform/dvb_t.rs` (new `waveform` module): DVB-T energy-dispersal PRBS
  whitener (`DvbTEnergyDispersal`; 1+X^14+X^15, MSB-first, first byte 0x03
  known-answer), the Figure-9a constellation mapping
  (`dvb_t_map_symbol`/`dvb_t_demap_symbol`, QPSK/16-QAM/64-QAM), 2K numerology
  (n_fft=2048, 1512 data carriers), the 45 continual-pilot indices with the w_k
  PRBS (X^11+X^2+1), `GuardInterval`, narrowband fs-scaling
  (`fs = BW·2048/1705`; 333 kHz/1 MHz/2 MHz constants), `dvb_t_mcs_table`
  (QPSK/16-QAM × K=7 conv × RS(204,188)), and `dvb_t_config` assembling the full
  2K `OfdmConfig`.
- `ScramblerKind::DvbTEnergyDispersal` wired into the frame scramble path
  (byte-domain, before the outer FEC); Python `OfdmConfig.with_dvb_t_scrambler`
  and a `"dvb_t"`/`"convolutional_k7"` inner-FEC selector.
- End-to-end DVB-T frame roundtrip tests (333 kHz/1 MHz/2 MHz + AWGN), a
  DVB-T-exact-constellation-through-OFDM hard-decision test, and a DVB-T 2K
  throughput benchmark (`frame_chain` generalized to take a cfg/preamble).

### Changed

- `InnerFec::Convolutional` gains a `code` field selecting the mother code
  (K=5 or DvbK7); the convolutional coder is generalized over `ConvCode`.

## [0.0.46] - 2026-08-01

Dual-mode (stream/frame) FEC primitives — the first step toward NB-DVB-T
(narrowband DVB-T) support. Adds stateful, resettable channel-coding blocks that
work in both the existing frame-oriented COFDM link and a continuous-stream
pipeline, so a DVB-T stream chain can be built on them without per-frame
interleaver-flush overhead. No change to existing frame-layer output (byte-
identical).

### Added

- Streaming Forney convolutional byte interleaver (`ConvInterleaver` /
  `ConvDeinterleaver` in `fec/interleaver.rs`): per-branch delay-line FIFOs with a
  `feed`/`flush`/`reset` interface, `conv_roundtrip_delay`, and a `dvbt()`
  (`I=12`, `M=17`) helper. Stream mode carries delay-line state across the whole
  stream (zero per-frame overhead); frame mode is `reset`+`feed`+`flush` per unit.
- `InterleaverKind::Convolutional { branches, depth }`, wired through the frame
  layer's interleave/deinterleave and block-size accounting in reset-per-frame
  mode; exposed to Python as `OfdmConfig.with_conv_interleaver`.
- Streaming additive PN scrambler (`PnScramblerStream` in `fec/scrambler.rs`) that
  carries the LFSR register across `feed` calls for continuous energy dispersal,
  with `reset`. Bit-identical to `PnScrambler` (`feed(&whole) == scramble(&whole)`;
  chunked feeds == one-shot).

### Changed

- `PnScrambler::scramble` refactored to share a per-byte `scramble_byte` helper
  with the new streaming variant — output unchanged.

## [0.0.45] - 2026-07-28

COFDM FEC performance pass: the concatenated-FEC decode path is substantially
faster, with no change to decoded output (bit-exact) or the on-air default
coding. Adds an opt-in min-sum LDPC decode rule.

### Added

- Selectable LDPC check-node decode rule (`DecodeRule::SumProduct` (default) /
  `MinSum` / `ScaledMinSum(α)`), chosen per link via
  `OfdmConfig::with_ldpc_decode_rule` (Python: `with_ldpc_decode_rule`). Applies
  to the payload decode only (the header always uses sum-product); scaled
  min-sum runs ~2× faster for a ≲0.3 dB coding-gain cost.
- `CodecCache`: a per-link cache of constructed FEC codes, shareable across a
  modulator/demodulator pair (`OfdmFrameMod::with_cache`,
  `OfdmFrameStreamDemod::with_cache`, and an optional `cache` argument to the
  batch `demodulate_frame`); exposed to Python as `CodecCache`.
- `Gf256::shared()`: a process-wide GF(2^8) table singleton.
- COFDM FEC/interleave/scrambler throughput and LDPC decode-rule SNR benchmarks
  (`throughput::fec`, `snr::ldpc_decode_rule`).

### Changed

- The COFDM frame layer builds each FEC code once per link instead of once per
  frame (`Ldpc`/`Bch`/`ReedSolomon` are cached, and `Bch`/`ReedSolomon` share
  one `Gf256`), removing the dominant per-frame construction cost — the LDPC
  parity-check matrix build alone was milliseconds per frame.
- The LDPC sum-product decoder stores its per-edge messages in a flat
  compressed-sparse-row layout, caches `tanh(msg/2)` per edge, and precomputes
  its Tanner-graph edge indices — a bit-exact ~2–3× decode-throughput
  improvement on the fixed-family codes.
- `demodulate_frame` takes an additional optional `cache` argument
  (`None` preserves the previous per-call behavior).
- Documentation: acronym glossary moved to `docs/acronyms.md`; all
  OFDM/COFDM performance tables refreshed; new COFDM modulator/demodulator and
  Python usage guides, including non-default FEC/interleave/scramble configs.

### Tests

- 292 default (`cargo test --release`) and 360 with `--features throughput`;
  new min-sum roundtrip and decode-rule-equivalence unit tests.

## [0.0.44] - 2026-07-27

COFDM: a concatenated-FEC, framed (MAC-layer) link built on the OFDM physical
layer. See [docs/ofdm.md](docs/ofdm.md).

### Added

- New waveform-agnostic `fec/` module: `Gf256` (GF(2^8) arithmetic),
  parameterized fixed-family LDPC (`Ldpc`/`LdpcCode`, distinct from FT8's
  `codec/ldpc.rs`), binary BCH (`Bch`), Reed–Solomon (`ReedSolomon`, DVB-T
  RS(204,188) t=8), punctured convolutional coding with an LLR-domain soft
  Viterbi (`conv`, rates 1/2–7/8), a block interleaver (`BlockInterleaver`), a
  PN scrambler (`PnScrambler`), and the frame/scheme types (`FramePacket`,
  `FrameMetadata`, `RxError`, and the `OuterFec`/`InnerFec`/`InterleaverKind`/
  `CrcKind`/`ScramblerKind`/`ScramblerPos`/`HeaderFormat` descriptors).
- Generic `crc16` (CRC-16/CCITT-FALSE) and `crc32` (CRC-32/ISO-HDLC) in
  `codec/crc.rs`.
- `OfdmConfig` frame-layer fields and `with_*` builder methods (outer/inner
  FEC, the two interleavers, header format, header/payload CRC, scrambler and
  its chain position), all defaulted-off so the bare OFDM symbol pipeline is
  unchanged; `OfdmConfig::validate()` for the frame-layer configuration.
- COFDM frame transmitter `OfdmFrameMod` and `Mcs`/`McsTable` (per-frame
  adaptive modulation and coding), the batch `demodulate_frame`, and the
  streaming receiver `OfdmFrameStreamDemod` (feed/flush, with preamble
  acquisition, CFO correction, training-symbol equalization, concatenated
  decode, and CRC — mirroring `Ft8StreamDecoder`'s shape).
- Python bindings for the frame layer (`FramePacket`, `McsTable`,
  `OfdmFrameMod`, `OfdmFrameStreamDemod`, `demodulate_frame`) plus the
  `OfdmConfig.with_*` FEC/CRC/scrambler/header configuration methods.
- `docs/ofdm.md` (OFDM PHY conventions moved out of `docs/design.md`, plus the
  new COFDM frame-layer design); COFDM throughput and frame-error-rate tables
  in `docs/performance.md`; new acronym-glossary rows (BCH, BM, COFDM, GF, MAC,
  MCS, MMSE, PN, RS, LFSR).

## [0.0.43] - 2026-07-19

Post-review hardening of the OFDM stack: one correctness fix plus test and
documentation improvements. No public API changes.

### Fixed

- `CarrierGrid::from_plan` now validates the `CarrierPlan` and panics on an
  invalid one (out-of-range carrier index, data/pilot overlap, or empty data
  set). Previously `validate()` was only enforced in the Python bindings, so
  the Rust pipeline silently accepted an overlapping data/pilot carrier —
  `GridMap` then overwrote the data symbol with the pilot value, a
  wrong-result bug with no other signal. This single check guards every OFDM
  construction path (`OfdmMod`/`OfdmDemod`/`OfdmEqualizer`).

### Changed

- Removed a dead, unreachable circular-wrap branch from the pilot-interpolation
  equalizer (`interpolate_at`) and corrected its documentation to describe the
  actual behavior: linear interpolation between bracketing pilots, with a
  nearest-pilot hold at the band edges (no wrap across bin 0). Refreshed
  `OfdmConfig` and `EqualizerMethod` docs that still carried mid-development
  "this release adds…" language.

### Tests

- Added Rust coverage for previously untested paths: TX/RX gain application
  (baseband and RF-upconversion), the IFFT `1/N` scale in isolation,
  `OfdmMod::modulate` partial-symbol zero-padding, and the
  `PerSymbolPilotInterp` empty-pilots no-op and out-of-span nearest-pilot
  fallback. Added `carrier_grid_from_plan_panics_on_{overlap,out_of_range}`.
- Added Python `TestOfdmEqualizer` covering the fused-equalizer path reachable
  only from `PyOfdmDemod`: the unknown-equalizer error, `pilot_interp`
  selection, `estimate_channel()` + multipath demodulation, and a
  with/without check proving `estimate_channel` is load-bearing.
- Tightened tests that passed structurally rather than by verifying: the CFO
  aliasing test now asserts the predicted aliased value (not an
  `atan2`-guaranteed bound); the no-false-positive-on-noise test sweeps
  multiple seeds; EVM and soft-LLR tests are anchored against the transmitted
  bits; a new test pins the EVM dB formula exactly; and the CFO round-trip
  asserts estimate accuracy.
- Fixed pre-existing `needless_range_loop` clippy lints in the OFDM unit tests
  surfaced under `cargo clippy --all-targets`.

## [0.0.42] - 2026-07-18

### Added

- Python bindings for the full OFDM stack (`src/python/ofdm.rs`):
  `OfdmConfig`, `OfdmMod`, `OfdmDemod` (with `demodulate`,
  `demodulate_soft`, `estimate_channel`), `OfdmRxFrame`,
  `build_ofdm_rx_frame`, `ofdm_sync`, and `generate_ofdm_preamble`,
  re-exported from `orion_sdr` alongside the rest of the public API.
  This is Release I / Phase 9 of the OFDM support roadmap, and stabilizes
  the OFDM public API surface across Rust and Python.
- `python/tests/test_ofdm.py`: 11 tests covering config/waveform
  round trip, packet sync + CFO acquisition, and soft-symbol/RX
  frame diagnostics through the Python bindings.
- OFDM/multicarrier coverage across `docs/*.md` and `README.md`:
  Rust and Python API references, usage examples (all verified against
  the built extension), freshly measured throughput/BER/packet-sync
  performance data, an expanded acronym glossary, and concrete
  numerology guidance (CP length, subcarrier spacing, `n_fft`) for
  choosing `CarrierPlan`/`OfdmConfig` parameters.

### Changed

- Crate framing updated to "HF-through-EHF" across `Cargo.toml`,
  `pyproject.toml`, `CLAUDE.md`, and `README.md`, replacing the
  narrower "HF-to-UHF" description.

### Fixed

- Stale documentation: incorrect Python binding/test counts, a
  reference to a nonexistent `BasicChain` scheduler, and an outdated
  test-gating file path in `docs/conventions.md`.

## [0.0.41] - 2026-07-18

### Added

- `bpsk_soft_llr`, `qpsk_soft_llr`, `qam_soft_llr<BITS>`, and
  `OfdmSoftDemod` (`src/demodulate/ofdm.rs`): max-log soft (LLR) demapping
  per constellation order, dispatched by `ConstellationOrder`. Positive
  LLR indicates the bit is more likely 0, matching the crate-wide LLR
  convention. `OfdmSoftDemod` is a separate type from `OfdmDecider` (not
  a mode flag), mirroring the existing `Ft8Demod`/`Ft8Codec::decode_soft`
  split. No mandatory FEC ships in this release — soft LLRs are the
  deliverable, directly usable by an external/user-supplied FEC layer.
  This is Release H / Phase 8 of the OFDM support roadmap.
- Unit test `ofdm_soft_llr_sign_matches_hard_decision`, checking LLR sign
  against `OfdmDecider`'s hard output across all five constellation
  orders (BPSK, QPSK, QAM-16/64/256).

### Changed

- Loosened `axis_scale`/`build_axis_table` in `src/modulate/qam.rs` to
  `pub(crate)` so the soft-LLR path reuses `QamMapper`/`QamDecider`'s
  exact Gray-coded amplitude table instead of duplicating it.

### Fixed

- The QAM max-log LLR formula initially computed `d0_sq - d1_sq`
  (distance-to-nearest-bit=0-point minus distance-to-nearest-bit=1-point)
  where it needed `d1_sq - d0_sq` — the sign was inverted relative to the
  "positive LLR ⇒ bit more likely 0" convention. Caught immediately by
  `ofdm_soft_llr_sign_matches_hard_decision` before merge.

## [0.0.40] - 2026-07-18

### Added

- `OfdmEqualizer` and `EqualizerMethod` (`src/demodulate/ofdm.rs`):
  frequency-domain channel equalization for OFDM, supporting
  frequency-selective (multipath) channels. `TrainingSymbolHold` (the
  default) estimates the channel once from the shared training symbol
  and holds it for the packet — the correct choice, not just the
  simplest, for this feature's predominantly line-of-sight VHF–EHF/L–Ka
  target bands. `PerSymbolPilotInterp` is the opt-in for genuinely
  time-varying channels, re-estimating every data symbol via
  frequency-domain linear interpolation between `CarrierGrid`'s pilot
  bins. `OfdmEqualizer` is a standalone `Block`, sitting between
  `FftBlock` and `GridExtract`, not fused into `OfdmDemod`, so it can be
  swapped or disabled independently. This is Release G / Phase 7 of the
  OFDM support roadmap.
- Unit tests (known static channel correction, pilot interpolation), a
  roundtrip test recovering exact bits through a synthetic 2-tap FIR
  multipath channel (delay spread within `cp_len`, this release's
  explicit scope limit), and multipath BER-vs-SNR characterization
  sweeps for QPSK and QAM-16.

### Changed

- Loosened `sync::ofdm_sync::training_symbol_freq_pattern` to
  `pub(crate)` so `OfdmEqualizer` divides by the exact same known
  frequency-domain pattern the integer-CFO estimator correlates
  against, reusing Release F's shared training symbol directly.

## [0.0.39] - 2026-07-18

### Added

- `TrainingSymbol` and `OfdmPreamble::with_training_symbol()`
  (`src/sync/ofdm_sync.rs`): a dedicated known training symbol (full
  `n_fft` + CP, known value on every subcarrier bin) for wide-range
  integer-CFO recovery, reused by a later release's channel estimator.
  `generate_ofdm_preamble()` emits it when present.
- Integer-CFO estimation in `ofdm_sync()`: after the fractional CFO and
  timing already found, the training symbol is FFT'd and correlated
  against its known frequency-domain pattern across candidate circular
  bin shifts, run only on the top few timing candidates to bound cost.
  This is Release F / Phase 6 of the OFDM support roadmap.
- `OfdmSyncResult::integer_cfo_bins: i32`: whole subcarrier-spacing
  units, split from `cfo_hz` (which remains fractional-only). Total CFO
  is `cfo_hz + integer_cfo_bins as f32 * subcarrier_spacing`.
- Unit tests (multi-spacing integer-CFO recovery, combined
  fractional+integer total-CFO accuracy), a roundtrip test recovering
  exact bits through a CFO well beyond Release E's ±½-spacing capture
  range, and a wide-CFO acquisition-vs-SNR characterization sweep.

### Fixed

- Clarified (via test correction, not an algorithm change) that the
  Schmidl & Cox fractional estimator's ambiguity period is
  `fs / repeat_len`, not `fs / n_fft` as initially assumed in a Release E
  test — for a preamble with `repeat_len = n_fft / 2` that's two
  subcarrier spacings, so `cfo_hz` alone can legitimately land outside
  ±½ a spacing in some cases. `ofdm_sync`'s combined
  fractional+integer total was already correct.

## [0.0.38] - 2026-07-18

### Added

- `OfdmPreamble`, `OfdmSyncResult`, `generate_ofdm_preamble()`, and
  `ofdm_sync()` (`src/sync/ofdm_sync.rs`, new file): packet sync and
  fractional CFO/timing acquisition for OFDM via a Schmidl & Cox-style
  repeated-segment preamble, generic and not tied to any standard. This
  is Release E / Phase 5 of the OFDM support roadmap.
- CFO capture range is documented and locked in by regression test as
  unambiguous within ±½ the subcarrier spacing (±`fs / (2 · repeat_len)`);
  wider offsets alias, with integer-CFO recovery deferred to a later
  release.
- Unit tests (known-offset acquisition, CFO estimate accuracy, aliasing
  beyond the documented capture bound, no false positives on noise), a
  full roundtrip test driving an unknown packet start and CFO correction
  through `OfdmDemod`/`OfdmDecider`, and an acquisition-probability-vs-SNR
  characterization sweep.

### Fixed

- The Schmidl & Cox correlation-phase timing metric alone forms a wide
  plateau (not a sharp peak) for a purely periodic preamble, since the
  correlation window stays fully coherent at any offset inside the
  repeated structure, not only at the true start. `ofdm_sync` now breaks
  the tie using the correlated window's own energy, which is maximized
  only where every correlated sample is real preamble signal, giving a
  sample-exact, unbiased timing estimate.

## [0.0.37] - 2026-07-18

### Added

- `OfdmDemod` (`src/demodulate/ofdm.rs`, new file): OFDM receiver pipeline —
  `CyclicPrefixRemove` → `FftBlock` → `GridExtract`, the exact inverse of
  `OfdmMod`'s TX chain, plus a scalar gain correction mirroring
  `BpskDemod`'s `gain`/`set_gain()`. Explicitly scoped to known packet
  start, no CFO, AWGN/flat channel only — no acquisition or equalization
  yet. This is Release D / Phase 4 of the OFDM support roadmap.
- `OfdmDecider`: hard-decision `C32 → u8`, dispatching to the existing
  `BpskDecider`/`QpskDecider`/`QamDecider<BITS>` by `ConstellationOrder`,
  the receive-side mirror of `OfdmMod`'s internal mapper dispatch.
- `OfdmRxFrame` and `build_ofdm_rx_frame`: per-packet RX diagnostics.
  `evm_db` is populated now (hard-decided bits re-mapped to their ideal
  constellation points and compared against the soft symbols); `cfo_hz`,
  `timing_offset_samples`, and `channel_mse` stay `None` until acquisition
  (Release E/F) and equalization (Release G) land.
- Noiseless and AWGN-flat-channel OFDM roundtrip tests, 50-trial Monte
  Carlo BER-vs-SNR regression thresholds (`tests/roundtrip/ofdm_snr.rs`),
  a full BER-vs-SNR characterization sweep
  (`tests/performance/snr/ofdm.rs`), and throughput coverage extended to
  the full TX+RX+decide chain.

### Changed

- Loosened `modulate::ofdm::MapperKind` to `pub(crate)` so `OfdmRxFrame`'s
  EVM computation can reuse the TX-side per-order mapper dispatch instead
  of duplicating it.

## [0.0.36] - 2026-07-18

### Added

- `ConstellationOrder` (`src/modulate/ofdm.rs`): BPSK/QPSK/QAM-16/64/256
  selector with `bits_per_symbol()`, used by the OFDM transmitter's data
  carriers.
- `OfdmConfig` finalized with `fs`, `rf_hz`, `gain`, and `constellation`
  fields plus `bits_per_ofdm_symbol()`/`samples_per_ofdm_symbol()` helpers.
- `OfdmMod` (`Block<In=u8,Out=C32>`): OFDM transmitter pipeline — symbol
  mapper (reuses `BpskMapper`/`QpskMapper`/`QamMapper<BITS>` verbatim via a
  `match`, no `dyn` dispatch) → `GridMap` → `IfftBlock` →
  `CyclicPrefixInsert` → optional `Rotator` for RF upconversion (`rf_hz ==
  0.0` gives baseband passthrough, matching `BpskMod`'s convention). Includes
  a `modulate()` convenience wrapper that zero-pads a final partial symbol,
  mirroring `Ft8Mod::modulate()`. This is Release C / Phase 3 of the OFDM
  support roadmap.
- `util::wb_spectrum_snr_db`: wideband-occupancy SNR estimate that compares
  mean in-band power across an occupied-bandwidth window against the
  out-of-band median, for signals (like OFDM) that spread energy across many
  bins rather than concentrating it in one.
- Unit tests for `OfdmMod` (symbol length, partial-chunk no-op, multi-symbol
  batching, null-carrier silence via a test-local reference FFT, cyclic-prefix
  content, RF-upconversion spectral shift) and for `wb_spectrum_snr_db`.
  Throughput benchmarks for QPSK and QAM-64 OFDM modulation.

### Changed

- Renamed `util::spectrum_snr_db` to `util::nb_spectrum_snr_db` (narrowband
  single-tone SNR) to distinguish it from the new wideband variant.

## [0.0.35] - 2026-07-18

### Added

- `CarrierPlan` (`src/multicarrier/config.rs`): OFDM resource-grid
  description with signed carrier indices, a builder API
  (`with_data_carriers`, `with_pilot_carriers`), and `validate()` checking
  for out-of-range indices, data/pilot overlap, and an empty data set. Bin 0
  (DC) is implicitly null unless explicitly assigned a role.
- `CarrierGrid` (`src/multicarrier/grid.rs`): resolves a `CarrierPlan`'s
  signed carrier indices to natural rustfft bin order once at construction.
- `GridMap`/`GridExtract` (`src/multicarrier/grid.rs`): TX/RX `Block` pair
  that scatters dense data symbols into the full FFT-bin vector (nulls
  zeroed, pilots inserted) and gathers data-carrier bins back into a dense
  stream. This is Release B / Phase 2 of the OFDM support roadmap.
- `OfdmConfig` (`src/modulate/ofdm.rs`, new file): initial OFDM waveform
  configuration, currently holding just `carrier_plan`; further fields are
  added in later releases as they're needed.
- Unit tests for the new grid types: carrier-plan validation (overlap,
  out-of-range, empty data set, well-formed), negative-index bin wrapping,
  data-bin ordering, grid map/extract round trip, null-zeroing and
  pilot-writing behavior, and partial-chunk no-op handling.

## [0.0.34] - 2026-07-18

### Added

- `multicarrier` module (`src/multicarrier/`): waveform-agnostic FFT-domain
  primitives shared by OFDM and future multicarrier waveforms (SC-FDMA,
  OTFS). `FftBlock`/`IfftBlock` — allocation-free forward/inverse FFT
  blocks with a cached `rustfft` plan and scratch buffer, unity-gain
  forward transform and `1/N` scale folded into the inverse's output copy.
  `CyclicPrefixInsert`/`CyclicPrefixRemove` — cyclic-prefix insertion and
  removal for OFDM symbols. This is Release A / Phase 1 of the OFDM
  support roadmap.
- Unit tests for the new primitives (FFT/IFFT round trip, DC-bin impulse
  response, partial-chunk no-op behavior, multi-symbol chunking, cyclic
  prefix content/round-trip/zero-length) and throughput benchmarks at
  representative FFT sizes (64/1024/4096).

### Changed

- Move `.claude/plans/` out of the repository (tracked externally instead).

### Removed

- Drop the unused `realfft` dependency; the FFT pipeline is complex-valued
  end-to-end and has no use for a real-input-optimized transform.

## [0.0.33] - 2026-04-13

### Added

- `MorseEncoder` in `codec::morse`: ITU-R M.1677 Morse code encoder that
  converts ASCII text into a keying envelope (0.0/1.0 sample buffer) for
  driving `CwKeyedMod`. Supports configurable WPM, element-duration jitter,
  dash weighting, and inter-element spacing to simulate human operator
  characteristics. Builder API: `with_jitter`, `with_dash_weight`,
  `with_char_space`, `with_word_space`.
- Unit tests for Morse encoder: timing accuracy, SOS envelope length, dash
  weight, jitter bounds, word/char spacing, case insensitivity, unknown
  character handling, WPM range.

## [0.0.32] - 2026-04-13

### Changed

- Opt into Node.js 24 on GitHub Actions runners via
  `FORCE_JAVASCRIPT_ACTIONS_TO_NODE24=true` in both CI and publish
  workflows, ahead of the June 2026 forced cutover.

## [0.0.31] - 2026-04-13

### Added

- GitHub Actions CI workflow: fmt/check/clippy, unit+roundtrip tests
  on stable and beta Rust, pytest smoke test (Linux), and
  throughput/SNR regression job gated to main-branch pushes only.

### Fixed

- Cargo test alias filters corrected (were using nonexistent
  `tests::*` path prefix).
- Two `clippy::is_multiple_of` lints in psk31 demod and sync
  (new in Rust 1.94).
- Maturin `develop` venv handling in CI pytest job.

## [0.0.30] - 2026-04-10

### Changed

- Clean up clippy warnings across the crate (iterator-style loops,
  `copy_from_slice`/`fill`, `clamp`, targeted `#[allow]`s with rationale
  on public sync entry points).
- Drop the unused `simd` feature and the nightly `core::simd` code path
  in `FirLowpass`; the scalar implementation is the only one now.

## [0.0.29] - 2026-04-10

### Added

- `Ft8StreamDecoder` in `codec::ft8`: streaming FT8/FT4 frame decoder
  that accumulates 12 kHz IQ samples and runs the full
  sync → LDPC → CRC → `unpack77` pipeline when a frame's worth of
  samples is buffered, or on demand via `flush()`. Mode-specific
  constructors (`new_ft8` / `new_ft4`) and a shared
  `CallsignHashTable` across frames so hashed nonstandard callsigns
  resolve in later frames.
- `Ft8DecodeResult` value type carrying the decoded `Ft8Message`,
  carrier frequency, and SNR score.
- Unit and roundtrip tests for `Ft8StreamDecoder` covering buffer
  bookkeeping, full-frame feeds, chunked feeds, and both standard and
  free-text messages for FT8 and FT4.

## [0.0.28] - 2026-04-07

### Added

- `Psk31Stream`: streaming BPSK31/QPSK31 decode pipeline
  (`new_bpsk`/`new_qpsk`, `feed(iq)→String`, `flush()→String`) in
  `codec::psk31`
- Spectral analysis utilities in `util`: `power_spectrum`,
  `spectrum_snr_db`, `spectrum_bw_hz`, `best_sync`, `SIGNAL_THRESHOLD`,
  `PSK31_BW_HZ`
- Python bindings: `Psk31Stream`, `Bpsk31Decider`, `best_psk31_sync`
  with type stubs and 10 pytest tests
- Copyright banners (SPDX MIT OR Apache-2.0) on all `.rs` and `.md`
  files

### Changed

- Renamed `codec::psk31_conv` → `codec::psk31`
- Migrated all tests from `src/tests/` to top-level `tests/` directory
  (unit, roundtrip, performance entry points with shared `common/`)
- Consolidated per-modulation tests: `psk31_snr`/`psk31_stream` →
  `psk31`; `ft8_snr`/`message` → `ft8`/`ft4`; gray → `ft8`/`ft4`
- Made `hard_decide_dqpsk` public for external test access
- Updated docs: `api.md` (Utilities section, Psk31Stream), `source.md`
  (test layout), `python.md` (PSK31 examples, updated counts)

## [0.0.27] - 2026-04-05

### Fixed

- Varicode table: replaced 38 incorrect entries with canonical fldigi source
  (pskvaricode.cxx). Affected control chars 9-31, uppercase U-Y, Z, brackets,
  underscore, backtick, braces, tilde, DEL.
- Varicode decoder: shift register cap changed from `MAX_BITS` to `MAX_BITS + 1`
  to correctly decode 10-bit codewords (%, &, ?, @, Z, ^, backtick, {, }, ~).
- `VARICODE_MAX_BITS` reduced from 11 to 10 to match the canonical table.

### Changed

- QPSK31 demodulator reverted from coherent to differential detection for
  streaming decode compatibility. `Qpsk31Demod` now outputs differential
  products `[Re(d), Im(d)]`; `Qpsk31Decider` uses non-coherent `viterbi_decode`.
  SNR 100% threshold: -6 dB (was -7 dB coherent).
- `DQPSK_EXP` constant in `psk31_conv.rs` made public.
- Removed `QPSK31_PHASE_STEP_F32` from `modulate/psk31.rs` (use `DQPSK_EXP`).
- Updated docs: performance SNR table, API descriptions, features, source layout.
- Fixed all markdownlint issues across `**/*.md`.

### Added

- `StreamingViterbi`: fixed-lag sliding-window Viterbi decoder for incremental
  QPSK31 decode. Non-coherent DQPSK branch metric, traceback depth 32, exported
  as `orion_sdr::codec::StreamingViterbi`.
- Varicode tests: `varicode_table_no_collisions`, `varicode_no_internal_zero_pairs`,
  `varicode_stream_roundtrip_all_printable`, expanded `varicode_decode_roundtrip`
  to all 128 ASCII values.
- `streaming_viterbi_matches_batch` and `streaming_viterbi_text_roundtrip` unit tests.
- `roundtrip_bpsk31_all_ascii`: full 128-code-point modulate-demod-varicode roundtrip.

## [0.0.26] - 2026-03-26

### Changed

- `actions/download-artifact` bumped from v6 to v7 in the publish workflow,
  completing the Node.js 24 upgrade across all three artifact actions
  (`checkout@v6`, `upload-artifact@v6`, `download-artifact@v7`).
- Updated `release-prep` skill to infer the next patch version automatically
  when no version argument is provided.

## [0.0.25] - 2026-03-25

### Changed

- Upgraded GitHub Actions workflow to Node.js 24: `actions/checkout`,
  `actions/upload-artifact`, and `actions/download-artifact` bumped from v4
  to v6; removed `FORCE_JAVASCRIPT_ACTIONS_TO_NODE24` workaround env var.

## [0.0.24] - 2026-03-25

### Changed

- `Qpsk31Demod` now outputs phase-corrected absolute phasors `[Re(sym_c), Im(sym_c)]`
  instead of differential products `[Re(d), Im(d)]`; `Qpsk31Decider::flush()` calls
  the new `viterbi_decode_coherent` decoder, eliminating the ~3 dB noise-product penalty
  of differential detection.  QPSK31 50%/100% decode thresholds improve from −9/−6 dB
  to ≈−12.5/−7 dB SNR/2500 Hz; CI regression threshold tightened to −7 dB.
- AFC phase discriminant in `Qpsk31Demod` updated to operate on the absolute phasor
  rather than the differential product, consistent with coherent mode.
- Throughput: QPSK31 587 Msps (was 603); BPSK31 658 Msps (unchanged).

### Added

- `viterbi_decode_coherent(soft, phase_steps)` in `src/codec/psk31_conv.rs`: coherent
  Viterbi MLSE that tracks a hypothesised absolute phasor per trellis state; branch
  metric is `|sym_c − hyp|²` rather than a differential Euclidean distance.
- `QPSK31_PHASE_STEP_F32` constant in `src/modulate/psk31.rs` (pub(crate)) for use
  by the coherent Viterbi decoder.

## [0.0.23] - 2026-03-24

### Fixed

- QPSK31 Viterbi branch metric now uses DQPSK constellation phasors `(±1, 0)` /
  `(0, ±1)` as expected values instead of `(±1, ±1)`.  The DQPSK constellation
  places all energy on a single axis per symbol; the old metric made half the coded
  bits undecidable from current evidence alone, costing ~19 dB.  QPSK31 100% decode
  threshold improves from +13 dB to −6 dB SNR/2500 Hz; QPSK31 now outperforms
  BPSK31 by ~2 dB as theory predicts.

### Added

- Decision-directed AFC (first-order PLL, K=0.05, B_L ≈ 0.78 Hz) to `Bpsk31Demod`
  and `Qpsk31Demod`: tracks residual carrier phase drift at each symbol boundary,
  keeping the differential detector coherent across the frame.
- `hard_decide_dbpsk` and `hard_decide_dqpsk` helper functions in
  `src/demodulate/psk31.rs` (used by the AFC loop; also unit-tested).

### Changed

- Updated CI regression threshold: QPSK31 −6 dB SNR/2500 Hz (was +13 dB).
- Updated throughput table in `docs/performance.md` with current measurements
  (BPSK31 ~670 Msps, QPSK31 ~603 Msps; ~20% reduction from pre-AFC baseline
  due to `sin_cos()` per symbol dump).

## [0.0.22] - 2026-03-24

### Changed

- Replaced Hann-weighted integrate-and-dump (final quarter) in `Bpsk31Demod` and
  `Qpsk31Demod` with decision-feedback matched filtering over the full 256-sample
  symbol period: `corrected[n] = s[n] − prev_sym·(1−h[n])`, `sym = Σ h[n]·corrected[n] / Σ h[n]²`
- Improved sensitivity by ~1–2 dB: BPSK31 100% decode at −5 dB, QPSK31 at +13 dB SNR/2500 Hz
- Updated CI regression thresholds and `docs/performance.md` SNR table accordingly

### Added

- `HalfCosineMf` in `src/dsp/fir.rs`: complex-split FIR with unit-energy half-cosine taps,
  exported as `orion_sdr::dsp::HalfCosineMf`

## [0.0.21] - 2026-03-24

### Changed

- Replaced peak sampling in `Bpsk31Demod` and `Qpsk31Demod` with Hann-weighted
  integrate-and-dump over the final quarter of each symbol period (`n ∈
  [3·sps/4, sps)`), improving sensitivity by ~18 dB for both modes
- Updated CI regression thresholds: BPSK31 −4 dB, QPSK31 +14 dB SNR/2500 Hz
  (previously +14 dB and +32 dB); QPSK31 now correctly outperforms BPSK31
- Updated PSK31 SNR sweep ranges and `docs/performance.md` sensitivity table
  to reflect the improved demodulator

## [0.0.20] - 2026-03-24

### Added

- PSK31 SNR sensitivity characterisation: 50-trial Monte Carlo sweep for BPSK31
  and QPSK31, feature-gated (`src/tests/performance/snr/psk31.rs`)
- PSK31 SNR CI regression tests: fixed thresholds at 100% success level (BPSK31
  +14 dB, QPSK31 +32 dB, SNR/2500 Hz) (`src/tests/roundtrip/psk31_snr.rs`)
- SNR sensitivity table and explanatory notes added to `docs/performance.md`

## [0.0.19] - 2026-03-24

### Added

- PSK31 Python and Rust API reference in `docs/api.md`; fixed all compact table
  separator rows in the file to comply with MD060.

## [0.0.18] - 2026-03-23

### Added

- PSK31 full stack (BPSK31 + QPSK31) at 31.25 baud:
  - Varicode codec: IZ8BLY/G3PLX canonical table, `VaricodeEncoder`, `VaricodeDecoder`
    (`src/codec/varicode.rs`)
  - Convolutional codec: rate-1/2 K=5 (G0=25, G1=23) encoder and soft Viterbi decoder
    (`src/codec/psk31_conv.rs`)
  - BPSK31/QPSK31 modulators with Hann-windowed half-cosine crossfade pulse shaping
    (`src/modulate/psk31.rs`)
  - BPSK31/QPSK31 demodulators using peak-sampling differential detection
    (`src/demodulate/psk31.rs`)
  - Waterfall-based energy-persistence carrier sync (`psk31_sync`,
    `src/sync/psk31_sync.rs`)
  - PyO3 bindings: `VaricodeEncoder`, `VaricodeDecoder`, `Bpsk31Mod`, `Bpsk31Demod`,
    `Qpsk31Mod`, `Qpsk31Demod`, `psk31_sync` (`src/python/psk31.rs`)
  - 122 Rust tests (unit, roundtrip, throughput) and 20 new Python tests

## [0.0.17] - 2026-03-23

### Changed

- Synchronized release-prep and release skills from orion-sdr-view: added branch
  guard, PR-merge workflow, and removed co-author trailer from release commits
- Updated GitHub Actions workflow to use Node.js 24

## [0.0.16] - 2026-03-15

### Added

- FT8/FT4 full stack: CPFSK waveform mod/demod, CRC-14 + LDPC(174,91) + Gray codec,
  Costas-array frame sync with soft LLR extraction, and 77-bit message packing
  (standard Type 1/2, free text, telemetry, non-standard callsigns)
- Python bindings for the complete FT8/FT4 stack: `Ft8Mod`, `Ft8Demod`, `Ft8Codec`,
  `Ft4Mod`, `Ft4Demod`, `Ft4Codec`, `ft8_sync`, `ft4_sync`, `ft8_pack_standard`,
  `ft8_pack_free_text`, `ft8_pack_telemetry`, `ft8_unpack`
- SNR sensitivity tests: sweeping characterisation (feature-gated, always passes,
  prints curve) and fixed CI thresholds (FT8 −15 dB, FT4 −11 dB, SNR/2500 Hz)
- `performance/` test module replacing `throughput/`, with `throughput/` and `snr/`
  subdirectories; `cargo test-throughput` alias updated accordingly
- Acronym glossary in `docs/design.md`

### Changed

- `pyo3` and `numpy` are now optional dependencies activated only by the
  `extension-module` feature; `cargo test --release` no longer requires a Python
  interpreter at build time
- `docs/throughput.md` renamed to `docs/performance.md`; all references updated

## [0.0.15] - 2026-03-08

### Added

- BPSK, QPSK, QAM-16/64/256 modulators and demodulators (Rust + PyO3 bindings)
  - `BpskMapper`, `BpskMod`, `BpskDemod`, `BpskDecider` (1 bit/symbol)
  - `QpskMapper`, `QpskMod`, `QpskDemod`, `QpskDecider` (2 bits/symbol, Gray-coded)
  - `QamMapper<BITS>`, `QamMod`, `QamDecider<BITS>`, `QamDemod` (const-generic, 4/6/8 bits/symbol, Gray-coded, unit-energy normalized)
  - Python classes: `BpskMod`, `BpskDemod`, `QpskMod`, `QpskDemod`, `QamMod`, `QamDemod`
- `IqToIqChain<B>` graph scheduler for C32→C32 pipelines
- Throughput tests for all digital modes (BPSK ~253 Msps, QPSK ~317 Msps, QAM-16 ~209 Msps, QAM-64 ~92 Msps, QAM-256 ~73 Msps)
- Python unit and roundtrip tests for all digital modes (54 tests total)
- PEP 561 type stubs for 6 new digital classes
- Release-prep and release skills (`skills/release-prep/`, `skills/release/`)

### Changed

- Docs updated to cover digital modes: `docs/features.md`, `docs/design.md`,
  `docs/modulate.md`, `docs/demodulate.md`, `docs/throughput.md`, `docs/python.md`
- Fixed incorrect API examples in `docs/demodulate.md`
  (removed fictional `push_iq`, `push_audio`, `demod_mut`, `set_deemph_tau_us`,
  `set_limiter` calls)

## [0.0.14] - 2026-03-08

### Added

- GitHub Actions workflow (`publish.yml`) to build and publish wheels for
  Linux (x86-64, aarch64), macOS (x86-64, arm64), and Windows (x86-64)
  on every `v*` tag push, using `maturin-action` and OIDC trusted publishing

## [0.0.13] - 2026-03-08

### Added

- Restructured docs into `docs/` directory (source, design, conventions,
  throughput, features, commands, demodulate, modulate, python, api)
- `CHANGELOG.md` with git-accurate dates in Keep a Changelog format
- Mixed maturin Python package layout (`python/orion_sdr/`) with `__init__.py`,
  `py.typed`, and type stub (moved from root `orion_sdr.pyi`)
- `pyproject.toml` with maturin config and pytest settings
- pytest test suite: unit tests (shape, dtype, validation, setters, isolation)
  and roundtrip SNR tests for all 5 modes
- Cargo test aliases (`test-unit`, `test-roundtrip`, `test-throughput`)
- `.markdownlint.json` (MD024 `siblings_only`)
- `.venv` and `*.so`/`*.pyd` added to `.gitignore`

## [0.0.12] - 2026-03-01

### Added

- PyO3 bindings for mod/demod functionality
- Python type stubs (PEP 561)
- Claude configuration (`CLAUDE.md`)

### Changed

- Package name used by Python changed to `orion-sdr` / `orion_sdr`
- More optimizations on all mod/demod paths (fused `LpDcCascade`, phasor recurrence NCO, atan2 approximation, loop unrolling)
- Updated throughput results

## [0.0.11] - 2025-09-03

### Changed

- Optimized AM mod and demod path
- Updated throughput results

## [0.0.10] - 2025-09-01

### Changed

- Optimized SSB mod and demod path
- Added throughput results to README

## [0.0.9] - 2025-09-01

### Changed

- Subdivided modulator and demodulator code into per-mode modules

## [0.0.8] - 2025-09-01

### Added

- CW, SSB, FM, PM modulators

### Changed

- Reorganized source into module tree

## [0.0.7] - 2025-08-30

### Added

- Audio to IQ chain
- AM modulator with tests

### Changed

- Adopted `{Mode}{Approach}{Demod|Mod}` naming convention

## [0.0.6] - 2025-08-30

### Added

- FM and PM demodulators with tests

### Changed

- Updated changelog and README

## [0.0.5] - 2025-08-12

### Added

- Graph scheduler
- AGC, FIR decimator
- CW and AM demodulators with tests

## [0.0.4] - 2025-08-12

### Changed

- Updated roadmap

## [0.0.3] - 2025-08-12

### Changed

- Updated description

## [0.0.2] - 2025-08-12

### Added

- API implementation and basic test

## [0.0.1] - 2025-08-12

### Added

- Initial API, project structure, roadmap
