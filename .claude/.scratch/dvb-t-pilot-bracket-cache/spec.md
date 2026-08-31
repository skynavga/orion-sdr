# DVB-T scattered-pilot bracket cache

## Status (manually added post facto)

Implemented in PR #55, released v0.0.67.

## Problem

`OfdmEqualizer::interpolate_at` (`src/demodulate/ofdm.rs:514`) locates each
data carrier's bracketing pilots via `partition_point` binary search, called
once per data carrier (~1512×/symbol for DVB-T 2K) inside
`interpolate_from_pilots`. Pilot carrier positions are fixed for a given
(transmission mode, phase = symbol index mod 4) — EN 300 744's scattered-pilot
pattern is `k mod 12 == 3*(phase mod 4)` — so rediscovering the bracket
relationship independently for every symbol is wasted work.

`set_pilot_bins` also currently reallocates two `Vec`s
(`current_pilot_bins().to_vec()`, `data_bins().to_vec()`) and re-sorts the
pilot array by bin on every call (`dvb_t_frame.rs:511-513`,
`ofdm_frame.rs:325-327`), even though only 4 distinct sorted layouts exist per
transmission mode.

## Decision

Replace the per-carrier binary search with a precomputed bracket table:

1. Extend `OfdmEqualizer::set_pilot_bins` with an explicit `phase: usize`
   (0-3) parameter.
2. On a cache miss for that phase, build the full data-carrier →
   `(left_bin, right_bin, weight)` mapping: sort the pilot/data bins (neither
   arrives pre-sorted, so this dominates the one-time build cost at
   `O((data + pilots)·log(data + pilots))`), then bracket every data carrier
   against them in one linear merge pass, `O(data + pilots)` — cache the
   result.
3. On a cache hit, skip straight to O(1) lookup per data carrier — only pilot
   *values* (received/known ratios) are refreshed each symbol; positions
   never change for a given phase.
4. Cache scope is per `OfdmEqualizer` instance, i.e., per `decode_inner` call
   (one DVB-T frame) — `ScatteredPilotExtractor`/`OfdmEqualizer` are
   reconstructed fresh every frame today; this is *not* changed as part of
   this work (see Non-goals).
5. Delete `interpolate_at` and its binary search entirely once the new path
   is verified — no fallback, no dual implementation.

See `.claude/docs/adr/0001-dvb-t-pilot-bracket-cache-in-equalizer.md` for why this
lives in `OfdmEqualizer` rather than as a DVB-T-local bypass.

## API impact

`OfdmEqualizer::set_pilot_bins`'s signature changes (breaking). Confirmed
safe:

- Not bound to Python (`src/python/ofdm.rs` has no `set_pilot_bins`
  pymethod).
- `orion-sdr-view` (Cargo.toml `orion-sdr = "0.0.64"`) never calls below
  `DvbTFrameStreamDemod`/`OfdmFrameStreamDemod` — grep confirmed zero
  references to `OfdmEqualizer`/`set_pilot_bins`/`demodulate::ofdm` in its
  source.
- In-repo callers to update: `src/demodulate/dvb_t_frame.rs:513`,
  `src/demodulate/ofdm_frame.rs:327`, `tests/unit/ofdm.rs:766-830` (generic
  non-DVB-T `PerSymbolPilotInterp` tests — pass a stable phase, e.g. `0`),
  `tests/roundtrip/dvb_t.rs:717`.

## Verification

- Add a test in `tests/` with an independent brute-force reference
  implementation of the old bracket-and-lerp logic; assert the new cached
  path matches it (bit-exact or epsilon-bounded) across all 4 phases with
  randomized pilot/channel values, before deleting `interpolate_at`. Per
  project convention, no `#[cfg(test)]` code in `src/`.
- Reuse the existing `throughput_dvb_t_conformant_frame` test
  (`tests/performance/throughput/dvbt.rs:57`) as before/after evidence — no
  new isolated micro-benchmark.
- Update `docs/performance.md`'s RX-cost description (currently states
  "O(data·log pilots) per symbol" binary search) to describe the new
  complexity/behavior, in place, present-tense, no change narrative.

## Non-goals

- **Cross-frame caching.** `ScatteredPilotExtractor`/`OfdmEqualizer` are
  rebuilt fresh on every `decode_inner` call (i.e., every DVB-T frame), so
  the bracket-table cache is rebuilt once per frame rather than once per
  receiver lifetime. Hoisting the cache onto `DvbTFrameDemod` to survive
  across frames was considered and rejected: the marginal gain is <1% of the
  total win (rebuilding four ~1512-step tables per frame is negligible next
  to the ~770K comparisons/frame the binary search was doing), and it would
  require restructuring `DvbTFrameDemod`'s persistent state and its current
  "everything fresh per frame" construction pattern.
- **Dynamic transmission-mode/guard-interval reconfiguration.** Confirmed:
  neither exists in this codebase today. Transmission mode is a hardcoded
  compile-time constant (`DVB_T_N_FFT = 2048`, 2K only); guard interval is
  fixed at `DvbTFrameDemod` construction for its whole life. TPS-decoded
  guard/mode values are read only for reporting, never fed back to
  reconfigure a running instance. This work does not need to (and does not)
  anticipate live TPS-driven reconfiguration.
- **Cold-acquisition phase verification/recovery.** Investigating this work
  surfaced that `ScatteredGridCycle.phase` is a free-running counter reset to
  0 by fiat at the start of every frame decode, with no verification against
  TPS or blind pilot-correlation — a real gap for cold OTA acquisition, but
  orthogonal to this change and not urgent (no capture hardware until
  ~2027-02-14, no OTA work planned before FY27 Q2/Q3). Tracked in the
  `project_ota_capture_hardware` agent memory, not as a separate
  `.claude/.scratch/` issue.

## Domain notes

- MCS (DVB-T's TPS-signalled constellation + HP/LP code rate) and waveform
  configuration (guard interval + transmission mode) are TPS-signalled
  independently, both with one-superframe advance notice, but only waveform
  configuration's transmission-mode field affects pilot carrier positions —
  MCS changes have zero interaction with this cache. See `.claude/CONTEXT.md`.
