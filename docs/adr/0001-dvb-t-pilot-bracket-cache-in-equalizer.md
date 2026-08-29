# Cache DVB-T scattered-pilot bracket tables inside OfdmEqualizer, not in DVB-T-specific code

DVB-T's per-symbol channel estimation located each data carrier's bracketing
pilots via a binary search (`partition_point`) over a freshly re-sorted pilot
array, on every OFDM symbol — even though pilot carrier *positions* are fixed
for a given (transmission mode, phase = symbol index mod 4), and DVB-T is the
only real caller of `OfdmEqualizer`'s `PerSymbolPilotInterp` equalizer method
today. We extended `OfdmEqualizer::set_pilot_bins` with an explicit
`phase: usize` parameter and an internal per-phase bracket-table cache,
rather than moving pilot-position bookkeeping and channel estimation into
DVB-T-specific code (`ScatteredGridCycle`) and bypassing the equalizer
entirely.

## Considered Options

- **DVB-T-local bypass**: `ScatteredGridCycle` owns precomputed bracket
  tables and computes channel estimates directly, feeding only the finished
  estimate into `OfdmEqualizer`. Rejected: it would carve DVB-T out into a
  parallel channel-estimation implementation, blurring the documented
  "`OfdmEqualizer` (generic) → four-phase scattered-pilot orchestrators (DVB-T
  specialization)" boundary (`docs/dvb.md`), for no compensating benefit —
  DVB-T is the sole real consumer of this code path either way.

## Consequences

- `OfdmEqualizer::set_pilot_bins`'s signature is a breaking change. Confirmed
  safe: it isn't Python-bound, and `orion-sdr-view` (the only external Rust
  consumer) never calls below `DvbTFrameStreamDemod`/`OfdmFrameStreamDemod`.
- The cache is scoped to a single `decode_inner` call (one DVB-T frame), not
  the life of the receiver, since `ScatteredPilotExtractor`/`OfdmEqualizer`
  are already reconstructed fresh per frame today. Hoisting further was
  considered and rejected — it would only shave <1% more cost off an
  already-eliminated bottleneck, at the price of restructuring
  `DvbTFrameDemod`'s persistent state and its fresh-per-frame construction
  pattern.
