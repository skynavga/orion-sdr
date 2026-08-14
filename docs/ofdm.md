<!--
  Copyright (c) 2026 G & R Associates LLC
  SPDX-License-Identifier: MIT OR Apache-2.0
-->

# OFDM / COFDM Design

This document covers the multicarrier (OFDM) physical layer and the coded,
framed link (COFDM) built on top of it. Design conventions decided here also
apply to the planned SC-FDMA/OTFS waveforms that will share the
`multicarrier/` module. See [design.md](design.md) for crate-wide patterns and
[acronyms.md](acronyms.md) for the acronym glossary.

## Multicarrier / OFDM Pipeline

OFDM is the first of a planned family of multicarrier waveforms sharing the
`multicarrier/` module (waveform-agnostic FFT-domain primitives); DFT-s-OFDM
(SC-FDMA) and OTFS are expected to follow and reuse `CarrierPlan`/`CarrierGrid`
verbatim. The conventions below were decided during OFDM's implementation and
apply to those future waveforms too.

**FFT normalization.** Unity-gain forward FFT (`FftBlock`); `1/N` scale
folded into the inverse FFT's output copy (`IfftBlock`), not a separate
normalization pass. The forward direction matches `util::power_spectrum()`.
The inverse convention is the standard OFDM choice: it keeps a transmitted
symbol's amplitude independent of `n_fft` and makes `IFFT(1/N)` then
`FFT(unity)` round-trip exactly.

**Carrier indexing.** Natural rustfft bin order internally (bin 0 = DC,
negative frequencies wrap into `n_fft/2..n_fft`); a **signed** carrier-index
convention (e.g. `-26..=26`) at the `CarrierGrid` public API boundary, with
`bin = carrier_idx.rem_euclid(n_fft)` computed once per carrier at grid
construction — never per-sample, and no `fftshift` pass is ever run over a
full FFT buffer. DC (bin 0) is implicitly null unless explicitly included in
a carrier plan's data or pilot carriers.

**Numerology is caller-owned.** `CarrierPlan` bakes in no standard's
subcarrier spacing, CP length, or carrier count, and the crate ships no
numerology calculator — OFDM's target bands (VHF through EHF, including the
L/S/X/Ku/Ka microwave bands) span orders of magnitude in delay spread and
Doppler spread, so no single default would fit. Derive the three caller-owned
parameters from the link budget directly:

- **`cp_len`** (in samples) must exceed the channel's delay spread
  `τ_max · fs`, with margin — e.g. `cp_len ≈ 1.25 · τ_max · fs` — since a
  shorter CP causes inter-symbol interference `OfdmEqualizer` cannot correct
  (see the multipath scope note below).
- **Subcarrier spacing** `Δf = fs / n_fft` should stay well under the
  channel's coherence bandwidth `1 / τ_max` (so each subcarrier sees flat
  fading) and comfortably above the link's Doppler spread `f_D` (so a
  symbol's phase stays coherent across one OFDM symbol period); as a rule of
  thumb, keep `Δf` at least an order of magnitude above `f_D`.
- **`n_fft`** follows from the target `Δf` and `fs`: `n_fft = fs / Δf`,
  rounded to a size `rustfft` handles efficiently (a power of two, or at
  least highly composite).

Once `n_fft`/`cp_len`/carrier layout are chosen, they flow straight into
`CarrierPlan::new(n_fft, cp_len).with_data_carriers(...)`
(`multicarrier/config.rs`) and `OfdmConfig` (`modulate/ofdm.rs`); nothing
downstream re-derives or second-guesses them.

**Edge-carrier guard band (out-of-band emission).** Plain OFDM's out-of-band
spectrum decays only as `~1/f` (each subcarrier is a rectangular-windowed
sinusoid, so a `sinc`), and the loudest skirt generators are the carriers at the
band edges. Because the COFDM carrier layout is caller-owned (no standard
mandates edge pilots), the cheapest way to clean up the emission is to leave a
guard band of null carriers at each edge, pulling those generators inward.
`CarrierPlan::with_contiguous_data(edge_guard, include_dc)` builds the
data-carrier span for exactly this: it fills a contiguous span leaving
`edge_guard` null carriers at each edge (beyond the always-null Nyquist bin),
skips DC unless `include_dc`, and skips any index already in `pilot_carriers` so
data and pilots never overlap.

`include_dc = false` is the conventional choice and the right default — a
direct-conversion front end puts its LO leakage and DC offset on bin 0, so most
numerologies spend the carrier rather than fight the impairment. `true` is fully
supported: every stage that needs to know which bins are occupied reads
`occupied_bins()`, the preamble's training symbol included. It is worth taking
when the link does not sit at baseband DC — one modulated at `rf_hz = 0.0` and
upconverted wholesale by a `Rotator`, where bin 0 lands on the RF carrier rather
than on the receiver's LO. Occupying DC does **not** buy DC-offset immunity:
nothing in the crate estimates or removes a receiver-side DC offset yet.

Use it *instead of* `with_data_carriers` but
*alongside* `with_pilot_carriers` (call `with_pilot_carriers` first so the fill
can exclude the pilot indices). `edge_guard == 0` reproduces the full-fill span,
so it is a regression-safe default. `validate_edge_guard(g)` optionally asserts
no data/pilot carrier intrudes into the guard.

- **Sizing.** `edge_guard ≈ ceil(0.02 … 0.05 · n_fft)` per edge is the useful
  range: it trades a few percent of throughput (each nulled carrier removes
  `bits_per_ofdm_symbol` capacity) for the strongest `sinc` generators moving
  inward by that many bins, lowering the skirt onset. It narrows the occupied
  bandwidth below `fs` but leaves `fs`, `n_fft`, and the CP fraction untouched —
  the CP is a *time-domain* guard, orthogonal to this *frequency-domain* edge
  guard; do not conflate the two.
- **Limits.** Nulling shifts where the skirt starts; it does **not** change the
  `~1/f` sidelobe decay. For deep suppression (tens of dB across the decay),
  combine it with time-domain symbol windowing (raised-cosine / Tukey
  overlap-add).
- **DVB-T cannot use this.** DVB-T's extreme carriers (active indices `0` and
  `1704`) are *mandatory* continual pilots that conformant receivers rely on
  (`waveform/dvb_t.rs`); they cannot be nulled or moved inward. Symbol
  windowing is DVB-T's only skirt-suppression lever.

**Symbol windowing (out-of-band emission, both chains).** The deeper
skirt-suppression lever is a time-domain raised-cosine (Tukey) taper applied to
each symbol's edges, softening the symbol-boundary discontinuity behind OFDM's
`~1/f` skirt. It is an **`orion-sdr` original capability, not a standard
mechanism** — DVB-T and DVB-T2 both specify a rectangular OFDM symbol (EN 302 755
defines the per-carrier PSD as a plain `sinc²`, with no windowing/roll-off;
the `1/128 … 19/256` figures often seen are DVB-T2 *guard-interval* fractions,
not roll-offs). So a standards-conformant receiver expects a rectangular symbol;
our windowing must therefore stay strictly **RX-transparent** and is off by
default.

The taper is applied by `SymbolWindow` (`multicarrier/symbol_window.rs`), a
stateless per-symbol `Block` — same-length (symbols abut, no overlap-add), so it
needs no cross-call state and respects the block-boundary contract below. The
modulator windows every CP-bearing symbol; for the COFDM frame it **skips the raw
Schmidl & Cox preamble repeats** (which the receiver correlates sample-for-sample
and must not see altered), starting at the training symbol. DVB-T is
preamble-less, so every symbol is windowed.

RX transparency is achieved by a paired **FFT-window back-off**, not by avoiding
an RX change. With back-off `b` the receiver reads the window
`symbol[cp_len - b .. cp_len - b + n_fft]` (`SymbolFft::with_window_backoff`),
sliding it into the guard; a symmetric taper of `roll_off ≤ min(cp_len - b, b)`
samples per side then falls entirely on guard samples the FFT never integrates,
maximized at **`b = cp_len/2 ⇒ roll_off = cp_len/2`**. The back-off imposes a
per-bin phase ramp `exp(-j2πkb/n_fft)` (FFT shift theorem), so it is transparent
**only on the equalized path** (streaming demod / DVB-T scattered), where the
training/pilot channel estimate is measured at the same back-off and divides the
ramp out. A bare, unequalized demod must keep back-off `0`.

- **Config.** `CarrierPlan::with_window_roll_off` (samples) is the geometry home,
  inherited by every profile. `OfdmConfig` exposes `with_symbol_window(roll_off)`
  plus two fraction forms — `with_symbol_window_beta_guard(β)` = `round(β·cp_len)`
  (β∈[0,0.5], β=0.5 is the max-transparent taper) and
  `with_symbol_window_beta_tu(β)` = `round(β·n_fft)` (the DVB-family Tu-relative
  convention) — and `with_rx_window_backoff(b)`. DVB-T uses builders on the
  frame/super-frame mod/demod objects (`DvbTFrameMod::with_symbol_window`,
  `DvbTFrameDemod::with_rx_window_backoff`, and the super-frame equivalents).
- **Sizing and payoff.** The taper is bounded by `cp_len/2` (a fraction of the
  guard), so suppression is real but **modest** — a measured ~11 dB drop in the
  sidelobe skirt a few carriers beyond the band edge, *not* the immediate edge
  nor the far noise floor. (Further out it does better, since it changes the
  skirt's *decay rate*: ~32 dB measured in the deep out-of-band region. The
  payoff grows with distance from the band edge.) It also competes with the
  equalizer's delay-spread budget (both live in the guard). For COFDM it
  composes with the edge guard above (nulling moves the skirt inward; windowing
  lowers its rolloff); combined beats either alone. Default `roll_off = 0`
  leaves on-air output byte-identical.
- **How large a back-off can be, and what caps it.** The `b ≤ cp_len` clamp on
  `SymbolFft` is not the binding constraint on every path. What the receiver has
  to do is *undo* the ramp `exp(-j2πkb/n_fft)` from its channel estimate, so the
  ceiling depends on how that estimate is obtained:
  - `TrainingSymbolHold` (the COFDM default) measures every bin from the
    training symbol at the same back-off, so it absorbs any `b` the guard
    allows — verified up to `b = cp_len`.
  - `PerSymbolPilotInterp` (the DVB-T scattered path) only samples the channel
    every `pilot_spacing` carriers and interpolates between them. The ramp
    advances `θ = 2π·b·pilot_spacing/n_fft` per gap, and past `π` the
    interpolation aliases, giving `b < n_fft / (2·pilot_spacing)`
    (`SymbolFft::max_pilot_safe_backoff`) — **85 samples for DVB-T 2K**,
    whatever the guard interval.

  Note which way round that is: holding one full-resolution estimate is the
  *stronger* option for window back-off, and a pilot-interpolated equalizer is
  the one that costs budget.

  **The aliasing limit is not the usable limit.** Because the interpolation is
  *linear*, it approximates the ramp's arc by a chord and is wrong by
  `1 − cos(θ/2)` in between pilots — a graded error that bites long before `θ`
  reaches `π`. Measured on DVB-T 2K: `b = 32` (θ = 68°) is free, `b = 42`
  (θ = 90°, i.e. `n_fft/(4·pilot_spacing)`) costs ~1 dB, `b = 64` costs ~6 dB,
  and at the aliasing cap itself the link does not close at any SNR. So the rule
  of thumb for a pilot-interpolated equalizer is **`b ≤ n_fft/(4·pilot_spacing)`**,
  half the aliasing bound; see
  [performance.md](performance.md#the-rx-window-back-off-costs-sensitivity-well-before-it-aliases).
  A guard longer than `2 × 42 = 84` samples buys DVB-T no additional shaping room.

**Baseband spectral mask (out-of-band emission, both chains).** The third and
deepest lever is an optional TX low-pass across the **assembled** stream, run
after CP insertion and after any symbol taper: `TxLowpass`
(`multicarrier/tx_lowpass.rs`), a Kaiser-windowed linear-phase FIR over complex
samples (`dsp::FirLowpassIq`). Where windowing attacks the skirt *indirectly*
(smoothing the symbol seam, and so capped by the taper length the guard allows),
a mask attacks it directly in the frequency domain — so its attenuation stacks
on top of windowing's rather than sharing the same budget.

It is applied with `FirLowpassIq::filter_aligned`: same length, group delay
compensated. Stream length and symbol boundaries are unchanged, so the fixed-`sps`
strided receiver is untouched.

- **What the receiver sees.** With an FFT window at back-off `b`, each windowed
  sample combines symbol samples spanning `±d` around the window (`d` = the
  filter's group delay). While that reach stays inside the symbol, the cyclic
  prefix makes it an exact *circular* convolution, so the FFT sees each
  subcarrier scaled by a single complex `H[k]` — which the pilot/training
  estimate divides out like any other channel. Nothing about decoding changes,
  and unlike the back-off there is no TX/RX value to keep matched.
- **But it does need the back-off.** Centring the response makes half of it a
  pre-echo, so at `b = 0` there is no room for it at all: the budget is the same
  one windowing uses, now shared —
  `roll_off + group_delay ≤ min(cp_len − b, b)` (`TxLowpass::fits_guard`),
  maximized at `b = cp_len/2`. The RX FFT-window back-off is therefore the
  enabler for *both* TX shaping levers, not a windowing-only requirement.
  Overrunning the budget degrades gradually (a little inter-symbol leakage the
  equalizer cannot invert) rather than failing abruptly.
- **It needs somewhere to filter.** A mask can only attenuate bandwidth the
  signal does not occupy. A COFDM plan filling every bin out to Nyquist leaves
  no room for a transition, so pair it with the edge guard above: *the guard
  makes the room, the mask uses it.* DVB-T has the room built in (1705 of 2048
  bins are active).
- **Acquisition budget.** The mask is applied to the whole burst, preamble
  included — a real transmitter band-limits everything it emits, and filtering
  only part of one would put the spectral step back. A Schmidl & Cox preamble's
  repetition survives wherever the taps see only repeated samples, so the second
  sizing rule is `group_delay ≪ repeat_len`. Preamble-less waveforms (DVB-T,
  which acquires from the CP) are bound only by the guard budget.
- **Config.** `OfdmConfig::with_tx_lowpass(TxLowpass)`, or
  `with_tx_lowpass_null_band(num_taps, stopband_db)` which reads the occupied
  band edge off the carrier plan and places the transition against it.
  `num_taps` stays the caller's choice because *it* is what the guard budget
  constrains; `TxLowpass::taps_for_null_band` suggests a length and
  `transition_fits` says whether it is long enough. DVB-T uses
  `DvbTFrameMod::with_tx_lowpass` (and the super-frame equivalent). Absent by
  default ⇒ byte-identical output.
- **Payoff.** Measured on a 256-point COFDM link (`cp_len = 64`, 31-carrier edge
  guard, 65-tap 60 dB mask, `roll_off = 32`), mean out-of-band power in the
  mask's stop band: baseline −30 dB, windowing −62 dB, **mask −96 dB**, both
  −116 dB. On a conformant DVB-T frame the null band drops 66 dB with in-band
  power unchanged to within 0.1 dB. Note the two levers act in different places
  — inside the mask's own transition the taper is the better lever; past it the
  mask wins by tens of dB — which is why both ship.

**Preamble construction.** The S&C repeats are built in the **frequency
domain**: loading only bins that are multiples of `k = n_fft / repeat_len`
makes the inverse transform repeat with period `repeat_len` by construction, so
the repetition the receiver correlates on is exact rather than approximate.
Restricting those bins to the carrier plan's occupied span is what keeps the
preamble inside the same band as the payload.

The training symbol is band-limited to the same band, but from the plan itself
rather than from its edges: **it loads exactly `CarrierPlan::occupied_bins()` —
every data and pilot carrier, and nothing else.** A band *half-width* can only
describe a symmetric span, so it cannot say whether bin 0 is live, and the
training symbol used to null DC unconditionally while
`with_contiguous_data(_, true)` handed DC out as a data carrier. Two places
disagreeing about which bins are occupied is the failure mode; deriving both
from one accessor makes it unrepresentable. It also lets an asymmetric or sparse
plan train only the bins it actually uses.

The S&C repeats keep DC out whatever the plan says, and owe it no such
agreement: they are correlated for timing and CFO and never used to estimate a
channel, and a loaded bin 0 is a constant offset across the segment —
identically self-similar at every lag, so it broadens the timing plateau while
adding nothing to localize on.

This matters spectrally, not just tidily. A time-domain pseudo-random sequence
is white across the full Nyquist band, and a training symbol loading every bin
is a full-band pedestal — either one sits *outside* the occupied band at full
amplitude and swamps whatever the shaping levers achieve there. Measured on a
256-point COFDM link, an unshaped preamble accounted for **70 dB** of
out-of-band excess, reduced to 24.6 dB by band-limiting.

Band-limiting also amplitude-matches the training symbol for free: an OFDM
symbol's time-domain RMS is `sqrt(loaded bins) / n_fft`, so loading the
occupied bins rather than every bin brings it to a data symbol's level. Its
former excess was that difference, not a gain. Loading exactly the plan's bins
makes the match exact rather than close: a data symbol loads that same set at
unit *average* energy (the constellations are normalized and pilots are
conventionally unit-magnitude), so the two levels agree by construction, and
adding or removing DC moves both counts together. The S&C repeats are boosted 2x
above data level (`SC_PREAMBLE_BOOST`) so they remain the energy peak the
timing tie-break above assumes; transmitting a preamble hot is ordinary
practice, and 802.11 boosts its short training field for the same reason.

**`cfg.gain` reaches the preamble.** It is applied to the generated preamble
exactly as `OfdmMod` applies it to every data symbol. The preamble and payload
must share one amplitude scale or the frame is undecodable twice over: the S&C
metric normalises against received energy, so a quiet preamble collapses the
score; and `TrainingSymbolHold` estimates the channel from the training symbol,
so a scale mismatch there leaves the demapper's LLRs miscalibrated by that
factor.

**CFO acquisition capture range.** `ofdm_sync`'s Schmidl & Cox fractional
estimator is unambiguous only within `±fs / (2 · repeat_len)` — note this is
**not** always `±½` the subcarrier spacing; it equals that only when
`repeat_len = n_fft / 2`. Wider offsets alias and require the integer-CFO
stage (a dedicated training symbol, FFT'd and correlated against its known
frequency-domain pattern across candidate bin shifts) to resolve. Because a
purely periodic S&C preamble correlates against itself at any offset fully
inside its repeated structure — not only the true start — `ofdm_sync` breaks
timing ties using the correlated window's own energy, which peaks only where
every correlated sample is real preamble signal.

That energy ratio **ranks** candidates; it is deliberately not folded into the
reported `score`, which stays the raw phase-coherence metric the acceptance
threshold is compared against. Multiplying it in made acquisition depend on
whatever else was loud in the search range: a preamble at ordinary signal level
scored 0.54 rather than 1.00 merely because the payload matched it for energy,
and any louder transient elsewhere — a corrupted burst, an adjacent signal —
suppressed a valid candidate below threshold entirely, returning nothing at all
rather than an error.

**Channel estimation default.** For OFDM's predominantly line-of-sight,
terrestrial-microwave/satellite target bands, a channel estimate taken once
per packet from the training symbol and held constant
(`EqualizerMethod::TrainingSymbolHold`) is the default — not merely the
simplest option, but the physically correct one given static/slowly-varying
multipath is the dominant impairment. Per-symbol pilot-interpolated
re-estimation (`PerSymbolPilotInterp`) is the explicit opt-in for genuinely
time-varying channels (fast-moving aeronautical or LEO geometries).

**Block-boundary contract.** Every FFT-domain `Block` (`FftBlock`,
`IfftBlock`, `CyclicPrefixInsert`/`Remove`, `GridMap`/`GridExtract`,
`OfdmEqualizer`) operates on exactly one symbol's worth of input per
`process()` call and is a no-op on partial input, with no cross-call
buffering. `OfdmMod`/`OfdmDemod` drive their sub-blocks directly through
owned scratch buffers rather than the generic chain schedulers
(`IqToIqChain`/etc.), since those schedulers assume near-1:1 sample flow and
would silently truncate a rate-expanding stage like the IFFT+CP.

## COFDM Frame (MAC) Layer

The frame layer turns raw OFDM symbols into a coded, framed link: a
`FramePacket` (metadata + opaque byte payload) is serialized to IQ by
`OfdmFrameMod` and recovered by either the batch `OfdmFrameDemod` (known
start) or the streaming `OfdmFrameStreamDemod` (unknown start, CFO, and
multipath). The waveform-agnostic pieces — FEC codes, interleavers, the PN
scrambler, CRCs, and the `FramePacket`/`RxError` types — live in the `fec/`
module and are reusable by future waveforms; only `OfdmFrameMod`/
`OfdmFrameStreamDemod` and the MCS table are OFDM-specific.

**Concatenated FEC, configured on `OfdmConfig`.** Coding is a four-stage
concatenation, each stage independently selectable:

```text
payload → CRC → [scramble] → outer FEC → outer interleave →
           inner FEC → inner interleave → [scramble] → symbol map
```

reversed on receive. The stages are set by builder methods on `OfdmConfig`
(`with_outer_fec`, `with_inner_fec`, `with_outer_interleaver`,
`with_inner_interleaver`, `with_scrambler`/`with_scrambler_pos`,
`with_payload_crc`/`with_header_crc`, `with_header_format`), all defaulting to
"absent" so the bare symbol pipeline (`OfdmMod`/`OfdmDemod`) is unaffected.
`OfdmConfig::validate()` rejects inconsistent combinations.

Codes available (`fec/`):

- **Inner (soft-decision):** parameterized fixed-family **LDPC**
  (`LdpcCode::N512R12`/`N576R23`/`N512R34`, rates 1/2, 2/3, 3/4 — `orion-sdr`'s
  own constructive staircase codes, distinct from the FT8 LDPC in
  `codec/ldpc.rs`), or **punctured convolutional** (zero-tail rate-1/2 K=5
  mother code, puncture rates 1/2, 2/3, 3/4, 5/6, 7/8, LLR-domain soft
  Viterbi). The inner decoder consumes the demapper's LLRs directly.
- **Outer (algebraic, hard-decision):** binary **BCH(n,k,t)** or
  **Reed–Solomon** (DVB-T's RS(204,188) t=8 and shortened variants), both over
  GF(2^8).

**LDPC decode rule.** The inner LDPC decoder's check-node update is selectable
via `OfdmConfig::with_ldpc_decode_rule` (`DecodeRule::SumProduct` /
`MinSum` / `ScaledMinSum(α)`). The **on-air default is exact sum-product**
(`2·atanh(∏ tanh(msg/2))`); it is the reference and is what the header always
uses (the header is decoded first, before the MCS is known, so it takes no coding-
gain risk). `ScaledMinSum(α≈0.75)` is an opt-in that replaces the transcendental
product with `α·∏sign·min|msg|` — measured at ~2× decode throughput for a ≲0.3 dB
coding-gain cost (the speed/coding-gain trade is characterized in
[performance.md](performance.md), "LDPC decode rule: sum-product vs. min-sum"). Only the
payload honors the configured rule; making min-sum the *default* would require its
own SNR-threshold revalidation and is deliberately not done. This choice is
receiver-side only — the transmitter is unaffected.

**Interleaver domains.** The two deinterleavers operate in *different*
domains, which is why `BlockInterleaver::permute` is generic over `T`: the
**inner** deinterleaver runs in the **LLR (`f32`) domain**, before the
soft-input inner decoder (standard for LDPC/convolutional); the **outer**
deinterleaver runs in the **hard byte (`u8`) domain**, after inner decode and
before the algebraic outer decoder (BCH/RS are Berlekamp–Massey, not
LLR-consuming) — matching DVB-T's Viterbi → Forney byte-deinterleaver → RS
ordering. The general invariant (deinterleave before the code it protects, in
whatever metric that decoder consumes) is universal; only the domain varies.

**CRC placement.** The CRC is appended to the payload bytes **before** FEC
encoding (CRC-under-FEC), so on receive it validates the fully error-corrected
payload — the standard COFDM ordering, not a bare on-air trailer. Header and
payload have independent, config-selectable CRCs (`CrcKind::Crc16`/`Crc32`/
`None`).

**Scrambler.** An additive PN (LFSR) whitener with a configurable polynomial,
register width, and seed (`SeedMode::Fixed` or `PerFrameRandom` — the latter
signaled in the header). Its chain position is config-driven
(`ScramblerPos::BeforeOuterFec`, the DVB energy-dispersal position and the
default, or `AfterInnerFec`), since standards differ. `PerFrameRandom` requires
a header (validated).

**On-air frame layout** (transmit order):

```text
[ S&C preamble + training symbol ][ header symbols ][ payload symbols ]
```

The header (present unless `HeaderFormat::NoHeader`) is a fixed, MCS-independent
block — BPSK + rate-1/2 LDPC, a bespoke `orion-sdr` format — carrying
`mcs_index`, `payload_len`, `sequence_num`, `flags`, and a scrambler seed, plus
a header CRC. Because it is fixed-size and coded independently of the payload
MCS, the receiver always decodes it first to learn the payload's length and
coding. The payload is then coded per the MCS the header selected, looked up in
the shared `McsTable` (which maps each `mcs_index` to a
constellation + inner/outer-FEC triple, enabling per-frame adaptive coding).

**Streaming receiver.** `OfdmFrameStreamDemod` mirrors `Ft8StreamDecoder`'s
accumulate-and-drain shape (`feed`/`flush`/`clear`/`view_buf`). Each `feed`
appends samples, runs `ofdm_sync` to locate a preamble, corrects total CFO
(`fractional + integer·spacing`) with a `Rotator`, estimates the channel from
the training symbol (`OfdmEqualizer::TrainingSymbolHold`), decodes the frame,
and drains its samples — looping to yield multiple frames per call. A frame
whose payload has not fully arrived is *held* (an internal
incomplete-vs-failed distinction) rather than mis-reported, so a frame split
across `feed` calls completes on a later call. Successful decodes carry CFO and
timing diagnostics.

**Baseband only.** `OfdmFrameMod`, `OfdmFrameDemod` and `OfdmFrameStreamDemod`
assert `rf_hz == 0.0` at construction. `OfdmMod` honours `rf_hz` correctly — one
rotator, one continuous stream — but the frame layer cannot: `TxLowpass` is
centred on DC and deletes an already-upconverted signal, the preamble generator
does not apply it, and a fresh `OfdmMod` per block restarts the rotator at phase
0 (a step at every seam). The receiver never applies it either — `rf_hz` appears
nowhere in `demodulate` — so an IF-modulated frame could not be decoded even if
the transmit side were right. Modulate at baseband, shape, then upconvert the
whole burst once with a continuous `Rotator`.

**Residual carrier tracking.** The Schmidl & Cox estimate has variance, and the
default `TrainingSymbolHold` measures the channel once from the training symbol
and holds it for the whole frame — so a residual offset `e` integrates to
`2*pi*e*T` of constellation rotation by the end of a frame of duration `T`, with
nothing to notice it. A few Hz is already tens of degrees by the last symbol of a
50 ms frame, against QPSK's 45-degree decision boundary, and the failures look
exactly like an FEC cliff.

`remove_common_phase_error` removes it on the equalized path, in two passes over
the payload's data symbols: a decision-directed tracking loop (each symbol
de-rotated by the phase predicted from those before it, so the residual *at the
point of decision* stays under a degree and the decisions stay valid), then a
least-squares fit of accumulated phase against symbol index — a carrier offset is
a straight line, so fitting one pools every symbol's estimate and drops the
loop's start-up transient. Correcting per symbol rather than per frame moves the
budget from the frame duration to one symbol.

Measured in [performance.md](performance.md) (`snr::cofdm_stream_fer`):
error-free reception moves from 25 dB in-band SNR to 20 dB, with every point
below it several times better. Note that a *better initial estimate* is not an
alternative — S&C variance is set by the preamble's total correlated energy, so
correlating at lag `3L` instead of `L` measured a gain of 1.0x.

**Acceptance: the strongest end-to-end check wins.** `ChainOutcome::is_valid`
decides whether a decoded block can be trusted, and `inner_ok` is deliberately
not part of it: that flag reports whether the inner decoder's parity checks
converged — how hard it worked — not whether the answer is right. A CRC decides
on its own, since it covers the recovered payload end to end. Without one
(DVB-T's shape) the outer code decides. With neither, `inner_ok` is all there
is. `crc_present`/`outer_present` are recorded because `CrcKind::None` reports
`crc_ok = true` for "nothing checked", which a naive rule would read as a pass.

Requiring `inner_ok` discarded frames whose payload was verifiably correct —
measured, a byte-exact CRC-32-verified payload with **both** FEC stages
reporting non-convergence and 14.7% of channel bits wrong.

**Per-frame diagnostics.** `RxFrame::diagnostics` (`OfdmRxFrame`) carries
`cfo_hz` and `timing_offset_samples`, the `sync_score` the frame was acquired
at, `evm_db` measured on the payload's own hard decisions, and the payload's
`inner_fec_ok`/`outer_fec_ok` reported separately. Two are opt-in because they
cost per frame:

- `with_channel_estimate` — the per-bin channel `H[k] = received[k] / known[k]`,
  an `n_fft`-sized allocation. A power delay profile, and from it delay spread
  and echo-within-guard, is its inverse FFT. (`channel_mse` stays `None`: a
  scalar needs a reference a single-shot estimate does not have.)
- `with_error_rates` — true `channel_ber` and `inner_ber`, obtained by
  re-encoding the decoded frame. A frame that passed its CRC is ground truth,
  so the difference from what arrived at each stage is a real rate. **No prior
  knowledge of the payload is needed**, which is what makes it usable over the
  air rather than only against a known test vector; only frames that decode are
  measured, so a rising rate that stops reporting is itself the signal the link
  has given up. Measured cost: +0.4% of a frame decode.

**Block-size bookkeeping.** Because FEC and interleaving change bit counts and
fragment into fixed codeword blocks (LDPC N, BCH/RS codewords), the
transmitter and receiver agree on every intermediate length via a deterministic
`BlockPlan` computed from the payload byte count and coding config — needed to
trim zero-padding and to know how many OFDM symbols the payload occupies.

**Scope note — high-order QAM over multipath.** QAM-16 (and above) through a
real frequency-selective channel is bounded by `OfdmEqualizer`'s per-bin
*zero-forcing* division, which amplifies noise at spectral nulls; QAM-16's
tight amplitude margins tolerate a milder channel than QPSK before the residual
error exceeds them, **regardless of the inner FEC** (confirmed across both the
LDPC+BCH and RS+convolutional paths). QAM-16 decodes cleanly on flat/AWGN
channels; deep multipath at high-order QAM is a known limitation (an MMSE
equalizer and `|H|²`-weighted LLRs are candidate remedies). QPSK is the robust
choice for strong multipath.
