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
data and pilots never overlap. Use it *instead of* `with_data_carriers` but
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
