<!--
  Copyright (c) 2026 G & R Associates LLC
  SPDX-License-Identifier: MIT OR Apache-2.0
-->

# DVB-T / NB-DVB-T Design

This document covers the DVB-T (and narrowband DVB-T, "NB-DVB-T", for amateur
DATV) waveform: how the standard's 2K-mode structure maps onto the crate's
generic `fec/`, `multicarrier/`, `modulate/`, and `demodulate/` primitives, and
the conventions specific to it. DVB-T is **not** a generic COFDM link — it is a
concrete standard (ETSI EN 300 744 V1.6.2) with fixed numerology, its own
signalling, and preamble-less acquisition. See [ofdm.md](ofdm.md) for the
generic OFDM/COFDM pipeline this builds on, [design.md](design.md) for
crate-wide patterns, and [terminology.md](terminology.md) for the glossary.

## Structure and where it lives

DVB-T is assembled from the generic primitives plus a **per-standard parameter
table + orchestrator** — it introduces no new generic abstraction. The
standard-specific pieces live under `src/waveform/`; the reusable stages stay in
their generic modules and are configured/called by the DVB-T code.

| Concern | Generic primitive (reused) | DVB-T specialization |
| --- | --- | --- |
| Numerology, pilots, mapping | `multicarrier::CarrierPlan`, `modulate::qam` | `waveform::dvb_t` (2K plan, Figure-9a map/LLR) |
| Payload FEC | `fec::{ReedSolomon, conv, interleaver}` | RS(204,188) + K=7 conv + Forney I=12 via `encode_chain`/`decode_chain` |
| Energy dispersal | `fec::scrambler` (PRBS shape) | `waveform::dvb_t_ts` (TS-packet framing) |
| Channel estimation | `OfdmEqualizer` (`set_pilot_bins`) | four-phase scattered-pilot orchestrators |
| Signalling | — (no generic equivalent) | `waveform::dvb_t_tps` (TPS + GF(2^7) BCH) |
| Acquisition | `sync::dvb_t_gi_sync` (generic CP estimator) | called with the 2K `(n_fft, cp_len)` |
| Frame assembly | `encode_chain`/`decode_chain` | `{modulate,demodulate}::dvb_t_frame` |
| Super-frame | — (orchestrates the above) | `{modulate,demodulate}::dvb_t_super_frame` |
| Streaming RX | `OfdmFrameStreamDemod` (feed/flush template) | `demodulate::dvb_t_stream` |

The **frame assemblers are direction-split** to match the crate convention:
`modulate::dvb_t_frame` (TX) and `demodulate::dvb_t_frame` (RX), sharing
`DvbTFrameParams` and the FEC constants from `waveform::dvb_t` — the same way
`{modulate,demodulate}::ofdm_frame` split while sharing `Mcs`/`BlockPlan`. Two
higher layers orchestrate the single frame: `{modulate,demodulate}::
dvb_t_super_frame` (the four-frame super-frame) and `demodulate::dvb_t_stream` (a
`feed`/`flush` receiver over a continuous frame run) — see **Frame transport**.

## 2K numerology (fixed)

`n_fft = 2048`, `Kmax = 1704` (1705 active carriers, indices `0..=1704`), of
which **exactly 1512 carry data** in every symbol. The rest are pilots and TPS:
45 continual pilots, scattered pilots, and 17 TPS carriers. Active carrier `a`
maps to the crate's signed DC-centered index via `signed = a − 852`. Guard
interval ∈ {1/32, 1/16, 1/8, 1/4} → `cp_len ∈ {64, 128, 256, 512}`.

**Narrowband scaling** changes only the sample rate: `occupied_BW = fs ·
1705/2048`, so `fs = BW · 2048/1705`. The 2K structure is unchanged. Amateur
modes: ~333 kHz (≈400 kS/s), ~1 MHz (≈1.20 MS/s), ~2 MHz (≈2.40 MS/s). 333 kHz
is below PlutoSDR's ~521 kS/s continuous-TX floor — valid for the library, not
for continuous Pluto TX.

## Pilots and the constant-1512 invariant

Continual pilots (EN 300 744 Table 7) sit at fixed positions on every symbol.
**Scattered pilots** move each symbol, cycling over four phases: for symbol
index `l`, carrier `k` is a scattered pilot iff `k mod 12 == 3·(l mod 4)`
(Kmin = 0 in 2K mode). Both are boosted (`±4/3`, `E[c·c*] = 16/9`) and take the
same `w_k` reference sequence (X¹¹+X²+1).

The standard fixes the scattered spacing so **the useful-carrier count is
constant (1512) across all four phases** — only *which physical bins* are data
vs. pilot rotates. This is the invariant that lets the frame layer's count-based
bookkeeping (`bits_per_ofdm_symbol`, `block_plan`) stay valid while the grid
rotates: `dvb_t_2k_plans(guard)` returns the four phase-grids, each asserted to
carry exactly 1512 data carriers (continual + phase-`p` scattered + 17 TPS
reserved). `ScatteredPilotMapper`/`ScatteredPilotExtractor` own the four grids +
a symbol-phase counter (frame-layer orchestrators, **not** `Block`s), and the
equalizer's `set_pilot_bins` installs each symbol's pilot set before `process`.

The **17 TPS carriers are reserved as non-data but are not channel-estimation
references.** The four grids record them as boosted `w_k` pilots so the 1512-data
invariant holds, but the modulator overwrites those bins with data-power DBPSK TPS
cells (§4.6), not the boosted value — so dividing the received cell by the grid's
known `w_k` would give a wrong channel ratio that the interpolator then smears onto
the data carriers straddling each TPS carrier. `ScatteredPilotExtractor::
current_pilot_bins()` therefore returns the continual + scattered pilots **only**;
the equalizer interpolates its estimate across the TPS bins from those real
references. (TPS is demodulated separately and differentially, off the raw
pre-equalization bins, so it needs no channel estimate of its own.)

## Payload FEC chain

Bit-exact to EN 300 744 §4.3: MPEG-TS → energy dispersal → RS(204,188) t=8 →
Forney convolutional interleaver (I=12, M=17) → K=7 punctured convolutional
(G0 = 0o171, G1 = 0o133) → constellation map. The K=7 code is `ConvCode::DvbK7`;
one 188-byte TS packet is exactly one RS information block (no stuffing).

**Energy dispersal** (`waveform::dvb_t_ts`): the PRBS 1+X¹⁴+X¹⁵ (init
`100101010000000`, MSB-first) re-initializes every 8 TS packets; the first
packet of each group has its sync byte inverted 0x47→0xB8 (the descrambler's
re-init signal); the other sync bytes are left unrandomized but the PRBS keeps
clocking over them. Known answer: the first randomized byte after the inverted
sync is `0x03`.

**Constellation** (Figure 9a): even bits (y0,y2,y4) → I axis, odd (y1,y3,y5) →
Q, each Gray-mapped per axis — distinct from the generic mapper's bit
assignment, so DVB-T has its own `dvb_t_map_symbol`/`dvb_t_soft_llr`. The
soft-LLR path gives the DVB-T frame real soft-decision coding gain.

**Frame filling** (§4.4 / §4.3.1): a single frame is 68 symbols, and §4.4
requires **every** symbol to carry data — a compliant signal never leaves data
carriers zeroed (§4.3.1: randomization stays active even with no program input).
A short payload is therefore stuffed with **MPEG-2 null packets** (PID `0x1FFF`,
`0xFF` payload — `ts_null_packet`/`ts_stuff_null_packets`) before energy
dispersal; the RX trims the recovered payload back to its original length, so the
stuffing is transparent. The standard's stronger "exact integer number of RS
packets, **no** stuffing" property (§4.7, Table 16 — e.g. 252 RS packets per 2K
QPSK-r1/2 super-frame) holds only over the full 4-frame **super-frame** (a single
frame is fractional for some rates), so exact-fit belongs to the super-frame path,
not the single-frame assembler here.

Stuffing cannot land on the last carrier exactly. The coded stream grows by a
fixed step per added packet (1632 bits / code rate) while the frame's capacity is
a multiple of the symbol size, so the two coincide at **no** integer packet count
in any mode — at 68-symbol QPSK r3/4, 83 packets code to 204 552 bits and 84 to
206 728, against a capacity of 205 632. `dvb_t_frame_fill` (in `waveform::dvb_t`,
with the other shared TX/RX frame constants) states the rule once for the
modulator, the demodulator and downstream payload sizing: take the **largest**
packet count whose coded stream still fits, and fill the carriers past it by
repeating the coded stream's head.

Both halves matter. Fitting rather than overshooting means nothing is encoded and
then dropped, so a receiver reconstructing what was transmitted — see [Receive
diagnostics and the probe](#receive-diagnostics-and-the-probe) — never asks its
decoder for bits that never went on air. Filling by **repeat** rather than with
zeros keeps the tail whitened: energy dispersal is applied at the TS layer ahead
of the FEC, so repeating coded bits stays dispersed, whereas a zero fill would put
a whole OFDM symbol on one constellation point (at QPSK r1/2 the remainder runs
past a symbol's 1512 data carriers). The repeat is never decoded — every block
plan on the receive side ends at the coded stream's length, which is where the
repeat begins.

## TPS signalling

`waveform::dvb_t_tps` (EN 300 744 §4.6). 17 TPS carriers each carry the **same**
DBPSK bit per symbol (differential *along the symbol axis*), spelling a 68-bit
word per 68-symbol frame (four frames = one super-frame). The word carries the
constellation, code rate, guard interval, transmission mode, frame number, and
cell id, protected by a shortened **BCH(67,53) t = 2 over GF(2^7)** (generator
`h(x) = x¹⁴+x⁹+x⁸+x⁶+x⁵+x⁴+x²+x+1 = 0x4377`, primitive poly `x⁷+x³+1`). The BCH
is a self-contained ~40-line code (cold path — once per frame), deliberately
**not** a parameterization of the hot GF(2^8) `Gf256`/`Bch`. DBPSK is
initialized per frame from `w_k` (absolute `Re = 2(½−w_k)`, normal power); the
decoder averages the 17 carriers, so it needs no absolute phase reference.

## Acquisition (preamble-less)

A conformant DVB-T frame carries **no Schmidl & Cox preamble**. `sync::
dvb_t_gi_sync` locates the symbol boundary and fractional CFO from the cyclic
prefix using the **van de Beek ML estimator**: it maximizes `Λ(d) = |γ(d)| − ρ·
Φ(d)`, where `γ = Σ r[d+k]·conj(r[d+n_fft+k])` (CP correlation), `Φ = ½Σ(|·|²)`
(window energy), and `ρ = SNR/(SNR+1)` weights the energy term. CFO comes from
`−∠γ`. `GiSyncConfig { rho, max_symbols }` tunes it: `max_symbols = 1` is the
strict single-symbol estimator; the default bounds coherent accumulation to a
few symbols (gr-dtv-style) to sharpen a batch lock while staying robust to
residual CFO (coherent summation over many symbols cancels under CFO). This
estimator is generic (any CP-based OFDM waveform can call it); frame/super-frame
lock comes from the TPS word (frame number + alternating sync).

**Integer CFO.** The guard-interval estimator resolves the CFO only within ±½ a
subcarrier; a real front end can be off by whole subcarriers, which slides the
entire spectrum by that integer `k`. `sync::dvb_t_integer_cfo` recovers `k` from
the **45 continual pilots** — they sit at fixed carrier positions on every symbol
(§4.5.4) and are boosted (16/9 power), so after fractional correction and symbol
alignment it FFTs a symbol and, for each trial shift `k ∈ [−max, max]`, sums the
energy landing at the continual-pilot bins shifted by `k`; the maximizing `k` is
the integer CFO. This is the DVB-T-native counterpart to the OFDM preamble path's
training-symbol integer-CFO recovery, which a preamble-less frame cannot use.
Correction is a **link-constant builder flag** on the receiver:
`DvbTFrameDemod::new(params).with_integer_cfo_correction(true)` (and the same on
the super-frame and streaming receivers). When enabled, the demod runs the
estimate right after its own guard-interval acquisition and rotates the buffer by
`−k·fs/n_fft` internally before decoding; off by default, since a clean link needs
no correction and the estimate/rotate is then skipped. (Always-on, the estimate
costs on the order of a few percent of the decode — a continual-pilot search per
frame.) The pilot peak is modest — 45 of 1705 carriers, boosted only ~1.78× — so
the winning shift sits ~1.7× above the all-shifts mean; the demod accumulates the
pilot energy over several symbols (sum `|X|²` per bin) to firm it up under noise.
The estimator `sync::dvb_t_integer_cfo` (returning `k` and a confidence ratio) is
also public for standalone use. See [demodulate.md](demodulate.md) for the flag.

## Spectral shaping (symbol windowing)

DVB-T's out-of-band skirt can be reduced with the crate's raised-cosine symbol
windowing — one of two OOB levers available to DVB-T (the other being the
baseband spectral mask below), since its extreme carriers are mandatory continual
pilots that cannot be nulled the way the COFDM edge-carrier guard nulls its own.
Windowing is **not** a DVB-T/DVB-T2 feature (both standards define a rectangular
symbol; see [ofdm.md](ofdm.md) for the EN 302 755 detail), so this is an
`orion-sdr`-internal, RX-transparent, off-by-default option.

Because a DVB-T frame is preamble-less, **every** symbol is a CP-bearing OFDM
symbol and every one is windowed (no S&C region to skip). The taper touches only
time-domain guard samples, so the continual/scattered/TPS pilots and the
subcarrier allocation are untouched. It is a matched TX/RX pair:
`DvbTFrameMod::with_symbol_window(roll_off)` applies the taper, and
`DvbTFrameDemod::with_rx_window_backoff(b)` slides the RX FFT window into the
guard so the taper falls outside it (`roll_off = b = cp_len/2` is the
transparent operating point; the scattered-pilot channel estimate, measured at
the same back-off, corrects the induced phase ramp). Both propagate through the
super-frame mod/demod to every constituent frame. Defaults (`0`/`0`) leave the
on-air frame byte-identical. See [ofdm.md](ofdm.md) for the shared `SymbolWindow`
/ `SymbolFft` geometry and the transparency argument.

## Spectral shaping (baseband mask)

The deeper lever is an optional TX low-pass across the assembled frame:
`DvbTFrameMod::with_tx_lowpass(TxLowpass)`, propagated by
`DvbTSuperFrameMod::with_tx_lowpass`. DVB-T is a good fit for it — only 1705 of
2048 bins are active, so the standard leaves a genuine null band for the
transition to live in, and the per-symbol scattered-pilot equalizer absorbs the
filter like any other channel. `DvbTFrameMod::tx_lowpass_for_2k(num_taps,
stopband_db)` places the cutoff against the fixed `±852` band edge.

Measured on a conformant frame (G1/8, 89-tap 60 dB mask): mean power in the null
band past the mask's transition drops **66 dB**, with in-band power unchanged to
within 0.1 dB and TPS, pilots, and payload all recovered. See
[ofdm.md](ofdm.md) for the transparency argument and the guard budget.

On the **super-frame** the mask is applied once over the concatenated four
frames, not per frame: the three interior seams are continuous on air, and
filtering each frame separately would leave the filter's edge transient at each.

### Choosing a guard interval for DATV

The two TX shaping levers share one budget,
`roll_off + group_delay ≤ min(cp_len − b, b)`, so a longer guard *looks* like it
buys a longer taper and a sharper mask. It does not, and the reason is worth
stating precisely, because there are two different ceilings on `b` and only one
of them is the one that matters.

The **aliasing** ceiling is `DVB_T_MAX_RX_WINDOW_BACKOFF` = `n_fft/(2·12)` = 85:
past it the scattered-pilot interpolation cannot represent the back-off's phase
ramp at all. The **practical** ceiling is much lower. The equalizer interpolates
*linearly* between pilots 12 carriers apart, and the ramp advances
`θ = 2π·b·12/2048` per gap — so it approximates an arc by a chord, with a
fractional magnitude error `1 − cos(θ/2)` in between. That error is graded, not a
cliff, and it is already crippling well below 85 (measured in
[performance.md](performance.md#the-rx-window-back-off-costs-sensitivity-well-before-it-aliases)):

| `b` | θ per gap | interp. error | Sensitivity cost |
| ---: | ---: | ---: | --- |
| 32 | 68° | 17% | free |
| 42 | 89° | 28% | ~1 dB |
| 64 | 135° | 62% | ~6 dB |
| 85 | 179° | 100% | link does not close |

So the usable slack per guard interval is:

| Guard | `cp_len` | Free `b` | Slack (free) | Slack (≤1 dB) |
| --- | ---: | ---: | ---: | ---: |
| G1/32 | 64 | 32 | 32 | 32 |
| G1/16 | 128 | 32 | 32 | 42 |
| G1/8 | 256 | 32 | 32 | 42 |
| G1/4 | 512 | 32 | 32 | 42 |

It saturates at **G1/16**, not G1/8, and at **42 samples**, not 85. Past G1/16
the extra guard buys delay-spread tolerance only — nothing for shaping. Two
consequences for a DATV transmitter:

- **G1/16 is the shaping sweet spot**; choose beyond it for the multipath
  environment, not for the spectrum.
- **The practical mask is ~45 taps** (group delay 22), plus a taper of up to ~20.
  An 89-tap mask needs 44 samples of group delay by itself, which only fits at
  `b ≥ 44` — already into the penalty region, so it costs more sensitivity than
  the extra sharpness is worth.

All four guards are TPS-signalled and auto-detected, so the guard choice itself
is transmitter-side with no receiver cost; the back-off is the part the receiver
has to be told about.

### Acquisition under shaping

Both TX levers perturb guard-interval acquisition, in the same direction. A
symbol taper attenuates each symbol's leading cyclic-prefix samples but not their
unwindowed copies in the interior, so the van de Beek correlation peaks **early**
by roughly a third of `roll_off`; a long mask smears the correlation similarly.

Where the receiver has lead-in that is free — the peak lands a few samples early
and the backed-off window absorbs it. Where the frame begins at sample 0 it is
not: the `[0, period)` search cannot express a negative phase, so the peak
surfaces at `period − δ`, which is the right phase but the *next* symbol.
`DvbTSuperFrameDemod` slices every constituent frame with zero lead-in, and
`DvbTFrameStreamDemod` re-acquires inside the slice it just acquired, so both hit
this squarely.

`sync::dvb_t_gi_sync` unwinds it, reporting the period boundary at or before the
peak when that boundary sits within `cp_len/2` of it *and* the boundary's own
single-symbol correlation reaches half the peak's. The second condition is what
keeps a genuine lead-in from collapsing to the origin. See
`GiSyncConfig::origin_score_ratio` for the derivation and the measured margins.

## Header formats

`HeaderFormat::DvbTps` sits alongside the retained `OrionSdr` (default) and
`NoHeader`. Only `OrionSdr` prepends a decodable header block
(`HeaderFormat::has_header_block()`); `DvbTps` carries no separate header — the
parameters ride on the TPS carriers. Because a cold receiver needs the MCS to
demap symbol 0 but TPS is only complete after 68 symbols, a `DvbTps` link takes
its MCS from configuration (like `NoHeader`) and TPS **verifies** it — matching
real receivers, which acquire on assumptions then confirm via TPS. The
conformant `DvbTps` frame is emitted/decoded by `{modulate,demodulate}::
dvb_t_frame`, not the generic `OfdmFrameMod` (which is preamble + OrionSdr
oriented).

## Frame transport

Three layers assemble the on-air stream, each over the one below:

- **Single frame** — `modulate::DvbTFrameMod` / `demodulate::DvbTFrameDemod`
  (above). One 68-symbol frame carrying a TS payload; a short payload is
  null-packet-stuffed to the largest count that fits and the remaining carriers
  repeat the coded stream's head, so every carrier is filled and nothing encoded
  is discarded (**Frame filling**, above). `DvbTFrameMod::modulate` returns
  the IQ plus `n_symbols` / `samples_per_symbol`; `DvbTFrameDemod::decode`
  GI-acquires it at an unknown offset and recovers the payload + TPS word (and,
  with the integer-CFO flag set, removes a whole-subcarrier offset first).
- **Super-frame** — `{modulate,demodulate}::dvb_t_super_frame` sequences **four**
  consecutive frames (numbers 0–3) with the standard's super-frame structure: the
  TPS sync word alternates each frame (frames 1 & 3 use `TPS_SYNC_WORD_13`, 2 & 4
  use `TPS_SYNC_WORD_24`, §4.6.2.2), the **16-bit** cell id is split across the
  frames (b15..b8 in 1 & 3, b7..b0 in 2 & 4, §4.6.2.10) and reassembled on RX, and
  the frame-number sequence `0,1,2,3` is verified. `DvbTSuperFrameParams` carries
  the full 16-bit cell id; the payload is split into four parts and the RX trims
  each back and concatenates them. Each frame codes its own part independently;
  the standard's byte-continuous stream (§4.7 Table 16 — an RS packet may straddle
  a frame boundary for rates whose per-frame count is fractional, e.g. QPSK r3/4
  = 94.5 packets/frame) is a continuous-FEC refinement left to a future streaming
  super-frame path.
- **Streaming RX** — `demodulate::dvb_t_stream::DvbTFrameStreamDemod` is a
  `feed`/`flush` receiver over a continuous run of frames, mirroring
  `OfdmFrameStreamDemod`: it accumulates IQ, GI-acquires the next frame at the
  front of the buffer, decodes it, drains its samples, and loops — holding a
  partially-arrived frame until a later `feed` completes it. The frame geometry
  (`n_symbols`, `payload_len`) is fixed at construction, exactly as the batch
  entry point takes it; `feed` is chunk-boundary-invariant (chunked input decodes
  identically to one-shot).

## Receive diagnostics and the probe

`DvbTRxFrame::diagnostics` (`DvbTRxDiagnostics`) is the measured quality ladder,
and `DvbTRxProbe` is the constellation / correction view behind it. Both mirror
what the generic COFDM path exposes (`OfdmRxFrame`, `OfdmRxProbe`), with three
DVB-T-specific departures argued below.

**Free versus gated.** Six rungs are read straight off work the decode already
does and are always populated: `cfo_hz`, `sync_score`, `timing_offset_samples`,
`integer_cfo_bins`, `outer_fec_ok`, `rs_corrected_bytes`. Three cost work the
decode would not otherwise do and are behind `with_error_rates(true)` on
`DvbTFrameDemod`, `DvbTSuperFrameDemod` and `DvbTFrameStreamDemod`: `evm_db`,
`channel_ber` (CBER) and `inner_ber` (IBER). The probe is gated separately, by
choice of method — `feed_probed` / `flush_probed` — so the unprobed `feed` gains
no runtime branch.

`None` never means zero. A rung that goes absent exactly when the link fails
must stay distinguishable from one reporting a perfect result, or a dead link
renders as a flawless one. `integer_cfo_bins` makes the same distinction within
a single rung: `Some(0)` is "the estimator ran and the link is on frequency",
`None` is "no estimate exists".

**Three deliberate absences.**

- **No `inner_fec_ok`.** DVB-T's inner code is always `ConvCode::DvbK7`, and
  `ChainOutcome::inner_ok` is documented as always `true` for the convolutional
  arm — its soft Viterbi has no per-block convergence flag. Exposing it would put
  a permanently-green lock on a display that no link condition could move. The
  meaningful post-inner measurement is `inner_ber`.
- **No `crc_ok`.** DVB-T carries no CRC, so the chain's `crc_ok` reports `true`
  because nothing was checked. Reed–Solomon is the integrity check here, which is
  why `outer_fec_ok` — not `crc_ok` — is the frame-good signal.
- **No sequence number on a probe frame, and no codeword geometry.** DVB-T has no
  frame header to carry a monotonic counter; `DvbTProbeFrame` carries the whole
  `TpsWord` instead, whose `frame_number` wraps 0..3 every super-frame.
  Synthesising a counter from it would invite gap arithmetic that cannot work.
  The convolutional inner code likewise terminates once per frame and has no
  codeword boundaries to draw, so the generic path's `codeword_bits` pair is
  omitted rather than carried as permanent zeroes.

**A measurement decodes the whole frame; a plain decode does not.** `DvbTFrameMod`
stuffs null TS packets up to the frame's data carriers, so a receiver told
`payload_len = 184` decodes a *prefix* — 39 180 of 205 632 coded bits for a
68-symbol QPSK frame. That is fine for recovering the payload, but a
BER or a correction map has to reproduce what the transmitter actually sent, and
with a Forney(12,17) outer interleaver no prefix of a re-encode does: each output
byte draws from twelve branches at different depths, so the first coded bits
already depend on TS bytes from codewords the prefix never recovers. Re-encoding
the prefix puts zeros where the transmitter had data, and measures a CBER around
0.25 on a *noiseless* link. So when any gated rung is requested the demod decodes
the full frame — roughly 5× the FEC work, which is exactly why it sits behind the
same gate.

The whole-frame plan it builds comes from `dvb_t_frame_fill`, the same rule the
modulator stuffed by, so it describes exactly the bits that were transmitted:
never more (the coded stream fits within the carriers by construction) and never
fewer (the carriers past it are a repeat, outside the plan). The receiver's
re-encode is therefore bit-exact and both BERs read **exactly** zero on a
noiseless link, in every mode and at every payload size — swept by
`error_rates_decode_every_mode` and `error_rates_decode_across_payload_sizes`. An
approximate zero would not be a rounding artifact but a misaligned truth
reference.

One consequence is worth knowing: a prefix decode carries a **structural RS
correction floor**. The Forney deinterleaver's tail draws on codewords the prefix
does not cover, and Reed–Solomon quietly repairs the shortfall — one byte of the
eight RS(204,188) can correct, spent before the channel has done anything. A
whole-frame decode has no shortfall and reports zero. Pinned by
`a_prefix_decode_carries_a_structural_rs_correction_floor`.

**Measured cost** (`throughput::dvbt`, G1/32 QPSK R1/2). Diagnostics off runs at
15.85 Msps on a sparse frame: the unmeasured demap loop is byte-for-byte the one
that shipped before any of this existed, the EVM branch being hoisted to once per
OFDM symbol rather than once per data carrier.

The gated path always decodes the whole frame, so **its cost does not depend on
`payload_len` — it is ~7.1–7.5 Msps either way — and the overhead percentage is
really a statement about the baseline**:

| Payload | Fill | `with_error_rates` | `feed_probed` |
| ---: | ---: | ---: | ---: |
| 184 B | 2% | +112% | +105% |
| 9 724 B | ~100% | **+7.0%** | **+4.9%** |

At a realistic DATV fill that is the same neighbourhood as the generic COFDM
path (+3.3% / +4.3%), which never shows the sparse case because it sizes the
frame to the payload rather than to a fixed TPS block. A caller sending one TS
packet per 68-symbol frame is paying for 98% stuffing in the plain decode too —
it is simply invisible there, because the plain decode skips it.
[performance.md](performance.md) carries the same measurement with the absolute
throughputs alongside.

The modulator runs at ~88 Msps. `dvb_t_map_symbol` is allocation-free — it folds
the axis de-interleave into the label pack rather than materializing two `Vec`s
per constellation point, which at ~103 k points per frame would otherwise
dominate the mapping loop.

**What `cfo_hz` reports.** The total offset — acquisition's fractional estimate
plus any whole-subcarrier offset removed ahead of it — matching `OfdmRxFrame`.
Reporting the post-correction residual would describe the receiver's internal
state rather than the link. Note the demod *estimates* the fractional CFO but
does not *apply* it, so a link more than roughly a fifth of a subcarrier off
needs `with_integer_cfo_correction`.

## Conformance and testing

**Spec-exact + self-verified**: every stage is bit-exact to EN 300 744 where a
known answer exists (energy-dispersal PRBS, QAM points, BCH parity, TPS tables,
continual-pilot/TPS carrier indices — all checked against the standard), and the
whole chain is decoded by orion-sdr's own conformant RX in-repo (no DVB hardware
in CI). The capstone `roundtrip_dvb_t_2k_tps_end_to_end` modulates a TS payload
into a preamble-less frame, GI-acquires it at an unknown offset, and recovers
both the payload and every TPS-signalled parameter; `dvb_t_tps_frame_survives_
awgn` repeats it through AWGN. Scattered-pilot channel tracking is proven
load-bearing by a multipath pair (decodes with scattered pilots, fails on
continual-only). The super-frame and streaming layers have their own roundtrips:
`roundtrip_dvb_t_super_frame_end_to_end` (payload + 16-bit cell id + frame
sequence) and the `dvb_t_stream_*` suite (chunk-boundary-invariance, a continuous
multi-frame run, partial-frame holding). External-IQ validation against published
captures is an opportunistic, non-CI-gating follow-up.

The diagnostics and probe have their own suite, `roundtrip::dvb_t_probe`, written
so a rung has to *move* rather than merely be present: a rung that is always
`Some(0.0)` passes an `is_some()` check and tells an operator nothing. `cfo_hz`
is checked against an
injected fractional offset (and against an injected integer one, to prove it
reports the total rather than the residual); `rs_corrected_bytes` against
injected noise; the correction map is required to reproduce `channel_ber`
exactly, since it is that scalar's per-bit expansion and not a second
measurement of it. Probing is proven not to change what decodes, to partition its
flat buffers into non-overlapping per-frame spans, and to be
chunk-boundary-invariant. Noise in this suite is scaled to the frame's own mean
power (~4.4e-4), not given absolutely — an absolute "0.1" is 225× the signal, and
every frame then dies in TPS decode long before the payload is interestingly
degraded.
