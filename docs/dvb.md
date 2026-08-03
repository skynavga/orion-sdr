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
crate-wide patterns, and [acronyms.md](acronyms.md) for the glossary.

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

The **frame assemblers are direction-split** to match the crate convention:
`modulate::dvb_t_frame` (TX) and `demodulate::dvb_t_frame` (RX), sharing
`DvbTFrameParams` and the FEC constants from `waveform::dvb_t` — the same way
`{modulate,demodulate}::ofdm_frame` split while sharing `Mcs`/`BlockPlan`.

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
estimator is generic (any CP-based OFDM waveform can call it); integer-CFO and
frame lock come from the scattered pilots + TPS downstream.

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
continual-only). External-IQ validation against published captures is an
opportunistic, non-CI-gating follow-up.
