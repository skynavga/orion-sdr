<!--
  Copyright (c) 2026 G & R Associates LLC
  SPDX-License-Identifier: MIT OR Apache-2.0
-->

# Throughput

Measurements taken on Apple M2 Pro, release build (`opt-level=3`, `lto=fat`,
`codegen-units=1`), no SIMD.  Results are ordered by throughput (descending)
within each table.

## v0.0.44 Results

### Analog modes (65536 samples × 30 passes)

| Mode         | Msps |
|--------------|-----:|
| CW           |  137 |
| AM-PowerSqrt |  107 |
| PM           |  125 |
| SSB-USB      |  138 |
| FM           |  103 |
| AM-AbsApprox |   79 |

### Digital modes (full pipeline: mapper → mod → demod → decider, 65536 sym × 30 passes)

| Mode    | Msps |
|---------|-----:|
| QPSK    |  328 |
| BPSK    |  249 |
| QAM-16  |  261 |
| QAM-64  |  162 |
| QAM-256 |  141 |

BPSK and QPSK are faster than the analog modes because the pipeline is purely
multiply-heavy with no transcendentals.  QAM decider throughput decreases with
order because the threshold scan is O(M) per axis (M = levels/axis = 2^(BITS/2)):
QAM-256 (M=16) does 4× more comparisons per symbol than QAM-16 (M=4).

### PSK31 (full roundtrip, 4096 sym × 256 sps × 20 passes)

| Mode   | Msps |
|--------|-----:|
| BPSK31 |  678 |
| QPSK31 |  598 |

Both modes measure the full roundtrip: `modulate_bits` → `process` (demod) → `process`
(decider / Viterbi flush).  The AFC loop adds a `sin_cos()` call per symbol dump.
QPSK31 uses the non-coherent Viterbi (`viterbi_decode`), which allocates a
`prev_state_table`; the gap between BPSK31 and QPSK31 is due to the Viterbi
trellis computation (16 states × 2 branches per symbol).

### PSK31 SNR sensitivity (50 trials/point, release build)

SNR is relative to noise in a 2500 Hz reference bandwidth (same convention as FT8/FT4).
Both modes modulate a 7-character text string with preamble=64 and postamble=32 bits.
Pipeline: `psk31_sync` (carrier detection) → `Bpsk31Demod` or `Qpsk31Demod` (whole signal)
→ Varicode decode → text search.

| SNR (dB/2500 Hz) | BPSK31 success% | QPSK31 success% |
| ---: | ---: | ---: |
| −16 | — | 0% |
| −14 | — | 0% |
| −13 | — | 0% |
| −12 | 2% | 0% |
| −11 | — | 4% |
| −10 | 20% | 28% |
| **−9** | 38% | 60% |
| **−8** | 60% | 84% |
| −7 | 88% | 98% |
| **−6** | 98% | **100%** |
| **−5** | **100%** | 100% |
| −4 | 100% | 100% |
| −2 | 100% | 100% |
| 0 | 100% | 100% |

100% decode points: BPSK31 = −5 dB, QPSK31 = −6 dB (used as CI regression thresholds).

Both modes use differential detection.  QPSK31 outperforms BPSK31 by ~1 dB at the
100% point due to the convolutional code's coding gain.

Both demodulators use decision-feedback matched filtering over the full sps=256 symbol
period combined with a symbol-rate decision-directed PLL (AFC).  For each sample n in
the symbol, the known previous-phasor contribution is subtracted before accumulation
(`corrected[n] = s[n] − prev_sym·(1−h[n])`), yielding a clean estimate of the current
phasor.  A first-order AFC loop (K=0.05, B_L ≈ 0.78 Hz) tracks residual carrier phase
drift at each symbol boundary.  BPSK31 outputs `Re(d)` (one soft value per symbol);
QPSK31 outputs `[Re(d), Im(d)]` differential products for the Viterbi decoder.

The QPSK31 Viterbi decoder is also available in streaming form (`StreamingViterbi`)
for incremental decode with fixed-lag traceback (depth=32 symbols).

The remaining gap to the published G3PLX reference (BPSK31 −10 dB, QPSK31 ~−11 dB)
is due to differences in test methodology (single-frame vs. multi-frame averaging).

### FT8/FT4 (frame-at-a-time, 20 passes; "Msps" = frame samples / wall time)

| Stage | FT8 Msps | FT4 Msps |
| --- | ---: | ---: |
| mod only | 274 | 265 |
| demod only | 35 | 80 |
| codec encode only | 327 957 | 148 874 |
| codec decode only | 430 848 | 52 495 |
| full roundtrip (encode → mod → demod → decode) | 31 | 64 |

Frame sizes: FT8 = 151 680 samples (79 sym × 1920); FT4 = 60 480 samples
(105 sym × 576).  Codec encode/decode times are sub-millisecond and optimized
away at constant input in release mode; they are not the throughput bottleneck.

Demod dominates: 8 Goertzel correlators × 79 symbols for FT8 vs. 4 × 105 for
FT4.  FT4 demod has higher Msps because the frame is 2.5× shorter, more than
compensating for the extra Costas blocks.

### FT8/FT4 SNR sensitivity (50 trials/point, release build, single AWGN seed per trial)

SNR is relative to noise in a 2500 Hz reference bandwidth, matching the WSJT-X convention.

| SNR (dB/2500 Hz) | FT8 success% | FT4 success% |
| ---: | ---: | ---: |
| −26 | 0% | 0% |
| −22 | 0% | 0% |
| −20 | 8% | 0% |
| −19 | 36% | 0% |
| −18 | 70% | 0% |
| −17 | 92% | 0% |
| −16 | 98% | 2% |
| −15 | **100%** | 6% |
| −14 | 100% | 36% |
| −13 | 100% | 86% |
| −12 | 100% | 94% |
| −11 | 100% | **100%** |
| −10 | 100% | 100% |

100% decode points: FT8 = −15 dB, FT4 = −11 dB (used as CI regression thresholds).

These are ~6 dB above the WSJT-X published limits (−21 dB FT8, −17 dB FT4). The gap
is expected: WSJT-X averages over many frames in a 15-second window with Doppler
tracking and iterative decoding. This decoder processes a single frame with no
iterative refinement.

### Multicarrier primitives (representative FFT sizes)

| Primitive | n_fft | Msps |
| --- | ---: | ---: |
| FFT/IFFT round trip | 64 | 324 |
| FFT/IFFT round trip | 1024 | 256 |
| FFT/IFFT round trip | 4096 | 210 |
| CyclicPrefix insert/remove round trip | 1024 | 5066 |

`FftBlock`/`IfftBlock` cache their `rustfft` plan and scratch buffer (allocated
once in `new()`); cyclic-prefix insertion/removal is a pure copy, hence its
much higher throughput. FFT round-trip throughput falls off gradually with
`n_fft` as expected for an O(N log N) transform amortized per sample.

### OFDM (n_fft=1024, cp_len=128, full pipeline unless noted)

| Stage | QPSK Msps | QAM-64 Msps |
| --- | ---: | ---: |
| mod only | 321 | 258 |
| full roundtrip (mod → demod → decide) | 164 | 102 |

`OfdmMod`'s TX-only throughput sits close to the multicarrier FFT primitive's
own ceiling, since the mapper/grid/CP stages are comparatively cheap. The full
roundtrip adds `OfdmDemod`'s inverse chain (CP-remove → FFT → grid-extract →
decide); QAM-64's decider does more per-axis comparisons than QPSK's, same
pattern as the single-carrier QAM throughput above.

### OFDM BER vs. noise scale (n_fft=64, cp_len=8, 50 trials/point, flat channel)

`noise_scale` is AWGN power relative to the time-domain signal's own power
(OFDM has no single reference bandwidth the way FT8/PSK31 do, since occupied
bandwidth is caller-chosen via `CarrierPlan`); `equiv_snr_dB = -10·log10(noise_scale)`
is printed alongside for a rough per-sample SNR reference.

| noise_scale | equiv. SNR (dB) | QPSK BER | QAM-16 BER | QAM-64 BER |
| ---: | ---: | ---: | ---: | ---: |
| 0.001 | 30.0 | 0.00000 | 0.00000 | 0.00000 |
| 0.005 | 23.0 | 0.00000 | 0.00000 | 0.00030 |
| 0.01 | 20.0 | 0.00000 | 0.00000 | 0.00603 |
| 0.02 | 17.0 | 0.00000 | 0.00030 | 0.03088 |
| 0.05 | 13.0 | 0.00000 | 0.01417 | 0.09158 |
| 0.1 | 10.0 | 0.00102 | 0.05254 | 0.15011 |
| 0.2 | 7.0 | 0.01465 | 0.11202 | 0.21496 |
| 0.5 | 3.0 | 0.08179 | 0.20461 | 0.30923 |
| 1.0 | 0.0 | 0.16259 | 0.28046 | 0.36817 |
| 2.0 | −3.0 | 0.24454 | 0.34919 | 0.41163 |

Uncoded BER curves as expected: higher-order constellations (QAM-64) degrade
faster than lower-order ones (QPSK) at the same noise scale, since their
decision regions are closer together for the same average symbol energy. These
are the raw `OfdmSoftDemod`/`OfdmDecider` LLR numbers; the COFDM frame layer's
concatenated FEC (below) improves on them substantially at moderate SNR.

### OFDM BER under multipath (n_fft=64, cp_len=8, 2-tap synthetic FIR channel, `TrainingSymbolHold` equalizer)

| noise_scale | equiv. SNR (dB) | QPSK BER | QAM-16 BER |
| ---: | ---: | ---: | ---: |
| 0.001 | 30.0 | 0.00000 | 0.00077 |
| 0.005 | 23.0 | 0.00067 | 0.03425 |
| 0.01 | 20.0 | 0.00805 | 0.08134 |
| 0.02 | 17.0 | 0.03464 | 0.14809 |
| 0.05 | 13.0 | 0.12148 | 0.24696 |
| 0.1 | 10.0 | 0.21604 | 0.32077 |
| 0.2 | 7.0 | 0.31528 | 0.38725 |
| 0.5 | 3.0 | 0.40805 | 0.44331 |
| 1.0 | 0.0 | 0.44635 | 0.46945 |
| 2.0 | −3.0 | 0.47264 | 0.48340 |

The `TrainingSymbolHold` equalizer (one channel estimate per packet, held
constant — the default for this feature's line-of-sight VHF–EHF target
bands) noticeably raises BER relative to the flat-channel table above at
matched noise scales, since a 2-tap channel spreads each subcarrier's SNR
unevenly across the band. This is expected: a per-bin equalizer corrects
channel *gain/phase*, not the SNR penalty of nulls the channel introduces at
some subcarriers.

### OFDM packet-sync acquisition probability (50 trials/point)

`ofdm_sync`'s Schmidl & Cox-style repeated-segment preamble, timing +
fractional CFO only, vs. the same preamble extended with a training symbol
for wide-range integer-CFO recovery (applied CFO = 5.3 subcarrier spacings,
well beyond the fractional-only ±½-spacing capture range).

| noise_scale | Fractional-only lock% | Fractional+integer lock% (5.3-spacing CFO) |
| ---: | ---: | ---: |
| 0.01 | 100.0% | 100.0% |
| 0.05 | 100.0% | 100.0% |
| 0.1 | 94.0% | 98.0% |
| 0.2 | 88.0% | 74.0% |
| 0.5 | 8.0% | 18.0% |
| 1.0 | 0.0% | 0.0% |
| 2.0 | 0.0% | 0.0% |
| 5.0 | 0.0% | 0.0% |

Both curves degrade sharply past `noise_scale ≈ 0.2`, driven by the timing
metric's tie-break (correlated-window energy) losing discrimination as AWGN
dominates the preamble's own energy — timing lock is the limiting factor at
low SNR, not the CFO estimators layered on top of it.

### COFDM frame throughput (n_fft=64, cp_len=8, QPSK payload, 96-byte payload, 200 passes)

Full frame pipeline: `OfdmFrameMod::modulate_frame` (CRC → concatenated FEC
encode → interleave → map → preamble/header) and `demodulate_frame` (soft-demap
→ deinterleave → concatenated FEC decode → CRC). Two concatenations are shown.
"Msps" is total frame samples / wall time.

| Config | mod Msps | demod Msps | frame samples |
| --- | ---: | ---: | ---: |
| LDPC(n512r12) + BCH(t=8) | 0.5 | 0.2 | 2584 |
| Convolutional r1/2 + RS(60,52) | 0.7 | 0.4 | 1936 |

Throughput is two-to-three orders of magnitude below the bare OFDM PHY (164
Msps roundtrip above) because per-frame FEC dominates: the LDPC sum-product
decoder runs up to 50 belief-propagation iterations per codeword, and the
convolutional path runs a full-block soft Viterbi. This is decode-side cost —
`modulate` is consistently ~2× faster than `demodulate`. Larger payloads
amortize the fixed preamble/header overhead but not the per-codeword decode.

### COFDM frame-error-rate vs. noise scale (n_fft=64, cp_len=8, QPSK payload, 100 trials/point, flat channel)

Frame-error-rate (whole-frame CRC pass/fail) for the two concatenations, same
`noise_scale` convention as the uncoded OFDM tables above. Compare against the
uncoded QPSK column of "OFDM BER vs. noise scale": the FEC drives *frame* errors
to zero at noise scales where the uncoded *bit*-error rate is already nonzero.

| noise_scale | equiv. SNR (dB) | LDPC+BCH FER | Conv+RS FER |
| ---: | ---: | ---: | ---: |
| 0.02 | 17.0 | 0.000 | 0.000 |
| 0.05 | 13.0 | 0.000 | 0.000 |
| 0.1 | 10.0 | 0.000 | 0.000 |
| 0.2 | 7.0 | 0.000 | 0.000 |
| 0.3 | 5.2 | 0.010 | 0.000 |
| 0.5 | 3.0 | 0.300 | 0.010 |

Both concatenations hold FER = 0 through `noise_scale = 0.2` (equiv. SNR 7 dB),
where uncoded QPSK already shows BER ≈ 0.015 — the concatenated FEC's coding
gain. The convolutional + Reed–Solomon pairing is the more robust of the two at
the cliff edge (`noise_scale ≥ 0.3`): its soft Viterbi plus RS symbol-error
correction degrades more gracefully than the LDPC + BCH pairing here. (Frame
errors are all-or-nothing per the payload CRC, so FER rises steeply once the
inner code can no longer clear the channel — characteristic of a coded packet
link, unlike the smooth uncoded BER curves.)

## COFDM FEC block throughput (work-in-progress — branch `feature/ofdm-optimize-cofdm`)

> **In-progress optimization tracking, not a released result.** The numbers below
> are the per-block FEC/interleave/scrambler benchmarks added by that branch
> (`throughput::fec`), recorded so the optimization commits (`R2`…) can be tracked
> against an `R1` baseline. Measured on Apple M2 Pro, release build,
> `--test-threads=1`, 200 passes/point. **"Msps" here = information bits processed
> per second** (before coding on Tx, after decoding on Rx) — comparable across code
> rates, and NOT the sample-domain figure used elsewhere in this doc. This section
> is folded into the released tables above (and this heading removed) at the end of
> the branch; until then it reflects the current tip, not v0.0.44.

**Current status:** R8 landed — the LDPC decoder's per-edge message arrays (`msg`,
`ext`) moved from a jagged `Vec<Vec<f32>>` to a flat CSR layout (one contiguous
buffer + a `check_start` offset table), removing a pointer-chase per check and giving
the check-/variable-node loops contiguous memory. Bit-exact (pure layout change; the
full suite passes unchanged). Measured on the isolated `LDPC-Decode` fixtures, on top
of R5: **~1.2–1.8× further speedup** for sum-product (`N512R12` ~13→~24 Msps), and
**~1.4–2.2×** for the min-sum rules — CSR helps min-sum more, since with the
transcendentals gone (R8a) memory access is a larger share of its runtime. Prior
steps: R8a/R8b (min-sum scaffold, measurement, and the opt-in
`with_ldpc_decode_rule`), R5 (tanh caching), R7 (interleaver hoist), R2/R3/R4 (GF/code
caching, edge-index precompute). Two candidates were **dropped**: R7's byte-wise
scrambler (flat) and R6's Horner syndromes (a regression). Deltas vs. the R1 baseline
are in the "Since baseline" columns.

### Per-block, single direction (§1.1)

| Block | Variant | Tx Msps | Rx Msps | Since baseline |
| --- | --- | ---: | ---: | --- |
| LDPC | N512R12 (rate 1/2) | 457 | ~24 | **~2.0×** (R5 ~1.1× + R8 CSR ~1.8×) |
| LDPC | N576R23 (rate 2/3) | 577 | ~25 | **~2.0×** (R5 ~1.3× + R8 CSR ~1.3×) |
| LDPC | N512R34 (rate 3/4) | 640 | ~11 | **~3.0×** (R5 ~2.3× + R8 CSR ~1.2×) |
| Convolutional | R1/2 | 610 | 27.1 | baseline |
| Convolutional | R2/3 | 347 | 27.3 | baseline |
| Convolutional | R3/4 | 384 | 26.1 | baseline |
| Convolutional | R5/6 | 345 | 28.4 | baseline |
| Convolutional | R7/8 | 328 | 28.8 | baseline |
| BCH | t=8 | 99.6 | 27.1 | baseline |
| Reed–Solomon | RS(204,188) | 799 | 165 | baseline |
| Reed–Solomon | RS(60,52) | 1126 | 140 | baseline |
| Interleaver | u8, 32×32 (kernel) | 5088 | 6083 | baseline |
| Interleaver | f32, 32×32 (kernel) | 4668 | 7042 | baseline |
| Interleave-Bits | chain driver, 8+ blocks | ~1700 | — | R7 ~1.05–1.08× |
| Scrambler | width 7 | 196 | (self-inverse) | R7 flat (dropped) |
| Scrambler | width 15 | 198 | (self-inverse) | R7 flat (dropped) |
| Scrambler | width 32 | 202 | (self-inverse) | R7 flat (dropped) |

Rx (decode) dominates cost across every code, as expected: LDPC sum-product decode
is 1–2 orders of magnitude slower than its direct staircase encode, and it is the
single largest term in the COFDM decode path. `LDPC-Decode N512R34` remains the
slowest block (~8.8 Msps after R5, up from ~3.7 at R1) — the rate-3/4 code's denser
checks and tighter margin run the most BP work per info bit. The interleaver kernels
are already near memory-bandwidth-bound (thousands of Msps); R7's target was per-chunk
*allocation churn* in the chain driver (`interleave_bits`), not the kernel — measured
by the separate `Interleave-Bits chain` row.

**R4 finding (edge-index precompute).** Removing the decoder's per-iteration
`position()` scans left `LDPC-Decode` flat within run-to-run noise: the check-node
update's tanh/atanh product (deg·(deg−1) `fast_tanh` calls per check per iteration)
dominates, so the O(edges·degree) scan it eliminated was never the bottleneck on
these low-degree codes. Kept because it is bit-exact, removes real redundant work,
and becomes load-bearing under R8a's min-sum rule (no transcendentals — the scans
would then dominate).

**R5 result (tanh caching).** Caching `tanh(msg/2)` per edge collapses the check-node
transcendental count from deg·(deg−1) to deg per check per iteration. This *is* the
bottleneck R4 identified, so the win is real and scales with check degree and
iteration count: `N512R34` (rate 3/4, degree-5 checks, iterates the most under the
error-injected fixture) gains ~2.3× (~3.8→~8.8 Msps); `N576R23` ~1.3× and `N512R12`
~1.1×. Warm-run steady-state numbers; the first run of each fixture is a cold-cache
outlier discarded. Bit-exact — the cached values equal the previous inline
`fast_tanh` calls and the leave-one-out products multiply in the same order.

**R8 result (flat CSR edge storage).** Replacing the decoder's jagged
`Vec<Vec<f32>>` message arrays (`msg`, `ext`) with a single contiguous buffer plus a
`check_start` offset table removes one pointer-chase per check and lets the
check-/variable-node loops walk contiguous memory. On top of R5 (same isolated
fixtures, same-session warm A/B against the R5 tip): sum-product gains **~1.2–1.8×**
(`N512R12` ~13→~24 Msps, `N576R23` ~19→~25, `N512R34` ~9.4→~11), and the min-sum rules
gain **~1.4–2.2×** — CSR helps min-sum more because, with R8a's transcendentals gone,
memory layout is a larger fraction of its cost. This is the change the plan flagged as
conditional ("only if R4+R5 leave a measurable gap"); the gap was real. Bit-exact — a
pure storage-layout change, identical values in identical order. (Contrary to an
earlier hunch that R5 had already captured the decode win and R8 could be skipped —
the layout effect turned out to be substantial and independent of the arithmetic
savings. Investigated on request; kept on the evidence.)

**R6/R7 findings (dropped candidates).** Two bit-exact changes were implemented,
measured against the immediately-preceding commit, and reverted because the perf
tests are the gate: **R6** rewrote the BCH/RS syndrome loops with Horner's method
(fewer operations) but *regressed* decode ~2–4× — the original `if c != 0 { … pow }`
form both skips zero positions and keeps its per-position terms independent, which a
superscalar CPU pipelines, whereas Horner serializes everything through its
`acc = acc·x + c` dependency chain; op-count fell but dependency-chain length rose,
and here the latter dominates. **R7's scrambler** rewrite (assemble the PN byte, then
one XOR) came out flat — the `count_ones` LFSR advance dominates and the compiler
already handled the original. Only R7's interleaver allocation-hoist survived, as a
small measured win.

### LDPC decode rule: sum-product vs. min-sum (§1.4, R8a investigation)

R8a adds a selectable check-node `DecodeRule` purely to *measure* the min-sum trade;
`SumProduct` stays the on-air default. **Speed axis** (`throughput::fec`
`throughput_ldpc_decode_rules`, error-injected fixture, warm steady-state Msps):

| Code | sum-product | min-sum | scaled-min-sum(0.75) |
| --- | ---: | ---: | ---: |
| N512R12 | ~8.4 | ~14.9 (**1.8×**) | ~12.6 (1.5×) |
| N576R23 | ~12.3 | ~21.3 (1.7×) | ~21.7 (1.8×) |
| N512R34 | ~7.4 | ~17.8 (2.4×) | ~19.0 (**2.6×**) |

Min-sum drops the transcendentals entirely (`∏sign · min|msg|`), giving ~1.7–2.6×
decode throughput; the densest/most-iterating code (`N512R34`) gains most. (This is
also where R4's edge-index precompute finally earns its keep — with the tanh/atanh
gone, the old `position()` scans would dominate.)

**Coding-gain axis** (`snr::ldpc_decode_rule`, BPSK-over-AWGN on the codeword, 200
trials/point, mean post-decode BER; all rules see the identical noise realization):

| Es/N0 (dB) | N512R12 SP | MS | SMS(0.75) | N512R34 SP | MS | SMS(0.75) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| −2.5 | 0.070 | 0.086 | 0.074 | 0.143 | 0.160 | 0.152 |
| −2.0 | 0.033 | 0.041 | 0.040 | 0.128 | 0.145 | 0.137 |
| −1.5 | 0.015 | 0.019 | 0.020 | 0.114 | 0.130 | 0.123 |
| −1.0 | 0.0036 | 0.0043 | 0.0061 | 0.100 | 0.114 | 0.108 |
| 0.0 | 0.0002 | 0.0001 | 0.0002 | 0.060 | 0.073 | 0.065 |

Reading the waterfall's horizontal (dB) shift at matched BER: plain **min-sum costs
~0.3–0.5 dB** of coding gain vs. sum-product, and **scaled-min-sum(0.75) recovers most
of it — within ~0.15–0.3 dB** across the slope, tracking sum-product closely
(especially on the denser `N512R34`). So scaled-min-sum buys ~2× decode speed for a
sub-0.3-dB penalty — the "essentially free" regime. Whether to expose it as an opt-in
mode is R8b's call; the on-air default stays sum-product regardless (flipping it would
need its own CI SNR-threshold update).

### Paired forward→inverse roundtrip (§1.2, correctness asserted each pass)

| Roundtrip | Variant | Msps | Since baseline |
| --- | --- | ---: | --- |
| LDPC enc→dec | N512R12 | 207 | baseline |
| LDPC enc→dec | N576R23 | 253 | baseline |
| LDPC enc→dec | N512R34 | 282 | baseline |
| Conv enc→dec | R1/2 | 25.1 | baseline |
| Conv enc→dec | R2/3…R7/8 | 24.3–25.4 | baseline |
| BCH enc→dec (t errors) | t=8 | 21.9 | baseline |
| RS enc→dec (t errors) | RS(204,188) | 141 | baseline |
| RS enc→dec (t errors) | RS(60,52) | 128 | baseline |
| Interleaver round trip | u8/f32 32×32 | 1533 (f32) | baseline |
| Scrambler round trip | width 32 | (see per-block) | baseline |

The LDPC roundtrip runs faster than the standalone `LDPC-Decode` block because its
decode fixture feeds clean codewords (BP converges in one syndrome check and exits),
whereas the standalone decode fixture injects soft errors to force realistic BP
iteration — the two measure different, deliberately chosen decoder workloads.

### Per-frame code-object construction cost (§1.3, Finding A)

Constructing a code — `Ldpc::new` above all — is the dominant per-frame FEC cost.
Before R2/R3 the concatenated-FEC chain rebuilt its `Ldpc`/`Bch`/`ReedSolomon` from
scratch on **every frame**; R3's per-instance `CodecCache` now builds each once per
link. These microbenchmarks measure construction in isolation.

| Construction | R1 baseline | R2 | Since baseline | Notes |
| --- | ---: | ---: | --- | --- |
| `Ldpc::new(N512R12)` | ~2.7 ms | ~2.7 ms | unchanged | sparse-H build + HashSet 4-cycle guard (R4 may trim; R3 caches it) |
| `Bch::new(t=8)` | ~3.4 µs (0.29 Mc/s) | ~3.1 µs (0.32 Mc/s) | ~1.1× | shared `Gf256`; still dominated by conjugacy-class LCMs |
| `ReedSolomon::dvb()` | ~0.7 µs (1.45 Mc/s) | ~0.41 µs (2.4–2.6 Mc/s) | **~1.7×** | shared `Gf256` |
| LDPC ×64 frames, rebuilt each frame | 1× (baseline) | — | — | pre-R3 per-frame behavior |
| LDPC ×64 frames, built once & reused | **~5000×** faster | — | — | the amortized behavior R3's `CodecCache` delivers |

R2 removes the per-construction `Gf256` table build from the algebraic codes
(`ReedSolomon` ~1.7×; `Bch` ~1.1×, since its cost is the generator-polynomial LCMs
R2 does not touch). `Ldpc::new` stays ~2.7 ms — three orders of magnitude above the
algebraic codes — so the win is not making it cheaper but calling it *once per link*
instead of once per frame, which R3 does. "Mc/s" = millions of constructions/s.

### Full COFDM frame chain (§1.2) — the end-to-end R3 result

The real per-link path: `OfdmFrameMod::modulate_frame` and batch `demodulate_frame`
over many frames on one modulator instance (n_fft=64, cp_len=8, QPSK payload,
96-byte payload, LDPC(n512r12)+BCH(t=8)). "Msps" = frame IQ samples / wall time,
directly comparable to the "COFDM frame throughput" table above.

Both paths reuse a warm `CodecCache` across the measured frames — the modulator
holds one on its `OfdmFrameMod`, and the demodulator is driven through
`demodulate_frame`'s optional caller-owned cache (`Some(&cache)`), reused across
passes. This keeps the two on equal footing: both measure steady-state, codes-warm
throughput, not per-frame code construction.

| Path | pre-optimization (doc table above) | R2+R3 (warm cache) | Since baseline |
| --- | ---: | ---: | --- |
| `modulate_frame` | 0.5 | ~40–56 | **~80–110×** |
| `demodulate_frame` | 0.2 | ~33–57 | **~150–280×** |

With the cache warm on both sides, Tx and Rx land in the same order of magnitude —
two-plus orders above the pre-optimization frame table — which the per-frame
FEC-construction removal (R2+R3) accounts for. The ranges are wide because at this
payload the whole frame is ~a few hundred µs, so timer granularity and scheduling
dominate the spread; the figures are order-of-magnitude, not precise. This
benchmark's decode runs on a *noiseless* roundtrip, so each LDPC codeword is already
valid and the sum-product decoder exits on its initial syndrome check **without
entering the belief-propagation loop** — which is why R4 (a BP-inner-loop change)
does not move this number, and why the isolated, error-injected `LDPC-Decode`
fixture above is the honest measure of decode-kernel work.

> **Note on the earlier skew.** An interim R3 benchmark measured the demodulator
> through `demodulate_frame` with a *fresh* cache per call while the modulator's
> cache stayed warm, so the Rx figure (~0.94 Msps) was ~98% the ~2.7 ms LDPC
> rebuild per call — construction, not decode — making the mod/demod comparison
> apples-to-oranges. `demodulate_frame` now takes an optional caller-owned
> `CodecCache` (`None` builds a per-call one, as before; `Some(&cache)` reuses one
> across a batch), and the benchmark passes a persistent cache so both directions
> are measured warm. The `modulate` figure uses a per-pass-varying seed with full
> output consumption so it is not constant-folded.

## Running the Benchmarks

```bash
cargo test-throughput
```

Or with a custom minimum floor (Msps):

```bash
ORION_SDR_THROUGHPUT_MINSPS=50 cargo test --release --features throughput throughput -- --nocapture --test-threads=1
```

To run only FT8/FT4 throughput tests:

```bash
cargo test --release --features throughput "throughput::ft" -- --nocapture --test-threads=1
```

To run only OFDM/multicarrier throughput tests:

```bash
cargo test --release --features throughput "throughput::ofdm" -- --nocapture --test-threads=1
cargo test --release --features throughput "throughput::multicarrier" -- --nocapture --test-threads=1
```

To run only the COFDM FEC/interleave/scrambler block benchmarks:

```bash
cargo test --release --features throughput "throughput::fec" -- --nocapture --test-threads=1
```

To run the SNR sensitivity / acquisition-probability sweeps (prints full
curves, always passes — these are measurement runs, not assertions):

```bash
cargo test --release --features throughput "snr::" -- --nocapture --test-threads=1
```

To run just the OFDM SNR/acquisition sweeps:

```bash
cargo test --release --features throughput "snr::ofdm" -- --nocapture --test-threads=1
```

To run the LDPC decode-rule (min-sum) coding-gain sweep:

```bash
cargo test --release --features throughput "snr::ldpc_decode_rule" -- --nocapture --test-threads=1
```

To run the CI SNR regression tests (fixed thresholds, part of the default
`cargo test --release` run, no `throughput` feature needed):

```bash
cargo test --release "roundtrip::ofdm_snr"
cargo test --release "snr_2500hz"          # FT8/PSK31 fixed-threshold decode tests
```

Always use `--release` for throughput benchmarks — debug builds are ~10× slower and
not representative.  The SNR sweeps can be run in debug; they are slow but the
sensitivity numbers are valid.
