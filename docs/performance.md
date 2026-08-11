<!--
  Copyright (c) 2026 G & R Associates LLC
  SPDX-License-Identifier: MIT OR Apache-2.0
-->

# Throughput

Measurements taken on Apple M2 Pro, release build (`opt-level=3`, `lto=fat`,
`codegen-units=1`), no SIMD.  Results are ordered by throughput (descending)
within each table.

## v0.0.59 Results

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
| mod only | 320 | 261 |
| full roundtrip (mod → demod → decide) | 159 | 98 |

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
| 0.001 | 30.0 | 0.00000 | 0.00000 |
| 0.005 | 23.0 | 0.00000 | 0.00077 |
| 0.01 | 20.0 | 0.00000 | 0.00539 |
| 0.02 | 17.0 | 0.00024 | 0.02287 |
| 0.05 | 13.0 | 0.00853 | 0.07659 |
| 0.1 | 10.0 | 0.03606 | 0.13788 |
| 0.2 | 7.0 | 0.09481 | 0.21275 |
| 0.5 | 3.0 | 0.21729 | 0.31129 |
| 1.0 | 0.0 | 0.30799 | 0.37877 |
| 2.0 | −3.0 | 0.38424 | 0.42654 |

The `TrainingSymbolHold` equalizer (one channel estimate per packet, held
constant — the default for this feature's line-of-sight VHF–EHF target
bands) still raises BER relative to the flat-channel table above at matched
noise scales, since a 2-tap channel spreads each subcarrier's SNR unevenly
across the band. A per-bin equalizer corrects channel *gain/phase*, not the SNR
penalty of nulls the channel introduces at some subcarriers.

**These curves improved sharply in v0.0.58** — QPSK at `noise_scale = 0.05`
fell from 0.12148 to 0.00853, a 14x reduction, and QAM-16 from 0.24696 to
0.07659. The band-limited preamble is the cause: acquisition timing is more
accurate (see the sync table below, where wide-CFO lock at `noise_scale = 0.2`
went 74% → 100%), and a timing error under multipath costs far more than under
a flat channel because it adds inter-symbol interference on top of the
frequency-selective fade.

### OFDM packet-sync acquisition probability (50 trials/point)

`ofdm_sync`'s Schmidl & Cox-style repeated-segment preamble, timing +
fractional CFO only, vs. the same preamble extended with a training symbol
for wide-range integer-CFO recovery (applied CFO = 5.3 subcarrier spacings,
well beyond the fractional-only ±½-spacing capture range).

| noise_scale | Fractional-only lock% | Fractional+integer lock% (5.3-spacing CFO) |
| ---: | ---: | ---: |
| 0.01 | 100.0% | 100.0% |
| 0.05 | 100.0% | 100.0% |
| 0.1 | 92.0% | 100.0% |
| 0.2 | 84.0% | 100.0% |
| 0.5 | 12.0% | 46.0% |
| 1.0 | 0.0% | 0.0% |
| 2.0 | 0.0% | 0.0% |
| 5.0 | 0.0% | 0.0% |

The fractional-only curve degrades past `noise_scale ≈ 0.2` as AWGN comes to
dominate the preamble's own energy — timing lock is the limiting factor at low
SNR, not the CFO estimators layered on top of it.

**The wide-CFO curve improved markedly in v0.0.58**, from 74% to 100% lock at
`noise_scale = 0.2` and 18% to 46% at 0.5, and now holds 100% two points
further than the fractional-only path. Two changes contribute: the training
symbol it depends on is band-limited and amplitude-matched to the data, so the
integer-CFO correlation is cleaner; and `ofdm_sync` no longer folds the
correlated-window energy ratio into the reported score, so acceptance is a
question about phase coherence rather than about whether the preamble happened
to be the loudest thing in the buffer.

### COFDM frame throughput (n_fft=64, cp_len=8, QPSK payload, 96-byte payload, 200 passes)

Full frame pipeline: `OfdmFrameMod::modulate_frame` (CRC → concatenated FEC
encode → interleave → map → preamble/header) and `OfdmFrameDemod::decode`
(soft-demap → deinterleave → concatenated FEC decode → CRC). Two concatenations
are shown.
"Msps" is total frame samples / wall time.

Both paths reuse a warm per-link `CodecCache` across the measured frames (the
modulator holds one; the batch demodulator is driven with a persistent
caller-owned cache), so the figures are steady-state and not dominated by
per-frame FEC-code construction. Warm-run steady-state numbers.

| Config | mod Msps | demod Msps | frame samples |
| --- | ---: | ---: | ---: |
| LDPC(n512r12) + BCH(t=8) | ~82 | ~52 | 2584 |
| Convolutional r1/2 + RS(60,52) | ~90 | ~19 | 1936 |

Decode is the limiting side of both concatenations, but for different reasons.
The LDPC+BCH demodulator (~52 Msps) runs the sum-product decoder, whose per-edge
message storage is a flat contiguous buffer and whose check-node update caches
`tanh(msg/2)` per edge — so its cost is the belief-propagation iteration itself,
not memory or transcendental overhead. The Conv+RS demodulator (~19 Msps) is
slower because the punctured convolutional inner code runs a full-block soft
Viterbi over the whole payload every frame, which the LDPC path's per-codeword
belief propagation and the shared code cache do not lighten. Both are far above
the per-frame FEC-*construction* cost, which the `CodecCache` removes entirely
(see "Per-frame code-object construction cost" below). Larger payloads amortize
the fixed preamble/header overhead but not the per-codeword decode.

### COFDM frame-error-rate vs. noise scale (n_fft=64, cp_len=8, QPSK payload, 100 trials/point, flat channel)

Frame-error-rate (whole-frame CRC pass/fail) for the two concatenations. Here
`noise_scale` is AWGN power relative to the **payload's** power — see the note
below the table, which is why these figures are not comparable point-for-point
with pre-v0.0.58 ones. Compare against the uncoded QPSK column of "OFDM BER vs.
noise scale": the FEC drives *frame* errors to zero at noise scales where the
uncoded *bit*-error rate is already substantial.

| noise_scale | equiv. SNR (dB) | LDPC+BCH FER | Conv+RS FER |
| ---: | ---: | ---: | ---: |
| 0.2 | 7.0 | 0.000 | 0.000 |
| 0.5 | 3.0 | 0.000 | 0.000 |
| 0.6 | 2.2 | 0.050 | 0.050 |
| 0.7 | 1.5 | 0.480 | 0.500 |
| 0.8 | 1.0 | 0.960 | 0.930 |
| 0.9 | 0.5 | 1.000 | 1.000 |
| 1.0 | 0.0 | 1.000 | 1.000 |

**The cliff moved out by roughly a factor of two in v0.0.58.** Both
concatenations previously began failing at `noise_scale = 0.3`; they now hold
FER = 0 through 0.5 and break between 0.6 and 0.8. Two causes, and the second
matters when comparing against the older table:

- Frame acceptance no longer requires inner-FEC convergence (v0.0.57). A frame
  whose payload the CRC vouches for is accepted however the stages beneath
  fared, which recovers frames that were previously discarded while correct.
- The noise reference changed meaning. `noise_scale` here is relative to the
  **payload's** power. Before the preamble was band-limited it was ~30 dB hotter
  than the payload and full-band, so a buffer-mean reference injected
  substantially more noise for the same nominal figure. Numbers either side of
  v0.0.58 are not directly comparable at a given `noise_scale`.

Both hold FER = 0 well past the point where uncoded QPSK shows BER ≈ 0.08
(`noise_scale = 0.5`) — the concatenated FEC's coding gain. The two
concatenations now track each other closely through the cliff rather than
Conv+RS holding a clear edge.

Measured with the batch `OfdmFrameDemod` at a known start, so this is a
measurement of the FEC rather than of acquisition; the sync table above covers
acquisition separately. Feeding the streaming receiver instead folds in sync
failures and reports 0.35 rather than 0.000 at `noise_scale = 0.02` for the
same link.

Regenerate with `snr::cofdm_fer` (see the SNR sweep command at the top). This
table previously had no committed test behind it and had drifted a full noise
decade out of date as a result.

## COFDM FEC block throughput

Per-block benchmarks for the individual channel-coding blocks (`throughput::fec`),
measured on Apple M2 Pro, release build, `--test-threads=1`, 200 passes/point.
**"Msps" in this section = information bits processed per second** (before coding on
Tx, after decoding on Rx) — comparable across code rates, and NOT the sample-domain
figure used elsewhere in this doc. Numbers are warm-run steady-state; the first run
of each fixture is a cold-cache outlier and is discarded.

### COFDM streaming-receiver frame-error rate (60 trials/point)

`OfdmFrameStreamDemod` over the same link as the frame-error-rate table above,
but through the receiver a caller actually gets: acquisition, equalization and
residual-carrier tracking included, rather than the batch demodulator handed a
known frame start. Backed by `snr::cofdm_stream_fer`. "In-band SNR" references
the noise to the **payload's** power, not the buffer mean — the preamble is
deliberately hotter, so a buffer-mean reference injects more noise than the
nominal figure claims.

| In-band SNR (dB) | FER | mean CBER |
| ---: | ---: | ---: |
| 6 | 0.800 | 0.07354 |
| 8 | 0.367 | 0.03358 |
| 10 | 0.133 | 0.00885 |
| 12 | 0.050 | 0.00120 |
| 15 | 0.017 | 0.00002 |
| 20 | 0.000 | 0.00000 |
| 25 | 0.000 | 0.00000 |
| 30 | 0.000 | 0.00000 |

**These curves improved sharply in v0.0.59**, when the receiver began tracking
residual carrier phase across a frame (`remove_common_phase_error`). Measured on
this fixture with tracking disabled, FER was 0.083 at 20 dB, 0.350 at 15 dB,
0.550 at 12 dB and 0.717 at 10 dB — error-free reception started at 25 dB rather
than 20, and every point below it was several times worse.

The cause is that the Schmidl & Cox carrier estimate has variance while
`TrainingSymbolHold` measures the channel once and holds it for the whole frame,
so a residual offset integrated into constellation rotation that nothing
corrected — on this 53.8 ms frame, a few Hz is already tens of degrees by the
last symbol. The failures looked exactly like an FEC cliff, which is why the gap
against the batch-demodulator table above is the number worth watching: that one
isolates the concatenated FEC, this one includes everything the receiver adds.

### Per-block, single direction

| Block | Variant | Tx Msps | Rx Msps |
| --- | --- | ---: | ---: |
| LDPC | N512R12 (rate 1/2) | 457 | ~24 |
| LDPC | N576R23 (rate 2/3) | 577 | ~25 |
| LDPC | N512R34 (rate 3/4) | 640 | ~11 |
| Convolutional | rate 1/2 | 610 | 27.1 |
| Convolutional | rate 2/3 | 347 | 27.3 |
| Convolutional | rate 3/4 | 384 | 26.1 |
| Convolutional | rate 5/6 | 345 | 28.4 |
| Convolutional | rate 7/8 | 328 | 28.8 |
| BCH | t=8 | 99.6 | 27.1 |
| Reed–Solomon | RS(204,188) | 799 | 165 |
| Reed–Solomon | RS(60,52) | 1126 | 140 |
| Interleaver | u8, 32×32 (kernel) | 5088 | 6083 |
| Interleaver | f32, 32×32 (kernel) | 4668 | 7042 |
| Interleave-Bits | chain driver, 8+ blocks | ~1700 | — |
| Scrambler | width 7 / 15 / 32 | 196 / 198 / 202 | (self-inverse) |

Decode is the limiting direction for every code: the algebraic Berlekamp–Massey and
the LDPC belief propagation cost far more than the systematic encoders. `LDPC-Decode
N512R34` is the slowest block (~11 Msps) — the rate-3/4 code's denser checks and
tighter error margin run the most belief-propagation work per information bit.

**LDPC decoder structure.** The sum-product decoder stores its per-edge messages in a
flat contiguous buffer with a `check_start` offset table (a compressed-sparse-row
layout), so the check- and variable-node loops walk contiguous memory rather than a
jagged `Vec<Vec<f32>>`; and its check-node update caches `tanh(msg/2)` once per edge
per iteration instead of recomputing the transcendental inside every leave-one-out
product (an O(deg²)→O(deg) reduction). The Tanner-graph edge indices are precomputed
at construction, so the inner loops index the message arrays directly with no
per-iteration search. Together these keep the decode cost at the belief-propagation
iteration itself; the densest, most-iterating code benefits most.

**Interleaver / scrambler.** The `BlockInterleaver` kernels are memory-bandwidth-bound
(thousands of Msps). The chain-driver `interleave_bits` (which fragments into blocks,
pads, and permutes) builds its interleaver and scratch buffers once per call rather
than per block — the `Interleave-Bits chain` row measures that path, since the default
MCS uses no interleaver and so the frame chain does not exercise it. The scrambler is
an additive LFSR whitener; its throughput is set by the per-byte `count_ones` feedback
advance.

### Paired forward→inverse roundtrip (correctness asserted each pass)

| Roundtrip | Variant | Msps |
| --- | --- | ---: |
| LDPC enc→dec | N512R12 | 207 |
| LDPC enc→dec | N576R23 | 253 |
| LDPC enc→dec | N512R34 | 282 |
| Conv enc→dec | rate 1/2 | 25.1 |
| Conv enc→dec | rate 2/3…7/8 | 24.3–25.4 |
| BCH enc→dec (t errors) | t=8 | 21.9 |
| RS enc→dec (t errors) | RS(204,188) | 141 |
| RS enc→dec (t errors) | RS(60,52) | 128 |
| Interleaver round trip | u8/f32 32×32 | 1533 (f32) |
| Scrambler round trip | width 32 | (see per-block) |

The LDPC roundtrip runs faster than the standalone `LDPC-Decode` block because it
feeds clean codewords: belief propagation converges on the initial syndrome check and
exits, whereas the standalone decode fixture injects soft errors to force realistic
iteration. The two measure different, deliberately chosen decoder workloads.

### LDPC decode rule: sum-product vs. min-sum

The LDPC inner decoder's check-node rule is selectable via
`OfdmConfig::with_ldpc_decode_rule` (`DecodeRule::SumProduct` / `MinSum` /
`ScaledMinSum(α)`); sum-product is the default. Min-sum replaces the transcendental
tanh product with `∏sign · min|msg|`, trading a little coding gain for a cheaper
update. **Speed** (`throughput::fec` `throughput_ldpc_decode_rules`, error-injected
fixture, warm steady-state Msps):

| Code | sum-product | min-sum | scaled-min-sum(0.75) |
| --- | ---: | ---: | ---: |
| N512R12 | ~8.4 | ~14.9 (1.8×) | ~12.6 (1.5×) |
| N576R23 | ~12.3 | ~21.3 (1.7×) | ~21.7 (1.8×) |
| N512R34 | ~7.4 | ~17.8 (2.4×) | ~19.0 (2.6×) |

Min-sum runs ~1.7–2.6× faster, and the densest/most-iterating code (`N512R34`) gains
most. **Coding gain** (`snr::ldpc_decode_rule`, BPSK-over-AWGN on the codeword, 200
trials/point, mean post-decode BER; all rules see the identical noise realization):

| Es/N0 (dB) | N512R12 SP | MS | SMS(0.75) | N512R34 SP | MS | SMS(0.75) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| −2.5 | 0.070 | 0.086 | 0.074 | 0.143 | 0.160 | 0.152 |
| −2.0 | 0.033 | 0.041 | 0.040 | 0.128 | 0.145 | 0.137 |
| −1.5 | 0.015 | 0.019 | 0.020 | 0.114 | 0.130 | 0.123 |
| −1.0 | 0.0036 | 0.0043 | 0.0061 | 0.100 | 0.114 | 0.108 |
| 0.0 | 0.0002 | 0.0001 | 0.0002 | 0.060 | 0.073 | 0.065 |

Reading the waterfall's horizontal (dB) shift at matched BER: plain min-sum costs
~0.3–0.5 dB of coding gain, while scaled-min-sum(0.75) recovers most of it — within
~0.15–0.3 dB of sum-product across the slope (closest on the denser `N512R34`). So
scaled-min-sum buys ~2× decode speed for a sub-0.3-dB penalty. It is an opt-in choice
on the payload decode; the header always decodes with sum-product, and sum-product is
the on-air default.

### Per-frame code-object construction cost

Constructing a code — `Ldpc::new` above all — would otherwise be the dominant
per-frame FEC cost. The per-link `CodecCache` builds each `Ldpc`/`Bch`/`ReedSolomon`
once and reuses it across every frame, and `Bch`/`ReedSolomon` share a single process-
wide `Gf256` rather than rebuilding its log/antilog tables per construction. These
microbenchmarks measure construction in isolation.

| Construction | Cost | Notes |
| --- | ---: | --- |
| `Ldpc::new(N512R12)` | ~2.7 ms | sparse-H build + HashSet 4-cycle guard; three orders of magnitude above the algebraic codes |
| `Bch::new(t=8)` | ~3.1 µs (0.32 Mc/s) | dominated by generator-polynomial conjugacy-class LCMs |
| `ReedSolomon::dvb()` | ~0.41 µs (2.4–2.6 Mc/s) | 2t linear-factor multiplies over the shared `Gf256` |
| LDPC ×64 frames, built once & reused | **~5000×** vs. rebuilding each frame | the amortized behavior the `CodecCache` delivers |

`Ldpc::new`'s ~2.7 ms is why per-frame reconstruction is untenable: the cache turns it
into a one-time per-link cost. "Mc/s" = millions of constructions per second.

### Full COFDM frame chain — reading the table above

The COFDM frame-chain figures are published **once**, in the "COFDM frame
throughput" table above. They used to
appear twice, and the two copies drifted: one was refreshed in v0.0.58 and the
other kept quoting pre-0.0.58 numbers (~87/~58 and ~97/~29 against the measured
~82/~52 and ~90/~19). A number published in two places is a number that will
disagree with itself, so this section keeps only the caveats that belong with it.

The modulate figure uses a per-pass-varying seed and consumes the whole output, so it
is not constant-folded. The demodulate figure runs on a noiseless roundtrip: each
LDPC codeword arrives already valid, so the sum-product decoder exits on its initial
syndrome check without entering the belief-propagation loop — this measures the
frame-chain plumbing (soft-demap, deinterleave, syndrome check, CRC) with the cache
warm, not the worst-case iteration count. For decode-kernel cost under real error
loads, see the isolated, error-injected `LDPC-Decode` fixtures above. The Conv+RS
demodulator is slower because its convolutional inner code runs a full-block soft
Viterbi over the whole payload every frame — work the LDPC path's per-codeword belief
propagation and shared code cache do not lighten.

## DVB-T / NB-DVB-T

DVB-T runs the 2K structure (n_fft = 2048) — a ~30× larger FFT than the small
COFDM test plans above, so absolute Msps are lower and FFT-dominated. See
[dvb.md](dvb.md) for the waveform design.

### DVB-T 2K payload-FEC chain, narrowband bandwidth sweep (184-byte payload, 200 passes)

The DVB-T payload FEC (K=7 conv + Forney I=12 + RS(204,188) + energy dispersal)
over the 2K carrier map via the generic frame layer, at the three amateur
bandwidth modes. Because NB-DVB-T is a **pure fs-scaling of the same 2K
structure**, the bandwidth mode changes only the sample-rate metadata, not the
per-sample compute — so throughput is essentially identical across modes (the
small spread is run-to-run noise). "Msps" is total frame samples / wall time; the
roundtrip column modulates then decodes one frame per pass, so it is set by the
slower (decode) direction.

| Bandwidth mode | fs (MS/s) | mod Msps | demod Msps | roundtrip Msps |
| --- | ---: | ---: | ---: | ---: |
| 333 kHz | 0.400 | ~66 | ~5.8 | ~5.4 |
| 1 MHz | 1.201 | ~66 | ~5.8 | ~5.4 |
| 2 MHz | 2.402 | ~66 | ~5.8 | ~5.4 |

The takeaway: the amateur bandwidth choice is an RF/hardware concern (channel
occupancy, PlutoSDR's ~521 kS/s continuous-TX floor rules out 333 kHz), **not a
compute one** — the DSP cost is identical.

### DVB-T conformant frame, end to end (184-byte TS payload, 68-symbol frame, 50 passes)

The full conformant preamble-less frame (`DvbTFrameMod` ↔ `DvbTFrameDemod`): TS
packetization + energy dispersal, payload FEC, Figure-9a soft-decision through the
four-phase scattered-pilot grid, TPS signalling, and guard-interval acquisition on
RX. "Msps" is frame samples / wall time; the roundtrip row modulates then
GI-acquires and decodes one frame per pass.

| Path | Msps |
| --- | ---: |
| modulate | ~33 |
| demodulate (incl. GI acquisition) | ~13 |
| roundtrip (mod → GI-acquire → demod) | ~9.5 |

The modulate figure is lower than the generic 2K chain above (~33 vs ~66) because
the conformant modulator **fills the frame**: a short payload is stuffed with null
TS packets (§4.4 — every data carrier must carry data), so it runs the full
payload FEC over a whole frame's worth of RS packets rather than one packet plus
zeros. The RX adds per-symbol scattered-pilot equalization, DVB-T-exact soft LLRs,
TPS DBPSK recovery, and the guard-interval correlation search on top of the FFT
chain; the per-symbol pilot interpolation is its dominant cost, locating each data
carrier's bracketing pilots by binary search over the sorted pilot set
(O(data·log pilots) per symbol, for 1512 data carriers × ~176 reference pilots)
and reusing a ratio scratch buffer across symbols, so the estimate is a fraction
of the FFT rather than an O(data·pilots) scan. The roundtrip (~9.5 Msps) is set by
the two directions in series.

### DVB-T super-frame, end to end (700-byte payload, 4 frames, 20 passes)

The conformant super-frame (`DvbTSuperFrameMod` ↔ `DvbTSuperFrameDemod`): four
consecutive frames (numbers 0–3) with the alternating TPS sync word and the 16-bit
cell id split across them.

| Path | Msps |
| --- | ---: |
| modulate | ~32 |
| demodulate | ~13 |
| roundtrip | ~9.5 |

A super-frame is the single-frame conformant path run four times plus the
multi-frame sequencing and cross-frame checks (frame-number sequence, cell-id
reassembly), which are negligible next to the per-frame FFT/FEC/equalize work — so
its throughput tracks the conformant single frame's, with no new bottleneck.

### DVB-T streaming receiver (4-frame stream, feed/flush, 20 passes)

The streaming receiver (`DvbTFrameStreamDemod`, `feed`/`flush`) decoding a
continuous run of frames: it accumulates IQ, guard-interval-acquires the next
frame at the front of the buffer, decodes it, and drains its samples.

| Path | Msps |
| --- | ---: |
| streaming demodulate (feed → decode → drain) | ~12 |

Slightly below the batch conformant demodulate (~13): the per-frame decode work
is identical, and the small gap is the streaming buffer management — the repeated
front-of-buffer GI search and the per-frame `drain` of consumed samples.

### DVB-T integer-CFO correction overhead (184-byte frame, 50 passes)

`DvbTFrameDemod::with_integer_cfo_correction(true)` folds a continual-pilot
integer-CFO estimate into every decode (after the RX's own guard-interval
acquisition), for a link that may sit a whole subcarrier or more off frequency. The
overhead is the always-on cost of that estimate, measured as a plain flag-off
decode vs. a flag-on decode over the **same clean buffer** (offset 0, so the
estimate returns k = 0 and the rotate is skipped — the normal locked-link case).

| Path | Msps |
| --- | ---: |
| decode, correction off | ~13.1 |
| decode, correction on (always estimating) | ~12.5 |

That is **a few percent** (measured ~4–6% across six runs — the added work, an FFT
of a few symbols plus a 45-bin continual-pilot search per frame, is small enough
that the ratio is dominated by decode jitter). It is off by default (a clean link
needs no correction), so the common path pays nothing; when a capture carries an
integer offset, the flag removes it internally rather than the caller running a
pre-pass. The estimator `sync::dvb_t_integer_cfo` is also public for standalone use.

### DVB-T conformant frame, decode-vs-SNR (GI 1/8, 30 trials/point)

Frame-decode success and post-decode payload BER vs. per-sample SNR, for the
conformant frame. QPSK r1/2 and 16-QAM r3/4 payloads, each a genuine full-payload
frame (a short payload is filled with null TS packets, so every carrier decodes).

| SNR (dB) | QPSK r1/2 decode% | 16-QAM r3/4 decode% |
| ---: | ---: | ---: |
| 2 | 0% | 0% |
| 4 | 100% | 0% |
| 10 | 100% | 0% |
| 12 | 100% | 77% |
| 15 | 100% | 100% |

Payload BER is 0 whenever a frame decodes at all (the FEC either clears the
channel or the frame fails as a unit). The waterfall is ordered by nominal
robustness: the redundant QPSK r1/2 config locks by ~4 dB, while the denser
16-QAM r3/4 needs ~12–15 dB — the whole frame of real coded data must clear, so
these are steeper full-payload cliffs than a mostly-empty frame would show.

Channel estimation excludes the 17 TPS carriers from the equalizer's pilot
reference set: the modulator transmits data-power DBPSK on them, not the boosted
`w_k` pilot value the grid records, so using them as references would divide the
received cell by the wrong known value and corrupt the interpolated estimate on
the data carriers straddling each TPS carrier. The equalizer interpolates its
estimate across the TPS bins from the true continual + scattered pilots instead;
the noiseless correctness guard is
`roundtrip::dvb_t::dvb_t_equalizer_noiseless_clean_*`.

## Out-of-band emission (spectral shaping)

Three off-by-default levers reduce OFDM's `~1/f` out-of-band skirt: an
edge-carrier guard band, a TX symbol-window taper, and a TX baseband spectral
mask. [ofdm.md](ofdm.md) has the geometry and the transparency arguments;
[modulate.md](modulate.md#choosing-the-numbers) has the sizing recipe. This
section is what they measure.

All figures are **mean power in a stated band, in dB relative to the in-band
level**, read through a 4-term Blackman–Harris analysis window. The window is not
a detail: a rectangular slice has its own ~−35 dB leakage floor and would hide a
60 dB mask completely — what got measured would be the leakage of the *analysis*,
not of the signal. Anything claiming deep stop-band attenuation has to be read
through a window whose sidelobes sit below the attenuation claimed.

### COFDM, each lever added in turn (n_fft 256, cp_len 64, 31-carrier edge guard, 65-tap 60 dB mask, roll_off 32)

Two bands, because the taper and the mask do not act in the same place. The mask
leaves its own transition deliberately unattenuated, so close to the band edge
the taper is the useful lever; past the transition the mask takes over by tens
of dB. Reporting only one band would misrepresent one of them.

| Configuration | Near edge (just outside the carriers) | Stop band (past the mask's transition) |
| --- | ---: | ---: |
| Baseline (no lever) | +5.9 | +5.8 |
| + edge guard | −21.3 | −29.7 |
| + symbol windowing | −30.8 | −61.7 |
| + baseband mask (no taper) | −26.1 | −95.6 |
| **All three** | **−33.8** | **−115.6** |

Read down the stop-band column: the edge guard buys ~35 dB by moving the loudest
`sinc` generators inward, windowing another ~32 dB by lowering the skirt's decay
*rate*, and the mask a further ~34 dB by attenuating what is left directly in the
frequency domain. They stack because the mechanisms are independent — which is
the whole argument for shipping all three. Note the fourth row against the third
in the *near-edge* column: the mask alone is **worse** there than the taper
alone, since near-edge energy sits inside its transition band.

Fixtures: `unit::ofdm::all_three_spectral_levers_stack` and
`tx_lowpass_drops_out_of_band_below_the_windowing_floor` (both print their
numbers under `--nocapture`).

### COFDM frame, as configured from Python (n_fft 256, cp_len 64, edge guard 31, 45-tap 60 dB mask, roll_off 8)

The same effect through the frame layer and the PyO3 bindings, on a shorter
filter sized to leave room for a taper (group delay 22 + roll_off 8 = 30, inside
the 32-sample slack at `backoff = cp_len/2`).

| Configuration | Stop band (\|f\|/fs ∈ [0.47, 0.5]) | In band (\|f\|/fs ≤ 0.36) |
| --- | ---: | ---: |
| Baseline (edge guard only) | −25.3 | +6.1 |
| + symbol windowing | −35.6 | +6.0 |
| + baseband mask | −91.0 | +6.1 |
| **Both** | **−101.0** | **+6.0** |
| *no edge guard, for contrast* | *+5.6* | — |

In-band power moves by less than 0.2 dB across every configuration: the shaping
is paid for out of the guard interval, not out of the payload. The last row is
why the edge guard is a prerequisite rather than an optional extra — with
`edge_guard = 0` the plan fills every bin to Nyquist, so there is no null band
for a mask's transition to live in and [0.47, 0.5] still holds real carriers.

Fixture: `python/tests/test_spectral_shaping.py::TestCofdmSpectrum`.

### DVB-T conformant frame (2K, G1/8, 89-tap 60 dB mask)

DVB-T needs no edge guard — 343 of its 2048 bins are already inactive, which is
the null band the mask works in — and could not have one anyway, since its
extreme carriers are mandatory continual pilots.

| Configuration | Null band (past the transition) | In band |
| --- | ---: | ---: |
| Plain | −15.7 | +0.5 |
| + baseband mask | **−81.8** | +0.5 |

**66 dB** of null-band attenuation with in-band power unchanged to within 0.1 dB,
and TPS, pilots, and payload all recovered. Fixture:
`roundtrip::dvb_t::dvb_t_tx_lowpass_attenuates_the_null_band`.

### Cost: what shaping does to modulator throughput (DVB-T 2K, G1/8, 184-byte payload, 50 passes)

Both levers are post-passes over an already-assembled frame, so the honest number
is the delta against the same modulator with shaping off — everything upstream
(TS packetization, payload FEC, mapping, pilots, TPS, IFFT, CP insertion) is
identical work.

| Configuration | mod Msps | vs. plain |
| --- | ---: | ---: |
| plain | ~35 | — |
| symbol windowing (roll_off 16) | ~35 | free |
| 45-tap mask | ~18.5 | −47% |
| roll_off 16 + 89-tap mask | ~10.8 | −69% |

**The taper is free; the mask is not.** The taper is `O(roll_off)` per symbol and
touches 32 of 2304 samples; the mask is `O(num_taps)` per sample across the whole
frame, so its cost scales linearly with the filter length the guard budget
affords — and a longer guard, which buys a sharper mask, also buys a more
expensive one. Halving the filter (89 → 45 taps) nearly halves the overhead.

This is a **transmit-side compute** cost only. Receiving a shaped signal costs no
extra compute: the back-off is a change of FFT-window offset, and the mask is
absorbed by the channel estimate the equalizer was already computing. It does,
however, cost *sensitivity* — see below.

Fixture: `throughput::dvbt::throughput_dvb_t_spectral_shaping_cost`.

### The RX window back-off costs sensitivity well before it aliases

Measured on DVB-T 2K, G1/8, QPSK r1/2, 20 trials/point.

`DVB_T_MAX_RX_WINDOW_BACKOFF` = 85 is where the scattered-pilot interpolation
*aliases* (`n_fft / (2 · 12)`). It is **not** where the back-off becomes usable
in noise, and the gap between the two is large. A back-off `b` puts a phase ramp
on the spectrum that advances `2π·b·12/2048` per pilot gap, and the equalizer
interpolates *linearly* between pilots — so it approximates an arc by a chord,
with a fractional magnitude error of `1 − cos(θ/2)` in between. Aliasing needs
`θ = 180°`; the error is already crippling well below it.

| `b` | θ per pilot gap | interp. error | 100% decode at | penalty |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 0° | 0% | 4 dB | — |
| 32 | 68° | 17% | 4 dB | 0 dB |
| 40 | 84° | 26% | 5 dB | ~1 dB |
| 42 | 89° | 28% | 5 dB | ~1 dB |
| 48 | 101° | 37% | 6 dB | ~2 dB |
| 56 | 118° | 49% | 6 dB | ~2 dB |
| 64 | 135° | 62% | 10 dB | ~6 dB |
| 85 | 179° | 100% | never (0% at 15 dB) | — |

Two practical rules follow, and they matter more than the aliasing cap:

- **`b ≤ 32` is free** on DVB-T 2K. Keep the back-off here if you can.
- **`b ≤ n_fft / (4 · pilot_spacing)` = 42** costs ≤1 dB — the `θ ≤ 90°` rule.
  Past it the cost climbs fast, and at the aliasing cap itself the link does not
  close at any SNR.

So the usable shaping slack, `min(cp_len − b, b)`, saturates at **42 samples from
G1/16 onward** — not the 85 the aliasing cap suggests. That in turn caps the
practical mask at roughly 45 taps (group delay 22) plus a taper of ~20 samples.
An 89-tap mask needs 44 samples of group delay alone, which only fits at `b ≥ 44`
— already into the penalty region.

**None of this applies to COFDM.** `TrainingSymbolHold` measures every bin from
the training symbol, so it interpolates nothing and absorbs any `b` the guard
allows; `b = cp_len/2` remains right there. This is a cost of *pilot
interpolation*, which is the DVB-T scattered path's estimator.

### Shaping itself is free once it fits the budget

Same link (DVB-T 2K, G1/8, QPSK r1/2), 30 trials/point. With the shaping sized to fit — `roll_off` 8 + a 45-tap mask (group delay 22) =
30 samples, inside the 32-sample slack at `b = 32`:

| Configuration | 3 dB | 4 dB | 5 dB | 6 dB | 8 dB |
| --- | ---: | ---: | ---: | ---: | ---: |
| plain, `b = 0` | 53% | 100% | 100% | 100% | 100% |
| plain, `b = 32` | 0% | 93% | 100% | 100% | 100% |
| taper 8 + 45-tap mask, `b = 32` | 0% | 87% | 100% | 100% | 100% |

The whole cost is the back-off's (~0.5 dB); adding the taper and the mask on top
of it costs essentially nothing. That is the shaping's central claim, and it holds
**only while the budget holds**. The same comparison with an over-budget
configuration (`b = 64`, 89-tap mask, `roll_off` 16) decodes 53% at 10 dB against
plain's 100% — about 6 dB of loss, all of it bought by the oversized back-off
that oversized filter demanded.

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

To run only the DVB-T / NB-DVB-T waveform benchmarks (bandwidth sweep, conformant
frame, super-frame, streaming receiver, integer-CFO overhead):

```bash
cargo test --release --features throughput "throughput::dvbt" -- --nocapture --test-threads=1
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
