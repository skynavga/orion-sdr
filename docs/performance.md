<!--
  Copyright (c) 2026 G & R Associates LLC
  SPDX-License-Identifier: MIT OR Apache-2.0
-->

# Throughput

Measurements taken on Apple M2 Pro, release build (`opt-level=3`, `lto=fat`,
`codegen-units=1`), no SIMD.  Results are ordered by throughput (descending)
within each table.

## v0.0.43 Results

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
decision regions are closer together for the same average symbol energy.
This release ships soft LLRs (`OfdmSoftDemod`) but no mandatory FEC — an
external/user-supplied FEC layer over these LLRs would improve on these
uncoded numbers substantially, especially at moderate SNR.

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

To run the SNR sensitivity / acquisition-probability sweeps (prints full
curves, always passes — these are measurement runs, not assertions):

```bash
cargo test --release --features throughput "snr::" -- --nocapture --test-threads=1
```

To run just the OFDM SNR/acquisition sweeps:

```bash
cargo test --release --features throughput "snr::ofdm" -- --nocapture --test-threads=1
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
