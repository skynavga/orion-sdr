# Throughput

Measurements taken on Apple M2 Pro, release build (`opt-level=3`, `lto=fat`,
`codegen-units=1`), no SIMD.  Results are ordered by throughput (descending).

## v0.0.25 Results

### Analog modes (2-run mean, 65536 samples × 30 passes)

| Mode         | Msps |
|--------------|-----:|
| CW           |  163 |
| AM-AbsApprox |  160 |
| AM-PowerSqrt |  158 |
| PM           |  138 |
| SSB-USB      |  129 |
| FM           |  123 |

### Digital modes (full pipeline: mapper → mod → demod → decider, 65536 sym × 30 passes)

| Mode    | Msps |
|---------|-----:|
| QPSK    |  344 |
| BPSK    |  328 |
| QAM-16  |  258 |
| QAM-64  |  134 |
| QAM-256 |   95 |

BPSK and QPSK are faster than the analog modes because the pipeline is purely
multiply-heavy with no transcendentals.  QAM decider throughput decreases with
order because the threshold scan is O(M) per axis (M = levels/axis = 2^(BITS/2)):
QAM-256 (M=16) does 4× more comparisons per symbol than QAM-16 (M=4).

### PSK31 (2-run mean, 4096 sym × 256 sps × 20 passes)

| Mode   | Msps |
|--------|-----:|
| BPSK31 |  658 |
| QPSK31 |  587 |

Both modes measure the full roundtrip: `modulate_bits` → `process` (demod) → `process`
(decider / Viterbi flush).  The AFC loop adds a `sin_cos()` call per symbol dump.
QPSK31 uses the coherent Viterbi (`viterbi_decode_coherent`), which allocates a
`hyp_table` (hypothesised phasor per state per symbol) in addition to the
`prev_state_table`; this accounts for the ~11% gap between BPSK31 and QPSK31.

### PSK31 SNR sensitivity (50 trials/point, release build)

SNR is relative to noise in a 2500 Hz reference bandwidth (same convention as FT8/FT4).
Both modes modulate a 7-character text string with preamble=64 and postamble=32 bits.
Pipeline: `psk31_sync` (carrier detection) → `Bpsk31Demod` or `Qpsk31Demod` (whole signal)
→ Varicode decode → text search.

| SNR (dB/2500 Hz) | BPSK31 success% | QPSK31 success% |
| ---: | ---: | ---: |
| −16 | 0% | 0% |
| −14 | 0% | 14% |
| −13 | 0% | 42% |
| −12 | 2% | 70% |
| −11 | 0% | 86% |
| −10 | 14% | 94% |
| **−9** | 34% | 96% |
| **−8** | 58% | 98% |
| −7 | 82% | **100%** |
| −6 | 98% | 100% |
| **−5** | **100%** | 100% |
| −4 | 100% | 100% |
| −2 | 100% | 100% |
| 0 | 100% | 100% |

50% decode points: BPSK31 ≈ −8 dB, QPSK31 ≈ −12.5 dB.
100% decode points: BPSK31 = −5 dB, QPSK31 = −7 dB (used as CI regression thresholds).

QPSK31 coherent outperforms BPSK31 by ~4.5 dB at the 100% point and ~4.5 dB at
the 50% point.  BPSK31 uses differential detection; QPSK31 uses coherent Viterbi MLSE
(`viterbi_decode_coherent`) which tracks a hypothesised absolute phasor per trellis
state, eliminating the ~3 dB noise-product penalty of differential detection.

Both demodulators use decision-feedback matched filtering over the full sps=256 symbol
period combined with a symbol-rate decision-directed PLL (AFC).  For each sample n in
the symbol, the known previous-phasor contribution is subtracted before accumulation
(`corrected[n] = s[n] − prev_sym·(1−h[n])`), yielding a clean estimate of the current
phasor.  A first-order AFC loop (K=0.05, B_L ≈ 0.78 Hz) tracks residual carrier phase
drift at each symbol boundary.  The QPSK31 AFC discriminant operates on the absolute
phasor (not the differential product), consistent with coherent mode.
The remaining gap to the published G3PLX reference (BPSK31 −10 dB, QPSK31 ~−11 dB)
is due to differences in test methodology (single-frame vs. multi-frame averaging).

### FT8/FT4 (frame-at-a-time, 20 passes; "Msps" = frame samples / wall time)

| Stage | FT8 Msps | FT4 Msps |
| --- | ---: | ---: |
| mod only | 266 | 265 |
| demod only | 30 | 60 |
| codec encode only | — | — |
| codec decode only | — | — |
| full roundtrip (encode → mod → demod → decode) | 27 | 49 |

Frame sizes: FT8 = 151 680 samples (79 sym × 1920); FT4 = 60 480 samples
(105 sym × 576).  The codec encode/decode times are sub-millisecond and
optimized away at constant input in release mode; they are not the bottleneck.

Demod dominates: 8 Goertzel correlators × 79 symbols for FT8 vs. 4 × 105 for
FT4.  FT4 demod has higher Msps because the frame is 2.5× shorter, more than
compensating for the extra Costas blocks.

### FT8/FT4 SNR sensitivity (50 trials/point, debug build, single AWGN seed per trial)

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

50% decode points: FT8 ≈ −19 dB, FT4 ≈ −13 dB.
100% decode points: FT8 = −15 dB, FT4 = −11 dB (used as CI regression thresholds).

These are ~6 dB above the WSJT-X published limits (−21 dB FT8, −17 dB FT4). The gap
is expected: WSJT-X averages over many frames in a 15-second window with Doppler
tracking and iterative decoding. This decoder processes a single frame with no
iterative refinement.

## Running the Benchmarks

```bash
cargo test-throughput
```

Or with a custom minimum floor (Msps):

```bash
ORION_SDR_THROUGHPUT_MINSPS=50 cargo test --release --features throughput tests::performance::throughput -- --nocapture --test-threads=1
```

To run only FT8/FT4 throughput tests:

```bash
cargo test --release --features throughput "performance::throughput::ft" -- --nocapture --test-threads=1
```

To run only PSK31 throughput tests:

```bash
cargo test --release --features throughput "performance::throughput::psk31" -- --nocapture --test-threads=1
```

To run the SNR sensitivity sweep (prints full curve, always passes):

```bash
cargo test --lib --features throughput "performance::snr" -- --nocapture --test-threads=1
```

To run the CI SNR regression tests (fixed thresholds, included in `cargo test --lib`):

```bash
cargo test --lib "roundtrip::ft8_snr"
cargo test --lib "roundtrip::psk31_snr"
```

Always use `--release` for throughput benchmarks — debug builds are ~10× slower and
not representative.  The SNR sweep can be run in debug; it is slow (~2 min) but the
sensitivity numbers are valid.
