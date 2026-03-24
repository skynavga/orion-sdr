# Throughput

Measurements taken on Apple M2 Pro, release build (`opt-level=3`, `lto=fat`,
`codegen-units=1`), no SIMD.  Results are ordered by throughput (descending).

## v0.0.20 Results

### Analog modes (9-run mean ±stdev, 65536 samples × 30 passes)

| Mode         | Msps    |
|--------------|---------|
| CW           | 149 ±3  |
| AM-AbsApprox | 149 ±4  |
| AM-PowerSqrt | 147 ±2  |
| PM           | 127 ±3  |
| FM           | 117 ±3  |
| SSB-USB      | 117 ±4  |

### Digital modes (full pipeline: mapper → mod → demod → decider, 65536 sym × 30 passes)

| Mode    | Msps |
|---------|-----:|
| QPSK    |  317 |
| BPSK    |  253 |
| QAM-16  |  209 |
| QAM-64  |   92 |
| QAM-256 |   73 |

BPSK and QPSK are faster than the analog modes because the pipeline is purely
multiply-heavy with no transcendentals.  QAM decider throughput decreases with
order because the threshold scan is O(M) per axis (M = levels/axis = 2^(BITS/2)):
QAM-256 (M=16) does 4× more comparisons per symbol than QAM-16 (M=4).

### PSK31 (10-run mean ±stdev, 4096 sym × 256 sps × 20 passes)

| Mode   | Msps    |
|--------|---------|
| BPSK31 | 817 ±7  |
| QPSK31 | 810 ±2  |

Both modes measure the full roundtrip: `modulate_bits` → `process` (demod) → `process`
(decider / Viterbi flush).  BPSK31 and QPSK31 have nearly identical throughput because
the bottleneck is the Hann-windowed pulse shaping in `write_symbol`, which is the same
for both.  The Viterbi decoder in QPSK31 adds negligible cost at 4096 symbols.

### PSK31 SNR sensitivity (50 trials/point, release build)

SNR is relative to noise in a 2500 Hz reference bandwidth (same convention as FT8/FT4).
Both modes modulate a 7-character text string with preamble=64 and postamble=32 bits.
Pipeline: `psk31_sync` (carrier detection) → `Bpsk31Demod` or `Qpsk31Demod` (whole signal)
→ Varicode decode → text search.

| SNR (dB/2500 Hz) | BPSK31 success% | QPSK31 success% |
| ---: | ---: | ---: |
| −12 | 0% | — |
| −10 | 0% | — |
| −9 | 12% | — |
| −8 | 20% | — |
| **−7** | 42% | — |
| −6 | 66% | — |
| −5 | 84% | — |
| **−4** | **100%** | — |
| 4 | 100% | 4% |
| 6 | 100% | 4% |
| 7 | 100% | 16% |
| 8 | 100% | 18% |
| **9** | 100% | 38% |
| 10 | 100% | 72% |
| 12 | 100% | 94% |
| **14** | 100% | **100%** |

50% decode points: BPSK31 ≈ −7 dB, QPSK31 ≈ +9 dB.
100% decode points: BPSK31 = −4 dB, QPSK31 = +14 dB (used as CI regression thresholds).

The demodulator uses Hann-weighted integrate-and-dump over the final quarter of each
symbol period (n ∈ [3·sps/4, sps), 64 samples at 8 kHz).  Relative to the previous
peak-sampling design, this yields approximately 18 dB improvement for both modes.
QPSK31 now outperforms BPSK31 as the G3PLX specification predicts (its rate-1/2
convolutional code recovers the 3 dB DQPSK penalty with coding gain to spare).
The remaining gap to the published G3PLX reference (BPSK31 −10 dB, QPSK31 ~−11 dB)
is due to the integration window covering only 25% of the symbol period; a full
matched filter (Phase 2) would close the remaining ~6 dB.

### FT8/FT4 (frame-at-a-time, 20 passes; "Msps" = frame samples / wall time)

| Stage | FT8 Msps | FT4 Msps |
| --- | ---: | ---: |
| mod only | 266 | 222 |
| demod only | 29 | 45 |
| codec encode only | — | — |
| codec decode only | — | — |
| full roundtrip (encode → mod → demod → decode) | 27 | 44 |

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
