# Throughput

Measurements taken on Apple M2 Pro, release build (`opt-level=3`, `lto=fat`,
`codegen-units=1`), no SIMD.  Results are ordered by throughput (descending).

## v0.0.15 Results

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

### FT8/FT4 (frame-at-a-time, 20 passes; "Msps" = frame samples / wall time)

| Stage | FT8 Msps | FT4 Msps |
|---|---:|---:|
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

## Running the Benchmarks

```bash
cargo test-throughput
```

Or with a custom minimum floor (Msps):

```bash
ORION_SDR_THROUGHPUT_MINSPS=50 cargo test --release --features throughput tests::throughput -- --nocapture --test-threads=1
```

To run only FT8/FT4 throughput tests:

```bash
cargo test --release --features throughput "throughput::ft" -- --nocapture --test-threads=1
```

Always use `--release` — debug builds are ~10× slower and not representative.
