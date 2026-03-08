# Throughput

Measurements taken on Apple M2 Pro, release build (`opt-level=3`, `lto=fat`,
`codegen-units=1`), no SIMD, averaged over 9 runs.  Results are ordered
by throughput (descending).

## v0.0.14 Results

### Analog modes (9-run mean ±stdev)

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

## Running the Benchmarks

```bash
cargo test-throughput
```

Or with a custom minimum floor (Msps):

```bash
ORION_SDR_THROUGHPUT_MINSPS=50 cargo test --release --features throughput tests::throughput -- --nocapture --test-threads=1
```

Always use `--release` — debug builds are ~10× slower and not representative.
