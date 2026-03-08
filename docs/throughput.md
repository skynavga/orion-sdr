# Throughput

Measurements taken on Apple M2 Pro, release build (`opt-level=3`, `lto=fat`,
`codegen-units=1`), no SIMD, averaged over 9 runs.

## v0.0.14 Results

| Mode         | Msps    |
|--------------|---------|
| CW           | 149 ±3  |
| AM-AbsApprox | 149 ±4  |
| AM-PowerSqrt | 147 ±2  |
| PM           | 127 ±3  |
| FM           | 117 ±3  |
| SSB-USB      | 117 ±4  |

## Running the Benchmarks

```bash
cargo test-throughput
```

Or with a custom minimum floor (Msps):

```bash
ORION_SDR_THROUGHPUT_MINSPS=50 cargo test --release --features throughput tests::throughput -- --nocapture --test-threads=1
```

Always use `--release` — debug builds are ~10× slower and not representative.
