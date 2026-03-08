# Build and Test Commands

## Building

```bash
# Debug build
cargo build

# Release build
cargo build --release
```

## Testing

```bash
# Standard tests (always run before committing)
cargo test --release

# Run only unit tests
cargo test-unit

# Run only roundtrip tests
cargo test-roundtrip

# Run throughput benchmarks
cargo test-throughput
```

See `.cargo/config.toml` for the alias definitions. Always use `--release` for
throughput runs — debug builds are ~10× slower and not representative.

## Python Extension

```bash
# Build a wheel
maturin build --release

# Install into an active virtualenv (editable/develop mode)
maturin develop --release
```

## Python Tests

```bash
# Install the extension into your active virtualenv first
maturin develop --release

# Run all Python tests
pytest

# Run with verbose output
pytest -v
```
