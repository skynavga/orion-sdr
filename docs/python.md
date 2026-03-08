# Python Bindings

All analog and digital modulators and demodulators are available as a native Python
extension via PyO3: CW, AM, SSB, FM, PM (analog) and BPSK, QPSK, QAM-16/64/256 (digital)
— 16 classes in total.

## Installation

**Prerequisites:** a [Rust toolchain](https://rustup.rs) (stable, 1.75+) and Python 3.12+.

### Set up a virtualenv

A `.venv` at the project root is the recommended setup. Maturin finds it automatically
without needing `source .venv/bin/activate`.

```bash
python3 -m venv .venv
.venv/bin/pip install maturin numpy pytest
```

### Build and install (editable/develop mode)

```bash
maturin develop --release
```

This compiles the Rust extension and installs `orion_sdr` into `.venv` in editable
mode — re-run it after any change to `src/python/`.

### Build a distributable wheel

```bash
maturin build --release
pip install target/wheels/orion_sdr-*.whl
```

Both paths produce an `orion_sdr` package importable as `import orion_sdr`.

### VS Code setup

Select `.venv/bin/python` as the interpreter so that Pylance resolves imports and
displays type stub completions:

1. Open the Command Palette (`⇧⌘P`) → **Python: Select Interpreter**
2. Choose **Enter interpreter path…** → `.venv/bin/python`

Or add it directly to your workspace settings:

```json
{
  "python.defaultInterpreterPath": "${workspaceFolder}/.venv/bin/python"
}
```

## Testing

The Python tests live in `python/tests/` and are discovered automatically by pytest
(configured in `pyproject.toml`). The extension must be built before running tests.

```bash
# Build (or rebuild after Rust changes) then run all tests
maturin develop --release && .venv/bin/pytest

# Verbose output
.venv/bin/pytest -v

# Run only unit tests
.venv/bin/pytest python/tests/test_unit.py

# Run only roundtrip SNR tests
.venv/bin/pytest python/tests/test_roundtrip.py

# Run a single test by name
.venv/bin/pytest -k test_roundtrip_fm_quadrature
```

If you have the `.venv` activated (`source .venv/bin/activate`) you can drop the
`.venv/bin/` prefix and use `pytest` directly.

### Test structure

| File | What it covers |
|------|----------------|
| `conftest.py` | Shared helpers: `real_tone`, `complex_tone`, `snr_db`, `tail`, fixtures |
| `test_unit.py` | Output shape/dtype, input validation, setters, instance isolation |
| `test_roundtrip.py` | Mod→demod SNR for CW, AM, SSB, FM, PM; noiseless bit-exact roundtrips for BPSK, QPSK, QAM-16/64/256 |

## Python Source Layout

```text
python/
  orion_sdr/
    __init__.py       ← re-exports everything from the native extension
    __init__.pyi      ← hand-authored PEP 561 type stub (all 16 classes)
    py.typed          ← PEP 561 marker; tells type checkers this package is typed
    orion_sdr.so      ← compiled native extension (built by maturin, not in repo)
  tests/
    __init__.py
    ...               ← pytest tests for the Python interface
```

After `maturin develop` or wheel installation, the `site-packages` layout mirrors
`python/orion_sdr/` above.

## Type Stubs

The package ships a [PEP 561](https://peps.python.org/pep-0561/) inline stub at
`python/orion_sdr/__init__.pyi`. It is hand-authored — update it whenever PyO3
bindings in `src/python/` change (new classes, renamed parameters, added methods).

No extra configuration is required — mypy, pyright, and pylance all pick up the stub
automatically once the package is installed.

Verify with mypy:

```bash
pip install mypy
```

```python
# check_orion_sdr.py
import orion_sdr as sdr, numpy as np
mod = sdr.AmDsbMod(fs=48_000, rf_hz=0.0, carrier_level=1.0, modulation_index=0.8)
audio: np.ndarray = np.zeros(1024, dtype=np.float32)
iq = mod.process(audio)
reveal_type(iq)
```

```bash
$ mypy check_orion_sdr.py
check_orion_sdr.py:5: note: Revealed type is "numpy.ndarray[..., numpy.dtype[numpy.complexfloating[...]]]"
Success: no issues found in 1 source file
```

Verify with pyright / pylance (VS Code): open any `.py` file that imports `orion_sdr`,
hover over a constructor or `process()` call, and the parameter names, types, and
docstring should appear in the tooltip.

## Demodulator Examples

### Analog

```python
import orion_sdr as sdr
import numpy as np

# Synthetic IQ block (replace with real samples from your SDR)
iq = np.zeros(4096, dtype=np.complex64)

# CW envelope demodulator — tracks amplitude of a tone near 700 Hz
cw = sdr.CwEnvelopeDemod(sample_rate=48_000, tone_hz=700, env_bw_hz=300)
cw.set_gain(1.5)
audio = cw.process(iq)   # → float32 ndarray, same length as input

# AM envelope demodulator (PowerSqrt, higher fidelity)
am = sdr.AmEnvelopeDemod(fs=48_000, audio_bw_hz=5_000)
audio = am.process(iq)

# AM envelope demodulator (AbsApprox, slightly faster, small amplitude error)
am_fast = sdr.AmEnvelopeDemod(fs=48_000, audio_bw_hz=5_000, abs_approx=True)
audio = am_fast.process(iq)

# SSB product demodulator — BFO at 0 Hz for a pre-tuned baseband signal
ssb = sdr.SsbProductDemod(fs=48_000, bfo_hz=0.0, audio_bw_hz=2_800)
audio = ssb.process(iq)

# NBFM quadrature demodulator — 2.5 kHz peak deviation, 5 kHz audio BW
fm = sdr.FmQuadratureDemod(fs=48_000, dev_hz=2_500, audio_bw_hz=5_000)
audio = fm.process(iq)

# PM quadrature demodulator — sensitivity k scales phase difference to audio
pm = sdr.PmQuadratureDemod(fs=48_000, k=0.8, audio_bw_hz=5_000)
audio = pm.process(iq)
```

### Digital

```python
import orion_sdr as sdr
import numpy as np

# IQ symbols at baseband, 1 sample per symbol (carrier-removed)
iq = np.zeros(4096, dtype=np.complex64)

# BPSK hard-decision demodulator — 1 bit per symbol → uint8 array of 0/1
bpsk = sdr.BpskDemod(gain=1.0)
bits = bpsk.process(iq)   # → uint8 ndarray, shape (4096,)

# QPSK hard-decision demodulator — 2 bits per symbol → uint8 array of 0/1
qpsk = sdr.QpskDemod(gain=1.0)
bits = qpsk.process(iq)   # → uint8 ndarray, shape (8192,)

# QAM-16 hard-decision demodulator — 4 bits per symbol
qam16 = sdr.QamDemod(order=16, gain=1.0)
bits = qam16.process(iq)   # → uint8 ndarray, shape (16384,)

# QAM-64 — 6 bits per symbol
qam64 = sdr.QamDemod(order=64, gain=1.0)
bits = qam64.process(iq)   # → uint8 ndarray, shape (24576,)

# QAM-256 — 8 bits per symbol
qam256 = sdr.QamDemod(order=256, gain=1.0)
bits = qam256.process(iq)   # → uint8 ndarray, shape (32768,)
```

## Modulator Examples

### Analog

```python
import orion_sdr as sdr
import numpy as np

audio = np.zeros(4096, dtype=np.float32)

# AM DSB modulator — full carrier (A3E), 80% modulation index, baseband output
am_mod = sdr.AmDsbMod(fs=48_000, rf_hz=0.0, carrier_level=1.0, modulation_index=0.8)
am_mod.set_gain(1.0)
am_mod.set_clamp(True)   # clamp modulated envelope to ±1
iq = am_mod.process(audio)   # → complex64 ndarray

# CW keyed modulator — input is a 0..1 keying envelope (not raw audio)
cw_mod = sdr.CwKeyedMod(sample_rate=48_000, tone_hz=700, rise_ms=5.0, fall_ms=5.0)
key = np.ones(4096, dtype=np.float32)   # key-down
iq = cw_mod.process(key)

# FM phase-accumulator modulator — 2.5 kHz deviation, baseband output
fm_mod = sdr.FmPhaseAccumMod(sample_rate=48_000, deviation_hz=2_500, rf_hz=0.0)
fm_mod.set_deviation(5_000)   # change deviation after construction
iq = fm_mod.process(audio)

# PM direct-phase modulator — kp maps ±1 audio to ±kp radians of phase
pm_mod = sdr.PmDirectPhaseMod(sample_rate=48_000, kp_rad_per_unit=0.8, rf_hz=0.0)
pm_mod.set_sensitivity(1.0)
iq = pm_mod.process(audio)

# SSB phasing modulator — USB, audio IF at 1.5 kHz, baseband RF output
ssb_mod = sdr.SsbPhasingMod(fs=48_000, audio_bw_hz=2_800, audio_if_hz=1_500,
                             rf_hz=0.0, usb=True)
iq = ssb_mod.process(audio)
```

### Digital

```python
import orion_sdr as sdr
import numpy as np

# BPSK modulator — 1 bit per symbol; input uint8 array (LSB of each byte)
bpsk_mod = sdr.BpskMod(fs=1.0, rf_hz=0.0, gain=1.0)
bits = np.array([0, 1, 0, 1, 1, 0], dtype=np.uint8)
iq = bpsk_mod.process(bits)   # → complex64, shape (6,): (+1,0),(−1,0),...

# QPSK modulator — 2 bits per symbol; len(bits) must be even
qpsk_mod = sdr.QpskMod(fs=1.0, rf_hz=0.0, gain=1.0)
bits = np.zeros(512, dtype=np.uint8)
iq = qpsk_mod.process(bits)   # → complex64, shape (256,)

# QAM-16 modulator — 4 bits per symbol
qam16_mod = sdr.QamMod(order=16, fs=1.0, rf_hz=0.0, gain=1.0)
bits = np.zeros(1024, dtype=np.uint8)
iq = qam16_mod.process(bits)   # → complex64, shape (256,)

# QAM-64 — 6 bits per symbol
qam64_mod = sdr.QamMod(order=64, fs=1.0, rf_hz=0.0, gain=1.0)
bits = np.zeros(1536, dtype=np.uint8)
iq = qam64_mod.process(bits)   # → complex64, shape (256,)

# QAM-256 — 8 bits per symbol; set rf_hz to upconvert in one step
qam256_mod = sdr.QamMod(order=256, fs=48_000.0, rf_hz=12_000.0, gain=1.0)
bits = np.zeros(2048, dtype=np.uint8)
iq = qam256_mod.process(bits)   # → complex64, shape (256,), centred at 12 kHz
```

## Round-Trip Examples

### Analog (SSB)

```python
import orion_sdr as sdr
import numpy as np

fs = 48_000
n  = 8192

# Generate a 1 kHz test tone
t = np.arange(n, dtype=np.float32) / fs
audio_in = np.sin(2 * np.pi * 1_000 * t)

# Modulate → demodulate (SSB USB)
mod   = sdr.SsbPhasingMod(fs=fs, audio_bw_hz=2_800, audio_if_hz=1_500,
                           rf_hz=0.0, usb=True)
demod = sdr.SsbProductDemod(fs=fs, bfo_hz=0.0, audio_bw_hz=2_800)

iq        = mod.process(audio_in)
audio_out = demod.process(iq)

print(audio_in.shape, audio_out.shape, audio_out.dtype)
# (8192,) (8192,) float32
```

### Digital (BPSK)

```python
import orion_sdr as sdr
import numpy as np

n = 256
bits_in = np.array([(i & 1) for i in range(n)], dtype=np.uint8)

mod   = sdr.BpskMod(fs=1.0, rf_hz=0.0, gain=1.0)
demod = sdr.BpskDemod(gain=1.0)

iq       = mod.process(bits_in)    # → complex64, shape (256,)
bits_out = demod.process(iq)       # → uint8,     shape (256,)

np.testing.assert_array_equal(bits_in, bits_out)
print("BPSK noiseless roundtrip: perfect recovery")
```

## Notes

- All `process()` calls are synchronous and hold the GIL. For high-throughput pipelines,
  call from a dedicated thread or release the GIL at a higher level.
- Each instance is stateful (filter state, NCO phase, AGC level). Create separate
  instances for independent signal paths.
- `rf_hz=0.0` produces baseband (DC-centered) IQ. Set a non-zero value to upconvert
  in one step.
- `CwKeyedMod` expects a keying envelope (0 = key up, 1 = key down), not audio.
  Derive the envelope from PTT state or a CW decoder.
- Digital mod/demod classes operate at **1 sample per symbol** with no pulse shaping.
  Insert a matched filter or interpolator between the modulator and channel if needed.
- `QamMod` and `QamDemod` raise `ValueError` for orders other than 16, 64, or 256.
- Bit arrays are `uint8` with one bit per byte (value 0 or 1). The LSB of each byte
  is used; higher bits are ignored by the modulators.
