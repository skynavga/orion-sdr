# Python Bindings

All 5 demodulators and all 5 modulators are available as a native Python extension via PyO3.

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
| `test_roundtrip.py` | Mod→demod SNR for CW, AM, SSB, FM, PM (thresholds match Rust tests) |

## Python Source Layout

```text
python/
  orion_sdr/
    __init__.py       ← re-exports everything from the native extension
    __init__.pyi      ← hand-authored PEP 561 type stub (all 10 classes)
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

## Modulator Examples

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

## Round-Trip Example

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

## Notes

- All `process()` calls are synchronous and hold the GIL. For high-throughput pipelines,
  call from a dedicated thread or release the GIL at a higher level.
- Each instance is stateful (filter state, NCO phase, AGC level). Create separate
  instances for independent signal paths.
- `rf_hz=0.0` produces baseband (DC-centered) IQ. Set a non-zero value to upconvert
  in one step.
- `CwKeyedMod` expects a keying envelope (0 = key up, 1 = key down), not audio.
  Derive the envelope from PTT state or a CW decoder.
