# Python Bindings

The complete orion-sdr stack is available as a native Python extension via PyO3: analog
modes (CW, AM, SSB, FM, PM), digital modes (BPSK, QPSK, QAM-16/64/256), and the full
FT8/FT4 pipeline (waveform, codec, sync, message packing) — 28 classes and functions in
total.

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
| ---- | -------------- |
| `conftest.py` | Shared helpers: `real_tone`, `complex_tone`, `snr_db`, `tail`, fixtures |
| `test_unit.py` | Output shape/dtype, input validation, setters, instance isolation |
| `test_roundtrip.py` | Mod→demod SNR for CW, AM, SSB, FM, PM; noiseless bit-exact roundtrips for BPSK, QPSK, QAM-16/64/256 |
| `test_ft8.py` | FT8/FT4 waveform shape/roundtrip, codec encode/decode, sync detection, message roundtrips, full-stack test |

## Python Source Layout

```text
python/
  orion_sdr/
    __init__.py       ← re-exports everything from the native extension
    __init__.pyi      ← hand-authored PEP 561 type stub (all 28 classes/functions)
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

## FT8 / FT4

The full FT8 and FT4 encode/decode pipeline is exposed at four levels:
waveform, codec, frame sync, and message packing.

### Waveform

```python
import orion_sdr as sdr
import numpy as np

# FT8 modulator — input: 58 tone indices (uint8, 0–7)
mod8 = sdr.Ft8Mod(fs=12_000, base_hz=1_000.0, rf_hz=0.0, gain=1.0)
tones = np.zeros(58, dtype=np.uint8)
iq = mod8.modulate(tones)   # → complex64, shape (151_680,)

# FT8 demodulator — input: at least 151 680 samples
demod8 = sdr.Ft8Demod(fs=12_000, base_hz=1_000.0)
tones_out = demod8.demodulate(iq)   # → uint8, shape (58,)

# FT4 modulator — input: 87 tone indices (uint8, 0–3)
mod4 = sdr.Ft4Mod(fs=12_000, base_hz=1_000.0, rf_hz=0.0, gain=1.0)
tones4 = np.zeros(87, dtype=np.uint8)
iq4 = mod4.modulate(tones4)   # → complex64, shape (60_480,)

# FT4 demodulator
demod4 = sdr.Ft4Demod(fs=12_000, base_hz=1_000.0)
tones4_out = demod4.demodulate(iq4)   # → uint8, shape (87,)
```

### Codec

`Ft8Codec` and `Ft4Codec` are stateless; all methods are static.

```python
import orion_sdr as sdr

payload = bytes(10)   # 77-bit payload in 10 bytes

# Encode payload → tone indices
tones = sdr.Ft8Codec.encode(payload)        # → uint8[58]

# Hard-decision decode (no noise tolerance)
result = sdr.Ft8Codec.decode_hard(tones)    # → bytes[10] or None

# Soft-decision decode (use after ft8_sync for noise resilience)
llr = ...                                   # float32[174] from ft8_sync
result = sdr.Ft8Codec.decode_soft(llr)      # → bytes[10] or None

# FT4 — same interface
tones4 = sdr.Ft4Codec.encode(payload)       # → uint8[87]
result4 = sdr.Ft4Codec.decode_hard(tones4)
```

### Frame sync

`ft8_sync` searches an IQ buffer for FT8 frame candidates and returns
soft LLRs ready for `Ft8Codec.decode_soft`.

```python
import orion_sdr as sdr
import numpy as np

iq = np.zeros(151_680, dtype=np.complex64)   # your received buffer

candidates = sdr.ft8_sync(
    iq,
    12_000.0,   # fs
    950.0,      # base_hz — search starts here
    1_100.0,    # max_hz  — search ends here
    0,          # t_min (symbol offset)
    0,          # t_max  (0 = end of buffer)
    5,          # max_cand — return up to 5 best hits
)

for c in candidates:
    print(c["time_sym"], c["freq_bin"], c["score"])
    result = sdr.Ft8Codec.decode_soft(c["llr"])   # llr: float32[174]
    if result is not None:
        print("decoded:", result.hex())
```

Use `ft4_sync` for FT4; the signature and return format are identical.

### Message packing

```python
import orion_sdr as sdr

# Pack a standard QSO message (callsign + callsign + grid/report/token)
payload = sdr.ft8_pack_standard("KD9ABC", "W9XYZ", "FN31")   # → bytes[10]
payload = sdr.ft8_pack_standard("CQ", "W9XYZ", "+07")
payload = sdr.ft8_pack_standard("KD9ABC", "W9XYZ", "RR73")

# Pack free text (up to 13 chars, base-42 alphabet)
payload = sdr.ft8_pack_free_text("CQ DX")                    # → bytes[10]

# Pack telemetry (arbitrary 9 bytes → 71 bits)
payload = sdr.ft8_pack_telemetry(b'\x12\x34\x56\x78\x9a\xbc\xde\xf0\x11')

# Unpack any payload
msg = sdr.ft8_unpack(payload)
# Standard:   {"type": "standard", "call_to": "KD9ABC", "call_de": "W9XYZ", "extra": "FN31"}
# FreeText:   {"type": "free_text", "text": "CQ DX"}
# Telemetry:  {"type": "telemetry", "data": b'\x12\x34...'}
```

### Full-stack example

```python
import orion_sdr as sdr
import numpy as np

FS = 12_000.0
BASE_HZ = 1_000.0

# 1. Pack message
payload = sdr.ft8_pack_standard("KD9ABC", "W9XYZ", "FN31")

# 2. Encode to tone indices
tones = sdr.Ft8Codec.encode(payload)

# 3. Modulate to IQ
mod = sdr.Ft8Mod(FS, BASE_HZ, rf_hz=0.0, gain=1.0)
iq = mod.modulate(tones)

# 4. (transmit / receive / add noise here)

# 5. Sync search
candidates = sdr.ft8_sync(iq, FS, BASE_HZ - 50, BASE_HZ + 100, 0, 0, 5)

# 6. Soft decode
if candidates:
    payload_out = sdr.Ft8Codec.decode_soft(candidates[0]["llr"])

# 7. Unpack
if payload_out:
    msg = sdr.ft8_unpack(payload_out)
    print(msg["call_to"], msg["call_de"], msg["extra"])
    # → KD9ABC W9XYZ FN31
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
