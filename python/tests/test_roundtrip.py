"""Roundtrip SNR tests for the orion_sdr Python extension.

Each test modulates a tone, demodulates it, and asserts that SNR in the
recovered audio exceeds a minimum threshold. Thresholds match the Rust
roundtrip tests in src/tests/roundtrip.rs.
"""

import numpy as np
import pytest

import orion_sdr as sdr
from .conftest import FS, real_tone, snr_db, tail


N = 32_768  # long enough for reliable SNR estimates after transient


# ---------------------------------------------------------------------------
# CW: CwKeyedMod → CwEnvelopeDemod
# ---------------------------------------------------------------------------

def test_roundtrip_cw_envelope():
    """Recovered keying envelope should have >14 dB ON/OFF contrast."""
    n = 24_000
    pitch_hz = 700.0
    key_f = 5.0  # 5 Hz square wave keying

    key_env = np.array(
        [(1.0 if (k * key_f / FS) % 1.0 < 0.5 else 0.0) for k in range(n)],
        dtype=np.float32,
    )

    iq = sdr.CwKeyedMod(FS, pitch_hz, 3.0, 3.0).process(key_env)
    audio = sdr.CwEnvelopeDemod(FS, pitch_hz, 300.0).process(iq)

    skip = int(0.100 * FS)
    a = audio[skip:]
    k = key_env[skip:]

    on_rms  = float(np.sqrt(np.mean(a[k > 0.5] ** 2)))
    off_rms = float(np.sqrt(np.mean(a[k < 0.5] ** 2))) + 1e-12
    contrast_db = 20.0 * np.log10(on_rms / off_rms)

    assert contrast_db > 14.0, f"CW ON/OFF contrast {contrast_db:.1f} dB < 14 dB"


# ---------------------------------------------------------------------------
# AM: AmDsbMod → AmEnvelopeDemod
# ---------------------------------------------------------------------------

def test_roundtrip_am_envelope():
    f_mod = 1_000.0
    audio_in = real_tone(FS, f_mod, N, 0.5)

    iq = sdr.AmDsbMod(FS, 0.0, 0.8, 0.5).process(audio_in)
    audio_out = sdr.AmEnvelopeDemod(FS, 5_000.0).process(iq)

    snr = snr_db(tail(audio_out), FS, f_mod)
    assert snr > 24.0, f"AM roundtrip SNR {snr:.1f} dB < 24 dB"


def test_roundtrip_am_envelope_abs_approx():
    """AbsApprox variant should also achieve acceptable SNR."""
    f_mod = 1_000.0
    audio_in = real_tone(FS, f_mod, N, 0.5)

    iq = sdr.AmDsbMod(FS, 0.0, 0.8, 0.5).process(audio_in)
    audio_out = sdr.AmEnvelopeDemod(FS, 5_000.0, abs_approx=True).process(iq)

    snr = snr_db(tail(audio_out), FS, f_mod)
    assert snr > 20.0, f"AM AbsApprox roundtrip SNR {snr:.1f} dB < 20 dB"


# ---------------------------------------------------------------------------
# SSB: SsbPhasingMod (USB) → SsbProductDemod
# ---------------------------------------------------------------------------

def test_roundtrip_ssb_usb():
    f_audio = 1_200.0
    audio_bw_hz = 2_800.0
    audio_if_hz = 1_500.0
    audio_in = real_tone(FS, f_audio, N, 0.4)

    iq = sdr.SsbPhasingMod(FS, audio_bw_hz, audio_if_hz, 0.0, True).process(audio_in)
    audio_out = sdr.SsbProductDemod(FS, audio_if_hz, audio_bw_hz).process(iq)

    skip = int(0.120 * FS)
    snr = snr_db(audio_out[skip:], FS, f_audio)
    assert snr > 18.0, f"SSB USB roundtrip SNR {snr:.1f} dB < 18 dB"


# ---------------------------------------------------------------------------
# FM: FmPhaseAccumMod → FmQuadratureDemod
# ---------------------------------------------------------------------------

def test_roundtrip_fm_quadrature():
    f_mod = 1_000.0
    audio_in = real_tone(FS, f_mod, N, 0.5)

    iq = sdr.FmPhaseAccumMod(FS, 2_500.0, 0.0).process(audio_in)
    audio_out = sdr.FmQuadratureDemod(FS, 2_500.0, 5_000.0).process(iq)

    snr = snr_db(tail(audio_out), FS, f_mod)
    assert snr > 20.0, f"FM roundtrip SNR {snr:.1f} dB < 20 dB"


# ---------------------------------------------------------------------------
# PM: PmDirectPhaseMod → PmQuadratureDemod
# ---------------------------------------------------------------------------

def test_roundtrip_pm_quadrature():
    f_mod = 900.0
    audio_in = real_tone(FS, f_mod, N, 0.5)

    iq = sdr.PmDirectPhaseMod(FS, 0.9, 0.0).process(audio_in)
    audio_out = sdr.PmQuadratureDemod(FS, 0.9, 5_000.0).process(iq)

    snr = snr_db(tail(audio_out), FS, f_mod)
    assert snr > 18.0, f"PM roundtrip SNR {snr:.1f} dB < 18 dB"
