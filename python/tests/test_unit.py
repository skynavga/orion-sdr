"""Unit tests for the orion_sdr Python extension.

Covers:
- Output shape and dtype for every demodulator and modulator
- Stateful behaviour (instances are independent)
- Input validation (wrong dtype, wrong ndim, non-contiguous)
- Setter methods (set_gain, set_deviation, set_sensitivity, set_clamp)
"""

import numpy as np
import pytest

import orion_sdr as sdr
from .conftest import FS, complex_tone, real_tone


N = 4096  # block size used throughout


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def iq_block(n: int = N) -> np.ndarray:
    return np.zeros(n, dtype=np.complex64)


def audio_block(n: int = N) -> np.ndarray:
    return np.zeros(n, dtype=np.float32)


# ---------------------------------------------------------------------------
# Output shape and dtype — demodulators (IQ → audio)
# ---------------------------------------------------------------------------

class TestDemodOutputShape:
    def test_cw(self):
        out = sdr.CwEnvelopeDemod(FS, 700, 300).process(iq_block())
        assert out.shape == (N,) and out.dtype == np.float32

    def test_am(self):
        out = sdr.AmEnvelopeDemod(FS, 5_000).process(iq_block())
        assert out.shape == (N,) and out.dtype == np.float32

    def test_am_abs_approx(self):
        out = sdr.AmEnvelopeDemod(FS, 5_000, abs_approx=True).process(iq_block())
        assert out.shape == (N,) and out.dtype == np.float32

    def test_ssb(self):
        out = sdr.SsbProductDemod(FS, 0.0, 2_800).process(iq_block())
        assert out.shape == (N,) and out.dtype == np.float32

    def test_fm(self):
        out = sdr.FmQuadratureDemod(FS, 2_500, 5_000).process(iq_block())
        assert out.shape == (N,) and out.dtype == np.float32

    def test_pm(self):
        out = sdr.PmQuadratureDemod(FS, 0.8, 5_000).process(iq_block())
        assert out.shape == (N,) and out.dtype == np.float32


# ---------------------------------------------------------------------------
# Output shape and dtype — modulators (audio → IQ)
# ---------------------------------------------------------------------------

class TestModOutputShape:
    def test_cw(self):
        out = sdr.CwKeyedMod(FS, 700, 5.0, 5.0).process(audio_block())
        assert out.shape == (N,) and out.dtype == np.complex64

    def test_am(self):
        out = sdr.AmDsbMod(FS, 0.0, 1.0, 0.8).process(audio_block())
        assert out.shape == (N,) and out.dtype == np.complex64

    def test_fm(self):
        out = sdr.FmPhaseAccumMod(FS, 2_500, 0.0).process(audio_block())
        assert out.shape == (N,) and out.dtype == np.complex64

    def test_pm(self):
        out = sdr.PmDirectPhaseMod(FS, 0.8, 0.0).process(audio_block())
        assert out.shape == (N,) and out.dtype == np.complex64

    def test_ssb(self):
        out = sdr.SsbPhasingMod(FS, 2_800, 1_500, 0.0, True).process(audio_block())
        assert out.shape == (N,) and out.dtype == np.complex64


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

class TestInputValidation:
    """process() should raise ValueError for bad input arrays."""

    # wrong dtype
    def test_demod_wrong_dtype(self):
        dem = sdr.CwEnvelopeDemod(FS, 700, 300)
        with pytest.raises((ValueError, TypeError)):
            dem.process(np.zeros(N, dtype=np.complex128))

    def test_mod_wrong_dtype(self):
        mod = sdr.AmDsbMod(FS, 0.0, 1.0, 0.8)
        with pytest.raises((ValueError, TypeError)):
            mod.process(np.zeros(N, dtype=np.float64))

    # wrong ndim
    def test_demod_2d_input(self):
        dem = sdr.FmQuadratureDemod(FS, 2_500, 5_000)
        with pytest.raises((ValueError, TypeError)):
            dem.process(np.zeros((N, 1), dtype=np.complex64))

    def test_mod_2d_input(self):
        mod = sdr.FmPhaseAccumMod(FS, 2_500, 0.0)
        with pytest.raises((ValueError, TypeError)):
            mod.process(np.zeros((N, 1), dtype=np.float32))

    # non-contiguous
    def test_demod_non_contiguous(self):
        dem = sdr.SsbProductDemod(FS, 0.0, 2_800)
        strided = np.zeros(N * 2, dtype=np.complex64)[::2]  # non-C-contiguous
        with pytest.raises((ValueError, TypeError)):
            dem.process(strided)

    def test_mod_non_contiguous(self):
        mod = sdr.CwKeyedMod(FS, 700, 5.0, 5.0)
        strided = np.zeros(N * 2, dtype=np.float32)[::2]
        with pytest.raises((ValueError, TypeError)):
            mod.process(strided)


# ---------------------------------------------------------------------------
# Setter methods
# ---------------------------------------------------------------------------

class TestSetters:
    def test_cw_demod_set_gain(self):
        dem = sdr.CwEnvelopeDemod(FS, 700, 300)
        iq = complex_tone(FS, 700, N)
        out_default = dem.process(iq.copy())
        dem2 = sdr.CwEnvelopeDemod(FS, 700, 300)
        dem2.set_gain(2.0)
        out_boosted = dem2.process(iq.copy())
        # boosted output should have higher RMS
        assert np.sqrt(np.mean(out_boosted**2)) > np.sqrt(np.mean(out_default**2))

    def test_am_mod_set_gain(self):
        mod = sdr.AmDsbMod(FS, 0.0, 1.0, 0.8)
        audio = real_tone(FS, 1_000, N, 0.5)
        out1 = mod.process(audio.copy())
        mod2 = sdr.AmDsbMod(FS, 0.0, 1.0, 0.8)
        mod2.set_gain(2.0)
        out2 = mod2.process(audio.copy())
        rms1 = float(np.sqrt(np.mean(np.abs(out1)**2)))
        rms2 = float(np.sqrt(np.mean(np.abs(out2)**2)))
        assert rms2 > rms1

    def test_fm_mod_set_deviation(self):
        mod = sdr.FmPhaseAccumMod(FS, 2_500, 0.0)
        mod.set_deviation(5_000)  # should not raise

    def test_pm_mod_set_sensitivity(self):
        mod = sdr.PmDirectPhaseMod(FS, 0.8, 0.0)
        mod.set_sensitivity(1.0)  # should not raise

    def test_am_mod_set_clamp(self):
        mod = sdr.AmDsbMod(FS, 0.0, 1.0, 0.8)
        mod.set_clamp(True)   # should not raise
        mod.set_clamp(False)


# ---------------------------------------------------------------------------
# Independence of instances (state isolation)
# ---------------------------------------------------------------------------

class TestInstanceIsolation:
    def test_two_fm_demod_instances_are_independent(self):
        iq = complex_tone(FS, 1_000, N)
        d1 = sdr.FmQuadratureDemod(FS, 2_500, 5_000)
        d2 = sdr.FmQuadratureDemod(FS, 2_500, 5_000)
        # process the same block through both; results must be equal
        out1 = d1.process(iq.copy())
        out2 = d2.process(iq.copy())
        np.testing.assert_array_equal(out1, out2)

    def test_second_process_call_differs_from_first(self):
        """Stateful filter: second call on silence after a tone should differ."""
        dem = sdr.CwEnvelopeDemod(FS, 700, 300)
        iq_tone = complex_tone(FS, 700, N)
        iq_zero = np.zeros(N, dtype=np.complex64)
        dem.process(iq_tone)          # prime the filter state
        out_after = dem.process(iq_zero)
        # envelope should decay, not be identically zero immediately
        assert np.any(out_after != 0.0)
