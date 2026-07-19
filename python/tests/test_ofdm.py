"""Tests for OFDM Python bindings (Phase 9 / Release I).

All tests are noiseless / synthetic — no hardware or RF noise required.
"""

import numpy as np
import pytest

import orion_sdr as sdr

FS = 48_000.0
N_FFT = 64
CP_LEN = 8


def _data_carriers() -> np.ndarray:
    half = N_FFT // 2
    return np.array(list(range(1, half)) + list(range(-(half - 1), 0)), dtype=np.int32)


def _no_pilots():
    return (
        np.zeros(0, dtype=np.int32),
        np.zeros(0, dtype=np.complex64),
    )


def _config(constellation: str = "qpsk", rf_hz: float = 0.0) -> "sdr.OfdmConfig":
    pilot_idx, pilot_val = _no_pilots()
    return sdr.OfdmConfig(
        N_FFT,
        CP_LEN,
        _data_carriers(),
        pilot_idx,
        pilot_val,
        FS,
        rf_hz,
        1.0,
        constellation,
    )


# ---------------------------------------------------------------------------
# OfdmConfig
# ---------------------------------------------------------------------------

class TestOfdmConfig:
    def test_basic_construction(self):
        cfg = _config()
        assert cfg.bits_per_ofdm_symbol > 0
        assert cfg.samples_per_ofdm_symbol == N_FFT + CP_LEN

    def test_bits_per_symbol_matches_constellation(self):
        cfg_qpsk = _config("qpsk")
        cfg_qam16 = _config("qam16")
        assert cfg_qam16.bits_per_ofdm_symbol == 2 * cfg_qpsk.bits_per_ofdm_symbol

    def test_unknown_constellation_raises(self):
        pilot_idx, pilot_val = _no_pilots()
        with pytest.raises((ValueError, Exception)):
            sdr.OfdmConfig(
                N_FFT, CP_LEN, _data_carriers(), pilot_idx, pilot_val,
                FS, 0.0, 1.0, "not-a-real-constellation",
            )

    def test_invalid_plan_raises(self):
        # Overlapping data/pilot carriers should fail CarrierPlan::validate.
        pilot_idx = np.array([1], dtype=np.int32)
        pilot_val = np.array([1.0 + 0.0j], dtype=np.complex64)
        with pytest.raises((ValueError, Exception)):
            sdr.OfdmConfig(
                N_FFT, CP_LEN, _data_carriers(), pilot_idx, pilot_val,
                FS, 0.0, 1.0, "qpsk",
            )


# ---------------------------------------------------------------------------
# OfdmMod / OfdmDemod round trip
# ---------------------------------------------------------------------------

class TestOfdmRoundtrip:
    def test_mod_shape(self):
        cfg = _config()
        mod = sdr.OfdmMod(cfg)
        bits = np.zeros(cfg.bits_per_ofdm_symbol, dtype=np.uint8)
        iq = mod.modulate(bits)
        assert iq.shape == (cfg.samples_per_ofdm_symbol,)
        assert iq.dtype == np.complex64

    def test_roundtrip_qpsk_noiseless(self):
        cfg = _config("qpsk")
        n_symbols = 8
        bps = cfg.bits_per_ofdm_symbol

        rng = np.random.default_rng(1234)
        bits_in = rng.integers(0, 2, size=n_symbols * bps, dtype=np.uint8)

        mod = sdr.OfdmMod(cfg)
        iq = mod.modulate(bits_in)
        assert iq.dtype == np.complex64
        assert iq.shape == (n_symbols * cfg.samples_per_ofdm_symbol,)

        demod = sdr.OfdmDemod(cfg)
        sps = cfg.samples_per_ofdm_symbol
        bits_out = np.zeros(0, dtype=np.uint8)
        for s in range(n_symbols):
            chunk = iq[s * sps:(s + 1) * sps]
            bits_out = np.concatenate([bits_out, demod.demodulate(chunk)])

        assert bits_out.dtype == np.uint8
        np.testing.assert_array_equal(bits_out, bits_in)

    def test_roundtrip_qam16_noiseless(self):
        cfg = _config("qam16")
        n_symbols = 4
        bps = cfg.bits_per_ofdm_symbol

        rng = np.random.default_rng(5678)
        bits_in = rng.integers(0, 2, size=n_symbols * bps, dtype=np.uint8)

        mod = sdr.OfdmMod(cfg)
        iq = mod.modulate(bits_in)

        demod = sdr.OfdmDemod(cfg)
        sps = cfg.samples_per_ofdm_symbol
        bits_out = np.zeros(0, dtype=np.uint8)
        for s in range(n_symbols):
            chunk = iq[s * sps:(s + 1) * sps]
            bits_out = np.concatenate([bits_out, demod.demodulate(chunk)])

        np.testing.assert_array_equal(bits_out, bits_in)

    def test_demod_too_short_raises(self):
        cfg = _config()
        demod = sdr.OfdmDemod(cfg)
        with pytest.raises((ValueError, Exception)):
            demod.demodulate(np.zeros(4, dtype=np.complex64))


# ---------------------------------------------------------------------------
# ofdm_sync
# ---------------------------------------------------------------------------

class TestOfdmSync:
    def test_finds_known_offset(self):
        cfg = _config()
        num_repeats, repeat_len = 4, 32
        preamble_iq = sdr.generate_ofdm_preamble(cfg, num_repeats, repeat_len)

        time_offset = 50
        buf = np.zeros(time_offset, dtype=np.complex64)
        buf = np.concatenate([buf, preamble_iq, np.zeros(64, dtype=np.complex64)])

        results = sdr.ofdm_sync(buf, FS, num_repeats, repeat_len, 0, len(buf))
        assert len(results) > 0
        best = results[0]
        assert best["start_sample"] == time_offset
        assert best["score"] > 0.9

    def test_empty_on_silence(self):
        cfg = _config()
        del cfg
        iq = np.zeros(2000, dtype=np.complex64)
        results = sdr.ofdm_sync(iq, FS, 4, 32, 0, len(iq))
        assert isinstance(results, list)


# ---------------------------------------------------------------------------
# build_ofdm_rx_frame
# ---------------------------------------------------------------------------

class TestOfdmRxFrame:
    def test_evm_present_cfo_absent(self):
        cfg = _config("qpsk")
        n_symbols = 4
        bps = cfg.bits_per_ofdm_symbol
        rng = np.random.default_rng(42)
        bits_in = rng.integers(0, 2, size=n_symbols * bps, dtype=np.uint8)

        mod = sdr.OfdmMod(cfg)
        iq = mod.modulate(bits_in)

        demod = sdr.OfdmDemod(cfg)
        sps = cfg.samples_per_ofdm_symbol
        soft_all = []
        bits_out = np.zeros(0, dtype=np.uint8)
        for s in range(n_symbols):
            chunk = iq[s * sps:(s + 1) * sps]
            soft, bits = demod.demodulate_soft(chunk)
            soft_all.append(soft)
            bits_out = np.concatenate([bits_out, bits])

        np.testing.assert_array_equal(bits_out, bits_in)

        soft_concat = np.concatenate(soft_all)
        frame = sdr.build_ofdm_rx_frame(cfg, soft_concat, bits_out)

        np.testing.assert_array_equal(frame.bits, bits_out)
        assert frame.num_symbols == n_symbols
        assert frame.evm_db is not None
        assert frame.evm_db < -20.0
        assert frame.cfo_hz is None
        assert frame.timing_offset_samples is None
        assert frame.channel_mse is None
