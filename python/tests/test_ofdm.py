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
# OfdmDemod equalizer paths (fused CP-remove + FFT + equalize + extract +
# decide). The pure-Rust OfdmDemod has no equalizer; PyOfdmDemod fuses one,
# so these paths are only reachable — and only testable — from Python.
# ---------------------------------------------------------------------------

def _apply_fir_channel(iq: np.ndarray, taps: np.ndarray) -> np.ndarray:
    """Causal FIR channel; taps[0] is the direct path. Delay spread
    (len(taps)-1) must stay within cp_len for the per-bin equalizer to hold."""
    out = np.convolve(iq, taps)[: len(iq)]
    return out.astype(np.complex64)


class TestOfdmEqualizer:
    def test_unknown_equalizer_raises(self):
        cfg = _config()
        with pytest.raises((ValueError, Exception)):
            sdr.OfdmDemod(cfg, "not-a-real-equalizer")

    def test_pilot_interp_constructs_and_roundtrips_flat(self):
        # The "pilot_interp" branch is never selected from Python elsewhere.
        # On a flat channel it must still round-trip (with a couple of pilots
        # present so interpolation has anchors).
        pilot_idx = np.array([10, -10], dtype=np.int32)
        pilot_val = np.array([1.0 + 0.0j, 1.0 + 0.0j], dtype=np.complex64)
        # Data carriers excluding the pilot bins.
        half = N_FFT // 2
        data = [c for c in list(range(1, half)) + list(range(-(half - 1), 0))
                if c not in (10, -10)]
        cfg = sdr.OfdmConfig(
            N_FFT, CP_LEN, np.array(data, dtype=np.int32),
            pilot_idx, pilot_val, FS, 0.0, 1.0, "qpsk",
        )

        n_symbols = 6
        bps = cfg.bits_per_ofdm_symbol
        rng = np.random.default_rng(2024)
        bits_in = rng.integers(0, 2, size=n_symbols * bps, dtype=np.uint8)

        mod = sdr.OfdmMod(cfg)
        iq = mod.modulate(bits_in)

        demod = sdr.OfdmDemod(cfg, "pilot_interp")
        sps = cfg.samples_per_ofdm_symbol
        bits_out = np.zeros(0, dtype=np.uint8)
        for s in range(n_symbols):
            chunk = iq[s * sps:(s + 1) * sps]
            bits_out = np.concatenate([bits_out, demod.demodulate(chunk)])

        np.testing.assert_array_equal(bits_out, bits_in)

    def test_training_symbol_estimate_channel_multipath(self):
        # End-to-end fused equalized demod over a real frequency-selective
        # channel: generate a preamble WITH a training symbol, pass data +
        # training through a 2-tap FIR channel, estimate the channel from the
        # received training symbol, then demodulate the data and recover bits.
        # QAM-16 so the equalizer is genuinely load-bearing (see
        # test_estimate_channel_is_load_bearing).
        cfg = _config("qam16")
        num_repeats, repeat_len = 4, 32
        sps = cfg.samples_per_ofdm_symbol

        n_symbols = 6
        bps = cfg.bits_per_ofdm_symbol
        rng = np.random.default_rng(99)
        bits_in = rng.integers(0, 2, size=n_symbols * bps, dtype=np.uint8)

        mod = sdr.OfdmMod(cfg)
        data_iq = mod.modulate(bits_in)
        preamble_iq = sdr.generate_ofdm_preamble(
            cfg, num_repeats, repeat_len, N_FFT, CP_LEN
        )

        # Known preamble layout: [num_repeats*repeat_len S&C samples][training
        # symbol of sps samples]. Concatenate preamble + data, then channelize.
        clean = np.concatenate([preamble_iq, data_iq]).astype(np.complex64)
        taps = np.array([0.8 + 0.1j, 0.0 + 0.0j, 0.25 - 0.15j], dtype=np.complex64)
        assert len(taps) - 1 <= CP_LEN
        channeled = _apply_fir_channel(clean, taps)

        training_start = num_repeats * repeat_len
        training_iq = channeled[training_start:training_start + sps]

        demod = sdr.OfdmDemod(cfg, "training_symbol")
        demod.estimate_channel(training_iq)

        data_start = training_start + sps
        bits_out = np.zeros(0, dtype=np.uint8)
        for s in range(n_symbols):
            chunk = channeled[data_start + s * sps:data_start + (s + 1) * sps]
            bits_out = np.concatenate([bits_out, demod.demodulate(chunk)])

        np.testing.assert_array_equal(bits_out, bits_in)

    def test_estimate_channel_is_load_bearing(self):
        # Without estimate_channel(), the held estimate is identity, so a
        # frequency-selective channel corrupts the bits; with it, they are
        # recovered exactly. This proves estimate_channel is not a silent
        # no-op. Uses QAM-16 (dense enough that per-bin gain/phase errors
        # cross decision boundaries — QPSK's 45deg margin is too forgiving of
        # this mild a channel to make a reliable "must corrupt" assertion).
        cfg = _config("qam16")
        num_repeats, repeat_len = 4, 32
        sps = cfg.samples_per_ofdm_symbol
        bps = cfg.bits_per_ofdm_symbol
        rng = np.random.default_rng(7)
        bits_in = rng.integers(0, 2, size=6 * bps, dtype=np.uint8)

        mod = sdr.OfdmMod(cfg)
        data_iq = mod.modulate(bits_in)
        preamble_iq = sdr.generate_ofdm_preamble(
            cfg, num_repeats, repeat_len, N_FFT, CP_LEN
        )
        clean = np.concatenate([preamble_iq, data_iq]).astype(np.complex64)
        taps = np.array([0.8 + 0.1j, 0.0 + 0.0j, 0.25 - 0.15j], dtype=np.complex64)
        channeled = _apply_fir_channel(clean, taps)

        training_start = num_repeats * repeat_len
        training_iq = channeled[training_start:training_start + sps]
        data_start = training_start + sps

        def demod_bits(estimate: bool) -> np.ndarray:
            demod = sdr.OfdmDemod(cfg, "training_symbol")
            if estimate:
                demod.estimate_channel(training_iq)
            out = np.zeros(0, dtype=np.uint8)
            for s in range(6):
                chunk = channeled[data_start + s * sps:data_start + (s + 1) * sps]
                out = np.concatenate([out, demod.demodulate(chunk)])
            return out

        errors_no_est = int(np.count_nonzero(demod_bits(False) != bits_in))
        assert errors_no_est > 0, "multipath with no channel estimate should corrupt bits"

        # With the estimate, the same channel is fully corrected.
        np.testing.assert_array_equal(demod_bits(True), bits_in)


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
