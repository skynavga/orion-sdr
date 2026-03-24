"""Tests for PSK31 (BPSK31 + QPSK31) Python bindings.

All tests are noiseless / synthetic — no hardware or RF noise required.
"""

import numpy as np
import pytest

import orion_sdr as sdr

FS = 8_000.0
BAUD = 31.25
SPS = int(round(FS / BAUD))  # 256


# ---------------------------------------------------------------------------
# Varicode codec
# ---------------------------------------------------------------------------

class TestVaricode:
    def test_encoder_drain_preamble(self):
        enc = sdr.VaricodeEncoder()
        enc.push_preamble(8)
        bits = enc.drain_bits()
        assert bits.dtype == np.uint8
        assert len(bits) == 8
        assert np.all(bits == 0)

    def test_encoder_drain_byte(self):
        enc = sdr.VaricodeEncoder()
        enc.push_byte(ord('e'))  # 'e' = 0b11 (len=2) in Varicode
        bits = enc.drain_bits()
        assert bits.dtype == np.uint8
        assert len(bits) == 2
        np.testing.assert_array_equal(bits, [1, 1])

    def test_encoder_is_empty(self):
        enc = sdr.VaricodeEncoder()
        assert enc.is_empty()
        enc.push_preamble(4)
        assert not enc.is_empty()
        enc.drain_bits()
        assert enc.is_empty()

    def test_decoder_roundtrip(self):
        enc = sdr.VaricodeEncoder()
        dec = sdr.VaricodeDecoder()

        text = b"HELLO"
        for b in text:
            enc.push_byte(b)
        # Add inter-character gaps to flush the decoder.
        enc.push_postamble(2)
        # Feed a terminating 00 to flush last character.
        enc.push_preamble(2)
        bits = enc.drain_bits()

        dec.push_bits(bits)
        decoded = dec.pop_bytes()
        # Check that 'HELLO' appears in decoded output.
        assert b"HELLO" in decoded or decoded.startswith(b"HELLO"), \
            f"decoded: {decoded!r}"

    def test_decoder_space(self):
        # Space = codeword 0b1 (len=1) followed by "00".
        dec = sdr.VaricodeDecoder()
        bits = np.array([1, 0, 0], dtype=np.uint8)
        dec.push_bits(bits)
        decoded = dec.pop_bytes()
        assert decoded == b' ', f"expected space, got {decoded!r}"


# ---------------------------------------------------------------------------
# BPSK31 modulator
# ---------------------------------------------------------------------------

class TestBpsk31Mod:
    def test_output_shape(self):
        mod = sdr.Bpsk31Mod(FS, 0.0, 1.0)
        bits = np.zeros(10, dtype=np.uint8)
        iq = mod.modulate_bits(bits)
        assert iq.dtype == np.complex64
        assert len(iq) == 10 * SPS

    def test_baseband_im_small(self):
        # At rf_hz=0 and all 1-bits (no phase changes) the imaginary part should be ~0.
        mod = sdr.Bpsk31Mod(FS, 0.0, 1.0)
        bits = np.ones(8, dtype=np.uint8)
        iq = mod.modulate_bits(bits)
        # Skip the first and last symbol (edge effects from Hann window).
        mid = iq[SPS:7*SPS]
        assert np.max(np.abs(mid.imag)) < 0.05, \
            f"Im too large: {np.max(np.abs(mid.imag))}"

    def test_modulate_text_shape(self):
        mod = sdr.Bpsk31Mod(FS, 0.0, 1.0)
        iq = mod.modulate_text(b"HI", preamble_bits=8, postamble_bits=8)
        assert iq.dtype == np.complex64
        assert len(iq) > 0

    def test_reset(self):
        mod = sdr.Bpsk31Mod(FS, 0.0, 1.0)
        mod.set_gain(2.0)
        mod.reset()
        bits = np.ones(4, dtype=np.uint8)
        iq = mod.modulate_bits(bits)
        # After reset, with gain=2.0, the amplitude should be ~2.0 in steady state.
        mid = iq[SPS:3*SPS]
        amp = np.abs(mid).mean()
        assert 1.8 < amp < 2.2, f"amplitude after reset: {amp}"


# ---------------------------------------------------------------------------
# BPSK31 demodulator
# ---------------------------------------------------------------------------

class TestBpsk31Demod:
    def test_noiseless_roundtrip(self):
        bits_in = np.array([(i & 1) for i in range(64)], dtype=np.uint8)

        mod = sdr.Bpsk31Mod(FS, 0.0, 1.0)
        iq = mod.modulate_bits(bits_in)

        demod = sdr.Bpsk31Demod(FS, 0.0, 1.0)
        soft = demod.process(iq)

        assert soft.dtype == np.float32
        assert len(soft) == len(bits_in)

        # Hard decision: positive → 1, negative → 0.
        hard = (soft >= 0).astype(np.uint8)

        # Skip first bit (differential startup transient).
        errors = np.sum(hard[1:] != bits_in[1:])
        assert errors == 0, f"BER non-zero: {errors} errors in {len(bits_in)-1} bits"

    def test_output_length(self):
        mod = sdr.Bpsk31Mod(FS, 0.0, 1.0)
        iq = mod.modulate_bits(np.zeros(20, dtype=np.uint8))
        demod = sdr.Bpsk31Demod(FS, 0.0, 1.0)
        soft = demod.process(iq)
        assert len(soft) == 20


# ---------------------------------------------------------------------------
# QPSK31 modulator
# ---------------------------------------------------------------------------

class TestQpsk31Mod:
    def test_output_shape(self):
        mod = sdr.Qpsk31Mod(FS, 0.0, 1.0)
        bits = np.zeros(16, dtype=np.uint8)
        iq = mod.modulate_bits(bits)
        assert iq.dtype == np.complex64
        # Each input bit → 2 coded bits → 1 QPSK symbol → sps samples.
        assert len(iq) == 16 * SPS

    def test_modulate_text(self):
        mod = sdr.Qpsk31Mod(FS, 0.0, 1.0)
        iq = mod.modulate_text(b"HI", preamble_bits=8, postamble_bits=8)
        assert iq.dtype == np.complex64
        assert len(iq) > 0

    def test_nonzero_power(self):
        mod = sdr.Qpsk31Mod(FS, 0.0, 1.0)
        iq = mod.modulate_bits(np.zeros(32, dtype=np.uint8))
        power = np.mean(np.abs(iq) ** 2)
        assert power > 0.01, f"expected nonzero power, got {power}"


# ---------------------------------------------------------------------------
# QPSK31 demodulator
# ---------------------------------------------------------------------------

class TestQpsk31Demod:
    def test_noiseless_roundtrip(self):
        bits_in = np.array([(i & 1) for i in range(32)], dtype=np.uint8)

        mod = sdr.Qpsk31Mod(FS, 0.0, 1.0)
        iq = mod.modulate_bits(bits_in)

        demod = sdr.Qpsk31Demod(FS, 0.0, 1.0)
        # Process returns soft dibits.
        soft = demod.process(iq)
        assert soft.dtype == np.float32

        # flush() runs Viterbi and returns decoded bits.
        bits_out = demod.flush()
        assert bits_out.dtype == np.uint8
        assert len(bits_out) == len(bits_in)

        skip = 5  # Viterbi startup
        errors = np.sum(bits_out[skip:] != bits_in[skip:])
        assert errors == 0, f"BER non-zero: {errors} errors"

    def test_reset_clears_state(self):
        bits_in = np.ones(16, dtype=np.uint8)
        mod = sdr.Qpsk31Mod(FS, 0.0, 1.0)
        iq = mod.modulate_bits(bits_in)

        demod = sdr.Qpsk31Demod(FS, 0.0, 1.0)
        demod.process(iq)
        demod.reset()
        # After reset, flush should return nothing.
        bits_out = demod.flush()
        assert len(bits_out) == 0


# ---------------------------------------------------------------------------
# psk31_sync
# ---------------------------------------------------------------------------

class TestPsk31Sync:
    def test_returns_list(self):
        noise = np.zeros(int(FS * 1.0), dtype=np.complex64)
        results = sdr.psk31_sync(noise, FS, 500.0, 1500.0)
        assert isinstance(results, list)

    def test_carrier_detection(self):
        # Use a carrier that aligns exactly with a waterfall bin (base_hz + k * PSK31_BAUD).
        # base_hz=900.0, bin 3 → 900.0 + 3 * 31.25 = 993.75 Hz.
        carrier_hz = 900.0 + 3 * 31.25  # 993.75 Hz — exact bin alignment
        mod = sdr.Bpsk31Mod(FS, carrier_hz, 1.0)
        text = b"CQ CQ CQ DE TEST"
        iq_sig = mod.modulate_text(text, preamble_bits=64, postamble_bits=32)

        # Size the buffer to hold the full signal (signal is ~7s at 31.25 baud).
        total = max(int(FS * 8.0), len(iq_sig))
        buf = np.zeros(total, dtype=np.complex64)
        buf[:len(iq_sig)] = iq_sig

        results = sdr.psk31_sync(buf, FS, 900.0, 1100.0,
                                  min_carrier_syms=4, peak_margin_db=3.0,
                                  n_bits=256, max_cand=5)

        assert len(results) > 0, "sync should detect the BPSK31 carrier"
        best = results[0]
        assert abs(best["carrier_hz"] - carrier_hz) < 40.0, \
            f"carrier_hz {best['carrier_hz']} too far from {carrier_hz}"

    def test_result_keys(self):
        carrier_hz = 1000.0
        mod = sdr.Bpsk31Mod(FS, carrier_hz, 1.0)
        iq_sig = mod.modulate_text(b"TEST", preamble_bits=64, postamble_bits=32)

        total = int(FS * 3.0)
        buf = np.zeros(total, dtype=np.complex64)
        offset = int(FS * 0.25)
        end = min(offset + len(iq_sig), total)
        buf[offset:end] = iq_sig[:end - offset]

        results = sdr.psk31_sync(buf, FS, 900.0, 1100.0,
                                  min_carrier_syms=4, peak_margin_db=3.0,
                                  n_bits=128, max_cand=3)
        if results:
            r = results[0]
            assert "time_sym"   in r
            assert "freq_bin"   in r
            assert "carrier_hz" in r
            assert "score"      in r
            assert "soft_bits"  in r
            assert r["soft_bits"].dtype == np.float32

    def test_soft_bits_shape(self):
        carrier_hz = 1000.0
        n_bits = 128
        mod = sdr.Bpsk31Mod(FS, carrier_hz, 1.0)
        iq_sig = mod.modulate_text(b"TESTING 123", preamble_bits=64, postamble_bits=32)

        total = int(FS * 3.0)
        buf = np.zeros(total, dtype=np.complex64)
        offset = int(FS * 0.25)
        end = min(offset + len(iq_sig), total)
        buf[offset:end] = iq_sig[:end - offset]

        results = sdr.psk31_sync(buf, FS, 900.0, 1100.0,
                                  min_carrier_syms=4, peak_margin_db=3.0,
                                  n_bits=n_bits, max_cand=3)
        if results:
            assert len(results[0]["soft_bits"]) <= n_bits
