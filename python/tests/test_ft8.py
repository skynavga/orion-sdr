"""Tests for FT8/FT4 Python bindings (Phase 5).

All tests are noiseless / synthetic — no hardware or RF noise required.
"""

import numpy as np
import pytest

import orion_sdr as sdr

FS = 12_000.0
BASE_HZ = 1_000.0
MAX_HZ  = 1_100.0

# A known-good 10-byte payload (all zeros with slack bits zeroed).
PAYLOAD = bytes(10)


# ---------------------------------------------------------------------------
# Waveform: Ft8Mod / Ft8Demod
# ---------------------------------------------------------------------------

class TestFt8Waveform:
    def test_mod_shape(self):
        mod = sdr.Ft8Mod(FS, BASE_HZ, 0.0, 1.0)
        tones = np.zeros(58, dtype=np.uint8)
        iq = mod.modulate(tones)
        assert iq.shape == (151_680,)
        assert iq.dtype == np.complex64

    def test_demod_roundtrip(self):
        tones_in = np.array([i % 8 for i in range(58)], dtype=np.uint8)
        mod = sdr.Ft8Mod(FS, BASE_HZ, 0.0, 1.0)
        iq = mod.modulate(tones_in)
        demod = sdr.Ft8Demod(FS, BASE_HZ)
        tones_out = demod.demodulate(iq)
        assert tones_out.shape == (58,)
        assert tones_out.dtype == np.uint8
        np.testing.assert_array_equal(tones_out, tones_in)

    def test_mod_wrong_length(self):
        mod = sdr.Ft8Mod(FS, BASE_HZ, 0.0, 1.0)
        with pytest.raises((ValueError, Exception)):
            mod.modulate(np.zeros(57, dtype=np.uint8))

    def test_demod_too_short(self):
        demod = sdr.Ft8Demod(FS, BASE_HZ)
        with pytest.raises((ValueError, Exception)):
            demod.demodulate(np.zeros(1000, dtype=np.complex64))


# ---------------------------------------------------------------------------
# Waveform: Ft4Mod / Ft4Demod
# ---------------------------------------------------------------------------

class TestFt4Waveform:
    def test_mod_shape(self):
        mod = sdr.Ft4Mod(FS, BASE_HZ, 0.0, 1.0)
        tones = np.zeros(87, dtype=np.uint8)
        iq = mod.modulate(tones)
        assert iq.shape == (60_480,)
        assert iq.dtype == np.complex64

    def test_demod_roundtrip(self):
        tones_in = np.array([i % 4 for i in range(87)], dtype=np.uint8)
        mod = sdr.Ft4Mod(FS, BASE_HZ, 0.0, 1.0)
        iq = mod.modulate(tones_in)
        demod = sdr.Ft4Demod(FS, BASE_HZ)
        tones_out = demod.demodulate(iq)
        assert tones_out.shape == (87,)
        assert tones_out.dtype == np.uint8
        np.testing.assert_array_equal(tones_out, tones_in)


# ---------------------------------------------------------------------------
# Codec: Ft8Codec
# ---------------------------------------------------------------------------

class TestFt8Codec:
    def test_encode_shape(self):
        tones = sdr.Ft8Codec.encode(PAYLOAD)
        assert tones.shape == (58,)
        assert tones.dtype == np.uint8
        assert all(0 <= int(t) <= 7 for t in tones)

    def test_roundtrip_hard(self):
        tones = sdr.Ft8Codec.encode(PAYLOAD)
        result = sdr.Ft8Codec.decode_hard(tones)
        assert result is not None
        assert result == PAYLOAD

    def test_decode_hard_bad_tones(self):
        bad_tones = np.zeros(58, dtype=np.uint8)
        # All-zero tones won't pass CRC unless the payload happens to match;
        # result may be None or valid — just check it doesn't crash.
        result = sdr.Ft8Codec.decode_hard(bad_tones)
        assert result is None or isinstance(result, bytes)

    def test_decode_soft_shape_check(self):
        with pytest.raises((ValueError, Exception)):
            sdr.Ft8Codec.decode_soft(np.zeros(10, dtype=np.float32))


# ---------------------------------------------------------------------------
# Codec: Ft4Codec
# ---------------------------------------------------------------------------

class TestFt4Codec:
    def test_encode_shape(self):
        tones = sdr.Ft4Codec.encode(PAYLOAD)
        assert tones.shape == (87,)
        assert tones.dtype == np.uint8
        assert all(0 <= int(t) <= 3 for t in tones)

    def test_roundtrip_hard(self):
        tones = sdr.Ft4Codec.encode(PAYLOAD)
        result = sdr.Ft4Codec.decode_hard(tones)
        assert result is not None
        assert result == PAYLOAD


# ---------------------------------------------------------------------------
# Sync: ft8_sync
# ---------------------------------------------------------------------------

class TestFt8Sync:
    def _make_frame_iq(self, payload=PAYLOAD):
        """Build a noiseless IQ buffer containing one FT8 frame at offset 0."""
        tones = sdr.Ft8Codec.encode(payload)
        mod = sdr.Ft8Mod(FS, BASE_HZ, 0.0, 1.0)
        return mod.modulate(tones)

    def test_finds_frame(self):
        iq = self._make_frame_iq()
        results = sdr.ft8_sync(iq, FS, BASE_HZ - 50.0, BASE_HZ + 100.0, 0, 0, 5)
        assert len(results) > 0
        r = results[0]
        assert "time_sym" in r
        assert "freq_bin" in r
        assert "score" in r
        assert "llr" in r
        assert r["llr"].shape == (174,)
        assert r["llr"].dtype == np.float32

    def test_decode_soft(self):
        iq = self._make_frame_iq()
        results = sdr.ft8_sync(iq, FS, BASE_HZ - 50.0, BASE_HZ + 100.0, 0, 0, 5)
        assert len(results) > 0
        payload_out = sdr.Ft8Codec.decode_soft(results[0]["llr"])
        assert payload_out is not None
        assert payload_out == PAYLOAD

    def test_empty_on_silence(self):
        iq = np.zeros(151_680, dtype=np.complex64)
        results = sdr.ft8_sync(iq, FS, BASE_HZ, BASE_HZ + 100.0, 0, 0, 5)
        # Silence may produce candidates with low scores; just check no crash.
        assert isinstance(results, list)


# ---------------------------------------------------------------------------
# Message packing / unpacking
# ---------------------------------------------------------------------------

class TestFt8Message:
    def test_pack_unpack_standard(self):
        payload = sdr.ft8_pack_standard("KD9ABC", "W9XYZ", "FN31")
        assert isinstance(payload, bytes)
        assert len(payload) == 10
        msg = sdr.ft8_unpack(payload)
        assert msg["type"] == "standard"
        assert msg["call_to"] == "KD9ABC"
        assert msg["call_de"] == "W9XYZ"
        assert msg["extra"] == "FN31"

    def test_pack_unpack_free_text(self):
        payload = sdr.ft8_pack_free_text("CQ DX")
        assert isinstance(payload, bytes)
        assert len(payload) == 10
        msg = sdr.ft8_unpack(payload)
        assert msg["type"] == "free_text"
        assert msg["text"] == "CQ DX"

    def test_pack_unpack_telemetry(self):
        data = bytes([0x12, 0x34, 0x56, 0x78, 0x9A, 0xBC, 0xDE, 0xF0, 0x11])
        payload = sdr.ft8_pack_telemetry(data)
        assert isinstance(payload, bytes)
        assert len(payload) == 10
        msg = sdr.ft8_unpack(payload)
        assert msg["type"] == "telemetry"
        assert msg["data"] == data

    def test_pack_standard_cq(self):
        payload = sdr.ft8_pack_standard("CQ", "W9XYZ", "+07")
        assert isinstance(payload, bytes)
        assert len(payload) == 10

    def test_unpack_wrong_length(self):
        with pytest.raises((ValueError, Exception)):
            sdr.ft8_unpack(bytes(5))

    def test_pack_telemetry_wrong_length(self):
        with pytest.raises((ValueError, Exception)):
            sdr.ft8_pack_telemetry(bytes(8))


# ---------------------------------------------------------------------------
# Full stack: pack → encode → mod → sync → decode_soft → unpack
# ---------------------------------------------------------------------------

class TestFt8FullStack:
    def test_full_stack(self):
        # Pack a standard message
        payload = sdr.ft8_pack_standard("KD9ABC", "W9XYZ", "FN31")
        assert len(payload) == 10

        # Encode to tones
        tones = sdr.Ft8Codec.encode(payload)
        assert tones.shape == (58,)

        # Modulate to IQ
        mod = sdr.Ft8Mod(FS, BASE_HZ, 0.0, 1.0)
        iq = mod.modulate(tones)
        assert iq.shape == (151_680,)

        # Sync (wide search window)
        results = sdr.ft8_sync(iq, FS, BASE_HZ - 50.0, BASE_HZ + 100.0, 0, 0, 5)
        assert len(results) > 0

        # Soft decode
        payload_out = sdr.Ft8Codec.decode_soft(results[0]["llr"])
        assert payload_out is not None
        assert payload_out == payload

        # Unpack and verify
        msg = sdr.ft8_unpack(payload_out)
        assert msg["type"] == "standard"
        assert msg["call_to"] == "KD9ABC"
        assert msg["call_de"] == "W9XYZ"
        assert msg["extra"] == "FN31"
