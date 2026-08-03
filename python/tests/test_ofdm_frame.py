"""Tests for the COFDM frame (MAC) layer Python bindings (Release R6).

All tests are noiseless / synthetic. They exercise the frame modulator, the
streaming frame receiver (acquisition + CFO + equalization), the batch
demodulator, and the FEC/CRC/scrambler configuration surface.
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


def _base_config(constellation: str = "bpsk") -> "sdr.OfdmConfig":
    return sdr.OfdmConfig(
        N_FFT,
        CP_LEN,
        _data_carriers(),
        np.zeros(0, dtype=np.int32),
        np.zeros(0, dtype=np.complex64),
        FS,
        0.0,
        1.0,
        constellation,
    )


def _ldpc_bch_config() -> "sdr.OfdmConfig":
    return (
        _base_config()
        .with_inner_fec("ldpc", "n512r12")
        .with_outer_fec("bch", 8)
        .with_payload_crc("crc32")
        .with_header_crc("crc16")
    )


def _sample_payload(n: int) -> np.ndarray:
    return np.frombuffer(bytes((i * 37 + 11) & 0xFF for i in range(n)), dtype=np.uint8)


# ---------------------------------------------------------------------------
# Configuration surface
# ---------------------------------------------------------------------------


class TestFrameConfig:
    def test_builders_chain_and_validate(self):
        cfg = _ldpc_bch_config()
        cfg.validate_frame()  # no exception

    def test_per_frame_seed_without_header_rejected(self):
        cfg = (
            _base_config()
            .with_header_format("none")
            .with_scrambler(0b1001, 7, per_frame_random=True)
        )
        with pytest.raises(ValueError):
            cfg.validate_frame()

    def test_unknown_inner_fec_rejected(self):
        with pytest.raises(ValueError):
            _base_config().with_inner_fec("bogus")

    def test_unknown_crc_rejected(self):
        with pytest.raises(ValueError):
            _base_config().with_payload_crc("crc7")


# ---------------------------------------------------------------------------
# MCS table
# ---------------------------------------------------------------------------


class TestMcsTable:
    def test_default_ladder(self):
        t = sdr.McsTable.default_ladder()
        assert t.len == 4

    def test_custom_table(self):
        t = sdr.McsTable()
        t.add("qpsk", inner_kind="ldpc", inner_code="n512r12", outer_kind="bch", outer_a=8)
        t.add("qam16", inner_kind="convolutional", inner_code="1/2", outer_kind="reed_solomon", outer_a=60, outer_b=8)
        assert t.len == 2

    def test_empty_table_rejected_at_use(self):
        cfg = _ldpc_bch_config()
        empty = sdr.McsTable()
        with pytest.raises(ValueError):
            sdr.OfdmFrameMod(cfg, empty)


# ---------------------------------------------------------------------------
# Frame round trips
# ---------------------------------------------------------------------------


class TestFrameRoundtrip:
    def test_batch_ldpc_bch(self):
        cfg = _ldpc_bch_config()
        table = sdr.McsTable.default_ladder()
        mod = sdr.OfdmFrameMod(cfg, table)
        payload = _sample_payload(40)
        frame = sdr.FramePacket(payload, sequence_num=0x1234, mcs_index=0)
        iq = mod.modulate_frame(frame)
        assert iq.dtype == np.complex64

        # Batch demod expects IQ starting after the preamble+training.
        preamble = sdr.generate_ofdm_preamble(cfg, 4, 16, N_FFT, CP_LEN)
        body = iq[len(preamble):]
        got = sdr.OfdmFrameDemod(cfg, table).decode(body)
        assert np.array_equal(got.payload, payload)
        assert got.sequence_num == 0x1234

    def test_scaled_min_sum_decode_rule(self):
        # The opt-in scaled-min-sum LDPC rule decodes the payload identically.
        cfg = _ldpc_bch_config().with_ldpc_decode_rule("scaled_min_sum", 0.75)
        table = sdr.McsTable.default_ladder()
        mod = sdr.OfdmFrameMod(cfg, table)
        payload = _sample_payload(40)
        frame = sdr.FramePacket(payload, sequence_num=0x99, mcs_index=0)
        iq = mod.modulate_frame(frame)
        preamble = sdr.generate_ofdm_preamble(cfg, 4, 16, N_FFT, CP_LEN)
        body = iq[len(preamble):]
        got = sdr.OfdmFrameDemod(cfg, table).decode(body)
        assert np.array_equal(got.payload, payload)

    def test_unknown_decode_rule_rejected(self):
        with pytest.raises(ValueError):
            _base_config().with_ldpc_decode_rule("bogus")

    def test_shared_codec_cache_roundtrip(self):
        # One CodecCache shared across the modulator, the streaming receiver,
        # and the batch demod builds each FEC code once and still decodes
        # identically to the private-cache path.
        cfg = _ldpc_bch_config()
        table = sdr.McsTable.default_ladder()
        cache = sdr.CodecCache()

        mod = sdr.OfdmFrameMod(cfg, table, cache=cache)
        # A stream receiver accepting the shared cache must at least construct.
        sdr.OfdmFrameStreamDemod(cfg, table, cache=cache)

        payload = _sample_payload(40)
        frame = sdr.FramePacket(payload, sequence_num=0x55AA, mcs_index=0)
        iq = mod.modulate_frame(frame)
        preamble = sdr.generate_ofdm_preamble(cfg, 4, 16, N_FFT, CP_LEN)
        body = iq[len(preamble):]

        # Shared-cache batch decode, and the no-cache path, agree.
        got_shared = sdr.OfdmFrameDemod(cfg, table, cache=cache).decode(body)
        got_default = sdr.OfdmFrameDemod(cfg, table).decode(body)
        assert np.array_equal(got_shared.payload, payload)
        assert np.array_equal(got_default.payload, got_shared.payload)
        assert got_shared.sequence_num == 0x55AA

    def test_stream_unknown_start(self):
        cfg = _ldpc_bch_config()
        table = sdr.McsTable.default_ladder()
        mod = sdr.OfdmFrameMod(cfg, table)
        payload = _sample_payload(40)
        frame = sdr.FramePacket(payload, sequence_num=42, mcs_index=0)
        iq = mod.modulate_frame(frame)

        buf = np.concatenate(
            [np.zeros(101, dtype=np.complex64), iq, np.zeros(64, dtype=np.complex64)]
        )
        rx = sdr.OfdmFrameStreamDemod(cfg, table)
        frames = rx.feed(buf)
        assert len(frames) == 1
        assert np.array_equal(frames[0].payload, payload)
        assert frames[0].sequence_num == 42

    def test_stream_back_to_back(self):
        cfg = _ldpc_bch_config()
        table = sdr.McsTable.default_ladder()
        mod = sdr.OfdmFrameMod(cfg, table)
        p0 = _sample_payload(24)
        p1 = _sample_payload(32)
        iq0 = mod.modulate_frame(sdr.FramePacket(p0, sequence_num=1))
        iq1 = mod.modulate_frame(sdr.FramePacket(p1, sequence_num=2))
        buf = np.concatenate(
            [np.zeros(40, dtype=np.complex64), iq0, iq1, np.zeros(64, dtype=np.complex64)]
        )
        rx = sdr.OfdmFrameStreamDemod(cfg, table)
        frames = rx.feed(buf)
        assert len(frames) == 2
        assert np.array_equal(frames[0].payload, p0)
        assert np.array_equal(frames[1].payload, p1)
        assert (frames[0].sequence_num, frames[1].sequence_num) == (1, 2)

    def test_rs_convolutional(self):
        cfg = (
            _base_config()
            .with_payload_crc("crc32")
        )
        table = sdr.McsTable()
        table.add(
            "qpsk",
            inner_kind="convolutional",
            inner_code="1/2",
            outer_kind="reed_solomon",
            outer_a=60,
            outer_b=8,
        )
        mod = sdr.OfdmFrameMod(cfg, table)
        payload = _sample_payload(40)
        frame = sdr.FramePacket(payload, sequence_num=100, mcs_index=0)
        iq = mod.modulate_frame(frame)
        rx = sdr.OfdmFrameStreamDemod(cfg, table)
        buf = np.concatenate([np.zeros(30, dtype=np.complex64), iq, np.zeros(64, dtype=np.complex64)])
        frames = rx.feed(buf)
        assert len(frames) == 1
        assert np.array_equal(frames[0].payload, payload)

    def test_feed_with_errors_reports_corruption(self):
        cfg = _ldpc_bch_config()
        table = sdr.McsTable.default_ladder()
        mod = sdr.OfdmFrameMod(cfg, table)
        frame = sdr.FramePacket(_sample_payload(40), sequence_num=5, mcs_index=0)
        iq = mod.modulate_frame(frame)
        buf = np.concatenate([np.zeros(16, dtype=np.complex64), iq, np.zeros(64, dtype=np.complex64)])
        # Corrupt the payload region beyond FEC's reach.
        preamble = sdr.generate_ofdm_preamble(cfg, 4, 16, N_FFT, CP_LEN)
        start = 16 + len(preamble) + 12 * cfg.samples_per_ofdm_symbol
        buf[start : start + 400] = -3.0 * buf[start : start + 400]
        rx = sdr.OfdmFrameStreamDemod(cfg, table)
        results = rx.feed_with_errors(buf)
        # No successful frame; at least one error surfaced.
        assert all(f is None for (f, _e) in results)
        assert any(e is not None for (_f, e) in results)


class TestScrambler:
    def test_scrambler_round_trips(self):
        for pos in ("before_outer", "after_inner"):
            cfg = (
                _ldpc_bch_config()
                .with_scrambler(0b1001, 7, seed=0x7F, position=pos)
            )
            table = sdr.McsTable.default_ladder()
            mod = sdr.OfdmFrameMod(cfg, table)
            payload = _sample_payload(24)
            frame = sdr.FramePacket(payload, sequence_num=2, mcs_index=0)
            iq = mod.modulate_frame(frame)
            rx = sdr.OfdmFrameStreamDemod(cfg, table)
            buf = np.concatenate([np.zeros(20, dtype=np.complex64), iq, np.zeros(64, dtype=np.complex64)])
            frames = rx.feed(buf)
            assert len(frames) == 1, f"scrambler pos {pos}"
            assert np.array_equal(frames[0].payload, payload)
