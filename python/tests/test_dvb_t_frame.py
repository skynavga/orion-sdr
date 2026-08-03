# Copyright (c) 2026 G & R Associates LLC
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Tests for the conformant DVB-T on-air frame Python bindings.

All tests are noiseless / synthetic. They exercise the batch frame modulator and
demodulator (`dvb_t_frame_modulate` / `dvb_t_frame_demodulate`, EN 300 744): a
preamble-less frame placed at an unknown offset, guard-interval-acquired, with
both the payload and the TPS-signalled parameters recovered.
"""

import numpy as np
import pytest

import orion_sdr as sdr


def _sample_payload(n: int) -> np.ndarray:
    return np.frombuffer(bytes((i * 37 + 11) & 0xFF for i in range(n)), dtype=np.uint8)


def _place_with_offset(iq: np.ndarray, sps: int, lead: int = 200) -> np.ndarray:
    """Lead-in silence + frame + a trailing symbol of silence, so the RX must
    guard-interval-acquire rather than start at sample 0."""
    return np.concatenate(
        [
            np.zeros(lead, dtype=np.complex64),
            iq,
            np.zeros(sps, dtype=np.complex64),
        ]
    )


# ---------------------------------------------------------------------------
# Params surface
# ---------------------------------------------------------------------------


def test_params_roundtrip_getters():
    p = sdr.DvbTFrameParams("1/8", "qpsk", "1/2", frame_number=2, cell_id=0x5A)
    assert p.guard == "1/8"
    assert p.constellation == "qpsk"
    assert p.code_rate == "1/2"
    assert p.frame_number == 2
    assert p.cell_id == 0x5A


def test_params_defaults():
    p = sdr.DvbTFrameParams("1/32", "qam16", "3/4")
    assert p.frame_number == 0
    assert p.cell_id == 0


@pytest.mark.parametrize(
    "guard,const,rate",
    [
        ("bad", "qpsk", "1/2"),
        ("1/8", "qam256", "1/2"),  # not a DVB-T constellation
        ("1/8", "qpsk", "9/10"),
    ],
)
def test_params_reject_bad_strings(guard, const, rate):
    with pytest.raises(ValueError):
        sdr.DvbTFrameParams(guard, const, rate)


# ---------------------------------------------------------------------------
# Modulate / demodulate roundtrip
# ---------------------------------------------------------------------------


def test_frame_shape():
    p = sdr.DvbTFrameParams("1/8", "qpsk", "1/2")
    frame = sdr.dvb_t_frame_modulate(p, _sample_payload(184))
    # A frame is padded to at least a full 68-symbol TPS block.
    assert frame.n_symbols >= 68
    assert frame.samples_per_symbol == 2048 + 256  # n_fft + cp_len(1/8)
    assert frame.iq.shape == (frame.n_symbols * frame.samples_per_symbol,)
    assert frame.iq.dtype == np.complex64


@pytest.mark.parametrize(
    "guard,const,rate",
    [
        ("1/8", "qpsk", "1/2"),
        ("1/32", "qpsk", "2/3"),
        ("1/8", "qam16", "3/4"),
        ("1/4", "qam16", "1/2"),
    ],
)
def test_roundtrip_recovers_payload_and_tps(guard, const, rate):
    p = sdr.DvbTFrameParams(guard, const, rate, frame_number=1, cell_id=0x33)
    payload = _sample_payload(184)
    frame = sdr.dvb_t_frame_modulate(p, payload)
    buf = _place_with_offset(frame.iq, frame.samples_per_symbol)

    rx = sdr.dvb_t_frame_demodulate(p, buf, frame.n_symbols, len(payload))
    assert rx.payload == payload.tobytes()
    assert rx.tps.frame_number == 1
    assert rx.tps.constellation == const
    assert rx.tps.code_rate == rate
    assert rx.tps.guard == guard
    assert rx.tps.cell_id == 0x33


def test_payload_returned_as_bytes():
    p = sdr.DvbTFrameParams("1/8", "qpsk", "1/2")
    payload = _sample_payload(184)
    frame = sdr.dvb_t_frame_modulate(p, payload)
    buf = _place_with_offset(frame.iq, frame.samples_per_symbol)
    rx = sdr.dvb_t_frame_demodulate(p, buf, frame.n_symbols, len(payload))
    assert isinstance(rx.payload, bytes)
    assert len(rx.payload) == len(payload)


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


def test_demodulate_too_short_raises():
    p = sdr.DvbTFrameParams("1/8", "qpsk", "1/2")
    frame = sdr.dvb_t_frame_modulate(p, _sample_payload(184))
    # A buffer far too short for the declared symbol count.
    with pytest.raises(ValueError):
        sdr.dvb_t_frame_demodulate(
            p, np.zeros(5000, dtype=np.complex64), frame.n_symbols, 184
        )


# ---------------------------------------------------------------------------
# NB bandwidth helpers (pure fs-scaling of the fixed 2K structure)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "mode,occupied",
    [("333khz", 333_000.0), ("1mhz", 1_000_000.0), ("2mhz", 2_000_000.0)],
)
def test_nb_bandwidth(mode, occupied):
    assert sdr.nb_bandwidth_occupied_hz(mode) == pytest.approx(occupied)
    # fs = occupied * 2048 / 1705.
    assert sdr.nb_bandwidth_fs(mode) == pytest.approx(occupied * 2048.0 / 1705.0, rel=1e-4)


def test_nb_bandwidth_rejects_unknown():
    with pytest.raises(ValueError):
        sdr.nb_bandwidth_fs("5mhz")
