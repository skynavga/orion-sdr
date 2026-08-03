# Copyright (c) 2026 G & R Associates LLC
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Tests for the DVB-T super-frame and streaming-receiver Python bindings.

All tests are noiseless / synthetic. They exercise the four-frame super-frame
objects (DvbTSuperFrameMod/DvbTSuperFrameDemod, with the 16-bit cell-id split) and
the streaming receiver (DvbTFrameStreamDemod feed/flush).
"""

import numpy as np
import pytest

import orion_sdr as sdr


def _payload(n: int) -> np.ndarray:
    return np.frombuffer(bytes((i * 37 + 11) & 0xFF for i in range(n)), dtype=np.uint8)


# ---------------------------------------------------------------------------
# Super-frame
# ---------------------------------------------------------------------------


def test_super_frame_params():
    p = sdr.DvbTSuperFrameParams("1/8", "qam16", "3/4", cell_id=0x1234)
    assert p.guard == "1/8"
    assert p.constellation == "qam16"
    assert p.code_rate == "3/4"
    assert p.cell_id == 0x1234


def test_super_frame_params_default_cell_id():
    p = sdr.DvbTSuperFrameParams("1/32", "qpsk", "1/2")
    assert p.cell_id == 0


def test_super_frame_roundtrip():
    p = sdr.DvbTSuperFrameParams("1/8", "qpsk", "1/2", cell_id=0xBEEF)
    payload = _payload(700)
    sf = sdr.DvbTSuperFrameMod(p).modulate(payload)
    assert sf.n_symbols == 4 * sf.symbols_per_frame
    assert sum(sf.frame_payload_lens) == len(payload)
    assert sf.iq.shape == (sf.n_symbols * sf.samples_per_symbol,)

    rx = sdr.DvbTSuperFrameDemod(p).decode(
        sf.iq, sf.symbols_per_frame, sf.frame_payload_lens
    )
    assert rx.payload == payload.tobytes()
    assert rx.cell_id == 0xBEEF


def test_super_frame_cell_id_split():
    # A cell id whose two bytes differ, to prove the split/reassembly.
    p = sdr.DvbTSuperFrameParams("1/8", "qam16", "3/4", cell_id=0xABCD)
    payload = _payload(400)
    sf = sdr.DvbTSuperFrameMod(p).modulate(payload)
    rx = sdr.DvbTSuperFrameDemod(p).decode(
        sf.iq, sf.symbols_per_frame, sf.frame_payload_lens
    )
    assert rx.cell_id == 0xABCD
    assert rx.payload == payload.tobytes()


def test_super_frame_demodulate_garbage_raises():
    p = sdr.DvbTSuperFrameParams("1/8", "qpsk", "1/2")
    payload = _payload(400)
    sf = sdr.DvbTSuperFrameMod(p).modulate(payload)
    with pytest.raises(ValueError):
        sdr.DvbTSuperFrameDemod(p).decode(
            np.zeros(1000, dtype=np.complex64), sf.symbols_per_frame, sf.frame_payload_lens
        )


# ---------------------------------------------------------------------------
# Streaming receiver
# ---------------------------------------------------------------------------


def _stream_of(params, payloads):
    """Lead-in silence + the frames for `payloads` back to back + trailing
    silence. All frames share a symbol count (equal-size payloads)."""
    modulator = sdr.DvbTFrameMod(params)
    frames = [modulator.modulate(pl) for pl in payloads]
    n = frames[0].n_symbols
    assert all(f.n_symbols == n for f in frames)
    parts = [np.zeros(200, dtype=np.complex64)]
    parts += [f.iq for f in frames]
    parts.append(np.zeros(frames[0].samples_per_symbol, dtype=np.complex64))
    return np.concatenate(parts), n


def test_stream_single_frame():
    p = sdr.DvbTFrameParams("1/8", "qpsk", "1/2")
    payload = _payload(184)
    iq, n = _stream_of(p, [payload])
    rx = sdr.DvbTFrameStreamDemod(p, n, len(payload))
    got = rx.feed(iq) + rx.flush()
    assert len(got) == 1
    assert got[0].payload == payload.tobytes()


def test_stream_multiple_frames_chunked():
    # Three frames fed in awkward chunks must all decode in order.
    p = sdr.DvbTFrameParams("1/8", "qpsk", "1/2")
    payloads = [
        np.frombuffer(bytes((i * 13 + k * 40 + 1) & 0xFF for i in range(184)), np.uint8)
        for k in range(3)
    ]
    iq, n = _stream_of(p, payloads)
    rx = sdr.DvbTFrameStreamDemod(p, n, 184)
    got = []
    for chunk in np.array_split(iq, 7):
        got += rx.feed(np.ascontiguousarray(chunk))
    got += rx.flush()
    assert [f.payload for f in got] == [pl.tobytes() for pl in payloads]


def test_stream_holds_partial_frame():
    p = sdr.DvbTFrameParams("1/8", "qpsk", "1/2")
    payload = _payload(184)
    iq, n = _stream_of(p, [payload])
    rx = sdr.DvbTFrameStreamDemod(p, n, len(payload))
    half = len(iq) // 2
    assert rx.feed(np.ascontiguousarray(iq[:half])) == []
    assert rx.buffered > 0
    got = rx.feed(np.ascontiguousarray(iq[half:])) + rx.flush()
    assert len(got) == 1
    assert got[0].payload == payload.tobytes()


def test_stream_clear():
    p = sdr.DvbTFrameParams("1/8", "qpsk", "1/2")
    payload = _payload(184)
    iq, n = _stream_of(p, [payload])
    rx = sdr.DvbTFrameStreamDemod(p, n, len(payload))
    rx.feed(np.ascontiguousarray(iq[: len(iq) // 3]))
    assert rx.buffered > 0
    rx.clear()
    assert rx.buffered == 0
