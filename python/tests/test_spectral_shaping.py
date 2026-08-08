# Copyright (c) 2026 G & R Associates LLC
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Tests for the out-of-band spectral-shaping Python bindings.

Three off-by-default levers reduce OFDM's ``~1/f`` out-of-band skirt, and all
three are reachable from Python:

===================  ==================================  ====================
Lever                COFDM                               DVB-T
===================  ==================================  ====================
Edge-carrier guard   ``OfdmConfig(..., edge_guard=g)``    n/a (edge carriers
                                                          are mandatory pilots)
Symbol-window taper  ``OfdmConfig.with_symbol_window``    ``DvbTFrameMod
                                                          .with_symbol_window``
Baseband mask        ``OfdmConfig.with_tx_lowpass``       ``DvbTFrameMod
                                                          .with_tx_lowpass``
===================  ==================================  ====================

with ``with_rx_window_backoff`` on the receiving side of both. These tests cover
the sizing helpers a caller uses to pick the numbers, that every lever is off by
default (byte-identical output), that a shaped stream still round-trips, and —
the load-bearing part — that the mask measurably attenuates the stop band.

All tests are noiseless / synthetic.
"""

import numpy as np
import pytest

import orion_sdr as sdr

FS = 240_000.0


# ---------------------------------------------------------------------------
# Spectrum measurement
# ---------------------------------------------------------------------------


def _mean_band_power_db(iq: np.ndarray, lo_norm: float, hi_norm: float) -> float:
    """Mean power (dB) over ``lo_norm <= |f|/fs <= hi_norm``, read through a
    4-term Blackman-Harris window (sidelobes ~ -92 dB).

    The window is not optional. A rectangular analysis slice has its own ~-35 dB
    leakage floor, which would hide a 60 dB mask completely: what got measured
    would be the leakage of the *analysis*, not of the signal. Any claim of deep
    stop-band attenuation has to be read through a window whose sidelobes sit
    below the attenuation being claimed.
    """
    n = len(iq)
    x = 2.0 * np.pi * np.arange(n) / n
    w = (
        0.35875
        - 0.48829 * np.cos(x)
        + 0.14128 * np.cos(2.0 * x)
        - 0.01168 * np.cos(3.0 * x)
    )
    spec = np.fft.fft(np.asarray(iq, dtype=np.complex128) * w)
    freq = np.abs(np.fft.fftfreq(n))  # normalized, folded to [0, 0.5]
    sel = (freq >= lo_norm) & (freq <= hi_norm)
    assert sel.any(), f"empty measurement band [{lo_norm}, {hi_norm}]"
    return float(10.0 * np.log10((np.abs(spec[sel]) ** 2).mean() + 1e-30))


def _sample_payload(n: int) -> np.ndarray:
    return np.frombuffer(bytes((i * 37 + 11) & 0xFF for i in range(n)), dtype=np.uint8)


# ---------------------------------------------------------------------------
# COFDM: sizing helpers
# ---------------------------------------------------------------------------


def _cofdm_config(n_fft=256, cp_len=64, edge_guard=31, constellation="qpsk"):
    """A COFDM config whose data span is generated with an edge guard. Passing an
    empty `data_carriers` with `edge_guard` set is what asks for the contiguous
    fill; a non-empty array alongside it is an error, so a caller's own layout is
    never silently discarded."""
    return sdr.OfdmConfig(
        n_fft,
        cp_len,
        np.zeros(0, dtype=np.int32),
        np.zeros(0, dtype=np.int32),
        np.zeros(0, dtype=np.complex64),
        FS,
        0.0,
        1.0,
        constellation,
        edge_guard=edge_guard,
    )


class TestCofdmSizing:
    def test_edge_guard_narrows_the_occupied_band(self):
        # The guard is the lever that *makes the room* a mask filters into, and
        # occupied_half_carriers is how it reads back: n_fft/2 - 1 - edge_guard.
        assert _cofdm_config(edge_guard=0).occupied_half_carriers == 127
        assert _cofdm_config(edge_guard=31).occupied_half_carriers == 96

    def test_edge_guard_rejects_an_explicit_carrier_list(self):
        with pytest.raises(ValueError):
            sdr.OfdmConfig(
                64,
                8,
                np.array([1, 2, 3], dtype=np.int32),
                np.zeros(0, dtype=np.int32),
                np.zeros(0, dtype=np.complex64),
                FS,
                0.0,
                1.0,
                "qpsk",
                edge_guard=4,
            )

    def test_suggested_taps_track_the_room_available(self):
        # A wider null band needs a *shorter* filter to reach the stop band
        # inside it; a deeper stop band needs a longer one. Both directions
        # matter, since the length is what the guard budget constrains.
        narrow = _cofdm_config(edge_guard=8).tx_lowpass_suggested_taps(60.0)
        wide = _cofdm_config(edge_guard=48).tx_lowpass_suggested_taps(60.0)
        assert wide < narrow

        shallow = _cofdm_config(edge_guard=31).tx_lowpass_suggested_taps(40.0)
        deep = _cofdm_config(edge_guard=31).tx_lowpass_suggested_taps(80.0)
        assert shallow < deep

    def test_group_delay_is_the_odd_clamped_half_length(self):
        cfg = _cofdm_config()
        assert cfg.tx_lowpass_group_delay(65) == 32
        # Even lengths are forced odd, so 64 and 65 share a group delay.
        assert cfg.tx_lowpass_group_delay(64) == cfg.tx_lowpass_group_delay(65)

    def test_fits_guard_is_the_shared_budget(self):
        # roll_off + group_delay <= min(cp_len - backoff, backoff), so a filter
        # that fits on its own can stop fitting once a taper is added.
        cfg = _cofdm_config(cp_len=64)
        assert cfg.tx_lowpass_fits_guard(65)  # 32 + 0 <= 32
        assert not cfg.tx_lowpass_fits_guard(65, 32)  # 32 + 32 > 32
        assert cfg.tx_lowpass_fits_guard(33, 16)  # 16 + 16 <= 32

    def test_backoff_zero_leaves_no_slack_at_all(self):
        # The mask is applied group-delay-COMPENSATED, so half its response is a
        # pre-echo. A receiver pinned at the cyclic-prefix boundary has nowhere
        # to put that: the back-off is the enabler for both TX levers, not a
        # windowing-only requirement.
        cfg = _cofdm_config(cp_len=64)
        assert not cfg.tx_lowpass_fits_guard(3, 0, backoff=0)


# ---------------------------------------------------------------------------
# COFDM: defaults off, and round trips
# ---------------------------------------------------------------------------


def _frame_iq(cfg, payload, *, repeat_len=32):
    table = sdr.McsTable.default_ladder()
    mod = sdr.OfdmFrameMod(cfg, table, num_repeats=4, repeat_len=repeat_len)
    return mod.modulate_frame(sdr.FramePacket(payload, sequence_num=0xC1, mcs_index=1))


def _cofdm_frame_config(constellation="bpsk"):
    """The geometry the shaping round trips use: n_fft 128 / cp_len 32, with a
    20-carrier edge guard so the mask has a null band to work in."""
    return _cofdm_config(
        n_fft=128, cp_len=32, edge_guard=20, constellation=constellation
    )


class TestCofdmShapingDefaults:
    def test_all_three_levers_default_off(self):
        # A config that asks for none of this must emit exactly what it emitted
        # before the levers existed — the regression-safety property the whole
        # feature rests on.
        cfg = _cofdm_frame_config()
        payload = _sample_payload(48)
        assert np.array_equal(_frame_iq(cfg, payload), _frame_iq(cfg, payload))

    def test_each_lever_changes_the_waveform(self):
        # The mirror image: proving the builders are not silent no-ops.
        cfg = _cofdm_frame_config()
        payload = _sample_payload(48)
        plain = _frame_iq(cfg, payload)

        windowed = _frame_iq(cfg.with_symbol_window(6), payload)
        masked = _frame_iq(cfg.with_tx_lowpass(15, 40.0), payload)

        assert len(windowed) == len(plain), "the taper is a same-length post-pass"
        assert len(masked) == len(plain), "the mask is a same-length post-pass"
        assert not np.array_equal(windowed, plain)
        assert not np.array_equal(masked, plain)
        assert not np.array_equal(windowed, masked)


class TestCofdmShapingRoundtrip:
    @pytest.mark.parametrize(
        "roll_off,taps",
        [
            (0, 21),  # mask alone: group delay 10 <= 16
            (6, 15),  # mask + taper: 7 + 6 <= 16
            (16, 0),  # taper alone at the max-transparent roll-off
        ],
    )
    def test_shaped_frame_still_decodes(self, roll_off, taps):
        cp_len = 32
        backoff = cp_len // 2
        cfg = (
            _cofdm_frame_config()
            .with_inner_fec("ldpc", "n512r12")
            .with_outer_fec("bch", 8)
            .with_payload_crc("crc32")
            .with_header_crc("crc16")
            .with_rx_window_backoff(backoff)
        )
        if roll_off:
            cfg = cfg.with_symbol_window(roll_off)
        if taps:
            assert cfg.tx_lowpass_fits_guard(taps, roll_off, backoff)
            cfg = cfg.with_tx_lowpass(taps, 40.0)

        payload = _sample_payload(48)
        iq = _frame_iq(cfg, payload)
        buf = np.concatenate(
            [
                np.zeros(40, dtype=np.complex64),
                iq,
                np.zeros(128, dtype=np.complex64),
            ]
        )
        rx = sdr.OfdmFrameStreamDemod(
            cfg, sdr.McsTable.default_ladder(), num_repeats=4, repeat_len=32
        )
        frames = rx.feed(buf)
        assert len(frames) == 1, "the shaped frame must still acquire and decode"
        assert np.array_equal(frames[0].payload, payload)


# Spectrum geometry: n_fft 256 / cp_len 64 with a 31-carrier edge guard, so the
# occupied band ends at |f|/fs = 96/256 = 0.375 and a 45-tap 60 dB mask reaches
# its stop band around 0.455. Measuring over [0.47, 0.5] therefore reads the stop
# band proper, past the transition. Group delay 22 + an 8-sample taper = 30, which
# fits the 32-sample slack at backoff = cp_len/2.
_SPEC_TAPS, _SPEC_ROLL_OFF = 45, 8
_STOP_BAND = (0.47, 0.5)


def _spectrum_config(edge_guard=31):
    return _cofdm_config(n_fft=256, cp_len=64, edge_guard=edge_guard)


def _stop_band_db(cfg, payload):
    """Mean stop-band power of a modulated frame, over a stationary 4096-sample
    slice past the preamble and the filter's leading edge transient."""
    sps = cfg.samples_per_ofdm_symbol
    body = _frame_iq(cfg, payload)[4 * sps : 4 * sps + 4096]
    assert len(body) == 4096, "need a full analysis block"
    return _mean_band_power_db(body, *_STOP_BAND)


class TestCofdmSpectrum:
    def test_mask_drops_the_stop_band_below_the_taper_floor(self):
        # The point of the mask. Symbol windowing works on the symbol seam, so it
        # is capped by the taper the guard allows; the mask attacks the same
        # energy directly in the frequency domain, is not bound by that ceiling,
        # and the two stack because the mechanisms are independent.
        #
        # Measured in the mask's STOP band, which is how an emission mask is
        # specified in the first place — at a stated offset from the band edge.
        # The transition is deliberately unattenuated, so measuring inside it
        # would understate the mask and flatter the taper.
        cfg = _spectrum_config()
        assert cfg.occupied_half_carriers == 96  # 256/2 - 1 - 31
        assert cfg.tx_lowpass_fits_guard(_SPEC_TAPS, _SPEC_ROLL_OFF, 32)

        payload = _sample_payload(256)
        power = {
            name: _stop_band_db(c, payload)
            for name, c in {
                "baseline": cfg,
                "taper": cfg.with_symbol_window(_SPEC_ROLL_OFF),
                "mask": cfg.with_tx_lowpass(_SPEC_TAPS, 60.0),
                "both": cfg.with_symbol_window(_SPEC_ROLL_OFF).with_tx_lowpass(
                    _SPEC_TAPS, 60.0
                ),
            }.items()
        }

        # Observed on this geometry: -25 / -36 / -91 / -101 dB. Assert roughly
        # half of each observed margin, so the test states the effect without
        # pinning exact numbers to a particular FFT implementation.
        assert power["taper"] < power["baseline"] - 5.0, power
        assert power["mask"] < power["taper"] - 25.0, power
        assert power["both"] < power["mask"] - 4.0, power

    def test_edge_guard_alone_lowers_the_skirt(self):
        # Track A on its own: nulling the outermost carriers moves the loudest
        # sinc generators inward, so at a fixed offset from Nyquist the guarded
        # plan radiates far less. It does not change the ~1/f decay *rate* —
        # that is what the other two levers are for.
        #
        # It is also what makes the mask possible at all: with edge_guard = 0 the
        # plan fills every bin out to Nyquist, so [0.47, 0.5] still holds real
        # carriers and there is no null band for a transition to live in.
        payload = _sample_payload(256)
        guarded = _stop_band_db(_spectrum_config(edge_guard=31), payload)
        unguarded = _stop_band_db(_spectrum_config(edge_guard=0), payload)
        assert guarded < unguarded - 15.0, (guarded, unguarded)

    def test_shaping_leaves_in_band_power_alone(self):
        # Shaping that cost in-band power would be trading the payload for the
        # skirt. Both levers must be invisible inside the occupied band.
        cfg = _spectrum_config()
        payload = _sample_payload(256)
        sps = cfg.samples_per_ofdm_symbol

        def in_band(c):
            body = _frame_iq(c, payload)[4 * sps : 4 * sps + 4096]
            return _mean_band_power_db(body, 0.0, 0.36)

        base = in_band(cfg)
        shaped = in_band(
            cfg.with_symbol_window(_SPEC_ROLL_OFF).with_tx_lowpass(_SPEC_TAPS, 60.0)
        )
        assert abs(base - shaped) < 0.5, (base, shaped)


# ---------------------------------------------------------------------------
# DVB-T: sizing helpers
# ---------------------------------------------------------------------------


class TestDvbTSizing:
    @pytest.mark.parametrize(
        "guard,cp_len", [("1/32", 64), ("1/16", 128), ("1/8", 256), ("1/4", 512)]
    )
    def test_cp_len_per_guard(self, guard, cp_len):
        assert sdr.dvb_t_cp_len(guard) == cp_len

    def test_cp_len_rejects_unknown_guard(self):
        with pytest.raises(ValueError):
            sdr.dvb_t_cp_len("1/64")

    def test_backoff_ceiling_is_the_pilot_grid(self):
        # 2048 / (2 * 12) = 85, set by the scattered-pilot spacing rather than by
        # the guard. This is the number that makes the shaping budget saturate.
        assert sdr.dvb_t_max_rx_window_backoff() == 85

    def test_shaping_slack_saturates_at_g1_8(self):
        # The practical consequence: 32 / 64 / 85 / 85 samples of slack for
        # G1/32 ... G1/4. Going G1/32 -> G1/8 more than doubles the budget;
        # G1/8 -> G1/4 adds nothing, so the extra guard buys delay-spread
        # tolerance only. G1/8 is the sweet spot for crowded DATV band plans.
        cap = sdr.dvb_t_max_rx_window_backoff()
        slack = {}
        for guard in ("1/32", "1/16", "1/8", "1/4"):
            cp_len = sdr.dvb_t_cp_len(guard)
            b = min(cp_len // 2, cap)
            slack[guard] = min(cp_len - b, b)
        assert slack == {"1/32": 32, "1/16": 64, "1/8": 85, "1/4": 85}

    def test_suggested_taps_and_group_delay(self):
        taps = sdr.dvb_t_tx_lowpass_suggested_taps(60.0)
        assert taps % 2 == 1, "a linear-phase FIR is forced odd"
        assert sdr.dvb_t_tx_lowpass_group_delay(taps) == (taps - 1) // 2
        # A deeper stop band costs length.
        assert sdr.dvb_t_tx_lowpass_suggested_taps(80.0) > taps

    def test_fits_guard_matches_the_budget(self):
        # 89 taps -> group delay 44. With a 16-sample taper that needs 60 samples
        # of slack: available at G1/8 with b = 64, not at G1/32 where the whole
        # guard is 64 and the best slack is 32.
        assert sdr.dvb_t_tx_lowpass_group_delay(89) == 44
        assert sdr.dvb_t_tx_lowpass_fits_guard("1/8", 89, 16, 64)
        assert not sdr.dvb_t_tx_lowpass_fits_guard("1/32", 89, 16, 32)


# ---------------------------------------------------------------------------
# DVB-T: defaults off, round trips, and the null band
# ---------------------------------------------------------------------------


def _place(iq: np.ndarray, sps: int, lead: int = 200) -> np.ndarray:
    """Lead-in silence + frame + a trailing symbol, so the RX must
    guard-interval-acquire rather than assume sample 0."""
    return np.concatenate(
        [
            np.zeros(lead, dtype=np.complex64),
            iq,
            np.zeros(sps, dtype=np.complex64),
        ]
    )


class TestDvbTShaping:
    def test_both_levers_default_off(self):
        p = sdr.DvbTFrameParams("1/8", "qpsk", "1/2")
        payload = _sample_payload(184)
        a = sdr.DvbTFrameMod(p).modulate(payload).iq
        b = sdr.DvbTFrameMod(p).modulate(payload).iq
        np.testing.assert_array_equal(a, b)

        taper = sdr.DvbTFrameMod(p).with_symbol_window(16).modulate(payload).iq
        mask = sdr.DvbTFrameMod(p).with_tx_lowpass(89, 60.0).modulate(payload).iq
        assert len(taper) == len(a) and len(mask) == len(a)
        assert not np.array_equal(taper, a)
        assert not np.array_equal(mask, a)

    def test_backoff_defaults_to_zero_and_reads_back(self):
        p = sdr.DvbTFrameParams("1/8", "qpsk", "1/2")
        assert sdr.DvbTFrameDemod(p).rx_window_backoff == 0
        assert sdr.DvbTFrameDemod(p).with_rx_window_backoff(64).rx_window_backoff == 64

    @pytest.mark.parametrize(
        "guard,backoff,roll_off,taps",
        [
            ("1/32", 32, 0, 45),  # mask alone, group delay 22 <= 32
            ("1/8", 64, 16, 89),  # mask + taper, 44 + 16 <= 64
            ("1/8", 64, 32, 0),  # taper alone
        ],
    )
    def test_shaped_frame_round_trips(self, guard, backoff, roll_off, taps):
        p = sdr.DvbTFrameParams(guard, "qpsk", "1/2", frame_number=1, cell_id=0x33)
        assert backoff <= sdr.dvb_t_max_rx_window_backoff()
        mod = sdr.DvbTFrameMod(p)
        if roll_off:
            mod = mod.with_symbol_window(roll_off)
        if taps:
            assert sdr.dvb_t_tx_lowpass_fits_guard(guard, taps, roll_off, backoff)
            mod = mod.with_tx_lowpass(taps, 60.0)

        payload = _sample_payload(184)
        frame = mod.modulate(payload)
        buf = _place(frame.iq, frame.samples_per_symbol)

        rx = (
            sdr.DvbTFrameDemod(p)
            .with_rx_window_backoff(backoff)
            .decode(buf, frame.n_symbols, len(payload))
        )
        assert rx.payload == payload.tobytes()
        # The shaping touches only guard samples, so the TPS carriers and the
        # pilot grid come through untouched.
        assert rx.tps.guard == guard
        assert rx.tps.frame_number == 1
        assert rx.tps.cell_id == 0x33

    def test_backoff_past_the_pilot_ceiling_fails(self):
        # G1/8 has cp_len 256, so the "b = cp_len/2" rule of thumb would suggest
        # 128 — already past the 85-sample pilot ceiling. Past it the equalizer's
        # interpolated estimate aliases and the decode dies, however much guard
        # is free. This is why the ceiling is documented as a hard cap.
        p = sdr.DvbTFrameParams("1/8", "qpsk", "1/2")
        payload = _sample_payload(184)
        frame = sdr.DvbTFrameMod(p).modulate(payload)
        buf = _place(frame.iq, frame.samples_per_symbol)

        ok = (
            sdr.DvbTFrameDemod(p)
            .with_rx_window_backoff(64)
            .decode(buf, frame.n_symbols, len(payload))
        )
        assert ok.payload == payload.tobytes()

        with pytest.raises(ValueError):
            sdr.DvbTFrameDemod(p).with_rx_window_backoff(128).decode(
                buf, frame.n_symbols, len(payload)
            )

    def test_mask_attenuates_the_null_band(self):
        # DVB-T needs no edge guard: only 1705 of 2048 bins are active, so the
        # standard leaves a genuine null band for the transition to live in.
        # In-band power must be essentially untouched while the null band drops.
        p = sdr.DvbTFrameParams("1/8", "qpsk", "1/2")
        payload = _sample_payload(184)
        plain = sdr.DvbTFrameMod(p).modulate(payload)
        masked = sdr.DvbTFrameMod(p).with_tx_lowpass(89, 60.0).modulate(payload)

        sps = plain.samples_per_symbol
        a = plain.iq[2 * sps : 2 * sps + 8192]
        b = masked.iq[2 * sps : 2 * sps + 8192]

        # Occupied band edge is 852/2048 = 0.416; the 89-tap mask reaches its
        # stop band around 0.457, so 0.47 is safely past the transition.
        null_a = _mean_band_power_db(a, 0.47, 0.5)
        null_b = _mean_band_power_db(b, 0.47, 0.5)
        assert null_b < null_a - 30.0, (null_a, null_b)

        in_a = _mean_band_power_db(a, 0.0, 0.40)
        in_b = _mean_band_power_db(b, 0.0, 0.40)
        assert abs(in_a - in_b) < 0.5, "in-band power must be untouched"


class TestDvbTSuperFrameShaping:
    def test_masked_super_frame_round_trips(self):
        # 45 taps (group delay 22) rather than the longer filters the single-frame
        # tests use: the super-frame demod slices each frame at an exact boundary
        # with no lead-in, so it has no tolerance for the acquisition bias a
        # long filter's cyclic-prefix smearing introduces. See
        # TestDvbTZeroLeadInAcquisition for the mechanism.
        p = sdr.DvbTSuperFrameParams("1/8", "qpsk", "1/2", cell_id=0xBEEF)
        payload = _sample_payload(700)
        sf = sdr.DvbTSuperFrameMod(p).with_tx_lowpass(45, 60.0).modulate(payload)
        # Same-length post-pass: the super-frame is still a uniform block.
        assert len(sf.iq) == sf.n_symbols * sf.samples_per_symbol

        rx = (
            sdr.DvbTSuperFrameDemod(p)
            .with_rx_window_backoff(64)
            .decode(sf.iq, sf.symbols_per_frame, sf.frame_payload_lens)
        )
        assert rx.payload == payload.tobytes()
        assert rx.cell_id == 0xBEEF

    def test_mask_runs_across_the_frame_seams(self):
        # The mask is applied ONCE over the four concatenated frames, not per
        # frame: the three interior seams are continuous on air, and per-frame
        # filtering would leave the filter's edge transient at every one. That is
        # observable — the samples straddling a seam differ from what a filter
        # restarted at the seam would produce.
        p = sdr.DvbTSuperFrameParams("1/8", "qpsk", "1/2", cell_id=0x0102)
        payload = _sample_payload(700)
        taps = 45
        whole = sdr.DvbTSuperFrameMod(p).with_tx_lowpass(taps, 60.0).modulate(payload).iq
        plain = sdr.DvbTSuperFrameMod(p).modulate(payload).iq

        seam = len(whole) // 4
        d = sdr.dvb_t_tx_lowpass_group_delay(taps)
        assert (
            np.abs(whole[seam - d : seam + d] - plain[seam - d : seam + d]).max() > 1e-6
        )


class TestDvbTStreamShaping:
    def test_streaming_receiver_takes_a_back_off(self):
        # The streaming path needs the back-off too, or a shaped stream arrives
        # with the shaping inside the FFT window. Frame acquisition is unaffected:
        # the back-off moves only the per-symbol FFT window, not frame boundaries.
        p = sdr.DvbTFrameParams("1/8", "qpsk", "1/2")
        payload = _sample_payload(184)
        frame = sdr.DvbTFrameMod(p).with_tx_lowpass(89, 60.0).modulate(payload)
        buf = _place(frame.iq, frame.samples_per_symbol, lead=101)

        rx = sdr.DvbTFrameStreamDemod(
            p, frame.n_symbols, len(payload), rx_window_backoff=64
        )
        assert rx.rx_window_backoff == 64
        frames = rx.feed(buf)
        assert len(frames) == 1
        assert frames[0].payload == payload.tobytes()

    def test_stream_default_back_off_is_zero(self):
        p = sdr.DvbTFrameParams("1/8", "qpsk", "1/2")
        rx = sdr.DvbTFrameStreamDemod(p, 68, 184)
        assert rx.rx_window_backoff == 0


class TestDvbTBackOffSensitivity:
    """The back-off's aliasing cap is not its usable limit, and only noise shows it.

    `dvb_t_max_rx_window_backoff()` = 85 is where the scattered-pilot
    interpolation aliases. But that interpolation is *linear* between pilots 12
    carriers apart, while the back-off puts a phase ramp advancing
    ``theta = 2*pi*b*12/2048`` per gap on the spectrum — so it approximates an arc
    by a chord and is wrong by ``1 - cos(theta/2)`` in between. That error is
    graded and bites long before aliasing: measured, b=32 is free, b=42 costs
    ~1 dB, b=64 costs ~6 dB, and b=85 never closes.

    Every *noiseless* round trip in this file passes at b=64 — the FEC has margin
    to spare when there is no noise to spend it on. These tests add noise, which
    is the only way the limit is visible.
    """

    GUARD, SNR_DB, TRIALS = "1/8", 6.0, 12

    def _decode_rate(self, backoff, snr_db=None, seed=0xBEEF):
        p = sdr.DvbTFrameParams(self.GUARD, "qpsk", "1/2")
        payload = _sample_payload(184)
        frame = sdr.DvbTFrameMod(p).modulate(payload)
        demod = sdr.DvbTFrameDemod(p).with_rx_window_backoff(backoff)
        power = float(np.mean(np.abs(frame.iq) ** 2))
        n0 = power / (10.0 ** ((snr_db if snr_db is not None else self.SNR_DB) / 10.0))
        rng = np.random.default_rng(seed)
        ok = 0
        for _ in range(self.TRIALS):
            noise = rng.normal(0, np.sqrt(n0 / 2), len(frame.iq)) + 1j * rng.normal(
                0, np.sqrt(n0 / 2), len(frame.iq)
            )
            buf = _place(
                (frame.iq + noise).astype(np.complex64), frame.samples_per_symbol
            )
            try:
                rx = demod.decode(buf, frame.n_symbols, len(payload))
                ok += rx.payload == payload.tobytes()
            except ValueError:
                pass
        return ok / self.TRIALS

    def test_a_free_back_off_decodes(self):
        # theta = 68 deg, 17% interpolation error: no measurable cost.
        assert self._decode_rate(32) == 1.0

    def test_the_aliasing_cap_is_unusable(self):
        # theta = 179 deg: the interpolated estimate collapses toward zero
        # mid-gap. Not marginal — it does not decode at any SNR.
        cap = sdr.dvb_t_max_rx_window_backoff()
        assert self._decode_rate(cap, snr_db=15.0) == 0.0

    def test_the_cost_is_monotonic_in_the_interpolation_error(self):
        # The ordering is what the model predicts, and what sizing guidance
        # depends on: more phase advance per pilot gap, worse decode.
        assert self._decode_rate(32) >= self._decode_rate(48) >= self._decode_rate(64)

    def test_budget_legal_shaping_costs_nothing_extra(self):
        # The central claim: once the shaping fits the slack, it is free. Group
        # delay 22 (45 taps) + roll_off 8 = 30 <= min(256-32, 32) = 32.
        p = sdr.DvbTFrameParams(self.GUARD, "qpsk", "1/2")
        payload = _sample_payload(184)
        shaped = (
            sdr.DvbTFrameMod(p)
            .with_symbol_window(8)
            .with_tx_lowpass(45, 60.0)
            .modulate(payload)
        )
        assert sdr.dvb_t_tx_lowpass_fits_guard(self.GUARD, 45, 8, 32)

        demod = sdr.DvbTFrameDemod(p).with_rx_window_backoff(32)
        power = float(np.mean(np.abs(shaped.iq) ** 2))
        n0 = power / (10.0 ** (self.SNR_DB / 10.0))
        rng = np.random.default_rng(0xBEEF)
        ok = 0
        for _ in range(self.TRIALS):
            noise = rng.normal(0, np.sqrt(n0 / 2), len(shaped.iq)) + 1j * rng.normal(
                0, np.sqrt(n0 / 2), len(shaped.iq)
            )
            buf = _place(
                (shaped.iq + noise).astype(np.complex64), shaped.samples_per_symbol
            )
            rx = demod.decode(buf, shaped.n_symbols, len(payload))
            ok += rx.payload == payload.tobytes()
        assert ok == self.TRIALS


class TestDvbTZeroLeadInAcquisition:
    """Pins a known limitation of guard-interval acquisition under TX shaping.

    Symbol windowing biases the van de Beek ML timing estimate a few samples
    EARLY (measured ~roll_off/2: a 16-sample taper acquires ~8 samples early).
    That bias is harmless in itself — a backed-off FFT window absorbs it. It stops
    being harmless when the frame begins at sample 0 of the buffer: the search
    range is ``[0, symbol_period)``, so "slightly negative" is not representable
    and the argmax wraps to *nearly a whole symbol late*. Right symbol phase,
    wrong symbol.

    The mask is not affected the same way — it is linear-phase and applied
    group-delay-compensated, so it introduces no timing bias — but a long filter
    smears the cyclic-prefix correlation enough to move the argmax off zero, which
    is equally fatal where there is no slack.

    Two callers have no slack, and both are exercised below:

    * ``DvbTSuperFrameDemod`` hands each frame a sub-buffer starting exactly at
      that frame's first sample, and the last frame's sub-buffer is exactly one
      frame long — so any nonzero offset overruns it.
    * ``DvbTFrameStreamDemod`` acquires, then hands ``DvbTFrameDemod`` a slice
      starting at what it just acquired, which re-runs the search with almost no
      lead-in.

    Delete these tests if acquisition is made robust to a negative timing bias.
    """

    def test_a_tapered_frame_needs_a_few_samples_of_lead_in(self):
        p = sdr.DvbTFrameParams("1/8", "qpsk", "1/2")
        payload = _sample_payload(184)
        frame = sdr.DvbTFrameMod(p).with_symbol_window(16).modulate(payload)

        def decodes(lead):
            buf = _place(frame.iq, frame.samples_per_symbol, lead=lead)
            try:
                rx = (
                    sdr.DvbTFrameDemod(p)
                    .with_rx_window_backoff(64)
                    .decode(buf, frame.n_symbols, len(payload))
                )
                return rx.payload == payload.tobytes()
            except ValueError:
                return False

        assert not decodes(0), "known limitation: no lead-in, no lock"
        assert decodes(16), "a handful of lead-in samples is enough"
        assert decodes(200)

    def test_an_unshaped_frame_needs_no_lead_in(self):
        # The control: without shaping the argmax sits exactly at 0, so the same
        # zero-lead-in buffer decodes. This is what makes the above a shaping
        # interaction rather than a general acquisition weakness.
        p = sdr.DvbTFrameParams("1/8", "qpsk", "1/2")
        payload = _sample_payload(184)
        frame = sdr.DvbTFrameMod(p).modulate(payload)
        buf = _place(frame.iq, frame.samples_per_symbol, lead=0)
        rx = sdr.DvbTFrameDemod(p).decode(buf, frame.n_symbols, len(payload))
        assert rx.payload == payload.tobytes()

    def test_a_tapered_super_frame_does_not_round_trip(self):
        # The consequence for the super-frame demod, which slices at exact frame
        # boundaries. `with_symbol_window` is still bound and still shapes the
        # transmitted spectrum correctly — it is the paired receiver that cannot
        # currently acquire it.
        p = sdr.DvbTSuperFrameParams("1/8", "qpsk", "1/2", cell_id=0xBEEF)
        payload = _sample_payload(700)
        sf = sdr.DvbTSuperFrameMod(p).with_symbol_window(16).modulate(payload)
        with pytest.raises(ValueError):
            sdr.DvbTSuperFrameDemod(p).with_rx_window_backoff(64).decode(
                sf.iq, sf.symbols_per_frame, sf.frame_payload_lens
            )

    def test_a_long_mask_also_overruns_the_last_super_frame_frame(self):
        # Same zero-slack mechanism, reached by filter length rather than by a
        # taper: an 89-tap mask (group delay 44) moves the argmax off zero, and
        # the last frame's sub-buffer is exactly one frame long. The 45-tap mask
        # used in TestDvbTSuperFrameShaping stays inside the tolerance.
        p = sdr.DvbTSuperFrameParams("1/8", "qpsk", "1/2", cell_id=0xBEEF)
        payload = _sample_payload(700)
        sf = sdr.DvbTSuperFrameMod(p).with_tx_lowpass(89, 60.0).modulate(payload)
        with pytest.raises(ValueError):
            sdr.DvbTSuperFrameDemod(p).with_rx_window_backoff(64).decode(
                sf.iq, sf.symbols_per_frame, sf.frame_payload_lens
            )
