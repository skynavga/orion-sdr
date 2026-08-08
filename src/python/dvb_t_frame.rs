// Copyright (c) 2026 G & R Associates LLC
// SPDX-License-Identifier: MIT OR Apache-2.0

// src/python/dvb_t_frame.rs — PyO3 bindings for the conformant, preamble-less
// DVB-T on-air frame (ETSI EN 300 744).
//
// Exposes the stateful frame/super-frame modulator and demodulator objects
// (`modulate::{DvbTFrameMod, DvbTSuperFrameMod}` /
// `demodulate::{DvbTFrameDemod, DvbTSuperFrameDemod}`) and the streaming receiver,
// the shared `DvbTFrameParams`, the recovered `DvbTRxFrame`/`TpsWord`, and the
// `NbBandwidth` sample-rate helper. Transmission parameters (guard interval,
// constellation, code rate) are passed as strings, matching the convention used
// by the config-level DVB-T bindings in `python/ofdm.rs`. Integer-CFO correction
// is a construction-time flag on the demod objects (`with_integer_cfo_correction`).
//
// The out-of-band spectral-shaping levers are builders too: `with_symbol_window`
// and `with_tx_lowpass` on the modulators, `with_rx_window_backoff` on the
// demodulators. `TxLowpass` itself is not a Python class — the DVB-T band edge is
// fixed, so the mask is specified by `(num_taps, stopband_db)` and built
// internally by `DvbTFrameMod::tx_lowpass_for_2k`. The `dvb_t_*` module functions
// below expose the sizing arithmetic (cp_len, the back-off ceiling, a suggested
// tap count, group delay, the guard-budget check) so a Python caller can pick
// those numbers without re-deriving them.

use num_complex::Complex32;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;

use crate::demodulate::{DvbTFrameDemod, DvbTFrameStreamDemod, DvbTSuperFrameDemod};
use crate::fec::PunctureRate;
use crate::modulate::{ConstellationOrder, DvbTFrameMod, DvbTSuperFrameMod, DvbTSuperFrameParams};
use crate::multicarrier::TxLowpass;
use crate::waveform::dvb_t::{
    DVB_T_MAX_RX_WINDOW_BACKOFF, DvbTFrameParams, DvbTLinkParams, GuardInterval, NbBandwidth,
};
use crate::waveform::dvb_t_tps::TpsWord;

// ── String <-> enum helpers (crate Python convention) ───────────────────────

fn parse_guard(s: &str) -> PyResult<GuardInterval> {
    match s {
        "1/32" => Ok(GuardInterval::G1_32),
        "1/16" => Ok(GuardInterval::G1_16),
        "1/8" => Ok(GuardInterval::G1_8),
        "1/4" => Ok(GuardInterval::G1_4),
        other => Err(PyValueError::new_err(format!(
            "unknown guard interval {other:?} (expected 1/32, 1/16, 1/8, 1/4)"
        ))),
    }
}

fn guard_str(g: GuardInterval) -> &'static str {
    match g {
        GuardInterval::G1_32 => "1/32",
        GuardInterval::G1_16 => "1/16",
        GuardInterval::G1_8 => "1/8",
        GuardInterval::G1_4 => "1/4",
    }
}

fn parse_dvb_t_constellation(s: &str) -> PyResult<ConstellationOrder> {
    match s {
        "qpsk" => Ok(ConstellationOrder::Qpsk),
        "qam16" => Ok(ConstellationOrder::Qam16),
        "qam64" => Ok(ConstellationOrder::Qam64),
        other => Err(PyValueError::new_err(format!(
            "unknown DVB-T constellation {other:?} (expected qpsk, qam16, qam64)"
        ))),
    }
}

fn constellation_str(c: ConstellationOrder) -> PyResult<&'static str> {
    match c {
        ConstellationOrder::Qpsk => Ok("qpsk"),
        ConstellationOrder::Qam16 => Ok("qam16"),
        ConstellationOrder::Qam64 => Ok("qam64"),
        other => Err(PyValueError::new_err(format!(
            "{other:?} is not a DVB-T constellation"
        ))),
    }
}

fn parse_rate(s: &str) -> PyResult<PunctureRate> {
    match s {
        "1/2" => Ok(PunctureRate::R1_2),
        "2/3" => Ok(PunctureRate::R2_3),
        "3/4" => Ok(PunctureRate::R3_4),
        "5/6" => Ok(PunctureRate::R5_6),
        "7/8" => Ok(PunctureRate::R7_8),
        other => Err(PyValueError::new_err(format!(
            "unknown code rate {other:?} (expected 1/2, 2/3, 3/4, 5/6, 7/8)"
        ))),
    }
}

fn rate_str(r: PunctureRate) -> &'static str {
    match r {
        PunctureRate::R1_2 => "1/2",
        PunctureRate::R2_3 => "2/3",
        PunctureRate::R3_4 => "3/4",
        PunctureRate::R5_6 => "5/6",
        PunctureRate::R7_8 => "7/8",
    }
}

// ── DvbTFrameParams ─────────────────────────────────────────────────────────

/// Transmission parameters for a conformant DVB-T frame: guard interval,
/// constellation, code rate, and the TPS-signalled frame number and cell id.
/// `guard`/`constellation`/`code_rate` are strings (e.g. `"1/8"`, `"qpsk"`,
/// `"1/2"`).
#[pyclass(name = "DvbTFrameParams", skip_from_py_object)]
#[derive(Clone)]
pub struct PyDvbTFrameParams {
    inner: DvbTFrameParams,
}

#[pymethods]
impl PyDvbTFrameParams {
    #[new]
    #[pyo3(signature = (guard, constellation, code_rate, frame_number = 0, cell_id = 0))]
    fn new(
        guard: &str,
        constellation: &str,
        code_rate: &str,
        frame_number: u8,
        cell_id: u8,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: DvbTFrameParams {
                link: DvbTLinkParams {
                    guard: parse_guard(guard)?,
                    constellation: parse_dvb_t_constellation(constellation)?,
                    code_rate: parse_rate(code_rate)?,
                },
                frame_number,
                cell_id,
            },
        })
    }

    #[getter]
    fn guard(&self) -> &'static str {
        guard_str(self.inner.guard())
    }
    #[getter]
    fn constellation(&self) -> PyResult<&'static str> {
        constellation_str(self.inner.constellation())
    }
    #[getter]
    fn code_rate(&self) -> &'static str {
        rate_str(self.inner.code_rate())
    }
    #[getter]
    fn frame_number(&self) -> u8 {
        self.inner.frame_number
    }
    #[getter]
    fn cell_id(&self) -> u8 {
        self.inner.cell_id
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(format!(
            "DvbTFrameParams(guard={:?}, constellation={:?}, code_rate={:?}, frame_number={}, cell_id={})",
            guard_str(self.inner.guard()),
            constellation_str(self.inner.constellation())?,
            rate_str(self.inner.code_rate()),
            self.inner.frame_number,
            self.inner.cell_id,
        ))
    }
}

// ── TpsWord (recovered) ─────────────────────────────────────────────────────

/// The transmission parameters recovered from a frame's TPS carriers.
#[pyclass(name = "TpsWord", skip_from_py_object)]
#[derive(Clone)]
pub struct PyTpsWord {
    inner: TpsWord,
}

#[pymethods]
impl PyTpsWord {
    #[getter]
    fn frame_number(&self) -> u8 {
        self.inner.frame_number
    }
    #[getter]
    fn constellation(&self) -> PyResult<&'static str> {
        constellation_str(self.inner.constellation)
    }
    #[getter]
    fn code_rate(&self) -> &'static str {
        rate_str(self.inner.code_rate_hp)
    }
    #[getter]
    fn guard(&self) -> &'static str {
        guard_str(self.inner.guard)
    }
    #[getter]
    fn cell_id(&self) -> u8 {
        self.inner.cell_id
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(format!(
            "TpsWord(frame_number={}, constellation={:?}, code_rate={:?}, guard={:?}, cell_id={})",
            self.inner.frame_number,
            constellation_str(self.inner.constellation)?,
            rate_str(self.inner.code_rate_hp),
            guard_str(self.inner.guard),
            self.inner.cell_id,
        ))
    }
}

// ── DvbTFrame (modulated) ───────────────────────────────────────────────────

/// A modulated DVB-T frame: the time-domain IQ plus the numerology a receiver
/// needs to acquire it.
#[pyclass(name = "DvbTFrame")]
pub struct PyDvbTFrame {
    iq: Vec<Complex32>,
    #[pyo3(get)]
    n_symbols: usize,
    #[pyo3(get)]
    samples_per_symbol: usize,
}

#[pymethods]
impl PyDvbTFrame {
    /// The time-domain baseband IQ (no preamble; a whole number of OFDM symbols).
    #[getter]
    fn iq<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<Complex32>> {
        self.iq.clone().into_pyarray(py)
    }
}

// ── DvbTRxFrame (demodulated) ───────────────────────────────────────────────

/// The recovered contents of a DVB-T frame: the TS payload and the TPS word.
#[pyclass(name = "DvbTRxFrame")]
pub struct PyDvbTRxFrame {
    payload: Vec<u8>,
    #[pyo3(get)]
    tps: PyTpsWord,
}

#[pymethods]
impl PyDvbTRxFrame {
    /// The recovered TS payload bytes (depacketized, trimmed to `payload_len`).
    #[getter]
    fn payload<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new(py, &self.payload)
    }
}

// ── DvbTFrameMod / DvbTFrameDemod ───────────────────────────────────────────

/// A conformant, preamble-less DVB-T frame modulator. Built from
/// `DvbTFrameParams`; `modulate(payload)` produces one `DvbTFrame` per call.
/// Out-of-band spectral shaping is off by default — see `with_symbol_window`
/// and `with_tx_lowpass`.
#[pyclass(name = "DvbTFrameMod")]
pub struct PyDvbTFrameMod {
    inner: DvbTFrameMod,
}

#[pymethods]
impl PyDvbTFrameMod {
    #[new]
    fn new(params: &PyDvbTFrameParams) -> Self {
        Self {
            inner: DvbTFrameMod::new(params.inner),
        }
    }

    /// Returns a modulator that applies a `roll_off`-sample raised-cosine taper
    /// to each symbol's edges, reducing out-of-band emission. `0` (the default)
    /// leaves the on-air frame byte-identical. DVB-T is preamble-less, so every
    /// symbol is CP-bearing and every one is tapered.
    ///
    /// Only RX-transparent when paired with a matching back-off on the
    /// demodulator: `roll_off = backoff = cp_len/2` is the transparent operating
    /// point, subject to the `dvb_t_max_rx_window_backoff()` ceiling.
    fn with_symbol_window(&self, roll_off: usize) -> Self {
        Self {
            inner: self.inner.clone().with_symbol_window(roll_off),
        }
    }

    /// Returns a modulator that applies a TX baseband low-pass (spectral mask)
    /// across the assembled frame, after any symbol taper. Absent by default.
    ///
    /// The cutoff is placed against DVB-T's fixed `±852`-of-2048 band edge, so
    /// only the filter length — the quantity the guard budget constrains — and
    /// the stop-band target are the caller's. Unlike the taper this is not
    /// bounded by the windowing ceiling: it attenuates out-of-band energy
    /// directly in the frequency domain, so its gain stacks on top.
    ///
    /// It needs no decoding change (the scattered-pilot equalizer absorbs the
    /// filter like any other channel), but its group delay must land in guard
    /// the receiver discards. Size it with `dvb_t_tx_lowpass_suggested_taps` and
    /// check it with `dvb_t_tx_lowpass_fits_guard`.
    #[pyo3(signature = (num_taps, stopband_db = 60.0))]
    fn with_tx_lowpass(&self, num_taps: usize, stopband_db: f32) -> Self {
        Self {
            inner: self
                .inner
                .clone()
                .with_tx_lowpass(DvbTFrameMod::tx_lowpass_for_2k(num_taps, stopband_db)),
        }
    }

    /// Modulates `payload` (the MPEG-TS payload bytes) into one conformant,
    /// preamble-less DVB-T frame. Returns a `DvbTFrame` (`.iq`, `.n_symbols`,
    /// `.samples_per_symbol`).
    fn modulate(&self, payload: PyReadonlyArray1<'_, u8>) -> PyResult<PyDvbTFrame> {
        let frame = self.inner.modulate(payload.as_slice()?);
        Ok(PyDvbTFrame {
            iq: frame.iq,
            n_symbols: frame.n_symbols,
            samples_per_symbol: frame.samples_per_symbol,
        })
    }
}

/// A conformant, preamble-less DVB-T frame demodulator. Built from
/// `DvbTFrameParams`; `decode(iq, n_symbols, payload_len)` recovers one frame.
/// Integer-CFO correction is off by default — enable it with
/// `with_integer_cfo_correction(True)` (a link-constant builder that returns a new
/// demod).
#[pyclass(name = "DvbTFrameDemod")]
pub struct PyDvbTFrameDemod {
    inner: DvbTFrameDemod,
}

#[pymethods]
impl PyDvbTFrameDemod {
    #[new]
    fn new(params: &PyDvbTFrameParams) -> Self {
        Self {
            inner: DvbTFrameDemod::new(params.inner),
        }
    }

    /// Returns a demod with internal integer-CFO correction enabled (or disabled).
    /// A link-constant knob: when on, `decode` estimates the whole-subcarrier
    /// offset from the continual pilots and rotates it out before demapping.
    fn with_integer_cfo_correction(&self, on: bool) -> Self {
        Self {
            inner: self.inner.clone().with_integer_cfo_correction(on),
        }
    }

    /// Whether internal integer-CFO correction is enabled.
    #[getter]
    fn integer_cfo_correction(&self) -> bool {
        self.inner.integer_cfo_correction()
    }

    /// Returns a demod whose per-symbol FFT window sits `backoff` samples
    /// earlier in the guard (default `0` = the standard CP-boundary window).
    ///
    /// This is the receiver half of the TX shaping pair: a symbol taper and a
    /// baseband mask both live in guard samples, and only a backed-off window
    /// leaves them outside the FFT. It also buys pre-echo and timing-error
    /// tolerance on its own. The scattered-pilot estimate is measured at the
    /// same back-off, so the phase ramp the shift induces is corrected.
    ///
    /// Capped at `dvb_t_max_rx_window_backoff()` (85) by the scattered-pilot
    /// spacing — **not** by the guard interval.
    fn with_rx_window_backoff(&self, backoff: usize) -> Self {
        Self {
            inner: self.inner.clone().with_rx_window_backoff(backoff),
        }
    }

    /// The receiver FFT-window back-off in samples.
    #[getter]
    fn rx_window_backoff(&self) -> usize {
        self.inner.rx_window_backoff()
    }

    /// Demodulates one conformant DVB-T frame from `iq`, acquiring the symbol grid
    /// from the guard interval (no preamble). `n_symbols` is the frame's symbol
    /// count (from the paired `DvbTFrameMod.modulate` result's `DvbTFrame`);
    /// `payload_len` is the original payload byte count for trimming. Raises
    /// `ValueError` on any acquisition/decode failure. The returned `DvbTRxFrame`
    /// exposes `.payload` (bytes) and `.tps` (the recovered `TpsWord`).
    fn decode(
        &self,
        iq: PyReadonlyArray1<'_, Complex32>,
        n_symbols: usize,
        payload_len: usize,
    ) -> PyResult<PyDvbTRxFrame> {
        let rx = self
            .inner
            .decode(iq.as_slice()?, n_symbols, payload_len)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(PyDvbTRxFrame {
            payload: rx.payload,
            tps: PyTpsWord { inner: rx.tps },
        })
    }
}

// ── NB bandwidth helpers ────────────────────────────────────────────────────

/// The sample rate (S/s) for a narrowband DVB-T bandwidth mode: `"333khz"`,
/// `"1mhz"`, or `"2mhz"`. NB-DVB-T is a pure fs-scaling of the fixed 2K
/// structure — `fs = occupied_hz · 2048/1705`.
#[pyfunction]
#[pyo3(name = "nb_bandwidth_fs")]
fn nb_bandwidth_fs(mode: &str) -> PyResult<f32> {
    Ok(parse_nb_bandwidth(mode)?.fs())
}

/// The nominal occupied RF bandwidth (Hz) for a narrowband DVB-T mode
/// (`"333khz"`, `"1mhz"`, `"2mhz"`).
#[pyfunction]
#[pyo3(name = "nb_bandwidth_occupied_hz")]
fn nb_bandwidth_occupied_hz(mode: &str) -> PyResult<f32> {
    Ok(parse_nb_bandwidth(mode)?.occupied_hz())
}

fn parse_nb_bandwidth(s: &str) -> PyResult<NbBandwidth> {
    match s {
        "333khz" | "333k" => Ok(NbBandwidth::Bw333kHz),
        "1mhz" | "1m" => Ok(NbBandwidth::Bw1MHz),
        "2mhz" | "2m" => Ok(NbBandwidth::Bw2MHz),
        other => Err(PyValueError::new_err(format!(
            "unknown NB bandwidth {other:?} (expected 333khz, 1mhz, 2mhz)"
        ))),
    }
}

// ── Spectral-shaping sizing helpers ─────────────────────────────────────────
//
// The arithmetic a caller needs to choose `roll_off`, `num_taps` and `backoff`
// without re-deriving it. `TxLowpass` is not a Python class (DVB-T's band edge is
// fixed, so a mask is fully specified by its length and stop-band target), so
// these are module functions rather than methods.

/// The cyclic-prefix length in samples for a DVB-T 2K guard interval:
/// 64 / 128 / 256 / 512 for `"1/32"` / `"1/16"` / `"1/8"` / `"1/4"`. This is the
/// guard the two TX shaping levers and the RX window back-off share.
#[pyfunction]
#[pyo3(name = "dvb_t_cp_len")]
fn dvb_t_cp_len(guard: &str) -> PyResult<usize> {
    Ok(parse_guard(guard)?.cp_len_2k())
}

/// The largest usable RX FFT-window back-off for DVB-T 2K: **85 samples**,
/// whatever the guard interval.
///
/// The cap comes from the scattered-pilot grid, not the guard. The back-off
/// induces a per-bin phase ramp that the equalizer removes from its channel
/// estimate, but that estimate is only sampled every 12 carriers; past
/// `n_fft / (2 · 12)` the interpolation aliases. So the shaping budget
/// saturates: 32 / 64 / 85 / 85 samples for G1/32 … G1/4, which makes G1/8 the
/// sweet spot — the full budget at a quarter of G1/4's guard overhead.
#[pyfunction]
#[pyo3(name = "dvb_t_max_rx_window_backoff")]
fn dvb_t_max_rx_window_backoff() -> usize {
    DVB_T_MAX_RX_WINDOW_BACKOFF
}

/// The shortest mask whose transition fits inside DVB-T's null band (the 343 of
/// 2048 bins the standard leaves inactive) at `stopband_db` — a starting point
/// for `DvbTFrameMod.with_tx_lowpass`, to be checked against the guard budget
/// with `dvb_t_tx_lowpass_fits_guard`.
///
/// A shorter filter cannot reach its stop band before Nyquist; a longer one
/// reaches it sooner, at the cost of more group delay to fit in the guard.
#[pyfunction]
#[pyo3(name = "dvb_t_tx_lowpass_suggested_taps")]
#[pyo3(signature = (stopband_db = 60.0))]
fn dvb_t_tx_lowpass_suggested_taps(stopband_db: f32) -> usize {
    TxLowpass::taps_for_null_band(
        crate::waveform::dvb_t::DVB_T_N_FFT,
        crate::waveform::dvb_t::DVB_T_KMAX / 2,
        stopband_db,
    )
}

/// A mask's group delay in samples, `(num_taps − 1) / 2` after the odd/≥3 clamp
/// the designer applies — the filter's reach on each side of a sample, and the
/// quantity the guard budget has to cover.
#[pyfunction]
#[pyo3(name = "dvb_t_tx_lowpass_group_delay")]
fn dvb_t_tx_lowpass_group_delay(num_taps: usize) -> usize {
    TxLowpass::new(0.25, num_taps, 60.0).group_delay()
}

/// Whether a `num_taps` mask and a `roll_off`-sample symbol taper both fit in
/// the guard samples a receiver at `backoff` discards:
///
/// ```text
/// roll_off + group_delay <= min(cp_len - backoff, backoff)
/// ```
///
/// Pass `roll_off = 0` when symbol windowing is off. The slack is maximized at
/// `backoff = cp_len/2` — but on DVB-T that is only reachable up to
/// `dvb_t_max_rx_window_backoff()`. Overrunning the budget degrades gradually
/// (a little inter-symbol leakage the equalizer cannot invert) rather than
/// failing abruptly, but it is the budget to design against.
#[pyfunction]
#[pyo3(name = "dvb_t_tx_lowpass_fits_guard")]
#[pyo3(signature = (guard, num_taps, roll_off, backoff))]
fn dvb_t_tx_lowpass_fits_guard(
    guard: &str,
    num_taps: usize,
    roll_off: usize,
    backoff: usize,
) -> PyResult<bool> {
    let cp_len = parse_guard(guard)?.cp_len_2k();
    Ok(DvbTFrameMod::tx_lowpass_for_2k(num_taps, 60.0).fits_guard(cp_len, roll_off, backoff))
}

// ── Super-frame ─────────────────────────────────────────────────────────────

/// Transmission parameters for a conformant DVB-T super-frame (four frames).
/// Like [`DvbTFrameParams`] but with the **full 16-bit** cell id, which is split
/// across the frames (b15..b8 in frames 1 & 3, b7..b0 in frames 2 & 4).
#[pyclass(name = "DvbTSuperFrameParams", skip_from_py_object)]
#[derive(Clone)]
pub struct PyDvbTSuperFrameParams {
    inner: DvbTSuperFrameParams,
}

#[pymethods]
impl PyDvbTSuperFrameParams {
    #[new]
    #[pyo3(signature = (guard, constellation, code_rate, cell_id = 0))]
    fn new(guard: &str, constellation: &str, code_rate: &str, cell_id: u16) -> PyResult<Self> {
        Ok(Self {
            inner: DvbTSuperFrameParams {
                link: DvbTLinkParams {
                    guard: parse_guard(guard)?,
                    constellation: parse_dvb_t_constellation(constellation)?,
                    code_rate: parse_rate(code_rate)?,
                },
                cell_id,
            },
        })
    }

    #[getter]
    fn guard(&self) -> &'static str {
        guard_str(self.inner.guard())
    }
    #[getter]
    fn constellation(&self) -> PyResult<&'static str> {
        constellation_str(self.inner.constellation())
    }
    #[getter]
    fn code_rate(&self) -> &'static str {
        rate_str(self.inner.code_rate())
    }
    #[getter]
    fn cell_id(&self) -> u16 {
        self.inner.cell_id
    }
}

/// A modulated DVB-T super-frame: the IQ of four consecutive frames plus the
/// numerology a receiver needs to re-slice them.
#[pyclass(name = "DvbTSuperFrame")]
pub struct PyDvbTSuperFrame {
    iq: Vec<Complex32>,
    #[pyo3(get)]
    symbols_per_frame: usize,
    #[pyo3(get)]
    samples_per_symbol: usize,
    frame_payload_lens: [usize; 4],
}

#[pymethods]
impl PyDvbTSuperFrame {
    /// The time-domain IQ of all four frames, concatenated (no preamble).
    #[getter]
    fn iq<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<Complex32>> {
        self.iq.clone().into_pyarray(py)
    }
    /// The payload byte count carried by each of the four frames, in order.
    #[getter]
    fn frame_payload_lens(&self) -> [usize; 4] {
        self.frame_payload_lens
    }
    /// OFDM symbols across the whole super-frame (`4 · symbols_per_frame`).
    #[getter]
    fn n_symbols(&self) -> usize {
        4 * self.symbols_per_frame
    }
}

/// The recovered contents of a DVB-T super-frame: the concatenated payload and
/// the reassembled 16-bit cell id.
#[pyclass(name = "DvbTRxSuperFrame")]
pub struct PyDvbTRxSuperFrame {
    payload: Vec<u8>,
    #[pyo3(get)]
    cell_id: u16,
}

#[pymethods]
impl PyDvbTRxSuperFrame {
    /// The four frames' payloads, concatenated.
    #[getter]
    fn payload<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new(py, &self.payload)
    }
}

/// A conformant DVB-T super-frame modulator (four frames, alternating TPS sync +
/// a 16-bit cell id split across them). Built from `DvbTSuperFrameParams`;
/// `modulate(payload)` produces one `DvbTSuperFrame` per call.
#[pyclass(name = "DvbTSuperFrameMod")]
pub struct PyDvbTSuperFrameMod {
    inner: DvbTSuperFrameMod,
}

#[pymethods]
impl PyDvbTSuperFrameMod {
    #[new]
    fn new(params: &PyDvbTSuperFrameParams) -> Self {
        Self {
            inner: DvbTSuperFrameMod::new(params.inner),
        }
    }

    /// Returns a modulator that tapers every symbol of every constituent frame
    /// (see `DvbTFrameMod.with_symbol_window`). The taper is per-symbol, so it
    /// simply propagates to each frame.
    fn with_symbol_window(&self, roll_off: usize) -> Self {
        Self {
            inner: self.inner.clone().with_symbol_window(roll_off),
        }
    }

    /// Returns a modulator that applies a TX baseband mask to the super-frame
    /// (see `DvbTFrameMod.with_tx_lowpass` for sizing).
    ///
    /// Note the scope: unlike the taper, the mask runs **once over the four
    /// concatenated frames**, not per frame. The three interior frame seams are
    /// continuous on air, and filtering each frame alone would leave the
    /// filter's edge transient at every one of them.
    #[pyo3(signature = (num_taps, stopband_db = 60.0))]
    fn with_tx_lowpass(&self, num_taps: usize, stopband_db: f32) -> Self {
        Self {
            inner: self
                .inner
                .clone()
                .with_tx_lowpass(DvbTFrameMod::tx_lowpass_for_2k(num_taps, stopband_db)),
        }
    }

    /// Modulates `payload` into one conformant DVB-T super-frame. Returns a
    /// `DvbTSuperFrame` (`.iq`, `.symbols_per_frame`, `.samples_per_symbol`,
    /// `.frame_payload_lens`).
    fn modulate(&self, payload: PyReadonlyArray1<'_, u8>) -> PyResult<PyDvbTSuperFrame> {
        let sf = self.inner.modulate(payload.as_slice()?);
        Ok(PyDvbTSuperFrame {
            iq: sf.iq,
            symbols_per_frame: sf.symbols_per_frame,
            samples_per_symbol: sf.samples_per_symbol,
            frame_payload_lens: sf.frame_payload_lens,
        })
    }
}

/// A conformant DVB-T super-frame demodulator. Built from `DvbTSuperFrameParams`;
/// `decode(iq, symbols_per_frame, frame_payload_lens)` recovers one super-frame.
/// Integer-CFO correction is off by default — enable it with
/// `with_integer_cfo_correction(True)` (delegated to each constituent frame).
#[pyclass(name = "DvbTSuperFrameDemod")]
pub struct PyDvbTSuperFrameDemod {
    inner: DvbTSuperFrameDemod,
}

#[pymethods]
impl PyDvbTSuperFrameDemod {
    #[new]
    fn new(params: &PyDvbTSuperFrameParams) -> Self {
        Self {
            inner: DvbTSuperFrameDemod::new(params.inner),
        }
    }

    /// Returns a super-frame demod with internal integer-CFO correction enabled
    /// (or disabled) on every constituent frame.
    fn with_integer_cfo_correction(&self, on: bool) -> Self {
        Self {
            inner: self.inner.clone().with_integer_cfo_correction(on),
        }
    }

    /// Whether internal integer-CFO correction is enabled.
    #[getter]
    fn integer_cfo_correction(&self) -> bool {
        self.inner.integer_cfo_correction()
    }

    /// Returns a super-frame demod with the FFT-window back-off applied to every
    /// constituent frame (see `DvbTFrameDemod.with_rx_window_backoff`).
    fn with_rx_window_backoff(&self, backoff: usize) -> Self {
        Self {
            inner: self.inner.clone().with_rx_window_backoff(backoff),
        }
    }

    /// The receiver FFT-window back-off in samples.
    #[getter]
    fn rx_window_backoff(&self) -> usize {
        self.inner.rx_window_backoff()
    }

    /// Demodulates one conformant DVB-T super-frame from `iq`. `symbols_per_frame`
    /// and `frame_payload_lens` come from the paired `DvbTSuperFrameMod.modulate`
    /// result. Verifies the frame-number sequence 0,1,2,3, reassembles the 16-bit
    /// cell id, and concatenates the payloads. Raises `ValueError` on failure. The
    /// returned `DvbTRxSuperFrame` exposes `.payload` (bytes) and `.cell_id`.
    fn decode(
        &self,
        iq: PyReadonlyArray1<'_, Complex32>,
        symbols_per_frame: usize,
        frame_payload_lens: [usize; 4],
    ) -> PyResult<PyDvbTRxSuperFrame> {
        let rx = self
            .inner
            .decode(iq.as_slice()?, symbols_per_frame, frame_payload_lens)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(PyDvbTRxSuperFrame {
            payload: rx.payload,
            cell_id: rx.cell_id,
        })
    }
}

// ── Streaming receiver ──────────────────────────────────────────────────────

/// A streaming DVB-T receiver. Push IQ with `feed()`; it guard-interval-acquires
/// and decodes each fixed-size frame as its samples arrive, returning the
/// completed ones. `flush()` runs a final pass over the residual buffer. Pass
/// `integer_cfo_correction=True` to enable internal integer-CFO correction on each
/// decoded frame, and `rx_window_backoff=b` to receive a spectrally-shaped stream
/// (both are link-constant knobs, set once here).
#[pyclass(name = "DvbTFrameStreamDemod")]
pub struct PyDvbTFrameStreamDemod {
    inner: DvbTFrameStreamDemod,
    integer_cfo: bool,
    rx_window_backoff: usize,
}

#[pymethods]
impl PyDvbTFrameStreamDemod {
    /// Builds a receiver for a link whose frames are `n_symbols` OFDM symbols
    /// carrying `payload_len` payload bytes each, under `params`. When
    /// `integer_cfo_correction` is `True`, each frame's whole-subcarrier CFO is
    /// estimated and removed internally before decoding. `rx_window_backoff`
    /// slides each symbol's FFT window that many samples earlier into the guard
    /// — set it to match a transmitter running `with_symbol_window` and/or
    /// `with_tx_lowpass`. Frame acquisition is unaffected either way.
    #[new]
    #[pyo3(signature = (params, n_symbols, payload_len, integer_cfo_correction = false, rx_window_backoff = 0))]
    fn new(
        params: &PyDvbTFrameParams,
        n_symbols: usize,
        payload_len: usize,
        integer_cfo_correction: bool,
        rx_window_backoff: usize,
    ) -> Self {
        Self {
            inner: DvbTFrameStreamDemod::new(params.inner, n_symbols, payload_len)
                .with_integer_cfo_correction(integer_cfo_correction)
                .with_rx_window_backoff(rx_window_backoff),
            integer_cfo: integer_cfo_correction,
            rx_window_backoff,
        }
    }

    /// Whether internal integer-CFO correction is enabled.
    #[getter]
    fn integer_cfo_correction(&self) -> bool {
        self.integer_cfo
    }

    /// The receiver FFT-window back-off in samples.
    #[getter]
    fn rx_window_backoff(&self) -> usize {
        self.rx_window_backoff
    }

    /// Feeds IQ and returns the frames that completed. Frames that failed to
    /// decode are omitted; use `feed_with_errors` to see the reasons.
    fn feed(&mut self, iq: PyReadonlyArray1<'_, Complex32>) -> PyResult<Vec<PyDvbTRxFrame>> {
        Ok(self
            .inner
            .feed(iq.as_slice()?)
            .into_iter()
            .filter_map(|r| {
                r.ok().map(|f| PyDvbTRxFrame {
                    payload: f.payload,
                    tps: PyTpsWord { inner: f.tps },
                })
            })
            .collect())
    }

    /// Like `feed`, but returns a list of `(frame_or_None, error_or_None)` tuples
    /// so callers can observe decode failures.
    fn feed_with_errors(
        &mut self,
        iq: PyReadonlyArray1<'_, Complex32>,
    ) -> PyResult<Vec<(Option<PyDvbTRxFrame>, Option<String>)>> {
        Ok(self
            .inner
            .feed(iq.as_slice()?)
            .into_iter()
            .map(|r| match r {
                Ok(f) => (
                    Some(PyDvbTRxFrame {
                        payload: f.payload,
                        tps: PyTpsWord { inner: f.tps },
                    }),
                    None,
                ),
                Err(e) => (None, Some(e.to_string())),
            })
            .collect())
    }

    /// Runs a final decode pass over the residual buffer.
    fn flush(&mut self) -> Vec<PyDvbTRxFrame> {
        self.inner
            .flush()
            .into_iter()
            .filter_map(|r| {
                r.ok().map(|f| PyDvbTRxFrame {
                    payload: f.payload,
                    tps: PyTpsWord { inner: f.tps },
                })
            })
            .collect()
    }

    /// Number of accumulated (not-yet-consumed) samples.
    #[getter]
    fn buffered(&self) -> usize {
        self.inner.len()
    }

    /// Discards the accumulated buffer.
    fn clear(&mut self) {
        self.inner.clear();
    }
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyDvbTFrameParams>()?;
    m.add_class::<PyDvbTFrame>()?;
    m.add_class::<PyDvbTRxFrame>()?;
    m.add_class::<PyTpsWord>()?;
    m.add_class::<PyDvbTFrameMod>()?;
    m.add_class::<PyDvbTFrameDemod>()?;
    m.add_class::<PyDvbTSuperFrameParams>()?;
    m.add_class::<PyDvbTSuperFrame>()?;
    m.add_class::<PyDvbTRxSuperFrame>()?;
    m.add_class::<PyDvbTSuperFrameMod>()?;
    m.add_class::<PyDvbTSuperFrameDemod>()?;
    m.add_class::<PyDvbTFrameStreamDemod>()?;
    m.add_function(wrap_pyfunction!(nb_bandwidth_fs, m)?)?;
    m.add_function(wrap_pyfunction!(nb_bandwidth_occupied_hz, m)?)?;
    m.add_function(wrap_pyfunction!(dvb_t_cp_len, m)?)?;
    m.add_function(wrap_pyfunction!(dvb_t_max_rx_window_backoff, m)?)?;
    m.add_function(wrap_pyfunction!(dvb_t_tx_lowpass_suggested_taps, m)?)?;
    m.add_function(wrap_pyfunction!(dvb_t_tx_lowpass_group_delay, m)?)?;
    m.add_function(wrap_pyfunction!(dvb_t_tx_lowpass_fits_guard, m)?)?;
    Ok(())
}
