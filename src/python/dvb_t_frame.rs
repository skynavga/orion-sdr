// Copyright (c) 2026 G & R Associates LLC
// SPDX-License-Identifier: MIT OR Apache-2.0

// src/python/dvb_t_frame.rs — PyO3 bindings for the conformant, preamble-less
// DVB-T on-air frame (ETSI EN 300 744).
//
// Exposes the batch frame modulator/demodulator
// (`modulate::dvb_t_frame_modulate` / `demodulate::dvb_t_frame_demodulate`),
// the shared `DvbTFrameParams`, the recovered `DvbTRxFrame`/`TpsWord`, and the
// `NbBandwidth` sample-rate helper. Transmission parameters (guard interval,
// constellation, code rate) are passed as strings, matching the convention used
// by the config-level DVB-T bindings in `python/ofdm.rs`.

use num_complex::Complex32;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;

use crate::demodulate::{
    DvbTFrameStreamDemod, dvb_t_frame_demodulate, dvb_t_super_frame_demodulate,
};
use crate::fec::PunctureRate;
use crate::modulate::{
    ConstellationOrder, DvbTSuperFrameParams, dvb_t_frame_modulate, dvb_t_super_frame_modulate,
};
use crate::waveform::dvb_t::{DvbTFrameParams, GuardInterval, NbBandwidth};
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
                guard: parse_guard(guard)?,
                constellation: parse_dvb_t_constellation(constellation)?,
                code_rate: parse_rate(code_rate)?,
                frame_number,
                cell_id,
            },
        })
    }

    #[getter]
    fn guard(&self) -> &'static str {
        guard_str(self.inner.guard)
    }
    #[getter]
    fn constellation(&self) -> PyResult<&'static str> {
        constellation_str(self.inner.constellation)
    }
    #[getter]
    fn code_rate(&self) -> &'static str {
        rate_str(self.inner.code_rate)
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
            guard_str(self.inner.guard),
            constellation_str(self.inner.constellation)?,
            rate_str(self.inner.code_rate),
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

// ── Free functions ──────────────────────────────────────────────────────────

/// Modulates `payload` (the MPEG-TS payload bytes) into one conformant,
/// preamble-less DVB-T frame. Returns a `DvbTFrame` (`.iq`, `.n_symbols`,
/// `.samples_per_symbol`).
#[pyfunction]
#[pyo3(name = "dvb_t_frame_modulate")]
fn dvb_t_frame_modulate_py(
    params: &PyDvbTFrameParams,
    payload: PyReadonlyArray1<'_, u8>,
) -> PyResult<PyDvbTFrame> {
    let frame = dvb_t_frame_modulate(params.inner, payload.as_slice()?);
    Ok(PyDvbTFrame {
        iq: frame.iq,
        n_symbols: frame.n_symbols,
        samples_per_symbol: frame.samples_per_symbol,
    })
}

/// Demodulates one conformant DVB-T frame from `iq`, acquiring the symbol grid
/// from the guard interval (no preamble). `params` supplies the cold-start MCS;
/// `n_symbols` is the frame's symbol count (from the paired modulate call's
/// `DvbTFrame.n_symbols`); `payload_len` is the original payload byte count for
/// trimming. Raises `ValueError` on any acquisition/decode failure. The returned
/// `DvbTRxFrame` exposes `.payload` (bytes) and `.tps` (the recovered `TpsWord`).
#[pyfunction]
#[pyo3(name = "dvb_t_frame_demodulate")]
fn dvb_t_frame_demodulate_py(
    params: &PyDvbTFrameParams,
    iq: PyReadonlyArray1<'_, Complex32>,
    n_symbols: usize,
    payload_len: usize,
) -> PyResult<PyDvbTRxFrame> {
    let rx = dvb_t_frame_demodulate(params.inner, iq.as_slice()?, n_symbols, payload_len)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(PyDvbTRxFrame {
        payload: rx.payload,
        tps: PyTpsWord { inner: rx.tps },
    })
}

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
                guard: parse_guard(guard)?,
                constellation: parse_dvb_t_constellation(constellation)?,
                code_rate: parse_rate(code_rate)?,
                cell_id,
            },
        })
    }

    #[getter]
    fn guard(&self) -> &'static str {
        guard_str(self.inner.guard)
    }
    #[getter]
    fn constellation(&self) -> PyResult<&'static str> {
        constellation_str(self.inner.constellation)
    }
    #[getter]
    fn code_rate(&self) -> &'static str {
        rate_str(self.inner.code_rate)
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

/// Modulates `payload` into one conformant DVB-T super-frame (four frames,
/// alternating TPS sync + a 16-bit cell id split across them). Returns a
/// `DvbTSuperFrame` (`.iq`, `.symbols_per_frame`, `.samples_per_symbol`,
/// `.frame_payload_lens`).
#[pyfunction]
#[pyo3(name = "dvb_t_super_frame_modulate")]
fn dvb_t_super_frame_modulate_py(
    params: &PyDvbTSuperFrameParams,
    payload: PyReadonlyArray1<'_, u8>,
) -> PyResult<PyDvbTSuperFrame> {
    let sf = dvb_t_super_frame_modulate(params.inner, payload.as_slice()?);
    Ok(PyDvbTSuperFrame {
        iq: sf.iq,
        symbols_per_frame: sf.symbols_per_frame,
        samples_per_symbol: sf.samples_per_symbol,
        frame_payload_lens: sf.frame_payload_lens,
    })
}

/// Demodulates one conformant DVB-T super-frame from `iq`. `symbols_per_frame`
/// and `frame_payload_lens` come from the paired `dvb_t_super_frame_modulate`
/// result. Verifies the frame-number sequence 0,1,2,3, reassembles the 16-bit
/// cell id, and concatenates the payloads. Raises `ValueError` on failure. The
/// returned `DvbTRxSuperFrame` exposes `.payload` (bytes) and `.cell_id`.
#[pyfunction]
#[pyo3(name = "dvb_t_super_frame_demodulate")]
fn dvb_t_super_frame_demodulate_py(
    params: &PyDvbTSuperFrameParams,
    iq: PyReadonlyArray1<'_, Complex32>,
    symbols_per_frame: usize,
    frame_payload_lens: [usize; 4],
) -> PyResult<PyDvbTRxSuperFrame> {
    let rx = dvb_t_super_frame_demodulate(
        params.inner,
        iq.as_slice()?,
        symbols_per_frame,
        frame_payload_lens,
    )
    .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(PyDvbTRxSuperFrame {
        payload: rx.payload,
        cell_id: rx.cell_id,
    })
}

// ── Streaming receiver ──────────────────────────────────────────────────────

/// A streaming DVB-T receiver. Push IQ with `feed()`; it guard-interval-acquires
/// and decodes each fixed-size frame as its samples arrive, returning the
/// completed ones. `flush()` runs a final pass over the residual buffer.
#[pyclass(name = "DvbTFrameStreamDemod")]
pub struct PyDvbTFrameStreamDemod {
    inner: DvbTFrameStreamDemod,
}

#[pymethods]
impl PyDvbTFrameStreamDemod {
    /// Builds a receiver for a link whose frames are `n_symbols` OFDM symbols
    /// carrying `payload_len` payload bytes each, under `params`.
    #[new]
    fn new(params: &PyDvbTFrameParams, n_symbols: usize, payload_len: usize) -> Self {
        Self {
            inner: DvbTFrameStreamDemod::new(params.inner, n_symbols, payload_len),
        }
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
    m.add_class::<PyDvbTSuperFrameParams>()?;
    m.add_class::<PyDvbTSuperFrame>()?;
    m.add_class::<PyDvbTRxSuperFrame>()?;
    m.add_class::<PyDvbTFrameStreamDemod>()?;
    m.add_function(wrap_pyfunction!(dvb_t_frame_modulate_py, m)?)?;
    m.add_function(wrap_pyfunction!(dvb_t_frame_demodulate_py, m)?)?;
    m.add_function(wrap_pyfunction!(dvb_t_super_frame_modulate_py, m)?)?;
    m.add_function(wrap_pyfunction!(dvb_t_super_frame_demodulate_py, m)?)?;
    m.add_function(wrap_pyfunction!(nb_bandwidth_fs, m)?)?;
    m.add_function(wrap_pyfunction!(nb_bandwidth_occupied_hz, m)?)?;
    Ok(())
}
