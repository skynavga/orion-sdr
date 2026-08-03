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

use crate::demodulate::dvb_t_frame_demodulate;
use crate::fec::PunctureRate;
use crate::modulate::{ConstellationOrder, dvb_t_frame_modulate};
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

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyDvbTFrameParams>()?;
    m.add_class::<PyDvbTFrame>()?;
    m.add_class::<PyDvbTRxFrame>()?;
    m.add_class::<PyTpsWord>()?;
    m.add_function(wrap_pyfunction!(dvb_t_frame_modulate_py, m)?)?;
    m.add_function(wrap_pyfunction!(dvb_t_frame_demodulate_py, m)?)?;
    m.add_function(wrap_pyfunction!(nb_bandwidth_fs, m)?)?;
    m.add_function(wrap_pyfunction!(nb_bandwidth_occupied_hz, m)?)?;
    Ok(())
}
