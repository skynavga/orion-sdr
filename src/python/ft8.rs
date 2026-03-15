// src/python/ft8.rs — PyO3 bindings for FT8/FT4 waveform, codec, sync, and message layers.

use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList};
use numpy::{PyReadonlyArray1, PyArray1, IntoPyArray};
use num_complex::Complex32;

use crate::modulate::ft8::{Ft8Frame, Ft8Mod, FT8_DATA_SYMS, FT8_FRAME_LEN};
use crate::modulate::ft4::{Ft4Frame, Ft4Mod, FT4_DATA_SYMS, FT4_FRAME_LEN};
use crate::demodulate::ft8::Ft8Demod;
use crate::demodulate::ft4::Ft4Demod;
use crate::codec::ft8::Ft8Codec;
use crate::codec::ft4::Ft4Codec;
use crate::codec::ldpc::N;
use crate::message::{CallsignHashTable, Ft8Message, GridField, gridfield_to_str, pack77, unpack77};
use crate::message::message::NonstdExtra;

// ── Ft8Mod ────────────────────────────────────────────────────────────────────

#[pyclass(name = "Ft8Mod")]
pub struct PyFt8Mod(Ft8Mod);

#[pymethods]
impl PyFt8Mod {
    #[new]
    fn new(fs: f32, base_hz: f32, rf_hz: f32, gain: f32) -> Self {
        Self(Ft8Mod::new(fs, base_hz, rf_hz, gain))
    }

    /// Modulate 58 tone indices into a 151 680-sample complex64 IQ waveform.
    fn modulate<'py>(
        &self,
        py: Python<'py>,
        tones: PyReadonlyArray1<'py, u8>,
    ) -> PyResult<Bound<'py, PyArray1<Complex32>>> {
        let t = tones.as_slice()?;
        if t.len() != FT8_DATA_SYMS {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Ft8Mod.modulate: expected {} tone indices, got {}",
                FT8_DATA_SYMS,
                t.len()
            )));
        }
        let mut arr = [0u8; FT8_DATA_SYMS];
        arr.copy_from_slice(t);
        let frame = Ft8Frame::new(arr);
        let iq = self.0.modulate(&frame);
        Ok(iq.into_pyarray(py))
    }
}

// ── Ft8Demod ──────────────────────────────────────────────────────────────────

#[pyclass(name = "Ft8Demod")]
pub struct PyFt8Demod(Ft8Demod);

#[pymethods]
impl PyFt8Demod {
    #[new]
    fn new(fs: f32, base_hz: f32) -> Self {
        Self(Ft8Demod::new(fs, base_hz))
    }

    /// Demodulate a 151 680-sample IQ block into 58 tone indices (uint8).
    fn demodulate<'py>(
        &self,
        py: Python<'py>,
        iq: PyReadonlyArray1<'py, Complex32>,
    ) -> PyResult<Bound<'py, PyArray1<u8>>> {
        let input = iq.as_slice()?;
        if input.len() < FT8_FRAME_LEN {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Ft8Demod.demodulate: input too short ({} < {})",
                input.len(),
                FT8_FRAME_LEN
            )));
        }
        match self.0.demodulate(input) {
            Some(frame) => Ok(frame.0.to_vec().into_pyarray(py)),
            None => Err(pyo3::exceptions::PyValueError::new_err(
                "Ft8Demod.demodulate: demodulation failed",
            )),
        }
    }
}

// ── Ft8Codec ──────────────────────────────────────────────────────────────────

#[pyclass(name = "Ft8Codec")]
pub struct PyFt8Codec;

#[pymethods]
impl PyFt8Codec {
    #[new]
    fn new() -> Self { Self }

    /// Encode a 10-byte payload into 58 Gray-coded tone indices (uint8).
    #[staticmethod]
    fn encode<'py>(py: Python<'py>, payload: &[u8]) -> PyResult<Bound<'py, PyArray1<u8>>> {
        if payload.len() != 10 {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Ft8Codec.encode: expected 10-byte payload, got {}",
                payload.len()
            )));
        }
        let mut arr = [0u8; 10];
        arr.copy_from_slice(payload);
        let frame = Ft8Codec::encode(&arr);
        Ok(frame.0.to_vec().into_pyarray(py))
    }

    /// Hard-decision decode 58 tone indices → bytes[10] or None.
    #[staticmethod]
    fn decode_hard<'py>(
        py: Python<'py>,
        tones: PyReadonlyArray1<'_, u8>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let t = tones.as_slice()?;
        if t.len() != FT8_DATA_SYMS {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Ft8Codec.decode_hard: expected {} tones, got {}",
                FT8_DATA_SYMS,
                t.len()
            )));
        }
        let mut arr = [0u8; FT8_DATA_SYMS];
        arr.copy_from_slice(t);
        let frame = Ft8Frame::new(arr);
        match Ft8Codec::decode_hard(&frame) {
            Some(p) => Ok(PyBytes::new(py, &p).into_any()),
            None => Ok(py.None().into_bound(py)),
        }
    }

    /// Soft-decision decode float32[174] LLRs → bytes[10] or None.
    #[staticmethod]
    fn decode_soft<'py>(
        py: Python<'py>,
        llr: PyReadonlyArray1<'_, f32>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let l = llr.as_slice()?;
        if l.len() != N {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Ft8Codec.decode_soft: expected {} LLRs, got {}",
                N,
                l.len()
            )));
        }
        let mut arr = [0.0f32; N];
        arr.copy_from_slice(l);
        match Ft8Codec::decode_soft(&arr) {
            Some(p) => Ok(PyBytes::new(py, &p).into_any()),
            None => Ok(py.None().into_bound(py)),
        }
    }
}

// ── Ft4Mod ────────────────────────────────────────────────────────────────────

#[pyclass(name = "Ft4Mod")]
pub struct PyFt4Mod(Ft4Mod);

#[pymethods]
impl PyFt4Mod {
    #[new]
    fn new(fs: f32, base_hz: f32, rf_hz: f32, gain: f32) -> Self {
        Self(Ft4Mod::new(fs, base_hz, rf_hz, gain))
    }

    /// Modulate 87 tone indices into a 60 480-sample complex64 IQ waveform.
    fn modulate<'py>(
        &self,
        py: Python<'py>,
        tones: PyReadonlyArray1<'py, u8>,
    ) -> PyResult<Bound<'py, PyArray1<Complex32>>> {
        let t = tones.as_slice()?;
        if t.len() != FT4_DATA_SYMS {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Ft4Mod.modulate: expected {} tone indices, got {}",
                FT4_DATA_SYMS,
                t.len()
            )));
        }
        let mut arr = [0u8; FT4_DATA_SYMS];
        arr.copy_from_slice(t);
        let frame = Ft4Frame::new(arr);
        let iq = self.0.modulate(&frame);
        Ok(iq.into_pyarray(py))
    }
}

// ── Ft4Demod ──────────────────────────────────────────────────────────────────

#[pyclass(name = "Ft4Demod")]
pub struct PyFt4Demod(Ft4Demod);

#[pymethods]
impl PyFt4Demod {
    #[new]
    fn new(fs: f32, base_hz: f32) -> Self {
        Self(Ft4Demod::new(fs, base_hz))
    }

    /// Demodulate a 60 480-sample IQ block into 87 tone indices (uint8).
    fn demodulate<'py>(
        &self,
        py: Python<'py>,
        iq: PyReadonlyArray1<'py, Complex32>,
    ) -> PyResult<Bound<'py, PyArray1<u8>>> {
        let input = iq.as_slice()?;
        if input.len() < FT4_FRAME_LEN {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Ft4Demod.demodulate: input too short ({} < {})",
                input.len(),
                FT4_FRAME_LEN
            )));
        }
        match self.0.demodulate(input) {
            Some(frame) => Ok(frame.0.to_vec().into_pyarray(py)),
            None => Err(pyo3::exceptions::PyValueError::new_err(
                "Ft4Demod.demodulate: demodulation failed",
            )),
        }
    }
}

// ── Ft4Codec ──────────────────────────────────────────────────────────────────

#[pyclass(name = "Ft4Codec")]
pub struct PyFt4Codec;

#[pymethods]
impl PyFt4Codec {
    #[new]
    fn new() -> Self { Self }

    /// Encode a 10-byte payload into 87 Gray-coded tone indices (uint8).
    #[staticmethod]
    fn encode<'py>(py: Python<'py>, payload: &[u8]) -> PyResult<Bound<'py, PyArray1<u8>>> {
        if payload.len() != 10 {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Ft4Codec.encode: expected 10-byte payload, got {}",
                payload.len()
            )));
        }
        let mut arr = [0u8; 10];
        arr.copy_from_slice(payload);
        let frame = Ft4Codec::encode(&arr);
        Ok(frame.0.to_vec().into_pyarray(py))
    }

    /// Hard-decision decode 87 tone indices → bytes[10] or None.
    #[staticmethod]
    fn decode_hard<'py>(
        py: Python<'py>,
        tones: PyReadonlyArray1<'_, u8>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let t = tones.as_slice()?;
        if t.len() != FT4_DATA_SYMS {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Ft4Codec.decode_hard: expected {} tones, got {}",
                FT4_DATA_SYMS,
                t.len()
            )));
        }
        let mut arr = [0u8; FT4_DATA_SYMS];
        arr.copy_from_slice(t);
        let frame = Ft4Frame::new(arr);
        match Ft4Codec::decode_hard(&frame) {
            Some(p) => Ok(PyBytes::new(py, &p).into_any()),
            None => Ok(py.None().into_bound(py)),
        }
    }

    /// Soft-decision decode float32[174] LLRs → bytes[10] or None.
    #[staticmethod]
    fn decode_soft<'py>(
        py: Python<'py>,
        llr: PyReadonlyArray1<'_, f32>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let l = llr.as_slice()?;
        if l.len() != N {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Ft4Codec.decode_soft: expected {} LLRs, got {}",
                N,
                l.len()
            )));
        }
        let mut arr = [0.0f32; N];
        arr.copy_from_slice(l);
        match Ft4Codec::decode_soft(&arr) {
            Some(p) => Ok(PyBytes::new(py, &p).into_any()),
            None => Ok(py.None().into_bound(py)),
        }
    }
}

// ── ft8_sync ──────────────────────────────────────────────────────────────────

/// Synchronise an FT8 IQ buffer and return up to `max_cand` candidates.
///
/// Returns a list of dicts:
///   {"time_sym": int, "freq_bin": int, "score": float, "llr": float32[174]}
#[pyfunction]
#[pyo3(signature = (iq, fs, base_hz, max_hz, t_min, t_max, max_cand))]
pub fn ft8_sync<'py>(
    py: Python<'py>,
    iq: PyReadonlyArray1<'py, Complex32>,
    fs: f32,
    base_hz: f32,
    max_hz: f32,
    t_min: i32,
    t_max: i32,
    max_cand: usize,
) -> PyResult<Bound<'py, PyList>> {
    let input = iq.as_slice()?;
    let results = crate::sync::ft8_sync::ft8_sync(input, fs, base_hz, max_hz, t_min, t_max, max_cand);

    let list = PyList::empty(py);
    for r in results {
        let d = PyDict::new(py);
        d.set_item("time_sym", r.time_sym)?;
        d.set_item("freq_bin", r.freq_bin)?;
        d.set_item("score", r.score)?;
        let llr_arr = r.llr.to_vec().into_pyarray(py);
        d.set_item("llr", llr_arr)?;
        list.append(d)?;
    }
    Ok(list)
}

// ── ft4_sync ──────────────────────────────────────────────────────────────────

/// Synchronise an FT4 IQ buffer and return up to `max_cand` candidates.
///
/// Returns a list of dicts:
///   {"time_sym": int, "freq_bin": int, "score": float, "llr": float32[174]}
#[pyfunction]
#[pyo3(signature = (iq, fs, base_hz, max_hz, t_min, t_max, max_cand))]
pub fn ft4_sync<'py>(
    py: Python<'py>,
    iq: PyReadonlyArray1<'py, Complex32>,
    fs: f32,
    base_hz: f32,
    max_hz: f32,
    t_min: i32,
    t_max: i32,
    max_cand: usize,
) -> PyResult<Bound<'py, PyList>> {
    let input = iq.as_slice()?;
    let results = crate::sync::ft4_sync::ft4_sync(input, fs, base_hz, max_hz, t_min, t_max, max_cand);

    let list = PyList::empty(py);
    for r in results {
        let d = PyDict::new(py);
        d.set_item("time_sym", r.time_sym)?;
        d.set_item("freq_bin", r.freq_bin)?;
        d.set_item("score", r.score)?;
        let llr_arr = r.llr.to_vec().into_pyarray(py);
        d.set_item("llr", llr_arr)?;
        list.append(d)?;
    }
    Ok(list)
}

// ── Message functions ─────────────────────────────────────────────────────────

/// Pack a standard FT8 message (two callsigns + extra) → bytes[10].
///
/// `extra` may be a Maidenhead grid ("FN31"), signal report ("+07", "-12"),
/// R-prefixed report ("R+05"), or token ("RRR", "RR73", "73").
#[pyfunction]
pub fn ft8_pack_standard<'py>(
    py: Python<'py>,
    call_to: &str,
    call_de: &str,
    extra: &str,
) -> PyResult<Bound<'py, PyBytes>> {
    let gf = str_to_gridfield(extra);
    let msg = Ft8Message::Standard {
        call_to: call_to.to_string(),
        call_de: call_de.to_string(),
        extra: gf,
    };
    let mut ht = CallsignHashTable::new();
    match pack77(&msg, &mut ht) {
        Some(p) => Ok(PyBytes::new(py, &p)),
        None => Err(pyo3::exceptions::PyValueError::new_err(
            "ft8_pack_standard: failed to pack message (invalid callsign?)",
        )),
    }
}

/// Pack a free-text FT8 message (up to 13 chars, base-42) → bytes[10].
#[pyfunction]
pub fn ft8_pack_free_text<'py>(py: Python<'py>, text: &str) -> PyResult<Bound<'py, PyBytes>> {
    let msg = Ft8Message::FreeText(text.to_string());
    let mut ht = CallsignHashTable::new();
    match pack77(&msg, &mut ht) {
        Some(p) => Ok(PyBytes::new(py, &p)),
        None => Err(pyo3::exceptions::PyValueError::new_err(
            "ft8_pack_free_text: failed to pack message (text too long or invalid chars?)",
        )),
    }
}

/// Pack a telemetry FT8 message (9 bytes of arbitrary data) → bytes[10].
#[pyfunction]
pub fn ft8_pack_telemetry<'py>(py: Python<'py>, data: &[u8]) -> PyResult<Bound<'py, PyBytes>> {
    if data.len() != 9 {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "ft8_pack_telemetry: expected 9 bytes, got {}",
            data.len()
        )));
    }
    let mut arr = [0u8; 9];
    arr.copy_from_slice(data);
    let msg = Ft8Message::Telemetry(arr);
    let mut ht = CallsignHashTable::new();
    match pack77(&msg, &mut ht) {
        Some(p) => Ok(PyBytes::new(py, &p)),
        None => Err(pyo3::exceptions::PyValueError::new_err(
            "ft8_pack_telemetry: pack failed",
        )),
    }
}

/// Unpack a 10-byte FT8/FT4 payload → dict.
///
/// Standard:   {"type": "standard", "call_to": str, "call_de": str, "extra": str}
/// FreeText:   {"type": "free_text", "text": str}
/// Telemetry:  {"type": "telemetry", "data": bytes}
/// Unknown:    {"type": "unknown", "payload": bytes}
#[pyfunction]
pub fn ft8_unpack<'py>(py: Python<'py>, payload: &[u8]) -> PyResult<Bound<'py, PyDict>> {
    if payload.len() != 10 {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "ft8_unpack: expected 10 bytes, got {}",
            payload.len()
        )));
    }
    let mut arr = [0u8; 10];
    arr.copy_from_slice(payload);
    let ht = CallsignHashTable::new();
    let msg = unpack77(&arr, &ht);

    let d = PyDict::new(py);
    match msg {
        Ft8Message::Standard { call_to, call_de, extra } => {
            d.set_item("type", "standard")?;
            d.set_item("call_to", call_to)?;
            d.set_item("call_de", call_de)?;
            d.set_item("extra", gridfield_to_str(&extra))?;
        }
        Ft8Message::FreeText(text) => {
            d.set_item("type", "free_text")?;
            d.set_item("text", text)?;
        }
        Ft8Message::Telemetry(data) => {
            d.set_item("type", "telemetry")?;
            d.set_item("data", PyBytes::new(py, &data))?;
        }
        Ft8Message::NonStd { call_to, call_de, extra } => {
            d.set_item("type", "nonstd")?;
            d.set_item("call_to", call_to)?;
            d.set_item("call_de", call_de)?;
            let extra_str = match extra {
                NonstdExtra::RRR      => "RRR",
                NonstdExtra::RR73     => "RR73",
                NonstdExtra::Seventy3 => "73",
                NonstdExtra::None     => "",
            };
            d.set_item("extra", extra_str)?;
        }
        Ft8Message::Unknown(p) => {
            d.set_item("type", "unknown")?;
            d.set_item("payload", PyBytes::new(py, &p))?;
        }
    }
    Ok(d)
}

// ── helpers ───────────────────────────────────────────────────────────────────

fn str_to_gridfield(s: &str) -> GridField {
    match s {
        "RRR"  => GridField::RRR,
        "RR73" => GridField::RR73,
        "73"   => GridField::Seventy3,
        ""     => GridField::None,
        _ => {
            let bytes = s.as_bytes();
            // R-prefixed report ("R+07", "R-12")
            if bytes.first() == Some(&b'R') && s.len() >= 2 {
                if let Ok(v) = s[1..].parse::<i8>() {
                    return GridField::RReport(v);
                }
            }
            // Plain signal report ("+07", "-12")
            if s.starts_with('+') || s.starts_with('-') {
                if let Ok(v) = s.parse::<i8>() {
                    return GridField::Report(v);
                }
            }
            // Otherwise treat as Maidenhead grid
            GridField::Grid(s.to_string())
        }
    }
}
