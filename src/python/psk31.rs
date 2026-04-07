// Copyright (c) 2026 G & R Associates LLC
// SPDX-License-Identifier: MIT OR Apache-2.0

// src/python/psk31.rs — PyO3 bindings for PSK31 (BPSK31 + QPSK31).

use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList};
use numpy::{PyReadonlyArray1, PyArray1, IntoPyArray};
use num_complex::Complex32;

use crate::codec::varicode::{VaricodeEncoder as RsVaricodeEncoder, VaricodeDecoder as RsVaricodeDecoder};
use crate::codec::psk31::Psk31Stream as RsPsk31Stream;
use crate::modulate::psk31::{Bpsk31Mod as RsBpsk31Mod, Qpsk31Mod as RsQpsk31Mod, PSK31_PREAMBLE_BITS, PSK31_POSTAMBLE_BITS, PSK31_BAUD};
use crate::demodulate::psk31::{Bpsk31Demod as RsBpsk31Demod, Bpsk31Decider as RsBpsk31Decider, Qpsk31Demod as RsQpsk31Demod, Qpsk31Decider as RsQpsk31Decider};
use crate::core::Block;

// ── VaricodeEncoder ───────────────────────────────────────────────────────────

#[pyclass(name = "VaricodeEncoder")]
pub struct PyVaricodeEncoder(RsVaricodeEncoder);

#[pymethods]
impl PyVaricodeEncoder {
    #[new]
    fn new() -> Self {
        Self(RsVaricodeEncoder::new())
    }

    fn push_preamble(&mut self, n: usize) {
        self.0.push_preamble(n);
    }

    fn push_byte(&mut self, b: u8) {
        self.0.push_byte(b);
    }

    fn push_postamble(&mut self, n: usize) {
        self.0.push_postamble(n);
    }

    /// Drain all pending bits into a uint8 numpy array.
    fn drain_bits<'py>(&mut self, py: Python<'py>) -> Bound<'py, PyArray1<u8>> {
        self.0.drain_bits().into_pyarray(py)
    }

    fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

// ── VaricodeDecoder ───────────────────────────────────────────────────────────

#[pyclass(name = "VaricodeDecoder")]
pub struct PyVaricodeDecoder(RsVaricodeDecoder);

#[pymethods]
impl PyVaricodeDecoder {
    #[new]
    fn new() -> Self {
        Self(RsVaricodeDecoder::new())
    }

    /// Feed a uint8 numpy array of bits (0/1) into the decoder.
    fn push_bits(&mut self, bits: PyReadonlyArray1<'_, u8>) -> PyResult<()> {
        let b = bits.as_slice()?;
        for &bit in b {
            self.0.push_bit(bit);
        }
        Ok(())
    }

    /// Drain all decoded bytes as a Python `bytes` object.
    fn pop_bytes<'py>(&mut self, py: Python<'py>) -> Bound<'py, PyBytes> {
        let mut chars = Vec::new();
        while let Some(c) = self.0.pop_char() {
            chars.push(c);
        }
        PyBytes::new(py, &chars)
    }
}

// ── Bpsk31Mod ─────────────────────────────────────────────────────────────────

#[pyclass(name = "Bpsk31Mod")]
pub struct PyBpsk31Mod(RsBpsk31Mod);

#[pymethods]
impl PyBpsk31Mod {
    #[new]
    fn new(fs: f32, rf_hz: f32, gain: f32) -> Self {
        Self(RsBpsk31Mod::new(fs, rf_hz, gain))
    }

    fn set_gain(&mut self, g: f32) { self.0.set_gain(g); }
    fn reset(&mut self) { self.0.reset(); }

    /// Modulate text bytes into a complex64 IQ waveform.
    #[pyo3(signature = (text, preamble_bits=PSK31_PREAMBLE_BITS, postamble_bits=PSK31_POSTAMBLE_BITS))]
    fn modulate_text<'py>(
        &mut self,
        py: Python<'py>,
        text: &[u8],
        preamble_bits: usize,
        postamble_bits: usize,
    ) -> Bound<'py, PyArray1<Complex32>> {
        self.0.modulate_text(text, preamble_bits, postamble_bits).into_pyarray(py)
    }

    /// Modulate raw differential bits into a complex64 IQ waveform.
    fn modulate_bits<'py>(
        &mut self,
        py: Python<'py>,
        bits: PyReadonlyArray1<'_, u8>,
    ) -> PyResult<Bound<'py, PyArray1<Complex32>>> {
        let b = bits.as_slice()?;
        Ok(self.0.modulate_bits(b).into_pyarray(py))
    }
}

// ── Bpsk31Demod ───────────────────────────────────────────────────────────────

#[pyclass(name = "Bpsk31Demod")]
pub struct PyBpsk31Demod(RsBpsk31Demod);

#[pymethods]
impl PyBpsk31Demod {
    #[new]
    fn new(fs: f32, rf_hz: f32, gain: f32) -> Self {
        Self(RsBpsk31Demod::new(fs, rf_hz, gain))
    }

    fn set_gain(&mut self, g: f32) { self.0.set_gain(g); }
    fn reset(&mut self) { self.0.reset(); }

    /// Process a complex64 IQ array and return float32 soft bits.
    fn process<'py>(
        &mut self,
        py: Python<'py>,
        iq: PyReadonlyArray1<'_, Complex32>,
    ) -> PyResult<Bound<'py, PyArray1<f32>>> {
        let input = iq.as_slice()?;
        // Output: at most one soft bit per sps samples.
        let max_out = input.len(); // generous upper bound
        let mut out = vec![0.0f32; max_out];
        let wr = self.0.process(input, &mut out);
        out.truncate(wr.out_written);
        Ok(out.into_pyarray(py))
    }
}

// ── Qpsk31Mod ─────────────────────────────────────────────────────────────────

#[pyclass(name = "Qpsk31Mod")]
pub struct PyQpsk31Mod(RsQpsk31Mod);

#[pymethods]
impl PyQpsk31Mod {
    #[new]
    fn new(fs: f32, rf_hz: f32, gain: f32) -> Self {
        Self(RsQpsk31Mod::new(fs, rf_hz, gain))
    }

    fn set_gain(&mut self, g: f32) { self.0.set_gain(g); }
    fn reset(&mut self) { self.0.reset(); }

    #[pyo3(signature = (text, preamble_bits=PSK31_PREAMBLE_BITS, postamble_bits=PSK31_POSTAMBLE_BITS))]
    fn modulate_text<'py>(
        &mut self,
        py: Python<'py>,
        text: &[u8],
        preamble_bits: usize,
        postamble_bits: usize,
    ) -> Bound<'py, PyArray1<Complex32>> {
        self.0.modulate_text(text, preamble_bits, postamble_bits).into_pyarray(py)
    }

    fn modulate_bits<'py>(
        &mut self,
        py: Python<'py>,
        bits: PyReadonlyArray1<'_, u8>,
    ) -> PyResult<Bound<'py, PyArray1<Complex32>>> {
        let b = bits.as_slice()?;
        Ok(self.0.modulate_bits(b).into_pyarray(py))
    }
}

// ── Qpsk31Demod ───────────────────────────────────────────────────────────────

#[pyclass(name = "Qpsk31Demod")]
pub struct PyQpsk31Demod {
    demod: RsQpsk31Demod,
    decider: RsQpsk31Decider,
}

#[pymethods]
impl PyQpsk31Demod {
    #[new]
    fn new(fs: f32, rf_hz: f32, gain: f32) -> Self {
        Self {
            demod: RsQpsk31Demod::new(fs, rf_hz, gain),
            decider: RsQpsk31Decider::new(),
        }
    }

    fn set_gain(&mut self, g: f32) { self.demod.set_gain(g); }

    fn reset(&mut self) {
        self.demod.reset();
        self.decider = RsQpsk31Decider::new();
    }

    /// Process IQ samples and buffer the resulting soft dibits.
    /// Returns the soft dibit float32 array (interleaved Re/Im pairs).
    fn process<'py>(
        &mut self,
        py: Python<'py>,
        iq: PyReadonlyArray1<'_, Complex32>,
    ) -> PyResult<Bound<'py, PyArray1<f32>>> {
        let input = iq.as_slice()?;
        let max_out = input.len() * 2;
        let mut soft = vec![0.0f32; max_out];
        let wr = self.demod.process(input, &mut soft);
        soft.truncate(wr.out_written);
        // Also buffer in the decider for later flush.
        self.decider.process(&soft, &mut vec![]);
        Ok(soft.into_pyarray(py))
    }

    /// Run Viterbi on accumulated soft dibits and return decoded bits.
    fn flush<'py>(&mut self, py: Python<'py>) -> Bound<'py, PyArray1<u8>> {
        let mut bits = Vec::new();
        self.decider.flush(&mut bits);
        bits.into_pyarray(py)
    }
}

// ── Bpsk31Decider ────────────────────────────────────────────────────────────

#[pyclass(name = "Bpsk31Decider")]
pub struct PyBpsk31Decider(RsBpsk31Decider);

#[pymethods]
impl PyBpsk31Decider {
    #[new]
    fn new() -> Self {
        Self(RsBpsk31Decider::new())
    }

    /// Threshold soft bits to hard decisions. Returns uint8 array.
    fn process<'py>(
        &mut self,
        py: Python<'py>,
        soft: PyReadonlyArray1<'_, f32>,
    ) -> PyResult<Bound<'py, PyArray1<u8>>> {
        let input = soft.as_slice()?;
        let mut out = vec![0u8; input.len()];
        let wr = self.0.process(input, &mut out);
        out.truncate(wr.out_written);
        Ok(out.into_pyarray(py))
    }
}

// ── Psk31Stream ──────────────────────────────────────────────────────────────

/// Streaming PSK31 decoder.
///
/// Wires demod → decider/viterbi → varicode into a single feed/flush API.
/// Use `Psk31Stream("bpsk", fs, carrier_hz, gain)` for BPSK31 or
/// `Psk31Stream("qpsk", ...)` for QPSK31.
#[pyclass(name = "Psk31Stream")]
pub struct PyPsk31Stream(RsPsk31Stream);

#[pymethods]
impl PyPsk31Stream {
    #[new]
    #[pyo3(signature = (mode, fs, carrier_hz, gain=1.0))]
    fn new(mode: &str, fs: f32, carrier_hz: f32, gain: f32) -> PyResult<Self> {
        match mode.to_lowercase().as_str() {
            "bpsk" | "bpsk31" => Ok(Self(RsPsk31Stream::new_bpsk(fs, carrier_hz, gain))),
            "qpsk" | "qpsk31" => Ok(Self(RsPsk31Stream::new_qpsk(fs, carrier_hz, gain))),
            _ => Err(pyo3::exceptions::PyValueError::new_err(
                format!("mode must be 'bpsk' or 'qpsk', got '{}'", mode),
            )),
        }
    }

    /// Feed IQ samples and return any newly decoded text.
    fn feed(&mut self, iq: PyReadonlyArray1<'_, Complex32>) -> PyResult<String> {
        let input = iq.as_slice()?;
        Ok(self.0.feed(input))
    }

    /// Flush the decoder and return any remaining text.
    fn flush(&mut self) -> String {
        self.0.flush()
    }
}

// ── best_sync ────────────────────────────────────────────────────────────────

/// Pick the best PSK31 sync result nearest to `carrier_hz`.
///
/// Takes the list of dicts returned by `psk31_sync()` and returns the best
/// candidate as a dict, or None if no candidate is within 2×baud of the carrier.
#[pyfunction]
#[pyo3(signature = (candidates, carrier_hz, baud=PSK31_BAUD))]
pub fn best_psk31_sync<'py>(
    _py: Python<'py>,
    candidates: &Bound<'py, PyList>,
    carrier_hz: f32,
    baud: f32,
) -> PyResult<Option<Bound<'py, PyDict>>> {
    // Convert Python dicts to Psk31SyncResult structs.
    let mut results = Vec::new();
    for item in candidates.iter() {
        let d = item.cast::<PyDict>()?;
        let hz: f32 = d.get_item("carrier_hz")?.unwrap().extract()?;
        let ts: usize = d.get_item("time_sym")?.unwrap().extract()?;
        let fb: usize = d.get_item("freq_bin")?.unwrap().extract()?;
        let sc: f32 = d.get_item("score")?.unwrap().extract()?;
        results.push(crate::sync::psk31_sync::Psk31SyncResult {
            carrier_hz: hz,
            time_sym: ts,
            freq_bin: fb,
            score: sc,
            soft_bits: vec![], // not needed for best_sync selection
        });
    }

    match crate::util::best_sync(&results, carrier_hz, baud) {
        Some((hz, time_sym)) => {
            // Find and return the matching original dict.
            for item in candidates.iter() {
                let d = item.cast::<PyDict>()?;
                let ts: usize = d.get_item("time_sym")?.unwrap().extract()?;
                let ch: f32 = d.get_item("carrier_hz")?.unwrap().extract()?;
                if ts == time_sym && (ch - hz).abs() < 0.01 {
                    return Ok(Some(d.clone()));
                }
            }
            Ok(None)
        }
        None => Ok(None),
    }
}

// ── psk31_sync ────────────────────────────────────────────────────────────────

/// Scan for PSK31 carriers in an IQ buffer.
///
/// Returns a list of dicts:
///   {"time_sym": int, "freq_bin": int, "carrier_hz": float, "score": float, "soft_bits": float32[N]}
#[pyfunction]
#[pyo3(signature = (iq, fs, base_hz, max_hz, min_carrier_syms=8, peak_margin_db=6.0, n_bits=1024, max_cand=10))]
pub fn psk31_sync<'py>(
    py: Python<'py>,
    iq: PyReadonlyArray1<'py, Complex32>,
    fs: f32,
    base_hz: f32,
    max_hz: f32,
    min_carrier_syms: usize,
    peak_margin_db: f32,
    n_bits: usize,
    max_cand: usize,
) -> PyResult<Bound<'py, PyList>> {
    let input = iq.as_slice()?;
    let results = crate::sync::psk31_sync::psk31_sync(
        input, fs, base_hz, max_hz,
        min_carrier_syms, peak_margin_db, n_bits, max_cand,
    );

    let list = PyList::empty(py);
    for r in results {
        let d = PyDict::new(py);
        d.set_item("time_sym",   r.time_sym)?;
        d.set_item("freq_bin",   r.freq_bin)?;
        d.set_item("carrier_hz", r.carrier_hz)?;
        d.set_item("score",      r.score)?;
        let sb = r.soft_bits.into_pyarray(py);
        d.set_item("soft_bits",  sb)?;
        list.append(d)?;
    }
    Ok(list)
}
