// src/python/psk31.rs — PyO3 bindings for PSK31 (BPSK31 + QPSK31).

use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList};
use numpy::{PyReadonlyArray1, PyArray1, IntoPyArray};
use num_complex::Complex32;

use crate::codec::varicode::{VaricodeEncoder as RsVaricodeEncoder, VaricodeDecoder as RsVaricodeDecoder};
use crate::modulate::psk31::{Bpsk31Mod as RsBpsk31Mod, Qpsk31Mod as RsQpsk31Mod, PSK31_PREAMBLE_BITS, PSK31_POSTAMBLE_BITS};
use crate::demodulate::psk31::{Bpsk31Demod as RsBpsk31Demod, Qpsk31Demod as RsQpsk31Demod, Qpsk31Decider as RsQpsk31Decider};
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
