use pyo3::prelude::*;
use numpy::{PyReadonlyArray1, PyArray1, IntoPyArray};
use num_complex::Complex32;
use crate::core::Block;

// ── CwEnvelopeDemod ───────────────────────────────────────────────────────────

#[pyclass(name = "CwEnvelopeDemod")]
pub struct PyCwEnvelopeDemod(crate::demodulate::CwEnvelopeDemod);

#[pymethods]
impl PyCwEnvelopeDemod {
    #[new]
    fn new(sample_rate: f32, tone_hz: f32, env_bw_hz: f32) -> Self {
        Self(crate::demodulate::CwEnvelopeDemod::new(sample_rate, tone_hz, env_bw_hz))
    }
    fn set_gain(&mut self, g: f32) { self.0.set_gain(g); }
    fn process<'py>(&mut self, py: Python<'py>, iq: PyReadonlyArray1<'py, Complex32>)
        -> PyResult<Bound<'py, PyArray1<f32>>>
    {
        let input = iq.as_slice()?;
        let mut out = vec![0.0f32; input.len()];
        self.0.process(input, &mut out);
        Ok(out.into_pyarray(py))
    }
}

// ── AmEnvelopeDemod ───────────────────────────────────────────────────────────

#[pyclass(name = "AmEnvelopeDemod")]
pub struct PyAmEnvelopeDemod(crate::demodulate::AmEnvelopeDemod);

#[pymethods]
impl PyAmEnvelopeDemod {
    /// `abs_approx=True` uses k1=0.9482, k2=0.3920 (Proakis & Manolakis coefficients).
    #[new]
    #[pyo3(signature = (fs, audio_bw_hz, abs_approx = false))]
    fn new(fs: f32, audio_bw_hz: f32, abs_approx: bool) -> Self {
        let inner = crate::demodulate::AmEnvelopeDemod::new(fs, audio_bw_hz);
        if abs_approx {
            Self(inner.with_abs_approx(0.9482, 0.3920))
        } else {
            Self(inner)
        }
    }
    fn process<'py>(&mut self, py: Python<'py>, iq: PyReadonlyArray1<'py, Complex32>)
        -> PyResult<Bound<'py, PyArray1<f32>>>
    {
        let input = iq.as_slice()?;
        let mut out = vec![0.0f32; input.len()];
        self.0.process(input, &mut out);
        Ok(out.into_pyarray(py))
    }
}

// ── SsbProductDemod ───────────────────────────────────────────────────────────

#[pyclass(name = "SsbProductDemod")]
pub struct PySsbProductDemod(crate::demodulate::SsbProductDemod);

#[pymethods]
impl PySsbProductDemod {
    #[new]
    fn new(fs: f32, bfo_hz: f32, audio_bw_hz: f32) -> Self {
        Self(crate::demodulate::SsbProductDemod::new(fs, bfo_hz, audio_bw_hz))
    }
    fn process<'py>(&mut self, py: Python<'py>, iq: PyReadonlyArray1<'py, Complex32>)
        -> PyResult<Bound<'py, PyArray1<f32>>>
    {
        let input = iq.as_slice()?;
        let mut out = vec![0.0f32; input.len()];
        self.0.process(input, &mut out);
        Ok(out.into_pyarray(py))
    }
}

// ── FmQuadratureDemod ─────────────────────────────────────────────────────────

#[pyclass(name = "FmQuadratureDemod")]
pub struct PyFmQuadratureDemod(crate::demodulate::FmQuadratureDemod);

#[pymethods]
impl PyFmQuadratureDemod {
    #[new]
    fn new(fs: f32, dev_hz: f32, audio_bw_hz: f32) -> Self {
        Self(crate::demodulate::FmQuadratureDemod::new(fs, dev_hz, audio_bw_hz))
    }
    fn process<'py>(&mut self, py: Python<'py>, iq: PyReadonlyArray1<'py, Complex32>)
        -> PyResult<Bound<'py, PyArray1<f32>>>
    {
        let input = iq.as_slice()?;
        let mut out = vec![0.0f32; input.len()];
        self.0.process(input, &mut out);
        Ok(out.into_pyarray(py))
    }
}

// ── PmQuadratureDemod ─────────────────────────────────────────────────────────

#[pyclass(name = "PmQuadratureDemod")]
pub struct PyPmQuadratureDemod(crate::demodulate::PmQuadratureDemod);

#[pymethods]
impl PyPmQuadratureDemod {
    #[new]
    fn new(fs: f32, k: f32, audio_bw_hz: f32) -> Self {
        Self(crate::demodulate::PmQuadratureDemod::new(fs, k, audio_bw_hz))
    }
    fn process<'py>(&mut self, py: Python<'py>, iq: PyReadonlyArray1<'py, Complex32>)
        -> PyResult<Bound<'py, PyArray1<f32>>>
    {
        let input = iq.as_slice()?;
        let mut out = vec![0.0f32; input.len()];
        self.0.process(input, &mut out);
        Ok(out.into_pyarray(py))
    }
}
