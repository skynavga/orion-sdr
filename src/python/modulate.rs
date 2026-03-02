use pyo3::prelude::*;
use numpy::{PyReadonlyArray1, PyArray1, IntoPyArray};
use num_complex::Complex32;
use crate::core::Block;

// ── AmDsbMod ──────────────────────────────────────────────────────────────────

#[pyclass(name = "AmDsbMod")]
pub struct PyAmDsbMod(crate::modulate::AmDsbMod);

#[pymethods]
impl PyAmDsbMod {
    #[new]
    fn new(fs: f32, rf_hz: f32, carrier_level: f32, modulation_index: f32) -> Self {
        Self(crate::modulate::AmDsbMod::new(fs, rf_hz, carrier_level, modulation_index))
    }
    fn set_gain(&mut self, g: f32) { self.0.set_gain(g); }
    fn set_clamp(&mut self, on: bool) { self.0.set_clamp(on); }
    fn process<'py>(&mut self, py: Python<'py>, audio: PyReadonlyArray1<'py, f32>)
        -> PyResult<Bound<'py, PyArray1<Complex32>>>
    {
        let input = audio.as_slice()?;
        let mut out = vec![Complex32::new(0.0, 0.0); input.len()];
        self.0.process(input, &mut out);
        Ok(out.into_pyarray(py))
    }
}

// ── CwKeyedMod ────────────────────────────────────────────────────────────────

#[pyclass(name = "CwKeyedMod")]
pub struct PyCwKeyedMod(crate::modulate::CwKeyedMod);

#[pymethods]
impl PyCwKeyedMod {
    #[new]
    fn new(sample_rate: f32, tone_hz: f32, rise_ms: f32, fall_ms: f32) -> Self {
        Self(crate::modulate::CwKeyedMod::new(sample_rate, tone_hz, rise_ms, fall_ms))
    }
    fn set_gain(&mut self, g: f32) { self.0.set_gain(g); }
    /// Input: keying envelope 0..1 as float32 array.
    fn process<'py>(&mut self, py: Python<'py>, audio: PyReadonlyArray1<'py, f32>)
        -> PyResult<Bound<'py, PyArray1<Complex32>>>
    {
        let input = audio.as_slice()?;
        let mut out = vec![Complex32::new(0.0, 0.0); input.len()];
        self.0.process(input, &mut out);
        Ok(out.into_pyarray(py))
    }
}

// ── FmPhaseAccumMod ───────────────────────────────────────────────────────────

#[pyclass(name = "FmPhaseAccumMod")]
pub struct PyFmPhaseAccumMod(crate::modulate::FmPhaseAccumMod);

#[pymethods]
impl PyFmPhaseAccumMod {
    #[new]
    fn new(sample_rate: f32, deviation_hz: f32, rf_hz: f32) -> Self {
        Self(crate::modulate::FmPhaseAccumMod::new(sample_rate, deviation_hz, rf_hz))
    }
    fn set_deviation(&mut self, hz: f32) { self.0.set_deviation(hz); }
    fn set_gain(&mut self, g: f32) { self.0.set_gain(g); }
    fn process<'py>(&mut self, py: Python<'py>, audio: PyReadonlyArray1<'py, f32>)
        -> PyResult<Bound<'py, PyArray1<Complex32>>>
    {
        let input = audio.as_slice()?;
        let mut out = vec![Complex32::new(0.0, 0.0); input.len()];
        self.0.process(input, &mut out);
        Ok(out.into_pyarray(py))
    }
}

// ── PmDirectPhaseMod ──────────────────────────────────────────────────────────

#[pyclass(name = "PmDirectPhaseMod")]
pub struct PyPmDirectPhaseMod(crate::modulate::PmDirectPhaseMod);

#[pymethods]
impl PyPmDirectPhaseMod {
    #[new]
    fn new(sample_rate: f32, kp_rad_per_unit: f32, rf_hz: f32) -> Self {
        Self(crate::modulate::PmDirectPhaseMod::new(sample_rate, kp_rad_per_unit, rf_hz))
    }
    fn set_gain(&mut self, g: f32) { self.0.set_gain(g); }
    fn set_sensitivity(&mut self, kp: f32) { self.0.set_sensitivity(kp); }
    fn process<'py>(&mut self, py: Python<'py>, audio: PyReadonlyArray1<'py, f32>)
        -> PyResult<Bound<'py, PyArray1<Complex32>>>
    {
        let input = audio.as_slice()?;
        let mut out = vec![Complex32::new(0.0, 0.0); input.len()];
        self.0.process(input, &mut out);
        Ok(out.into_pyarray(py))
    }
}

// ── SsbPhasingMod ─────────────────────────────────────────────────────────────

#[pyclass(name = "SsbPhasingMod")]
pub struct PySsbPhasingMod(crate::modulate::SsbPhasingMod);

#[pymethods]
impl PySsbPhasingMod {
    #[new]
    fn new(fs: f32, audio_bw_hz: f32, audio_if_hz: f32, rf_hz: f32, usb: bool) -> Self {
        Self(crate::modulate::SsbPhasingMod::new(fs, audio_bw_hz, audio_if_hz, rf_hz, usb))
    }
    fn process<'py>(&mut self, py: Python<'py>, audio: PyReadonlyArray1<'py, f32>)
        -> PyResult<Bound<'py, PyArray1<Complex32>>>
    {
        let input = audio.as_slice()?;
        let mut out = vec![Complex32::new(0.0, 0.0); input.len()];
        self.0.process(input, &mut out);
        Ok(out.into_pyarray(py))
    }
}
