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

// ── BpskMod ───────────────────────────────────────────────────────────────────

/// Combined BPSK mapper + waveform stage.
/// Input: uint8 array of bits (LSB of each byte used).
/// Output: complex64 IQ array (one symbol per input bit).
#[pyclass(name = "BpskMod")]
pub struct PyBpskMod {
    mapper: crate::modulate::BpskMapper,
    waveform: crate::modulate::BpskMod,
}

#[pymethods]
impl PyBpskMod {
    #[new]
    fn new(fs: f32, rf_hz: f32, gain: f32) -> Self {
        Self {
            mapper: crate::modulate::BpskMapper::new(),
            waveform: crate::modulate::BpskMod::new(fs, rf_hz, gain),
        }
    }
    fn set_gain(&mut self, g: f32) { self.waveform.set_gain(g); }
    fn process<'py>(&mut self, py: Python<'py>, bits: PyReadonlyArray1<'py, u8>)
        -> PyResult<Bound<'py, PyArray1<Complex32>>>
    {
        let input = bits.as_slice()?;
        let n = input.len();
        let mut syms = vec![Complex32::new(0.0, 0.0); n];
        let mut iq   = vec![Complex32::new(0.0, 0.0); n];
        self.mapper.process(input, &mut syms);
        self.waveform.process(&syms, &mut iq);
        Ok(iq.into_pyarray(py))
    }
}

// ── QpskMod ───────────────────────────────────────────────────────────────────

/// Combined QPSK mapper + waveform stage.
/// Input: uint8 array of bits (LSB of each byte); consumed in pairs.
/// Output: complex64 IQ array (one symbol per two input bits).
#[pyclass(name = "QpskMod")]
pub struct PyQpskMod {
    mapper: crate::modulate::QpskMapper,
    waveform: crate::modulate::QpskMod,
}

#[pymethods]
impl PyQpskMod {
    #[new]
    fn new(fs: f32, rf_hz: f32, gain: f32) -> Self {
        Self {
            mapper: crate::modulate::QpskMapper::new(),
            waveform: crate::modulate::QpskMod::new(fs, rf_hz, gain),
        }
    }
    fn set_gain(&mut self, g: f32) { self.waveform.set_gain(g); }
    fn process<'py>(&mut self, py: Python<'py>, bits: PyReadonlyArray1<'py, u8>)
        -> PyResult<Bound<'py, PyArray1<Complex32>>>
    {
        let input = bits.as_slice()?;
        let n_syms = input.len() / 2;
        let mut syms = vec![Complex32::new(0.0, 0.0); n_syms];
        let mut iq   = vec![Complex32::new(0.0, 0.0); n_syms];
        self.mapper.process(input, &mut syms);
        self.waveform.process(&syms, &mut iq);
        Ok(iq.into_pyarray(py))
    }
}

// ── QamMod ────────────────────────────────────────────────────────────────────

/// Enum holding one of the three concrete QamMapper instantiations.
enum QamMapperInner {
    Qam16(crate::modulate::Qam16Mapper),
    Qam64(crate::modulate::Qam64Mapper),
    Qam256(crate::modulate::Qam256Mapper),
}

impl QamMapperInner {
    fn bits(&self) -> usize {
        match self { Self::Qam16(_) => 4, Self::Qam64(_) => 6, Self::Qam256(_) => 8 }
    }
    fn process(&mut self, input: &[u8], output: &mut [Complex32]) {
        match self {
            Self::Qam16(m)  => { m.process(input, output); }
            Self::Qam64(m)  => { m.process(input, output); }
            Self::Qam256(m) => { m.process(input, output); }
        }
    }
}

/// Combined QAM mapper + waveform stage.
///
/// *order* must be 16, 64, or 256.
/// Input: uint8 array of bits (LSB of each byte); consumed log2(order) bytes per symbol.
/// Output: complex64 IQ array (one symbol per log2(order) input bits).
#[pyclass(name = "QamMod")]
pub struct PyQamMod {
    mapper: QamMapperInner,
    waveform: crate::modulate::QamMod,
}

#[pymethods]
impl PyQamMod {
    #[new]
    fn new(order: u32, fs: f32, rf_hz: f32, gain: f32) -> PyResult<Self> {
        let mapper = match order {
            16  => QamMapperInner::Qam16(crate::modulate::Qam16Mapper::new()),
            64  => QamMapperInner::Qam64(crate::modulate::Qam64Mapper::new()),
            256 => QamMapperInner::Qam256(crate::modulate::Qam256Mapper::new()),
            _   => return Err(pyo3::exceptions::PyValueError::new_err(
                "QamMod: order must be 16, 64, or 256"
            )),
        };
        Ok(Self { mapper, waveform: crate::modulate::QamMod::new(fs, rf_hz, gain) })
    }
    fn set_gain(&mut self, g: f32) { self.waveform.set_gain(g); }
    fn process<'py>(&mut self, py: Python<'py>, bits: PyReadonlyArray1<'py, u8>)
        -> PyResult<Bound<'py, PyArray1<Complex32>>>
    {
        let input = bits.as_slice()?;
        let bits_per_sym = self.mapper.bits();
        let n_syms = input.len() / bits_per_sym;
        let mut syms = vec![Complex32::new(0.0, 0.0); n_syms];
        let mut iq   = vec![Complex32::new(0.0, 0.0); n_syms];
        self.mapper.process(input, &mut syms);
        self.waveform.process(&syms, &mut iq);
        Ok(iq.into_pyarray(py))
    }
}
