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

// ── BpskDemod ─────────────────────────────────────────────────────────────────

/// Combined BPSK soft-demod + hard decider.
/// Input: complex64 IQ array (carrier-removed baseband, 1 sample per symbol).
/// Output: uint8 array of bits (0 or 1), one per input symbol.
#[pyclass(name = "BpskDemod")]
pub struct PyBpskDemod {
    demod: crate::demodulate::BpskDemod,
    decider: crate::demodulate::BpskDecider,
}

#[pymethods]
impl PyBpskDemod {
    #[new]
    fn new(gain: f32) -> Self {
        Self {
            demod: crate::demodulate::BpskDemod::new(gain),
            decider: crate::demodulate::BpskDecider::new(),
        }
    }
    fn set_gain(&mut self, g: f32) { self.demod.set_gain(g); }
    fn process<'py>(&mut self, py: Python<'py>, iq: PyReadonlyArray1<'py, Complex32>)
        -> PyResult<Bound<'py, PyArray1<u8>>>
    {
        let input = iq.as_slice()?;
        let n = input.len();
        let mut soft = vec![Complex32::new(0.0, 0.0); n];
        let mut bits = vec![0u8; n];
        self.demod.process(input, &mut soft);
        self.decider.process(&soft, &mut bits);
        Ok(bits.into_pyarray(py))
    }
}

// ── QpskDemod ─────────────────────────────────────────────────────────────────

/// Combined QPSK soft-demod + hard decider.
/// Input: complex64 IQ array (carrier-removed baseband, 1 sample per symbol).
/// Output: uint8 array of bits (0 or 1); two bits per input symbol,
///         interleaved as [b0_I, b0_Q, b1_I, b1_Q, …].
#[pyclass(name = "QpskDemod")]
pub struct PyQpskDemod {
    demod: crate::demodulate::QpskDemod,
    decider: crate::demodulate::QpskDecider,
}

#[pymethods]
impl PyQpskDemod {
    #[new]
    fn new(gain: f32) -> Self {
        Self {
            demod: crate::demodulate::QpskDemod::new(gain),
            decider: crate::demodulate::QpskDecider::new(),
        }
    }
    fn set_gain(&mut self, g: f32) { self.demod.set_gain(g); }
    fn process<'py>(&mut self, py: Python<'py>, iq: PyReadonlyArray1<'py, Complex32>)
        -> PyResult<Bound<'py, PyArray1<u8>>>
    {
        let input = iq.as_slice()?;
        let n = input.len();
        let mut soft = vec![Complex32::new(0.0, 0.0); n];
        let mut bits = vec![0u8; n * 2];
        self.demod.process(input, &mut soft);
        self.decider.process(&soft, &mut bits);
        Ok(bits.into_pyarray(py))
    }
}

// ── QamDemod ──────────────────────────────────────────────────────────────────

/// Enum holding one of the three concrete QamDecider instantiations.
enum QamDeciderInner {
    Qam16(crate::demodulate::Qam16Decider),
    Qam64(crate::demodulate::Qam64Decider),
    Qam256(crate::demodulate::Qam256Decider),
}

impl QamDeciderInner {
    fn bits(&self) -> usize {
        match self { Self::Qam16(_) => 4, Self::Qam64(_) => 6, Self::Qam256(_) => 8 }
    }
    fn process(&mut self, input: &[Complex32], output: &mut [u8]) {
        match self {
            Self::Qam16(d)  => { d.process(input, output); }
            Self::Qam64(d)  => { d.process(input, output); }
            Self::Qam256(d) => { d.process(input, output); }
        }
    }
}

/// Combined QAM soft-demod + hard decider.
///
/// *order* must be 16, 64, or 256.
/// Input: complex64 IQ array (carrier-removed baseband, 1 sample per symbol).
/// Output: uint8 array of bits (0 or 1); log2(order) bits per input symbol.
#[pyclass(name = "QamDemod")]
pub struct PyQamDemod {
    demod: crate::demodulate::QamDemod,
    decider: QamDeciderInner,
}

#[pymethods]
impl PyQamDemod {
    #[new]
    fn new(order: u32, gain: f32) -> PyResult<Self> {
        let decider = match order {
            16  => QamDeciderInner::Qam16(crate::demodulate::Qam16Decider::new()),
            64  => QamDeciderInner::Qam64(crate::demodulate::Qam64Decider::new()),
            256 => QamDeciderInner::Qam256(crate::demodulate::Qam256Decider::new()),
            _   => return Err(pyo3::exceptions::PyValueError::new_err(
                "QamDemod: order must be 16, 64, or 256"
            )),
        };
        Ok(Self { demod: crate::demodulate::QamDemod::new(gain), decider })
    }
    fn set_gain(&mut self, g: f32) { self.demod.set_gain(g); }
    fn process<'py>(&mut self, py: Python<'py>, iq: PyReadonlyArray1<'py, Complex32>)
        -> PyResult<Bound<'py, PyArray1<u8>>>
    {
        let input = iq.as_slice()?;
        let n = input.len();
        let bits_per_sym = self.decider.bits();
        let mut soft = vec![Complex32::new(0.0, 0.0); n];
        let mut bits = vec![0u8; n * bits_per_sym];
        self.demod.process(input, &mut soft);
        self.decider.process(&soft, &mut bits);
        Ok(bits.into_pyarray(py))
    }
}
