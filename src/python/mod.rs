use pyo3::prelude::*;

mod demodulate;
mod modulate;

#[pymodule]
fn orion_sdr(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // analog demodulators
    m.add_class::<demodulate::PyCwEnvelopeDemod>()?;
    m.add_class::<demodulate::PyAmEnvelopeDemod>()?;
    m.add_class::<demodulate::PySsbProductDemod>()?;
    m.add_class::<demodulate::PyFmQuadratureDemod>()?;
    m.add_class::<demodulate::PyPmQuadratureDemod>()?;
    // digital demodulators
    m.add_class::<demodulate::PyBpskDemod>()?;
    m.add_class::<demodulate::PyQpskDemod>()?;
    m.add_class::<demodulate::PyQamDemod>()?;
    // analog modulators
    m.add_class::<modulate::PyAmDsbMod>()?;
    m.add_class::<modulate::PyCwKeyedMod>()?;
    m.add_class::<modulate::PyFmPhaseAccumMod>()?;
    m.add_class::<modulate::PyPmDirectPhaseMod>()?;
    m.add_class::<modulate::PySsbPhasingMod>()?;
    // digital modulators
    m.add_class::<modulate::PyBpskMod>()?;
    m.add_class::<modulate::PyQpskMod>()?;
    m.add_class::<modulate::PyQamMod>()?;
    Ok(())
}
