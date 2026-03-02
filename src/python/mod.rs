use pyo3::prelude::*;

mod demodulate;
mod modulate;

#[pymodule]
fn orion_sdr(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<demodulate::PyCwEnvelopeDemod>()?;
    m.add_class::<demodulate::PyAmEnvelopeDemod>()?;
    m.add_class::<demodulate::PySsbProductDemod>()?;
    m.add_class::<demodulate::PyFmQuadratureDemod>()?;
    m.add_class::<demodulate::PyPmQuadratureDemod>()?;
    m.add_class::<modulate::PyAmDsbMod>()?;
    m.add_class::<modulate::PyCwKeyedMod>()?;
    m.add_class::<modulate::PyFmPhaseAccumMod>()?;
    m.add_class::<modulate::PyPmDirectPhaseMod>()?;
    m.add_class::<modulate::PySsbPhasingMod>()?;
    Ok(())
}
