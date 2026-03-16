use pyo3::prelude::*;

mod demodulate;
mod ft8;
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
    // FT8/FT4 waveform
    m.add_class::<ft8::PyFt8Mod>()?;
    m.add_class::<ft8::PyFt8Demod>()?;
    m.add_class::<ft8::PyFt8Codec>()?;
    m.add_class::<ft8::PyFt4Mod>()?;
    m.add_class::<ft8::PyFt4Demod>()?;
    m.add_class::<ft8::PyFt4Codec>()?;
    // FT8/FT4 sync
    m.add_function(wrap_pyfunction!(ft8::ft8_sync, m)?)?;
    m.add_function(wrap_pyfunction!(ft8::ft4_sync, m)?)?;
    // FT8/FT4 message packing
    m.add_function(wrap_pyfunction!(ft8::ft8_pack_standard, m)?)?;
    m.add_function(wrap_pyfunction!(ft8::ft8_pack_free_text, m)?)?;
    m.add_function(wrap_pyfunction!(ft8::ft8_pack_telemetry, m)?)?;
    m.add_function(wrap_pyfunction!(ft8::ft8_unpack, m)?)?;
    Ok(())
}
