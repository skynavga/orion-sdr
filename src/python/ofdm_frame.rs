// Copyright (c) 2026 G & R Associates LLC
// SPDX-License-Identifier: MIT OR Apache-2.0

// src/python/ofdm_frame.rs — PyO3 bindings for the COFDM frame (MAC) layer.
//
// Exposes the frame modulator (`OfdmFrameMod`) and the streaming frame
// receiver (`OfdmFrameStreamDemod`, feed/flush like `Psk31Stream`), plus the
// `FramePacket`/`Mcs` support types. The FEC/interleaver/scrambler/CRC/header
// scheme is configured on `OfdmConfig` via its `with_*` methods (see
// `python/ofdm.rs`); an `McsTable` maps each frame's `mcs_index` to a
// (constellation, inner FEC, outer FEC) triple.

use num_complex::Complex32;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use super::ofdm::PyOfdmConfig;
use crate::demodulate::{OfdmFrameStreamDemod, demodulate_frame};
use crate::fec::{FrameMetadata, FramePacket, InnerFec, LdpcCode, OuterFec, PunctureRate};
use crate::modulate::{ConstellationOrder, Mcs, McsTable, OfdmFrameMod};
use crate::sync::OfdmPreamble;

// ── FramePacket ─────────────────────────────────────────────────────────────

/// A MAC-layer frame: metadata (sequence number, MCS index, flags) plus an
/// opaque byte payload.
#[pyclass(name = "FramePacket", skip_from_py_object)]
#[derive(Clone)]
pub struct PyFramePacket {
    sequence_num: u32,
    mcs_index: u8,
    flags: u8,
    payload: Vec<u8>,
}

#[pymethods]
impl PyFramePacket {
    #[new]
    #[pyo3(signature = (payload, sequence_num = 0, mcs_index = 0, flags = 0))]
    fn new(
        payload: PyReadonlyArray1<'_, u8>,
        sequence_num: u32,
        mcs_index: u8,
        flags: u8,
    ) -> PyResult<Self> {
        Ok(Self {
            sequence_num,
            mcs_index,
            flags,
            payload: payload.as_slice()?.to_vec(),
        })
    }

    #[getter]
    fn payload<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<u8>> {
        self.payload.clone().into_pyarray(py)
    }
    #[getter]
    fn sequence_num(&self) -> u32 {
        self.sequence_num
    }
    #[getter]
    fn mcs_index(&self) -> u8 {
        self.mcs_index
    }
    #[getter]
    fn flags(&self) -> u8 {
        self.flags
    }
}

impl PyFramePacket {
    fn to_frame(&self) -> FramePacket {
        FramePacket {
            metadata: FrameMetadata {
                sequence_num: self.sequence_num,
                mcs_index: self.mcs_index,
                flags: self.flags,
            },
            payload: self.payload.clone(),
        }
    }

    fn from_frame(frame: FramePacket) -> Self {
        Self {
            sequence_num: frame.metadata.sequence_num,
            mcs_index: frame.metadata.mcs_index,
            flags: frame.metadata.flags,
            payload: frame.payload,
        }
    }
}

// ── McsTable ──────────────────────────────────────────────────────────────

/// Maps each frame's `mcs_index` to a modulation-and-coding scheme. Build it
/// by adding entries; the sender and receiver must share the same table.
#[pyclass(name = "McsTable", skip_from_py_object)]
#[derive(Clone)]
pub struct PyMcsTable {
    entries: Vec<Mcs>,
}

#[pymethods]
impl PyMcsTable {
    #[new]
    fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    /// The default ladder: BPSK/QPSK/QAM-16/QAM-64, each with an LDPC(n512r12)
    /// inner code and a BCH(t=8) outer code.
    #[staticmethod]
    fn default_ladder() -> Self {
        let t = McsTable::default_ladder();
        // Reconstruct entries via public accessors.
        let mut entries = Vec::new();
        for i in 0..t.len() {
            entries.push(t.get(i as u8).unwrap());
        }
        Self { entries }
    }

    /// Appends an MCS entry. `constellation` is `"bpsk"|"qpsk"|"qam16"|
    /// "qam64"|"qam256"`; `inner`/`outer` mirror
    /// `OfdmConfig.with_inner_fec`/`with_outer_fec` (`inner_kind`, `inner_code`,
    /// `outer_kind`, `outer_a`, `outer_b`).
    #[pyo3(signature = (constellation, inner_kind = "none", inner_code = "", outer_kind = "none", outer_a = 0, outer_b = 0))]
    #[allow(clippy::too_many_arguments)]
    fn add(
        &mut self,
        constellation: &str,
        inner_kind: &str,
        inner_code: &str,
        outer_kind: &str,
        outer_a: usize,
        outer_b: usize,
    ) -> PyResult<()> {
        let c = match constellation {
            "bpsk" => ConstellationOrder::Bpsk,
            "qpsk" => ConstellationOrder::Qpsk,
            "qam16" => ConstellationOrder::Qam16,
            "qam64" => ConstellationOrder::Qam64,
            "qam256" => ConstellationOrder::Qam256,
            other => {
                return Err(PyValueError::new_err(format!(
                    "McsTable.add: unknown constellation {other:?}"
                )));
            }
        };
        let inner = match inner_kind {
            "none" => InnerFec::None,
            "ldpc" => InnerFec::Ldpc(parse_ldpc(inner_code)?),
            "convolutional" | "conv" => InnerFec::Convolutional {
                rate: parse_rate(inner_code)?,
            },
            other => {
                return Err(PyValueError::new_err(format!(
                    "McsTable.add: unknown inner FEC {other:?}"
                )));
            }
        };
        let outer = match outer_kind {
            "none" => OuterFec::None,
            "bch" => OuterFec::Bch { t: outer_a },
            "reed_solomon" | "rs" => OuterFec::ReedSolomon {
                n: outer_a,
                n_parity: outer_b,
            },
            other => {
                return Err(PyValueError::new_err(format!(
                    "McsTable.add: unknown outer FEC {other:?}"
                )));
            }
        };
        self.entries.push(Mcs::new(c, inner, outer));
        Ok(())
    }

    #[getter]
    fn len(&self) -> usize {
        self.entries.len()
    }
}

impl PyMcsTable {
    fn to_table(&self) -> PyResult<McsTable> {
        if self.entries.is_empty() {
            return Err(PyValueError::new_err(
                "McsTable must have at least one entry",
            ));
        }
        Ok(McsTable::new(self.entries.clone()))
    }
}

fn parse_ldpc(s: &str) -> PyResult<LdpcCode> {
    match s {
        "n512r12" => Ok(LdpcCode::N512R12),
        "n576r23" => Ok(LdpcCode::N576R23),
        "n512r34" => Ok(LdpcCode::N512R34),
        other => Err(PyValueError::new_err(format!(
            "unknown LDPC code {other:?}"
        ))),
    }
}

fn parse_rate(s: &str) -> PyResult<PunctureRate> {
    match s {
        "1/2" => Ok(PunctureRate::R1_2),
        "2/3" => Ok(PunctureRate::R2_3),
        "3/4" => Ok(PunctureRate::R3_4),
        "5/6" => Ok(PunctureRate::R5_6),
        "7/8" => Ok(PunctureRate::R7_8),
        other => Err(PyValueError::new_err(format!(
            "unknown puncture rate {other:?}"
        ))),
    }
}

fn build_preamble(
    num_repeats: usize,
    repeat_len: usize,
    n_fft: usize,
    cp_len: usize,
) -> OfdmPreamble {
    OfdmPreamble::new(num_repeats, repeat_len).with_training_symbol(n_fft, cp_len)
}

// ── OfdmFrameMod ────────────────────────────────────────────────────────────

/// COFDM frame transmitter: serializes a `FramePacket` to a flat IQ stream
/// (`[preamble + training][header][payload]`), applying the concatenated FEC
/// chain configured on the `OfdmConfig` and selected per frame by the MCS
/// table.
#[pyclass(name = "OfdmFrameMod")]
pub struct PyOfdmFrameMod {
    inner: OfdmFrameMod,
}

#[pymethods]
impl PyOfdmFrameMod {
    /// The preamble is `num_repeats` × `repeat_len` Schmidl & Cox segments
    /// followed by a training symbol sized to the config's FFT/CP.
    #[new]
    #[pyo3(signature = (cfg, mcs_table, num_repeats = 4, repeat_len = 16))]
    fn new(
        cfg: &PyOfdmConfig,
        mcs_table: &PyMcsTable,
        num_repeats: usize,
        repeat_len: usize,
    ) -> PyResult<Self> {
        let config = cfg.inner_config();
        let pre = build_preamble(
            num_repeats,
            repeat_len,
            config.carrier_plan.n_fft(),
            config.carrier_plan.cp_len(),
        );
        Ok(Self {
            inner: OfdmFrameMod::new(config, mcs_table.to_table()?, pre),
        })
    }

    /// Modulates a whole frame into IQ. `per_frame_seed` supplies the
    /// scrambler seed for a per-frame-random configuration (ignored otherwise).
    #[pyo3(signature = (frame, per_frame_seed = 0))]
    fn modulate_frame<'py>(
        &self,
        py: Python<'py>,
        frame: &PyFramePacket,
        per_frame_seed: u32,
    ) -> Bound<'py, PyArray1<Complex32>> {
        let iq = self.inner.modulate_frame(&frame.to_frame(), per_frame_seed);
        iq.into_pyarray(py)
    }
}

// ── OfdmFrameStreamDemod ────────────────────────────────────────────────────

/// Streaming COFDM frame receiver. Push IQ with `feed()`; it locates
/// preambles, corrects CFO, estimates the channel from the training symbol,
/// decodes each frame, and returns the completed ones. `flush()` runs a final
/// pass over the residual buffer.
#[pyclass(name = "OfdmFrameStreamDemod")]
pub struct PyOfdmFrameStreamDemod {
    inner: OfdmFrameStreamDemod,
}

#[pymethods]
impl PyOfdmFrameStreamDemod {
    #[new]
    #[pyo3(signature = (cfg, mcs_table, num_repeats = 4, repeat_len = 16))]
    fn new(
        cfg: &PyOfdmConfig,
        mcs_table: &PyMcsTable,
        num_repeats: usize,
        repeat_len: usize,
    ) -> PyResult<Self> {
        let config = cfg.inner_config();
        let pre = build_preamble(
            num_repeats,
            repeat_len,
            config.carrier_plan.n_fft(),
            config.carrier_plan.cp_len(),
        );
        Ok(Self {
            inner: OfdmFrameStreamDemod::new(config, mcs_table.to_table()?, pre),
        })
    }

    /// Feeds IQ and returns the frames that completed. Frames that failed to
    /// decode (bad CRC/FEC/header) are omitted; use `feed_with_errors` to see
    /// the error reasons.
    fn feed<'py>(
        &mut self,
        py: Python<'py>,
        iq: PyReadonlyArray1<'py, Complex32>,
    ) -> PyResult<Vec<PyFramePacket>> {
        let results = self.inner.feed(iq.as_slice()?);
        let _ = py;
        Ok(results
            .into_iter()
            .filter_map(|r| r.ok().map(|f| PyFramePacket::from_frame(f.packet)))
            .collect())
    }

    /// Like `feed`, but returns a list of `(frame_or_None, error_or_None)`
    /// tuples so callers can observe decode failures.
    fn feed_with_errors(
        &mut self,
        iq: PyReadonlyArray1<'_, Complex32>,
    ) -> PyResult<Vec<(Option<PyFramePacket>, Option<String>)>> {
        let results = self.inner.feed(iq.as_slice()?);
        Ok(results
            .into_iter()
            .map(|r| match r {
                Ok(f) => (Some(PyFramePacket::from_frame(f.packet)), None),
                Err(e) => (None, Some(e.to_string())),
            })
            .collect())
    }

    /// Runs a final decode pass over the residual buffer.
    fn flush(&mut self) -> Vec<PyFramePacket> {
        self.inner
            .flush()
            .into_iter()
            .filter_map(|r| r.ok().map(|f| PyFramePacket::from_frame(f.packet)))
            .collect()
    }

    /// Number of accumulated (not-yet-consumed) samples.
    #[getter]
    fn buffered(&self) -> usize {
        self.inner.len()
    }

    /// Discards the accumulated buffer.
    fn clear(&mut self) {
        self.inner.clear();
    }
}

// ── demodulate_frame (batch) ────────────────────────────────────────────────

/// Batch-demodulates a single frame at a known start (`iq[0]` is the first
/// sample after the preamble+training, already synchronized). Raises
/// `ValueError` on a decode failure.
#[pyfunction]
#[pyo3(name = "demodulate_frame")]
fn demodulate_frame_py(
    cfg: &PyOfdmConfig,
    mcs_table: &PyMcsTable,
    iq: PyReadonlyArray1<'_, Complex32>,
) -> PyResult<PyFramePacket> {
    let config = cfg.inner_config();
    let table = mcs_table.to_table()?;
    let frame = demodulate_frame(&config, &table, iq.as_slice()?)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(PyFramePacket::from_frame(frame))
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyFramePacket>()?;
    m.add_class::<PyMcsTable>()?;
    m.add_class::<PyOfdmFrameMod>()?;
    m.add_class::<PyOfdmFrameStreamDemod>()?;
    m.add_function(wrap_pyfunction!(demodulate_frame_py, m)?)?;
    Ok(())
}
