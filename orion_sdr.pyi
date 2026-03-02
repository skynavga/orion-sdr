"""
Type stubs for the orion-sdr native extension module.

All classes live in the flat ``orion_sdr`` namespace.  IQ arrays use
``numpy.complex64``; audio arrays use ``numpy.float32``.  Every
``process()`` call returns a new 1-D array of the same length as the
input.  Arrays must be 1-D and C-contiguous; a wrong dtype or layout
raises ``ValueError``.
"""

import numpy as np
from numpy.typing import NDArray

# ---------------------------------------------------------------------------
# Demodulators  (IQ complex64 → audio float32)
# ---------------------------------------------------------------------------

class CwEnvelopeDemod:
    """One-pole envelope detector for CW signals.

    Tracks the instantaneous magnitude of the IQ input with a low-pass
    time constant derived from *env_bw_hz*.  *tone_hz* is accepted for
    API symmetry but is not used internally (pre-tune the signal before
    passing it in).
    """

    def __init__(self, sample_rate: float, tone_hz: float, env_bw_hz: float) -> None: ...
    def set_gain(self, g: float) -> None: ...
    def process(self, iq: NDArray[np.complex64]) -> NDArray[np.float32]: ...

class AmEnvelopeDemod:
    """AM envelope demodulator with 4th-order IIR low-pass and DC blocker.

    Two envelope methods are available:

    * ``abs_approx=False`` (default) — ``sqrt(I² + Q²)`` after the LP filter
      (*PowerSqrt*); highest fidelity.
    * ``abs_approx=True`` — ``k1·|I| + k2·|Q|`` approximation (*AbsApprox*,
      k1=0.9482, k2=0.3920); slightly faster with a small amplitude error.
    """

    def __init__(
        self,
        fs: float,
        audio_bw_hz: float,
        abs_approx: bool = False,
    ) -> None: ...
    def process(self, iq: NDArray[np.complex64]) -> NDArray[np.float32]: ...

class SsbProductDemod:
    """SSB product detector with BFO rotator and 4th-order IIR audio filter.

    Set *bfo_hz* to 0 for a signal already tuned to baseband, or to a small
    offset to shift the recovered audio pitch.
    """

    def __init__(self, fs: float, bfo_hz: float, audio_bw_hz: float) -> None: ...
    def process(self, iq: NDArray[np.complex64]) -> NDArray[np.float32]: ...

class FmQuadratureDemod:
    """FM quadrature (phase-difference) discriminator.

    Output is scaled so that ±*dev_hz* of instantaneous frequency deviation
    maps to roughly ±1.0.  A 4th-order IIR low-pass at *audio_bw_hz* follows
    the discriminator.
    """

    def __init__(self, fs: float, dev_hz: float, audio_bw_hz: float) -> None: ...
    def process(self, iq: NDArray[np.complex64]) -> NDArray[np.float32]: ...

class PmQuadratureDemod:
    """PM quadrature demodulator (instantaneous phase difference).

    *k* scales the recovered phase difference to the output audio level.
    A 4th-order IIR low-pass at *audio_bw_hz* follows the discriminator.
    """

    def __init__(self, fs: float, k: float, audio_bw_hz: float) -> None: ...
    def process(self, iq: NDArray[np.complex64]) -> NDArray[np.float32]: ...

# ---------------------------------------------------------------------------
# Modulators  (audio float32 → IQ complex64)
# ---------------------------------------------------------------------------

class AmDsbMod:
    """AM double-sideband modulator with optional carrier and RF upconversion.

    * *carrier_level* — 1.0 produces full carrier (A3E); 0.0 gives DSB-SC.
    * *modulation_index* — values ≤ 1.0 are recommended to avoid
      over-modulation.
    * *rf_hz* — set to 0.0 for baseband IQ output.
    """

    def __init__(
        self,
        fs: float,
        rf_hz: float,
        carrier_level: float,
        modulation_index: float,
    ) -> None: ...
    def set_gain(self, g: float) -> None: ...
    def set_clamp(self, on: bool) -> None:
        """Clamp the modulated envelope to ±1 to prevent over-modulation."""
        ...
    def process(self, audio: NDArray[np.float32]) -> NDArray[np.complex64]: ...

class CwKeyedMod:
    """CW keyed carrier modulator with shaped rise/fall envelope.

    The input array is a **keying envelope** in the range 0..1 (not raw
    audio): 1.0 = key down, 0.0 = key up.  Rise and fall times are
    smoothed by one-pole filters with time constants *rise_ms* / *fall_ms*
    to suppress key clicks.
    """

    def __init__(
        self,
        sample_rate: float,
        tone_hz: float,
        rise_ms: float,
        fall_ms: float,
    ) -> None: ...
    def set_gain(self, g: float) -> None: ...
    def process(self, audio: NDArray[np.float32]) -> NDArray[np.complex64]: ...

class FmPhaseAccumMod:
    """FM modulator using a phasor-recurrence phase accumulator.

    Each sample multiplies a running phasor by ``exp(j·2π·kf·x/fs)``
    where ``kf = deviation_hz``.  The phasor is renormalized every 1024
    samples to prevent amplitude drift.  Set *rf_hz* to 0.0 for baseband
    output.
    """

    def __init__(
        self,
        sample_rate: float,
        deviation_hz: float,
        rf_hz: float,
    ) -> None: ...
    def set_deviation(self, hz: float) -> None: ...
    def set_gain(self, g: float) -> None: ...
    def process(self, audio: NDArray[np.float32]) -> NDArray[np.complex64]: ...

class PmDirectPhaseMod:
    """PM modulator: instantaneous phase φ = kp · x[n].

    *kp_rad_per_unit* maps ±1.0 audio to ±kp radians of carrier phase.
    Set *rf_hz* to 0.0 for baseband output.
    """

    def __init__(
        self,
        sample_rate: float,
        kp_rad_per_unit: float,
        rf_hz: float,
    ) -> None: ...
    def set_gain(self, g: float) -> None: ...
    def set_sensitivity(self, kp: float) -> None:
        """Update the phase sensitivity (rad per unit input amplitude)."""
        ...
    def process(self, audio: NDArray[np.float32]) -> NDArray[np.complex64]: ...

class SsbPhasingMod:
    """SSB phasing modulator (Weaver-style IIR variant).

    Audio is up-converted to *audio_if_hz* via a complex rotator, split
    into I and Q paths through matched 4th-order IIR low-pass filters,
    then combined to select the desired sideband.  Set *usb=True* for
    upper sideband, *usb=False* for lower sideband.  Set *rf_hz* to 0.0
    for baseband IQ output.
    """

    def __init__(
        self,
        fs: float,
        audio_bw_hz: float,
        audio_if_hz: float,
        rf_hz: float,
        usb: bool,
    ) -> None: ...
    def process(self, audio: NDArray[np.float32]) -> NDArray[np.complex64]: ...
