"""
Type stubs for the orion-sdr native extension module.

All classes live in the flat ``orion_sdr`` namespace.  IQ arrays use
``numpy.complex64``; audio arrays use ``numpy.float32``; bit arrays use
``numpy.uint8`` (one bit per byte, value 0 or 1).  Every ``process()``
call returns a new 1-D array; arrays must be 1-D and C-contiguous, or a
``ValueError`` is raised.
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
# Digital demodulators  (IQ complex64 → bits uint8)
# ---------------------------------------------------------------------------

class BpskDemod:
    """BPSK demodulator: hard-decision slicer.

    Input: complex64 IQ array, carrier-removed baseband, 1 sample per symbol.
    Output: uint8 bit array — one bit (0 or 1) per input symbol.
    Decision rule: Re(z) ≥ 0 → 0, Re(z) < 0 → 1.

    *gain* scales the soft metric before slicing (use 1.0 for normalized input).
    """

    def __init__(self, gain: float = 1.0) -> None: ...
    def set_gain(self, g: float) -> None: ...
    def process(self, iq: NDArray[np.complex64]) -> NDArray[np.uint8]: ...

class QpskDemod:
    """QPSK demodulator: hard-decision slicer.

    Input: complex64 IQ array, carrier-removed baseband, 1 sample per symbol.
    Output: uint8 bit array — two bits per input symbol, interleaved as
    ``[b0_I, b0_Q, b1_I, b1_Q, …]``.  Matches the Gray coding of ``QpskMod``.

    *gain* scales the soft metric before slicing (use 1.0 for normalized input).
    """

    def __init__(self, gain: float = 1.0) -> None: ...
    def set_gain(self, g: float) -> None: ...
    def process(self, iq: NDArray[np.complex64]) -> NDArray[np.uint8]: ...

class QamDemod:
    """QAM demodulator: hard-decision slicer for square QAM constellations.

    *order* must be 16, 64, or 256 (raises ``ValueError`` otherwise).
    Input: complex64 IQ array, carrier-removed baseband, 1 sample per symbol.
    Output: uint8 bit array — ``log2(order)`` bits per input symbol, laid out
    as ``log2(order)/2`` I-axis bits then ``log2(order)/2`` Q-axis bits
    (MSB-first within each axis Gray index).  Matches ``QamMod`` bit layout.

    *gain* scales the soft metric before slicing (use 1.0 for normalized input).
    """

    def __init__(self, order: int, gain: float = 1.0) -> None: ...
    def set_gain(self, g: float) -> None: ...
    def process(self, iq: NDArray[np.complex64]) -> NDArray[np.uint8]: ...

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

# ---------------------------------------------------------------------------
# Digital modulators  (bits uint8 → IQ complex64)
# ---------------------------------------------------------------------------

class BpskMod:
    """BPSK modulator: Gray-coded constellation mapper + waveform stage.

    Input: uint8 bit array (LSB of each byte used), one bit per symbol.
    Output: complex64 IQ array of the same length.
    Constellation: bit 0 → (+1, 0), bit 1 → (−1, 0).

    Set *rf_hz* to 0.0 for baseband output; non-zero upconverts via an
    internal ``Rotator`` (phasor recurrence, no per-sample trig).
    """

    def __init__(self, fs: float, rf_hz: float, gain: float = 1.0) -> None: ...
    def set_gain(self, g: float) -> None: ...
    def process(self, bits: NDArray[np.uint8]) -> NDArray[np.complex64]: ...

class QpskMod:
    """QPSK modulator: Gray-coded constellation mapper + waveform stage.

    Input: uint8 bit array (LSB of each byte); consumed in pairs ``[b0, b1]``.
    Output: complex64 IQ array of length ``len(bits) // 2``.
    Constellation is normalized to unit energy (each axis ±1/√2).

    Set *rf_hz* to 0.0 for baseband output.
    """

    def __init__(self, fs: float, rf_hz: float, gain: float = 1.0) -> None: ...
    def set_gain(self, g: float) -> None: ...
    def process(self, bits: NDArray[np.uint8]) -> NDArray[np.complex64]: ...

class QamMod:
    """Square QAM modulator: Gray-coded constellation mapper + waveform stage.

    *order* must be 16, 64, or 256 (raises ``ValueError`` otherwise).
    Input: uint8 bit array (LSB of each byte); consumed ``log2(order)`` bytes
    per symbol.  Output: complex64 IQ array of length
    ``len(bits) // log2(order)``.

    Constellation is Gray-coded on each axis independently and normalized to
    unit average symbol energy.  Set *rf_hz* to 0.0 for baseband output.
    """

    def __init__(self, order: int, fs: float, rf_hz: float, gain: float = 1.0) -> None: ...
    def set_gain(self, g: float) -> None: ...
    def process(self, bits: NDArray[np.uint8]) -> NDArray[np.complex64]: ...

# ---------------------------------------------------------------------------
# FT8/FT4 waveform classes
# ---------------------------------------------------------------------------

class Ft8Mod:
    """FT8 frame modulator: 8-FSK CPFSK, 79 symbols, 151 680 samples at 12 kHz.

    Input: uint8 array of 58 tone indices (0–7).
    Output: complex64 IQ array of shape (151680,).
    """

    def __init__(self, fs: float, base_hz: float, rf_hz: float, gain: float = 1.0) -> None: ...
    def modulate(self, tones: NDArray[np.uint8]) -> NDArray[np.complex64]: ...

class Ft8Demod:
    """FT8 frame demodulator: Goertzel/dot-product tone detector.

    Input: complex64 IQ array of at least 151 680 samples.
    Output: uint8 array of 58 tone indices.
    Raises ``ValueError`` if input is too short or demodulation fails.
    """

    def __init__(self, fs: float, base_hz: float) -> None: ...
    def demodulate(self, iq: NDArray[np.complex64]) -> NDArray[np.uint8]: ...

class Ft8Codec:
    """FT8 channel codec: CRC-14 + LDPC(174,91) + Gray code.

    All methods are static; no per-instance state.
    """

    def __init__(self) -> None: ...
    @staticmethod
    def encode(payload: bytes) -> NDArray[np.uint8]:
        """Encode a 10-byte payload → uint8[58] Gray-coded tone indices."""
        ...
    @staticmethod
    def decode_hard(tones: NDArray[np.uint8]) -> bytes | None:
        """Hard-decision decode 58 tone indices → bytes[10], or None on failure."""
        ...
    @staticmethod
    def decode_soft(llr: NDArray[np.float32]) -> bytes | None:
        """Soft-decision decode float32[174] LLRs → bytes[10], or None on failure."""
        ...

class Ft4Mod:
    """FT4 frame modulator: 4-FSK CPFSK, 105 symbols, 60 480 samples at 12 kHz.

    Input: uint8 array of 87 tone indices (0–3).
    Output: complex64 IQ array of shape (60480,).
    """

    def __init__(self, fs: float, base_hz: float, rf_hz: float, gain: float = 1.0) -> None: ...
    def modulate(self, tones: NDArray[np.uint8]) -> NDArray[np.complex64]: ...

class Ft4Demod:
    """FT4 frame demodulator: Goertzel/dot-product tone detector.

    Input: complex64 IQ array of at least 60 480 samples.
    Output: uint8 array of 87 tone indices.
    Raises ``ValueError`` if input is too short or demodulation fails.
    """

    def __init__(self, fs: float, base_hz: float) -> None: ...
    def demodulate(self, iq: NDArray[np.complex64]) -> NDArray[np.uint8]: ...

class Ft4Codec:
    """FT4 channel codec: XOR scramble + CRC-14 + LDPC(174,91) + Gray code.

    All methods are static; no per-instance state.
    """

    def __init__(self) -> None: ...
    @staticmethod
    def encode(payload: bytes) -> NDArray[np.uint8]:
        """Encode a 10-byte payload → uint8[87] Gray-coded tone indices."""
        ...
    @staticmethod
    def decode_hard(tones: NDArray[np.uint8]) -> bytes | None:
        """Hard-decision decode 87 tone indices → bytes[10], or None on failure."""
        ...
    @staticmethod
    def decode_soft(llr: NDArray[np.float32]) -> bytes | None:
        """Soft-decision decode float32[174] LLRs → bytes[10], or None on failure."""
        ...

# ---------------------------------------------------------------------------
# FT8/FT4 sync functions
# ---------------------------------------------------------------------------

def ft8_sync(
    iq: NDArray[np.complex64],
    fs: float,
    base_hz: float,
    max_hz: float,
    t_min: int,
    t_max: int,
    max_cand: int,
) -> list[dict]:
    """Synchronise an FT8 IQ buffer and return up to *max_cand* frame candidates.

    Each candidate is a dict::

        {
            "time_sym": int,           # symbol offset of frame start
            "freq_bin": int,           # frequency bin of tone-0
            "score":    float,         # Costas match score
            "llr":      float32[174],  # soft LLRs for Ft8Codec.decode_soft
        }

    Pass each result's ``"llr"`` to ``Ft8Codec.decode_soft`` to recover the
    77-bit payload.
    """
    ...

def ft4_sync(
    iq: NDArray[np.complex64],
    fs: float,
    base_hz: float,
    max_hz: float,
    t_min: int,
    t_max: int,
    max_cand: int,
) -> list[dict]:
    """Synchronise an FT4 IQ buffer and return up to *max_cand* frame candidates.

    Same return shape as ``ft8_sync``.
    """
    ...

# ---------------------------------------------------------------------------
# FT8/FT4 message packing functions
# ---------------------------------------------------------------------------

def ft8_pack_standard(call_to: str, call_de: str, extra: str) -> bytes:
    """Pack a standard FT8/FT4 message → bytes[10].

    *extra* may be a Maidenhead grid (``"FN31"``), signal report (``"+07"``,
    ``"-12"``), R-prefixed report (``"R+05"``), or token
    (``"RRR"``, ``"RR73"``, ``"73"``).  Pass ``""`` for no extra field.

    Raises ``ValueError`` if the callsigns cannot be encoded.
    """
    ...

def ft8_pack_free_text(text: str) -> bytes:
    """Pack a free-text FT8/FT4 message (up to 13 chars) → bytes[10].

    Raises ``ValueError`` if the text is too long or contains invalid characters.
    """
    ...

def ft8_pack_telemetry(data: bytes) -> bytes:
    """Pack a telemetry FT8/FT4 message (exactly 9 bytes) → bytes[10].

    Raises ``ValueError`` if *data* is not exactly 9 bytes.
    """
    ...

def ft8_unpack(payload: bytes) -> dict:
    """Unpack a 10-byte FT8/FT4 payload → dict.

    The ``"type"`` key indicates the message type:

    * ``"standard"``  — ``{"type", "call_to", "call_de", "extra"}``
    * ``"free_text"`` — ``{"type", "text"}``
    * ``"telemetry"`` — ``{"type", "data"}`` (bytes[9])
    * ``"nonstd"``    — ``{"type", "call_to", "call_de", "extra"}``
    * ``"unknown"``   — ``{"type", "payload"}`` (bytes[10])

    Raises ``ValueError`` if *payload* is not exactly 10 bytes.
    """
    ...

# ---------------------------------------------------------------------------
# PSK31 codec classes
# ---------------------------------------------------------------------------

class VaricodeEncoder:
    """PSK31 Varicode encoder: push bytes, drain bit stream."""

    def __init__(self) -> None: ...
    def push_preamble(self, n: int) -> None:
        """Append *n* zero bits as preamble."""
        ...
    def push_byte(self, b: int) -> None:
        """Encode byte *b* and append its Varicode bits."""
        ...
    def push_postamble(self, n: int) -> None:
        """Append *n* zero bits as postamble."""
        ...
    def drain_bits(self) -> NDArray[np.uint8]:
        """Drain all pending bits into a uint8 array."""
        ...
    def is_empty(self) -> bool: ...

class VaricodeDecoder:
    """PSK31 Varicode decoder: push bits, pop decoded bytes."""

    def __init__(self) -> None: ...
    def push_bits(self, bits: NDArray[np.uint8]) -> None:
        """Feed a uint8 array of bits (0/1) into the decoder."""
        ...
    def pop_bytes(self) -> bytes:
        """Drain all decoded bytes."""
        ...

# ---------------------------------------------------------------------------
# PSK31 modulators / demodulators
# ---------------------------------------------------------------------------

class Bpsk31Mod:
    """BPSK31 modulator: differential phase encoding with Hann pulse shaping."""

    def __init__(self, fs: float, rf_hz: float, gain: float = 1.0) -> None: ...
    def set_gain(self, g: float) -> None: ...
    def reset(self) -> None: ...
    def modulate_text(
        self,
        text: bytes,
        preamble_bits: int = 32,
        postamble_bits: int = 32,
    ) -> NDArray[np.complex64]:
        """Encode text via Varicode and modulate to IQ."""
        ...
    def modulate_bits(self, bits: NDArray[np.uint8]) -> NDArray[np.complex64]:
        """Modulate raw differential bits to IQ."""
        ...

class Bpsk31Demod:
    """BPSK31 demodulator: matched-filter symbol detection."""

    def __init__(self, fs: float, rf_hz: float, gain: float = 1.0) -> None: ...
    def set_gain(self, g: float) -> None: ...
    def reset(self) -> None: ...
    def process(self, iq: NDArray[np.complex64]) -> NDArray[np.float32]:
        """Demodulate IQ to soft bits (one float per symbol)."""
        ...

class Bpsk31Decider:
    """BPSK31 hard-decision slicer: threshold soft bits at 0."""

    def __init__(self) -> None: ...
    def process(self, soft: NDArray[np.float32]) -> NDArray[np.uint8]:
        """Threshold soft bits to hard decisions."""
        ...

class Qpsk31Mod:
    """QPSK31 modulator: convolutional encoding + DQPSK + Hann pulse shaping."""

    def __init__(self, fs: float, rf_hz: float, gain: float = 1.0) -> None: ...
    def set_gain(self, g: float) -> None: ...
    def reset(self) -> None: ...
    def modulate_text(
        self,
        text: bytes,
        preamble_bits: int = 32,
        postamble_bits: int = 32,
    ) -> NDArray[np.complex64]:
        """Encode text via Varicode, convolutional-encode, and modulate to IQ."""
        ...
    def modulate_bits(self, bits: NDArray[np.uint8]) -> NDArray[np.complex64]:
        """Modulate raw bits (convolutional encoding + DQPSK) to IQ."""
        ...

class Qpsk31Demod:
    """QPSK31 demodulator with integrated Viterbi decider.

    Call ``process()`` to feed IQ samples (returns soft dibits for inspection),
    then ``flush()`` to run Viterbi and get decoded bits.
    """

    def __init__(self, fs: float, rf_hz: float, gain: float = 1.0) -> None: ...
    def set_gain(self, g: float) -> None: ...
    def reset(self) -> None: ...
    def process(self, iq: NDArray[np.complex64]) -> NDArray[np.float32]:
        """Demodulate IQ to soft dibits (interleaved Re/Im pairs)."""
        ...
    def flush(self) -> NDArray[np.uint8]:
        """Run Viterbi on accumulated dibits and return decoded bits."""
        ...

# ---------------------------------------------------------------------------
# PSK31 streaming decoder
# ---------------------------------------------------------------------------

class Psk31Stream:
    """Streaming PSK31 decoder: demod → decider/Viterbi → Varicode in one step.

    Use ``mode="bpsk"`` for BPSK31 or ``mode="qpsk"`` for QPSK31.
    """

    def __init__(
        self,
        mode: str,
        fs: float,
        carrier_hz: float,
        gain: float = 1.0,
    ) -> None: ...
    def feed(self, iq: NDArray[np.complex64]) -> str:
        """Feed IQ samples and return any newly decoded text."""
        ...
    def flush(self) -> str:
        """Flush the decoder and return any remaining text."""
        ...

# ---------------------------------------------------------------------------
# PSK31 sync functions
# ---------------------------------------------------------------------------

def psk31_sync(
    iq: NDArray[np.complex64],
    fs: float,
    base_hz: float,
    max_hz: float,
    min_carrier_syms: int = 8,
    peak_margin_db: float = 6.0,
    n_bits: int = 1024,
    max_cand: int = 10,
) -> list[dict]:
    """Scan for PSK31 carriers in an IQ buffer.

    Returns a list of candidate dicts::

        {
            "time_sym":   int,
            "freq_bin":   int,
            "carrier_hz": float,
            "score":      float,
            "soft_bits":  float32[N],
        }
    """
    ...

def best_psk31_sync(
    candidates: list[dict],
    carrier_hz: float,
    baud: float = 31.25,
) -> dict | None:
    """Pick the best PSK31 sync result nearest to *carrier_hz*.

    Takes the list returned by ``psk31_sync()`` and returns the best
    candidate dict, or ``None`` if no candidate is within 2×baud.
    """
    ...

# ---------------------------------------------------------------------------
# OFDM
# ---------------------------------------------------------------------------

class OfdmConfig:
    """OFDM waveform configuration: carrier plan + RF/constellation parameters.

    *data_carriers* and *pilot_carrier_indices* use the signed carrier-index
    convention (bin 0 = DC; e.g. ``-26..=26``). *pilot_carrier_indices* and
    *pilot_carrier_values* are parallel arrays of the same length — pass
    empty arrays for no pilots. *constellation* is one of ``"bpsk"``,
    ``"qpsk"``, ``"qam16"``, ``"qam64"``, ``"qam256"``.

    Raises ``ValueError`` for an unknown constellation or an invalid carrier
    plan (overlapping data/pilot carriers, out-of-range indices, or an empty
    data set).
    """

    def __init__(
        self,
        n_fft: int,
        cp_len: int,
        data_carriers: NDArray[np.int32],
        pilot_carrier_indices: NDArray[np.int32],
        pilot_carrier_values: NDArray[np.complex64],
        fs: float,
        rf_hz: float,
        gain: float,
        constellation: str,
    ) -> None: ...
    @property
    def bits_per_ofdm_symbol(self) -> int: ...
    @property
    def samples_per_ofdm_symbol(self) -> int: ...
    # ── COFDM frame-layer configuration (builder-style; each returns a new
    #    config with the field set, for the OfdmFrameMod/OfdmFrameStreamDemod). ──
    def with_outer_fec(self, kind: str, a: int = 0, b: int = 0) -> "OfdmConfig":
        """Set the outer FEC. *kind*: ``"none"`` | ``"bch"`` |
        ``"reed_solomon"``. For BCH, *a* is ``t``. For Reed–Solomon, *a* is
        ``n`` and *b* is ``n_parity`` (``= 2t``)."""
        ...
    def with_inner_fec(self, kind: str, code: str = "") -> "OfdmConfig":
        """Set the inner FEC. *kind*: ``"none"`` | ``"ldpc"`` |
        ``"convolutional"``. For LDPC, *code* is ``"n512r12"`` | ``"n576r23"``
        | ``"n512r34"``. For convolutional, *code* is a puncture rate
        ``"1/2"`` | ``"2/3"`` | ``"3/4"`` | ``"5/6"`` | ``"7/8"``."""
        ...
    def with_ldpc_decode_rule(
        self, kind: str, scale: float = 0.75
    ) -> "OfdmConfig":
        """Select the receiver's LDPC check-node decode rule: ``"sum_product"``
        (default, exact) | ``"min_sum"`` | ``"scaled_min_sum"``. *scale* applies
        only to ``"scaled_min_sum"`` (≈0.75 recovers most of the coding gain).
        Min-sum trades ≲0.3 dB of coding gain for ~2× decode throughput."""
        ...
    def with_interleaver(self, stage: str, rows: int, cols: int) -> "OfdmConfig":
        """Set a rectangular block interleaver on *stage* (``"inner"`` |
        ``"outer"``); ``rows``/``cols`` = 0 disables it."""
        ...
    def with_payload_crc(self, kind: str) -> "OfdmConfig":
        """Set the payload CRC: ``"none"`` | ``"crc16"`` | ``"crc32"``."""
        ...
    def with_header_crc(self, kind: str) -> "OfdmConfig":
        """Set the header CRC: ``"none"`` | ``"crc16"`` | ``"crc32"``."""
        ...
    def with_header_format(self, kind: str) -> "OfdmConfig":
        """Set the header format: ``"orion_sdr"`` | ``"none"``."""
        ...
    def with_scrambler(
        self,
        poly: int,
        width: int,
        seed: int = 1,
        per_frame_random: bool = False,
        position: str = "before_outer",
    ) -> "OfdmConfig":
        """Set an additive PN scrambler (``poly`` = 0 disables). *position*:
        ``"before_outer"`` | ``"after_inner"``."""
        ...
    def validate_frame(self) -> None:
        """Raise ``ValueError`` on an inconsistent frame-layer configuration."""
        ...

class OfdmMod:
    """OFDM transmitter: fused mapper + resource-grid mapping + IFFT + cyclic
    prefix + optional RF upconversion.

    Input: uint8 array of bits (LSB of each byte); consumed
    ``bits_per_ofdm_symbol`` at a time, zero-padding a final partial symbol.
    Output: complex64 IQ array, ``samples_per_ofdm_symbol`` samples/symbol.
    """

    def __init__(self, cfg: OfdmConfig) -> None: ...
    def modulate(self, bits: NDArray[np.uint8]) -> NDArray[np.complex64]: ...

class OfdmDemod:
    """OFDM receiver: fused cyclic-prefix removal + FFT + channel
    equalization + resource-grid extraction + hard-decision decoding.

    *equalizer* selects the channel-estimation method:

    * ``"training_symbol"`` (the default) — one estimate per packet, held
      constant. Call ``estimate_channel()`` once with a demodulated training
      symbol before the first ``demodulate()``/``demodulate_soft()`` call.
    * ``"pilot_interp"`` — re-estimated every symbol from in-band pilots
      (frequency-domain linear interpolation); no separate estimation call
      needed.

    Input: complex64 IQ array, ``samples_per_ofdm_symbol`` samples/symbol.
    Output: uint8 array of bits, ``bits_per_ofdm_symbol`` per symbol.
    Raises ``ValueError`` if input is shorter than one OFDM symbol.
    """

    def __init__(self, cfg: OfdmConfig, equalizer: str = "training_symbol") -> None: ...
    def estimate_channel(self, training_iq: NDArray[np.complex64]) -> None:
        """Estimate and hold the channel from a demodulated training symbol.

        Only meaningful for the ``"training_symbol"`` equalizer; a no-op
        under ``"pilot_interp"``.
        """
        ...
    def demodulate(self, iq: NDArray[np.complex64]) -> NDArray[np.uint8]: ...
    def demodulate_soft(
        self, iq: NDArray[np.complex64]
    ) -> tuple[NDArray[np.complex64], NDArray[np.uint8]]:
        """Like ``demodulate()``, but also returns the pre-decision soft
        symbols (post-equalization, post-grid-extract) as
        ``(soft_symbols, bits)``, for building an ``OfdmRxFrame`` via
        ``build_ofdm_rx_frame``.
        """
        ...

class OfdmRxFrame:
    """Per-packet OFDM RX diagnostics.

    Fields that require acquisition or equalization stay ``None`` until the
    caller has actually run those stages.
    """

    @property
    def bits(self) -> NDArray[np.uint8]: ...
    @property
    def num_symbols(self) -> int: ...
    @property
    def evm_db(self) -> float | None: ...
    @property
    def cfo_hz(self) -> float | None: ...
    @property
    def timing_offset_samples(self) -> int | None: ...
    @property
    def channel_mse(self) -> float | None: ...

def build_ofdm_rx_frame(
    cfg: OfdmConfig,
    soft_symbols: NDArray[np.complex64],
    bits: NDArray[np.uint8],
) -> OfdmRxFrame:
    """Build an ``OfdmRxFrame`` from demodulated soft symbols and their
    corresponding hard-decided bits (as returned by
    ``OfdmDemod.demodulate_soft``, concatenated across all symbols in the
    packet). ``evm_db`` is always populated; ``cfo_hz``,
    ``timing_offset_samples``, and ``channel_mse`` require acquisition/
    equalization info not carried by this function alone and are ``None``.
    """
    ...

def ofdm_sync(
    iq: NDArray[np.complex64],
    fs: float,
    num_repeats: int,
    repeat_len: int,
    search_start: int,
    search_end: int,
    training_n_fft: int | None = None,
    training_cp_len: int | None = None,
) -> list[dict]:
    """Search an OFDM IQ buffer for a repeated-segment (Schmidl & Cox-style)
    preamble match.

    Pass *training_n_fft*/*training_cp_len* to enable wide-range integer-CFO
    recovery via a dedicated training symbol (must match the values used
    with ``generate_ofdm_preamble``); omit both for fractional-CFO-only
    acquisition (capture range ±½ the subcarrier spacing).

    Returns a list of dicts, sorted by descending score::

        {
            "start_sample":     int,    # sample offset of the preamble start
            "cfo_hz":           float,  # fractional CFO estimate (Hz)
            "integer_cfo_bins": int,    # whole subcarrier-spacing units
            "score":            float,  # normalized timing-metric score
        }

    Total CFO is ``cfo_hz + integer_cfo_bins * (fs / n_fft)``.
    ``integer_cfo_bins`` is always 0 unless a training symbol was supplied.
    """
    ...

def generate_ofdm_preamble(
    cfg: OfdmConfig,
    num_repeats: int,
    repeat_len: int,
    training_n_fft: int | None = None,
    training_cp_len: int | None = None,
) -> NDArray[np.complex64]:
    """Generate a repeated-segment preamble (plus training symbol, if
    *training_n_fft*/*training_cp_len* are given) for prepending before OFDM
    data symbols. See ``ofdm_sync`` for the matching acquisition function.
    """
    ...

# ── COFDM frame (MAC) layer ────────────────────────────────────────────────

class FramePacket:
    """A MAC-layer frame: an opaque byte payload plus metadata.

    *payload* is a uint8 array. *sequence_num*, *mcs_index*, and *flags* are
    carried in the frame header (for the ``"orion_sdr"`` header format).
    """

    def __init__(
        self,
        payload: NDArray[np.uint8],
        sequence_num: int = 0,
        mcs_index: int = 0,
        flags: int = 0,
    ) -> None: ...
    @property
    def payload(self) -> NDArray[np.uint8]: ...
    @property
    def sequence_num(self) -> int: ...
    @property
    def mcs_index(self) -> int: ...
    @property
    def flags(self) -> int: ...

class McsTable:
    """Maps each frame's ``mcs_index`` to a modulation-and-coding scheme
    (constellation + inner/outer FEC). The transmitter and receiver must share
    the same table.
    """

    def __init__(self) -> None: ...
    @staticmethod
    def default_ladder() -> "McsTable":
        """BPSK/QPSK/QAM-16/QAM-64, each with an LDPC(n512r12) inner code and a
        BCH(t=8) outer code."""
        ...
    def add(
        self,
        constellation: str,
        inner_kind: str = "none",
        inner_code: str = "",
        outer_kind: str = "none",
        outer_a: int = 0,
        outer_b: int = 0,
    ) -> None:
        """Append an MCS entry. *constellation* is ``"bpsk"``…``"qam256"``;
        *inner*/*outer* mirror ``OfdmConfig.with_inner_fec``/``with_outer_fec``.
        """
        ...
    @property
    def len(self) -> int: ...

class CodecCache:
    """A shared cache of constructed FEC codes (LDPC/BCH/Reed-Solomon).

    Building a code (the LDPC parity-check matrix especially) costs
    milliseconds and depends only on its parameters, so it need only be done
    once per link. Pass one ``CodecCache`` to an ``OfdmFrameMod``, an
    ``OfdmFrameStreamDemod``, and/or ``demodulate_frame`` (via ``cache=``) to
    build each code once and reuse it across all of them — a transmitter and
    receiver on the same MCS then share the built codes. Omitting ``cache=``
    gives each object its own private cache, which still amortizes across that
    object's own calls.
    """

    def __init__(self) -> None: ...

class OfdmFrameMod:
    """COFDM frame transmitter: serializes a ``FramePacket`` to a flat IQ
    stream (``[preamble + training][header][payload]``), applying the
    concatenated FEC chain configured on *cfg* and selected per frame by
    *mcs_table*.
    """

    def __init__(
        self,
        cfg: OfdmConfig,
        mcs_table: McsTable,
        num_repeats: int = 4,
        repeat_len: int = 16,
        cache: CodecCache | None = None,
    ) -> None: ...
    def modulate_frame(
        self, frame: FramePacket, per_frame_seed: int = 0
    ) -> NDArray[np.complex64]:
        """Modulate a whole frame into IQ. *per_frame_seed* supplies the
        scrambler seed for a per-frame-random configuration."""
        ...

class OfdmFrameStreamDemod:
    """Streaming COFDM frame receiver. Push IQ with ``feed()``; it locates
    preambles, corrects CFO, estimates the channel from the training symbol,
    decodes each frame, and returns the completed ones.
    """

    def __init__(
        self,
        cfg: OfdmConfig,
        mcs_table: McsTable,
        num_repeats: int = 4,
        repeat_len: int = 16,
        cache: CodecCache | None = None,
    ) -> None: ...
    def feed(self, iq: NDArray[np.complex64]) -> list[FramePacket]:
        """Feed IQ; return the frames that completed. Failed decodes are
        omitted (see ``feed_with_errors``)."""
        ...
    def feed_with_errors(
        self, iq: NDArray[np.complex64]
    ) -> list[tuple[FramePacket | None, str | None]]:
        """Like ``feed``, but each result is ``(frame_or_None, error_or_None)``
        so decode failures are observable."""
        ...
    def flush(self) -> list[FramePacket]:
        """Run a final decode pass over the residual buffer."""
        ...
    @property
    def buffered(self) -> int: ...
    def clear(self) -> None: ...

def demodulate_frame(
    cfg: OfdmConfig,
    mcs_table: McsTable,
    iq: NDArray[np.complex64],
    cache: CodecCache | None = None,
) -> FramePacket:
    """Batch-demodulate a single frame at a known start (*iq*[0] is the first
    sample after the preamble+training). Raises ``ValueError`` on a decode
    failure. Pass ``cache=`` a ``CodecCache`` to reuse built FEC codes across a
    batch of calls (or share them with a modulator). See ``OfdmFrameStreamDemod``
    for the streaming path.
    """
    ...

# ---------------------------------------------------------------------------
# Conformant DVB-T on-air frame (EN 300 744)
# ---------------------------------------------------------------------------

class DvbTFrameParams:
    """Transmission parameters for a conformant DVB-T frame. *guard* is one of
    ``"1/32" | "1/16" | "1/8" | "1/4"``; *constellation* one of
    ``"qpsk" | "qam16" | "qam64"``; *code_rate* one of
    ``"1/2" | "2/3" | "3/4" | "5/6" | "7/8"``.
    """

    def __init__(
        self,
        guard: str,
        constellation: str,
        code_rate: str,
        frame_number: int = 0,
        cell_id: int = 0,
    ) -> None: ...
    @property
    def guard(self) -> str: ...
    @property
    def constellation(self) -> str: ...
    @property
    def code_rate(self) -> str: ...
    @property
    def frame_number(self) -> int: ...
    @property
    def cell_id(self) -> int: ...

class TpsWord:
    """The transmission parameters recovered from a frame's TPS carriers."""

    @property
    def frame_number(self) -> int: ...
    @property
    def constellation(self) -> str: ...
    @property
    def code_rate(self) -> str: ...
    @property
    def guard(self) -> str: ...
    @property
    def cell_id(self) -> int: ...

class DvbTFrame:
    """A modulated DVB-T frame: time-domain IQ plus its numerology."""

    @property
    def iq(self) -> NDArray[np.complex64]: ...
    @property
    def n_symbols(self) -> int: ...
    @property
    def samples_per_symbol(self) -> int: ...

class DvbTRxFrame:
    """The recovered contents of a DVB-T frame: TS payload and TPS word."""

    @property
    def payload(self) -> bytes: ...
    @property
    def tps(self) -> TpsWord: ...

def dvb_t_frame_modulate(
    params: DvbTFrameParams, payload: NDArray[np.uint8]
) -> DvbTFrame:
    """Modulate *payload* (MPEG-TS payload bytes) into one conformant,
    preamble-less DVB-T frame."""
    ...

def dvb_t_frame_demodulate(
    params: DvbTFrameParams,
    iq: NDArray[np.complex64],
    n_symbols: int,
    payload_len: int,
) -> DvbTRxFrame:
    """Demodulate one conformant DVB-T frame, acquiring the symbol grid from the
    guard interval (no preamble). *n_symbols* comes from the paired
    ``dvb_t_frame_modulate`` result; *payload_len* is the original payload byte
    count. Raises ``ValueError`` on any acquisition/decode failure.
    """
    ...

def nb_bandwidth_fs(mode: str) -> float:
    """Sample rate (S/s) for a narrowband DVB-T mode: ``"333khz" | "1mhz" |
    "2mhz"``. ``fs = occupied_hz * 2048/1705``."""
    ...

def nb_bandwidth_occupied_hz(mode: str) -> float:
    """Nominal occupied RF bandwidth (Hz) for a narrowband DVB-T mode."""
    ...
