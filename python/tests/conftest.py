"""Shared fixtures and helpers for orion_sdr pytest tests."""

import numpy as np
import pytest
from numpy.typing import NDArray


FS = 48_000.0  # default sample rate used across tests


def real_tone(fs: float, f_hz: float, n: int, amp: float = 1.0) -> NDArray[np.float32]:
    """Generate a real sinusoidal tone."""
    t = np.arange(n, dtype=np.float32) / fs
    return (amp * np.sin(2.0 * np.pi * f_hz * t)).astype(np.float32)


def complex_tone(fs: float, f_hz: float, n: int, amp: float = 1.0) -> NDArray[np.complex64]:
    """Generate a complex exponential tone (baseband IQ)."""
    t = np.arange(n, dtype=np.float32) / fs
    return (amp * np.exp(1j * 2.0 * np.pi * f_hz * t)).astype(np.complex64)


def snr_db(x: NDArray[np.float32], fs: float, f_hz: float) -> float:
    """Single-bin DFT SNR estimate: power at f_hz vs power at f_hz * 0.73."""
    n = len(x)
    k = np.arange(n, dtype=np.float64)

    def dft_power(f: float) -> float:
        w = -2.0 * np.pi * f / fs * k
        phasor = np.cos(w) + 1j * np.sin(w)
        c = float(np.abs(np.dot(phasor, x)) ** 2) / (n * n)
        return c

    p_sig = dft_power(f_hz)
    p_off = dft_power(f_hz * 0.73) + 1e-20
    return 10.0 * np.log10(p_sig / p_off)


def tail(x: NDArray, fraction: float = 0.75) -> NDArray:
    """Drop the leading transient, return the trailing fraction of the array."""
    return x[int(len(x) * (1.0 - fraction)):]


@pytest.fixture
def silence_iq() -> NDArray[np.complex64]:
    return np.zeros(4096, dtype=np.complex64)


@pytest.fixture
def silence_audio() -> NDArray[np.float32]:
    return np.zeros(4096, dtype=np.float32)
