"""Simulated ADC data, used instead of the device until the hardware is available."""

from __future__ import annotations

import numpy as np

from zulf_nmr.experiment.parser import ExperimentConfig

# Zero-field J-spectrum of a 13C-1H pair (e.g. 13C-formic acid): lines at J and 2J.
J_COUPLING_HZ = 222.0
LINES = ((J_COUPLING_HZ, 1.0), (2 * J_COUPLING_HZ, 0.3))  # (frequency Hz, relative amplitude)
T2_S = 2.0
NOISE_RMS = 0.5


def simulate_transient(config: ExperimentConfig, rng: np.random.Generator | None = None) -> np.ndarray:
    """One transient: decaying J-coupling lines plus white noise.

    Length and time step follow acq_sample_rate_hz and acq_duration_ms. Lines
    above 0.45 x the sample rate are left out (they would alias).
    """
    rng = rng or np.random.default_rng()
    hw = config.hardware_parameters
    n = int(round(hw.acq_sample_rate_hz * hw.acq_duration_ms / 1000))
    t = np.arange(n) / hw.acq_sample_rate_hz
    signal = np.zeros(n)
    for freq, amplitude in LINES:
        if freq < 0.45 * hw.acq_sample_rate_hz:
            phase = rng.uniform(-0.05, 0.05)  # small shot-to-shot phase jitter
            signal += amplitude * np.cos(2 * np.pi * freq * t + phase)
    signal *= np.exp(-t / T2_S)
    return signal + rng.normal(0.0, NOISE_RMS, n)

