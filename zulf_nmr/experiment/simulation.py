"""Simulated ADC data, used instead of the device until the hardware is available.

Every channel is a sine or a cosine (drawn once per channel for a whole set) of
constant amplitude whose frequency is set by the shim current of that channel.
Every transient gets its own random phase, frequency and amplitude offsets
within PHASE_JITTER_DEG, FREQUENCY_JITTER_HZ and AMPLITUDE_JITTER, so the
transients and their average differ.
"""

from __future__ import annotations

import numpy as np

from zulf_nmr.core.scpi import header_matches
from zulf_nmr.experiment.parser import ExperimentConfig
from zulf_nmr.experiment.sequence import acquisition_window_ms

# Zero-field J-coupling line of a 13C-1H pair (e.g. 13C-formic acid).
J_COUPLING_HZ = 222.0
# Toy model of the shims: the current of shim channel n (:SHIM:CURRent n, <mA>)
# leaves a residual field that shifts the line seen by ADC channel n by
# SHIM_SHIFT_HZ_PER_MA x current. ADC channels without a shim are not shifted.
SHIM_SHIFT_HZ_PER_MA = 2.0
# Random change of every transient, drawn separately for each channel:
PHASE_JITTER_DEG = 45.0  # phase offset within +/- this
FREQUENCY_JITTER_HZ = 0.75  # line frequency within +/- this
AMPLITUDE_JITTER = 0.75  # amplitude factor within 1 +/- this
# A simulated transient is delivered in SIM_CHUNKS parts over SIM_TRANSIENT_S
# seconds, so the GUI can show it filling up like a real acquisition.
SIM_TRANSIENT_S = 3.0
SIM_CHUNKS = 30


def shim_currents_mA(config: ExperimentConfig) -> dict[int, float]:
    """Shim current per channel when the acquisition starts: the last :SHIM:CURRent
    event for that channel at or before the start (all of them if there is no start)."""
    window = acquisition_window_ms(config)
    start = window[0] if window else float("inf")
    currents: dict[int, float] = {}
    for event in config.events:  # sorted by time
        header, _, args = event.action.strip().partition(" ")
        if event.time_ms > start or not header_matches(header, ":SHIM:CURRent"):
            continue
        channel, current = (a.strip() for a in args.split(","))  # validated by the parser
        currents[int(channel)] = float(current)
    return currents


def line_frequency_hz(config: ExperimentConfig, channel: int) -> float:
    """Frequency of the simulated line of an ADC channel, before the random jitter."""
    shift = SHIM_SHIFT_HZ_PER_MA * shim_currents_mA(config).get(channel, 0.0)
    return J_COUPLING_HZ + shift


def choose_waves(config: ExperimentConfig, rng: np.random.Generator | None = None) -> dict[int, str]:
    """'sin' or 'cos' for every channel, drawn once and shared by all transients of a set
    (a new draw per transient would shift the phase by 90 degrees, beyond PHASE_JITTER_DEG)."""
    rng = rng or np.random.default_rng()
    return {ch: "sin" if rng.random() < 0.5 else "cos" for ch in config.acquisition.channels}


def simulate_transient(config: ExperimentConfig, waves: dict[int, str] | None = None,
                       rng: np.random.Generator | None = None) -> np.ndarray:
    """One transient, shape (number_of_samples, len(channels)).

    Channel ch: a/ch x waves[ch](2 pi (f + df) t + dphi), with f from line_frequency_hz,
    t the time after :AWG:RUN and, drawn for every call, a within 1 +/- AMPLITUDE_JITTER,
    df within +/- FREQUENCY_JITTER_HZ and dphi within +/- PHASE_JITTER_DEG.
    waves defaults to a new choose_waves().
    """
    rng = rng or np.random.default_rng()
    waves = waves or choose_waves(config, rng)
    rate = config.hardware_parameters.acq_sample_rate_hz
    acq = config.acquisition
    t = acq.start_us / 1e6 + np.arange(acq.number_of_samples) / rate
    columns = []
    for ch in acq.channels:
        wave = np.sin if waves[ch] == "sin" else np.cos
        amplitude = rng.uniform(1 - AMPLITUDE_JITTER, 1 + AMPLITUDE_JITTER)
        frequency = line_frequency_hz(config, ch) + rng.uniform(-FREQUENCY_JITTER_HZ, FREQUENCY_JITTER_HZ)
        phase = np.deg2rad(rng.uniform(-PHASE_JITTER_DEG, PHASE_JITTER_DEG))
        columns.append(amplitude / ch * wave(2 * np.pi * frequency * t + phase))
    return np.column_stack(columns)
