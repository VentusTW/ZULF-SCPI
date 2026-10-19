"""Simulated ADC data, used instead of the device until the hardware is available.

Every channel is a sine or a cosine (drawn at random for every channel of every
transient) of constant amplitude whose frequency is set by the shim current of
that channel. Every next transient is changed a little (frequency, phase), so
the transients and their average differ.
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
# Change of every next transient of a set (transient 1 is unchanged): the line
# drifts and the phase moves by PHASE_STEP_RAD; on top of that every transient
# gets a random amplitude within 1 +/- AMPLITUDE_SPREAD.
DRIFT_HZ_PER_TRANSIENT = 0.5
PHASE_STEP_RAD = np.pi / 4  # 45 degrees
AMPLITUDE_SPREAD = 0.15
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


def line_frequency_hz(config: ExperimentConfig, channel: int, transient: int = 1) -> float:
    """Frequency of the simulated line of an ADC channel in a given transient (1-based)."""
    shift = SHIM_SHIFT_HZ_PER_MA * shim_currents_mA(config).get(channel, 0.0)
    return J_COUPLING_HZ + shift + DRIFT_HZ_PER_TRANSIENT * (transient - 1)


def simulate_transient(config: ExperimentConfig, transient: int = 1,
                       rng: np.random.Generator | None = None) -> np.ndarray:
    """Transient number `transient` (1-based), shape (number_of_samples, len(channels)).

    Channel ch: a/ch x {sin or cos}(2 pi f t + PHASE_STEP_RAD x (n-1)), with f from
    line_frequency_hz, t the time after :AWG:RUN, sin or cos drawn at random and
    a drawn from 1 +/- AMPLITUDE_SPREAD.
    """
    rng = rng or np.random.default_rng()
    rate = config.hardware_parameters.acq_sample_rate_hz
    acq = config.acquisition
    t = acq.start_us / 1e6 + np.arange(acq.number_of_samples) / rate
    phase = PHASE_STEP_RAD * (transient - 1)
    columns = []
    for ch in acq.channels:
        wave = np.sin if rng.random() < 0.5 else np.cos
        amplitude = rng.uniform(1 - AMPLITUDE_SPREAD, 1 + AMPLITUDE_SPREAD)
        columns.append(amplitude / ch * wave(2 * np.pi * line_frequency_hz(config, ch, transient) * t + phase))
    return np.column_stack(columns)
