"""Conversion of high-level events into binary blocks and SCPI commands."""

from __future__ import annotations

from dataclasses import dataclass

from zulf_nmr.core.scpi import header_matches, short_form
from zulf_nmr.experiment.parser import ExperimentConfig
from zulf_nmr.hardware.acquisition import Acquisition, duration_for_samples
from zulf_nmr.hardware.sequencer import ALLOWED_EVENTS, Sequencer


class _RecordingConnection:
    """Stands in for SCPIConnection and records what the drivers would send.

    Reusing the drivers keeps command formatting and range checks in one place.
    """

    def __init__(self) -> None:
        self.commands: list[str] = []

    def write(self, command: str) -> None:
        self.commands.append(command)

    def query(self, command: str, timeout_s: float | None = None) -> str:
        raise RuntimeError(f"Queries cannot be compiled into a sequence: {command!r}")

    def query_binary(self, command: str, timeout_s: float | None = None) -> bytes:
        raise RuntimeError(f"Queries cannot be compiled into a sequence: {command!r}")


@dataclass(frozen=True)
class CompiledSequence:
    """SCPI commands that run one experiment, in the order they must be sent."""

    setup: tuple[str, ...]  # clear the old sequence, configure the acquisition
    events: tuple[str, ...]  # :SEQuence:ADD <time>,'<event>' for every event
    start: str  # software trigger

    def commands(self) -> list[str]:
        return [*self.setup, *self.events, self.start]


def canonical_event(action: str) -> str:
    """Rewrite an event in canonical form: absolute short-form header, upper-case
    keywords, arguments without spaces, e.g. ':motor:move  measure' -> ':MOT:MOVE MEASURE'.
    """
    header, _, arg_text = action.strip().partition(" ")
    if not header.startswith(":"):
        header = ":" + header
    for pattern in ALLOWED_EVENTS:
        if header_matches(header, pattern):
            header = short_form(pattern)
            break
    else:
        raise ValueError(f"{action!r} is not an allowed sequence event")
    args = [a.strip() for a in arg_text.split(",")] if arg_text.strip() else []
    if header == short_form(":MOTor:MOVE"):
        args = [a.upper() for a in args]
    return f"{header} {','.join(args)}" if args else header


def acquisition_window_ms(config: ExperimentConfig) -> tuple[float, float] | None:
    """(start, end) of the acquisition in ms of the sequence, None if nothing triggers it.

    acquisition.start counts from the first :AWG:RUN event (:ACQuire:AUTostart);
    without :AWG:RUN a :ACQuire:RUN event starts the recording at once.
    """
    acq = config.acquisition
    duration_ms = acq.duration_s(config.hardware_parameters.acq_sample_rate_hz) * 1000
    for event in config.events:  # sorted by time
        header = event.action.strip().split(" ", 1)[0]
        if header_matches(header, ":AWG:RUN"):
            start = event.time_ms + acq.start_us / 1000
            return start, start + duration_ms
    for event in config.events:
        if header_matches(event.action.strip().split(" ", 1)[0], ":ACQuire:RUN"):
            return event.time_ms, event.time_ms + duration_ms
    return None


def compile_experiment(config: ExperimentConfig) -> CompiledSequence:
    """Convert a validated experiment into the SCPI commands that run it.

    The AWG waveform (awg_sequence) is not included yet - the .npz layout
    is not defined.
    """
    hw = config.hardware_parameters
    acq = config.acquisition

    rec = _RecordingConnection()
    acquisition = Acquisition(rec)
    sequencer = Sequencer(rec)

    sequencer.reset()
    acquisition.set_sample_rate(hw.acq_sample_rate_hz)
    # The device takes a duration in ms; it is rounded up so that at least
    # number_of_samples are recorded (the runner keeps exactly that many).
    acquisition.set_duration(duration_for_samples(acq.number_of_samples, hw.acq_sample_rate_hz))
    acquisition.set_autostart(acq.start_us, enable=True)  # us after :AWG:RUN
    setup = tuple(rec.commands)

    rec.commands.clear()
    for event in config.events:  # already sorted by time_ms
        sequencer.add(event.time_ms, canonical_event(event.action))
    events = tuple(rec.commands)

    rec.commands.clear()
    sequencer.run()
    (start,) = rec.commands

    return CompiledSequence(setup=setup, events=events, start=start)
