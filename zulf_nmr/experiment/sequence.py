"""Conversion of high-level events into binary blocks and SCPI commands."""

from __future__ import annotations

from dataclasses import dataclass

from zulf_nmr.core.scpi import header_matches, short_form
from zulf_nmr.experiment.parser import ExperimentConfig
from zulf_nmr.hardware.acquisition import Acquisition
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


def compile_experiment(config: ExperimentConfig) -> CompiledSequence:
    """Convert a validated experiment into the SCPI commands that run it.

    The AWG waveform (awg_sequence) is not included yet - the .npz layout
    is not defined.
    """
    hw = config.hardware_parameters

    rec = _RecordingConnection()
    acquisition = Acquisition(rec)
    sequencer = Sequencer(rec)

    sequencer.reset()
    acquisition.set_sample_rate(hw.acq_sample_rate_hz)
    acquisition.set_duration(hw.acq_duration_ms)
    # ASSUMED: acq_start_awg_time is in us, the unit of :ACQuire:AUTostart.
    acquisition.set_autostart(int(round(hw.acq_start_awg_time)), enable=True)
    setup = tuple(rec.commands)

    rec.commands.clear()
    for event in config.events:  # already sorted by time_ms
        sequencer.add(event.time_ms, canonical_event(event.action))
    events = tuple(rec.commands)

    rec.commands.clear()
    sequencer.run()
    (start,) = rec.commands

    return CompiledSequence(setup=setup, events=events, start=start)
