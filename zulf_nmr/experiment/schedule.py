"""Queue of experiment sets run one after another, with a delay between them."""

from __future__ import annotations

import json
import logging
import math
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from zulf_nmr.core.spectrometer import Spectrometer
from zulf_nmr.experiment.parser import (
    ConfigError,
    ExperimentConfig,
    HardwareLimits,
    load_experiment_config,
)
from zulf_nmr.experiment.runner import ExperimentAborted, ExperimentRunner, RunResult
from zulf_nmr.hardware.awg import AwgWaveform, load_awg_npz

log = logging.getLogger(__name__)

# Delay units -> seconds. The delay is timed by the PC, so sub-millisecond
# values are only as precise as the operating system's sleep.
TIME_UNITS: dict[str, float] = {"h": 3600.0, "min": 60.0, "s": 1.0, "ms": 1e-3, "us": 1e-6}


def parse_delay(value: str | float, unit: str) -> float:
    """Return the delay in seconds; raise ValueError for a bad number or unit."""
    if unit not in TIME_UNITS:
        raise ValueError(f"unknown time unit {unit!r}; use one of {', '.join(TIME_UNITS)}")
    try:
        number = float(str(value).strip().replace(",", "."))
    except ValueError:
        raise ValueError(f"{value!r} is not a number") from None
    if not math.isfinite(number) or number < 0:
        raise ValueError("delay must be a number >= 0")
    return number * TIME_UNITS[unit]


def format_delay(seconds: float) -> str:
    """Human-readable delay, e.g. 90 -> '1.5 min', 0.002 -> '2 ms'."""
    if seconds == 0:
        return "0 s"
    for unit in ("h", "min", "s", "ms", "us"):
        if seconds >= TIME_UNITS[unit] or unit == "us":
            return f"{seconds / TIME_UNITS[unit]:g} {unit}"
    return f"{seconds:g} s"


@dataclass
class QueueItem:
    """One set: an experiment file (with its AWG sequence) and the delay before the next set."""

    path: Path
    delay_value: str = "0"
    delay_unit: str = "s"
    config: ExperimentConfig | None = None
    awg: AwgWaveform | None = None
    errors: list[str] = field(default_factory=list)

    @property
    def name(self) -> str:
        return self.config.experiment_name if self.config else self.path.stem

    @property
    def is_valid(self) -> bool:
        """The experiment file and its AWG sequence are valid (the delay is checked separately)."""
        return self.config is not None and not self.errors

    @property
    def delay_s(self) -> float:
        return parse_delay(self.delay_value, self.delay_unit)

    @property
    def delay_error(self) -> str | None:
        try:
            self.delay_s
        except ValueError as e:
            return str(e)
        return None

    def reload(self, limits: HardwareLimits) -> None:
        """(Re)read and validate the experiment file and its AWG sequence."""
        self.config, self.awg, self.errors = None, None, []
        try:
            with open(self.path, encoding="utf-8") as f:
                data = json.load(f)
        except FileNotFoundError:
            self.errors = ["file not found"]
            return
        except (OSError, json.JSONDecodeError) as e:
            self.errors = [f"cannot read the file: {e}"]
            return
        if isinstance(data, dict) and "connection" in data and "events" not in data:
            self.errors = ["this is a connection file (system.json), not an experiment - "
                           "use 'Load connection' in the main window"]
            return
        try:
            config = load_experiment_config(self.path, limits=limits)
        except ConfigError as e:
            self.errors = list(e.errors)
            return
        awg = None
        if config.awg_sequence is not None:
            try:
                awg = load_awg_npz(
                    config.awg_sequence.file_path,
                    max_sample_rate_hz=limits.max_awg_sample_rate_hz,
                    max_amplitude_A=limits.max_awg_amplitude_A,
                )
            except ValueError as e:
                self.errors = [f"AWG sequence: {e}"]
                return
        self.config, self.awg = config, awg


def load_item(path: str | Path, limits: HardwareLimits, delay_value: str = "0", delay_unit: str = "s") -> QueueItem:
    item = QueueItem(Path(path), delay_value, delay_unit)
    item.reload(limits)
    return item


def queue_errors(items: list[QueueItem]) -> list[str]:
    """Every problem that prevents running the queue (the last set's delay is not used)."""
    if not items:
        return ["the queue is empty"]
    problems = []
    for i, item in enumerate(items, start=1):
        for message in item.errors or ([] if item.config else ["not loaded"]):
            problems.append(f"Set {i} ({item.path.name}): {message}")
        if i < len(items) and item.delay_error:
            problems.append(f"Set {i} ({item.path.name}): invalid delay - {item.delay_error}")
    return problems


class QueueRunner:
    """Runs the sets in order; after each set (except the last) waits its delay.

    on_status(text) reports what is happening (set started, waiting...);
    on_progress(set_index, transient, received, expected) forwards runner progress.
    """

    def __init__(
        self,
        spectrometer: Spectrometer,
        items: list[QueueItem],
        on_status: Callable[[str], None] | None = None,
        on_progress: Callable[[int, int, int, int], None] | None = None,
    ) -> None:
        problems = queue_errors(items)
        if problems:
            raise ValueError("; ".join(problems))
        self.spec = spectrometer
        self.items = list(items)
        self.on_status = on_status
        self.on_progress = on_progress
        self._abort = threading.Event()
        self._current: ExperimentRunner | None = None

    def abort(self) -> None:
        """Stop the running set (sends *RST) or the wait between sets."""
        self._abort.set()
        runner = self._current
        if runner is not None:
            runner.abort()

    def run(self) -> list[RunResult]:
        self._abort.clear()
        results: list[RunResult] = []
        n = len(self.items)
        for index, item in enumerate(self.items, start=1):
            if self._abort.is_set():
                raise ExperimentAborted("Queue aborted by user")
            self._status(f"Set {index}/{n}: {item.name} running")
            self._current = ExperimentRunner(
                self.spec,
                item.config,
                awg_waveform=item.awg,
                on_progress=lambda t, r, e, i=index: self.on_progress and self.on_progress(i, t, r, e),
            )
            try:
                results.append(self._current.run())
            finally:
                self._current = None
            if index < n and item.delay_s > 0:
                self._status(f"Waiting {format_delay(item.delay_s)} before set {index + 1}/{n}")
                if self._abort.wait(item.delay_s):
                    raise ExperimentAborted("Queue aborted by user")
        return results

    def _status(self, text: str) -> None:
        log.info(text)
        if self.on_status:
            self.on_status(text)
