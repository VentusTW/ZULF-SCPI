"""Queue of experiment sets run one after another, with a delay between them."""

from __future__ import annotations

import json
import logging
import math
import shutil
import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np

from zulf_nmr.core.spectrometer import Spectrometer
from zulf_nmr.experiment.parser import (
    ConfigError,
    ExperimentConfig,
    HardwareLimits,
    load_experiment_config,
)
from zulf_nmr.experiment.runner import ExperimentAborted, ExperimentRunner, average, save_csv
from zulf_nmr.experiment.simulation import simulate_set
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


@dataclass
class Transient:
    number: int  # nnn in the file name: 1, 2, 3... within its configuration
    set_index: int  # position in the queue, 1-based
    file: Path
    data: np.ndarray


@dataclass
class ResultGroup:
    """All sets of the queue that used the same configuration file."""

    name: str  # experiment_name, made unique within the run
    config_source: Path
    sample_rate_hz: float
    file_format: str
    config_file: Path
    awg_file: Path | None = None
    transients: list[Transient] = field(default_factory=list)
    average_file: Path | None = None
    average: np.ndarray | None = None


@dataclass
class QueueResult:
    """Where the results of one queue run were saved."""

    folder: Path
    groups: list[ResultGroup] = field(default_factory=list)  # in order of first use

    @property
    def files(self) -> list[Path]:
        """Every saved file, grouped by configuration: config, AWG, transients, average."""
        files: list[Path] = []
        for group in self.groups:
            files += [group.config_file, *([group.awg_file] if group.awg_file else [])]
            files += [t.file for t in group.transients]
            files += [group.average_file] if group.average_file else []
        return files


class QueueRunner:
    """Runs the sets in order; after each set (except the last) waits its delay.

    Results go to <results_root>/<stamp>/ where stamp = yyyymmdd_hhmmss of the start.
    Sets that use the same configuration file form a group; for every group:
        <stamp>_<name>_config.json                  - copy of the configuration file
        <stamp>_<name>_awg.npz                      - copy of its AWG sequence (if any)
        <stamp>_<name>_transient_<nnn>.<format>     - one per set; nnn counts from 001
                                                      separately for every configuration
        <stamp>_<name>_average.<format>             - mean of the group's transients
    name is the experiment_name; two different files with the same experiment_name
    get _2, _3... appended. The average is rewritten after every set, so it is up
    to date even if the queue is aborted.

    simulate=True produces simulated data instead of talking to the device
    (spectrometer may then be None).
    on_status(text) reports what is happening (set started, waiting...);
    on_progress(set_index, transient, received, expected) forwards runner progress.
    """

    def __init__(
        self,
        spectrometer: Spectrometer | None,
        items: list[QueueItem],
        results_root: str | Path,
        simulate: bool = False,
        on_status: Callable[[str], None] | None = None,
        on_progress: Callable[[int, int, int, int], None] | None = None,
    ) -> None:
        problems = queue_errors(items)
        if problems:
            raise ValueError("; ".join(problems))
        if spectrometer is None and not simulate:
            raise ValueError("a spectrometer is required unless simulate=True")
        self.spec = spectrometer
        self.items = list(items)
        self.results_root = Path(results_root)
        self.simulate = simulate
        self.on_status = on_status
        self.on_progress = on_progress
        self.result: QueueResult | None = None  # set as soon as the folder exists
        self._abort = threading.Event()
        self._current: ExperimentRunner | None = None

    def abort(self) -> None:
        """Stop the running set (sends *RST) or the wait between sets."""
        self._abort.set()
        runner = self._current
        if runner is not None:
            runner.abort()

    def run(self) -> QueueResult:
        self._abort.clear()
        stamp = f"{datetime.now():%Y%m%d_%H%M%S}"
        folder = self.results_root / stamp
        folder.mkdir(parents=True, exist_ok=True)
        self.result = result = QueueResult(folder)
        groups: dict[Path, ResultGroup] = {}

        n = len(self.items)
        for index, item in enumerate(self.items, start=1):
            if self._abort.is_set():
                raise ExperimentAborted("Queue aborted by user")
            group = groups.get(item.path.resolve())
            if group is None:
                group = groups[item.path.resolve()] = self._new_group(item, folder, stamp, result.groups)
                result.groups.append(group)

            mode = " (simulated)" if self.simulate else ""
            self._status(f"Set {index}/{n}: {item.name} running{mode}")
            data = self._acquire(item, index)

            number = len(group.transients) + 1
            name = f"{stamp}_{group.name}_transient_{number:03d}.{group.file_format}"
            path = save_csv(folder / name, data, group.sample_rate_hz)
            group.transients.append(Transient(number, index, path, data))
            group.average = average([t.data for t in group.transients])
            group.average_file = save_csv(
                folder / f"{stamp}_{group.name}_average.{group.file_format}", group.average, group.sample_rate_hz
            )

            if index < n and item.delay_s > 0:
                self._status(f"Waiting {format_delay(item.delay_s)} before set {index + 1}/{n}")
                if self._abort.wait(item.delay_s):
                    raise ExperimentAborted("Queue aborted by user")
        return result

    @staticmethod
    def _new_group(item: QueueItem, folder: Path, stamp: str, existing: list[ResultGroup]) -> ResultGroup:
        """Start a group for a configuration file and copy the file (and its AWG sequence)."""
        cfg = item.config
        used = {g.name for g in existing}
        name, k = cfg.experiment_name, 2
        while name in used:
            name, k = f"{cfg.experiment_name}_{k}", k + 1
        config_file = folder / f"{stamp}_{name}_config.json"
        shutil.copyfile(item.path, config_file)
        awg_file = None
        if item.awg is not None and item.awg.source_path is not None:
            awg_file = folder / f"{stamp}_{name}_awg{item.awg.source_path.suffix}"
            shutil.copyfile(item.awg.source_path, awg_file)
        return ResultGroup(
            name=name,
            config_source=item.path,
            sample_rate_hz=cfg.hardware_parameters.acq_sample_rate_hz,
            file_format=cfg.output.format,
            config_file=config_file,
            awg_file=awg_file,
        )

    def _acquire(self, item: QueueItem, index: int) -> np.ndarray:
        """Data of one set: the average of its output.average_count transients."""
        if self.simulate:
            return simulate_set(item.config)
        self._current = ExperimentRunner(
            self.spec,
            item.config,
            awg_waveform=item.awg,
            auto_save=False,  # saved here, with the queue's names
            on_progress=lambda t, r, e: self.on_progress and self.on_progress(index, t, r, e),
        )
        try:
            return self._current.run().average
        finally:
            self._current = None

    def _status(self, text: str) -> None:
        log.info(text)
        if self.on_status:
            self.on_status(text)
