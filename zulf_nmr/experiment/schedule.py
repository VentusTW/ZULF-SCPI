"""Queue of experiment sets run one after another, with a delay between them."""

from __future__ import annotations

import json
import logging
import math
import shutil
import threading
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np

from zulf_nmr.core.errors import english_error
from zulf_nmr.core.spectrometer import Spectrometer
from zulf_nmr.experiment.parser import (
    ConfigError,
    ExperimentConfig,
    HardwareLimits,
    load_experiment_config,
)
from zulf_nmr.experiment.runner import ExperimentAborted, ExperimentRunner, average, save_csv
from zulf_nmr.experiment.sequence import acquisition_window_ms
from zulf_nmr.experiment.simulation import SIM_CHUNKS, SIM_TRANSIENT_S, choose_waves, simulate_transient
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
            self.errors = [f"cannot read the file: {english_error(e)}"]
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
    number: int  # nnn in the file name: 1 .. output.average_count
    set_index: int  # position of its set in the queue, 1-based
    file: Path
    data: np.ndarray


@dataclass
class ResultGroup:
    """One set of the queue: its copied configuration, its transients and their average."""

    name: str  # experiment_name, plus _<occurrence> if it is in the queue more than once
    set_index: int  # position in the queue, 1-based
    config_source: Path
    sample_rate_hz: float
    channels: tuple[int, ...]  # acquisition.channels - the data columns, in this order
    file_format: str
    expected_transients: int  # output.average_count
    acquisition_start_s: float  # start of the recording, seconds after the sequence start (:SEQ:RUN)
    config_file: Path
    awg_file: Path | None = None
    transients: list[Transient] = field(default_factory=list)
    average_file: Path | None = None
    average: np.ndarray | None = None


@dataclass
class QueueResult:
    """Where the results of one queue run were saved."""

    folder: Path
    groups: list[ResultGroup] = field(default_factory=list)  # one per set, in queue order

    @property
    def files(self) -> list[Path]:
        """Every saved file, set by set: config, AWG, transients, average."""
        files: list[Path] = []
        for group in self.groups:
            files += [group.config_file, *([group.awg_file] if group.awg_file else [])]
            files += [t.file for t in group.transients]
            files += [group.average_file] if group.average_file else []
        return files


class QueueRunner:
    """Runs the sets in order; after each set (except the last) waits its delay.

    Results go to <results_root>/<stamp>/ where stamp = yyyymmdd_hhmmss of the start.
    Every set runs output.average_count transients and gets its own files:
        <stamp>_<name>_config.json                - copy of the configuration file
        <stamp>_<name>_awg.npz                    - copy of its AWG sequence (if any)
        <stamp>_<name>_transient_<nnn>.<format>   - nnn = 001 .. average_count
        <stamp>_<name>_average.<format>           - mean of this set's transients
    name is the experiment_name. If several sets have the same experiment_name, each
    gets _<occ> appended (_001 for the first, _002 for the second...), e.g.
    <stamp>_ZULF_Thermal_Example_002_transient_001.csv. Every transient is saved as
    soon as it is complete and the average rewritten, so an aborted queue keeps them.

    simulate=True produces simulated data instead of talking to the device
    (spectrometer may then be None).
    on_status(text) reports what is happening (set started, waiting...);
    on_progress(set_index, transient, received, expected) forwards runner progress;
    on_live(group, transient_number, data_so_far) shows a transient while it is being
    acquired (a few times per second); on_result(group, transient) is called after every
    saved transient (group.average is already updated). All callbacks run in the thread
    that called run().
    """

    def __init__(
        self,
        spectrometer: Spectrometer | None,
        items: list[QueueItem],
        results_root: str | Path,
        simulate: bool = False,
        on_status: Callable[[str], None] | None = None,
        on_progress: Callable[[int, int, int, int], None] | None = None,
        on_result: Callable[[ResultGroup, Transient], None] | None = None,
        on_live: Callable[[ResultGroup, int, np.ndarray], None] | None = None,
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
        self.on_result = on_result
        self.on_live = on_live
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
        totals = Counter(item.config.experiment_name for item in self.items)
        occurrences: Counter[str] = Counter()

        n = len(self.items)
        for index, item in enumerate(self.items, start=1):
            if self._abort.is_set():
                raise ExperimentAborted("Queue aborted by user")
            exp_name = item.config.experiment_name
            occurrences[exp_name] += 1
            name = exp_name if totals[exp_name] == 1 else f"{exp_name}_{occurrences[exp_name]:03d}"
            group = self._new_group(item, index, name, folder, stamp)
            result.groups.append(group)

            mode = " (simulated)" if self.simulate else ""
            self._status(f"Set {index}/{n}: {group.name} running, "
                         f"{group.expected_transients} transient(s){mode}")
            self._acquire(item, group, lambda number, data: self._save_transient(group, number, data, stamp))

            if index < n and item.delay_s > 0:
                self._status(f"Waiting {format_delay(item.delay_s)} before set {index + 1}/{n}")
                if self._abort.wait(item.delay_s):
                    raise ExperimentAborted("Queue aborted by user")
        return result

    @staticmethod
    def _new_group(item: QueueItem, index: int, name: str, folder: Path, stamp: str) -> ResultGroup:
        """Start the files of one set: copy its configuration file (and AWG sequence)."""
        cfg = item.config
        window = acquisition_window_ms(cfg)
        config_file = folder / f"{stamp}_{name}_config.json"
        shutil.copyfile(item.path, config_file)
        awg_file = None
        if item.awg is not None and item.awg.source_path is not None:
            awg_file = folder / f"{stamp}_{name}_awg{item.awg.source_path.suffix}"
            shutil.copyfile(item.awg.source_path, awg_file)
        return ResultGroup(
            name=name,
            set_index=index,
            config_source=item.path,
            sample_rate_hz=cfg.hardware_parameters.acq_sample_rate_hz,
            channels=cfg.acquisition.channels,
            file_format=cfg.output.format,
            expected_transients=cfg.output.average_count,
            acquisition_start_s=window[0] / 1000 if window else 0.0,
            config_file=config_file,
            awg_file=awg_file,
        )

    def _save_transient(self, group: ResultGroup, number: int, data: np.ndarray, stamp: str) -> None:
        folder = group.config_file.parent
        name = f"{stamp}_{group.name}_transient_{number:03d}.{group.file_format}"
        path = save_csv(folder / name, data, group.sample_rate_hz, group.channels)
        transient = Transient(number, group.set_index, path, data)
        group.transients.append(transient)
        group.average = average([t.data for t in group.transients])
        group.average_file = save_csv(
            folder / f"{stamp}_{group.name}_average.{group.file_format}", group.average, group.sample_rate_hz,
            group.channels,
        )
        if self.on_result:
            self.on_result(group, transient)

    def _acquire(self, item: QueueItem, group: ResultGroup,
                 on_transient: Callable[[int, np.ndarray], None]) -> None:
        """Run the output.average_count transients of one set, passing each to on_transient."""
        index = group.set_index

        def progress(number: int, received: int, expected: int) -> None:
            if self.on_progress:
                self.on_progress(index, number, received, expected)

        def live(number: int, data: np.ndarray) -> None:
            if self.on_live:
                self.on_live(group, number, data)

        if self.simulate:
            waves = choose_waves(item.config)  # sin or cos per channel, the same for the whole set
            for number in range(1, item.config.output.average_count + 1):
                on_transient(number, self._simulate(item, number, waves, progress, live))
            return
        self._current = ExperimentRunner(
            self.spec,
            item.config,
            awg_waveform=item.awg,
            auto_save=False,  # saved here, with the queue's names
            on_progress=progress,
            on_transient=on_transient,
            on_data=live,
        )
        try:
            self._current.run()
        finally:
            self._current = None

    def _simulate(self, item: QueueItem, number: int, waves: dict[int, str], progress, live) -> np.ndarray:
        """Simulated transient, delivered in parts over SIM_TRANSIENT_S like a real acquisition."""
        data = simulate_transient(item.config, waves)
        n = len(data)
        for part in range(1, SIM_CHUNKS + 1):
            if self._abort.wait(SIM_TRANSIENT_S / SIM_CHUNKS):
                raise ExperimentAborted("Queue aborted by user")
            received = n * part // SIM_CHUNKS
            progress(number, received, n)
            if part < SIM_CHUNKS:
                live(number, data[:received])
        return data

    def _status(self, text: str) -> None:
        log.info(text)
        if self.on_status:
            self.on_status(text)
