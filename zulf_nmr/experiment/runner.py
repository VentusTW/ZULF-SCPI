"""Main execution loop and data retrieval from the buffer."""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np

from zulf_nmr.core.spectrometer import Spectrometer
from zulf_nmr.experiment.parser import ExperimentConfig
from zulf_nmr.experiment.sequence import CompiledSequence, compile_experiment
from zulf_nmr.hardware.awg import AwgWaveform

log = logging.getLogger(__name__)


def decode_int16(chunk: bytes) -> np.ndarray:
    """ASSUMED chunk format: one ADC channel, little-endian int16 samples, no checksum."""
    return np.frombuffer(chunk, dtype="<i2").astype(float)


class ExperimentAborted(RuntimeError):
    pass


@dataclass
class RunResult:
    transients: list[np.ndarray] = field(default_factory=list)
    average: np.ndarray | None = None
    files: list[Path] = field(default_factory=list)


class ExperimentRunner:
    """Uploads the compiled sequence, starts it and collects the ADC data.

    One run = output.average_count transients. Every transient is saved as it
    arrives and the average file is rewritten after each one. An AWG waveform,
    if given, is uploaded once before the first transient.
    """

    def __init__(
        self,
        spectrometer: Spectrometer,
        config: ExperimentConfig,
        output_dir: str | Path | None = None,
        decode: Callable[[bytes], np.ndarray] = decode_int16,
        is_valid_chunk: Callable[[bytes], bool] = lambda chunk: True,
        poll_s: float = 0.05,
        timeout_margin_s: float = 30.0,
        on_progress: Callable[[int, int, int], None] | None = None,
        awg_waveform: AwgWaveform | None = None,
        auto_save: bool | None = None,
    ) -> None:
        """on_progress(transient, points_received, points_expected) is called after every chunk."""
        self.spec = spectrometer
        self.config = config
        self.output_dir = Path(output_dir) if output_dir is not None else spectrometer.data_output
        self.decode = decode
        self.is_valid_chunk = is_valid_chunk
        self.poll_s = poll_s
        self.timeout_margin_s = timeout_margin_s
        self.on_progress = on_progress
        self.awg_waveform = awg_waveform
        # None = output.auto_save from the experiment file; False when the caller saves the data.
        self.auto_save = config.output.auto_save if auto_save is None else auto_save
        self.sequence: CompiledSequence = compile_experiment(config)
        self._abort = threading.Event()

    # ---------------------------------------------------------------- public

    @property
    def expected_points(self) -> int:
        hw = self.config.hardware_parameters
        return int(round(hw.acq_sample_rate_hz * hw.acq_duration_ms / 1000))

    def abort(self) -> None:
        """Request a stop; safe to call from another thread (e.g. a GUI button)."""
        self._abort.set()

    def run(self) -> RunResult:
        """Run all transients and return the data; on error or abort send *RST."""
        self._abort.clear()
        result = RunResult()
        prefix = self._file_prefix()
        if self.auto_save:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            result.files.append(self._save_config(prefix))

        n = self.config.output.average_count
        try:
            if self.awg_waveform is not None:
                log.info("Uploading AWG sequence (%d samples)", self.awg_waveform.n_samples)
                self.spec.awg.upload(self.awg_waveform)
            for i in range(1, n + 1):
                log.info("Transient %d/%d", i, n)
                data = self._run_transient(i)
                result.transients.append(data)
                result.average = average(result.transients)
                if self.auto_save:
                    result.files.append(self._save_csv(f"{prefix}_transient_{i:03d}.csv", data))
                    avg_path = self._save_csv(f"{prefix}_average.csv", result.average)
                    if avg_path not in result.files:
                        result.files.append(avg_path)
        except BaseException:
            # Abort, error or Ctrl+C: stop everything on the device.
            try:
                self.spec.system.reset()
            except Exception:
                log.exception("Could not send *RST after failure")
            raise
        return result

    # ----------------------------------------------------------- one transient

    def _run_transient(self, index: int) -> np.ndarray:
        for command in (*self.sequence.setup, *self.sequence.events):
            self.spec.connection.write(command)
        self.spec.system.clear_status()
        self.spec.connection.write(self.sequence.start)

        expected = self.expected_points
        deadline = time.monotonic() + self._max_duration_s()
        blocks: list[np.ndarray] = []
        received = 0
        done = False

        while received < expected:
            self._check_abort()
            if time.monotonic() > deadline:
                raise TimeoutError(
                    f"Transient {index}: got {received}/{expected} points before the timeout"
                )
            chunk = self.spec.acquisition.fetch_chunk(self.is_valid_chunk)
            if chunk:
                samples = self.decode(chunk)
                blocks.append(samples)
                received += len(samples)
                if self.on_progress:
                    self.on_progress(index, received, expected)
                continue
            # Buffer empty: finished, or the device has not produced more yet.
            if done:
                log.warning("Transient %d: sequence done with %d/%d points", index, received, expected)
                break
            done = self.spec.system.is_operation_complete()
            if not done:
                time.sleep(self.poll_s)

        data = np.concatenate(blocks) if blocks else np.empty(0)
        return data[:expected]

    def _max_duration_s(self) -> float:
        last_event_ms = max((e.time_ms for e in self.config.events), default=0)
        return (last_event_ms + self.config.hardware_parameters.acq_duration_ms) / 1000 + self.timeout_margin_s

    def _check_abort(self) -> None:
        if self._abort.is_set():
            raise ExperimentAborted("Experiment aborted by user")

    # ------------------------------------------------------------------ saving

    def _file_prefix(self) -> str:
        return f"{datetime.now():%Y%m%d_%H%M%S}_{self.config.experiment_name}"

    def _save_config(self, prefix: str) -> Path:
        path = self.output_dir / f"{prefix}_config.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.config.raw, f, indent=2, ensure_ascii=False)
        return path

    def _save_csv(self, name: str, data: np.ndarray) -> Path:
        return save_csv(self.output_dir / name, data, self.config.hardware_parameters.acq_sample_rate_hz)


def save_csv(path: Path, data: np.ndarray, sample_rate_hz: float) -> Path:
    """Save samples with a time column: time_s,signal (or ch1, ch2... for 2-D data)."""
    t = np.arange(len(data)) / sample_rate_hz
    columns = data.reshape(len(data), -1)
    signal_names = ["signal"] if columns.shape[1] == 1 else [f"ch{i + 1}" for i in range(columns.shape[1])]
    np.savetxt(
        path,
        np.column_stack([t, columns]),
        delimiter=",",
        header=",".join(["time_s", *signal_names]),
        comments="",
        fmt="%.9g",
    )
    return path


def average(transients: list[np.ndarray]) -> np.ndarray:
    n = min(len(t) for t in transients)
    return np.mean([t[:n] for t in transients], axis=0)
