"""Reading back the results folder of an earlier queue run (see QueueRunner for the file names)."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from zulf_nmr.experiment.parser import ConfigError, HardwareLimits, parse_experiment
from zulf_nmr.experiment.schedule import QueueResult, ResultGroup, Transient
from zulf_nmr.experiment.sequence import acquisition_window_ms

# <stamp>_<name>_transient_<nnn>.<ext> / _average.<ext> / _config.json / _awg.npz
_FILE = re.compile(r"^(?P<stamp>\d{8}_\d{6})_(?P<name>.+?)_"
                   r"(?:transient_(?P<number>\d{3})|(?P<kind>average|config|awg))\.(?P<ext>\w+)$")
_CHANNEL = re.compile(r"^ch(\d+)$")


class HistoryError(ValueError):
    """The folder does not hold the results of a queue run."""


def load_results(folder: str | Path) -> tuple[QueueResult, list[str]]:
    """Read every set of a results folder; return the result and warnings about skipped files.

    Sets are ordered by the time their configuration was saved (the start of the set).
    The average is read from its file; if it is missing it is computed from the transients.
    """
    folder = Path(folder)
    if not folder.is_dir():
        raise HistoryError(f"{folder} is not a folder")
    sets: dict[str, dict] = {}
    warnings: list[str] = []
    for path in sorted(folder.iterdir()):
        match = _FILE.match(path.name) if path.is_file() else None
        if match is None:
            if path.is_file():
                warnings.append(f"{path.name}: not a result file - skipped")
            continue
        entry = sets.setdefault(match["name"], {"transients": {}, "stamp": match["stamp"]})
        if match["number"]:
            entry["transients"][int(match["number"])] = path
        else:
            entry[match["kind"]] = path

    sets = {name: e for name, e in sets.items() if e["transients"]}
    if not sets:
        subfolders = [p.name for p in folder.iterdir() if p.is_dir()]
        hint = f" - choose one of its run folders, e.g. {subfolders[-1]}" if subfolders else ""
        raise HistoryError(f"No transient files in {folder}{hint}")

    def started(item: tuple[str, dict]) -> float:
        path = item[1].get("config") or min(item[1]["transients"].values())
        return path.stat().st_mtime

    result = QueueResult(folder)
    for index, (name, entry) in enumerate(sorted(sets.items(), key=started), start=1):
        try:
            result.groups.append(_load_group(index, name, entry, warnings))
        except (OSError, ValueError) as e:
            warnings.append(f"{name}: {e} - set skipped")
    if not result.groups:
        raise HistoryError("No set could be read:\n" + "\n".join(warnings))
    return result, warnings


def _load_group(index: int, name: str, entry: dict, warnings: list[str]) -> ResultGroup:
    transients = []
    channels: tuple[int, ...] | None = None
    rate = None
    for number, path in sorted(entry["transients"].items()):
        data, file_channels, file_rate = _read_csv(path)
        if channels is not None and file_channels != channels:
            warnings.append(f"{path.name}: channels {file_channels} differ from {channels} - skipped")
            continue
        channels, rate = file_channels, rate or file_rate
        transients.append(Transient(number, index, path, data))

    expected = len(transients)
    start_s = 0.0
    config_file = entry.get("config")
    if config_file is not None:
        try:
            config = _read_config(config_file)
            rate = config.hardware_parameters.acq_sample_rate_hz
            expected = config.output.average_count
            window = acquisition_window_ms(config)
            start_s = window[0] / 1000 if window else 0.0
        except (OSError, ValueError) as e:
            warnings.append(f"{config_file.name}: {e} - time axis starts at 0")
    else:
        warnings.append(f"{name}: no _config.json - time axis starts at 0")

    average_file = entry.get("average")
    average = None
    if average_file is not None:
        average, avg_channels, _ = _read_csv(average_file)
        if avg_channels != channels:
            warnings.append(f"{average_file.name}: channels differ from the transients - recomputed")
            average = None
    if average is None:
        n = min(len(t.data) for t in transients)
        average = np.mean([t.data[:n] for t in transients], axis=0)

    return ResultGroup(
        name=name,
        set_index=index,
        config_source=config_file or transients[0].file,
        sample_rate_hz=rate,
        channels=channels,
        file_format=transients[0].file.suffix.lstrip("."),
        expected_transients=expected,
        acquisition_start_s=start_s,
        config_file=config_file or transients[0].file,
        awg_file=entry.get("awg"),
        transients=transients,
        average_file=average_file or transients[-1].file,
        average=average,
    )


def _read_csv(path: Path) -> tuple[np.ndarray, tuple[int, ...], float]:
    """(data of shape (N, channels), channel numbers, sample rate from the time column)."""
    with open(path, encoding="utf-8") as f:
        header = f.readline().strip().split(",")
    if not header or header[0] != "time_s":
        raise ValueError(f"{path.name}: expected a time_s column first, got {header[:1]}")
    channels = []
    for column in header[1:]:
        match = _CHANNEL.match(column)
        channels.append(int(match[1]) if match else len(channels) + 1)  # e.g. old "signal" column
    table = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
    if table.shape[1] != len(header):
        raise ValueError(f"{path.name}: {table.shape[1]} columns but {len(header)} names")
    t = table[:, 0]
    rate = 1 / float(np.median(np.diff(t))) if len(t) > 1 else 1.0
    return table[:, 1:], tuple(channels), rate


def _read_config(path: Path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    try:
        # The copied AWG file is not where the config points to; it is not needed here.
        return parse_experiment(data, HardwareLimits(), base_dir=path.parent, check_files=False)
    except ConfigError as e:
        raise ValueError("; ".join(e.errors[:3])) from None
