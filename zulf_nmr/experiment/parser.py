"""Loading and validation of the JSON file."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from zulf_nmr.core.connection import DEFAULT_SYSTEM_CONFIG
from zulf_nmr.core.scpi import header_matches
from zulf_nmr.hardware.digital import Digital
from zulf_nmr.hardware.motor import MotorPosition
from zulf_nmr.hardware.sequencer import ALLOWED_EVENTS
from zulf_nmr.hardware.shim import Shim

SUPPORTED_OUTPUT_FORMATS = ("csv",)
SUPPORTED_AWG_TYPES = ("npz",)
_FORBIDDEN_NAME_CHARS = set('<>:"/\\|?*')


class ConfigError(ValueError):
    """Raised with every problem found in a configuration file."""

    def __init__(self, path: Path | None, errors: list[str]) -> None:
        self.path = path
        self.errors = errors
        where = f" in {path}" if path else ""
        super().__init__(f"{len(errors)} error(s){where}:\n  - " + "\n  - ".join(errors))


# --------------------------------------------------------------- dataclasses


@dataclass(frozen=True)
class HardwareLimits:
    max_shim_current_mA: float = 100.0
    max_awg_amplitude_A: float = 1.0
    max_awg_sample_rate_hz: float = 10000
    motor_timeout_s: float = 15.0


@dataclass(frozen=True)
class ConnectionSettings:
    port: str
    baudrate: int = 115200
    timeout_s: float = 2.0


@dataclass(frozen=True)
class SystemConfig:
    connection: ConnectionSettings
    hardware_limits: HardwareLimits
    data_output: Path
    source_path: Path | None = None


@dataclass(frozen=True)
class OutputSettings:
    average_count: int = 1
    format: str = "csv"
    auto_save: bool = True


@dataclass(frozen=True)
class HardwareParameters:
    acq_sample_rate_hz: float
    acq_duration_ms: int
    acq_start_awg_time: float  # unit not defined yet (:ACQuire:AUTostart uses us)


@dataclass(frozen=True)
class Event:
    time_ms: int
    action: str


@dataclass(frozen=True)
class AwgSequence:
    type: str
    file_path: Path  # resolved against the config file's directory


@dataclass(frozen=True)
class ExperimentConfig:
    experiment_name: str
    metadata: dict[str, Any]
    output: OutputSettings
    hardware_parameters: HardwareParameters
    events: tuple[Event, ...]  # sorted by time_ms (stable)
    awg_sequence: AwgSequence | None
    source_path: Path | None = None
    raw: dict[str, Any] = field(default_factory=dict, repr=False)  # saved alongside data


# ----------------------------------------------------------------- loaders


def load_system_config(path: str | Path = DEFAULT_SYSTEM_CONFIG) -> SystemConfig:
    """Load and validate system.json; raise ConfigError listing every problem."""
    path = Path(path)
    v = _Validator()
    data = _read_json(path)

    v.check_keys(data, "", required={"connection"}, optional={"hardware_limits", "directories"})

    conn = v.section(data, "connection")
    v.check_keys(conn, "connection", required={"port"}, optional={"baudrate", "timeout_s"})
    connection = ConnectionSettings(
        port=v.string(conn, "port", "connection"),
        baudrate=v.integer(conn, "baudrate", "connection", default=115200, minimum=1),
        timeout_s=v.number(conn, "timeout_s", "connection", default=2.0, positive=True),
    )

    lim = v.section(data, "hardware_limits")
    defaults = HardwareLimits()
    v.check_keys(lim, "hardware_limits", optional=set(HardwareLimits.__dataclass_fields__))
    limits = HardwareLimits(
        **{
            name: v.number(lim, name, "hardware_limits", default=getattr(defaults, name), positive=True)
            for name in HardwareLimits.__dataclass_fields__
        }
    )

    dirs = v.section(data, "directories")
    v.check_keys(dirs, "directories", optional={"data_output"})
    data_output = v.string(dirs, "data_output", "directories", default="data/")

    v.raise_if_errors(path)
    return SystemConfig(connection, limits, Path(data_output), source_path=path)


def load_experiment_config(
    path: str | Path,
    limits: HardwareLimits | None = None,
    check_files: bool = True,
) -> ExperimentConfig:
    """Load and validate an experiment file (e.g. config_example.json).

    limits defaults to the hardware_limits of the default system.json.
    check_files=False skips the check that awg_sequence.file_path exists.
    Raises ConfigError listing every problem found.
    """
    path = Path(path)
    if limits is None:
        limits = load_system_config().hardware_limits
    data = _read_json(path)
    return parse_experiment(data, limits, base_dir=path.parent, check_files=check_files, source_path=path)


def parse_experiment(
    data: dict[str, Any],
    limits: HardwareLimits,
    base_dir: Path = Path("."),
    check_files: bool = True,
    source_path: Path | None = None,
) -> ExperimentConfig:
    """Validate an already loaded experiment dictionary."""
    v = _Validator()
    v.check_keys(
        data,
        "",
        required={"experiment_name", "hardware_parameters", "events"},
        optional={"metadata", "output", "awg_sequence"},
    )

    name = v.string(data, "experiment_name", "")
    if name and (_FORBIDDEN_NAME_CHARS & set(name)):
        v.error("experiment_name", f"must not contain any of {''.join(sorted(_FORBIDDEN_NAME_CHARS))} (used in file names)")

    metadata = v.section(data, "metadata")

    out = v.section(data, "output")
    v.check_keys(out, "output", optional={"average_count", "format", "auto_save"})
    output = OutputSettings(
        average_count=v.integer(out, "average_count", "output", default=1, minimum=1),
        format=v.choice(out, "format", "output", SUPPORTED_OUTPUT_FORMATS, default="csv"),
        auto_save=v.boolean(out, "auto_save", "output", default=True),
    )

    hw = v.section(data, "hardware_parameters")
    v.check_keys(
        hw,
        "hardware_parameters",
        required={"acq_sample_rate_hz", "acq_duration_ms", "acq_start_awg_time"},
    )
    hardware = HardwareParameters(
        acq_sample_rate_hz=v.number(hw, "acq_sample_rate_hz", "hardware_parameters", positive=True),
        acq_duration_ms=v.integer(hw, "acq_duration_ms", "hardware_parameters", minimum=1),
        acq_start_awg_time=v.number(hw, "acq_start_awg_time", "hardware_parameters", minimum=0),
    )

    events = []
    raw_events = data.get("events")
    if "events" in data and not isinstance(raw_events, list):
        v.error("events", "must be a list")
    for i, ev in enumerate(raw_events if isinstance(raw_events, list) else []):
        where = f"events[{i}]"
        if not isinstance(ev, dict):
            v.error(where, "must be an object with time_ms and action")
            continue
        v.check_keys(ev, where, required={"time_ms", "action"})
        time_ms = v.integer(ev, "time_ms", where, minimum=0)
        action = v.string(ev, "action", where)
        if action:
            for problem in validate_action(action, limits):
                v.error(f"{where}.action", f"{action!r}: {problem}")
        events.append(Event(time_ms, action))
    events.sort(key=lambda e: e.time_ms)

    awg_sequence = None
    awg = data.get("awg_sequence")
    if awg is not None:
        awg = v.section(data, "awg_sequence")
        v.check_keys(awg, "awg_sequence", required={"type", "file_path"})
        awg_type = v.choice(awg, "type", "awg_sequence", SUPPORTED_AWG_TYPES)
        file_path = v.string(awg, "file_path", "awg_sequence")
        if file_path:
            resolved = (base_dir / file_path).resolve()
            if check_files and not resolved.is_file():
                v.error("awg_sequence.file_path", f"file not found: {resolved}")
            awg_sequence = AwgSequence(awg_type, resolved)

    v.raise_if_errors(source_path)
    return ExperimentConfig(
        experiment_name=name,
        metadata=dict(metadata),
        output=output,
        hardware_parameters=hardware,
        events=tuple(events),
        awg_sequence=awg_sequence,
        source_path=source_path,
        raw=data,
    )


# -------------------------------------------------------- event validation


def validate_action(action: str, limits: HardwareLimits) -> list[str]:
    """Return the problems of one sequence event command (empty list = valid)."""
    header, _, arg_text = action.strip().partition(" ")
    if not header.startswith(":"):
        header = ":" + header
    args = [a.strip() for a in arg_text.split(",")] if arg_text.strip() else []

    if header_matches(header, ":MOTor:MOVE"):
        if len(args) != 1:
            return ["expects 1 argument: <pos>"]
        targets = [p.value for p in MotorPosition if p is not MotorPosition.OTHER]
        if args[0].upper() not in targets:
            return [f"position must be one of {', '.join(targets)}"]
        return []

    if header_matches(header, ":SHIM:CURRent"):
        if len(args) != 2:
            return ["expects 2 arguments: <ch>, <mA>"]
        problems = _check_channel(args[0], Shim.CHANNELS)
        current = _to_float(args[1])
        if current is None:
            problems.append(f"current {args[1]!r} is not a number")
        elif abs(current) > limits.max_shim_current_mA:
            problems.append(f"current {current} mA exceeds +/-{limits.max_shim_current_mA} mA")
        return problems

    if header_matches(header, ":DIGital:STATe"):
        if len(args) != 2:
            return ["expects 2 arguments: <ch>, <state>"]
        problems = _check_channel(args[0], Digital.CHANNELS)
        if args[1] not in ("0", "1"):
            problems.append(f"state must be 0 or 1, got {args[1]!r}")
        return problems

    for no_arg in (":AWG:RUN", ":AWG:RST", ":ACQuire:RUN"):
        if header_matches(header, no_arg):
            return [] if not args else ["takes no arguments"]

    return [f"not an allowed sequence event; allowed: {', '.join(ALLOWED_EVENTS)}"]


def _check_channel(text: str, channels: tuple[int, ...]) -> list[str]:
    try:
        if int(text) in channels:
            return []
    except ValueError:
        pass
    return [f"channel must be one of {channels}, got {text!r}"]


def _to_float(text: str) -> float | None:
    try:
        value = float(text)
    except ValueError:
        return None
    return value if math.isfinite(value) else None


# --------------------------------------------------------------- helpers


def _read_json(path: Path) -> dict[str, Any]:
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        raise ConfigError(path, ["file not found"]) from None
    except json.JSONDecodeError as e:
        raise ConfigError(path, [f"invalid JSON at line {e.lineno}, column {e.colno}: {e.msg}"]) from None
    if not isinstance(data, dict):
        raise ConfigError(path, ["top level must be a JSON object"])
    return data


_MISSING = object()


class _Validator:
    """Collects errors instead of stopping at the first one."""

    def __init__(self) -> None:
        self.errors: list[str] = []

    def error(self, where: str, message: str) -> None:
        self.errors.append(f"{where}: {message}" if where else message)

    def raise_if_errors(self, path: Path | None) -> None:
        if self.errors:
            raise ConfigError(path, self.errors)

    @staticmethod
    def _join(parent: str, key: str) -> str:
        return f"{parent}.{key}" if parent else key

    def check_keys(self, d: dict, where: str, required: set[str] = frozenset(), optional: set[str] = frozenset()) -> None:
        for key in sorted(required - d.keys()):
            self.error(self._join(where, key), "missing")
        for key in sorted(d.keys() - required - optional):
            self.error(self._join(where, key), "unknown key")

    def section(self, d: dict, key: str) -> dict:
        value = d.get(key, _MISSING)
        if value is _MISSING:
            return {}  # "missing" is reported by check_keys
        if not isinstance(value, dict):
            self.error(key, "must be an object")
            return {}
        return value

    def number(self, d, key, where, default=_MISSING, positive=False, minimum=None) -> float:
        value = d.get(key, default)
        if value is _MISSING:
            return math.nan
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            self.error(self._join(where, key), f"must be a number, got {value!r}")
            return math.nan
        if positive and value <= 0:
            self.error(self._join(where, key), f"must be > 0, got {value}")
        if minimum is not None and value < minimum:
            self.error(self._join(where, key), f"must be >= {minimum}, got {value}")
        return float(value)

    def integer(self, d, key, where, default=_MISSING, minimum=None) -> int:
        value = d.get(key, default)
        if value is _MISSING:
            return 0
        if isinstance(value, float) and value.is_integer():
            value = int(value)
        if isinstance(value, bool) or not isinstance(value, int):
            self.error(self._join(where, key), f"must be an integer, got {value!r}")
            return 0
        if minimum is not None and value < minimum:
            self.error(self._join(where, key), f"must be >= {minimum}, got {value}")
        return value

    def string(self, d, key, where, default=_MISSING) -> str:
        value = d.get(key, default)
        if value is _MISSING:
            return ""
        if not isinstance(value, str) or not value.strip():
            self.error(self._join(where, key), f"must be a non-empty string, got {value!r}")
            return ""
        return value

    def boolean(self, d, key, where, default=_MISSING) -> bool:
        value = d.get(key, default)
        if value is _MISSING:
            return False
        if not isinstance(value, bool):
            self.error(self._join(where, key), f"must be true or false, got {value!r}")
            return False
        return value

    def choice(self, d, key, where, choices, default=_MISSING) -> str:
        value = self.string(d, key, where, default)
        if value and value not in choices:
            self.error(self._join(where, key), f"must be one of {', '.join(choices)}, got {value!r}")
        return value
