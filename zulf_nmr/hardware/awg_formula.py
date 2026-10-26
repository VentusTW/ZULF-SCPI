"""AWG waveforms from formulas: a small, safe expression language and the .npz writer.

A formula gives the current of one channel in mA as a function of the time t in
seconds after :AWG:RUN, e.g. "200 * pulse(5*ms, 15*ms)" or "5 * sin(2*pi*300*t)".
data(n) is channel n of an opened .npz file (SourceData), so files without
formulas can be edited too, e.g. "0.5 * data(1) + 3 * sin(2*pi*50*t)".
Only numbers, the names in NAMES, arithmetic, comparisons and calls of the
functions in NAMES are accepted - the formula is checked before it is evaluated,
so it cannot run any other Python code.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from zulf_nmr.hardware.awg import (
    N_CHANNELS,
    RANGE_FULL_SCALE_A,
    AwgRange,
    AwgWaveform,
    load_awg_npz,
)

MAX_SAMPLES = 1_000_000


# ------------------------------------------------------------------ shapes


def _shapes(t: np.ndarray) -> dict[str, object]:
    """Waveform building blocks; times in seconds, frequencies in Hz."""

    def step(t0):
        """1 from t0 on, 0 before."""
        return (t >= t0).astype(float)

    def pulse(t0, t1):
        """1 from t0 to t1, 0 elsewhere."""
        return ((t >= t0) & (t < t1)).astype(float)

    def ramp(t0, t1):
        """0 before t0, rising linearly to 1 at t1, 1 after."""
        return np.clip((t - t0) / (t1 - t0), 0.0, 1.0)

    def gauss(t0, sigma):
        """Gaussian bell centred at t0, height 1."""
        return np.exp(-0.5 * ((t - t0) / sigma) ** 2)

    def hann(t0, t1):
        """Smooth window: 0 at t0, 1 in the middle, 0 at t1, 0 outside."""
        x = np.clip((t - t0) / (t1 - t0), 0.0, 1.0)
        return np.where((t >= t0) & (t < t1), np.sin(np.pi * x) ** 2, 0.0)

    def chirp(f0, f1, t0, t1):
        """Sine sweeping linearly from f0 to f1 between t0 and t1, 0 outside."""
        tau = t - t0
        phase = 2 * np.pi * (f0 * tau + 0.5 * (f1 - f0) / (t1 - t0) * tau**2)
        return np.where((t >= t0) & (t < t1), np.sin(phase), 0.0)

    return {"step": step, "pulse": pulse, "ramp": ramp, "gauss": gauss, "hann": hann, "chirp": chirp}


MATH = {
    "sin": np.sin, "cos": np.cos, "tan": np.tan,
    "asin": np.arcsin, "acos": np.arccos, "atan": np.arctan,
    "sinh": np.sinh, "cosh": np.cosh, "tanh": np.tanh,
    "exp": np.exp, "log": np.log, "log10": np.log10, "sqrt": np.sqrt,
    "abs": np.abs, "sign": np.sign, "floor": np.floor, "ceil": np.ceil, "round": np.round,
    "min": np.minimum, "max": np.maximum,
}
# Units: times in s, frequencies in Hz, currents in mA (the unit of a formula's result).
UNITS = {
    "s": 1.0, "sec": 1.0, "ms": 1e-3, "us": 1e-6, "ns": 1e-9,
    "Hz": 1.0, "kHz": 1e3, "MHz": 1e6,
    "mA": 1.0, "uA": 1e-3, "A": 1e3,
}
CONSTANTS = {"pi": np.pi, "e": np.e, **UNITS}

# A unit right after a number ("1.6ms", "2 s", "5kHz") is multiplied with it: "(1.6*ms)".
_NUMBER_UNIT = re.compile(
    r"(?<![\w.])((?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)\s*("
    + "|".join(sorted(UNITS, key=len, reverse=True)) + r")(?![\w.])"
)


def normalise(text: str) -> str:
    """The formula as Python: ^ is a power, a unit after a number multiplies it, µs is us."""
    text = text.replace("µ", "u").replace("μ", "u").replace("^", "**")
    # In brackets, so the unit stays with its number: t/15ms is t/(15*ms), not (t/15)*ms.
    return _NUMBER_UNIT.sub(lambda m: f"({m.group(1)}*{m.group(2)})", text)


SHAPES = ("step", "pulse", "ramp", "gauss", "hann", "chirp")
# Number of arguments of every function (the math ones take 1 unless listed).
ARITY = {"min": 2, "max": 2, "step": 1, "pulse": 2, "ramp": 2, "gauss": 2, "hann": 2, "chirp": 4, "data": 1}
NAMES = ("t", *CONSTANTS, *MATH, *SHAPES, "data")


@dataclass(frozen=True)
class SourceData:
    """Samples of an opened .npz file, used by data(n) in formulas."""

    currents_mA: np.ndarray  # (N, 4)
    sample_rate_hz: float
    name: str = ""

    def channel(self, n, t: np.ndarray) -> np.ndarray:
        """Channel n (1-4) at the times t: the sample in force at each time (the AWG
        holds every value until the next sample), 0 after the end of the file."""
        if not (isinstance(n, (int, float)) and float(n).is_integer() and 1 <= n <= N_CHANNELS):
            raise FormulaError(f"data() takes a channel number 1-{N_CHANNELS}, got {n!r}")
        index = np.floor(t * self.sample_rate_hz + 1e-9).astype(int)
        inside = (index >= 0) & (index < len(self.currents_mA))
        values = np.zeros_like(t)
        values[inside] = self.currents_mA[index[inside], int(n) - 1]
        return values

_ALLOWED_NODES = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Compare, ast.Call, ast.Name, ast.Load, ast.Constant,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod, ast.FloorDiv, ast.USub, ast.UAdd,
    ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.Eq, ast.NotEq,
)


class FormulaError(ValueError):
    pass


def check_formula(text: str) -> ast.Expression:
    """Parse a formula and make sure it uses only allowed names and operations."""
    if len(text) > 2000:
        raise FormulaError("formula is too long")
    try:
        tree = ast.parse(normalise(text), mode="eval")
    except SyntaxError as e:
        # Usually a formula still being typed. No character position: it would count in the
        # rewritten text (5ms -> (5*ms)), not in what the user sees.
        missing = text.count("(") - text.count(")")
        if missing > 0:
            raise FormulaError(f"missing {')' * missing}") from None
        if missing < 0:
            raise FormulaError(f"{-missing} ')' too many") from None
        end = text.rstrip()[-1:]
        if end and end in "+-*/^%,.":
            raise FormulaError(f"unfinished - ends with '{end}'") from None
        raise FormulaError("syntax error") from None
    called = {id(node.func) for node in ast.walk(tree) if isinstance(node, ast.Call)}
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            raise FormulaError(f"'{type(node).__name__}' is not allowed")
        if isinstance(node, ast.Name) and node.id in (*MATH, *SHAPES, "data") and id(node) not in called:
            raise FormulaError(f"{node.id} is a function - write {node.id}(...)")
        if isinstance(node, ast.Constant) and not isinstance(node.value, (int, float)):
            raise FormulaError(f"{node.value!r} is not a number")
        if isinstance(node, ast.Name) and node.id not in NAMES:
            raise FormulaError(f"unknown name '{node.id}'")
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in (*MATH, *SHAPES, "data"):
                raise FormulaError("only the listed functions can be called")
            if node.keywords:
                raise FormulaError("use positional arguments only")
            expected = ARITY.get(node.func.id, 1)
            if len(node.args) != expected:
                raise FormulaError(f"{node.func.id}() takes {expected} argument(s), got {len(node.args)}")
    return tree


def evaluate_formula(text: str, t: np.ndarray, source: SourceData | None = None) -> np.ndarray:
    """Values of a formula (mA) at the times t (s); an empty formula is 0."""
    if not text.strip():
        return np.zeros_like(t)
    tree = check_formula(text)

    def data(n):
        if source is None:
            raise FormulaError("data() needs a file - open one with Open...")
        return source.channel(n, t)

    names = {"t": t, **CONSTANTS, **MATH, **_shapes(t), "data": data}
    try:
        with np.errstate(all="ignore"):
            value = eval(compile(tree, "<formula>", "eval"), {"__builtins__": {}}, names)  # checked above
        value = np.broadcast_to(np.asarray(value, dtype=float), t.shape).copy()
    except FormulaError:
        raise
    except TypeError as e:
        raise FormulaError(f"wrong arguments: {e}") from None
    except Exception as e:
        raise FormulaError(str(e)) from None
    if not np.isfinite(value).all():
        bad = t[~np.isfinite(value)][0]
        raise FormulaError(f"not a finite number at t = {bad * 1000:g} ms (e.g. division by 0)")
    return value


# --------------------------------------------------------------- waveform


@dataclass
class ChannelFormula:
    formula: str = ""
    range_: AwgRange = AwgRange.HIGH


def sample_times(sample_rate_hz: float, duration_ms: float) -> np.ndarray:
    n = int(round(sample_rate_hz * duration_ms / 1000))
    if n < 1:
        raise ValueError("the duration is shorter than one sample")
    if n > MAX_SAMPLES:
        raise ValueError(f"{n} samples - at most {MAX_SAMPLES}")
    return np.arange(n) / sample_rate_hz


def build_waveform(
    channels: list[ChannelFormula],
    sample_rate_hz: float,
    duration_ms: float,
    max_sample_rate_hz: float = 10000,
    max_amplitude_A: float = 1.0,
    source: SourceData | None = None,
) -> tuple[AwgWaveform | None, list[str], list[str | None]]:
    """Evaluate the 4 channel formulas.

    Returns (waveform, general errors, one error or None per channel). The waveform
    is None only when the time axis is wrong; a channel with an error is 0 in it, so
    it can still be previewed - save it only when there are no errors at all.
    The checks are those of load_awg_npz.
    """
    errors: list[str] = []
    channel_errors: list[str | None] = [None] * N_CHANNELS
    if not 0 < sample_rate_hz <= max_sample_rate_hz:
        errors.append(f"sample rate must be in (0, {max_sample_rate_hz:g}] Hz")
    try:
        t = sample_times(sample_rate_hz, duration_ms) if not errors else None
    except ValueError as e:
        errors.append(str(e))
        t = None
    if t is None:
        return None, errors, channel_errors

    currents = np.zeros((len(t), N_CHANNELS))
    for i, ch in enumerate(channels):
        try:
            values_mA = evaluate_formula(ch.formula, t, source)
        except FormulaError as e:
            channel_errors[i] = str(e)
            continue
        full_scale_mA = RANGE_FULL_SCALE_A[ch.range_] * 1000
        limit_mA = min(full_scale_mA, max_amplitude_A * 1000)
        peak = float(np.abs(values_mA).max(initial=0))
        if peak > limit_mA:
            where = t[np.argmax(np.abs(values_mA))] * 1000
            channel_errors[i] = (f"{peak:.4g} mA at {where:g} ms exceeds +/-{limit_mA:g} mA "
                                 f"({'LOW' if ch.range_ == AwgRange.LOW else 'HIGH'} range)")
            continue
        currents[:, i] = values_mA / 1000
    waveform = AwgWaveform(currents, float(sample_rate_hz), tuple(ch.range_ for ch in channels))
    return waveform, errors, channel_errors


def save_waveform(path: str | Path, waveform: AwgWaveform, channels: list[ChannelFormula],
                  duration_ms: float, max_sample_rate_hz: float = 10000, max_amplitude_A: float = 1.0,
                  source: SourceData | None = None) -> AwgWaveform:
    """Write the .npz (with the formulas - and the data() source if they use it - so the
    file can be edited again) and read it back with load_awg_npz to make sure the
    experiment will accept it."""
    path = Path(path)
    extra = {}
    if source is not None and any("data" in ch.formula for ch in channels):
        extra = {"data_currents_A": source.currents_mA / 1000, "data_sample_rate_hz": source.sample_rate_hz}
    np.savez(
        path,
        currents_A=waveform.currents_A,
        sample_rate_hz=waveform.sample_rate_hz,
        ranges=np.array([int(r) for r in waveform.ranges]),
        formulas=np.array([ch.formula for ch in channels]),
        duration_ms=float(duration_ms),
        **extra,
    )
    return load_awg_npz(path, max_sample_rate_hz=max_sample_rate_hz, max_amplitude_A=max_amplitude_A)


def open_for_editing(path: str | Path, max_sample_rate_hz: float = 10000, max_amplitude_A: float = 1.0
                     ) -> tuple[list[ChannelFormula], float, float, SourceData | None]:
    """(channels, sample rate, duration in ms, data() source) to edit a .npz file.

    A file saved by this editor gives back its formulas. Any other AWG file (e.g.
    one made by a script) is checked with load_awg_npz and every channel becomes
    data(n) - its own samples - which can then be changed or replaced.
    """
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        files = set(data.files)
        if "formulas" in files:
            formulas = [str(f) for f in data["formulas"]]
            ranges = data["ranges"] if "ranges" in files else [AwgRange.HIGH] * N_CHANNELS
            rate = float(data["sample_rate_hz"])
            duration = (float(data["duration_ms"]) if "duration_ms" in files
                        else len(data["currents_A"]) / rate * 1000)
            source = None
            if "data_currents_A" in files:
                source = SourceData(np.asarray(data["data_currents_A"], float) * 1000,
                                    float(data["data_sample_rate_hz"]), path.name)
            return [ChannelFormula(f, AwgRange(int(r))) for f, r in zip(formulas, ranges)], rate, duration, source
    waveform = load_awg_npz(path, max_sample_rate_hz, max_amplitude_A)
    source = SourceData(waveform.currents_A * 1000, waveform.sample_rate_hz, path.name)
    channels = [ChannelFormula(f"data({n})" if np.any(waveform.currents_A[:, n - 1]) else "", r)
                for n, r in zip(range(1, N_CHANNELS + 1), waveform.ranges)]
    return channels, waveform.sample_rate_hz, waveform.duration_s * 1000, source
