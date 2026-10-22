"""Guessing a formula for sampled AWG data (see awg_formula for the formula language).

The channel is cut into active parts separated by zeros; every part is tried as
    a constant            ->  c*pulse(t0, t1)
    straight segments     ->  sum of dv*ramp(ta, tb)       (e.g. a trapezoid)
    a sine (maybe decaying) ->  A*sin(2*pi*f*(t-t0) + phi)*exp(-(t-t0)/T)*pulse(t0, t1)
The result is accepted only if the formula, evaluated like any other, matches every
sample within FIT_TOLERANCE of the channel's peak and within one DAC step (step_mA),
so the device gets practically the same codes. Anything else (e.g. a chirp) is not
recognised - the caller keeps data(n) for it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from zulf_nmr.hardware.awg_formula import FormulaError, evaluate_formula

FIT_TOLERANCE = 1e-3  # max error / channel peak
_MAX_GAP = 3  # zero samples inside one active part (e.g. a sine crossing zero)
_MAX_RAMPS = 8


@dataclass
class FitResult:
    formula: str | None  # None: no simple formula found
    max_error_mA: float  # of the formula, or of the best attempt
    note: str


def fit_channel(values_mA: np.ndarray, sample_rate_hz: float, step_mA: float | None = None) -> FitResult:
    """step_mA: one DAC step of the channel's range (full scale / 32767), if known."""
    values_mA = np.asarray(values_mA, dtype=float)
    peak = float(np.abs(values_mA).max(initial=0))
    if peak == 0:
        return FitResult("", 0.0, "all zero")
    t = np.arange(len(values_mA)) / sample_rate_hz
    tolerance = FIT_TOLERANCE * peak if step_mA is None else min(FIT_TOLERANCE * peak, step_mA)
    terms, notes = [], []
    for i0, i1 in _active_parts(values_mA, peak):
        term, error, note = _fit_part(values_mA, t, i0, i1, sample_rate_hz, tolerance)
        if term is None:
            return FitResult(None, error, f"{_ms(t[i0])}-{_ms(t[i1 - 1])} ms: {note}")
        terms.append(term)
        notes.append(note)
    formula = " + ".join(terms).replace("+ -", "- ")
    error = _error(formula, values_mA, t)
    if error > tolerance:
        return FitResult(None, error, "the parts do not add up")
    kinds = sorted(set(notes))
    return FitResult(formula, error, ", ".join(kinds))


# ------------------------------------------------------------------ parts


def _active_parts(values: np.ndarray, peak: float) -> list[tuple[int, int]]:
    """[start, end) sample ranges of non-zero data; short zero gaps do not split a part."""
    active = np.abs(values) > 1e-9 * peak
    idx = np.nonzero(active)[0]
    parts = []
    start = prev = idx[0]
    for i in idx[1:]:
        if i - prev > _MAX_GAP + 1:
            parts.append((start, prev + 1))
            start = i
        prev = i
    parts.append((start, prev + 1))
    return parts


def _fit_part(values, t, i0, i1, rate, tolerance):
    """(term, its max error over the part, kind) or (None, best error, reason)."""
    part = values[i0:i1]
    t0, t1 = _edge(i0, rate), _edge(i1, rate)
    window = np.zeros_like(values)
    window[i0:i1] = part

    candidates = []
    if np.ptp(part) <= tolerance:
        candidates.append((f"{_num(part.mean())}*pulse({t0}, {t1})", "pulse"))
    ramps = _ramps(values, t, i0, i1, rate, tolerance)
    if ramps:
        candidates.append((ramps, "ramps"))
    candidates.extend(_sines(part, t[i0:i1], t0, t1, rate))

    best = (None, np.inf, "no constant, straight or sine shape fits")
    for term, kind in candidates:
        error = _error(term, window, t)
        if error <= tolerance:
            return term, error, kind
        if error < best[1]:
            best = (None, error, f"not a constant, straight or sine shape (best: {kind}, "
                                 f"error {error:.3g} mA)")
    return best


def _ramps(values, t, i0, i1, rate, tolerance) -> str | None:
    """Straight segments from 0 back to 0: sum of dv*ramp(ta, tb), or None."""
    a, b = max(i0 - 1, 0), min(i1 + 1, len(values))  # include the zero on each side
    x = values[a:b]
    if abs(x[0]) > tolerance or abs(x[-1]) > tolerance:
        return None
    slope = np.diff(x)
    corners = [0] + [k for k in range(1, len(slope)) if abs(slope[k] - slope[k - 1]) > tolerance] + [len(x) - 1]
    if len(corners) - 1 > _MAX_RAMPS:
        return None
    terms = []
    for ka, kb in zip(corners, corners[1:]):
        dv = x[kb] - x[ka]
        if abs(dv) > tolerance:
            terms.append(f"{_num(dv)}*ramp({_ms(t[a + ka])}*ms, {_ms(t[a + kb])}*ms)")
    return " + ".join(terms) if terms else None


def _sines(part, tp, t0, t1, rate):
    """Sine candidates (decaying and not) for one part, from its strongest FFT line."""
    n = len(part)
    if n < 8:
        return []
    tau = tp - tp[0]
    n_fft = 16 * int(2 ** np.ceil(np.log2(n)))
    spectrum = np.abs(np.fft.rfft(part - part.mean(), n_fft))
    f0 = np.fft.rfftfreq(n_fft, 1 / rate)[np.argmax(spectrum[1:]) + 1]
    bin_hz = rate / n

    def fit(f, decay):
        env = np.exp(-decay * tau)
        basis = np.column_stack([np.sin(2 * np.pi * f * tau) * env, np.cos(2 * np.pi * f * tau) * env])
        coef, *_ = np.linalg.lstsq(basis, part, rcond=None)
        return np.abs(basis @ coef - part).max(), f, decay, coef

    # Coarse grid, then a few rounds of finer grids around the best point.
    freqs = np.linspace(max(f0 - bin_hz, bin_hz / 4), f0 + bin_hz, 41)
    decays = [0.0, *np.geomspace(0.1 / tau[-1], 20 / tau[-1], 30)]
    best = min((fit(f, d) for f in freqs for d in decays), key=lambda r: r[0])
    f_step, d_factor = freqs[1] - freqs[0], 1.3
    for _ in range(6):
        _, f_best, d_best, _ = best
        freqs = np.linspace(f_best - 2 * f_step, f_best + 2 * f_step, 21)
        decays = [0.0] if d_best == 0 else np.geomspace(d_best / d_factor, d_best * d_factor, 15)
        best = min((fit(f, d) for f in freqs for d in decays), key=lambda r: r[0])
        f_step, d_factor = f_step / 5, d_factor ** 0.25
    _, f, decay, (s, c) = best
    amplitude, phase = np.hypot(s, c), np.arctan2(c, s)
    start = f"{_ms(tp[0])}*ms"
    # The same sine written with 3, 4, ... significant digits: the first one that is
    # still within the tolerance is used, so the formula stays readable.
    out = []
    for digits in (3, 4, 5, 6, 9):
        sine = (f"{_num(amplitude, digits)}*sin(2*pi*{_num(f, digits)}*(t-{start})"
                f" + {_num(phase, digits)})")
        if decay > 0:
            out.append((f"{sine}*exp(-(t-{start})/{_num(1 / decay, digits)})*pulse({t0}, {t1})",
                        "decaying sine"))
        else:
            out.append((f"{sine}*pulse({t0}, {t1})", "sine"))
    return [(term.replace("+ -", "- "), kind) for term, kind in out]


# ---------------------------------------------------------------- helpers


def _error(formula: str, target: np.ndarray, t: np.ndarray) -> float:
    try:
        return float(np.abs(evaluate_formula(formula, t) - target).max())
    except FormulaError:
        return np.inf


def _edge(i: int, rate: float) -> str:
    """Boundary just before sample i (half a sample earlier, so it is safe from rounding)."""
    return f"{_ms((i - 0.5) / rate)}*ms"


def _ms(seconds: float) -> str:
    return f"{seconds * 1000:.10g}"


def _num(x: float, digits: int = 9) -> str:
    return f"{x:.{digits}g}"
