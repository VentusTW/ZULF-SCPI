"""AWG: clock, ranges and the upload of a sequence as :AWG:DATA? and :AWG:GATing? blocks.

The formats are those agreed with the firmware team (docs/protocol): every block is
<bin> = header version 2 | payload | CRC-32 (zulf_nmr.core.binary_block), sent in
chunks with consecutive chunk indexes; the device answers every chunk with err-<code>.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

import numpy as np

from zulf_nmr.core.binary_block import build_block, error_code
from zulf_nmr.core.errors import english_error

if TYPE_CHECKING:
    from zulf_nmr.core.connection import SCPIConnection

log = logging.getLogger(__name__)

N_CHANNELS = 4

# :AWG:DATA? payload version 2 (awg_data.md): records of
#   delay (UInt16) - distance from the previous record in samples (1 = the next sample;
#                    more = the device fills the samples between by linear approximation)
#   4 x DAC value (UInt16), channels X, Y, Z, S = 1, 2, 3, 4
DATA_PAYLOAD_VERSION = 2
DATA_RECORD = np.dtype([("delay", "<u2"), ("dac", "<u2", (N_CHANNELS,))])
# :AWG:GATing? payload version 2 (awg_gating.md): records of
#   delay (UInt16) - distance from the previous record in samples
#   4 x EN toggle offset (UInt16) - us from the start of that sample period to the toggle
#                                   of the channel's EN signal; NO_TOGGLE = do not toggle
GATING_PAYLOAD_VERSION = 2
GATING_RECORD = np.dtype([("delay", "<u2"), ("toggle_us", "<u2", (N_CHANNELS,))])
NO_TOGGLE = 0xFFFF
MAX_DELAY = 0xFFFF
# ASSUMED: records per chunk (the documents set no limit); 1000 records = 10 kB.
CHUNK_RECORDS = 1000
# ASSUMED: the first record's delay is counted from sample 0, so a record for sample 0
# has delay 0; the documents only define the distance between two records.
# ASSUMED: the DAC values are offset binary - 0 = -full scale, 0x8000 = 0 A,
# 0xFFFF = +full scale (the documents say UInt16 without the coding).
DAC_ZERO = 0x8000
DAC_HALF_SPAN = 0x7FFF
# A chunk whose checksum the device rejects is sent again this many times.
CRC_RETRIES = 3

# err-<code> replies of :AWG:DATA? and :AWG:GATing?
UPLOAD_ERRORS = {
    1: "repeated block - rejected",
    2: "block not in order - rejected",
    3: "header or payload structure not supported",
    4: "CRC mismatch",
    5: "too much data / out of memory",
    255: "general error",
}


class AwgRange(IntEnum):
    LOW = 0  # 10 mA full scale
    HIGH = 1  # 1 A full scale


RANGE_FULL_SCALE_A = {AwgRange.LOW: 0.010, AwgRange.HIGH: 1.0}


class AwgUploadError(IOError):
    """The device refused a chunk of the AWG sequence."""


def currents_to_codes(currents_A: np.ndarray, ranges: Sequence[AwgRange]) -> np.ndarray:
    """Convert an (N, 4) array of currents in A into UInt16 DAC codes (offset binary)."""
    currents_A = np.asarray(currents_A, dtype=float)
    if currents_A.ndim != 2 or currents_A.shape[1] != N_CHANNELS:
        raise ValueError(f"Expected currents of shape (N, {N_CHANNELS}), got {currents_A.shape}")
    if len(ranges) != N_CHANNELS:
        raise ValueError(f"Expected {N_CHANNELS} ranges, got {len(ranges)}")
    full_scale = np.array([RANGE_FULL_SCALE_A[AwgRange(r)] for r in ranges])
    over = np.abs(currents_A) > full_scale
    if over.any():
        ch = int(np.argwhere(over)[0][1]) + 1
        raise ValueError(f"AWG channel {ch} exceeds its range of +/-{full_scale[ch - 1]} A")
    return np.round(DAC_ZERO + currents_A / full_scale * DAC_HALF_SPAN).astype(np.uint16)


def data_records(codes: np.ndarray) -> np.ndarray:
    """:AWG:DATA? records of (N, 4) DAC codes.

    A sample that lies exactly on the straight line between its neighbours (on every
    channel) is left out: the device's linear approximation gives it back. Flat parts,
    ramps and pulses so take a few records instead of one per sample.
    """
    codes = np.asarray(codes, dtype=np.int64)
    n = len(codes)
    keep = np.ones(n, dtype=bool)
    if n > 2:
        keep[1:-1] = np.any(codes[2:] - 2 * codes[1:-1] + codes[:-2] != 0, axis=1)
    indexes = np.flatnonzero(keep)
    # A gap longer than a delay can say gets the samples at every MAX_DELAY.
    gaps = np.diff(indexes)
    if gaps.size and gaps.max() > MAX_DELAY:
        filled = [indexes[0]]
        for index in indexes[1:]:
            while index - filled[-1] > MAX_DELAY:
                filled.append(filled[-1] + MAX_DELAY)
            filled.append(index)
        indexes = np.array(filled)
    records = np.empty(len(indexes), dtype=DATA_RECORD)
    records["delay"] = np.diff(indexes, prepend=0)
    records["dac"] = codes[indexes]
    return records


def gating_records(enable: np.ndarray, sample_rate_hz: float) -> np.ndarray:
    """:AWG:GATing? records of an (N, 4) enable array (non-zero = the channel's EN on).

    Every channel starts off; a record is made for each sample where at least one channel
    changes, toggling it at the start of that sample period (offset 0 us).
    """
    on = np.asarray(enable) != 0
    if on.ndim != 2 or on.shape[1] != N_CHANNELS:
        raise ValueError(f"Expected enable of shape (N, {N_CHANNELS}), got {on.shape}")
    if 1e6 / sample_rate_hz > NO_TOGGLE:
        raise ValueError(f"A sample period of {1e6 / sample_rate_hz:g} us does not fit the toggle offset")
    changed = on ^ np.vstack([np.zeros((1, N_CHANNELS), dtype=bool), on[:-1]])
    rows: list[tuple[int, list[int]]] = []
    previous = 0
    for index in np.flatnonzero(changed.any(axis=1)):
        while index - previous > MAX_DELAY:  # filler record: nothing toggles
            previous += MAX_DELAY
            rows.append((previous, [NO_TOGGLE] * N_CHANNELS))
        rows.append((int(index), [0 if c else NO_TOGGLE for c in changed[index]]))
        previous = int(index)
    records = np.empty(len(rows), dtype=GATING_RECORD)
    indexes = np.array([index for index, _ in rows], dtype=np.int64)
    records["delay"] = np.diff(indexes, prepend=0)
    records["toggle_us"] = [toggles for _, toggles in rows]
    return records


def chunks(records: np.ndarray, size: int = CHUNK_RECORDS) -> list[np.ndarray]:
    """The records split into chunks of at most size records."""
    return [records[i:i + size] for i in range(0, len(records), size)] or [records[:0]]


# ASSUMED .npz layout of an AWG sequence file (not defined by the firmware):
#   currents_A      float (N, 4)  channel currents in A, one row per sample
#   sample_rate_hz  float scalar  DAC sample rate
#   ranges          int   (4,)    optional, AwgRange per channel (default: HIGH)
#   enable          int   (N, 4)  optional, non-zero = EN on (default: on for the whole sequence)
#   formulas        str   (4,)    optional, written by the AWG editor (not sent to the device)
#   duration_ms     float scalar  optional, written by the AWG editor
#   data_currents_A, data_sample_rate_hz  optional, the file a data(n) formula refers to


@dataclass(frozen=True)
class AwgWaveform:
    """AWG sequence loaded from an .npz file, ready to upload."""

    currents_A: np.ndarray  # (N, 4)
    sample_rate_hz: float
    ranges: tuple[AwgRange, ...]
    enable: np.ndarray | None = None
    source_path: Path | None = None

    @property
    def n_samples(self) -> int:
        return len(self.currents_A)

    @property
    def duration_s(self) -> float:
        return self.n_samples / self.sample_rate_hz

    def time_s(self) -> np.ndarray:
        return np.arange(self.n_samples) / self.sample_rate_hz

    def enable_or_default(self) -> np.ndarray:
        """The enable array; without one every channel is on for the whole sequence."""
        if self.enable is None:
            return np.ones((self.n_samples, N_CHANNELS), dtype=np.uint8)
        return np.asarray(self.enable)

    def data_records(self) -> np.ndarray:
        """:AWG:DATA? records (also checks every current against its range)."""
        return data_records(currents_to_codes(self.currents_A, self.ranges))

    def gating_records(self) -> np.ndarray:
        """:AWG:GATing? records of the enable signals."""
        return gating_records(self.enable_or_default(), self.sample_rate_hz)


def load_awg_npz(
    path: str | Path,
    max_sample_rate_hz: float = 10000,
    max_amplitude_A: float = 1.0,
) -> AwgWaveform:
    """Load and validate an AWG sequence .npz file; raise ValueError on any problem."""
    path = Path(path)
    try:
        with np.load(path, allow_pickle=False) as data:
            arrays = {key: data[key] for key in data.files}
    except FileNotFoundError:
        raise ValueError(f"{path}: file not found") from None
    except Exception as e:
        raise ValueError(f"{path}: not a valid .npz file ({english_error(e)})") from None

    missing = {"currents_A", "sample_rate_hz"} - arrays.keys()
    if missing:
        raise ValueError(f"{path}: missing array(s) {', '.join(sorted(missing))}")
    unknown = arrays.keys() - {"currents_A", "sample_rate_hz", "ranges", "enable", "formulas", "duration_ms",
                               "data_currents_A", "data_sample_rate_hz"}
    if unknown:
        raise ValueError(f"{path}: unknown array(s) {', '.join(sorted(unknown))}")

    currents = np.asarray(arrays["currents_A"], dtype=float)
    if currents.ndim != 2 or currents.shape[1] != N_CHANNELS or len(currents) == 0:
        raise ValueError(f"{path}: currents_A must have shape (N, {N_CHANNELS}), got {currents.shape}")
    if not np.isfinite(currents).all():
        raise ValueError(f"{path}: currents_A contains NaN or infinite values")
    if np.abs(currents).max() > max_amplitude_A:
        raise ValueError(f"{path}: AWG amplitude exceeds the limit of +/-{max_amplitude_A} A")

    rate = float(arrays["sample_rate_hz"])
    if not 0 < rate <= max_sample_rate_hz:
        raise ValueError(f"{path}: sample_rate_hz must be in (0, {max_sample_rate_hz}], got {rate:g}")

    try:
        raw_ranges = arrays.get("ranges", np.full(N_CHANNELS, AwgRange.HIGH))
        ranges = tuple(AwgRange(int(r)) for r in np.ravel(raw_ranges))
    except ValueError:
        raise ValueError(f"{path}: ranges must contain only 0 (LOW) or 1 (HIGH)") from None
    if len(ranges) != N_CHANNELS:
        raise ValueError(f"{path}: ranges must have {N_CHANNELS} values, got {len(ranges)}")

    enable = arrays.get("enable")
    if enable is not None and np.shape(enable) != currents.shape:
        raise ValueError(f"{path}: enable must have the shape of currents_A {currents.shape}, "
                         f"got {np.shape(enable)}")
    waveform = AwgWaveform(currents, rate, ranges, enable, source_path=path)
    try:
        waveform.data_records()  # range check
        waveform.gating_records()
    except ValueError as e:
        raise ValueError(f"{path}: {e}") from None
    return waveform


class Awg:
    """:AWG subsystem - 4-channel arbitrary waveform generator.

    Every command is sent with its absolute path (leading ':') in SCPI short form.
    """

    CHANNELS = (1, 2, 3, 4)

    def __init__(
        self,
        connection: SCPIConnection,
        max_sample_rate_hz: float = 10000,
        max_amplitude_A: float = 1.0,
    ) -> None:
        self._conn = connection
        self.max_sample_rate_hz = max_sample_rate_hz
        self.max_amplitude_A = max_amplitude_A

    # ------------------------------------------------------------ waveform

    def upload(self, waveform: AwgWaveform) -> None:
        """Reset the AWG, send the sample rate and the range of every channel, then the
        samples (:AWG:DATA?) and the enable signals (:AWG:GATing?) chunk by chunk."""
        if np.abs(waveform.currents_A).max(initial=0) > self.max_amplitude_A:
            raise ValueError(f"AWG amplitude exceeds the limit of +/-{self.max_amplitude_A} A")
        data, gating = waveform.data_records(), waveform.gating_records()
        # ASSUMED: :AWG:RST starts the chunk indexes of both commands from 0 again.
        self.reset()
        self.set_sample_rate(waveform.sample_rate_hz)
        for channel, range_ in zip(self.CHANNELS, waveform.ranges):
            self.set_range(channel, range_)
        self.send_records(":AWG:DATA?", DATA_PAYLOAD_VERSION, data)
        self.send_records(":AWG:GAT?", GATING_PAYLOAD_VERSION, gating)
        log.info("AWG sequence sent: %d samples as %d data and %d gating records",
                 waveform.n_samples, len(data), len(gating))

    def send_records(self, command: str, payload_version: int, records: np.ndarray) -> None:
        """Send records in chunks with indexes 0, 1, 2...; every chunk must be answered err-0.

        err-1 (repeated block) after a chunk was sent again means the device already has it
        (its first answer was lost). err-4 (CRC mismatch) sends the chunk again.
        """
        for index, chunk in enumerate(chunks(records)):
            block = build_block(payload_version, index, chunk.tobytes())
            for attempt in range(CRC_RETRIES + 1):
                try:
                    code = error_code(self._conn.query_block(command, block))
                except TimeoutError:
                    if attempt == CRC_RETRIES:
                        raise
                    log.warning("%s chunk %d: no answer - sent again", command, index)
                    continue
                if code == 0 or (code == 1 and attempt > 0):
                    break
                if code == 4 and attempt < CRC_RETRIES:
                    log.warning("%s chunk %d: CRC mismatch - sent again", command, index)
                    continue
                meaning = UPLOAD_ERRORS.get(code, "unknown error")
                raise AwgUploadError(f"{command} chunk {index}: err-{code} ({meaning})")

    # ---------------------------------------------------------------- clock

    def set_sample_rate(self, rate_hz: float) -> None:
        """:AWG:SAMPle:RATE <MHz> - baseline time step between DAC samples."""
        if not 0 < rate_hz <= self.max_sample_rate_hz:
            raise ValueError(
                f"AWG sample rate must be in (0, {self.max_sample_rate_hz}] Hz, got {rate_hz}"
            )
        self._conn.write(f":AWG:SAMP:RATE {rate_hz / 1e6:g}")

    def get_sample_rate(self) -> float:
        """:AWG:SAMPle:RATE? - sample rate in Hz (the device reports MHz)."""
        return self._query_float(":AWG:SAMP:RATE?") * 1e6

    # ---------------------------------------------------------------- range

    def set_range(self, channel: int, range_: AwgRange | int) -> None:
        """:AWG:RANGe <ch>, <bol> - 0 = LOW (10 mA), 1 = HIGH (1 A)."""
        self._check_channel(channel)
        self._conn.write(f":AWG:RANG {channel},{int(AwgRange(range_))}")

    def get_range(self, channel: int) -> AwgRange:
        """:AWG:RANGe? <ch>"""
        self._check_channel(channel)
        reply = self._conn.query(f":AWG:RANG? {channel}").strip()
        try:
            return AwgRange(int(reply))
        except ValueError:
            raise ValueError(f"Unexpected :AWG:RANGe? reply: {reply!r}") from None

    # -------------------------------------------------------------- control

    def run(self) -> None:
        """:AWG:RUN - software trigger of the AWG sequence."""
        self._conn.write(":AWG:RUN")

    def reset(self) -> None:
        """:AWG:RST - reset AWG data and flags."""
        self._conn.write(":AWG:RST")

    # -------------------------------------------------------------- helpers

    def _check_channel(self, channel: int) -> None:
        if channel not in self.CHANNELS:
            raise ValueError(f"AWG channel must be one of {self.CHANNELS}, got {channel!r}")

    def _query_float(self, command: str) -> float:
        reply = self._conn.query(command).strip()
        try:
            return float(reply)
        except ValueError:
            raise ValueError(f"Unexpected {command} reply: {reply!r}") from None
