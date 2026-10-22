"""Clock configuration and compilation of #<hdr><bin> packets for DMA."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

import numpy as np

if TYPE_CHECKING:
    from zulf_nmr.core.connection import SCPIConnection

N_CHANNELS = 4

# ASSUMED sample layout - the SCPI dictionary only says "4x16 bit of data,
# 4x2 bit of enable". Until the firmware fixes it, one sample is 5 little-endian
# 16-bit words (10 bytes, half-word aligned for DMA):
#   word 0..3 : signed DAC code of channel 1..4 (int16, full scale = range)
#   word 4    : enable bits, 2 bits per channel, channel 1 in bits 0-1
SAMPLE_DTYPE = np.dtype([("dac", "<i2", (N_CHANNELS,)), ("enable", "<u2")])


class AwgRange(IntEnum):
    LOW = 0  # 10 mA full scale
    HIGH = 1  # 1 A full scale


RANGE_FULL_SCALE_A = {AwgRange.LOW: 0.010, AwgRange.HIGH: 1.0}


def currents_to_codes(currents_A: np.ndarray, ranges: Sequence[AwgRange]) -> np.ndarray:
    """Convert an (N, 4) array of currents in A into int16 DAC codes."""
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
    return np.round(currents_A / full_scale * np.iinfo(np.int16).max).astype(np.int16)


def pack_samples(codes: np.ndarray, enable: np.ndarray | None = None) -> bytes:
    """Pack (N, 4) DAC codes and (N, 4) 2-bit enable values into the :AWG:DATA payload.

    enable defaults to 1 on every channel and sample.
    """
    codes = np.asarray(codes)
    if codes.ndim != 2 or codes.shape[1] != N_CHANNELS:
        raise ValueError(f"Expected codes of shape (N, {N_CHANNELS}), got {codes.shape}")
    if enable is None:
        enable = np.ones_like(codes, dtype=np.uint16)
    enable = np.asarray(enable)
    if enable.shape != codes.shape:
        raise ValueError(f"enable shape {enable.shape} does not match codes shape {codes.shape}")
    if ((enable < 0) | (enable > 3)).any():
        raise ValueError("enable values must fit in 2 bits (0-3)")

    samples = np.empty(len(codes), dtype=SAMPLE_DTYPE)
    samples["dac"] = codes
    shifts = np.arange(N_CHANNELS, dtype=np.uint16) * 2
    samples["enable"] = (enable.astype(np.uint16) << shifts).sum(axis=1, dtype=np.uint16)
    return samples.tobytes()


# ASSUMED .npz layout of an AWG sequence file (not defined by the firmware yet):
#   currents_A      float (N, 4)  channel currents in A, one row per sample
#   sample_rate_hz  float scalar  DAC sample rate
#   ranges          int   (4,)    optional, AwgRange per channel (default: HIGH)
#   enable          int   (N, 4)  optional, 2-bit enable per sample (default: 1)
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

    def payload(self) -> bytes:
        """Packed :AWG:DATA payload (also checks every current against its range)."""
        return pack_samples(currents_to_codes(self.currents_A, self.ranges), self.enable)


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
        raise ValueError(f"{path}: not a valid .npz file ({e})") from None

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

    waveform = AwgWaveform(currents, rate, ranges, arrays.get("enable"), source_path=path)
    waveform.payload()  # range and enable checks
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

    def load_data(self, payload: bytes) -> None:
        """:AWG:DATA #<hdr><bin> - upload an already packed waveform."""
        if len(payload) % SAMPLE_DTYPE.itemsize:
            raise ValueError(
                f"Payload length {len(payload)} is not a multiple of {SAMPLE_DTYPE.itemsize} bytes"
            )
        self._conn.write_binary(":AWG:DATA", payload)

    def load_waveform(
        self,
        currents_A: np.ndarray,
        ranges: Sequence[AwgRange],
        enable: np.ndarray | None = None,
    ) -> None:
        """Convert (N, 4) currents in A to DAC codes, pack them and send :AWG:DATA."""
        if np.abs(np.asarray(currents_A, dtype=float)).max(initial=0) > self.max_amplitude_A:
            raise ValueError(f"AWG amplitude exceeds the limit of +/-{self.max_amplitude_A} A")
        self.load_data(pack_samples(currents_to_codes(currents_A, ranges), enable))

    def upload(self, waveform: AwgWaveform) -> None:
        """Send the sample rate, the range of every channel and the data of a loaded sequence."""
        if np.abs(waveform.currents_A).max(initial=0) > self.max_amplitude_A:
            raise ValueError(f"AWG amplitude exceeds the limit of +/-{self.max_amplitude_A} A")
        payload = waveform.payload()
        self.set_sample_rate(waveform.sample_rate_hz)
        for channel, range_ in zip(self.CHANNELS, waveform.ranges):
            self.set_range(channel, range_)
        self.load_data(payload)

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
