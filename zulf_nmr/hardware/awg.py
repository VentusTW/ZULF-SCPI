"""Clock configuration and compilation of #<hdr><bin> packets for DMA."""

from __future__ import annotations

from enum import IntEnum
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
