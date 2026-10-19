"""ADC sampling and acquisition-time parameters."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Callable

import numpy as np

if TYPE_CHECKING:
    from zulf_nmr.core.connection import SCPIConnection

# ASSUMED - the SCPI dictionary has no ADC channel commands. The ADC has 4
# detection channels and always records all of them; one sample (frame) is
# 4 little-endian int16 values, channel 1 first. Channels not selected in the
# experiment file are dropped on the PC.
ADC_CHANNELS = (1, 2, 3, 4)
FRAME_DTYPE = np.dtype(("<i2", len(ADC_CHANNELS)))


def decode_frames(data: bytes) -> np.ndarray:
    """(N, 4) array of raw ADC values from whole frames; len(data) must be a multiple of the frame size."""
    if len(data) % FRAME_DTYPE.itemsize:
        raise ValueError(f"{len(data)} bytes is not a whole number of {FRAME_DTYPE.itemsize}-byte frames")
    return np.frombuffer(data, dtype="<i2").reshape(-1, len(ADC_CHANNELS)).astype(float)


def duration_for_samples(number_of_samples: int, sample_rate_hz: float) -> int:
    """:ACQuire:DURation value (whole ms) long enough to record number_of_samples."""
    # round() first so that e.g. 20000 samples at 1 kHz gives 20000 ms, not 20001 (float error)
    return max(1, math.ceil(round(number_of_samples / sample_rate_hz * 1000, 6)))


class Acquisition:
    """:ACQuire subsystem - ADC acquisition of the detection channels.

    Every command is sent with its absolute path (leading ':') in SCPI short form.
    """

    def __init__(self, connection: SCPIConnection, data_timeout_s: float | None = None) -> None:
        self._conn = connection
        # Data chunks can be large; None means the connection's default timeout.
        self.data_timeout_s = data_timeout_s

    # -------------------------------------------------------------- control

    def run(self) -> None:
        """:ACQuire:RUN - software trigger of the acquisition."""
        self._conn.write(":ACQ:RUN")

    def set_autostart(self, time_us: int, enable: bool = True) -> None:
        """:ACQuire:AUTostart <time>, <enable> - start acquisition at time_us of the AWG sequence."""
        if time_us < 0:
            raise ValueError(f"Autostart time must be >= 0 us, got {time_us}")
        self._conn.write(f":ACQ:AUT {int(time_us)},{int(enable)}")

    def get_autostart(self) -> tuple[int, bool]:
        """:ACQuire:AUTostart? - return (time_us, enabled)."""
        reply = self._conn.query(":ACQ:AUT?").strip()
        try:
            time_us, enable = (part.strip() for part in reply.split(","))
            if enable not in ("0", "1"):
                raise ValueError
            return int(float(time_us)), enable == "1"
        except ValueError:
            raise ValueError(f"Unexpected :ACQuire:AUTostart? reply: {reply!r}") from None

    # ------------------------------------------------------------ parameters

    def set_sample_rate(self, rate_hz: float) -> None:
        """:ACQuire:SAMPle:RATE <kHz> - ADC sampling frequency."""
        if rate_hz <= 0:
            raise ValueError(f"Sample rate must be positive, got {rate_hz} Hz")
        self._conn.write(f":ACQ:SAMP:RATE {rate_hz / 1e3:g}")

    def get_sample_rate(self) -> float:
        """:ACQuire:SAMPle:RATE? - sample rate in Hz (the device reports kHz)."""
        return self._query_float(":ACQ:SAMP:RATE?") * 1e3

    def set_duration(self, duration_ms: int) -> None:
        """:ACQuire:DURation <ms> - total acquisition time."""
        if duration_ms <= 0:
            raise ValueError(f"Duration must be positive, got {duration_ms} ms")
        self._conn.write(f":ACQ:DUR {int(duration_ms)}")

    def get_duration(self) -> int:
        """:ACQuire:DURation? - acquisition time in ms."""
        return int(self._query_float(":ACQ:DUR?"))

    # ------------------------------------------------------------------ data

    def fetch_new_chunk(self) -> bytes:
        """:ACQuire:DATa:NEW? - drop the last chunk from the buffer, return the oldest one."""
        return self._conn.query_binary(":ACQ:DAT:NEW?", self.data_timeout_s)

    def repeat_chunk(self) -> bytes:
        """:ACQuire:DATa:REPeat? - resend the oldest chunk without removing anything."""
        return self._conn.query_binary(":ACQ:DAT:REP?", self.data_timeout_s)

    def fetch_chunk(self, is_valid: Callable[[bytes], bool], retries: int = 3) -> bytes:
        """Fetch the next chunk; while is_valid() rejects it, ask for a repeat.

        is_valid is the checksum test - the checksum format is not defined yet.
        """
        chunk = self.fetch_new_chunk()
        for _ in range(retries):
            if is_valid(chunk):
                return chunk
            chunk = self.repeat_chunk()
        if is_valid(chunk):
            return chunk
        raise IOError(f"Data chunk failed validation after {retries} repeats")

    # -------------------------------------------------------------- helpers

    def _query_float(self, command: str) -> float:
        reply = self._conn.query(command).strip()
        try:
            return float(reply)
        except ValueError:
            raise ValueError(f"Unexpected {command} reply: {reply!r}") from None
