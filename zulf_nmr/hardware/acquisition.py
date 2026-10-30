"""ADC sampling and acquisition-time parameters."""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING

import numpy as np

from zulf_nmr.core.binary_block import BlockError, error_code, parse_block

if TYPE_CHECKING:
    from zulf_nmr.core.connection import SCPIConnection

log = logging.getLogger(__name__)

# :ACQuire:DATA:GET? <index> answers #<hdr><bin> (awg_acquire_data.md) with header version 2
# (its chunk index is the one asked for) and payload version 2: samples of 8 channels,
# UInt16 little-endian, channel 1 first, one sample after another. The ADC always records
# all channels; those not selected in the experiment file are dropped on the PC.
ADC_CHANNELS = (1, 2, 3, 4, 5, 6, 7, 8)
ACQ_PAYLOAD_VERSION = 2
FRAME_DTYPE = np.dtype(("<u2", len(ADC_CHANNELS)))
# err-<code> replies of :ACQuire:DATA:GET?
ACQ_ERRORS = {1: "invalid chunk index", 255: "general error"}
# A chunk that arrives damaged (or not at all) is asked for again this many times.
CHUNK_RETRIES = 3


class ChunkNotAvailable(LookupError):
    """err-1: the device has no chunk with that index (not recorded yet, or past the end)."""


def decode_frames(data: bytes) -> np.ndarray:
    """(N, 8) array of raw ADC values from whole frames; len(data) must be a multiple of the frame size."""
    if len(data) % FRAME_DTYPE.itemsize:
        raise ValueError(f"{len(data)} bytes is not a whole number of {FRAME_DTYPE.itemsize}-byte frames")
    return np.frombuffer(data, dtype="<u2").reshape(-1, len(ADC_CHANNELS)).astype(float)


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

    def fetch_chunk(self, index: int) -> np.ndarray:
        """:ACQuire:DATA:GET? <index> - the samples of chunk index, shape (N, 8).

        The same index can be asked for again at no risk, so a damaged chunk (bad checksum,
        wrong index) or a lost answer is simply requested again. Raises ChunkNotAvailable
        on err-1.
        """
        command = f":ACQ:DATA:GET? {index}"
        problem = ""
        for _ in range(CHUNK_RETRIES + 1):
            try:
                reply = self._conn.query_any(command, self.data_timeout_s)
            except TimeoutError:
                problem = "no answer"
                continue
            if isinstance(reply, str):
                code = error_code(reply)
                if code == 1:
                    raise ChunkNotAvailable(f"chunk {index} is not available")
                raise IOError(f"{command}: err-{code} ({ACQ_ERRORS.get(code, 'unknown error')})")
            try:
                block = parse_block(reply)
                if block.chunk_index is not None and block.chunk_index != index:
                    raise BlockError(f"got chunk {block.chunk_index} instead of {index}")
                if block.payload_version != ACQ_PAYLOAD_VERSION:
                    raise BlockError(f"payload version {block.payload_version} is not supported")
                return decode_frames(block.payload)
            except (BlockError, ValueError) as e:
                problem = str(e)
                log.warning("%s: %s - asked again", command, problem)
        raise IOError(f"{command}: {problem} after {CHUNK_RETRIES} repeats")

    # -------------------------------------------------------------- helpers

    def _query_float(self, command: str) -> float:
        reply = self._conn.query(command).strip()
        try:
            return float(reply)
        except ValueError:
            raise ValueError(f"Unexpected {command} reply: {reply!r}") from None
