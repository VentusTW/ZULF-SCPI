"""Compensation coil current configuration."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from zulf_nmr.core.connection import SCPIConnection


class Shim:
    """:SHIM subsystem - DC currents of the shimming coils.

    Every command is sent with its absolute path (leading ':') in SCPI short form.
    """

    CHANNELS = (1, 2, 3)

    def __init__(self, connection: SCPIConnection, max_current_mA: float = 100.0) -> None:
        self._conn = connection
        self.max_current_mA = max_current_mA

    def set_current(self, channel: int, current_mA: float) -> None:
        """:SHIM:CURRent <ch>, <mA> - set the DC current of a shimming coil."""
        self._check_channel(channel)
        if abs(current_mA) > self.max_current_mA:
            raise ValueError(
                f"Shim current {current_mA} mA exceeds the limit of +/-{self.max_current_mA} mA"
            )
        self._conn.write(f":SHIM:CURR {channel},{current_mA:g}")

    def get_current(self, channel: int) -> float:
        """:SHIM:CURRent? <ch> - return the configured DC current in mA."""
        self._check_channel(channel)
        reply = self._conn.query(f":SHIM:CURR? {channel}").strip()
        try:
            return float(reply)
        except ValueError:
            raise ValueError(f"Unexpected :SHIM:CURRent? reply: {reply!r}") from None

    def _check_channel(self, channel: int) -> None:
        if channel not in self.CHANNELS:
            raise ValueError(f"Shim channel must be one of {self.CHANNELS}, got {channel!r}")
