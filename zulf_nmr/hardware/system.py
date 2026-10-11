"""*IDN?, *OPC?, *RST and *CLS commands."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from zulf_nmr.core.connection import SCPIConnection


class System:
    """IEEE 488.2 common commands of the spectrometer."""

    def __init__(self, connection: SCPIConnection) -> None:
        self._conn = connection

    def identify(self) -> str:
        """*IDN? - return the device ID string."""
        return self._conn.query("*IDN?").strip()

    def is_operation_complete(self) -> bool:
        """*OPC? - return True when the sequence is done."""
        reply = self._conn.query("*OPC?").strip()
        if reply not in ("0", "1"):
            raise ValueError(f"Unexpected *OPC? reply: {reply!r}")
        return reply == "1"

    def reset(self) -> None:
        """*RST - instantly stop all events in the sequence and reset the device."""
        self._conn.write("*RST")

    def clear_status(self) -> None:
        """*CLS - empty the error queue and clear status bits."""
        self._conn.write("*CLS")
