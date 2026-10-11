"""Digital (TTL) outputs."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from zulf_nmr.core.connection import SCPIConnection


class Digital:
    """:DIGital subsystem - logic state of the digital outputs.

    Every command is sent with its absolute path (leading ':') in SCPI short form.
    """

    CHANNELS = (1, 2, 3, 4, 5)

    def __init__(self, connection: SCPIConnection) -> None:
        self._conn = connection

    def set_state(self, channel: int, state: bool | int) -> None:
        """:DIGital:STATe <ch>, <state> - 0 = LOW, 1 = HIGH."""
        self._check_channel(channel)
        if state not in (0, 1):  # also accepts True / False
            raise ValueError(f"Digital state must be 0 or 1, got {state!r}")
        self._conn.write(f":DIG:STAT {channel},{int(state)}")

    def get_state(self, channel: int) -> bool:
        """:DIGital:STATe? <ch> - True if the output is HIGH."""
        self._check_channel(channel)
        reply = self._conn.query(f":DIG:STAT? {channel}").strip()
        if reply not in ("0", "1"):
            raise ValueError(f"Unexpected :DIGital:STATe? reply: {reply!r}")
        return reply == "1"

    def _check_channel(self, channel: int) -> None:
        if channel not in self.CHANNELS:
            raise ValueError(f"Digital channel must be one of {self.CHANNELS}, got {channel!r}")
