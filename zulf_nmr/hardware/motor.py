"""Sample position control (OUT, MAGNET, MEASURE, EXTERNAL)."""

from __future__ import annotations

import time
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from zulf_nmr.core.connection import SCPIConnection


class MotorPosition(str, Enum):
    OUT = "OUT"
    MAGNET = "MAGNET"
    MEASURE = "MEASURE"
    EXTERNAL = "EXTERNAL"
    OTHER = "OTHER"  # reported by :MOTor:POSition? only, cannot be a target


class MotorState(str, Enum):
    BUSY = "BUSY"
    IDLE = "IDLE"


class Motor:
    """:MOTor subsystem - sample shuttle.

    Every command is sent with its absolute path (leading ':') in SCPI short
    form, so it does not depend on the device parser's current node.
    """

    def __init__(self, connection: SCPIConnection, timeout_s: float = 15.0) -> None:
        self._conn = connection
        self.timeout_s = timeout_s

    def zero(self, wait: bool = False) -> None:
        """:MOTor:ZERO - drive to the mechanical hard stop and set position to 0."""
        self._conn.write(":MOT:ZERO")
        if wait:
            self.wait_until_idle()

    def move(self, position: MotorPosition | str, wait: bool = False) -> None:
        """:MOTor:MOVE <pos> - move the sample to OUT, MAGNET, MEASURE or EXTERNAL."""
        if isinstance(position, MotorPosition):
            target = position
        else:
            try:
                target = MotorPosition(position.strip().upper())
            except ValueError:
                raise ValueError(f"Unknown motor position: {position!r}") from None
        if target is MotorPosition.OTHER:
            raise ValueError("OTHER is not a valid motor target")
        self._conn.write(f":MOT:MOVE {target.value}")
        if wait:
            self.wait_until_idle()

    def position(self) -> MotorPosition:
        """:MOTor:POSition? - physical position of the shuttle."""
        reply = self._conn.query(":MOT:POS?").strip().upper()
        try:
            return MotorPosition(reply)
        except ValueError:
            raise ValueError(f"Unexpected :MOTor:POSition? reply: {reply!r}") from None

    def status(self) -> MotorState:
        """:MOTor:STATus? - BUSY while moving, IDLE otherwise."""
        reply = self._conn.query(":MOT:STAT?").strip().upper()
        try:
            return MotorState(reply)
        except ValueError:
            raise ValueError(f"Unexpected :MOTor:STATus? reply: {reply!r}") from None

    def is_busy(self) -> bool:
        return self.status() is MotorState.BUSY

    def set_sample_size(self, size_mm: float) -> None:
        """:MOTor:CONFigure:SIZe <mm> - sample size used to offset the positions."""
        if size_mm <= 0:
            raise ValueError(f"Sample size must be positive, got {size_mm} mm")
        self._conn.write(f":MOT:CONF:SIZ {size_mm:g}")

    def wait_until_idle(self, timeout_s: float | None = None, poll_s: float = 0.1) -> None:
        """Poll :MOTor:STATus? until IDLE; raise TimeoutError after timeout_s."""
        timeout = self.timeout_s if timeout_s is None else timeout_s
        deadline = time.monotonic() + timeout
        while self.is_busy():
            if time.monotonic() > deadline:
                raise TimeoutError(f"Motor still BUSY after {timeout} s")
            time.sleep(poll_s)
