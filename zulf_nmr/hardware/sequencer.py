"""Device-side experiment sequence: timed events executed by the STM32."""

from __future__ import annotations

from typing import TYPE_CHECKING

from zulf_nmr.core.scpi import header_matches, quote, unquote

if TYPE_CHECKING:
    from zulf_nmr.core.connection import SCPIConnection

# Commands that :SEQuence:ADD accepts as <event>.
ALLOWED_EVENTS = (
    ":MOTor:MOVE",
    ":SHIM:CURRent",
    ":AWG:RUN",
    ":AWG:RST",
    ":ACQuire:RUN",
    ":DIGital:STATe",
)


def normalize_event(event: str) -> str:
    """Return event as an absolute command; raise ValueError if it may not be sequenced."""
    event = event.strip()
    if not event.startswith(":"):
        event = ":" + event
    header = event.split(maxsplit=1)[0]
    if not any(header_matches(header, allowed) for allowed in ALLOWED_EVENTS):
        raise ValueError(
            f"{header!r} cannot be a sequence event; allowed: {', '.join(ALLOWED_EVENTS)}"
        )
    return event


def _parse_time_event(reply: str, command: str) -> tuple[int, str]:
    try:
        time_ms, event = reply.split(",", 1)
        return int(float(time_ms)), unquote(event)
    except ValueError:
        raise ValueError(f"Unexpected {command} reply: {reply!r}") from None


class Sequencer:
    """:SEQuence subsystem - events scheduled on the device's ms clock.

    Every command is sent with its absolute path (leading ':') in SCPI short form.
    """

    def __init__(self, connection: SCPIConnection) -> None:
        self._conn = connection

    def add(self, time_ms: int, event: str) -> None:
        """:SEQuence:ADD <time>, <event> - schedule event at time_ms from the sequence start."""
        if time_ms < 0:
            raise ValueError(f"Event time must be >= 0 ms, got {time_ms}")
        self._conn.write(f":SEQ:ADD {int(time_ms)},{quote(normalize_event(event))}")

    def last(self) -> tuple[int, str]:
        """:SEQuence:LAST? - return (time_ms, event) of the last added event."""
        return _parse_time_event(self._conn.query(":SEQ:LAST?").strip(), ":SEQuence:LAST?")

    def reset(self) -> None:
        """:SEQuence:RST - remove all events from the sequence."""
        self._conn.write(":SEQ:RST")

    def run(self) -> None:
        """:SEQuence:RUN - software trigger that starts the experiment."""
        self._conn.write(":SEQ:RUN")

    def status(self) -> str:
        """:SEQuence:STATus? - the last event that started."""
        return unquote(self._conn.query(":SEQ:STAT?"))
