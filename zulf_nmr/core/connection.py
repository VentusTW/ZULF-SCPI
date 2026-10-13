"""PyVISA / PySerial wrapper for SCPI communication."""

from __future__ import annotations

import json
import logging
import queue
import sys
import threading
from pathlib import Path
from typing import Callable, Union

import serial

log = logging.getLogger(__name__)

# Folder holding config/ and data/: the package folder, or - in a PyInstaller
# executable, whose own files are unpacked to a temporary folder - the folder of the .exe.
APP_DIR = (
    Path(sys.executable).resolve().parent if getattr(sys, "frozen", False)
    else Path(__file__).resolve().parent.parent
)
DEFAULT_SYSTEM_CONFIG = APP_DIR / "config" / "system.json"

# A message received from the device: a text line (str) or an
# IEEE 488.2 definite-length binary block #<hdr><bin> (bytes, payload only).
Message = Union[str, bytes]


class SCPIConnection:
    """Serial (USB CDC) link to the STM32 spectrometer.

    A background thread listens on the port all the time and splits the
    incoming stream into messages. Replies to queries are returned by
    query() / query_binary(); every message is also passed to the optional
    on_message callback (called from the listener thread).
    """

    TERMINATOR = b"\n"
    _READ_POLL_S = 0.05

    def __init__(
        self,
        port: str,
        baudrate: int = 115200,
        timeout_s: float = 2.0,
        on_message: Callable[[Message], None] | None = None,
    ) -> None:
        self.port = port
        self.baudrate = baudrate
        self.timeout_s = timeout_s
        self.on_message = on_message

        self._serial: serial.SerialBase | None = None
        self._lock = threading.RLock()
        self._replies: queue.Queue[Message] = queue.Queue()
        self._listener: threading.Thread | None = None
        self._stop = threading.Event()
        self._listener_error: Exception | None = None

    @classmethod
    def from_config(
        cls,
        path: str | Path = DEFAULT_SYSTEM_CONFIG,
        on_message: Callable[[Message], None] | None = None,
    ) -> SCPIConnection:
        """Create a connection from the "connection" section of system.json."""
        with open(path, encoding="utf-8") as f:
            cfg = json.load(f)["connection"]
        return cls(
            port=cfg["port"],
            baudrate=cfg.get("baudrate", 115200),
            timeout_s=cfg.get("timeout_s", 2.0),
            on_message=on_message,
        )

    # ------------------------------------------------------------------ link

    @property
    def is_connected(self) -> bool:
        return (
            self._serial is not None
            and self._serial.is_open
            and self._listener is not None
            and self._listener.is_alive()
        )

    def connect(self) -> None:
        """Open the port, discard stale bytes and start the listener thread."""
        if self.is_connected:
            return
        self.close()
        # serial_for_url also accepts test URLs such as "loop://".
        self._serial = serial.serial_for_url(
            self.port,
            baudrate=self.baudrate,
            timeout=self._READ_POLL_S,
            write_timeout=self.timeout_s,
        )
        self._serial.reset_input_buffer()
        self._serial.reset_output_buffer()
        self._replies = queue.Queue()
        self._listener_error = None
        self._stop.clear()
        self._listener = threading.Thread(
            target=self._listen, name=f"SCPI listener {self.port}", daemon=True
        )
        self._listener.start()

    def close(self) -> None:
        self._stop.set()
        if self._listener is not None:
            self._listener.join(timeout=1.0)
            self._listener = None
        if self._serial is not None:
            self._serial.close()
            self._serial = None

    def __enter__(self) -> SCPIConnection:
        self.connect()
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -------------------------------------------------------------- commands

    def write(self, command: str) -> None:
        """Send a command that produces no reply."""
        self._send(command.encode("ascii") + self.TERMINATOR)

    def write_binary(self, command: str, payload: bytes) -> None:
        """Send a command followed by a #<hdr><bin> block, e.g. :AWG:DATA."""
        length = str(len(payload)).encode("ascii")
        if len(length) > 9:
            raise ValueError("Binary block too large for a definite-length header")
        header = b"#" + str(len(length)).encode("ascii") + length
        self._send(command.encode("ascii") + b" " + header + payload + self.TERMINATOR)

    def query(self, command: str, timeout_s: float | None = None) -> str:
        """Send a query and return the text reply without the terminator."""
        reply = self._query(command, timeout_s)
        if not isinstance(reply, str):
            raise TypeError(f"Expected text reply to {command!r}, got binary block")
        return reply

    def query_binary(self, command: str, timeout_s: float | None = None) -> bytes:
        """Send a query that answers with a #<hdr><bin> block; return the payload."""
        reply = self._query(command, timeout_s)
        if not isinstance(reply, bytes):
            raise TypeError(f"Expected binary block in reply to {command!r}, got {reply!r}")
        return reply

    def _query(self, command: str, timeout_s: float | None) -> Message:
        with self._lock:
            # Anything still queued arrived unrequested; it has already been
            # passed to on_message, so drop it to keep replies in step.
            while True:
                try:
                    stale = self._replies.get_nowait()
                except queue.Empty:
                    break
                log.debug("Discarding unrequested message: %r", stale)
            self.write(command)
            timeout = self.timeout_s if timeout_s is None else timeout_s
            try:
                return self._replies.get(timeout=timeout)
            except queue.Empty:
                self._raise_if_listener_failed()
                raise TimeoutError(f"No reply to {command!r} within {timeout} s") from None

    def _send(self, data: bytes) -> None:
        with self._lock:
            self._require_connected().write(data)

    def _require_connected(self) -> serial.SerialBase:
        self._raise_if_listener_failed()
        if not self.is_connected:
            raise ConnectionError("Not connected - call connect() first")
        return self._serial

    def _raise_if_listener_failed(self) -> None:
        if self._listener_error is not None:
            raise ConnectionError(f"Connection lost: {self._listener_error}") from self._listener_error

    # -------------------------------------------------------------- listener

    def _listen(self) -> None:
        buf = bytearray()
        try:
            while not self._stop.is_set():
                chunk = self._serial.read(max(1, self._serial.in_waiting))
                if not chunk:
                    continue
                buf += chunk
                for message in self._parse(buf):
                    self._dispatch(message)
        except Exception as e:  # port unplugged, driver error, malformed header...
            if not self._stop.is_set():
                log.error("SCPI listener stopped: %s", e)
                self._listener_error = e

    def _parse(self, buf: bytearray) -> list[Message]:
        """Extract complete messages from buf (consumed in place)."""
        messages: list[Message] = []
        while True:
            # Skip line endings left between messages (e.g. after a block).
            while buf[:1] in (b"\r", b"\n"):
                del buf[0]
            if not buf:
                break
            if buf[:1] == b"#":
                if len(buf) < 2:
                    break
                n_digits = int(chr(buf[1]))
                if n_digits == 0:
                    raise ValueError("Indefinite-length binary blocks (#0) are not supported")
                if len(buf) < 2 + n_digits:
                    break
                length = int(buf[2 : 2 + n_digits])
                start = 2 + n_digits
                if len(buf) < start + length:
                    break
                messages.append(bytes(buf[start : start + length]))
                del buf[: start + length]
            else:
                end = buf.find(self.TERMINATOR)
                if end < 0:
                    break
                messages.append(buf[:end].decode("ascii", errors="replace").rstrip("\r"))
                del buf[: end + 1]
        return messages

    def _dispatch(self, message: Message) -> None:
        if self.on_message is not None:
            try:
                self.on_message(message)
            except Exception:
                log.exception("on_message callback failed")
        self._replies.put(message)
