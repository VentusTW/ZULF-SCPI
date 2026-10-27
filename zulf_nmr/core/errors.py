"""English texts for operating-system errors.

Windows describes its errors in the language of the system (e.g. "Nie mozna odnalezc
okreslonego pliku" instead of "The system cannot find the file specified"), and those
texts end up in the messages of pyserial and of file operations. english_error() puts
an English description, chosen by the error number, in their place.
"""

from __future__ import annotations

import re

# Windows error codes (GetLastError / OSError.winerror)
WINDOWS_ERRORS = {
    2: "not found (no such file, or no such port - is the device connected?)",
    3: "the path was not found",
    5: "access denied (the port or file may be used by another program)",
    6: "invalid handle",
    21: "the device is not ready",
    22: "the device does not recognise the command",
    31: "the device is not working",
    32: "the file is used by another program",
    87: "invalid parameter",
    112: "there is not enough space on the disk",
    121: "the device did not respond in time",
    183: "the file already exists",
    433: "the device does not exist",
    995: "the operation was aborted",
    1167: "the device is not connected",
}
# C library error numbers (OSError.errno)
ERRNO_ERRORS = {
    2: "no such file or directory",
    5: "input/output error",
    13: "permission denied",
    17: "the file already exists",
    21: "is a directory",
    22: "invalid argument",
    28: "no space left on the device",
}

# FileNotFoundError(2, 'localised text', None, 2) - how pyserial nests the system error
_NESTED = re.compile(r"\w*Error\((\d+), '(?:[^'\\]|\\.)*'(?:, None, (\d+))?\)")
# [WinError 5] localised text  /  [Errno 2] text   (as str(OSError) starts)
_WIN = re.compile(r"\[WinError (\d+)\] [^:'\"\r\n]*")
_ERRNO = re.compile(r"\[Errno (\d+)\] [^:'\"\r\n]*")


def _describe(winerror: int | None, errno: int | None) -> str:
    if winerror is not None and winerror in WINDOWS_ERRORS:
        return WINDOWS_ERRORS[winerror]
    if errno is not None and errno in ERRNO_ERRORS:
        return ERRNO_ERRORS[errno]
    code = winerror if winerror is not None else errno
    return f"system error {code}"


def english_error(error: BaseException) -> str:
    """The message of error with every operating-system description in English."""
    # An OSError raised by the system itself (errno + its text); pyserial's errors carry
    # only a message, which english_text() handles.
    if isinstance(error, OSError) and (getattr(error, "winerror", None) or error.errno) and len(error.args) >= 2:
        text = _describe(getattr(error, "winerror", None), error.errno)
        if error.filename:
            text += f": {error.filename}"
        return text
    return english_text(str(error))


def english_text(text: str) -> str:
    """text with every embedded operating-system description replaced by English."""
    text = _NESTED.sub(lambda m: _describe(int(m.group(2)) if m.group(2) else None, int(m.group(1))), text)
    text = _WIN.sub(lambda m: _describe(int(m.group(1)), None), text)
    return _ERRNO.sub(lambda m: _describe(None, int(m.group(1))), text)
