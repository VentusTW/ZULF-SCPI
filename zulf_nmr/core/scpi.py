"""SCPI syntax helpers: header matching (long/short forms) and string parameters."""

from __future__ import annotations


def mnemonic_matches(token: str, mnemonic: str) -> bool:
    """True if token is the long or the short form of mnemonic (case-insensitive).

    mnemonic is written as in the SCPI dictionary, short form in upper case,
    e.g. "MOTor" accepts "MOTOR", "mot", "Mot" but not "MOTO".
    """
    short = "".join(c for c in mnemonic if c.isupper() or c.isdigit())
    return token.upper() in (short.upper(), mnemonic.upper())


def header_matches(header: str, pattern: str) -> bool:
    """True if a command header (e.g. ":mot:move") matches pattern (e.g. ":MOTor:MOVE").

    Both are treated as absolute paths; a trailing '?' must be present on both or neither.
    """
    header, pattern = header.strip(), pattern.strip()
    if header.endswith("?") != pattern.endswith("?"):
        return False
    tokens = header.rstrip("?").lstrip(":").split(":")
    mnemonics = pattern.rstrip("?").lstrip(":").split(":")
    return len(tokens) == len(mnemonics) and all(
        mnemonic_matches(t, m) for t, m in zip(tokens, mnemonics)
    )


def short_form(pattern: str) -> str:
    """Short form of a dictionary header, e.g. ":MOTor:CONFigure:SIZe" -> ":MOT:CONF:SIZ"."""
    query = "?" if pattern.endswith("?") else ""
    mnemonics = pattern.rstrip("?").lstrip(":").split(":")
    return ":" + ":".join(
        "".join(c for c in m if c.isupper() or c.isdigit()) for m in mnemonics
    ) + query


def quote(text: str) -> str:
    """Encode text as an SCPI string parameter: 'text' with embedded quotes doubled."""
    return "'" + text.replace("'", "''") + "'"


def unquote(text: str) -> str:
    """Decode an SCPI string parameter ('...' or "..."); unquoted text is returned stripped."""
    text = text.strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "'\"":
        q = text[0]
        return text[1:-1].replace(q + q, q)
    return text
