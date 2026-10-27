"""What the user last typed in the windows, restored at the next start.

Kept per Windows user in %APPDATA%\\ZULF-SCPI\\settings.json (~/.config/ZULF-SCPI on other
systems), so it is neither in the project folder nor next to the .exe. A missing or
damaged file only means the program starts with its defaults.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from zulf_nmr.core.errors import english_error

log = logging.getLogger(__name__)

SETTINGS_FILE = Path(os.environ.get("APPDATA") or Path.home() / ".config") / "ZULF-SCPI" / "settings.json"


def load_section(name: str) -> dict[str, Any]:
    """The saved values of one window; {} when there are none."""
    try:
        with open(SETTINGS_FILE, encoding="utf-8") as f:
            section = json.load(f).get(name, {})
    except FileNotFoundError:
        return {}
    except (OSError, ValueError, AttributeError) as e:
        log.warning("Saved settings not read from %s: %s", SETTINGS_FILE, english_error(e))
        return {}
    return section if isinstance(section, dict) else {}


def save_section(name: str, values: dict[str, Any]) -> None:
    """Replace the saved values of one window; the other windows' values are kept."""
    try:
        with open(SETTINGS_FILE, encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            data = {}
    except (OSError, ValueError):
        data = {}
    data[name] = values
    try:
        SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
        temporary = SETTINGS_FILE.with_suffix(".tmp")
        with open(temporary, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        temporary.replace(SETTINGS_FILE)  # never a half-written file
    except OSError as e:
        log.warning("Settings not saved to %s: %s", SETTINGS_FILE, english_error(e))


def text(section: dict[str, Any], key: str, default: str) -> str:
    """A saved text value, or the default when it is missing or not text."""
    value = section.get(key)
    return value if isinstance(value, str) else default
