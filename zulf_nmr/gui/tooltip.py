"""A tooltip: a small window with a text, shown after the mouse rests on a widget."""

from __future__ import annotations

import tkinter as tk

from zulf_nmr.gui import theme

DELAY_MS = 1000


class Tooltip:
    """Shows text under the mouse after it rests DELAY_MS on widget; leaving or clicking hides it."""

    def __init__(self, widget: tk.Widget, text: str, delay_ms: int = DELAY_MS) -> None:
        self.widget, self.text, self.delay_ms = widget, text, delay_ms
        self._pending: str | None = None
        self._window: tk.Toplevel | None = None
        widget.bind("<Enter>", self._schedule, add="+")
        widget.bind("<Leave>", self._hide, add="+")
        widget.bind("<ButtonPress>", self._hide, add="+")

    def _schedule(self, _event=None) -> None:
        self._cancel()
        self._pending = self.widget.after(self.delay_ms, self._show)

    def _cancel(self) -> None:
        if self._pending is not None:
            self.widget.after_cancel(self._pending)
            self._pending = None

    def _show(self) -> None:
        self._pending = None
        if self._window is not None or not self.widget.winfo_exists():
            return
        x, y = self.widget.winfo_pointerxy()
        self._window = window = tk.Toplevel(self.widget)
        window.wm_overrideredirect(True)
        window.attributes("-topmost", True)
        tk.Label(window, text=self.text, justify="left", background=theme.RAISED, foreground=theme.TEXT,
                 highlightthickness=1, highlightbackground=theme.BORDER, borderwidth=0, padx=8, pady=4,
                 font=theme.sans(9)).pack()
        window.wm_geometry(f"+{x + 12}+{y + 18}")

    def _hide(self, _event=None) -> None:
        self._cancel()
        if self._window is not None:
            self._window.destroy()
            self._window = None
