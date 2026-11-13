"""Loading indicator: a spinning ring and a text in the top bar while data are being read.

The main window makes one BusyIndicator and keeps it as its busy_indicator; any widget
marks work with busy(widget, text), as a context manager around work on the Tk thread or
with begin()/end() around work in another thread (the ring only turns while the Tk
thread is free)."""

from __future__ import annotations

import math
import tkinter as tk
from contextlib import contextmanager
from itertools import count
from typing import Iterator

from PIL import Image, ImageDraw, ImageTk

from zulf_nmr.gui import theme

SPINNER_PX = 18
SPINNER_FRAMES = 24
SPINNER_FRAME_MS = 40
SPINNER_ARC_DEG = 270
SPINNER_STROKE_PX = 2.4
OVERSAMPLE = 4


def _ring_frames(colour: str, track: str) -> list[ImageTk.PhotoImage]:
    """The frames of the turning ring: a faint full circle and a bright arc on it, drawn
    large and scaled down so the edges are smooth."""
    big = SPINNER_PX * OVERSAMPLE
    stroke = round(SPINNER_STROKE_PX * OVERSAMPLE)
    box = (stroke / 2, stroke / 2, big - 1 - stroke / 2, big - 1 - stroke / 2)
    frames = []
    for i in range(SPINNER_FRAMES):
        image = Image.new("RGBA", (big, big), (0, 0, 0, 0))
        draw = ImageDraw.Draw(image)
        draw.ellipse(box, outline=track, width=stroke)
        start = 360 * i / SPINNER_FRAMES
        draw.arc(box, start, start + SPINNER_ARC_DEG * (0.55 + 0.45 * math.sin(math.pi * i / SPINNER_FRAMES)),
                 fill=colour, width=stroke)
        frames.append(ImageTk.PhotoImage(image.resize((SPINNER_PX, SPINNER_PX), Image.LANCZOS)))
    return frames


class BusyIndicator(tk.Frame):
    """The spinning ring and the text of the newest unfinished work; hidden when there is
    none. begin(text) returns a token for end(token)."""

    def __init__(self, parent: tk.Misc, background: str) -> None:
        super().__init__(parent, background=background)
        self._ring = tk.Label(self, background=background, borderwidth=0)
        self._ring.pack(side="left")
        self._text = tk.Label(self, text="", background=background, foreground=theme.ACCENT, font=theme.mono(9))
        self._text.pack(side="left", padx=(6, 0))
        self._frames = _ring_frames(theme.ACCENT, theme.BORDER)
        self._work: dict[int, str] = {}
        self._tokens = count(1)
        self._frame = 0
        self._job: str | None = None
        self._pack_options: dict = {}

    def place_in(self, **pack_options) -> None:
        """Where the indicator is packed while it is shown."""
        self._pack_options = pack_options

    @property
    def active(self) -> bool:
        return bool(self._work)

    def recolor(self, mapping: dict[str, str]) -> None:
        background = mapping.get(self.cget("background"), self.cget("background"))
        for widget in (self, self._ring, self._text):
            widget.config(background=background)
        self._text.config(foreground=theme.ACCENT)
        self._frames = _ring_frames(theme.ACCENT, theme.BORDER)
        self._ring.config(image=self._frames[self._frame])

    def begin(self, text: str) -> int:
        token = next(self._tokens)
        self._work[token] = text
        self._text.config(text=text)
        if self._job is None:
            self.pack(**self._pack_options)
            self.winfo_toplevel().config(cursor="watch")
            self._turn()
        return token

    def end(self, token: int) -> None:
        self._work.pop(token, None)
        if self._work:
            self._text.config(text=next(reversed(self._work.values())))
            return
        if self._job is not None:
            self.after_cancel(self._job)
            self._job = None
        self.pack_forget()
        self.winfo_toplevel().config(cursor="")

    def _turn(self) -> None:
        self._frame = (self._frame + 1) % SPINNER_FRAMES
        self._ring.config(image=self._frames[self._frame])
        self._job = self.after(SPINNER_FRAME_MS, self._turn)


def indicator(widget: tk.Misc) -> BusyIndicator | None:
    return getattr(widget.winfo_toplevel(), "busy_indicator", None)


@contextmanager
def busy(widget: tk.Misc, text: str) -> Iterator[None]:
    """Show the indicator with text while the block runs (painted before it starts)."""
    shown = indicator(widget)
    token = shown.begin(text) if shown is not None else None
    if shown is not None:
        shown.update_idletasks()
    try:
        yield
    finally:
        if token is not None:
            shown.end(token)
