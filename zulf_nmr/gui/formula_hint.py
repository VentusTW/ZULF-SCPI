"""Greyed hints in a formula field: an example when it is empty, the missing arguments
of the function being typed, or the rest of a function name."""

from __future__ import annotations

import re
import tkinter as tk
from tkinter import ttk

# Argument names shown as hints; the one-argument math functions take "x".
SIGNATURES: dict[str, tuple[str, ...]] = {
    "step": ("t0",), "pulse": ("t0", "t1"), "ramp": ("t0", "t1"), "gauss": ("t0", "sigma"),
    "hann": ("t0", "t1"), "chirp": ("f0", "f1", "t0", "t1"), "data": ("n",),
    "min": ("a", "b"), "max": ("a", "b"),
    **{name: ("x",) for name in ("sin", "cos", "tan", "asin", "acos", "atan", "sinh", "cosh", "tanh",
                                 "exp", "log", "log10", "sqrt", "abs", "sign", "floor", "ceil", "round")},
}
HINT_COLOUR = "#9a9a9a"
_NAME_AT_END = re.compile(r"([A-Za-z_]\w*)$")


def formula_hint(text: str) -> str:
    """What to show greyed after the text (the cursor is at its end); "" for nothing.

    "pul" -> "se(t0, t1)", "pulse(" -> "t0, t1)", "pulse(5ms" -> ", t1)",
    "pulse(5ms, " -> "t1)", "chirp(100Hz, 800Hz, 1ms, 2ms" -> ")".
    """
    # The start of a function name that only one function has: complete it.
    match = _NAME_AT_END.search(text)
    if match and not text[:match.start()].rstrip().endswith(tuple("0123456789.")):
        word = match.group(1)
        if word in SIGNATURES:  # a whole name, no bracket yet
            return f"({', '.join(SIGNATURES[word])})"
        names = [n for n in SIGNATURES if n.startswith(word)]
        if len(names) == 1:
            name = names[0]
            return f"{name[len(word):]}({', '.join(SIGNATURES[name])})"

    # Innermost call still open: which argument is being typed.
    calls: list[list] = []  # [name, commas, characters typed since the last ( or ,]
    for i, ch in enumerate(text):
        if ch == "(":
            before = _NAME_AT_END.search(text[:i].rstrip())
            calls.append([before.group(1) if before else "", 0, ""])
        elif ch == ")":
            if calls:
                calls.pop()
        elif calls:
            if ch == ",":
                calls[-1][1] += 1
                calls[-1][2] = ""
            else:
                calls[-1][2] += ch
    if not calls or calls[-1][0] not in SIGNATURES:
        return ""
    name, index, current = calls[-1]
    params = SIGNATURES[name]
    if current.strip():  # an argument is being typed: show the ones after it
        rest = params[index + 1:]
        return (", " + ", ".join(rest) if rest else "") + ")"
    return ", ".join(params[index:]) + ")"


class GhostText:
    """Shows formula_hint (or the example, when the field is empty) greyed right after the
    text of a ttk.Entry. Only a picture: it is not part of the entry's value."""

    def __init__(self, entry: ttk.Entry, example: str) -> None:
        self.entry = entry
        self.example = example
        self.label = tk.Label(entry, text="", fg=HINT_COLOUR, bg="SystemWindow", bd=0, padx=0, pady=0,
                              font=entry.cget("font"), cursor="xterm")
        self.label.bind("<Button-1>", self._click)
        for event in ("<KeyRelease>", "<ButtonRelease-1>", "<FocusIn>", "<FocusOut>", "<Configure>"):
            entry.bind(event, lambda _e: self.update(), add="+")
        entry.after_idle(self.update)

    def update(self) -> None:
        text = self.entry.get()
        focused = self.entry.focus_get() is self.entry
        at_end = self.entry.index("insert") == len(text)
        if not text:
            hint = self.example
        elif focused and at_end:
            hint = formula_hint(text)
        else:
            hint = ""
        if not hint:
            self.label.place_forget()
            return
        x = self._end_x(text)
        width = self.entry.winfo_width()
        if x is None or x >= width - 4:
            self.label.place_forget()
            return
        self.label.config(text=hint)
        self.label.place(x=x, rely=0.5, anchor="w", width=min(self.label.winfo_reqwidth(), width - 4 - x))

    def _end_x(self, text: str) -> int | None:
        """x of the end of the text inside the entry, None if it is scrolled out of view."""
        if not text:
            box = self.entry.bbox(0)
            return box[0] if box else None
        box = self.entry.bbox(len(text) - 1)
        return box[0] + box[2] if box else None

    def _click(self, _event) -> None:
        self.entry.focus_set()
        self.entry.icursor("end")
        self.update()
