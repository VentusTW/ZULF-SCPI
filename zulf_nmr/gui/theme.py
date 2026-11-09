"""Dark theme of the program: ttk styles, fonts, colours of the classic Tk widgets and
helpers for the panels (cards, scrolling pages, matplotlib toolbars, dark title bars)."""

from __future__ import annotations

import ctypes
import sys
import tkinter as tk
from tkinter import font as tkfont
from tkinter import ttk

from zulf_nmr.utils.visualization import plot_style
from zulf_nmr.utils.visualization.plot_style import (
    ACCENT,
    BACKGROUND,
    BORDER,
    DANGER,
    DIM,
    FIELD,
    MUTED,
    OK,
    PANEL,
    RAISED,
    SURFACE,
    TEXT,
    WARNING,
)

THEME_NAME = "zulf-dark"
ACCENT_HOVER = "#4fd5f0"
ACCENT_TEXT = "#03161d"
ACCENT_SOFT = "#123b4a"
DANGER_HOVER = "#f06a6e"
DANGER_SOFT = "#3a1820"
RAISED_HOVER = "#22324f"
SELECTION = "#164e63"
ERROR_FIELD = "#3a1620"
INFO = "#60a5fa"

SANS_FAMILIES = ("Segoe UI", "Inter", "Helvetica", "Arial")
_fonts: dict[str, str] = {"sans": "TkDefaultFont", "mono": "TkFixedFont"}


def sans(size: int = 10, weight: str = "normal") -> tuple[str, int, str]:
    return _fonts["sans"], size, weight


def mono(size: int = 10, weight: str = "normal") -> tuple[str, int, str]:
    return _fonts["mono"], size, weight


def _pick(families: tuple[str, ...], available: set[str], fallback: str) -> str:
    return next((f for f in families if f in available), fallback)


def apply_theme(root: tk.Tk) -> None:
    """Create and use the dark ttk theme; set the colours of classic Tk widgets created later.
    Also makes matplotlib draw dark plots."""
    available = set(tkfont.families(root))
    _fonts["sans"] = _pick(SANS_FAMILIES, available, "TkDefaultFont")
    _fonts["mono"] = _pick((plot_style.mono_family(), *plot_style.MONO_FAMILIES), available, "TkFixedFont")
    plot_style.apply()

    root.configure(background=BACKGROUND)
    for name, value in {
        "*Background": BACKGROUND,
        "*Foreground": TEXT,
        "*Font": sans(10),
        "*selectBackground": SELECTION,
        "*selectForeground": TEXT,
        "*insertBackground": TEXT,
        "*highlightBackground": BACKGROUND,
        "*highlightColor": ACCENT,
        "*activeBackground": RAISED_HOVER,
        "*activeForeground": TEXT,
        "*disabledForeground": DIM,
        "*troughColor": FIELD,
        "*Text.background": FIELD,
        "*Text.relief": "flat",
        "*Text.borderWidth": 0,
        "*Listbox.background": FIELD,
        "*Listbox.relief": "flat",
        "*Entry.background": FIELD,
        "*Canvas.highlightThickness": 0,
        "*Button.relief": "flat",
        "*Button.highlightThickness": 0,
        "*Checkbutton.selectColor": SELECTION,
        "*Checkbutton.highlightThickness": 0,
        "*TCombobox*Listbox.background": FIELD,
        "*TCombobox*Listbox.foreground": TEXT,
        "*TCombobox*Listbox.selectBackground": SELECTION,
        "*TCombobox*Listbox.selectForeground": TEXT,
        "*TCombobox*Listbox.font": mono(10),
    }.items():
        root.option_add(name, value)

    style = ttk.Style(root)
    if THEME_NAME not in style.theme_names():
        style.theme_create(THEME_NAME, parent="clam")
    style.theme_use(THEME_NAME)
    _configure_styles(style)


def _configure_styles(style: ttk.Style) -> None:
    style.configure(".", background=BACKGROUND, foreground=TEXT, fieldbackground=FIELD,
                    bordercolor=BORDER, lightcolor=PANEL, darkcolor=PANEL, troughcolor=FIELD,
                    selectbackground=SELECTION, selectforeground=TEXT, insertcolor=TEXT,
                    focuscolor=ACCENT, font=sans(10))
    style.map(".", foreground=[("disabled", DIM)])

    for prefix, background in (("", BACKGROUND), ("Card.", PANEL), ("Surface.", SURFACE),
                               ("Raised.", RAISED)):
        style.configure(f"{prefix}TFrame", background=background)
        style.configure(f"{prefix}TLabel", background=background, foreground=TEXT)
        style.configure(f"{prefix}TCheckbutton", background=background, foreground=TEXT,
                        indicatorbackground=FIELD, indicatorforeground=ACCENT_TEXT,
                        upperbordercolor=BORDER, lowerbordercolor=BORDER, padding=(4, 2), indicatormargin=(0, 0, 6, 0),
                        font=mono(9))
        style.map(f"{prefix}TCheckbutton",
                  background=[("active", background)],
                  indicatorbackground=[("selected", ACCENT), ("active", RAISED)],
                  foreground=[("disabled", DIM)])

    style.configure("TButton", background=RAISED, foreground=TEXT, bordercolor=BORDER,
                    lightcolor=RAISED, darkcolor=RAISED, focuscolor=RAISED, padding=(12, 5),
                    font=mono(9), anchor="center")
    style.map("TButton",
              background=[("disabled", SURFACE), ("pressed", SELECTION), ("active", RAISED_HOVER)],
              lightcolor=[("disabled", SURFACE), ("active", RAISED_HOVER)],
              darkcolor=[("disabled", SURFACE), ("active", RAISED_HOVER)],
              bordercolor=[("focus", ACCENT_SOFT)],
              foreground=[("disabled", DIM)])
    for name, colour, hover, text, off in (("Accent", ACCENT, ACCENT_HOVER, ACCENT_TEXT, ACCENT_SOFT),
                                           ("Danger", DANGER, DANGER_HOVER, "#ffffff", DANGER_SOFT)):
        style.configure(f"{name}.TButton", background=colour, foreground=text, bordercolor=colour,
                        lightcolor=colour, darkcolor=colour, focuscolor=colour, font=mono(9, "bold"))
        style.map(f"{name}.TButton",
                  background=[("disabled", off), ("pressed", colour), ("active", hover)],
                  lightcolor=[("disabled", off), ("active", hover)],
                  darkcolor=[("disabled", off), ("active", hover)],
                  bordercolor=[("disabled", off), ("active", hover)],
                  foreground=[("disabled", DIM)])
    style.configure("Key.TButton", padding=(2, 3), font=mono(9))
    style.configure("Small.TButton", padding=(8, 2), font=mono(9))

    style.configure("TEntry", fieldbackground=FIELD, foreground=TEXT, bordercolor=BORDER,
                    lightcolor=FIELD, darkcolor=FIELD, insertcolor=TEXT, padding=(6, 4))
    style.map("TEntry", bordercolor=[("focus", ACCENT)], lightcolor=[("focus", FIELD)],
              fieldbackground=[("disabled", SURFACE)], foreground=[("disabled", DIM)])
    style.configure("Error.TEntry", fieldbackground=ERROR_FIELD, bordercolor=DANGER)
    style.map("Error.TEntry", bordercolor=[("focus", DANGER)], fieldbackground=[("disabled", SURFACE)])

    style.configure("TCombobox", fieldbackground=FIELD, background=RAISED, foreground=TEXT,
                    bordercolor=BORDER, lightcolor=FIELD, darkcolor=FIELD, arrowcolor=MUTED,
                    selectbackground=FIELD, selectforeground=TEXT, padding=(6, 4), insertcolor=TEXT)
    style.map("TCombobox",
              fieldbackground=[("readonly", FIELD), ("disabled", SURFACE)],
              background=[("active", RAISED_HOVER)],
              bordercolor=[("focus", ACCENT)],
              arrowcolor=[("active", ACCENT), ("disabled", DIM)],
              selectbackground=[("readonly", FIELD)], selectforeground=[("readonly", TEXT)],
              foreground=[("disabled", DIM)])

    style.configure("Treeview", background=FIELD, fieldbackground=FIELD, foreground=TEXT,
                    bordercolor=BORDER, lightcolor=FIELD, darkcolor=FIELD, rowheight=28, font=mono(9))
    style.map("Treeview", background=[("selected", SELECTION)], foreground=[("selected", TEXT)])
    style.configure("Treeview.Heading", background=SURFACE, foreground=MUTED, bordercolor=BORDER,
                    lightcolor=SURFACE, darkcolor=SURFACE, relief="flat", font=mono(8, "bold"),
                    padding=(6, 6))
    style.map("Treeview.Heading", background=[("active", RAISED)])

    style.configure("TScrollbar", background=RAISED, troughcolor=BACKGROUND, bordercolor=BACKGROUND,
                    lightcolor=RAISED, darkcolor=RAISED, arrowcolor=MUTED, gripcount=0, arrowsize=12)
    style.map("TScrollbar", background=[("active", RAISED_HOVER)], arrowcolor=[("active", ACCENT)])

    style.configure("TProgressbar", background=ACCENT, troughcolor=FIELD, bordercolor=BORDER,
                    lightcolor=ACCENT, darkcolor=ACCENT, thickness=8)

    style.configure("TLabelframe", background=PANEL, bordercolor=BORDER, lightcolor=PANEL, darkcolor=PANEL)
    style.configure("TLabelframe.Label", background=PANEL, foreground=MUTED, font=mono(8, "bold"))

    style.configure("TNotebook", background=BACKGROUND, bordercolor=BORDER, lightcolor=BACKGROUND,
                    darkcolor=BACKGROUND, tabmargins=(0, 0, 0, 0))
    style.configure("TNotebook.Tab", background=SURFACE, foreground=MUTED, bordercolor=BORDER,
                    lightcolor=SURFACE, darkcolor=SURFACE, padding=(14, 6), font=mono(9))
    style.map("TNotebook.Tab",
              background=[("selected", PANEL), ("active", RAISED)],
              foreground=[("selected", ACCENT), ("active", TEXT)],
              lightcolor=[("selected", PANEL)])

    style.configure("TSeparator", background=BORDER)


def card(parent: tk.Misc, padding: int = 14) -> tk.Frame:
    """A panel card: a lighter frame with a thin border; put widgets in its .body."""
    outer = tk.Frame(parent, background=PANEL, highlightthickness=1, highlightbackground=BORDER,
                     highlightcolor=BORDER, borderwidth=0)
    outer.body = ttk.Frame(outer, style="Card.TFrame", padding=padding)
    outer.body.pack(fill="both", expand=True)
    return outer


def heading(parent: tk.Misc, text: str, style: str = "Card.TLabel", size: int = 14) -> ttk.Label:
    return ttk.Label(parent, text=text, style=style, font=sans(size, "bold"))


def caption(parent: tk.Misc, text: str, style: str = "Card.TLabel", colour: str = MUTED,
            size: int = 8) -> ttk.Label:
    """Small upper-case monospace label, like the section names of the sidebar."""
    return ttk.Label(parent, text=text.upper(), style=style, foreground=colour, font=mono(size, "bold"))


class Badge(tk.Label):
    """Small coloured tag with upper-case monospace text (e.g. a state)."""

    def __init__(self, parent: tk.Misc, text: str = "", colour: str = MUTED) -> None:
        super().__init__(parent, font=mono(8, "bold"), padx=6, pady=1, borderwidth=0)
        self.set(text, colour)

    def set(self, text: str, colour: str = MUTED) -> None:
        self.config(text=text.upper(), foreground=colour, background=_soft(colour))


def _soft(colour: str) -> str:
    """colour mixed into the panel colour: a dark tinted background for a badge."""
    def rgb(c: str) -> tuple[int, int, int]:
        return int(c[1:3], 16), int(c[3:5], 16), int(c[5:7], 16)
    a, b = rgb(colour), rgb(PANEL)
    return "#" + "".join(f"{round(0.2 * x + 0.8 * y):02x}" for x, y in zip(a, b))


class StatusDot(tk.Canvas):
    """A round state indicator."""

    def __init__(self, parent: tk.Misc, background: str, size: int = 10, colour: str = DANGER) -> None:
        super().__init__(parent, width=size + 4, height=size + 4, background=background,
                         highlightthickness=0, borderwidth=0)
        self._dot = self.create_oval(2, 2, size + 2, size + 2, fill=colour, outline="")

    def set(self, colour: str) -> None:
        self.itemconfig(self._dot, fill=colour)


class ScrollPage(ttk.Frame):
    """A vertically scrolling area: put widgets in .inner; they get the full width."""

    def __init__(self, parent: tk.Misc, background: str = BACKGROUND, style: str = "TFrame") -> None:
        super().__init__(parent, style=style)
        self._scroll = ttk.Scrollbar(self, orient="vertical")
        self._scroll.pack(side="right", fill="y")
        self.view = tk.Canvas(self, background=background, highlightthickness=0,
                              yscrollcommand=self._scroll.set)
        self.view.pack(side="left", fill="both", expand=True)
        self._scroll.config(command=self.view.yview)
        self.inner = ttk.Frame(self.view, style=style)
        self._window = self.view.create_window(0, 0, window=self.inner, anchor="nw")
        self.inner.bind("<Configure>", lambda _e: self.view.config(scrollregion=self.view.bbox("all")))
        self.view.bind("<Configure>", lambda e: self.view.itemconfigure(self._window, width=e.width))
        self.bind_wheel(self.view)
        self.bind_wheel(self.inner)

    def bind_wheel(self, widget: tk.Misc) -> None:
        widget.bind("<MouseWheel>", lambda e: self.scroll(e.delta / 120), add="+")

    def scroll(self, notches: float) -> None:
        if self.view.yview() != (0.0, 1.0):
            self.view.yview_scroll(int(-notches * 3), "units")

    def to_top(self) -> None:
        """Back to the top once the new content has its size."""
        self.update_idletasks()
        self.view.config(scrollregion=self.view.bbox("all"))
        self.view.yview_moveto(0.0)


def style_toolbar(toolbar: tk.Frame, background: str = PANEL) -> None:
    """Matplotlib's toolbar on a panel: its background, light icons and a monospace readout."""
    toolbar.config(background=background, borderwidth=0)
    for widget in toolbar.winfo_children():
        try:
            widget.config(background=background)
        except tk.TclError:
            continue
        if isinstance(widget, (tk.Button, tk.Checkbutton)):
            widget.config(activebackground=RAISED_HOVER, foreground=TEXT, highlightthickness=0)
            if isinstance(widget, tk.Checkbutton):
                widget.config(selectcolor=SELECTION)
            if getattr(widget, "_image_file", None) is not None:
                toolbar._set_image_for_button(widget)
        elif isinstance(widget, tk.Label):
            widget.config(foreground=ACCENT, font=mono(9))
        elif isinstance(widget, tk.Frame):
            widget.config(background=BORDER)


def figure_widget(canvas, background: str = PANEL) -> tk.Canvas:
    """The Tk widget of a FigureCanvasTkAgg, without the white flash before the first draw."""
    widget = canvas.get_tk_widget()
    widget.config(background=background, highlightthickness=0)
    return widget


def dark_title_bar(window: tk.Misc) -> None:
    """Dark title bar of a window on Windows 10/11 (nothing elsewhere)."""
    if sys.platform != "win32":
        return
    try:
        window.update_idletasks()
        hwnd = ctypes.windll.user32.GetParent(window.winfo_id())
        on = ctypes.c_int(1)
        ctypes.windll.dwmapi.DwmSetWindowAttribute(hwnd, 20, ctypes.byref(on), ctypes.sizeof(on))
        r, g, b = (int(SURFACE[i:i + 2], 16) for i in (1, 3, 5))
        colour = ctypes.c_int(r | g << 8 | b << 16)
        ctypes.windll.dwmapi.DwmSetWindowAttribute(hwnd, 35, ctypes.byref(colour), ctypes.sizeof(colour))
    except (AttributeError, OSError, tk.TclError):
        pass


__all__ = [
    "ACCENT", "BACKGROUND", "BORDER", "DANGER", "DIM", "FIELD", "MUTED", "OK", "PANEL", "RAISED",
    "SURFACE", "TEXT", "WARNING", "INFO", "apply_theme", "sans", "mono", "card", "heading", "caption",
    "Badge", "StatusDot", "ScrollPage", "style_toolbar", "figure_widget", "dark_title_bar",
]
