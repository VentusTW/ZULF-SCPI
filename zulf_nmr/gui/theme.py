"""Theme of the program, dark or light: ttk styles, fonts, colours of the classic Tk
widgets and helpers for the panels (cards, scrolling pages, matplotlib toolbars, title
bars). The colours are this module's globals (BACKGROUND, PANEL, ACCENT...), taken from
the current palette of plot_style; switch() changes the palette and repaints the
widgets already made."""

from __future__ import annotations

import ctypes
import sys
import tkinter as tk
from typing import Callable
from tkinter import font as tkfont
from tkinter import ttk

from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
from PIL import Image, ImageDraw, ImageTk

from zulf_nmr.utils.visualization import plot_style

THEME_NAME = "zulf"
THEMES = ("dark", "light")
BACKGROUND = SURFACE = PANEL = RAISED = RAISED_HOVER = FIELD = BORDER = TEXT = NAV_TEXT = ""
MUTED = DIM = ACCENT = ACCENT_HOVER = ACCENT_TEXT = ACCENT_SOFT = DANGER = DANGER_HOVER = ""
DANGER_SOFT = OK = WARNING = INFO = SELECTION = ERROR_FIELD = CURSOR = SPAN = ""
CLOSE_ICON_PX = 14
_close_images: dict[str, ImageTk.PhotoImage] = {}
_COLOUR_OPTIONS = ("background", "foreground", "highlightbackground", "highlightcolor", "activebackground",
                   "activeforeground", "selectcolor", "selectbackground", "selectforeground", "insertbackground",
                   "troughcolor", "disabledforeground")


def _load() -> None:
    """Copy the colours of plot_style's current palette into this module."""
    globals().update({k: v for k, v in plot_style.PALETTES[plot_style.current].items() if isinstance(v, str)})


def current() -> str:
    return plot_style.current


_load()

SANS_FAMILIES = ("Segoe UI", "Inter", "Helvetica", "Arial")
_fonts: dict[str, str] = {"sans": "TkDefaultFont", "mono": "TkFixedFont"}


def sans(size: int = 10, weight: str = "normal") -> tuple[str, int, str]:
    return _fonts["sans"], size, weight


def mono(size: int = 10, weight: str = "normal") -> tuple[str, int, str]:
    return _fonts["mono"], size, weight


def _pick(families: tuple[str, ...], available: set[str], fallback: str) -> str:
    return next((f for f in families if f in available), fallback)


def apply_theme(root: tk.Tk, name: str = "dark") -> None:
    """Create and use the ttk theme with palette name ("dark" or "light"); set the colours
    of classic Tk widgets created later and of the plots."""
    available = set(tkfont.families(root))
    _fonts["sans"] = _pick(SANS_FAMILIES, available, "TkDefaultFont")
    _fonts["mono"] = _pick((plot_style.mono_family(), *plot_style.MONO_FAMILIES), available, "TkFixedFont")
    plot_style.use(name)
    _load()
    root.configure(background=BACKGROUND)
    _set_options(root)
    style = ttk.Style(root)
    if THEME_NAME not in style.theme_names():
        style.theme_create(THEME_NAME, parent="clam")
    style.theme_use(THEME_NAME)
    _configure_styles(style)
    _closable_tabs(root, style)


def _close_icon(colour: str) -> Image.Image:
    """A small "x" in colour on a transparent background (drawn 4x larger, then reduced)."""
    big = CLOSE_ICON_PX * 4
    image = Image.new("RGBA", (big, big), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    margin, width = big * 0.28, max(big // 9, 1)
    draw.line((margin, margin, big - margin, big - margin), fill=colour, width=width)
    draw.line((margin, big - margin, big - margin, margin), fill=colour, width=width)
    return image.resize((CLOSE_ICON_PX, CLOSE_ICON_PX), Image.LANCZOS)


def _close_colours() -> dict[str, str]:
    return {"normal": MUTED, "active": TEXT, "pressed": DANGER}


def _closable_tabs(root: tk.Tk, style: ttk.Style) -> None:
    """The style Closable.TNotebook: every tab has an "x" after its name (see
    closable_notebook)."""
    if _close_images:
        for state, colour in _close_colours().items():
            _close_images[state].paste(_close_icon(colour))
        return
    for state, colour in _close_colours().items():
        _close_images[state] = ImageTk.PhotoImage(_close_icon(colour), master=root)
    style.element_create("close", "image", _close_images["normal"],
                         ("active", "pressed", "!disabled", _close_images["pressed"]),
                         ("active", "!disabled", _close_images["active"]), border=6, sticky="")
    style.layout("Closable.TNotebook", style.layout("TNotebook"))
    style.layout("Closable.TNotebook.Tab", [
        ("Notebook.tab", {"sticky": "nswe", "children": [
            ("Notebook.padding", {"side": "top", "sticky": "nswe", "children": [
                ("Notebook.focus", {"side": "top", "sticky": "nswe", "children": [
                    ("Notebook.label", {"side": "left", "sticky": ""}),
                    ("Notebook.close", {"side": "left", "sticky": ""}),
                ]}),
            ]}),
        ]}),
    ])


def closable_notebook(parent: tk.Misc, on_close: Callable[[tk.Misc], None]) -> ttk.Notebook:
    """A notebook whose tabs close with their "x": on_close(tab widget) is called when the
    mouse is pressed and released on the same "x"."""
    notebook = ttk.Notebook(parent, style="Closable.TNotebook")
    pressed: list[int] = []

    def press(event) -> str | None:
        if "close" not in notebook.identify(event.x, event.y):
            return None
        pressed[:] = [notebook.index(f"@{event.x},{event.y}")]
        notebook.state(["pressed"])
        return "break"

    def release(event) -> None:
        if not notebook.instate(["pressed"]):
            return
        notebook.state(["!pressed"])
        if not pressed:
            return
        index = pressed.pop()
        if "close" in notebook.identify(event.x, event.y) and notebook.index(f"@{event.x},{event.y}") == index:
            on_close(notebook.nametowidget(notebook.tabs()[index]))

    notebook.bind("<ButtonPress-1>", press, add="+")
    notebook.bind("<ButtonRelease-1>", release, add="+")
    return notebook


def switch(root: tk.Tk, name: str) -> None:
    """Change to palette name and repaint every widget of root; the plots are redrawn by
    their panels (their figures keep the colours they were drawn with)."""
    old = plot_style.PALETTES[plot_style.current]
    plot_style.use(name)
    _load()
    new = plot_style.PALETTES[plot_style.current]
    mapping: dict[str, str] = {}
    for key, value in old.items():
        if isinstance(value, str):
            mapping.setdefault(value.lower(), new[key])
        elif isinstance(value, tuple):
            for a, b in zip(value, new[key]):
                mapping.setdefault(a.lower(), b)
        elif isinstance(value, dict):
            for k, a in value.items():
                mapping.setdefault(a.lower(), new[key][k])
    _set_options(root)
    _configure_styles(ttk.Style(root))
    _closable_tabs(root, ttk.Style(root))
    _recolor(root, mapping)
    title_bar(root)


def _mapped(value, mapping: dict[str, str]) -> str | None:
    text = str(value).lower()
    return mapping.get(text) if text.startswith("#") else None


def _recolor(widget: tk.Misc, mapping: dict[str, str]) -> None:
    hook = getattr(widget, "recolor", None)
    if callable(hook):
        hook(mapping)
    else:
        for option in _COLOUR_OPTIONS:
            try:
                new = _mapped(widget.cget(option), mapping)
            except tk.TclError:
                continue
            if new:
                widget.configure({option: new})
    if isinstance(widget, NavigationToolbar2Tk):
        style_toolbar(widget, _mapped(widget.cget("background"), mapping) or PANEL)
        return
    if isinstance(widget, tk.Text):
        for tag in widget.tag_names():
            for option in ("foreground", "background"):
                new = _mapped(widget.tag_cget(tag, option), mapping)
                if new:
                    widget.tag_configure(tag, {option: new})
    elif isinstance(widget, ttk.Treeview):
        for tag in widget.tk.splitlist(widget.tk.call(widget, "tag", "names")):
            for option in ("foreground", "background"):
                new = _mapped(widget.tag_configure(tag, option), mapping)
                if new:
                    widget.tag_configure(tag, **{option: new})
    elif isinstance(widget, tk.Canvas):
        for item in widget.find_all():
            for option in ("fill", "outline"):
                try:
                    new = _mapped(widget.itemcget(item, option), mapping)
                except tk.TclError:
                    continue
                if new:
                    widget.itemconfigure(item, {option: new})
    for child in widget.winfo_children():
        _recolor(child, mapping)


def _set_options(root: tk.Tk) -> None:
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
    style.configure("Field.Vertical.TScrollbar", troughcolor=FIELD, bordercolor=FIELD)

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


def caption(parent: tk.Misc, text: str, style: str = "Card.TLabel", colour: str | None = None,
            size: int = 8) -> ttk.Label:
    """Small upper-case monospace label, like the section names of the sidebar."""
    return ttk.Label(parent, text=text.upper(), style=style, foreground=colour or MUTED, font=mono(size, "bold"))


class Badge(tk.Label):
    """Small coloured tag with upper-case monospace text (e.g. a state)."""

    def __init__(self, parent: tk.Misc, text: str = "", colour: str | None = None) -> None:
        super().__init__(parent, font=mono(8, "bold"), padx=6, pady=1, borderwidth=0)
        self.set(text, colour)

    def set(self, text: str, colour: str | None = None) -> None:
        self._colour = colour or MUTED
        self.config(text=text.upper(), foreground=self._colour, background=_soft(self._colour))

    def recolor(self, mapping: dict[str, str]) -> None:
        self.set(self.cget("text"), mapping.get(self._colour.lower(), self._colour))


def _soft(colour: str) -> str:
    """colour mixed into the panel colour: a tinted background for a badge."""
    def rgb(c: str) -> tuple[int, int, int]:
        return int(c[1:3], 16), int(c[3:5], 16), int(c[5:7], 16)
    a, b = rgb(colour), rgb(PANEL)
    share = 0.2 if plot_style.current == "dark" else 0.12
    return "#" + "".join(f"{round(share * x + (1 - share) * y):02x}" for x, y in zip(a, b))


class StatusDot(tk.Canvas):
    """A round state indicator."""

    def __init__(self, parent: tk.Misc, background: str, size: int = 10, colour: str | None = None) -> None:
        super().__init__(parent, width=size + 4, height=size + 4, background=background,
                         highlightthickness=0, borderwidth=0)
        self._dot = self.create_oval(2, 2, size + 2, size + 2, fill=colour or DANGER, outline="")

    def set(self, colour: str) -> None:
        self.itemconfig(self._dot, fill=colour)


class ScrollPage(ttk.Frame):
    """A vertically scrolling area: put widgets in .inner; they get the full width."""

    def __init__(self, parent: tk.Misc, background: str | None = None, style: str = "TFrame") -> None:
        super().__init__(parent, style=style)
        self._scroll = ttk.Scrollbar(self, orient="vertical")
        self._scroll.pack(side="right", fill="y")
        self.view = tk.Canvas(self, background=background or BACKGROUND, highlightthickness=0,
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


def style_toolbar(toolbar: tk.Frame, background: str | None = None) -> None:
    """Matplotlib's toolbar on a panel: its background, icons that contrast with it and a
    monospace readout."""
    background = background or PANEL
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


def figure_widget(canvas, background: str | None = None) -> tk.Canvas:
    """The Tk widget of a FigureCanvasTkAgg, without a flash of white before the first draw."""
    widget = canvas.get_tk_widget()
    widget.config(background=background or PANEL, highlightthickness=0)
    return widget


def hide_title_icon(window: tk.Misc) -> None:
    """No icon in the title bar of a window on Windows: a dialog-style frame, no small icon
    of its own and a transparent small icon of its window class (the title bar falls back
    to it). The large icon, shown on the taskbar, stays."""
    if sys.platform != "win32":
        return
    try:
        window.update_idletasks()
        user32 = ctypes.windll.user32
        user32.SendMessageW.argtypes = (ctypes.c_void_p, ctypes.c_uint, ctypes.c_size_t, ctypes.c_void_p)
        hwnd = user32.GetParent(window.winfo_id())
        style = user32.GetWindowLongW(hwnd, -20)
        user32.SetWindowLongW(hwnd, -20, style | 0x0001)
        user32.SendMessageW(hwnd, 0x0080, 0, None)
        user32.CreateIcon.restype = ctypes.c_void_p
        user32.SetClassLongPtrW.argtypes = (ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p)
        size = 16
        and_mask = (ctypes.c_ubyte * (size * size // 8))(*([0xFF] * (size * size // 8)))
        xor_mask = (ctypes.c_ubyte * (size * size // 8))()
        blank = user32.CreateIcon(None, size, size, 1, 1, and_mask, xor_mask)
        if blank:
            user32.SetClassLongPtrW(hwnd, -34, blank)
        user32.SetWindowPos(hwnd, None, 0, 0, 0, 0, 0x0002 | 0x0001 | 0x0004 | 0x0020)
    except (AttributeError, OSError, tk.TclError):
        pass


def title_bar(window: tk.Misc) -> None:
    """Title bar of a window in the colours of the theme on Windows 10/11 (nothing elsewhere)."""
    if sys.platform != "win32":
        return
    try:
        window.update_idletasks()
        hwnd = ctypes.windll.user32.GetParent(window.winfo_id())
        dark = ctypes.c_int(1 if plot_style.current == "dark" else 0)
        ctypes.windll.dwmapi.DwmSetWindowAttribute(hwnd, 20, ctypes.byref(dark), ctypes.sizeof(dark))
        r, g, b = (int(SURFACE[i:i + 2], 16) for i in (1, 3, 5))
        colour = ctypes.c_int(r | g << 8 | b << 16)
        ctypes.windll.dwmapi.DwmSetWindowAttribute(hwnd, 35, ctypes.byref(colour), ctypes.sizeof(colour))
    except (AttributeError, OSError, tk.TclError):
        pass
