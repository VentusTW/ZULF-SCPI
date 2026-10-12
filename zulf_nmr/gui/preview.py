"""Modal window that shows a plot and asks the user to accept or reject it."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure


class PlotPreview(tk.Toplevel):
    """Show figure with a description; ask() returns True if the user accepted it."""

    def __init__(self, parent: tk.Misc, title: str, figure: Figure, description: str = "") -> None:
        super().__init__(parent)
        self.title(title)
        # As tall as the figure needs, but never taller than the screen.
        dpi = self.winfo_fpixels("1i")
        width = min(int(figure.get_figwidth() * dpi), self.winfo_screenwidth() - 80)
        height = min(int(figure.get_figheight() * dpi) + 90, self.winfo_screenheight() - 100)
        self.geometry(f"{width}x{height}+20+20")
        self.minsize(600, 400)
        self.transient(parent)
        self.accepted = False

        buttons = ttk.Frame(self, padding=8)
        buttons.pack(side="bottom", fill="x")
        if description:
            ttk.Label(buttons, text=description).pack(side="left")
        accept = ttk.Button(buttons, text="OK", command=self._accept)
        accept.pack(side="right")
        ttk.Button(buttons, text="Exit", command=self._reject).pack(side="right", padx=(0, 6))

        canvas = FigureCanvasTkAgg(figure, master=self)
        toolbar = NavigationToolbar2Tk(canvas, self, pack_toolbar=False)
        toolbar.update()
        toolbar.pack(side="bottom", fill="x")
        canvas.get_tk_widget().pack(side="top", fill="both", expand=True)
        canvas.draw()

        self.protocol("WM_DELETE_WINDOW", self._reject)
        self.bind("<Escape>", lambda _e: self._reject())
        accept.focus_set()

    def ask(self) -> bool:
        """Block until the window is closed (other windows stay inactive)."""
        self.grab_set()
        self.wait_window()
        return self.accepted

    def _accept(self) -> None:
        self.accepted = True
        self.destroy()

    def _reject(self) -> None:
        self.destroy()
