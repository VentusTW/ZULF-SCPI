"""Scrollable window with the plots of a finished queue run."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

from zulf_nmr.experiment.schedule import QueueResult
from zulf_nmr.utils.visualization.results_plotter import plot_results


class ResultsViewer(tk.Toplevel):
    """Every transient one below the other, then the average of its configuration, then the next one."""

    def __init__(self, parent: tk.Misc, result: QueueResult) -> None:
        super().__init__(parent)
        self.title(f"Results — {result.folder.name}")
        figure = plot_results(result)
        dpi = self.winfo_fpixels("1i")
        figure.set_dpi(dpi)
        fig_w, fig_h = int(figure.get_figwidth() * dpi), int(figure.get_figheight() * dpi)
        self.geometry(f"{min(fig_w + 40, self.winfo_screenwidth() - 80)}x"
                      f"{min(fig_h + 90, self.winfo_screenheight() - 100)}+40+20")

        bottom = ttk.Frame(self, padding=8)
        bottom.pack(side="bottom", fill="x")
        ttk.Label(bottom, text=str(result.folder)).pack(side="left")
        ttk.Button(bottom, text="Close", command=self.destroy).pack(side="right")

        # The figure keeps its full height; the window scrolls over it.
        holder = ttk.Frame(self)
        holder.pack(side="top", fill="both", expand=True)
        scroll = ttk.Scrollbar(holder, orient="vertical")
        scroll.pack(side="right", fill="y")
        self._view = tk.Canvas(holder, highlightthickness=0, yscrollcommand=scroll.set)
        self._view.pack(side="left", fill="both", expand=True)
        scroll.config(command=self._view.yview)

        inner = ttk.Frame(self._view)
        self._view.create_window(0, 0, window=inner, anchor="nw")
        canvas = FigureCanvasTkAgg(figure, master=inner)
        widget = canvas.get_tk_widget()
        widget.config(width=fig_w, height=fig_h)
        widget.pack()
        canvas.draw()

        toolbar = NavigationToolbar2Tk(canvas, self, pack_toolbar=False)
        toolbar.update()
        toolbar.pack(side="bottom", fill="x", before=bottom)

        inner.update_idletasks()
        self._view.config(scrollregion=(0, 0, fig_w, fig_h))
        for w in (self._view, widget):
            w.bind("<MouseWheel>", self._on_wheel)

    def _on_wheel(self, event: tk.Event) -> None:
        self._view.yview_scroll(int(-event.delta / 120) * 3, "units")
