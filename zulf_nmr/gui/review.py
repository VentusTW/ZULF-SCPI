"""Set review panel: the event timeline and the AWG channels of one set, accepted with OK
or rejected with Exit; after the whole queue is accepted, its sets can be browsed."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Callable

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

from zulf_nmr.core.scpi import header_matches
from zulf_nmr.experiment.parser import SystemConfig
from zulf_nmr.experiment.schedule import QueueItem, format_delay
from zulf_nmr.experiment.sequence import acquisition_window_ms
from zulf_nmr.gui import theme
from zulf_nmr.utils.visualization.awg_plotter import plot_awg
from zulf_nmr.utils.visualization.events_plotter import AcquisitionWindow, plot_events

ROW_PX = 34
TIMELINE_EXTRA_PX = 130
AWG_PX = 560
MIN_TIMELINE_PX = 260


def event_durations(item: QueueItem, system: SystemConfig) -> list[float | None]:
    """Duration of every event in ms: :AWG:RUN lasts as long as the AWG sequence, the
    others as configured in system.json (None = unknown)."""
    awg = item.awg
    return [
        awg.duration_s * 1000 if awg is not None and header_matches(e.action.strip().split(" ", 1)[0], ":AWG:RUN")
        else system.event_duration_ms(e.action)
        for e in item.config.events
    ]


def sequence_length_ms(item: QueueItem, system: SystemConfig) -> float:
    """From 0 to the end of the last event or of the acquisition, whichever is later."""
    ends = [e.time_ms + (d or 0) for e, d in zip(item.config.events, event_durations(item, system))]
    window = acquisition_window_ms(item.config)
    if window:
        ends.append(window[1])
    return max(ends, default=0.0)


def _ms_text(ms: float) -> str:
    return f"{ms / 1000:.6g} s ({ms:,.6g} ms)".replace(",", " ")


class ReviewPanel(ttk.Frame):
    """One set at a time. In review mode OK calls on_ok() and Exit on_exit(); with
    accepted=True (the whole queue is accepted) the sets can be stepped through instead."""

    def __init__(self, parent: tk.Misc, on_ok: Callable[[], None], on_exit: Callable[[], None],
                 on_run: Callable[[], None], on_abort: Callable[[], None],
                 on_queue: Callable[[], None]) -> None:
        super().__init__(parent, padding=(18, 16, 18, 16))
        self._on_ok, self._on_exit = on_ok, on_exit
        self._items: list[QueueItem] = []
        self._index = 0
        self._system: SystemConfig | None = None
        self._accepted = False
        self._locked = False
        self._figures: list[Figure] = []
        self._canvases: list[FigureCanvasTkAgg] = []
        self._cards: list[tk.Frame] = []
        self._build(on_run, on_abort, on_queue)
        self.show_empty("No set to review yet. Build the queue and press Review sets.")

    def _build(self, on_run, on_abort, on_queue) -> None:
        header = theme.card(self)
        header.pack(side="top", fill="x")
        body = header.body
        body.columnconfigure(0, weight=1)

        line = ttk.Frame(body, style="Card.TFrame")
        line.grid(row=0, column=0, sticky="ew")
        self._set_badge = theme.Badge(line, "set –/–", theme.ACCENT)
        self._set_badge.pack(side="left")
        self._file = ttk.Label(line, text="", style="Card.TLabel", foreground=theme.ACCENT, font=theme.mono(11))
        self._file.pack(side="left", padx=(10, 0))
        self._dash = ttk.Label(line, text="", style="Card.TLabel", foreground=theme.DIM, font=theme.sans(12))
        self._dash.pack(side="left", padx=6)
        self._name = ttk.Label(line, text="", style="Card.TLabel", font=theme.sans(16, "bold"))
        self._name.pack(side="left")

        badges = ttk.Frame(body, style="Card.TFrame")
        badges.grid(row=1, column=0, sticky="w", pady=(8, 6))
        self._valid_badge = theme.Badge(badges, "", theme.OK)
        self._valid_badge.pack(side="left")
        self._state_badge = theme.Badge(badges, "", theme.WARNING)
        self._state_badge.pack(side="left", padx=8)

        self._facts = ttk.Frame(body, style="Card.TFrame")
        self._facts.grid(row=2, column=0, sticky="ew")
        self._fact_values: dict[str, ttk.Label] = {}
        for keys in (("Sequence length", "Transients", "Delay to the next set"), ("Acquisition",), ("AWG",)):
            line = ttk.Frame(self._facts, style="Card.TFrame")
            line.pack(side="top", fill="x", pady=2)
            for key in keys:
                theme.caption(line, key + ":").pack(side="left")
                value = ttk.Label(line, text="", style="Card.TLabel", foreground=theme.ACCENT, font=theme.mono(9))
                value.pack(side="left", padx=(6, 28))
                self._fact_values[key] = value

        actions = ttk.Frame(body, style="Card.TFrame")
        actions.grid(row=0, column=1, rowspan=3, sticky="ne")
        top = ttk.Frame(actions, style="Card.TFrame")
        top.pack(side="top", anchor="e")
        self._prev_btn = ttk.Button(top, text="◀  Previous set", command=lambda: self._browse(-1))
        self._next_btn = ttk.Button(top, text="Next set  ▶", command=lambda: self._browse(1))
        self._queue_btn = ttk.Button(top, text="Edit queue", command=on_queue)
        self._queue_btn.pack(side="right")
        bottom = ttk.Frame(actions, style="Card.TFrame")
        bottom.pack(side="top", anchor="e", pady=(8, 0))
        self.run_btn = ttk.Button(bottom, text="▶  RUN QUEUE", style="Accent.TButton", width=16, command=on_run)
        self.run_btn.pack(side="left")
        self.abort_btn = ttk.Button(bottom, text="■  ABORT", style="Danger.TButton", width=10, command=on_abort)
        self.abort_btn.pack(side="left", padx=(8, 0))

        footer = theme.card(self, padding=10)
        footer.pack(side="bottom", fill="x", pady=(12, 0))
        ttk.Label(footer.body, text="ⓘ", style="Card.TLabel", foreground=theme.ACCENT,
                  font=theme.sans(13)).pack(side="left")
        self._hint = ttk.Label(footer.body, text="", style="Card.TLabel", font=theme.sans(10))
        self._hint.pack(side="left", padx=8)
        self.ok_btn = ttk.Button(footer.body, text="OK", style="Accent.TButton", width=12, command=self._ok)
        self.ok_btn.pack(side="right")
        self.exit_btn = ttk.Button(footer.body, text="Exit", width=12, command=self._exit)
        self.exit_btn.pack(side="right", padx=(0, 8))
        for button in (self.ok_btn, self.exit_btn):
            button.bind("<Return>", lambda _e: self._ok())
            button.bind("<Escape>", lambda _e: self._exit())

        self._page = theme.ScrollPage(self)
        self._page.pack(side="top", fill="both", expand=True, pady=(12, 0))
        self._empty = ttk.Label(self._page.inner, text="", foreground=theme.MUTED, font=theme.sans(11),
                                justify="center")

    @property
    def index(self) -> int:
        return self._index

    def show_empty(self, text: str) -> None:
        """No set: only a message (e.g. after a set was rejected)."""
        self._items, self._accepted = [], False
        self._clear_plots()
        self._set_badge.set("set –/–", theme.MUTED)
        for label in (self._file, self._dash, self._name):
            label.config(text="")
        self._name.config(text="Set review")
        self._valid_badge.pack_forget()
        self._state_badge.pack_forget()
        for value in self._fact_values.values():
            value.config(text="–")
        self._empty.config(text=text)
        self._empty.pack(side="top", pady=80)
        self._hint.config(text=text)
        self._update_buttons()

    def show(self, items: list[QueueItem], index: int, system: SystemConfig, accepted: bool) -> None:
        """Show set index of items: for review (accepted=False) or browsing an accepted queue."""
        self._items, self._index, self._system, self._accepted = items, index, system, accepted
        self._empty.pack_forget()
        item, total = items[index], len(items)
        config = item.config
        self._set_badge.set(f"set {index + 1}/{total}", theme.ACCENT)
        self._file.config(text=item.path.name)
        self._dash.config(text="—")
        self._name.config(text=config.experiment_name)
        self._valid_badge.set("validated (ok)", theme.OK)
        self._valid_badge.pack(side="left")
        if accepted:
            self._state_badge.set("accepted", theme.OK)
        else:
            self._state_badge.set("awaiting review", theme.WARNING)
        self._state_badge.pack(side="left", padx=8)

        rate = config.hardware_parameters.acq_sample_rate_hz
        acq = config.acquisition
        self._fact_values["Sequence length"].config(text=_ms_text(sequence_length_ms(item, system)))
        self._fact_values["Transients"].config(text=str(config.output.average_count))
        self._fact_values["Delay to the next set"].config(
            text=format_delay(item.delay_s) if index < total - 1 else "— (last set)")
        self._fact_values["Acquisition"].config(
            text=f"{acq.number_of_samples} samples @ {rate:g} Hz = {acq.duration_s(rate):g} s, "
                 f"CH {', '.join(map(str, acq.channels))}, starts {acq.start_us / 1000:g} ms after :AWG:RUN")
        awg = item.awg
        self._fact_values["AWG"].config(
            text=(f"{awg.source_path.name}, {len(awg.ranges)} channels @ {awg.sample_rate_hz:g} Hz, "
                  f"{awg.duration_s * 1000:g} ms") if awg is not None else "none")

        if accepted:
            self._hint.config(text=f"All {total} set(s) accepted. Press RUN QUEUE to start the queue.")
        elif index < total - 1:
            self._hint.config(text=f"Set {index + 1}/{total}. Next set starts {format_delay(item.delay_s)} "
                                   f"after this one ends. Press OK to accept.")
        else:
            self._hint.config(text=f"Set {index + 1}/{total} (last). Press OK to accept the queue.")
        self._draw(item, system)
        self._update_buttons()
        if not accepted:
            self.ok_btn.focus_set()

    def set_locked(self, locked: bool) -> None:
        """While the queue runs, the review cannot be answered."""
        self._locked = locked
        self._update_buttons()

    def set_run_state(self, can_run: bool, running: bool) -> None:
        self.run_btn.config(state="normal" if can_run and not running else "disabled")
        self.abort_btn.config(state="normal" if running else "disabled")

    def _update_buttons(self) -> None:
        reviewing = bool(self._items) and not self._accepted
        for button in (self.ok_btn, self.exit_btn):
            if reviewing:
                button.pack(side="right", padx=(0, 8) if button is self.exit_btn else 0)
                button.config(state="disabled" if self._locked else "normal")
            else:
                button.pack_forget()
        if self._accepted and len(self._items) > 1:
            if not self._prev_btn.winfo_manager():
                self._prev_btn.pack(side="left", padx=(0, 8))
                self._next_btn.pack(side="left", padx=(0, 8))
            self._prev_btn.config(state="normal" if self._index > 0 else "disabled")
            self._next_btn.config(state="normal" if self._index < len(self._items) - 1 else "disabled")
        else:
            self._prev_btn.pack_forget()
            self._next_btn.pack_forget()

    def _browse(self, step: int) -> None:
        index = self._index + step
        if self._accepted and 0 <= index < len(self._items):
            self.show(self._items, index, self._system, accepted=True)

    def _ok(self) -> None:
        if self._items and not self._accepted and not self._locked:
            self._on_ok()

    def _exit(self) -> None:
        if self._items and not self._accepted and not self._locked:
            self._on_exit()

    def _clear_plots(self) -> None:
        """Free the figures of the previous set."""
        for figure in self._figures:
            figure.clear()
        for card in self._cards:
            card.destroy()
        self._cards, self._canvases, self._figures = [], [], []

    def _plot_card(self, title: str, info: str, note: str, height_px: int) -> tuple[Figure, tk.Frame]:
        card = theme.card(self._page.inner, padding=12)
        card.pack(side="top", fill="x", pady=(0, 12))
        self._cards.append(card)
        head = ttk.Frame(card.body, style="Card.TFrame")
        head.pack(side="top", fill="x", pady=(0, 6))
        theme.heading(head, title).pack(side="left")
        if info:
            theme.Badge(head, info, theme.MUTED).pack(side="left", padx=10)
        if note:
            ttk.Label(head, text=note, style="Card.TLabel", foreground=theme.MUTED,
                      font=theme.mono(9)).pack(side="right")
        dpi = self.winfo_fpixels("1i")
        figure = Figure(figsize=(12, height_px / dpi), dpi=dpi, layout="constrained")
        self._figures.append(figure)
        canvas = FigureCanvasTkAgg(figure, master=card.body)
        widget = theme.figure_widget(canvas)
        widget.config(height=height_px)
        toolbar = NavigationToolbar2Tk(canvas, card.body, pack_toolbar=False)
        toolbar.update()
        theme.style_toolbar(toolbar)
        toolbar.pack(side="bottom", fill="x")
        widget.pack(side="top", fill="x")
        canvas.mpl_connect("scroll_event", lambda e: self._page.scroll(e.step))
        for w in (card, card.body, head, *head.winfo_children()):
            self._page.bind_wheel(w)
        self._canvases.append(canvas)
        return figure, card

    def _draw(self, item: QueueItem, system: SystemConfig) -> None:
        self._clear_plots()
        config, awg = item.config, item.awg
        window = acquisition_window_ms(config)
        channels = config.acquisition.channels
        timeline_acq = AcquisitionWindow(window[0], window[1], channels) if window else None
        rows = len(config.events) + (1 if timeline_acq else 0)
        figure, _card = self._plot_card(
            f"Sequence timeline — {config.experiment_name}", f"{len(config.events)} events",
            "event · time (+duration) · SCPI command", max(MIN_TIMELINE_PX, ROW_PX * rows + TIMELINE_EXTRA_PX))
        plot_events(config.events, title=config.experiment_name, ax=figure.add_subplot(),
                    durations_ms=event_durations(item, system), acquisition=timeline_acq, show_title=False)

        if awg is None:
            card = theme.card(self._page.inner, padding=12)
            card.pack(side="top", fill="x")
            self._cards.append(card)
            theme.heading(card.body, "AWG sequence").pack(side="left")
            ttk.Label(card.body, text="This set has no AWG sequence.", style="Card.TLabel",
                      foreground=theme.MUTED).pack(side="left", padx=12)
        else:
            acq = config.acquisition
            end_ms = acq.start_us / 1000 + acq.duration_s(config.hardware_parameters.acq_sample_rate_hz) * 1000
            figure, _card = self._plot_card(
                f"AWG sequence — {awg.source_path.name}",
                f"{awg.n_samples} samples / {awg.sample_rate_hz:g} Hz / {awg.duration_s * 1000:g} ms",
                f"acquisition: :AWG:RUN + {acq.start_us / 1000:g} ms", AWG_PX)
            plot_awg(awg, fig=figure, acquisition=(acq.start_us / 1000, end_ms, channels), show_title=False)
        for canvas in self._canvases:
            canvas.draw_idle()
        self._page.to_top()

