"""Queue panel: the experiment sets to run - add, remove, reorder, set the delays."""

from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Callable

from zulf_nmr.experiment.parser import HardwareLimits
from zulf_nmr.experiment.schedule import TIME_UNITS, QueueItem, format_delay, load_item, queue_errors
from zulf_nmr.gui import theme
from zulf_nmr.gui.busy import busy
from zulf_nmr.gui.tooltip import Tooltip


def copy_items(items: list[QueueItem]) -> list[QueueItem]:
    return [QueueItem(i.path, i.delay_value, i.delay_unit, i.config, i.awg, list(i.errors)) for i in items]


class QueuePanel(ttk.Frame):
    """Edits a list of sets; "Review sets" validates it and hands it to on_review(items).

    Every file is validated as soon as it is added, and every delay as it is typed.
    on_changed() is called after every change (the queue then needs a new review).
    """

    def __init__(self, parent: tk.Misc, limits: HardwareLimits, initial_dir: Path,
                 on_review: Callable[[list[QueueItem]], None], on_changed: Callable[[], None]) -> None:
        super().__init__(parent, padding=(18, 16, 18, 16))
        self._limits = limits
        self._initial_dir = initial_dir
        self._on_review = on_review
        self._on_changed = on_changed
        self._items: list[QueueItem] = []
        self._updating_fields = False
        self._locked = False
        self._build_ui()
        self._refresh()

    @property
    def items(self) -> list[QueueItem]:
        return self._items

    def set_items(self, items: list[QueueItem]) -> None:
        self._items = copy_items(items)
        self._refresh()
        if self._items:
            self._select(0)

    def add_files(self, paths: list[str | Path]) -> None:
        """Add experiment files at the end of the queue (each one validated)."""
        with busy(self, "Reading the experiment files…"):
            for path in paths:
                self._items.append(load_item(path, self._limits))
        self._refresh()
        if self._items:
            self._select(len(self._items) - 1)
        self._on_changed()

    def set_limits(self, limits: HardwareLimits) -> None:
        self._limits = limits

    def set_locked(self, locked: bool) -> None:
        """While the queue runs it cannot be changed."""
        self._locked = locked
        state = "disabled" if locked else "normal"
        for button in self._buttons:
            button.config(state=state)
        self._lock_note.config(text="The queue is running - abort it to change the sets" if locked else "")
        self._show_delay_state(self._selected_item())

    def _build_ui(self) -> None:
        card = theme.card(self)
        card.pack(side="top", fill="both", expand=True)
        body = card.body

        head = ttk.Frame(body, style="Card.TFrame")
        head.pack(side="top", fill="x")
        theme.heading(head, "Experiment queue").pack(side="left")
        self._count = theme.Badge(head, "empty", theme.MUTED)
        self._count.pack(side="left", padx=10)
        self._review_btn = ttk.Button(head, text="Review sets  ▸", style="Accent.TButton", command=self._ok)
        Tooltip(self._review_btn, "Sets run from top to bottom. Review shows every set; OK on the last one "
                                  "accepts the queue for RUN.")
        self._review_btn.pack(side="right")
        self._buttons = [self._review_btn]
        for text, command in (("Clear", self._clear), ("Move down", lambda: self._move(1)),
                              ("Move up", lambda: self._move(-1)), ("Remove", self._remove), ("Add…", self._add)):
            button = ttk.Button(head, text=text, command=command)
            button.pack(side="right", padx=(0, 6))
            self._buttons.append(button)

        hint = ttk.Frame(body, style="Card.TFrame")
        hint.pack(side="top", fill="x", pady=(6, 10))
        self._lock_note = ttk.Label(hint, text="", style="Card.TLabel", foreground=theme.WARNING,
                                    font=theme.mono(9))
        self._lock_note.pack(side="right")

        details_card = theme.card(body, padding=12)
        details_card.pack(side="bottom", fill="x", pady=(12, 0))
        theme.caption(details_card.body, "Selected set").pack(side="top", anchor="w")
        delay_row = ttk.Frame(details_card.body, style="Card.TFrame")
        delay_row.pack(side="top", fill="x", pady=(8, 8))
        ttk.Label(delay_row, text="Delay before the next set:", style="Card.TLabel").pack(side="left")
        self._delay_var = tk.StringVar()
        self._delay_entry = ttk.Entry(delay_row, textvariable=self._delay_var, width=10, font=theme.mono(10))
        self._delay_entry.pack(side="left", padx=(8, 4))
        self._unit_var = tk.StringVar(value="s")
        self._unit_box = ttk.Combobox(delay_row, textvariable=self._unit_var, values=list(TIME_UNITS),
                                      state="readonly", width=5, font=theme.mono(10))
        self._unit_box.pack(side="left")
        self._delay_msg = ttk.Label(delay_row, text="", style="Card.TLabel", font=theme.mono(9))
        self._delay_msg.pack(side="left", padx=10)
        self._delay_var.trace_add("write", lambda *_: self._on_delay_changed())
        self._unit_var.trace_add("write", lambda *_: self._on_delay_changed())

        self._details = tk.Text(details_card.body, height=6, wrap="word", state="disabled",
                                font=theme.mono(9), background=theme.FIELD, foreground=theme.TEXT,
                                borderwidth=0, highlightthickness=0, padx=8, pady=6)
        self._details.tag_config("error", foreground=theme.DANGER)
        self._details.tag_config("muted", foreground=theme.MUTED)
        self._details.pack(side="top", fill="x")

        table = ttk.Frame(body, style="Card.TFrame")
        table.pack(side="top", fill="both", expand=True)
        columns = ("no", "file", "experiment", "delay", "status")
        self._tree = ttk.Treeview(table, columns=columns, show="headings", selectmode="browse", height=8)
        for col, title, width, stretch in (
            ("no", "#", 40, False),
            ("file", "FILE", 220, True),
            ("experiment", "EXPERIMENT", 200, True),
            ("delay", "DELAY BEFORE NEXT SET", 190, False),
            ("status", "STATUS", 260, True),
        ):
            self._tree.heading(col, text=title, anchor="w")
            self._tree.column(col, width=width, stretch=stretch, anchor="w")
        self._tree.tag_configure("error", background=theme.ERROR_FIELD, foreground=theme.DANGER)
        self._tree.tag_configure("ok", foreground=theme.TEXT)
        scroll = ttk.Scrollbar(table, orient="vertical", command=self._tree.yview)
        self._tree.config(yscrollcommand=scroll.set)
        scroll.pack(side="right", fill="y")
        self._tree.pack(side="left", fill="both", expand=True)
        self._tree.bind("<<TreeviewSelect>>", lambda _e: self._on_select())
        self._tree.bind("<Delete>", lambda _e: self._remove())

    def _refresh(self) -> None:
        selected = self._selected_index()
        self._tree.delete(*self._tree.get_children())
        for i in range(len(self._items)):
            values, tags = self._row(i)
            self._tree.insert("", "end", iid=str(i), values=values, tags=tags)
        if selected is not None and self._items:
            self._select(min(selected, len(self._items) - 1))
        elif not self._items:
            self._show_details(None)
            self._show_delay_state(None)
        n = len(self._items)
        bad = sum(1 for i in range(n) if self._row(i)[1] == ("error",))
        if not n:
            self._count.set("empty", theme.MUTED)
        elif bad:
            self._count.set(f"{n} set(s) · {bad} with errors", theme.DANGER)
        else:
            self._count.set(f"{n} set(s)", theme.ACCENT)

    def _row(self, i: int) -> tuple[tuple, tuple[str, ...]]:
        """Column values and tags of row i. The delay of the last set is not used."""
        item = self._items[i]
        last = i == len(self._items) - 1
        delay_error = None if last else item.delay_error
        if item.errors:
            extra = f" (+{len(item.errors) - 1})" if len(item.errors) > 1 else ""
            status = f"Error: {item.errors[0]}{extra}"
        elif delay_error:
            status = f"Delay error: {delay_error}"
        else:
            status = "OK" + (f", AWG {item.awg.source_path.name}" if item.awg else "")
        if last:
            delay = "— (last set)"
        elif delay_error:
            delay = f"{item.delay_value} {item.delay_unit} (invalid)"
        else:
            delay = format_delay(item.delay_s)
        bad = bool(item.errors) or item.config is None or bool(delay_error)
        values = (i + 1, item.path.name, item.name if item.config else "", delay, status)
        return values, ("error",) if bad else ("ok",)

    def _selected_index(self) -> int | None:
        selection = self._tree.selection()
        return int(selection[0]) if selection else None

    def _selected_item(self) -> QueueItem | None:
        index = self._selected_index()
        return self._items[index] if index is not None and index < len(self._items) else None

    def _select(self, index: int) -> None:
        self._tree.selection_set(str(index))
        self._tree.see(str(index))
        self._on_select()

    def _on_select(self) -> None:
        item = self._selected_item()
        self._updating_fields = True
        self._delay_var.set(item.delay_value if item else "")
        self._unit_var.set(item.delay_unit if item else "s")
        self._updating_fields = False
        self._show_delay_state(item)
        self._show_details(item)

    def _show_details(self, item: QueueItem | None) -> None:
        self._details.config(state="normal")
        self._details.delete("1.0", "end")
        if item is None:
            self._details.insert("end", "No set selected. Add experiment files (.json) with Add…", "muted")
        else:
            self._details.insert("end", f"{item.path}\n", "muted")
            if item.errors:
                self._details.insert("end", "This file contains errors:\n", "error")
                for message in item.errors:
                    self._details.insert("end", f"  - {message}\n", "error")
            elif item.config is not None:
                cfg = item.config
                self._details.insert(
                    "end",
                    f"Experiment: {cfg.experiment_name}, {len(cfg.events)} events, "
                    f"{cfg.output.average_count} transient(s)\n",
                )
                acq, rate = cfg.acquisition, cfg.hardware_parameters.acq_sample_rate_hz
                self._details.insert(
                    "end",
                    f"Acquisition: {acq.number_of_samples} samples at {rate:g} Hz "
                    f"({acq.duration_s(rate):g} s), starts {acq.start_us} us after :AWG:RUN, "
                    f"channels {', '.join(map(str, acq.channels))}\n",
                )
                if item.awg is not None:
                    self._details.insert(
                        "end",
                        f"AWG sequence: {item.awg.source_path.name}, {item.awg.n_samples} samples, "
                        f"{item.awg.duration_s * 1000:g} ms\n",
                    )
                else:
                    self._details.insert("end", "No AWG sequence\n")
        self._details.config(state="disabled")

    def _on_delay_changed(self) -> None:
        if self._updating_fields:
            return
        index = self._selected_index()
        if index is None:
            return
        item = self._items[index]
        item.delay_value = self._delay_var.get()
        item.delay_unit = self._unit_var.get()
        self._show_delay_state(item)
        values, tags = self._row(index)
        self._tree.item(str(index), values=values, tags=tags)
        self._on_changed()

    def _show_delay_state(self, item: QueueItem | None) -> None:
        editable = item is not None and not self._locked
        self._delay_entry.config(state="normal" if editable else "disabled")
        self._unit_box.config(state="readonly" if editable else "disabled")
        error = item.delay_error if item is not None else None
        self._delay_entry.config(style="Error.TEntry" if error else "TEntry")
        if error:
            self._delay_msg.config(text=f"Invalid delay: {error}", foreground=theme.DANGER)
        elif item is not None and item is self._items[-1]:
            self._delay_msg.config(text="Not used - this is the last set", foreground=theme.MUTED)
        else:
            self._delay_msg.config(text="")

    def _add(self) -> None:
        paths = filedialog.askopenfilenames(
            parent=self,
            title="Add experiment configuration(s)",
            initialdir=self._initial_dir,
            filetypes=[("JSON", "*.json"), ("All files", "*.*")],
        )
        if not paths:
            return
        index = self._selected_index()
        insert_at = len(self._items) if index is None else index + 1
        bad: list[QueueItem] = []
        with busy(self, "Reading the experiment files…"):
            for offset, path in enumerate(paths):
                item = load_item(path, self._limits)
                self._items.insert(insert_at + offset, item)
                if item.errors:
                    bad.append(item)
        self._initial_dir = Path(paths[-1]).parent
        self._refresh()
        self._select(insert_at)
        self._on_changed()
        if bad:
            messagebox.showerror(
                "Configuration errors",
                "\n\n".join(f"{item.path.name}:\n" + "\n".join(f"  - {e}" for e in item.errors)
                            for item in bad),
                parent=self,
            )

    def _remove(self) -> None:
        index = self._selected_index()
        if index is None or self._locked:
            return
        del self._items[index]
        self._refresh()
        if self._items:
            self._select(min(index, len(self._items) - 1))
        self._on_changed()

    def _clear(self) -> None:
        if not self._items:
            return
        if not messagebox.askyesno("Clear the queue", f"Remove all {len(self._items)} set(s) from the queue?",
                                   parent=self):
            return
        self._items = []
        self._refresh()
        self._on_changed()

    def _move(self, step: int) -> None:
        index = self._selected_index()
        if index is None or not 0 <= index + step < len(self._items):
            return
        self._items[index], self._items[index + step] = self._items[index + step], self._items[index]
        self._refresh()
        self._select(index + step)
        self._on_changed()

    def _ok(self) -> None:
        if not self._items:
            messagebox.showerror("Empty queue", "Add at least one experiment configuration.", parent=self)
            return
        for item in self._items:
            item.reload(self._limits)
        self._refresh()
        problems = queue_errors(self._items)
        if problems:
            messagebox.showerror("Queue contains errors", "\n".join(problems), parent=self)
            return
        self._on_review(copy_items(self._items))
