"""Window for building the queue of experiment sets: add, remove, reorder, set delays."""

from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from zulf_nmr.experiment.parser import HardwareLimits
from zulf_nmr.experiment.schedule import TIME_UNITS, QueueItem, format_delay, load_item, queue_errors

_COLOR_ERROR = "#d0342c"
_ERROR_BG = "#fbe3e1"


class QueueEditor(tk.Toplevel):
    """Edit a copy of the queue; ask() returns the new list, or None if cancelled.

    Every file is validated as soon as it is added, and every delay as it is typed.
    """

    def __init__(
        self,
        parent: tk.Misc,
        items: list[QueueItem],
        limits: HardwareLimits,
        initial_dir: Path,
    ) -> None:
        super().__init__(parent)
        self.title("Experiment queue")
        self.geometry("900x560")
        self.minsize(700, 420)
        self.transient(parent)

        self._limits = limits
        self._initial_dir = initial_dir
        self._items = [QueueItem(i.path, i.delay_value, i.delay_unit, i.config, i.awg, list(i.errors))
                       for i in items]
        self._result: list[QueueItem] | None = None
        self._updating_fields = False

        self._build_ui()
        self._refresh()
        if self._items:
            self._select(0)

        self.protocol("WM_DELETE_WINDOW", self._cancel)
        self.bind("<Escape>", lambda _e: self._cancel())

    def ask(self) -> list[QueueItem] | None:
        self.grab_set()
        self.wait_window()
        return self._result

    # -------------------------------------------------------------------- UI

    def _build_ui(self) -> None:
        bottom = ttk.Frame(self, padding=8)
        bottom.pack(side="bottom", fill="x")
        ttk.Label(bottom, text="Sets run from top to bottom. After OK every set is shown "
                               "for review.").pack(side="left")
        ttk.Button(bottom, text="OK", command=self._ok).pack(side="right")
        ttk.Button(bottom, text="Cancel", command=self._cancel).pack(side="right", padx=(0, 6))

        middle = ttk.Frame(self, padding=(8, 8, 8, 0))
        middle.pack(side="top", fill="both", expand=True)

        side = ttk.Frame(middle)
        side.pack(side="right", fill="y", padx=(8, 0))
        for text, command in (
            ("Add…", self._add),
            ("Remove", self._remove),
            ("Move up", lambda: self._move(-1)),
            ("Move down", lambda: self._move(1)),
        ):
            ttk.Button(side, text=text, command=command, width=12).pack(fill="x", pady=(0, 4))

        columns = ("no", "file", "experiment", "delay", "status")
        self._tree = ttk.Treeview(middle, columns=columns, show="headings", selectmode="browse", height=8)
        for col, title, width, stretch in (
            ("no", "#", 36, False),
            ("file", "File", 220, True),
            ("experiment", "Experiment", 180, True),
            ("delay", "Delay before next set", 150, False),
            ("status", "Status", 200, True),
        ):
            self._tree.heading(col, text=title)
            self._tree.column(col, width=width, stretch=stretch, anchor="w")
        self._tree.tag_configure("error", background=_ERROR_BG, foreground=_COLOR_ERROR)
        self._tree.pack(side="top", fill="both", expand=True)
        self._tree.bind("<<TreeviewSelect>>", lambda _e: self._on_select())

        delay_row = ttk.Frame(middle, padding=(0, 8, 0, 4))
        delay_row.pack(side="top", fill="x")
        ttk.Label(delay_row, text="Delay before the next set:").pack(side="left")
        self._delay_var = tk.StringVar()
        self._delay_entry = tk.Entry(delay_row, textvariable=self._delay_var, width=10)
        self._delay_entry.pack(side="left", padx=(6, 4))
        self._unit_var = tk.StringVar(value="s")
        self._unit_box = ttk.Combobox(delay_row, textvariable=self._unit_var, values=list(TIME_UNITS),
                                      state="readonly", width=5)
        self._unit_box.pack(side="left")
        self._delay_msg = ttk.Label(delay_row, text="")
        self._delay_msg.pack(side="left", padx=8)
        self._entry_bg = self._delay_entry.cget("background")
        self._delay_var.trace_add("write", lambda *_: self._on_delay_changed())
        self._unit_var.trace_add("write", lambda *_: self._on_delay_changed())

        self._details = tk.Text(middle, height=7, wrap="word", state="disabled", font=("Consolas", 9))
        self._details.tag_config("error", foreground=_COLOR_ERROR)
        self._details.pack(side="top", fill="x")

    # ------------------------------------------------------------- list view

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
        return values, ("error",) if bad else ()

    def _selected_index(self) -> int | None:
        selection = self._tree.selection()
        return int(selection[0]) if selection else None

    def _select(self, index: int) -> None:
        self._tree.selection_set(str(index))
        self._tree.see(str(index))
        self._on_select()

    def _on_select(self) -> None:
        index = self._selected_index()
        item = self._items[index] if index is not None else None
        self._updating_fields = True
        self._delay_var.set(item.delay_value if item else "")
        self._unit_var.set(item.delay_unit if item else "s")
        self._updating_fields = False
        self._show_delay_state(item)
        self._show_details(item)

    def _show_details(self, item: QueueItem | None) -> None:
        self._details.config(state="normal")
        self._details.delete("1.0", "end")
        if item is not None:
            self._details.insert("end", f"{item.path}\n")
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

    # ---------------------------------------------------------------- delays

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

    def _show_delay_state(self, item: QueueItem | None) -> None:
        state = "normal" if item is not None else "disabled"
        self._delay_entry.config(state=state)
        self._unit_box.config(state="readonly" if item is not None else "disabled")
        error = item.delay_error if item is not None else None
        self._delay_entry.config(background=_ERROR_BG if error else self._entry_bg)
        if error:
            self._delay_msg.config(text=f"Invalid delay: {error}", foreground=_COLOR_ERROR)
        elif item is not None and item is self._items[-1]:
            self._delay_msg.config(text="Not used - this is the last set", foreground="#666666")
        else:
            self._delay_msg.config(text="")

    # --------------------------------------------------------------- actions

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
        for offset, path in enumerate(paths):
            item = load_item(path, self._limits)
            self._items.insert(insert_at + offset, item)
            if item.errors:
                bad.append(item)
        self._initial_dir = Path(paths[-1]).parent
        self._refresh()
        self._select(insert_at)
        if bad:
            messagebox.showerror(
                "Configuration errors",
                "\n\n".join(f"{item.path.name}:\n" + "\n".join(f"  - {e}" for e in item.errors)
                            for item in bad),
                parent=self,
            )

    def _remove(self) -> None:
        index = self._selected_index()
        if index is None:
            return
        del self._items[index]
        self._refresh()
        if self._items:
            self._select(min(index, len(self._items) - 1))

    def _move(self, step: int) -> None:
        index = self._selected_index()
        if index is None or not 0 <= index + step < len(self._items):
            return
        self._items[index], self._items[index + step] = self._items[index + step], self._items[index]
        self._refresh()
        self._select(index + step)

    def _ok(self) -> None:
        if not self._items:
            messagebox.showerror("Empty queue", "Add at least one experiment configuration.", parent=self)
            return
        for item in self._items:  # files may have changed since they were added
            item.reload(self._limits)
        self._refresh()
        problems = queue_errors(self._items)
        if problems:
            messagebox.showerror("Queue contains errors", "\n".join(problems), parent=self)
            return
        self._result = self._items
        self.destroy()

    def _cancel(self) -> None:
        self.destroy()
