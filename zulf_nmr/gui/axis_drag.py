"""Stretching an axis of a matplotlib plot with the mouse: press the left button on the
numbers of an axis (below a plot for x, left of it for y) and drag - right or up makes
the range smaller (zoom in), left or down larger (zoom out), around the value where the
drag started."""

from __future__ import annotations

import math

from matplotlib.backend_tools import Cursors

BAND_PX = 46
PIXELS_PER_E = 160.0


class AxisDrag:
    """Axis stretching for every 2D axes of a canvas.

    With manage_cursor=True it also shows a double arrow over the numbers of an axis;
    a plot that sets the mouse cursor itself asks cursor_at() instead. on_change(ax) is
    called after every change of a range (e.g. to redraw thinned curves)."""

    def __init__(self, canvas, toolbar=None, manage_cursor: bool = True, on_change=None) -> None:
        self.canvas = canvas
        self.toolbar = toolbar
        self.manage_cursor = manage_cursor
        self.on_change = on_change
        self.dragging = False
        self._drag = None
        self._cursor_set = False
        canvas.mpl_connect("button_press_event", self._press)
        canvas.mpl_connect("motion_notify_event", self._motion)
        canvas.mpl_connect("button_release_event", self._release)

    def _tool_active(self) -> bool:
        return bool(self.toolbar is not None and getattr(self.toolbar, "mode", None))

    def axis_at(self, event):
        """(axes, "x" or "y") whose numbers are under the mouse, or None."""
        if event.x is None or event.y is None:
            return None
        for ax in self.canvas.figure.axes:
            if ax.name != "rectilinear" or not ax.get_visible() or not ax.get_navigate():
                continue
            box = ax.bbox
            if box.x0 <= event.x <= box.x1 and box.y0 - BAND_PX <= event.y < box.y0:
                return ax, "x"
            if box.y0 <= event.y <= box.y1 and box.x0 - BAND_PX <= event.x < box.x0:
                return ax, "y"
        return None

    def cursor_at(self, event) -> Cursors | None:
        """The cursor for the mouse over an axis's numbers (None elsewhere)."""
        if self._tool_active():
            return None
        hit = self._drag[:2] if self._drag else self.axis_at(event)
        if hit is None:
            return None
        return Cursors.RESIZE_HORIZONTAL if hit[1] == "x" else Cursors.RESIZE_VERTICAL

    def _press(self, event) -> None:
        if event.button != 1 or self._tool_active():
            return
        hit = self.axis_at(event)
        if hit is None:
            return
        ax, which = hit
        low, high = ax.get_xlim() if which == "x" else ax.get_ylim()
        to_data = ax.transData.inverted()
        anchor = to_data.transform((event.x, event.y))[0 if which == "x" else 1]
        if self.toolbar is not None:
            self.toolbar.push_current()
        self._drag = (ax, which, event.x if which == "x" else event.y, (low, high), anchor)
        self.dragging = True

    def _motion(self, event) -> None:
        if self._drag is None:
            if self.manage_cursor:
                cursor = self.cursor_at(event)
                if cursor is not None:
                    self.canvas.set_cursor(cursor)
                    self._cursor_set = True
                elif self._cursor_set:
                    self.canvas.set_cursor(Cursors.POINTER)
                    self._cursor_set = False
            return
        ax, which, start, (low, high), anchor = self._drag
        position = event.x if which == "x" else event.y
        if position is None:
            return
        factor = math.exp(-(position - start) / PIXELS_PER_E)
        new = (anchor - (anchor - low) * factor, anchor + (high - anchor) * factor)
        if new[1] - new[0] <= 0 or not all(map(math.isfinite, new)):
            return
        if which == "x":
            ax.set_xlim(*new)
        else:
            ax.set_ylim(*new)
        if self.on_change:
            self.on_change(ax)
        self.canvas.draw_idle()

    def _release(self, _event) -> None:
        if self._drag is not None:
            self._drag = None
            self.dragging = False
