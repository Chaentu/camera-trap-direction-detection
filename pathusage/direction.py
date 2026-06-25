"""Direction-of-movement estimation.

The :class:`VirtualGrid` overlays an ``n_cols x n_rows`` tripwire grid on the
camera frame. For each frame in a sequence the grid column occupied by the
dominant subject centroid is recorded; the net column crossing gives the
direction, the Euclidean pixel displacement over time gives a speed estimate,
and a coarse speed threshold gives an activity hint.

If the grid sees no column crossing, a centroid-trajectory fallback uses the
net first-to-last horizontal (or vertical) displacement instead.
"""

from __future__ import annotations


def calib_sign(calib):
    """+1 if the inbound direction increases along the calibrated axis, else -1."""
    return 1 if calib["in_sign"] == "+" else -1


class VirtualGrid:
    """A tripwire grid for direction + speed estimation on one sequence."""

    def __init__(self, frame_w, frame_h, n_cols=5,
                 speed_slow=80, speed_fast=300):
        self.frame_w = frame_w
        self.frame_h = frame_h
        self.n_cols = n_cols
        self.speed_slow = speed_slow
        self.speed_fast = speed_fast
        # column boundaries at (i+1)/(n_cols+1) of the frame width
        self.col_th = [frame_w * (i + 1) / (n_cols + 1) for i in range(n_cols)]
        self.col_h, self.cx_h, self.cy_h, self.t_h = [], [], [], []

    def record(self, cx, cy, t_sec):
        """Record one detection: pixel centroid (cx, cy) and time offset (s)."""
        col = min(sum(cx > th for th in self.col_th), self.n_cols - 1)
        self.col_h.append(col)
        self.cx_h.append(cx)
        self.cy_h.append(cy)
        self.t_h.append(t_sec)

    def resolve(self, calib):
        """Return direction, pixel speed and activity hint from the grid."""
        n = len(self.col_h)
        if n < 2:
            return {"direction": "undefined", "speed_px_s": None,
                    "activity_hint": None, "n_points": n}
        col_disp = self.col_h[-1] - self.col_h[0]
        sign = calib_sign(calib)
        direction = ("undefined" if col_disp == 0 else
                     (calib["in_label"] if sign * col_disp > 0 else calib["out_label"]))
        dx = self.cx_h[-1] - self.cx_h[0]
        dy = self.cy_h[-1] - self.cy_h[0]
        dt = max(self.t_h[-1] - self.t_h[0], 1e-6)
        speed = (dx ** 2 + dy ** 2) ** 0.5 / dt
        hint = ("fast (MTB/ski)" if speed > self.speed_fast else
                "medium (ski/hiking)" if speed > self.speed_slow else
                "slow (snowshoe/hiking)")
        return {"direction": direction, "speed_px_s": round(speed, 2),
                "activity_hint": hint, "n_points": n}


def estimate_direction(seq_dets, dt_offsets, calib, frame_w, frame_h,
                       detect_conf=0.20, n_cols=5, min_move_px=15.0,
                       speed_slow=80, speed_fast=300):
    """Estimate the direction of a sequence (grid first, trajectory fallback).

    For each frame the highest-confidence person detection (category 2,
    confidence >= ``detect_conf``) is selected as the dominant subject
    centroid. The grid is resolved first; if it returns ``undefined`` the
    trajectory fallback is applied to the same centroids. Sequences without a
    detected person are left undefined by the caller.
    """
    grid = VirtualGrid(frame_w, frame_h, n_cols=n_cols,
                       speed_slow=speed_slow, speed_fast=speed_fast)
    for dets, t_off in zip(seq_dets, dt_offsets):
        persons = [d for d in dets if d["category"] == 2 and d["conf"] >= detect_conf]
        if not persons:
            continue
        best = max(persons, key=lambda d: d["conf"])
        grid.record(best["cx"], best["cy"], t_off)

    result = grid.resolve(calib)

    if result["direction"] == "undefined" and len(grid.cx_h) >= 2:
        dx = grid.cx_h[-1] - grid.cx_h[0]
        dy = grid.cy_h[-1] - grid.cy_h[0]
        if (dx ** 2 + dy ** 2) ** 0.5 >= min_move_px:
            disp = dx if calib["axis"] == "x" else dy
            sign = calib_sign(calib)
            result["direction"] = (calib["in_label"] if sign * disp > 0
                                   else calib["out_label"])
            result["fallback"] = True
    return result
