# ===============================
# FILE: src/vgr_sdk/core/utils.py
# ===============================
from __future__ import annotations
import math
import time
from dataclasses import dataclass
from typing import Iterable, Tuple, List


def clamp(v: float, lo: float, hi: float) -> float:
    """Clamp v to [lo, hi]."""
    return max(lo, min(hi, v))


def deg2rad(d: float) -> float:
    return d * math.pi / 180.0


def rad2deg(r: float) -> float:
    return r * 180.0 / math.pi


def ang_norm_deg(a: float) -> float:
    """Normalize degrees to (-180, 180]."""
    a = (a + 180.0) % 360.0 - 180.0
    return a


@dataclass
class SimpleTimer:
    """Convenient timing helper (ms)."""
    t0: float = time.perf_counter()

    def reset(self) -> None:
        self.t0 = time.perf_counter()

    def ms(self) -> float:
        return (time.perf_counter() - self.t0) * 1000.0


def polygon_area(points: Iterable[Tuple[float, float]]) -> float:
    """Signed area; positive if points are CCW."""
    pts = list(points)
    if len(pts) < 3:
        return 0.0
    s = 0.0
    for (x1, y1), (x2, y2) in zip(pts, pts[1:] + pts[:1]):
        s += x1 * y2 - x2 * y1
    return 0.5 * s


def poly_centroid(points: Iterable[Tuple[float, float]]) -> Tuple[float, float]:
    """Centroid of polygon (non-self-intersecting)."""
    pts = list(points)
    a = polygon_area(pts)
    if abs(a) < 1e-12:
        # fallback to average
        xs = sum(p[0] for p in pts) / len(pts)
        ys = sum(p[1] for p in pts) / len(pts)
        return xs, ys
    cx = 0.0
    cy = 0.0
    for (x1, y1), (x2, y2) in zip(pts, pts[1:] + pts[:1]):
        w = x1 * y2 - x2 * y1
        cx += (x1 + x2) * w
        cy += (y1 + y2) * w
    cx /= (6.0 * a)
    cy /= (6.0 * a)
    return cx, cy


def pca_axis(points: List[Tuple[float, float]]) -> Tuple[float, float]:
    """
    Return (major_axis_deg, minor_axis_deg) of 2D points via PCA.
    Angles measured in degrees CCW from +X.
    """
    import numpy as np

    if len(points) < 2:
        return 0.0, 90.0
    pts = np.asarray(points, dtype=np.float64)
    c = pts.mean(axis=0, keepdims=True)
    X = pts - c
    cov = X.T @ X / max(1, len(points) - 1)
    w, v = np.linalg.eigh(cov)  # ascending
    major = v[:, 1]  # eigenvector with largest eigenvalue
    ang = math.degrees(math.atan2(major[1], major[0]))
    return ang, ang_norm_deg(ang + 90.0)
