# ================================
# FILE: src/vgr_sdk/core/gripper.py
# ================================
from __future__ import annotations
from typing import Iterable, Tuple, Dict, Any, List
import math
import numpy as np

# Pure math helpers for grasp sizing from a detected shape.
# No ROS or intra-project imports.


def quad_minor_major_mm(quad_world_mm: Iterable[Iterable[float]]) -> Tuple[float, float]:
    """
    Given a 4-point quad in world mm (each [X,Y,Z]), estimate (minor, major) extents (mm).
    Uses average of opposing edge lengths as minor/major heuristics.
    """
    pts = np.asarray(quad_world_mm, dtype=np.float64)[:, :2]
    if pts.shape != (4, 2):
        return 0.0, 0.0
    # edges: 0-1, 1-2, 2-3, 3-0
    e01 = np.linalg.norm(pts[1] - pts[0])
    e12 = np.linalg.norm(pts[2] - pts[1])
    e23 = np.linalg.norm(pts[3] - pts[2])
    e30 = np.linalg.norm(pts[0] - pts[3])
    a = 0.5 * (e01 + e23)
    b = 0.5 * (e12 + e30)
    minor = min(a, b)
    major = max(a, b)
    return float(minor), float(major)


def recommend_opening_mm(
    minor_extent_mm: float,
    *,
    clearance_mm: float,
    stroke_min_mm: float,
    stroke_max_mm: float,
) -> float:
    """
    Compute recommended jaw opening from minor extent + clearance, clamped to stroke.
    """
    opening = minor_extent_mm + float(clearance_mm)
    opening = max(stroke_min_mm, min(stroke_max_mm, opening))
    return float(opening)


def grasp_axis_deg_from_quad_world(quad_world_mm: Iterable[Iterable[float]]) -> float:
    """
    Estimate grasp axis (deg) from world quad: take the minor axis direction.
    Returns angle CCW from +X in world.
    """
    pts = np.asarray(quad_world_mm, dtype=np.float64)[:, :2]
    if pts.shape != (4, 2):
        return 0.0
    # choose the two opposite edges corresponding to minor dimension
    e01 = pts[1] - pts[0]; l01 = np.linalg.norm(e01)
    e12 = pts[2] - pts[1]; l12 = np.linalg.norm(e12)
    e_axis = e01 if l01 < l12 else e12  # vector along minor axis edge
    ang = math.degrees(math.atan2(e_axis[1], e_axis[0]))
    # Grasp axis perpendicular to minor edge (close across minor dimension)
    return (ang + 90.0 + 360.0) % 360.0


def symmetric_grasp_points_world(
    center_world_mm: Iterable[float],
    axis_deg: float,
    jaw_distance_mm: float,
) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    """
    Return two symmetric points around center along the grasp axis with separation `jaw_distance_mm`.
    """
    cx, cy, cz = float(center_world_mm[0]), float(center_world_mm[1]), float(center_world_mm[2])
    half = 0.5 * float(jaw_distance_mm)
    ang = math.radians(float(axis_deg))
    dx, dy = math.cos(ang), math.sin(ang)
    p1 = (cx - half * dx, cy - half * dy, cz)
    p2 = (cx + half * dx, cy + half * dy, cz)
    return p1, p2
