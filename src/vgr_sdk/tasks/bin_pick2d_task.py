# =========================================
# FILE: src/vgr_sdk/tasks/bin_pick2d_task.py
# =========================================
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple
import math

from vgr_sdk.tasks.base_task import BaseTask
from vgr_sdk.core.message_types import VisionResult
from vgr_sdk.core.planner import Plan
from vgr_sdk.core.utils import polygon_area


@dataclass
class BinPick2DConfig:
    """
    Top-layer 2D bin picking with a fixed plane and wall clearance.

    - bin_polygon_world_mm : list of [X,Y] vertices (CCW recommended)
    - wall_clearance_mm    : required distance from detection center to any wall
    - place_xyzyaw         : drop pose (cartesian)
    """
    bin_polygon_world_mm: List[Tuple[float, float]]
    wall_clearance_mm: float
    place_xyzyaw: Tuple[float, float, float, float]


def _point_in_polygon(x: float, y: float, poly: List[Tuple[float, float]]) -> bool:
    """Even-odd rule (works for convex/concave, non-self-intersecting)."""
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        # check if ray intersects edge
        if (y1 > y) != (y2 > y):
            xin = x1 + (y - y1) * (x2 - x1) / (y2 - y1 + 1e-12)
            if xin > x:
                inside = not inside
    return inside


def _min_dist_to_edges(x: float, y: float, poly: List[Tuple[float, float]]) -> float:
    """Minimum Euclidean distance from point to polygon edges (2D)."""
    mind = float("inf")
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        # segment distance
        vx, vy = x2 - x1, y2 - y1
        wx, wy = x - x1, y - y1
        seg_len2 = vx * vx + vy * vy
        t = 0.0 if seg_len2 < 1e-12 else max(0.0, min(1.0, (wx * vx + wy * vy) / seg_len2))
        projx, projy = x1 + t * vx, y1 + t * vy
        d = math.hypot(x - projx, y - projy)
        if d < mind:
            mind = d
    return mind


class BinPick2DTask(BaseTask):
    """
    - Keeps detections whose center is inside the bin polygon AND at least wall_clearance away from edges.
    - Builds a pick→place plan to a configured drop pose.
    """

    def __init__(self, *, config: BinPick2DConfig, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        # sanity: polygon orientation isn't strictly required
        if len(self.config.bin_polygon_world_mm) < 3:
            raise ValueError("bin_polygon_world_mm must have >=3 points")

    def _inside_with_clearance(self, det: Dict[str, Any]) -> bool:
        c = det.get("center_world_mm")
        if not c:
            return False
        x, y = float(c[0]), float(c[1])
        poly = [(float(px), float(py)) for px, py in self.config.bin_polygon_world_mm]
        if not _point_in_polygon(x, y, poly):
            return False
        dmin = _min_dist_to_edges(x, y, poly)
        return dmin >= float(self.config.wall_clearance_mm)

    def select_candidate(self, enriched: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        for d in enriched:
            if self._inside_with_clearance(d):
                return d
        return None

    def plan_cycle(self, vr: VisionResult) -> Tuple[Optional[Plan], Optional[Dict[str, Any]]]:
        enriched = self.to_world_and_enrich(vr)
        if not enriched:
            return None, None
        pick = self.select_candidate(enriched)
        if pick is None:
            return None, None
        plan = self.build_pick_place(pick, self.config.place_xyzyaw)
        return plan, pick
