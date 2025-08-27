# ========================================
# FILE: src/vgr_sdk/tasks/cnc_tending_task.py
# ========================================
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple

from vgr_sdk.tasks.base_task import BaseTask
from vgr_sdk.core.message_types import VisionResult
from vgr_sdk.core.planner import Plan


@dataclass
class CNCTendingConfig:
    """
    CNC tending: pick a part from tray/table, place into a fixture/machine pose.

    - place_xyzyaw : Cartesian pose where the part should be inserted
    - align_theta  : if True, align part yaw with place yaw (uses detected yaw)
    - theta_tolerance_deg : allowed absolute delta between detected yaw and place yaw (when align_theta)
    - scale_tolerance     : acceptable scale drift (fraction). e.g., 0.2 -> ±20% vs nominal
    """
    place_xyzyaw: Tuple[float, float, float, float]
    align_theta: bool = True
    theta_tolerance_deg: float = 7.5
    scale_tolerance: float = 0.25  # 25%


class CNCTendingTask(BaseTask):
    """
    - Filters detections by yaw/scale tolerance (if enabled).
    - Builds a pick→place plan to the configured machine pose.
    """

    def __init__(self, *, config: CNCTendingConfig, **kwargs):
        super().__init__(**kwargs)
        self.config = config

    def _yaw_ok(self, det: Dict[str, Any]) -> bool:
        if not self.config.align_theta:
            return True
        yaw_det = float(det["pose_world"]["yaw_deg"])
        yaw_target = float(self.config.place_xyzyaw[3])
        return abs((yaw_det - yaw_target + 180.0) % 360.0 - 180.0) <= self.config.theta_tolerance_deg

    def _scale_ok(self, det: Dict[str, Any]) -> bool:
        p = det.get("pose", {})
        xs = float(p.get("x_scale", 1.0))
        ys = float(p.get("y_scale", 1.0))
        tol = float(self.config.scale_tolerance)
        return (1.0 - tol) <= xs <= (1.0 + tol) and (1.0 - tol) <= ys <= (1.0 + tol)

    def select_candidate(self, enriched: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        for d in enriched:
            if self._yaw_ok(d) and self._scale_ok(d):
                return d
        return None

    def plan_cycle(self, vr: VisionResult) -> Tuple[Optional[Plan], Optional[Dict[str, Any]]]:
        enriched = self.to_world_and_enrich(vr)
        if not enriched:
            return None, None

        pick = self.select_candidate(enriched)
        if pick is None:
            return None, None

        # Place yaw: either keep configured yaw or align exactly with detection yaw
        place_x, place_y, place_z, place_yaw = self.config.place_xyzyaw
        if self.config.align_theta:
            place_yaw = float(pick["pose_world"]["yaw_deg"])

        plan = self.build_pick_place(pick, (place_x, place_y, place_z, place_yaw))
        return plan, pick
