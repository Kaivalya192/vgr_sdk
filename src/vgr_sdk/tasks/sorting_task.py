# ======================================
# FILE: src/vgr_sdk/tasks/sorting_task.py
# ======================================
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple

from vgr_sdk.tasks.base_task import BaseTask
from vgr_sdk.core.message_types import VisionResult
from vgr_sdk.core.planner import Plan


@dataclass
class SortingConfig:
    """
    Mapping from object 'name' (as emitted by Vision) to a placement target.
    You can provide either:
      - fixed world pose (x_mm, y_mm, z_mm, yaw_deg), OR
      - a recorded joint pose name (resolved by your ROS executor), OR
      - a hybrid: place_xyzyaw for cartesian, with pre/post joint waypoints handled elsewhere.

    This class carries only the cartesian part. If you want joint targets, call
    BaseTask.resolve_pose_from_store(name) at the node/executor layer.
    """
    # cartesian drop poses per object name
    place_xyzyaw_by_name: Dict[str, Tuple[float, float, float, float]]
    # optional: default pose if name not found
    default_place_xyzyaw: Optional[Tuple[float, float, float, float]] = None


class SortingTask(BaseTask):
    """
    Select the best detection whose object name exists in the sorting map,
    then build a pick→place plan to the configured drop pose.
    """

    def __init__(self, *, config: SortingConfig, **kwargs):
        super().__init__(**kwargs)
        self.config = config

    def select_candidate(self, enriched: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        Choose the highest-quality detection whose name has a destination.
        `enriched` are items from BaseTask.to_world_and_enrich (flat dicts).
        """
        for d in enriched:
            name = str(d.get("name", ""))
            if name in self.config.place_xyzyaw_by_name:
                return d
        # fallback to default target if provided
        if self.config.default_place_xyzyaw and enriched:
            return enriched[0]
        return None

    def plan_cycle(self, vr: VisionResult) -> Tuple[Optional[Plan], Optional[Dict[str, Any]]]:
        """
        Convert VisionResult → enriched detections → select → plan.
        Returns (Plan or None, chosen_detection or None)
        """
        enriched = self.to_world_and_enrich(vr)
        if not enriched:
            return None, None

        pick = self.select_candidate(enriched)
        if pick is None:
            return None, None

        name = str(pick.get("name", ""))
        place_xyzyaw = self.config.place_xyzyaw_by_name.get(name, self.config.default_place_xyzyaw)
        if not place_xyzyaw:
            return None, None

        plan = self.build_pick_place(pick, place_xyzyaw)
        return plan, pick
