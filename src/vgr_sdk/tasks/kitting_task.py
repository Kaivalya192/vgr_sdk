# =======================================
# FILE: src/vgr_sdk/tasks/kitting_task.py
# =======================================
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple

from vgr_sdk.tasks.base_task import BaseTask
from vgr_sdk.core.message_types import VisionResult
from vgr_sdk.core.planner import Plan


@dataclass
class KittingConfig:
    """
    Kitting with named objects → ordered slot poses.

    - slots_by_name: dict of object name → list of slot poses (x,y,z,yaw)
                     (order defines the fill order)
    - allow_fallback: if True and the next slot list is exhausted, use the last slot again
                      (useful for demos); otherwise skip if full.
    """
    slots_by_name: Dict[str, List[Tuple[float, float, float, float]]]
    allow_fallback: bool = False


class KittingTask(BaseTask):
    """
    Keeps a simple per-object pointer for the next slot to fill.

    Usage:
      - Construct once and keep the instance alive for the job (so cursors persist).
      - Each call to `plan_cycle(vr)` tries to find a detection whose name has remaining slots.
    """

    def __init__(self, *, config: KittingConfig, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self._slot_idx: Dict[str, int] = {name: 0 for name in self.config.slots_by_name.keys()}

    def _next_slot_pose(self, name: str) -> Optional[Tuple[float, float, float, float]]:
        if name not in self.config.slots_by_name:
            return None
        idx = self._slot_idx.get(name, 0)
        slots = self.config.slots_by_name[name]
        if idx < len(slots):
            pose = slots[idx]
            self._slot_idx[name] = idx + 1
            return pose
        if self.config.allow_fallback and slots:
            # reuse last slot pose
            return slots[-1]
        return None

    def select_candidate(self, enriched: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Return first detection that still has an available slot."""
        for d in enriched:
            name = str(d.get("name", ""))
            if name in self.config.slots_by_name:
                # peek if a slot exists
                idx = self._slot_idx.get(name, 0)
                if idx < len(self.config.slots_by_name[name]) or self.config.allow_fallback:
                    return d
        return None

    def plan_cycle(self, vr: VisionResult) -> Tuple[Optional[Plan], Optional[Dict[str, Any]]]:
        enriched = self.to_world_and_enrich(vr)
        if not enriched:
            return None, None

        pick = self.select_candidate(enriched)
        if pick is None:
            return None, None

        name = str(pick.get("name", ""))
        place_xyzyaw = self._next_slot_pose(name)
        if not place_xyzyaw:
            return None, None

        plan = self.build_pick_place(pick, place_xyzyaw)
        return plan, pick
