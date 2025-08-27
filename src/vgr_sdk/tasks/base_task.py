# ==================================
# FILE: src/vgr_sdk/tasks/base_task.py
# ==================================
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple

from vgr_sdk.core.transform import GeomConfig, detection_img_to_world, extent_px_to_mm
from vgr_sdk.core.gripper import (
    quad_minor_major_mm,
    recommend_opening_mm,
    grasp_axis_deg_from_quad_world,
    symmetric_grasp_points_world,
)
from vgr_sdk.core.planner import Plan, PickPlaceParams, build_pick_place_plan
from vgr_sdk.core.pose_store import PoseStore
from vgr_sdk.core.message_types import VisionResult
from vgr_sdk.core.filters import gate_detections, sort_by_quality, dedupe_by_center_px


@dataclass
class GripperConfig:
    stroke_min_mm: float = 5.0
    stroke_max_mm: float = 80.0
    clearance_mm: float = 3.0


@dataclass
class QualityGates:
    min_inliers: int = 10
    min_score: float = 0.25
    min_center_dist_px: float = 40.0


class BaseTask:
    """
    Task-agnostic helpers:
    - Transform detections to world
    - Add grasp hints (opening, axis, points)
    - Build canonical pick→place plans

    Subclasses implement selection and target placement logic.
    """

    def __init__(
        self,
        *,
        geom: GeomConfig,
        pose_store: Optional[PoseStore] = None,
        gripper_cfg: Optional[GripperConfig] = None,
        gates: Optional[QualityGates] = None,
        plan_params: Optional[PickPlaceParams] = None,
    ):
        self.geom = geom
        self.pose_store = pose_store
        self.gripper_cfg = gripper_cfg or GripperConfig()
        self.gates = gates or QualityGates()
        self.plan_params = plan_params or PickPlaceParams()

    # ---------- detection processing ----------
    def to_world_and_enrich(self, vr: VisionResult) -> List[Dict[str, Any]]:
        """
        Flatten the VisionResult and enrich each detection with:
        - center_world_mm, quad_world_mm, pose_world
        - grasp: {minor_mm, major_mm, opening_mm, axis_deg, points_world[[X,Y,Z]*2]}
        """
        flat = vr.flat_detections()
        # per-object de-duplication in image space (already done by producer usually)
        flat = dedupe_by_center_px(flat, min_center_dist_px=self.gates.min_center_dist_px)
        flat = gate_detections(flat, min_inliers=self.gates.min_inliers, min_score=self.gates.min_score)
        # sort by quality for convenience
        flat = sort_by_quality(flat)

        enriched: List[Dict[str, Any]] = []
        for d in flat:
            # add world fields (pose_world, center_world_mm, quad_world_mm)
            dw = detection_img_to_world(d, self.geom)

            # compute grasp hints
            minor_mm, major_mm = 0.0, 0.0
            if dw.get("quad_world_mm"):
                minor_mm, major_mm = quad_minor_major_mm(dw["quad_world_mm"])
            else:
                # fallback: template size * affine scale * mm_per_px
                tpl = d.get("template_size")
                pose = d.get("pose", {})
                if tpl and len(tpl) == 2:
                    w_px, h_px = float(tpl[0]) * float(pose.get("x_scale", 1.0)), float(tpl[1]) * float(pose.get("y_scale", 1.0))
                    w_mm, h_mm = extent_px_to_mm(w_px, h_px, self.geom)
                    minor_mm, major_mm = (min(w_mm, h_mm), max(w_mm, h_mm))

            opening = recommend_opening_mm(
                minor_mm,
                clearance_mm=self.gripper_cfg.clearance_mm,
                stroke_min_mm=self.gripper_cfg.stroke_min_mm,
                stroke_max_mm=self.gripper_cfg.stroke_max_mm,
            )
            axis_deg = 0.0
            if dw.get("quad_world_mm"):
                axis_deg = grasp_axis_deg_from_quad_world(dw["quad_world_mm"])
            else:
                # fallback: use world yaw
                axis_deg = float(dw["pose_world"]["yaw_deg"])

            points_world = None
            if dw.get("center_world_mm"):
                points_world = list(
                    symmetric_grasp_points_world(dw["center_world_mm"], axis_deg, opening)
                )

            dw["grasp"] = {
                "minor_extent_mm": minor_mm,
                "major_extent_mm": major_mm,
                "recommended_open_mm": opening,
                "axis_deg": axis_deg,
                "points_world": points_world,
                "clearance_mm": self.gripper_cfg.clearance_mm,
            }
            enriched.append(dw)

        return enriched

    # ---------- planning ----------
    def build_pick_place(
        self,
        pick_det_world: Dict[str, Any],
        place_xyzyaw: Tuple[float, float, float, float],
        *,
        use_joint_transit: bool = False,
    ) -> Plan:
        """Create a canonical pick→place plan from a detection and a target place pose."""
        pw = pick_det_world["pose_world"]
        pick_xyzyaw = (float(pw["x_mm"]), float(pw["y_mm"]), float(pw["z_mm"]), float(pw["yaw_deg"]))
        plan = build_pick_place_plan(
            pick_xyzyaw=pick_xyzyaw,
            place_xyzyaw=place_xyzyaw,
            params=self.plan_params,
            use_joint_transit=use_joint_transit,
        )
        return plan

    # ---------- helpers ----------
    def resolve_pose_from_store(self, name: str) -> Optional[List[float]]:
        """
        Return a joint pose by name if PoseStore is present (else None).
        """
        if not self.pose_store:
            return None
        pe = self.pose_store.get(name)
        return list(pe.joints) if pe else None
