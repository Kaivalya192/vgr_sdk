# =================================
# FILE: src/vgr_sdk/core/planner.py
# =================================
"""
Minimal, transport-agnostic motion planning primitives for VGR V1.

This module *does not* talk to ROS or a specific robot. It just creates
high-level waypoint plans in world units (mm / deg) that your robot-side
driver can consume and turn into real trajectories.

Waypoints are tagged, so your executor can:
- inject gripper close/open at "GRASP"/"RELEASE"
- map LINEAR vs JOINT segments to appropriate controllers
- apply speed/accel presets per-segment
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Literal, Dict


MoveMode = Literal["JOINT", "LINEAR"]
Tag = Literal[
    "APPROACH", "DESCEND", "GRASP", "LIFT", "TRANSIT", "PREPLACE",
    "RELEASE", "RETREAT", "HOME", "CUSTOM"
]


@dataclass
class Waypoint:
    """Single Cartesian waypoint in world coordinates (mm/deg)."""
    x_mm: float
    y_mm: float
    z_mm: float
    yaw_deg: float
    mode: MoveMode = "LINEAR"
    speed_mm_s: float = 200.0
    tag: Tag = "CUSTOM"
    note: str = ""


@dataclass
class Plan:
    """A simple ordered list of waypoints with optional metadata."""
    waypoints: List[Waypoint] = field(default_factory=list)
    meta: Dict[str, float | int | str] = field(default_factory=dict)

    def add(self, wp: Waypoint) -> None:
        self.waypoints.append(wp)

    def extend(self, wps: List[Waypoint]) -> None:
        self.waypoints.extend(wps)

    def to_dict(self) -> Dict:
        return {
            "meta": dict(self.meta),
            "waypoints": [
                {
                    "x_mm": w.x_mm, "y_mm": w.y_mm, "z_mm": w.z_mm, "yaw_deg": w.yaw_deg,
                    "mode": w.mode, "speed_mm_s": w.speed_mm_s, "tag": w.tag, "note": w.note
                }
                for w in self.waypoints
            ],
        }


@dataclass
class PickPlaceParams:
    """Reusable heights/speeds for VGR pick/place cycles."""
    approach_above_mm: float = 60.0   # how high above pick/place to approach
    lift_after_pick_mm: float = 80.0  # lift height after grasp
    place_above_mm: float = 60.0      # pre-place height above target
    pick_speed_mm_s: float = 250.0
    transit_speed_mm_s: float = 350.0
    place_speed_mm_s: float = 250.0
    joint_transit_speed_mm_s: float = 9999.0  # informational; real joint speed is driver-specific


def build_pick_place_plan(
    *,
    pick_xyzyaw: tuple[float, float, float, float],
    place_xyzyaw: tuple[float, float, float, float],
    params: Optional[PickPlaceParams] = None,
    use_joint_transit: bool = False,
) -> Plan:
    """
    Create a canonical pick→place plan made of tagged waypoints:

    APPROACH(pick_above) → DESCEND(pick) → GRASP → LIFT(pick_above)
      → TRANSIT(place_above) → PREPLACE(place_above) → DESCEND(place)
      → RELEASE → RETREAT(place_above)

    Args:
        pick_xyzyaw  : (x_mm, y_mm, z_mm, yaw_deg) target for grasp
        place_xyzyaw : (x_mm, y_mm, z_mm, yaw_deg) target for placement
        params       : PickPlaceParams with heights/speeds
        use_joint_transit : if True, the long transit to place_above uses JOINT mode

    Returns:
        Plan with Waypoints. Your executor should trigger gripper actions at GRASP/RELEASE.
    """
    if params is None:
        params = PickPlaceParams()

    px, py, pz, pyaw = pick_xyzyaw
    qx, qy, qz, qyaw = place_xyzyaw

    plan = Plan(meta={"type": "pick_place"})

    # ---- Pick approach & descend
    pick_above = Waypoint(px, py, pz + params.approach_above_mm, pyaw,
                          mode="LINEAR", speed_mm_s=params.transit_speed_mm_s, tag="APPROACH")
    pick_pose  = Waypoint(px, py, pz, pyaw,
                          mode="LINEAR", speed_mm_s=params.pick_speed_mm_s, tag="DESCEND")
    plan.extend([pick_above, pick_pose])

    # ---- Grasp (marker)
    plan.add(Waypoint(px, py, pz, pyaw, mode="LINEAR", speed_mm_s=0.0, tag="GRASP", note="close gripper"))

    # ---- Lift
    lift_pose = Waypoint(px, py, pz + params.lift_after_pick_mm, pyaw,
                         mode="LINEAR", speed_mm_s=params.pick_speed_mm_s, tag="LIFT")
    plan.add(lift_pose)

    # ---- Transit to place_above
    place_above = Waypoint(qx, qy, qz + params.place_above_mm, qyaw,
                           mode="JOINT" if use_joint_transit else "LINEAR",
                           speed_mm_s=params.transit_speed_mm_s, tag="TRANSIT")
    plan.add(place_above)

    # ---- Optional pre-place hover (kept for symmetry & timing hooks)
    plan.add(Waypoint(qx, qy, qz + params.place_above_mm, qyaw,
                      mode="LINEAR", speed_mm_s=params.place_speed_mm_s, tag="PREPLACE"))

    # ---- Descend to place
    place_pose = Waypoint(qx, qy, qz, qyaw,
                          mode="LINEAR", speed_mm_s=params.place_speed_mm_s, tag="DESCEND")
    plan.add(place_pose)

    # ---- Release (marker)
    plan.add(Waypoint(qx, qy, qz, qyaw, mode="LINEAR", speed_mm_s=0.0,
                      tag="RELEASE", note="open gripper"))

    # ---- Retreat
    retreat = Waypoint(qx, qy, qz + params.place_above_mm, qyaw,
                       mode="LINEAR", speed_mm_s=params.transit_speed_mm_s, tag="RETREAT")
    plan.add(retreat)

    return plan
