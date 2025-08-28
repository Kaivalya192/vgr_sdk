# ======================================================
# FILE: src/vgr_sdk/drivers/gripper_addverb_backend.py
# ======================================================
from __future__ import annotations
import rospy
from typing import Dict

# Your action-style goal messages
from addverb_cobot_msgs.msg import GraspActionGoal, ReleaseActionGoal


class Backend:
    """
    Backend for scripts/gripper_node.py (~driver_module param) that maps the SDK
    API to Addverb cobot grasp/release goal topics.

    Mapping:
      - open_mm(width)     -> publish ReleaseActionGoal (width is ignored)
      - close_mm(width)    -> publish GraspActionGoal with a force derived from width
      - close_force(force) -> publish GraspActionGoal with provided force
      - get_status()       -> best-effort stub (no feedback topics wired here)

    ROS Params (private, ~):
      ~topic_grasp_goal   : /robotA/grasp_action/goal
      ~topic_release_goal : /robotA/release_action/goal
      ~min_force_n        : 20.0
      ~max_force_n        : 120.0
      ~stroke_min_mm      : set by gripper_node
      ~stroke_max_mm      : set by gripper_node
    """

    def __init__(self, stroke_min_mm: float, stroke_max_mm: float):
        self.stroke_min_mm = float(stroke_min_mm)
        self.stroke_max_mm = float(stroke_max_mm)

        self.topic_grasp = rospy.get_param("~topic_grasp_goal", "/robotA/grasp_action/goal")
        self.topic_release = rospy.get_param("~topic_release_goal", "/robotA/release_action/goal")
        self.min_force = float(rospy.get_param("~min_force_n", 20.0))
        self.max_force = float(rospy.get_param("~max_force_n", 120.0))

        self.pub_grasp = rospy.Publisher(self.topic_grasp, GraspActionGoal, queue_size=10)
        self.pub_release = rospy.Publisher(self.topic_release, ReleaseActionGoal, queue_size=10)

        # internal state (no feedback wired; keep last commanded)
        self._opening_mm = self.stroke_max_mm
        self._object = False

    # ---------- helpers ----------
    def _width_to_force(self, opening_mm: float) -> float:
        """
        Simple heuristic: narrower target -> higher force within [min,max].
        You can replace this with a calibrated mapping or table.
        """
        span = max(1e-6, self.stroke_max_mm - self.stroke_min_mm)
        alpha = 1.0 - (max(self.stroke_min_mm, min(opening_mm, self.stroke_max_mm)) - self.stroke_min_mm) / span
        return self.min_force + alpha * (self.max_force - self.min_force)

    # ---------- API ----------
    def open_mm(self, opening_mm: float) -> None:
        # Width is ignored by the release action; we store it for status
        self._opening_mm = float(opening_mm)
        msg = ReleaseActionGoal()
        self.pub_release.publish(msg)

    def close_mm(self, opening_mm: float) -> None:
        self._opening_mm = float(opening_mm)
        force = self._width_to_force(opening_mm)
        msg = GraspActionGoal()
        msg.goal.grasp_force = float(force)
        self.pub_grasp.publish(msg)
        self._object = True  # optimistic

    def close_force(self, force_n: float) -> None:
        msg = GraspActionGoal()
        msg.goal.grasp_force = float(force_n)
        self.pub_grasp.publish(msg)
        self._object = True

    def get_status(self) -> Dict:
        # If you have feedback topics, wire them and set these accordingly.
        return {
            "opening_mm": float(self._opening_mm),
            "object_detected": bool(self._object),
            "ok": True,
            "error": "",
        }
