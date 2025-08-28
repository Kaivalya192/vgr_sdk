#!/usr/bin/env python3
# =================================
# FILE: scripts/plan_executor_node.py
# =================================
from __future__ import annotations
import json
import rospy
from std_msgs.msg import String
from typing import Dict, Any, Tuple

# Robot + gripper backends
from vgr_sdk.drivers.robot_velocity_backend import Backend as RobotBackend
from vgr_sdk.core.robot_io import GripperInterface  # type protocol

# Reuse the existing gripper_node as a separate process, or import a backend directly.
# Here we talk to your /vgr_gripper topics via JSON command for simplicity.
from std_msgs.msg import Float64


class PlanExecutorNode:
    """
    Subscribes to /vgr/plan (JSON) produced by vgr_manager_node.py and executes
    the waypoint list using RobotBackend. Triggers the gripper at GRASP/RELEASE.

    ROS Params (~):
      ~sub_plan_topic      : /vgr/plan
      ~gripper_ns          : /vgr_gripper
      ~gripper_open_mm     : 40.0   (default open width at RELEASE if pick doesn't provide it)
      ~gripper_close_fallback_mm : 20.0 (if pick.grasp.recommended_open_mm missing)
    """

    def __init__(self):
        self.sub_plan_topic = rospy.get_param("~sub_plan_topic", "/vgr/plan")
        self.gripper_ns = rospy.get_param("~gripper_ns", "/vgr_gripper")
        self.default_open_mm = float(rospy.get_param("~gripper_open_mm", 40.0))
        self.fallback_close_mm = float(rospy.get_param("~gripper_close_fallback_mm", 20.0))

        # Robot backend (KDL IK + velocity control)
        self.robot = RobotBackend()

        # Gripper topics (talk to scripts/gripper_node.py)
        self.pub_open = rospy.Publisher(f"{self.gripper_ns}/open_mm", Float64, queue_size=10)
        self.pub_close = rospy.Publisher(f"{self.gripper_ns}/close_mm", Float64, queue_size=10)

        rospy.Subscriber(self.sub_plan_topic, String, self._on_plan, queue_size=3)
        rospy.loginfo(f"[plan_executor] listening on {self.sub_plan_topic}")

    # ---------- callbacks ----------
    def _on_plan(self, msg: String):
        try:
            payload = json.loads(msg.data)
        except Exception as e:
            rospy.logwarn(f"[plan_executor] bad plan JSON: {e}")
            return

        plan = payload.get("plan", {})
        pick = payload.get("pick", {})
        waypoints = plan.get("waypoints", [])
        if not waypoints:
            return

        # Resolve gripper closing width
        close_mm = self._pick_close_width(pick)
        open_mm = self.default_open_mm

        # Execute
        try:
            for wp in waypoints:
                tag = str(wp.get("tag", "")).upper()
                mode = str(wp.get("mode", "LINEAR")).upper()
                x = float(wp.get("x_mm", 0.0))
                y = float(wp.get("y_mm", 0.0))
                z = float(wp.get("z_mm", 0.0))
                yaw = float(wp.get("yaw_deg", 0.0))
                speed = float(wp.get("speed_mm_s", 200.0))

                if tag == "GRASP":
                    # Close gripper BEFORE moving on if desired; many cells close at contact height
                    self._gripper_close(close_mm)

                elif tag == "RELEASE":
                    self._gripper_open(open_mm)

                # Move between tags (we treat JOINT as a hint; we still target Cartesian pose with IK)
                self.robot.move_linear(x, y, z, yaw, speed=speed)
                # Wait until done
                while self.robot.is_busy() and not rospy.is_shutdown():
                    rospy.sleep(0.01)

        except Exception as e:
            rospy.logerr(f"[plan_executor] execution error: {e}")
            self.robot.stop()

    # ---------- helpers ----------
    def _pick_close_width(self, pick: Dict[str, Any]) -> float:
        try:
            g = pick.get("grasp", {})
            val = float(g.get("recommended_open_mm", self.fallback_close_mm))
            return val
        except Exception:
            return self.fallback_close_mm

    def _gripper_close(self, width_mm: float):
        self.pub_close.publish(Float64(data=float(width_mm)))

    def _gripper_open(self, width_mm: float):
        self.pub_open.publish(Float64(data=float(width_mm)))


def main():
    rospy.init_node("vgr_plan_executor", anonymous=False)
    PlanExecutorNode()
    rospy.spin()


if __name__ == "__main__":
    main()
