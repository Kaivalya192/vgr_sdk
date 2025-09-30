#!/usr/bin/env python3
# =================================
# FILE: scripts/plan_executor_node.py
# =================================
from __future__ import annotations
import json
import math
import rospy
from typing import Dict, Any, Optional
from std_msgs.msg import String

from vgr_sdk.drivers.robot_velocity_backend import Backend as RobotBackend
from addverb_cobot_msgs.msg import GraspActionGoal, ReleaseActionGoal


class PlanExecutorNode:
    def __init__(self):
        self.sub_plan_topic = rospy.get_param("~sub_plan_topic", "/vgr/plan")
        self.pub_state_topic = rospy.get_param("~pub_plan_state_topic", "/vgr/plan_state")

        self.backend_trans_units = str(rospy.get_param("~backend_trans_units", "m")).lower()
        self.backend_yaw_units   = str(rospy.get_param("~backend_yaw_units", "deg")).lower()

        self.topic_grasp_goal   = rospy.get_param("~topic_grasp_goal",   "/robotA/grasp_action/goal")
        self.topic_release_goal = rospy.get_param("~topic_release_goal", "/robotA/release_action/goal")

        self.default_grasp_force = float(rospy.get_param("~default_grasp_force_n", 100.0))
        self.post_grasp_wait_s   = float(rospy.get_param("~post_grasp_wait_s", 2.0))
        self.post_release_wait_s = float(rospy.get_param("~post_release_wait_s", 2.0))
        self.skip_zero_speed     = bool(rospy.get_param("~skip_zero_speed", True))

        self._pos_scale = 0.001 if self.backend_trans_units.startswith("m") else 1.0
        self._yaw_to_backend = (lambda d: d) if self.backend_yaw_units.startswith("deg") else (lambda d: math.radians(d))
        self._spd_scale = 0.001 if self.backend_trans_units.startswith("m") else 1.0

        self.robot = RobotBackend()
        self.pub_grasp   = rospy.Publisher(self.topic_grasp_goal,   GraspActionGoal,   queue_size=10)
        self.pub_release = rospy.Publisher(self.topic_release_goal, ReleaseActionGoal, queue_size=10)
        self.pub_state   = rospy.Publisher(self.pub_state_topic,    String,            queue_size=10)

        rospy.Subscriber(self.sub_plan_topic, String, self._on_plan, queue_size=3)

        rospy.loginfo("[plan_exec] Ready | listen=%s | units: trans=%s, yaw=%s | gripper(close)=%s | gripper(open)=%s",
                      self.sub_plan_topic, self.backend_trans_units, self.backend_yaw_units,
                      self.topic_grasp_goal, self.topic_release_goal)
        rospy.loginfo("[plan_exec] plan_state pub=%s", self.pub_state_topic)
        rospy.loginfo("[plan_exec] spinning")

        self._last_plan_payload: Optional[Dict[str, Any]] = None

    def _on_plan(self, msg: String):
        try:
            payload = json.loads(msg.data)
        except Exception as e:
            rospy.logwarn("[plan_exec] bad plan JSON: %s", e)
            return
        self._last_plan_payload = payload
        plan = payload.get("plan", {})
        waypoints = plan.get("waypoints", [])
        rospy.loginfo("[plan_exec] received plan with %d waypoints", len(waypoints))
        try:
            self.pub_state.publish(String(data="start"))
        except Exception:
            pass
        self.execute_latest()

    def execute_latest(self):
        if not self._last_plan_payload:
            return
        self._execute_payload(self._last_plan_payload)

    def _execute_payload(self, payload: Dict[str, Any]):
        plan = payload.get("plan", {})
        pick = payload.get("pick", {})
        waypoints = plan.get("waypoints", [])
        if not waypoints:
            try:
                self.pub_state.publish(String(data="done"))
            except Exception:
                pass
            return

        force_n = self._resolve_force(pick)
        rospy.loginfo("[plan_exec] executing %d waypoints (force=%.1fN)", len(waypoints), force_n)

        try:
            for idx, wp in enumerate(waypoints, start=1):
                tag   = str(wp.get("tag", "")).upper()
                mode  = str(wp.get("mode", "LINEAR")).upper()
                x_mm  = float(wp.get("x_mm", 0.0))
                y_mm  = float(wp.get("y_mm", 0.0))
                z_mm  = float(wp.get("z_mm", 0.0))
                yaw_d = float(wp.get("yaw_deg", 0.0))
                sp_mm = float(wp.get("speed_mm_s", 200.0))

                x_b   = x_mm * self._pos_scale
                y_b   = y_mm * self._pos_scale
                z_b   = z_mm * self._pos_scale
                yaw_b = self._yaw_to_backend(yaw_d)
                sp_b  = sp_mm * self._spd_scale

                should_move = True
                reason_skip = ""
                if tag in ("GRASP", "RELEASE"):
                    should_move = False
                    reason_skip = tag
                elif self.skip_zero_speed and abs(sp_mm) <= 1e-9:
                    should_move = False
                    reason_skip = "speed==0"

                label = f"{idx}/{len(waypoints)} {tag:9s}"
                if should_move:
                    u_lbl = "m/s" if self._pos_scale == 0.001 else "mm/s"
                    rospy.loginfo("[plan_exec] %3s -> x=%.3f y=%.3f z=%.3f yaw=%.3f  speed=%.3f %s",
                                  label, x_b, y_b, z_b, yaw_b, sp_b, u_lbl)
                else:
                    rospy.loginfo("[plan_exec] %3s (no motion: %s)", label, reason_skip)

                if tag == "GRASP":
                    self._gripper_close(force_n)
                    if self.post_grasp_wait_s > 0:
                        rospy.sleep(self.post_grasp_wait_s)
                    continue
                elif tag == "RELEASE":
                    self._gripper_open()
                    if self.post_release_wait_s > 0:
                        rospy.sleep(self.post_release_wait_s)
                    continue

                if should_move:
                    if mode == "LINEAR":
                        self.robot.move_linear(x_b, y_b, z_b, yaw_b, speed=sp_b)
                    else:
                        self.robot.move_linear(x_b, y_b, z_b, yaw_b, speed=sp_b)
                    while self.robot.is_busy() and not rospy.is_shutdown():
                        rospy.sleep(0.01)

            try:
                self.pub_state.publish(String(data="done"))
            except Exception:
                pass

        except Exception as e:
            rospy.logerr("[plan_exec] execution error: %s", e)
            try:
                self.robot.stop()
            except Exception:
                pass
            try:
                self.pub_state.publish(String(data="error"))
            except Exception:
                pass

    def _ensure_conn(self, pub, timeout=1.0):
        t0 = rospy.Time.now().to_sec()
        while pub.get_num_connections() == 0 and (rospy.Time.now().to_sec() - t0) < timeout and not rospy.is_shutdown():
            rospy.sleep(0.02)

    def _resolve_force(self, pick: Dict[str, Any]) -> float:
        try:
            g = pick.get("grasp", {})
            return float(g.get("force_n", self.default_grasp_force))
        except Exception:
            return self.default_grasp_force

    def _gripper_close(self, force_n: float):
        self._ensure_conn(self.pub_grasp, timeout=1.0)
        msg = GraspActionGoal()
        msg.goal.grasp_force = float(force_n)
        self.pub_grasp.publish(msg)
        rospy.loginfo("[plan_exec] GRASP -> GraspActionGoal(force=%.1fN)", force_n)

    def _gripper_open(self):
        self._ensure_conn(self.pub_release, timeout=1.0)
        msg = ReleaseActionGoal()
        self.pub_release.publish(msg)
        rospy.loginfo("[plan_exec] RELEASE -> ReleaseActionGoal()")


def main():
    rospy.init_node("vgr_plan_executor", anonymous=False)
    PlanExecutorNode()
    rospy.spin()


if __name__ == "__main__":
    main()
