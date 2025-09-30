#!/usr/bin/env python3
# ====================================
# FILE: scripts/driver_smoke_test.py
# ====================================
# source /home/pikapika/Yash_WS/Addverb_Heal_and_Syncro_Hardware/devel/setup.bash 
from __future__ import annotations
import threading
import time
import json
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional

import rospy
import yaml
from std_msgs.msg import String
from sensor_msgs.msg import JointState

# Addverb action goals (direct publish)
from addverb_cobot_msgs.msg import GraspActionGoal, ReleaseActionGoal

# Robot backend (KDL IK + velocity controller like your scripts)
from vgr_sdk.drivers.robot_velocity_backend import Backend as RobotBackend


# ------------ Data model ------------
@dataclass
class StepJoint:
    type: str  # "joint"
    joints: List[float]
    dwell_s: float = 0.0

@dataclass
class StepTCP:
    type: str  # "tcp"
    x_mm: float
    y_mm: float
    z_mm: float
    yaw_deg: float
    speed_mm_s: float = 200.0
    dwell_s: float = 0.0

@dataclass
class StepGripper:
    type: str  # "gripper"
    action: str  # "open" | "close"
    force_n: float  # grasp force in newtons (used for 'close'); ignored for 'open'

def _is_joint(step: Dict[str, Any]) -> bool:   return step.get("type") == "joint"
def _is_tcp(step: Dict[str, Any]) -> bool:     return step.get("type") == "tcp"
def _is_grip(step: Dict[str, Any]) -> bool:    return step.get("type") == "gripper"


# ------------ Node ------------
class DriverSmokeTest:
    """
    A tiny recorder/player to validate your robot + gripper drivers.

    Commands (publish JSON to ~cmd):
      {"cmd":"record_joint", "dwell_s":0.5}
      {"cmd":"record_tcp", "dwell_s":0.5, "speed_mm_s":150}
      {"cmd":"add_gripper", "action":"open"}
      {"cmd":"add_gripper", "action":"close", "force_n":100}
      {"cmd":"list"}
      {"cmd":"clear"}
      {"cmd":"save"}
      {"cmd":"load"}
      {"cmd":"play"}
      {"cmd":"stop"}

    Params (~):
      ~path                   : poses/driver_test.yaml
      ~default_tcp_speed      : 200.0
      ~default_dwell_s        : 0.0
      ~topic_grasp_goal       : /robotA/grasp_action/goal
      ~topic_release_goal     : /robotA/release_action/goal
      ~default_grasp_force_n  : 100.0   (used for 'close' when step doesn't specify force)
    """

    def __init__(self):
        self.path = rospy.get_param("~path", "poses/driver_test.yaml")
        self.default_tcp_speed = float(rospy.get_param("~default_tcp_speed", 200.0))
        self.default_dwell_s = float(rospy.get_param("~default_dwell_s", 0.0))

        # Direct Addverb action-goal topics
        self.topic_grasp_goal = rospy.get_param("~topic_grasp_goal", "/robotA/grasp_action/goal")
        self.topic_release_goal = rospy.get_param("~topic_release_goal", "/robotA/release_action/goal")
        self.default_grasp_force = float(rospy.get_param("~default_grasp_force_n", 100.0))

        # Robot + state
        self.robot = RobotBackend()
        self.joint_state = None  # type: Optional[List[float]]
        rospy.Subscriber("/joint_states", JointState, self._on_js, queue_size=50)

        # Gripper (direct action goals; stroke/width logic removed — use force only)
        self.pub_grasp_goal = rospy.Publisher(self.topic_grasp_goal, GraspActionGoal, queue_size=10)
        self.pub_release_goal = rospy.Publisher(self.topic_release_goal, ReleaseActionGoal, queue_size=10)

        # Command + status
        rospy.Subscriber("~cmd", String, self._on_cmd, queue_size=20)
        self.pub_status = rospy.Publisher("~status", String, queue_size=10)

        # Program
        self.steps: List[Dict[str, Any]] = []
        self._exec_th: Optional[threading.Thread] = None
        self._stop_flag = threading.Event()

        rospy.loginfo("[driver_smoke] Ready. Publish JSON to ~cmd to record/play. (cmd examples: list | record_joint | record_tcp | add_gripper | save | load | play | stop)")
        rospy.loginfo(f"[driver_smoke] Gripper direct publish: close -> {self.topic_grasp_goal} (force={self.default_grasp_force}N), open -> {self.topic_release_goal}")

    # ---------- Callbacks ----------
    def _on_js(self, msg: JointState):
        self.joint_state = list(msg.position)

    def _on_cmd(self, msg: String):
        try:
            cmd = json.loads(msg.data)
        except Exception as e:
            rospy.logwarn(f"[driver_smoke] bad JSON: {e}")
            return
        c = str(cmd.get("cmd","")).lower()
        if c == "list":
            self._print_list()
        elif c == "clear":
            self.steps.clear()
            self._status("cleared")
        elif c == "save":
            self._save()
        elif c == "load":
            self._load()
        elif c == "record_joint":
            self._record_joint(dwell_s=float(cmd.get("dwell_s", self.default_dwell_s)))
        elif c == "record_tcp":
            self._record_tcp(dwell_s=float(cmd.get("dwell_s", self.default_dwell_s)),
                             speed=float(cmd.get("speed_mm_s", self.default_tcp_speed)))
        elif c == "add_gripper":
            # new API: use force_n for close; open ignores it
            action = str(cmd.get("action","open")).lower()
            force_n = cmd.get("force_n", None)
            if force_n is None:
                # if not provided, use default grasp force for 'close'; ignored for 'open'
                force_n = float(self.default_grasp_force)
            else:
                force_n = float(force_n)
            self._add_gripper(action=action, force_n=force_n)
        elif c == "play":
            self._play()
        elif c == "stop":
            self._stop()
        else:
            rospy.logwarn(f"[driver_smoke] unknown cmd: {c}")

    # ---------- Actions ----------
    def _record_joint(self, *, dwell_s: float):
        js = self.joint_state or self.robot.get_joint_positions()
        if not js:
            self._status("no joint state yet")
            return
        step = StepJoint(type="joint", joints=[float(v) for v in js], dwell_s=float(dwell_s))
        self.steps.append(asdict(step))
        self._status(f"recorded JOINT + dwell {dwell_s:.3f}s")

    def _record_tcp(self, *, dwell_s: float, speed: float):
        x, y, z, yaw = self.robot.get_tcp_pose()
        step = StepTCP(type="tcp", x_mm=x, y_mm=y, z_mm=z, yaw_deg=yaw,
                       speed_mm_s=float(speed), dwell_s=float(dwell_s))
        self.steps.append(asdict(step))
        self._status(f"recorded TCP ({x:.1f},{y:.1f},{z:.1f},{yaw:.1f}) + dwell {dwell_s:.3f}s")

    def _add_gripper(self, *, action: str, force_n: float):
        if action not in ("open","close"):
            self._status("gripper action must be 'open' or 'close'")
            return
        step = StepGripper(type="gripper", action=action, force_n=float(force_n))
        self.steps.append(asdict(step))
        if action == "open":
            self._status("added GRIPPER open")
        else:
            self._status(f"added GRIPPER close force={force_n:.1f}N")

    def _save(self):
        data = {"version": 1, "steps": self.steps}
        with open(self.path, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, sort_keys=False)
        self._status(f"saved {self.path}")

    def _load(self):
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            loaded_steps = list(data.get("steps", []))

            # Support legacy 'width_mm' entries by ignoring them and mapping to default force
            normalized_steps: List[Dict[str, Any]] = []
            for s in loaded_steps:
                if _is_grip(s):
                    # If file has an old 'width_mm' field, drop it and set force to default or to provided force_n
                    if "force_n" in s:
                        force = float(s.get("force_n", self.default_grasp_force))
                    else:
                        # legacy width_mm present? ignore value and use default force
                        force = float(s.get("force_n", self.default_grasp_force))
                    normalized = {"type": "gripper", "action": s.get("action", "open"), "force_n": force}
                    normalized_steps.append(normalized)
                else:
                    normalized_steps.append(s)
            self.steps = normalized_steps

            self._status(f"loaded {self.path} ({len(self.steps)} steps)")
        except Exception as e:
            self._status(f"load failed: {e}")

    def _print_list(self):
        lines = []
        for i, s in enumerate(self.steps):
            if _is_joint(s):
                lines.append(f"{i:02d} JOINT dwell={s.get('dwell_s',0):.2f} n={len(s.get('joints',[]))}")
            elif _is_tcp(s):
                lines.append(f"{i:02d} TCP x={s['x_mm']:.1f} y={s['y_mm']:.1f} z={s['z_mm']:.1f} yaw={s['yaw_deg']:.1f} spd={s.get('speed_mm_s',0):.0f} dwell={s.get('dwell_s',0):.2f}")
            elif _is_grip(s):
                # show force if present; otherwise show default force
                force = s.get("force_n", self.default_grasp_force)
                lines.append(f"{i:02d} GRIP {s['action']} force={force:.1f}N")
            else:
                lines.append(f"{i:02d} ??? {s}")
        txt = "\n".join(lines) if lines else "(no steps)"
        rospy.loginfo("[driver_smoke] program:\n" + txt)
        self._status(f"{len(self.steps)} steps")

    # ---------- Execution ----------
    def _play(self):
        if self._exec_th and self._exec_th.is_alive():
            self._status("already playing")
            return
        if not self.steps:
            self._status("no steps to play")
            return

        self._stop_flag.clear()
        self._exec_th = threading.Thread(target=self._run_sequence, daemon=True)
        self._exec_th.start()

    def _stop(self):
        self._stop_flag.set()
        try:
            self.robot.stop()
        except Exception:
            pass
        self._status("stopping")

    def _ensure_conn(self, pub, timeout=1.0):
        # Wait briefly for a subscriber (the action server)
        t0 = time.time()
        while pub.get_num_connections() == 0 and (time.time() - t0) < timeout and not rospy.is_shutdown():
            rospy.sleep(0.02)

    def _run_sequence(self):
        self._status("playing")
        ok = True
        try:
            for idx, s in enumerate(self.steps):
                if self._stop_flag.is_set() or rospy.is_shutdown():
                    ok = False
                    break

                if _is_joint(s):
                    self.robot.move_joint(s["joints"], speed=0.0)  # speed handled by backend timing
                    while self.robot.is_busy() and not rospy.is_shutdown() and not self._stop_flag.is_set():
                        rospy.sleep(0.01)
                    self._dwell(s.get("dwell_s", 0.0))

                elif _is_tcp(s):
                    self.robot.move_linear(s["x_mm"], s["y_mm"], s["z_mm"], s["yaw_deg"],
                                           speed=float(s.get("speed_mm_s", self.default_tcp_speed)))
                    while self.robot.is_busy() and not rospy.is_shutdown() and not self._stop_flag.is_set():
                        rospy.sleep(0.01)
                    self._dwell(s.get("dwell_s", 0.0))

                elif _is_grip(s):
                    action = s.get("action", "open")
                    if action == "open":
                        self._ensure_conn(self.pub_release_goal)
                        self.pub_release_goal.publish(ReleaseActionGoal())
                        rospy.loginfo("[driver_smoke] GRIPPER open -> ReleaseActionGoal()")
                    else:
                        # Use provided force_n if present, otherwise fallback to node default
                        force = float(s.get("force_n", self.default_grasp_force))
                        self._ensure_conn(self.pub_grasp_goal)
                        g = GraspActionGoal()
                        g.goal.grasp_force = force
                        self.pub_grasp_goal.publish(g)
                        rospy.loginfo(f"[driver_smoke] GRIPPER close -> GraspActionGoal(force={force:.1f}N)")
                    # tiny settle time
                    rospy.sleep(0.2)
                else:
                    rospy.logwarn(f"[driver_smoke] unknown step: {s}")

            self._status("done" if ok else "stopped")
        except Exception as e:
            rospy.logerr(f"[driver_smoke] play error: {e}")
            self._status(f"error: {e}")
            try: self.robot.stop()
            except Exception: pass

    def _dwell(self, seconds: float):
        t0 = time.time()
        while (time.time() - t0) < float(seconds) and not rospy.is_shutdown() and not self._stop_flag.is_set():
            rospy.sleep(0.01)

    # ---------- Status ----------
    def _status(self, text: str):
        msg = {"status": text, "count": len(self.steps)}
        self.pub_status.publish(String(data=json.dumps(msg, separators=(",",":"))))
        rospy.loginfo(f"[driver_smoke] {text}")


def main():
    rospy.init_node("vgr_driver_smoke_test", anonymous=False)
    DriverSmokeTest()
    rospy.spin()


if __name__ == "__main__":
    main()
