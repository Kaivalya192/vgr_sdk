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

import os
import sys
import numpy as np

import rospy
import yaml
from std_msgs.msg import String, Float64MultiArray
from sensor_msgs.msg import JointState

# Addverb action goals (direct publish)
from addverb_cobot_msgs.msg import GraspActionGoal, ReleaseActionGoal

# Robot backend (KDL IK + velocity controller like your scripts) — used for record/play
from vgr_sdk.drivers.robot_velocity_backend import Backend as RobotBackend

# -------- Teleop deps (same as your working script) --------
import PyKDL as kdl
from urdf_parser_py.urdf import URDF
from kdl_parser_py.urdf import treeFromParam

# Allow importing your utils like in the standalone teleop
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.TRAJECTORY_PLANNERS.trajectory_planners import TrajectoryPlanner


# ------------ Data model (record/play) ------------
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
    Recorder/player + gripper + TCP teleop (KDL IK + quintic joint-velocity)

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

    Teleop (publish JSON to ~teleop):
        {"cmd":"teleop_start"}
        {"cmd":"teleop_stop"}
        {"cmd":"nudge","dx":0.005,"dy":0.0,"dz":0.0,"yaw":0.05}         # dx/dy/dz in meters, yaw in radians
        {"cmd":"nudge","roll":0.05}                                     # roll in radians (about world X)
        {"cmd":"nudge","pitch":0.05}                                    # pitch in radians (about world Y)
        {"cmd":"nudge","roll":0.02,"pitch":-0.02,"yaw":0.05}            # any combo of roll/pitch/yaw
        {"cmd":"recapture"}
        {"cmd":"print"}
        {"cmd":"stop"}                                                  # zero velocity once

    Params (~):
      # Record/Play
      ~path                   : poses/driver_test.yaml
      ~default_tcp_speed      : 200.0
      ~default_dwell_s        : 0.0
      ~topic_grasp_goal       : /robotA/grasp_action/goal
      ~topic_release_goal     : /robotA/release_action/goal
      ~default_grasp_force_n  : 100.0
      ~teleop_rp_time_scale   : 2.5        # multiply segment time for roll/pitch nudges

      # Teleop (mirrors your working script)
      ~base_link              : base_link
      ~tip_link               : tool_ff
      ~teleop_dt              : 0.01        # control period (s)
      ~teleop_segment_T       : 0.4         # seconds per nudge
      ~teleop_vel_topic       : /velocity_controller/command
    """

    def __init__(self):
        # -------- Record/Play params --------
        self.path = rospy.get_param("~path", "poses/driver_test.yaml")
        self.default_tcp_speed = float(rospy.get_param("~default_tcp_speed", 200.0))
        self.default_dwell_s = float(rospy.get_param("~default_dwell_s", 0.0))

        # Gripper topics
        self.topic_grasp_goal = rospy.get_param("~topic_grasp_goal", "/robotA/grasp_action/goal")
        self.topic_release_goal = rospy.get_param("~topic_release_goal", "/robotA/release_action/goal")
        self.default_grasp_force = float(rospy.get_param("~default_grasp_force_n", 100.0))

        # -------- Teleop params (KDL + vel traj) --------
        self.base_link = rospy.get_param("~base_link", "base_link")
        self.tip_link = rospy.get_param("~tip_link", "tool_ff")
        self.teleop_dt = float(rospy.get_param("~teleop_dt", 0.01))
        self.teleop_segment_T = float(rospy.get_param("~teleop_segment_T", 0.4))
        self.teleop_rp_time_scale = float(rospy.get_param("~teleop_rp_time_scale", 2.5))
        self.teleop_vel_topic = rospy.get_param("~teleop_vel_topic", "/velocity_controller/command")

        # -------- Robot backend for record/play --------
        self.robot = RobotBackend()

        # -------- State: record/play --------
        self.joint_state = None  # type: Optional[List[float]]
        rospy.Subscriber("/joint_states", JointState, self._on_js, queue_size=50)

        # Gripper pubs
        self.pub_grasp_goal = rospy.Publisher(self.topic_grasp_goal, GraspActionGoal, queue_size=10)
        self.pub_release_goal = rospy.Publisher(self.topic_release_goal, ReleaseActionGoal, queue_size=10)

        # Command + status
        rospy.Subscriber("~cmd", String, self._on_cmd, queue_size=20)
        self.pub_status = rospy.Publisher("~status", String, queue_size=10)

        # Program buffer
        self.steps: List[Dict[str, Any]] = []
        self._exec_th: Optional[threading.Thread] = None
        self._stop_flag = threading.Event()

        # -------- Teleop (KDL IK + velocity) --------
        # Build KDL chain from robot_description
        if not rospy.has_param("robot_description"):
            rospy.logwarn("[driver_smoke] robot_description param not found. Teleop will not start until URDF is available.")
        else:
            self.robot_model = URDF.from_parameter_server()
            ok, tree = treeFromParam("robot_description")
            if not ok:
                rospy.logerr("[driver_smoke] Failed to parse KDL tree from URDF.")
            else:
                self.chain = tree.getChain(self.base_link, self.tip_link)
                self.nj = self.chain.getNrOfJoints()
                self.fk_solver = kdl.ChainFkSolverPos_recursive(self.chain)
                self.ik_solver = kdl.ChainIkSolverPos_LMA(self.chain)
                rospy.loginfo("[driver_smoke] KDL chain created with %d joints", self.nj)

        # Teleop publisher (velocity)
        self.pub_vel = rospy.Publisher(self.teleop_vel_topic, Float64MultiArray, queue_size=50)
        self.traj = TrajectoryPlanner()

        # Teleop runtime state
        self.q_np: Optional[np.ndarray] = None         # latest joints (np array) for KDL
        self.target_F: Optional[kdl.Frame] = None      # target TCP frame
        self.active_vel_traj = None                    # list/array of joint velocities
        self.active_index = 0
        self.teleop_active = False
        self._teleop_lock = threading.Lock()

        # Timer to stream velocity samples while a segment is active
        self.timer = rospy.Timer(rospy.Duration(self.teleop_dt), self._teleop_tick)

        rospy.loginfo("[driver_smoke] Ready. Publish JSON to ~cmd to record/play. (cmd examples: list | record_joint | record_tcp | add_gripper | save | load | play | stop)")
        rospy.loginfo(f"[driver_smoke] Gripper direct publish: close -> {self.topic_grasp_goal} (force={self.default_grasp_force}N), open -> {self.topic_release_goal}")
        rospy.loginfo(f"[driver_smoke] Teleop topic: {rospy.get_name()}/teleop  (dt={self.teleop_dt}s, T={self.teleop_segment_T}s, vel_topic={self.teleop_vel_topic})")

        # Teleop command subscriber
        rospy.Subscriber("~teleop", String, self._on_teleop, queue_size=20)

    # ---------- Joint states ----------
    def _on_js(self, msg: JointState):
        # record/play uses this directly
        self.joint_state = list(msg.position)
        # teleop keeps np array with correct joint count if KDL is ready
        if hasattr(self, "nj") and self.joint_state is not None:
            self.q_np = np.array(self.joint_state[:self.nj]) if len(self.joint_state) >= self.nj else None

    # ---------- Utilities (KDL) ----------
    def _fk(self, q_np: np.ndarray) -> kdl.Frame:
        q = kdl.JntArray(self.nj)
        for i in range(self.nj):
            q[i] = float(q_np[i])
        out = kdl.Frame()
        self.fk_solver.JntToCart(q, out)
        return out

    def _ik(self, q_init_np: np.ndarray, target_F: kdl.Frame) -> Optional[np.ndarray]:
        q_init = kdl.JntArray(self.nj)
        for i in range(self.nj):
            q_init[i] = float(q_init_np[i])
        q_out = kdl.JntArray(self.nj)
        rc = self.ik_solver.CartToJnt(q_init, target_F, q_out)
        if rc >= 0:
            return np.array([q_out[i] for i in range(self.nj)])
        return None

    # ---- Target edits: translate only changes position; yaw edits rotate about world-Z ----
    def _shift_target_xyz(self, dx=0.0, dy=0.0, dz=0.0):
        # Keep R/P/Y exactly as-is; only translate the position.
        P = self.target_F.p
        self.target_F.p = kdl.Vector(P.x() + dx, P.y() + dy, P.z() + dz)

    def _yaw_target(self, d_yaw_rad: float):
        # Rotate about world Z by d_yaw and keep the existing roll/pitch relative to the new yaw.
        self.target_F.M = kdl.Rotation.RotZ(d_yaw_rad) * self.target_F.M
        
    def _apply_rpy_world(self, d_roll_rad: float = 0.0, d_pitch_rad: float = 0.0, d_yaw_rad: float = 0.0):
        """
        Apply small world-frame rotations to target_F in this exact order:
        Yaw (Z) -> Pitch (Y) -> Roll (X), i.e. Rz * Ry * Rx * M
        Any of d_* may be zero.
        """
        # Build R_total explicitly in the intended order
        R_total = kdl.Rotation.Identity()
        if d_yaw_rad:
            R_total = kdl.Rotation.RotZ(d_yaw_rad)
        if d_pitch_rad:
            R_total = (R_total * kdl.Rotation.RotY(d_pitch_rad)) if d_yaw_rad else kdl.Rotation.RotY(d_pitch_rad)
        if d_roll_rad:
            R_total = (R_total * kdl.Rotation.RotX(d_roll_rad)) if (d_yaw_rad or d_pitch_rad) else kdl.Rotation.RotX(d_roll_rad)

        # Left-multiply world-frame delta
        self.target_F.M = R_total * self.target_F.M


    def _print_tcp(self):
        try:
            cur_F = self._fk(self.q_np)
        except Exception:
            rospy.logwarn("[driver_smoke] _print_tcp: FK not available yet.")
            return
        cx, cy, cz = cur_F.p.x(), cur_F.p.y(), cur_F.p.z()
        cqx, cqy, cqz, cqw = cur_F.M.GetQuaternion()
        if self.target_F is None:
            rospy.loginfo("FK TCP: x=%.3f y=%.3f z=%.3f | quat=[%.3f %.3f %.3f %.3f]",
                          cx, cy, cz, cqx, cqy, cqz, cqw)
        else:
            tx, ty, tz = self.target_F.p.x(), self.target_F.p.y(), self.target_F.p.z()
            tqx, tqy, tqz, tqw = self.target_F.M.GetQuaternion()
            rospy.loginfo(
                "FK TCP: x=%.3f y=%.3f z=%.3f | quat=[%.3f %.3f %.3f %.3f]\n"
                "TGT   : x=%.3f y=%.3f z=%.3f | quat=[%.3f %.3f %.3f %.3f]",
                cx, cy, cz, cqx, cqy, cqz, cqw,
                tx, ty, tz, tqx, tqy, tqz, tqw
            )

    # ---------- Teleop velocity streaming ----------
    def _teleop_tick(self, _evt):
        with self._teleop_lock:
            if self.active_vel_traj is None:
                return
            if self.active_index < len(self.active_vel_traj):
                msg = Float64MultiArray()
                msg.data = self.active_vel_traj[self.active_index].tolist()
                self.pub_vel.publish(msg)
                self.active_index += 1
            else:
                # end of segment -> publish one zero-vel sample
                msg = Float64MultiArray()
                msg.data = [0.0] * self.nj
                self.pub_vel.publish(msg)
                self.active_vel_traj = None

    # ---------- Teleop command handling ----------
    def _on_teleop(self, msg: String):
        try:
            data = json.loads(msg.data)
        except Exception as e:
            rospy.logwarn(f"[driver_smoke] bad teleop JSON: {e}")
            return

        cmd = str(data.get("cmd", "")).lower()

        # KDL/URDF readiness checks
        if not hasattr(self, "chain") or not hasattr(self, "nj"):
            rospy.logwarn("[driver_smoke] teleop: KDL not ready (no robot_description/chain).")
            self._status("teleop not ready")
            return
        if self.q_np is None:
            rospy.logwarn("[driver_smoke] teleop: waiting for /joint_states...")
            self._status("teleop waiting joint_states")
            return

        if cmd == "teleop_start":
            # Capture current TCP via FK, keep full RPY as-is
            try:
                self.target_F = self._fk(self.q_np)
                self.teleop_active = True
                self._status("teleop started")
                rospy.loginfo("[driver_smoke] teleop started (target captured, keep full RPY)")
            except Exception as e:
                rospy.logwarn("[driver_smoke] teleop_start failed: %s", e)
                self._status("teleop start failed")

        elif cmd == "teleop_stop":
            with self._teleop_lock:
                # publish one zero-vel
                try:
                    msg0 = Float64MultiArray()
                    msg0.data = [0.0] * self.nj
                    self.pub_vel.publish(msg0)
                except Exception:
                    pass
                self.active_vel_traj = None
                self.active_index = 0
                self.teleop_active = False
            self._status("teleop stopped")
            rospy.loginfo("[driver_smoke] teleop stopped")

        elif cmd == "stop":
            # alias: stop streaming now
            with self._teleop_lock:
                try:
                    msg0 = Float64MultiArray()
                    msg0.data = [0.0] * self.nj
                    self.pub_vel.publish(msg0)
                except Exception:
                    pass
                self.active_vel_traj = None
                self.active_index = 0
            self._status("teleop stop")
            rospy.loginfo("[driver_smoke] teleop stop (zero velocity).")

        elif cmd == "recapture":
            try:
                self.target_F = self._fk(self.q_np)  # keep full RPY from FK
                self._status("teleop recaptured")
                rospy.loginfo("[driver_smoke] teleop: target recaptured (full RPY).")
                self._print_tcp()
            except Exception as e:
                rospy.logwarn("[driver_smoke] teleop recapture failed: %s", e)
                self._status("teleop recapture failed")

        elif cmd in ("print", "print_tcp"):
            self._print_tcp()
            self._status("teleop print")

        elif cmd == "nudge":
            if not self.teleop_active or self.target_F is None:
                rospy.logwarn("[driver_smoke] teleop nudge: call teleop_start first.")
                self._status("teleop inactive")
                return

            dx = float(data.get("dx", 0.0))      # meters
            dy = float(data.get("dy", 0.0))
            dz = float(data.get("dz", 0.0))
            yaw = float(data.get("yaw", 0.0))    # radians
            pitch = float(data.get("pitch", 0.0))# radians (world Y)
            roll = float(data.get("roll", 0.0))  # radians (world X)

            with self._teleop_lock:
                if self.active_vel_traj is not None:
                    rospy.logwarn("[driver_smoke] Busy executing previous nudge; please wait...")
                    self._status("teleop busy")
                    return

                # --- snapshot current target to allow clean revert on IK failure ---
                # Deep-copy target_F via (pos + quaternion) to avoid mutating references
                old_px, old_py, old_pz = self.target_F.p.x(), self.target_F.p.y(), self.target_F.p.z()
                oqx, oqy, oqz, oqw = self.target_F.M.GetQuaternion()

                # 1) edit target frame (translate)
                if dx or dy or dz:
                    self._shift_target_xyz(dx, dy, dz)

                # 2) edit orientation (world-frame): yaw (Z), then pitch (Y), then roll (X)
                if yaw or pitch or roll:
                    self._apply_rpy_world(d_roll_rad=roll, d_pitch_rad=pitch, d_yaw_rad=yaw)

                # 3) solve IK from current joints
                q_goal = self._ik(self.q_np, self.target_F)
                if q_goal is None:
                    rospy.logwarn("[driver_smoke] IK failed for requested nudge; reverting target.")
                    # revert entire pose (both position and orientation)
                    self.target_F.p = kdl.Vector(old_px, old_py, old_pz)
                    self.target_F.M = kdl.Rotation.Quaternion(oqx, oqy, oqz, oqw)
                    self._status("teleop IK failed")
                    return

                # 4) choose segment time:
                #    - stretch for roll/pitch (user asked to slow these)
                #    - (optional) auto-scale more if the joint move is large
                T = self.teleop_segment_T

                # If there's any roll or pitch request, slow the motion
                if (abs(roll) > 0.0) or (abs(pitch) > 0.0):
                    T = max(T * self.teleop_rp_time_scale, T)

                # Optional: also slow further if joint move is large
                dq = np.abs(q_goal - self.q_np)
                max_dq = float(np.max(dq)) if dq.size else 0.0
                if max_dq > np.deg2rad(8.0):
                    T = max(T, 0.8)
                if max_dq > np.deg2rad(15.0):
                    T = max(T, 1.2)

                # 5) plan + execute short velocity segment
                _, vel_traj, _ = self.traj.quintic_joint_trajectory(self.q_np.copy(), q_goal, T, self.teleop_dt)
                self.active_vel_traj = vel_traj
                self.active_index = 0
                self._status("teleop nudge")
                rospy.loginfo(
                    "[driver_smoke] Executing nudge: %d steps over %.2fs (dx=%.4f dy=%.4f dz=%.4f | roll=%.4f pitch=%.4f yaw=%.4f rad) max|dq|=%.3f rad",
                    len(vel_traj), T, dx, dy, dz, roll, pitch, yaw, max_dq
                )


        else:
            rospy.logwarn("[driver_smoke] teleop unknown cmd: %s", cmd)

    # ---------- Record/Play commands ----------
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
            action = str(cmd.get("action","open")).lower()
            force_n = cmd.get("force_n", None)
            force_n = float(self.default_grasp_force) if force_n is None else float(force_n)
            self._add_gripper(action=action, force_n=force_n)
        elif c == "play":
            self._play()
        elif c == "stop":
            self._stop()
        else:
            rospy.logwarn(f"[driver_smoke] unknown cmd: {c}")

    # ---------- Record/Play actions ----------
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
            normalized_steps: List[Dict[str, Any]] = []
            for s in loaded_steps:
                if _is_grip(s):
                    force = float(s.get("force_n", self.default_grasp_force))
                    normalized_steps.append({"type": "gripper", "action": s.get("action", "open"), "force_n": force})
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
                force = s.get("force_n", self.default_grasp_force)
                lines.append(f"{i:02d} GRIP {s['action']} force={force:.1f}N")
            else:
                lines.append(f"{i:02d} ??? {s}")
        txt = "\n".join(lines) if lines else "(no steps)"
        rospy.loginfo("[driver_smoke] program:\n" + txt)
        self._status(f"{len(self.steps)} steps")

    # ---------- Record/Play execution ----------
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
                    self.robot.move_joint(s["joints"], speed=0.0)
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
                        force = float(s.get("force_n", self.default_grasp_force))
                        self._ensure_conn(self.pub_grasp_goal)
                        g = GraspActionGoal()
                        g.goal.grasp_force = force
                        self.pub_grasp_goal.publish(g)
                        rospy.loginfo(f"[driver_smoke] GRIPPER close -> GraspActionGoal(force={force:.1f}N)")
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
        try:
            self.pub_status.publish(String(data=json.dumps(msg, separators=(",",":"))))
        except Exception:
            pass
        rospy.loginfo(f"[driver_smoke] {text}")


def main():
    rospy.init_node("vgr_driver_smoke_test", anonymous=False)
    DriverSmokeTest()
    rospy.spin()


if __name__ == "__main__":
    main()
