# ====================================================
# FILE: src/vgr_sdk/drivers/robot_velocity_backend.py
# ====================================================
from __future__ import annotations
import threading
import time
import numpy as np
import rospy
from typing import List, Tuple, Optional

from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState

import PyKDL as kdl
from urdf_parser_py.urdf import URDF
from kdl_parser_py.urdf import treeFromParam

# Your planner (same import pattern as your scripts)
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
# Expect user to provide utils/TRAJECTORY_PLANNERS/trajectory_planners.py in PYTHONPATH or repo
from utils.TRAJECTORY_PLANNERS.trajectory_planners import TrajectoryPlanner  # noqa: E402


class Backend:
    """
    RobotInterface implementation that:
      - Publishes joint velocities to /velocity_controller/command
      - Uses KDL IK (LMA) for Cartesian targets
      - Streams quintic joint-velocity trajectories (like your scripts)

    ROS Params (private, ~):
      ~vel_topic     : /velocity_controller/command
      ~base_link     : base_link
      ~tip_link      : tool_ff
      ~dt            : 0.001   (control period s)
      ~move_time_s   : 3.0     (nominal duration for a segment; overridden by waypoint speed if desired)

    Note:
      - get_tcp_pose() uses FK from KDL chain.
      - move_linear(x,y,z,yaw,speed) solves IK to a pose with yaw about Z (RPY = [0,0,yaw]).
      - move_joint(joints,speed) streams directly in joint space.
    """

    def __init__(self):
        # Params
        self.vel_topic = rospy.get_param("~vel_topic", "/velocity_controller/command")
        self.base_link = rospy.get_param("~base_link", "base_link")
        self.tip_link = rospy.get_param("~tip_link", "tool_ff")
        self.dt = float(rospy.get_param("~dt", 0.001))
        self.nominal_T = float(rospy.get_param("~move_time_s", 3.0))

        # IO
        self.pub_vel = rospy.Publisher(self.vel_topic, Float64MultiArray, queue_size=10)
        rospy.Subscriber("/joint_states", JointState, self._on_js, queue_size=50)

        # KDL Chain
        if not rospy.has_param("robot_description"):
            rospy.logerr("robot_description param not found.")
            raise RuntimeError("No robot_description on param server")
        self.robot_urdf = URDF.from_parameter_server()
        ok, tree = treeFromParam("robot_description")
        if not ok:
            raise RuntimeError("Failed to parse KDL tree")

        self.chain = tree.getChain(self.base_link, self.tip_link)
        self.nj = self.chain.getNrOfJoints()
        self.ik = kdl.ChainIkSolverPos_LMA(self.chain)
        self.fk = kdl.ChainFkSolverPos_recursive(self.chain)

        # State
        self.traj_planner = TrajectoryPlanner()
        self.current_joints = None  # type: Optional[np.ndarray]
        self._busy = False
        self._exec_th = None  # type: Optional[threading.Thread]
        self._stop_flag = threading.Event()

    # --------- RobotInterface methods ---------
    def get_joint_positions(self) -> List[float]:
        while self.current_joints is None and not rospy.is_shutdown():
            rospy.sleep(0.01)
        return list(self.current_joints) if self.current_joints is not None else []

    def get_tcp_pose(self) -> Tuple[float, float, float, float]:
        js = self.get_joint_positions()
        q = kdl.JntArray(self.nj)
        for i, v in enumerate(js):
            q[i] = v
        frame = kdl.Frame()
        self.fk.JntToCart(q, frame)
        p = frame.p
        roll, pitch, yaw = frame.M.GetRPY()
        return (float(p[0]), float(p[1]), float(p[2]), float(np.degrees(yaw)))

    def move_joint(self, joints: List[float], *, speed: float) -> None:
        self._start_joint_trajectory(np.array(self.get_joint_positions()),
                                     np.array(joints), T=self.nominal_T)

    def move_linear(self, x_mm: float, y_mm: float, z_mm: float, yaw_deg: float, *, speed: float) -> None:
        # IK target
        target = kdl.Frame(
            kdl.Rotation.RPY(0.0, 0.0, np.radians(float(yaw_deg))),
            kdl.Vector(float(x_mm), float(y_mm), float(z_mm))
        )
        q_init = kdl.JntArray(self.nj)
        cur = self.get_joint_positions()
        for i, v in enumerate(cur):
            q_init[i] = v
        q_out = kdl.JntArray(self.nj)
        ret = self.ik.CartToJnt(q_init, target, q_out)
        if ret < 0:
            rospy.logerr(f"[robot_backend] IK failed with code {ret}")
            raise RuntimeError("IK failed")
        q_goal = np.array([q_out[i] for i in range(self.nj)], dtype=float)
        self._start_joint_trajectory(np.array(cur), q_goal, T=self.nominal_T)

    def stop(self) -> None:
        self._stop_flag.set()
        self._busy = False
        # publish zeros once
        self._publish_vel(np.zeros(self.nj, dtype=float))

    def is_busy(self) -> bool:
        return bool(self._busy)

    # --------- internals ---------
    def _on_js(self, msg: JointState):
        self.current_joints = np.array(msg.position, dtype=float)

    def _publish_vel(self, v: np.ndarray):
        msg = Float64MultiArray()
        msg.data = v.tolist()
        self.pub_vel.publish(msg)

    def _start_joint_trajectory(self, q_start: np.ndarray, q_goal: np.ndarray, T: float):
        if self._exec_th and self._exec_th.is_alive():
            rospy.logwarn("[robot_backend] previous trajectory still running; stopping it")
            self.stop()
            if self._exec_th:
                self._exec_th.join(timeout=0.2)

        self._stop_flag.clear()
        self._busy = True

        def _runner():
            try:
                # (pos_traj, vel_traj, acc_traj)
                _, vtraj, _ = self.traj_planner.quintic_joint_trajectory(q_start, q_goal, T, self.dt)
                for i in range(len(vtraj)):
                    if self._stop_flag.is_set() or rospy.is_shutdown():
                        break
                    self._publish_vel(vtraj[i])
                    time.sleep(self.dt)
            finally:
                # stop and mark idle
                self._publish_vel(np.zeros_like(q_goal))
                self._busy = False

        self._exec_th = threading.Thread(target=_runner, daemon=True)
        self._exec_th.start()
