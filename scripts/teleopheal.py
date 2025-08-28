#!/usr/bin/env python3
"""
TCP Keyboard Teleop (IK + Quintic Vel Traj)

Run directly with Python (ROS 1 is used via rospy).

Fix: persistent raw keyboard mode so keystrokes are actually read (no newline needed),
with fallback to /dev/tty. This avoids the earlier issue where keys were echoed but
not captured.

- Captures current TCP pose from FK at startup
- Nudge TCP with keys (WASD, R/F, Q/E for yaw)
- Solves IK with KDL ChainIkSolverPos_LMA
- Generates a short quintic joint-velocity trajectory for each nudge
- Publishes Float64MultiArray on /velocity_controller/command

Requirements:
- ROS1, robot_description loaded (URDF)
- /joint_states publishing actual joint positions (same joint order as KDL chain)
- utils.TRAJECTORY_PLANNERS.trajectory_planners.TrajectoryPlanner available

Params (~):
  ~base_link           : base_link
  ~tip_link            : tool_ff
  ~dt                  : 0.01            (control period)
  ~segment_T           : 0.4             (sec per nudge)
  ~step_trans_mm       : 5.0             (translation step)
  ~step_yaw_deg        : 5.0             (yaw step)
  ~vel_topic           : /velocity_controller/command

Keymap:
  W/S: +X/-X   A/D: +Y/-Y   R/F: +Z/-Z
  Q/E: +Yaw/-Yaw (about base Z)
  Z/X: translation step -/+ (mm)   C/V: yaw step -/+ (deg)
  G  : recapture current TCP as target (from FK)
  P  : print current TCP (FK) and target TCP
  SPACE: stop (publish one zero-vel sample)
  H  : help
  ESC / Ctrl+C: quit
"""

import os
import sys
import select
import termios
import tty
import numpy as np
import PyKDL as kdl
import rospy
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
from urdf_parser_py.urdf import URDF
from kdl_parser_py.urdf import treeFromParam

# Add your utils path (same as your example)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.TRAJECTORY_PLANNERS.trajectory_planners import TrajectoryPlanner


def frame_to_xyzquat(F: kdl.Frame):
    x, y, z = F.p.x(), F.p.y(), F.p.z()
    qx, qy, qz, qw = F.M.GetQuaternion()
    return (x, y, z, qx, qy, qz, qw)


def rotz(delta_rad):
    return kdl.Rotation.RotZ(delta_rad)


class TcpKeyboardTeleop:
    def __init__(self):
        rospy.init_node("tcp_keyboard_teleop", anonymous=True)

        # --- Params ---
        self.base_link = rospy.get_param("~base_link", "base_link")
        self.tip_link = rospy.get_param("~tip_link", "tool_ff")
        self.dt = float(rospy.get_param("~dt", 0.01))
        self.segment_T = float(rospy.get_param("~segment_T", 0.4))
        self.step_trans_m = float(rospy.get_param("~step_trans_mm", 5.0)) / 1000.0  # default 5 mm
        self.step_yaw_rad = np.deg2rad(float(rospy.get_param("~step_yaw_deg", 5.0)))
        self.vel_topic = rospy.get_param("~vel_topic", "/velocity_controller/command")

        # --- Pub/Sub ---
        self.pub = rospy.Publisher(self.vel_topic, Float64MultiArray, queue_size=20)
        self.joint_state_sub = rospy.Subscriber("/joint_states", JointState, self._on_js)

        # --- URDF + KDL ---
        if not rospy.has_param("robot_description"):
            rospy.logerr("robot_description param not found.")
            sys.exit(1)
        self.robot = URDF.from_parameter_server()
        ok, tree = treeFromParam("robot_description")
        if not ok:
            rospy.logerr("Failed to parse KDL tree from URDF.")
            sys.exit(1)
        self.chain = tree.getChain(self.base_link, self.tip_link)
        self.nj = self.chain.getNrOfJoints()
        rospy.loginfo("KDL chain created with %d joints", self.nj)

        self.fk_solver = kdl.ChainFkSolverPos_recursive(self.chain)
        self.ik_solver = kdl.ChainIkSolverPos_LMA(self.chain)

        # --- Planner ---
        self.traj = TrajectoryPlanner()

        # --- State ---
        self.q = None                     # current joints (np array)
        self.active_vel_traj = None       # list/array of joint velocities
        self.active_index = 0
        self.target_F = None              # current target TCP frame (kdl.Frame)

        # --- Keyboard setup (persistent raw mode + no echo) ---
        self._kbd_fd = None
        self._kbd_from_devtty = False
        self._old_term = None
        self._keyboard_setup()

        # Wait for first joint state to seed FK
        rospy.loginfo("Waiting for /joint_states...")
        while self.q is None and not rospy.is_shutdown():
            rospy.sleep(0.05)

        # Capture current TCP via FK
        self.target_F = self._fk(self.q)
        x, y, z, qx, qy, qz, qw = frame_to_xyzquat(self.target_F)
        rospy.loginfo(
            "Captured current TCP (FK): x=%.3f y=%.3f z=%.3f | quat=[%.3f, %.3f, %.3f, %.3f]",
            x, y, z, qx, qy, qz, qw)

        self._print_help()

    # ----------------- Keyboard raw mode -----------------
    def _keyboard_setup(self):
        """Put terminal into noncanonical, no-echo mode and keep it there.
        Fallback to /dev/tty if stdin is not a TTY.
        """
        # Choose a fd that is a real TTY
        try:
            if hasattr(sys.stdin, "isatty") and sys.stdin.isatty():
                fd = sys.stdin.fileno()
            else:
                # fall back to /dev/tty
                fd = os.open('/dev/tty', os.O_RDONLY)
                self._kbd_from_devtty = True
        except Exception:
            # ultimate fallback
            fd = os.open('/dev/tty', os.O_RDONLY)
            self._kbd_from_devtty = True

        # Save + set raw-ish
        self._kbd_fd = fd
        self._old_term = termios.tcgetattr(fd)
        new = termios.tcgetattr(fd)
        new[3] = new[3] & ~(termios.ICANON | termios.ECHO)   # lflags: no canonical, no echo
        new[6][termios.VMIN] = 0
        new[6][termios.VTIME] = 0
        termios.tcsetattr(fd, termios.TCSANOW, new)
        # also set cbreak
        try:
            tty.setcbreak(fd)
        except Exception:
            pass

    def _keyboard_restore(self):
        try:
            if self._kbd_fd is not None and self._old_term is not None:
                termios.tcsetattr(self._kbd_fd, termios.TCSADRAIN, self._old_term)
            if self._kbd_from_devtty and self._kbd_fd is not None:
                os.close(self._kbd_fd)
        except Exception:
            pass

    def _read_key(self):
        try:
            r, _, _ = select.select([self._kbd_fd], [], [], 0.0)
            if r:
                ch = os.read(self._kbd_fd, 1).decode('utf-8', errors='ignore')
                return ch
        except Exception:
            return ''
        return ''

    # ----------------- ROS callbacks -----------------
    def _on_js(self, msg: JointState):
        # keep the last nj positions (in case of fixed joints in stream)
        self.q = np.array(msg.position[:self.nj])

    # ----------------- KDL helpers -------------------
    def _fk(self, q_np: np.ndarray) -> kdl.Frame:
        q = kdl.JntArray(self.nj)
        for i in range(self.nj):
            q[i] = float(q_np[i])
        out = kdl.Frame()
        self.fk_solver.JntToCart(q, out)
        return out

    def _ik(self, q_init_np: np.ndarray, target_F: kdl.Frame):
        q_init = kdl.JntArray(self.nj)
        for i in range(self.nj):
            q_init[i] = float(q_init_np[i])
        q_out = kdl.JntArray(self.nj)
        rc = self.ik_solver.CartToJnt(q_init, target_F, q_out)
        if rc >= 0:
            return np.array([q_out[i] for i in range(self.nj)])
        return None

    # ----------------- Motion ------------------------
    def _plan_and_start(self, q_start: np.ndarray, q_goal: np.ndarray):
        """Generate short vel trajectory and start executing."""
        _, vel_traj, _ = self.traj.quintic_joint_trajectory(q_start, q_goal, self.segment_T, self.dt)
        self.active_vel_traj = vel_traj
        self.active_index = 0
        rospy.loginfo("Executing nudge: %d steps over %.2fs", len(vel_traj), self.segment_T)

    def _tick_motion(self):
        """Publish next velocity sample if a trajectory is active."""
        if self.active_vel_traj is None:
            return
        if self.active_index < len(self.active_vel_traj):
            msg = Float64MultiArray()
            msg.data = self.active_vel_traj[self.active_index].tolist()
            self.pub.publish(msg)
            self.active_index += 1
        else:
            # end of segment -> publish one zero to settle
            msg = Float64MultiArray()
            msg.data = [0.0] * self.nj
            self.pub.publish(msg)
            self.active_vel_traj = None

    # ----------------- Target edits ------------------
    def _shift_target_xyz(self, dx=0.0, dy=0.0, dz=0.0):
        P = self.target_F.p
        self.target_F.p = kdl.Vector(P.x() + dx, P.y() + dy, P.z() + dz)

    def _yaw_target(self, d_yaw_rad: float):
        self.target_F.M = rotz(d_yaw_rad) * self.target_F.M  # world Z yaw

    # ----------------- UI / Help ---------------------
    def _print_help(self):
        rospy.loginfo(
            "--- TCP Keyboard Teleop ---"
            "W/S: +X/-X   A/D: +Y/-Y   R/F: +Z/-Z   Q/E: +Yaw/-Yaw (about base Z)"
            "Z/X: trans step -/+ (mm)   C/V: yaw step -/+ (deg)"
            "G: recapture current TCP as target   P: print TCP"
            "SPACE: stop (zero velocity once)   H: help   ESC/CTRL+C: quit"
            "Current steps: %.1f mm, %.1f deg",
            self.step_trans_m * 1000.0, np.rad2deg(self.step_yaw_rad)
        )

    def _print_tcp(self):
        cur_F = self._fk(self.q)
        cx, cy, cz, cqx, cqy, cqz, cqw = frame_to_xyzquat(cur_F)
        tx, ty, tz, tqx, tqy, tqz, tqw = frame_to_xyzquat(self.target_F)
        rospy.loginfo(
            "FK TCP: x=%.3f y=%.3f z=%.3f | quat=[%.3f %.3f %.3f %.3f]"
            "TGT   : x=%.3f y=%.3f z=%.3f | quat=[%.3f %.3f %.3f %.3f]",
            cx, cy, cz, cqx, cqy, cqz, cqw,
            tx, ty, tz, tqx, tqy, tqz, tqw
        )

    # ----------------- Main loop ---------------------
    def run(self):
        rate = rospy.Rate(1.0 / self.dt)
        try:
            while not rospy.is_shutdown():
                # 1) publish motion if active
                self._tick_motion()

                # 2) poll key
                ch = self._read_key()
                if ch:
                    code = ord(ch)
                    ch = ch.lower()

                    # quit
                    if code == 27:  # ESC
                        break

                    # translations
                    if ch == 'w': self._try_nudge(dx=+self.step_trans_m)
                    elif ch == 's': self._try_nudge(dx=-self.step_trans_m)
                    elif ch == 'a': self._try_nudge(dy=+self.step_trans_m)
                    elif ch == 'd': self._try_nudge(dy=-self.step_trans_m)
                    elif ch == 'r': self._try_nudge(dz=+self.step_trans_m)
                    elif ch == 'f': self._try_nudge(dz=-self.step_trans_m)

                    # yaw
                    elif ch == 'q': self._try_nudge(yaw=+self.step_yaw_rad)
                    elif ch == 'e': self._try_nudge(yaw=-self.step_yaw_rad)

                    # step size tweaks
                    elif ch == 'z':
                        self.step_trans_m = max(0.0005, self.step_trans_m * 0.5)
                        rospy.loginfo("step_trans: %.1f mm", self.step_trans_m * 1000.0)
                    elif ch == 'x':
                        self.step_trans_m = min(0.05, self.step_trans_m * 2.0)
                        rospy.loginfo("step_trans: %.1f mm", self.step_trans_m * 1000.0)
                    elif ch == 'c':
                        self.step_yaw_rad = max(np.deg2rad(0.5), self.step_yaw_rad * 0.5)
                        rospy.loginfo("step_yaw: %.1f deg", np.rad2deg(self.step_yaw_rad))
                    elif ch == 'v':
                        self.step_yaw_rad = min(np.deg2rad(30.0), self.step_yaw_rad * 2.0)
                        rospy.loginfo("step_yaw: %.1f deg", np.rad2deg(self.step_yaw_rad))

                    # status / help / capture
                    elif ch == 'p':
                        self._print_tcp()
                    elif ch == 'g':
                        self.target_F = self._fk(self.q)
                        rospy.loginfo("Target recaptured from FK.")
                        self._print_tcp()
                    elif ch == 'h':
                        self._print_help()

                    # stop
                    elif ch == ' ':
                        msg = Float64MultiArray()
                        msg.data = [0.0] * self.nj
                        self.pub.publish(msg)
                        self.active_vel_traj = None
                        rospy.loginfo("Stop (zero velocity).")

                rate.sleep()
        finally:
            # On exit, publish a final zero velocity for safety and restore terminal
            try:
                msg = Float64MultiArray()
                msg.data = [0.0] * self.nj
                self.pub.publish(msg)
            except Exception:
                pass
            self._keyboard_restore()

    # ----------------- Nudge op ----------------------
    def _try_nudge(self, dx=0.0, dy=0.0, dz=0.0, yaw=0.0):
        if self.active_vel_traj is not None:
            rospy.logwarn("Busy executing previous nudge; please wait...")
            return

        # 1) edit target frame
        if dx or dy or dz:
            self._shift_target_xyz(dx, dy, dz)
        if yaw:
            self._yaw_target(yaw)

        # 2) solve IK from current joints
        q_goal = self._ik(self.q, self.target_F)
        if q_goal is None:
            rospy.logwarn("IK failed for requested nudge; reverting target.")
            # revert the target frame if IK fails
            if dx or dy or dz:
                self._shift_target_xyz(-dx, -dy, -dz)
            if yaw:
                self._yaw_target(-yaw)
            return

        # 3) plan + execute short segment
        self._plan_and_start(self.q.copy(), q_goal)


if __name__ == '__main__':
    try:
        node = TcpKeyboardTeleop()
        node.run()
    except rospy.ROSInterruptException:
        pass
