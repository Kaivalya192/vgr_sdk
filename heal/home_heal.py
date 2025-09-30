#!/usr/bin/env python3
"""
Move to Target Pose Script

This script moves the HEAL robotic arm from its current position to a specified
Cartesian target pose using velocity control, with a 90° yaw rotation applied.
"""

import rospy
import numpy as np
import PyKDL as kdl
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
from urdf_parser_py.urdf import URDF
from kdl_parser_py.urdf import treeFromParam
import os
import sys

# Add custom utils directory to import trajectory planner
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.TRAJECTORY_PLANNERS.trajectory_planners import TrajectoryPlanner


class MoveToTargetCommander:
    def __init__(self):
        rospy.init_node("move_to_target_commander", anonymous=True)

        # Publisher for velocity control commands
        self.pub = rospy.Publisher(
            "/velocity_controller/command", Float64MultiArray, queue_size=10
        )

        # Subscriber to joint state feedback
        self.joint_state_sub = rospy.Subscriber(
            "/joint_states", JointState, self.joint_state_callback
        )

        # Trajectory planner (quintic polynomial)
        self.traj_planner = TrajectoryPlanner()

        # State Variables
        self.current_joint_state = None
        self.trajectory_generated = False
        self.velocity_traj = None
        self.current_index = 0
        self.dt = 0.001  # 1ms control loop

        # Load URDF and KDL chain for IK
        if not rospy.has_param("robot_description"):
            rospy.logerr("robot_description param not found.")
            exit(1)
        self.robot = URDF.from_parameter_server()
        success, tree = treeFromParam("robot_description")
        if not success:
            rospy.logerr("Failed to parse KDL tree from URDF.")
            exit(1)

        self.base_link = "base_link"
        self.tip_link = "tool_ff"
        self.chain = tree.getChain(self.base_link, self.tip_link)
        self.n_joints = self.chain.getNrOfJoints()
        rospy.loginfo("KDL chain created with %d joints", self.n_joints)

        # IK solver (Levenberg–Marquardt)
        self.ik_solver = kdl.ChainIkSolverPos_LMA(self.chain)

        # ---- Target pose from vision (mm, deg) ----
        x_mm   = rospy.get_param("~target_x_mm",   148.69202330706582)
        y_mm   = rospy.get_param("~target_y_mm",   365.8440789237809)
        z_mm   = rospy.get_param("~target_z_mm",   623.4898940655077)
        roll_d = rospy.get_param("~target_roll_deg",   0.0)   # about X
        pitch_d= rospy.get_param("~target_pitch_deg",  0.0)   # about Y
        yaw_d  = rospy.get_param("~target_yaw_deg",    19.937607757850362)  # about Z

        # Convert units (KDL expects meters/radians)
        target_pos = kdl.Vector(x_mm/1000.0, y_mm/1000.0, z_mm/1000.0)
        new_rot = kdl.Rotation.RPY(np.deg2rad(roll_d), np.deg2rad(pitch_d), np.deg2rad(yaw_d))

        self.target_frame = kdl.Frame(new_rot, target_pos)

        rospy.loginfo("Target position (m): x=%.6f y=%.6f z=%.6f",
                    target_pos.x(), target_pos.y(), target_pos.z())
        qx, qy, qz, qw = new_rot.GetQuaternion()
        rospy.loginfo("Target orientation (quat): x=%.6f y=%.6f z=%.6f w=%.6f", qx, qy, qz, qw)
        rospy.loginfo("Target RPY (deg): roll=%.3f pitch=%.3f yaw=%.3f", roll_d, pitch_d, yaw_d)

        rospy.loginfo(
            "Target position: (%.6f, %.6f, %.6f)",
            target_pos.x(),
            target_pos.y(),
            target_pos.z(),
        )

        # Wait for joint state to be received
        rospy.loginfo("Waiting for joint state...")
        while self.current_joint_state is None and not rospy.is_shutdown():
            rospy.sleep(0.1)
        rospy.loginfo("Joint state received. Ready to move.")

    def compute_ik(self, q_init):
        """
        Solve inverse kinematics for the target pose.
        Returns joint angles (numpy array) if successful, else None.
        """
        q_out = kdl.JntArray(self.n_joints)
        result = self.ik_solver.CartToJnt(q_init, self.target_frame, q_out)
        if result >= 0:
            return np.array([q_out[i] for i in range(self.n_joints)])
        else:
            rospy.logerr("IK failed with error code: %d", result)
            return None

    def joint_state_callback(self, msg):
        """
        Callback to update current joint state and generate trajectory.
        """
        self.current_joint_state = np.array(msg.position)

        if not self.trajectory_generated and self.current_joint_state is not None:
            q_init = kdl.JntArray(self.n_joints)
            for i in range(self.n_joints):
                q_init[i] = self.current_joint_state[i]

            target_joint_state = self.compute_ik(q_init)
            if target_joint_state is None:
                return

            T = 7  # seconds
            _, self.velocity_traj, _ = self.traj_planner.quintic_joint_trajectory(
                self.current_joint_state, target_joint_state, T, self.dt
            )

            self.trajectory_generated = True
            self.current_index = 0
            rospy.loginfo("Trajectory generated. Starting motion...")

    def run(self):
        """
        Main control loop to publish joint velocity commands.
        """
        rate = rospy.Rate(1.0 / self.dt)
        while not rospy.is_shutdown():
            if self.trajectory_generated and self.current_index < len(self.velocity_traj):
                msg = Float64MultiArray()
                msg.data = self.velocity_traj[self.current_index].tolist()
                self.pub.publish(msg)
                self.current_index += 1

                # Print progress occasionally
                if self.current_index % 1000 == 0:
                    progress = (self.current_index / len(self.velocity_traj)) * 100
                    rospy.loginfo("Motion progress: %.1f%%", progress)

            elif self.trajectory_generated and self.current_index >= len(
                self.velocity_traj
            ):
                # Trajectory complete
                rospy.loginfo("Target pose reached successfully!")
                rospy.signal_shutdown("Motion complete")

            rate.sleep()


if __name__ == "__main__":
    try:
        commander = MoveToTargetCommander()
        commander.run()
    except rospy.ROSInterruptException:
        pass
