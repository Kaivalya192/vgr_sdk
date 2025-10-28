import numpy as np
import scipy.linalg
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp


class TrajectoryPlanner:
    def __init__(self):
        pass

    def quintic_joint_trajectory(self, start_joints: np.ndarray, goal_joints: np.ndarray, T: float, dt: float = 0.01):
        """
        Computes a quintic polynomial trajectory for joint positions.

        Parameters:
        start_joints (np.ndarray): Initial joint positions (n-dimensional vector)
        goal_joints (np.ndarray): Final joint positions (n-dimensional vector)
        T (float): Total duration of motion
        dt (float): Time step for sampling trajectory

        Returns:
        np.ndarray: Joint position trajectory (N x n), where N = T/dt
        np.ndarray: Joint velocity trajectory (N x n)
        np.ndarray: Joint acceleration trajectory (N x n)
        """
        t = np.arange(0, T + dt, dt)
        N = len(t)
        
        # Boundary conditions
        J0, Jf = start_joints, goal_joints
        V0, Vf = np.zeros_like(J0), np.zeros_like(J0)  # Zero velocity at start and end
        A0, Af = np.zeros_like(J0), np.zeros_like(J0)  # Zero acceleration at start and end
        
        # Solve for coefficients of the quintic polynomial
        M = np.array([
            [1, 0, 0, 0, 0, 0],
            [1, T, T**2, T**3, T**4, T**5],
            [0, 1, 0, 0, 0, 0],
            [0, 1, 2*T, 3*T**2, 4*T**3, 5*T**4],
            [0, 0, 2, 0, 0, 0],
            [0, 0, 2, 6*T, 12*T**2, 20*T**3]
        ])
        
        n = J0.shape[0]
        coeffs = np.zeros((n, 6))  # Coefficients for each joint
        
        for i in range(n):
            b = np.array([J0[i], Jf[i], V0[i], Vf[i], A0[i], Af[i]])
            coeffs[i, :] = np.linalg.solve(M, b)
        
        joint_positions = np.zeros((N, n))
        joint_velocities = np.zeros((N, n))
        joint_accelerations = np.zeros((N, n))
        
        for i in range(N):
            t_i = t[i]
            T_vec = np.array([1, t_i, t_i**2, t_i**3, t_i**4, t_i**5])
            dT_vec = np.array([0, 1, 2*t_i, 3*t_i**2, 4*t_i**3, 5*t_i**4])
            ddT_vec = np.array([0, 0, 2, 6*t_i, 12*t_i**2, 20*t_i**3])
            
            for j in range(n):
                joint_positions[i, j] = np.dot(coeffs[j], T_vec)
                joint_velocities[i, j] = np.dot(coeffs[j], dT_vec)
                joint_accelerations[i, j] = np.dot(coeffs[j], ddT_vec)
        
        return joint_positions, joint_velocities, joint_accelerations

    def quintic_position_trajectory(self, start_pos: np.ndarray, goal_pos: np.ndarray, T: float, dt: float = 0.01):
        """
        Computes a quintic polynomial trajectory for position.

        Parameters:
        start_pos (np.ndarray): Initial position (x, y, z)
        goal_pos (np.ndarray): Final position (x, y, z)
        T (float): Total duration of motion
        dt (float): Time step for sampling trajectory

        Returns:
        np.ndarray: Position trajectory (N x 3), where N = T/dt
        np.ndarray: Velocity trajectory (N x 3)
        np.ndarray: Acceleration trajectory (N x 3)
        """
        t = np.arange(0, T + dt, dt)
        N = len(t)
        
        # Boundary conditions
        X0, Xf = start_pos, goal_pos
        V0, Vf = np.zeros(3), np.zeros(3)  # Zero velocity at start and end
        A0, Af = np.zeros(3), np.zeros(3)  # Zero acceleration at start and end
        
        # Solve for coefficients of the quintic polynomial
        M = np.array([
            [1, 0, 0, 0, 0, 0],
            [1, T, T**2, T**3, T**4, T**5],
            [0, 1, 0, 0, 0, 0],
            [0, 1, 2*T, 3*T**2, 4*T**3, 5*T**4],
            [0, 0, 2, 0, 0, 0],
            [0, 0, 2, 6*T, 12*T**2, 20*T**3]
        ])
        
        coeffs = np.zeros((3, 6))  # One set of coefficients for each dimension
        
        for i in range(3):
            b = np.array([X0[i], Xf[i], V0[i], Vf[i], A0[i], Af[i]])
            coeffs[i, :] = np.linalg.solve(M, b)
        
        # Generate trajectory
        position_traj = np.zeros((N, 3))
        velocity_traj = np.zeros((N, 3))
        acceleration_traj = np.zeros((N, 3))
        
        for i in range(N):
            t_i = t[i]
            T_vec = np.array([1, t_i, t_i**2, t_i**3, t_i**4, t_i**5])
            dT_vec = np.array([0, 1, 2*t_i, 3*t_i**2, 4*t_i**3, 5*t_i**4])
            ddT_vec = np.array([0, 0, 2, 6*t_i, 12*t_i**2, 20*t_i**3])
            
            for j in range(3):
                position_traj[i, j] = np.dot(coeffs[j], T_vec)
                velocity_traj[i, j] = np.dot(coeffs[j], dT_vec)
                acceleration_traj[i, j] = np.dot(coeffs[j], ddT_vec)
        
        return position_traj, velocity_traj, acceleration_traj

    def slerp_orientation_trajectory(self, start_quat: np.ndarray, goal_quat: np.ndarray, T: float, dt: float = 0.01):
        """
        Computes a SLERP (Spherical Linear Interpolation) trajectory for orientation.

        Parameters:
        start_quat (np.ndarray): Initial quaternion (x, y, z, w)
        goal_quat (np.ndarray): Final quaternion (x, y, z, w)
        T (float): Total duration of motion
        dt (float): Time step for sampling trajectory

        Returns:
        np.ndarray: Orientation trajectory as quaternions (N x 4), where N = T/dt
        """
        t = np.arange(0, T + dt, dt)
        N = len(t)
        
        # Define key times (e.g., start at time 0 and goal at time T)
        key_times = [0, T]
        
        # Create a Rotation object for the start and goal quaternions
        key_rots = R.from_quat([start_quat, goal_quat])
        
        # Create a Slerp instance
        slerp = Slerp(key_times, key_rots)
        
        # Interpolate for all time steps at once
        interp_rots = slerp(t)
        
        # Extract quaternions from the interpolated rotations
        orientation_traj = interp_rots.as_quat()
        
        return orientation_traj

    def quintic_joint_trajectory(self, start_joints: np.ndarray, goal_joints: np.ndarray, T: float, dt: float = 0.01):
        """
        Computes a quintic polynomial trajectory for joint positions.

        Parameters:
        start_joints (np.ndarray): Initial joint positions (n-dimensional vector)
        goal_joints (np.ndarray): Final joint positions (n-dimensional vector)
        T (float): Total duration of motion
        dt (float): Time step for sampling trajectory

        Returns:
        np.ndarray: Joint position trajectory (N x n), where N = T/dt
        np.ndarray: Joint velocity trajectory (N x n)
        np.ndarray: Joint acceleration trajectory (N x n)
        """
        t = np.arange(0, T + dt, dt)
        N = len(t)
        
        # Boundary conditions
        J0, Jf = start_joints, goal_joints
        V0, Vf = np.zeros_like(J0), np.zeros_like(J0)  # Zero velocity at start and end
        A0, Af = np.zeros_like(J0), np.zeros_like(J0)  # Zero acceleration at start and end
        
        # Solve for coefficients of the quintic polynomial
        M = np.array([
            [1, 0, 0, 0, 0, 0],
            [1, T, T**2, T**3, T**4, T**5],
            [0, 1, 0, 0, 0, 0],
            [0, 1, 2*T, 3*T**2, 4*T**3, 5*T**4],
            [0, 0, 2, 0, 0, 0],
            [0, 0, 2, 6*T, 12*T**2, 20*T**3]
        ])
        
        n = J0.shape[0]
        coeffs = np.zeros((n, 6))  # Coefficients for each joint
        
        for i in range(n):
            b = np.array([J0[i], Jf[i], V0[i], Vf[i], A0[i], Af[i]])
            coeffs[i, :] = np.linalg.solve(M, b)
        
        joint_positions = np.zeros((N, n))
        joint_velocities = np.zeros((N, n))
        joint_accelerations = np.zeros((N, n))
        
        for i in range(N):
            t_i = t[i]
            T_vec = np.array([1, t_i, t_i**2, t_i**3, t_i**4, t_i**5])
            dT_vec = np.array([0, 1, 2*t_i, 3*t_i**2, 4*t_i**3, 5*t_i**4])
            ddT_vec = np.array([0, 0, 2, 6*t_i, 12*t_i**2, 20*t_i**3])
            
            for j in range(n):
                joint_positions[i, j] = np.dot(coeffs[j], T_vec)
                joint_velocities[i, j] = np.dot(coeffs[j], dT_vec)
                joint_accelerations[i, j] = np.dot(coeffs[j], ddT_vec)
        
        return joint_positions, joint_velocities, joint_accelerations