# ================================
# FILE: src/vgr_sdk/core/robot_io.py
# ================================
"""
Transport-agnostic interfaces for robot & gripper I/O.

No ROS imports here—this module defines *protocols* and simple
dummy implementations useful for local testing or unit tests.

Your ROS nodes can implement these interfaces and wire them up
to actual topics/services/controllers.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol, runtime_checkable, List, Tuple, Optional


# --------- Protocols (interfaces) ---------
@runtime_checkable
class RobotInterface(Protocol):
    def get_joint_positions(self) -> List[float]: ...
    def get_tcp_pose(self) -> Tuple[float, float, float, float]:  # x_mm, y_mm, z_mm, yaw_deg
        ...
    def move_joint(self, joints: List[float], *, speed: float) -> None: ...
    def move_linear(self, x_mm: float, y_mm: float, z_mm: float, yaw_deg: float, *, speed: float) -> None: ...
    def stop(self) -> None: ...
    def is_busy(self) -> bool: ...


@runtime_checkable
class GripperInterface(Protocol):
    def open_mm(self, opening_mm: float) -> None: ...
    def close_mm(self, opening_mm: float) -> None: ...
    def close_force(self, force_n: float) -> None: ...
    def get_opening_mm(self) -> float: ...
    def is_object_detected(self) -> bool: ...


# --------- Dummy (logging) implementations ---------
@dataclass
class DummyRobot(RobotInterface):
    _joints: List[float]
    _tcp: Tuple[float, float, float, float]  # x,y,z,yaw
    _busy: bool = False

    def get_joint_positions(self) -> List[float]:
        return list(self._joints)

    def get_tcp_pose(self) -> Tuple[float, float, float, float]:
        return tuple(self._tcp)

    def move_joint(self, joints: List[float], *, speed: float) -> None:
        self._busy = True
        # emulate immediate move for testing
        self._joints = list(joints)
        self._busy = False
        self._tcp = self._tcp  # unchanged in dummy
        print(f"[DummyRobot] move_joint -> {joints} @ {speed}")

    def move_linear(self, x_mm: float, y_mm: float, z_mm: float, yaw_deg: float, *, speed: float) -> None:
        self._busy = True
        self._tcp = (float(x_mm), float(y_mm), float(z_mm), float(yaw_deg))
        self._busy = False
        print(f"[DummyRobot] move_linear -> xyz({x_mm:.1f},{y_mm:.1f},{z_mm:.1f}) yaw({yaw_deg:.1f}) @ {speed}")

    def stop(self) -> None:
        self._busy = False
        print("[DummyRobot] stop")

    def is_busy(self) -> bool:
        return self._busy


@dataclass
class DummyGripper(GripperInterface):
    _opening_mm: float = 0.0
    _object: bool = False

    def open_mm(self, opening_mm: float) -> None:
        self._opening_mm = float(opening_mm)
        self._object = False
        print(f"[DummyGripper] open -> {self._opening_mm:.1f} mm")

    def close_mm(self, opening_mm: float) -> None:
        self._opening_mm = float(opening_mm)
        # In a real driver, object detection would be set based on torque/current/width thresholds.
        self._object = True
        print(f"[DummyGripper] close -> {self._opening_mm:.1f} mm (simulated object)")

    def close_force(self, force_n: float) -> None:
        print(f"[DummyGripper] close_force -> {force_n:.1f} N (no width change)")

    def get_opening_mm(self) -> float:
        return self._opening_mm

    def is_object_detected(self) -> bool:
        return self._object
