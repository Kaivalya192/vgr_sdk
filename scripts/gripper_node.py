#!/usr/bin/env python3
# =================================
# FILE: scripts/gripper_node.py
# =================================
from __future__ import annotations
import json
import importlib
import rospy
from std_msgs.msg import String, Float64, Bool

# -------- Backend interfaces --------
class BaseBackend:
    def __init__(self, stroke_min_mm: float, stroke_max_mm: float):
        self.stroke_min_mm = float(stroke_min_mm)
        self.stroke_max_mm = float(stroke_max_mm)

    def open_mm(self, opening_mm: float) -> None:
        raise NotImplementedError

    def close_mm(self, opening_mm: float) -> None:
        raise NotImplementedError

    def close_force(self, force_n: float) -> None:
        raise NotImplementedError

    def get_status(self) -> dict:
        """Return dict with at least: opening_mm: float, object_detected: bool, ok: bool, error: str"""
        raise NotImplementedError


class SimBackend(BaseBackend):
    def __init__(self, stroke_min_mm: float, stroke_max_mm: float):
        super().__init__(stroke_min_mm, stroke_max_mm)
        self._opening_mm = float(stroke_min_mm)
        self._object = False
        self._ok = True
        self._err = ""

    def _clamp(self, v: float) -> float:
        if v < self.stroke_min_mm:
            return self.stroke_min_mm
        if v > self.stroke_max_mm:
            return self.stroke_max_mm
        return v

    def open_mm(self, opening_mm: float) -> None:
        self._opening_mm = self._clamp(opening_mm)
        self._object = False

    def close_mm(self, opening_mm: float) -> None:
        self._opening_mm = self._clamp(opening_mm)
        # Simulate object contact
        self._object = True

    def close_force(self, force_n: float) -> None:
        # Force close without changing width in this simple sim
        self._object = True

    def get_status(self) -> dict:
        return {
            "opening_mm": float(self._opening_mm),
            "object_detected": bool(self._object),
            "ok": bool(self._ok),
            "error": str(self._err),
        }


# -------- Node --------
class GripperNode:
    """
    Topics:
      ~cmd            (std_msgs/String, JSON): {"cmd":"open","open_mm":30} | {"cmd":"close","open_mm":20} | {"cmd":"close_force","force_n":50} | {"cmd":"status"}
      ~open_mm        (std_msgs/Float64): open to width (mm)
      ~close_mm       (std_msgs/Float64): close to width (mm)
      ~close_force_n  (std_msgs/Float64): close with force (N)
      ~status         (std_msgs/String, JSON): periodic status message

    Params:
      ~stroke_min_mm        : 5.0
      ~stroke_max_mm        : 80.0
      ~publish_rate_hz      : 2.0
      ~driver_module        : python import path to backend (optional); must expose class 'Backend'
                              with same interface as BaseBackend(open_mm/close_mm/close_force/get_status)
    """
    def __init__(self):
        self.stroke_min_mm = float(rospy.get_param("~stroke_min_mm", 5.0))
        self.stroke_max_mm = float(rospy.get_param("~stroke_max_mm", 80.0))
        self.rate_hz = float(rospy.get_param("~publish_rate_hz", 2.0))
        driver_module = rospy.get_param("~driver_module", "")

        # Backend
        if driver_module:
            try:
                mod = importlib.import_module(driver_module)
                Backend = getattr(mod, "Backend")
                self.backend = Backend(self.stroke_min_mm, self.stroke_max_mm)
                rospy.loginfo(f"[gripper_node] Using driver backend: {driver_module}.Backend")
            except Exception as e:
                rospy.logwarn(f"[gripper_node] Failed to load driver backend '{driver_module}': {e}; using SimBackend")
                self.backend = SimBackend(self.stroke_min_mm, self.stroke_max_mm)
        else:
            self.backend = SimBackend(self.stroke_min_mm, self.stroke_max_mm)
            rospy.loginfo("[gripper_node] Using SimBackend")

        # Topics
        self.pub_status = rospy.Publisher("~status", String, queue_size=10)
        rospy.Subscriber("~cmd", String, self._on_cmd, queue_size=10)
        rospy.Subscriber("~open_mm", Float64, self._on_open_mm, queue_size=10)
        rospy.Subscriber("~close_mm", Float64, self._on_close_mm, queue_size=10)
        rospy.Subscriber("~close_force_n", Float64, self._on_close_force, queue_size=10)

        self._timer = rospy.Timer(rospy.Duration(1.0 / max(0.1, self.rate_hz)), self._on_timer)

    # ---- Callbacks ----
    def _on_cmd(self, msg: String):
        try:
            cmd = json.loads(msg.data)
            c = str(cmd.get("cmd", "")).lower()
            if c == "open":
                self.backend.open_mm(float(cmd.get("open_mm", self.stroke_max_mm)))
            elif c == "close":
                self.backend.close_mm(float(cmd.get("open_mm", self.stroke_min_mm)))
            elif c in ("close_force", "force"):
                self.backend.close_force(float(cmd.get("force_n", 0.0)))
            elif c == "status":
                self._publish_status()
            else:
                rospy.logwarn(f"[gripper_node] Unknown cmd: {cmd}")
        except Exception as e:
            rospy.logwarn(f"[gripper_node] bad JSON on ~cmd: {e}")

    def _on_open_mm(self, msg: Float64):
        self.backend.open_mm(float(msg.data))

    def _on_close_mm(self, msg: Float64):
        self.backend.close_mm(float(msg.data))

    def _on_close_force(self, msg: Float64):
        self.backend.close_force(float(msg.data))

    def _on_timer(self, _evt):
        self._publish_status()

    # ---- Utils ----
    def _publish_status(self):
        st = self.backend.get_status()
        self.pub_status.publish(String(data=json.dumps(st, separators=(",", ":"))))


def main():
    rospy.init_node("vgr_gripper", anonymous=False)
    GripperNode()
    rospy.spin()


if __name__ == "__main__":
    main()
