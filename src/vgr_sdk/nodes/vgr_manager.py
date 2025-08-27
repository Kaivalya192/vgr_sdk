# ==================================
# FILE: src/vgr_sdk/nodes/vgr_manager.py
# ==================================
from __future__ import annotations
import json
import yaml
import rospy
from std_msgs.msg import String

from vgr_sdk.core.message_types import VisionResult
from vgr_sdk.core.transform import GeomConfig
from vgr_sdk.core.pose_store import PoseStore
from vgr_sdk.tasks.sorting_task import SortingTask, SortingConfig
from vgr_sdk.tasks.cnc_tending_task import CNCTendingTask, CNCTendingConfig
from vgr_sdk.tasks.bin_pick2d_task import BinPick2DTask, BinPick2DConfig
from vgr_sdk.tasks.kitting_task import KittingTask, KittingConfig


class VGRManagerNode:
    """
    Parameters (~ namespace):
      ~vision_geom     : path to config/vision_geom.yaml
      ~task_config     : path to config/tasks/<task>.yaml
      ~poses_path      : path to poses/recorded_poses.yaml
      ~task            : one of ["sorting","cnc","bin2d","kitting"]  (matches .yaml chosen)
      ~pub_plan_topic  : "/vgr/plan"
      ~sub_result_topic: "/vgr/vision_result"
    """
    def __init__(self):
        self.vision_geom_path = rospy.get_param("~vision_geom", "config/vision_geom.yaml")
        self.task_config_path = rospy.get_param("~task_config", "config/tasks/sorting.yaml")
        self.poses_path = rospy.get_param("~poses_path", "poses/recorded_poses.yaml")
        self.task_name = rospy.get_param("~task", "sorting")
        self.sub_topic = rospy.get_param("~sub_result_topic", "/vgr/vision_result")
        self.pub_topic = rospy.get_param("~pub_plan_topic", "/vgr/plan")

        self.geom = self._load_geom(self.vision_geom_path)
        self.pose_store = PoseStore(self.poses_path)

        self.task = self._build_task(self.task_name, self.task_config_path)

        self.sub = rospy.Subscriber(self.sub_topic, String, self._on_result, queue_size=10)
        self.pub_plan = rospy.Publisher(self.pub_topic, String, queue_size=10)

        rospy.loginfo(f"[vgr_manager] task={self.task_name} sub={self.sub_topic} pub={self.pub_topic}")

    # ------- loaders -------
    def _load_geom(self, path: str) -> GeomConfig:
        with open(path, "r", encoding="utf-8") as f:
            y = yaml.safe_load(f) or {}
        return GeomConfig(
            proc_width=int(y.get("proc_width", 640)),
            proc_height=int(y.get("proc_height", 480)),
            mm_per_px_x=float(y.get("mm_per_px_x", 0.25)),
            mm_per_px_y=float(y.get("mm_per_px_y", 0.25)),
            origin_px=tuple(y.get("origin_px", [0.0, 0.0])),
            origin_mm=tuple(y.get("origin_mm", [0.0, 0.0])),
            invert_x=bool(y.get("invert_x", False)),
            invert_y=bool(y.get("invert_y", False)),
            yaw_offset_deg=float(y.get("yaw_offset_deg", 0.0)),
            theta_sign=int(y.get("theta_sign", 1)),
            plane_z_mm=float(y.get("plane_z_mm", 0.0)),
        )

    def _build_task(self, name: str, path: str):
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

        if name == "sorting":
            m = cfg.get("map", {})  # {name: [x,y,z,yaw]}
            place = {k: tuple(map(float, v)) for k, v in m.items()}
            default = cfg.get("default")
            default = tuple(map(float, default)) if default else None
            return SortingTask(
                config=SortingConfig(place_xyzyaw_by_name=place, default_place_xyzyaw=default),
                geom=self.geom, pose_store=self.pose_store
            )

        if name == "cnc":
            place = tuple(map(float, cfg.get("place_xyzyaw", [0, 0, 0, 0])))
            align = bool(cfg.get("align_theta", True))
            tol = float(cfg.get("theta_tolerance_deg", 7.5))
            scale_tol = float(cfg.get("scale_tolerance", 0.25))
            return CNCTendingTask(
                config=CNCTendingConfig(place_xyzyaw=place, align_theta=align,
                                        theta_tolerance_deg=tol, scale_tolerance=scale_tol),
                geom=self.geom, pose_store=self.pose_store
            )

        if name == "bin2d":
            poly = [tuple(map(float, p)) for p in cfg.get("bin_polygon_world_mm", [])]
            clearance = float(cfg.get("wall_clearance_mm", 20.0))
            place = tuple(map(float, cfg.get("place_xyzyaw", [0, 0, 0, 0])))
            return BinPick2DTask(
                config=BinPick2DConfig(bin_polygon_world_mm=poly,
                                       wall_clearance_mm=clearance,
                                       place_xyzyaw=place),
                geom=self.geom, pose_store=self.pose_store
            )

        if name == "kitting":
            slots = {k: [tuple(map(float, p)) for p in v] for k, v in (cfg.get("slots_by_name", {}) or {}).items()}
            allow_fb = bool(cfg.get("allow_fallback", False))
            return KittingTask(
                config=KittingConfig(slots_by_name=slots, allow_fallback=allow_fb),
                geom=self.geom, pose_store=self.pose_store
            )

        raise ValueError(f"Unknown task '{name}'")

    # ------- callback -------
    def _on_result(self, msg: String):
        try:
            payload = json.loads(msg.data)
        except Exception:
            rospy.logwarn("[vgr_manager] invalid JSON on /vgr/vision_result")
            return
        vr = VisionResult.from_json(payload)

        # Plan cycle
        try:
            plan, pick = self.task.plan_cycle(vr)  # type: ignore[attr-defined]
        except Exception as e:
            rospy.logwarn(f"[vgr_manager] planning error: {e}")
            return

        if not plan or not pick:
            return

        # Publish plan as JSON
        out = {
            "plan": plan.to_dict(),
            "pick": pick,  # enriched detection with world fields + grasp hints
        }
        self.pub_plan.publish(String(data=json.dumps(out, separators=(",", ":"))))

        # Log quick summary
        g = pick.get("grasp", {})
        rospy.loginfo(f"[vgr_manager] plan ready: open={g.get('recommended_open_mm', 0):.1f} mm, "
                      f"pick=({pick['pose_world']['x_mm']:.1f},{pick['pose_world']['y_mm']:.1f},{pick['pose_world']['z_mm']:.1f}) "
                      f"→ waypoints={len(plan.waypoints)}")
