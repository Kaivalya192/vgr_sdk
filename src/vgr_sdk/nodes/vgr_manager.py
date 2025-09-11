# ==================================
# FILE: src/vgr_sdk/nodes/vgr_manager.py
# ==================================
from __future__ import annotations
import json, yaml, copy, socket
import rospy
from std_msgs.msg import String
from typing import Tuple, Optional, List, Dict, Any

from vgr_sdk.core.message_types import VisionResult
from vgr_sdk.core.transform import GeomConfig, px_to_world_xy, angle_img_to_world
from vgr_sdk.core.pose_store import PoseStore
from vgr_sdk.tasks.sorting_task import SortingTask, SortingConfig
from vgr_sdk.tasks.cnc_tending_task import CNCTendingTask, CNCTendingConfig
from vgr_sdk.tasks.bin_pick2d_task import BinPick2DTask, BinPick2DConfig
from vgr_sdk.tasks.kitting_task import KittingTask, KittingConfig


class VGRManagerNode:
    def __init__(self):
        self.vision_geom_path = rospy.get_param("~vision_geom", "config/vision_geom.yaml")
        self.task_config_path = rospy.get_param("~task_config", "config/tasks/sorting.yaml")
        self.poses_path = rospy.get_param("~poses_path", "poses/recorded_poses.yaml")
        self.task_name = rospy.get_param("~task", "sorting")
        self.sub_topic = rospy.get_param("~sub_result_topic", "/vgr/vision_result")
        self.pub_topic = rospy.get_param("~pub_plan_topic", "/vgr/plan")
        self.plan_state_topic = rospy.get_param("~sub_plan_state_topic", "/vgr/plan_state")

        self.pi_ip = rospy.get_param("~pi_ip", "10.1.149.182")
        self.pi_port = int(rospy.get_param("~pi_port", 40002))
        self.trigger_kind = rospy.get_param("~trigger_kind", "json")  # "json" or "text"
        self.auto_trigger_after_done = bool(rospy.get_param("~auto_trigger_after_done", True))

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        self.geom, self._swap_xy, self._prehome_xyzyaw, self._home_xyzyaw = self._load_geom(self.vision_geom_path)
        self.pose_store = PoseStore(self.poses_path)
        self.task = self._build_task(self.task_name, self.task_config_path)

        self.sub = rospy.Subscriber(self.sub_topic, String, self._on_result, queue_size=10)
        self.sub_plan = rospy.Subscriber(self.plan_state_topic, String, self._on_plan_state, queue_size=10)
        self.pub_plan = rospy.Publisher(self.pub_topic, String, queue_size=10)

        self._busy: bool = False
        self._last_payload_used: Optional[dict] = None

        rospy.loginfo(f"[vgr_manager] task={self.task_name} sub={self.sub_topic} pub={self.pub_topic} plan_state={self.plan_state_topic}")

    # ------- loaders -------
    def _load_geom(self, path: str) -> tuple[GeomConfig, bool, Optional[tuple], Optional[tuple]]:
        with open(path, "r", encoding="utf-8") as f:
            y = yaml.safe_load(f) or {}
        proc_width  = int(y.get("proc_width", 640))
        proc_height = int(y.get("proc_height", 480))
        mm_per_px_x = float(y.get("mm_per_px_x", 0.25))
        mm_per_px_y = float(y.get("mm_per_px_y", 0.25))
        origin_px   = list(y.get("origin_px", [0.0, 0.0]))
        origin_mm_raw = list(y.get("origin_mm", [0.0, 0.0]))
        invert_x    = bool(y.get("invert_x", False))
        invert_y    = bool(y.get("invert_y", False))
        theta_sign  = int(y.get("theta_sign", 1))
        yaw_offset  = float(y.get("yaw_offset_deg", 0.0))
        plane_z_mm  = float(y.get("plane_z_mm", 0.0))
        swap_xy     = bool(y.get("swap_xy", False))
        if swap_xy:
            proc_width, proc_height = proc_height, proc_width
            mm_per_px_x, mm_per_px_y = mm_per_px_y, mm_per_px_x
            origin_px = [origin_px[1], origin_px[0]]
            rospy.loginfo("[vgr_manager] swap_xy: TRUE")
        origin_mm_xy = (float(origin_mm_raw[0]), float(origin_mm_raw[1])) if len(origin_mm_raw) >= 2 else (0.0, 0.0)
        gc = GeomConfig(
            proc_width=proc_width, proc_height=proc_height,
            mm_per_px_x=mm_per_px_x, mm_per_px_y=mm_per_px_y,
            origin_px=tuple(origin_px), origin_mm=tuple(origin_mm_xy),
            invert_x=invert_x, invert_y=invert_y,
            yaw_offset_deg=yaw_offset, theta_sign=theta_sign, plane_z_mm=plane_z_mm
        )
        prehome_xyzyaw = None
        home_xyzyaw = None
        if len(origin_mm_raw) >= 4:
            x, y, z, yaw = map(float, origin_mm_raw[:4])
            prehome_xyzyaw = (x, y, z, yaw)
            home_xyzyaw = (x, y, z, yaw)
            rospy.loginfo("[vgr_manager] prehome/home: (%.1f, %.1f, %.1f, %.1f)", x, y, z, yaw)
        else:
            rospy.logwarn("[vgr_manager] origin_mm needs 4 values for homing.")
        return gc, swap_xy, prehome_xyzyaw, home_xyzyaw

    def _build_task(self, name: str, path: str):
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        if name == "sorting":
            m = cfg.get("map", {})
            place = {k: tuple(map(float, v)) for k, v in m.items()}
            default = cfg.get("default")
            default = tuple(map(float, default)) if default else None
            return SortingTask(config=SortingConfig(place_xyzyaw_by_name=place, default_place_xyzyaw=default),
                               geom=self.geom, pose_store=self.pose_store)
        if name == "cnc":
            place = tuple(map(float, cfg.get("place_xyzyaw", [0, 0, 0, 0])))
            align = bool(cfg.get("align_theta", True))
            tol = float(cfg.get("theta_tolerance_deg", 7.5))
            scale_tol = float(cfg.get("scale_tolerance", 0.25))
            return CNCTendingTask(config=CNCTendingConfig(place_xyzyaw=place, align_theta=align,
                                                         theta_tolerance_deg=tol, scale_tolerance=scale_tol),
                                  geom=self.geom, pose_store=self.pose_store)
        if name == "bin2d":
            poly = [tuple(map(float, p)) for p in cfg.get("bin_polygon_world_mm", [])]
            clearance = float(cfg.get("wall_clearance_mm", 20.0))
            place = tuple(map(float, cfg.get("place_xyzyaw", [0, 0, 0, 0])))
            return BinPick2DTask(config=BinPick2DConfig(bin_polygon_world_mm=poly,
                                                        wall_clearance_mm=clearance,
                                                        place_xyzyaw=place),
                                 geom=self.geom, pose_store=self.pose_store)
        if name == "kitting":
            slots = {k: [tuple(map(float, p)) for p in v] for k, v in (cfg.get("slots_by_name", {}) or {}).items()}
            allow_fb = bool(cfg.get("allow_fallback", False))
            return KittingTask(config=KittingConfig(slots_by_name=slots, allow_fallback=allow_fb),
                               geom=self.geom, pose_store=self.pose_store)
        raise ValueError(f"Unknown task '{name}'")

    # ------- helpers -------
    def _swap_detection_xy(self, det: dict) -> None:
        p = det.get("pose")
        if isinstance(p, dict) and ("x" in p) and ("y" in p):
            p["x"], p["y"] = p["y"], p["x"]
        c = det.get("center")
        if isinstance(c, (list, tuple)) and len(c) >= 2:
            det["center"] = [c[1], c[0]] + list(c[2:]) if len(c) > 2 else [c[1], c[0]]
        q = det.get("quad")
        if isinstance(q, (list, tuple)) and len(q) == 4:
            det["quad"] = [[pt[1], pt[0]] for pt in q]

    def _swap_payload_xy(self, payload: dict) -> dict:
        out = copy.deepcopy(payload)
        cam = out.get("camera", {})
        if "proc_width" in cam and "proc_height" in cam:
            cam["proc_width"], cam["proc_height"] = cam["proc_height"], cam["proc_width"]
        res = out.get("result", {})
        for obj in res.get("objects", []) or []:
            for det in obj.get("detections", []) or []:
                self._swap_detection_xy(det)
        return out

    def _mk_pose_dict(self, x, y, z, yaw_deg) -> Dict[str, Any]:
        return {"x_mm": x, "y_mm": y, "z_mm": z, "yaw_deg": yaw_deg}

    def _apply_pose_into_waypoint(self, wp: Dict[str, Any], xyzyaw: tuple) -> Dict[str, Any]:
        x, y, z, yaw = xyzyaw
        out = copy.deepcopy(wp)
        if "pose" in out and isinstance(out["pose"], dict):
            out["pose"].update(self._mk_pose_dict(x, y, z, yaw))
        else:
            out.update(self._mk_pose_dict(x, y, z, yaw))
        out["tag"] = out.get("tag", "move")
        out["motion"] = out.get("motion", "joint")
        return out

    def _append_homing_waypoints(self, waypoints: List[Dict[str, Any]]) -> None:
        if not waypoints:
            return
        if not (self._prehome_xyzyaw and self._home_xyzyaw):
            rospy.logwarn("[vgr_manager] homing skipped (origin_mm missing 4 values)")
            return
        last_wp = waypoints[-1]
        pre_wp  = self._apply_pose_into_waypoint(last_wp, self._prehome_xyzyaw)
        home_wp = self._apply_pose_into_waypoint(pre_wp,  self._home_xyzyaw)
        pre_wp["tag"]  = "prehome"
        home_wp["tag"] = "home"
        waypoints.extend([pre_wp, home_wp])

    def _trigger_detection(self):
        try:
            if self.trigger_kind == "text":
                self.sock.sendto(b"TRIGGER", (self.pi_ip, self.pi_port))
            else:
                msg = json.dumps({"cmd": "trigger"})
                self.sock.sendto(msg.encode("utf-8"), (self.pi_ip, self.pi_port))
            rospy.loginfo("[vgr_manager] UDP trigger sent to %s:%d", self.pi_ip, self.pi_port)
        except Exception as e:
            rospy.logwarn(f"[vgr_manager] trigger failed: {e}")

    def _select_next_target(self, payload: dict) -> Optional[Dict[str, Any]]:
        res = (payload or {}).get("result") or {}
        objs = [o for o in (res.get("objects") or []) if (o.get("detections") or [])]
        if not objs:
            return None
        objs.sort(key=lambda o: str(o.get("name", "")).lower())
        o0 = objs[0]
        dets = o0.get("detections") or []
        def det_key(d):
            sc = d.get("score", None)
            il = d.get("inliers", None)
            return (float(sc) if sc is not None else -1.0, int(il) if il is not None else -1)
        dets_sorted = sorted(dets, key=det_key, reverse=True)
        d0 = dets_sorted[0]
        return {
            "object_id": o0.get("object_id"),
            "name": o0.get("name"),
            "instance_id": d0.get("instance_id", None),
            "center": d0.get("center", None)
        }

    def _filter_payload_for_pick(self, payload: dict, sel: Dict[str, Any]) -> dict:
        out = {"version": payload.get("version"),
               "sdk": payload.get("sdk"),
               "session": payload.get("session"),
               "timestamp_ms": payload.get("timestamp_ms"),
               "camera": payload.get("camera"),
               "result": {"counts": {"objects": 1, "detections": 1}, "objects": []}}
        for obj in (payload.get("result", {}).get("objects") or []):
            if obj.get("name") != sel.get("name"):
                continue
            keep_det = None
            for d in (obj.get("detections") or []):
                if sel.get("instance_id") is not None and d.get("instance_id") == sel["instance_id"]:
                    keep_det = d; break
            if keep_det is None and sel.get("center") and (obj.get("detections") or []):
                cx, cy = sel["center"][:2]
                for d in obj["detections"]:
                    dc = d.get("center")
                    if isinstance(dc, (list, tuple)) and abs(dc[0]-cx) < 2.0 and abs(dc[1]-cy) < 2.0:
                        keep_det = d; break
            if keep_det is None and (obj.get("detections") or []):
                keep_det = (obj["detections"])[0]
            new_obj = {
                "object_id": obj.get("object_id"),
                "name": obj.get("name"),
                "template_size": obj.get("template_size"),
                "detections": [keep_det] if keep_det else []
            }
            out["result"]["objects"].append(new_obj)
            break
        return out

    def _publish_plan_for_payload(self, payload_used: dict):
        try:
            vr = VisionResult.from_json(payload_used)
        except Exception as e:
            rospy.logwarn(f"[vgr_manager] VisionResult parse failed: {e}")
            return
        try:
            plan, pick = self.task.plan_cycle(vr)  # type: ignore[attr-defined]
        except Exception as e:
            rospy.logwarn(f"[vgr_manager] planning error: {e}")
            return
        if not plan or not pick:
            return
        pd = plan.to_dict()
        wps = pd.get("waypoints", [])
        self._append_homing_waypoints(wps)
        pd["waypoints"] = wps
        out = {"plan": pd, "pick": pick}
        self.pub_plan.publish(String(data=json.dumps(out, separators=(",", ":"))))
        g = pick.get("grasp", {})
        rospy.loginfo(
            f"[vgr_manager] plan published (+homing): open={g.get('recommended_open_mm', 0):.1f} mm, "
            f"pick=({pick['pose_world']['x_mm']:.1f},{pick['pose_world']['y_mm']:.1f},{pick['pose_world']['z_mm']:.1f}) "
            f"→ waypoints={len(pd.get('waypoints', []))}"
        )
        self._busy = True

    # ------- callbacks -------
    def _on_plan_state(self, msg: String):
        s = (msg.data or "").strip().lower()
        if any(k in s for k in ("done", "success", "complete", "completed", "ok", "idle")):
            self._busy = False
            if self.auto_trigger_after_done:
                self._trigger_detection()

    def _on_result(self, msg: String):
        try:
            payload_raw = json.loads(msg.data)
        except Exception:
            rospy.logwarn("[vgr_manager] invalid JSON on /vgr/vision_result")
            return
        payload_used = self._swap_payload_xy(payload_raw) if self._swap_xy else payload_raw
        self._last_payload_used = payload_used
        if self._busy:
            return
        sel = self._select_next_target(payload_used)
        if not sel:
            return
        filtered = self._filter_payload_for_pick(payload_used, sel)
        self._publish_plan_for_payload(filtered)

def main():
    rospy.init_node("vgr_manager_node", anonymous=False)
    VGRManagerNode()
    rospy.spin()

if __name__ == "__main__":
    main()
