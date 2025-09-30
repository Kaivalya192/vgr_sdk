#!/usr/bin/env python3
import os, copy, math, json, yaml, rospy
from typing import Dict, Any, Optional, Tuple, List
from std_msgs.msg import String

def _dist(a, b):
    return math.hypot(float(a[0]) - float(a[1]), float(b[0]) - float(b[1]))

def _best_det(obj: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    dets = obj.get("detections") or []
    if not dets: return None
    def key(d):
        s = d.get("score"); i = d.get("inliers")
        return (float(s) if s is not None else -1.0, int(i) if i is not None else -1)
    return sorted(dets, key=key, reverse=True)[0]

def _size_px(det: Dict[str, Any], template_size: Optional[List[float]]) -> Optional[Tuple[float,float]]:
    q = det.get("quad")
    if isinstance(q, (list,tuple)) and len(q) == 4:
        w1 = math.hypot(q[1][0]-q[0][0], q[1][1]-q[0][1])
        w2 = math.hypot(q[2][0]-q[3][0], q[2][1]-q[3][1])
        h1 = math.hypot(q[2][0]-q[1][0], q[2][1]-q[1][1])
        h2 = math.hypot(q[3][0]-q[0][0], q[3][1]-q[0][1])
        return (0.5*(w1+w2), 0.5*(h1+h2))
    pose = det.get("pose") or {}
    if template_size and "x_scale" in pose and "y_scale" in pose:
        return (float(template_size[0])*float(pose["x_scale"]),
                float(template_size[1])*float(pose["y_scale"]))
    return None

def _parse_xyzyaw_param(raw) -> Optional[Tuple[float,float,float,float]]:
    if raw is None: return None
    if isinstance(raw, (list, tuple)) and len(raw) == 4:
        return tuple(map(float, raw))
    if isinstance(raw, str):
        parts = [p.strip() for p in raw.replace("[","").replace("]","").split(",")]
        if len(parts) == 4:
            return tuple(map(float, parts))
    return None

class CalibNode:
    def __init__(self):
        rospy.init_node("vgr_calibrate_from_vision", anonymous=False)
        self.topic_in       = rospy.get_param("~vision_result_topic", "/vgr/vision_result")
        self.yaml_in        = rospy.get_param("~vision_geom_in", "config/vision_geom.yaml")
        self.yaml_out       = rospy.get_param("~vision_geom_out", self.yaml_in)
        self.box_mm         = float(rospy.get_param("~box_size_mm", 40.0))   # 4x4 cm
        self.center_sep_mm  = float(rospy.get_param("~center_sep_mm", 80.0)) # 8 cm
        self.obj3_name      = str(rospy.get_param("~obj3_name", "Obj3"))
        self.obj4_name      = str(rospy.get_param("~obj4_name", "Obj4"))
        self.min_inliers    = int(rospy.get_param("~min_inliers", 10))
        self.center_axis    = str(rospy.get_param("~center_axis", "auto")).lower()  # auto|x|y

        raw_origin_mm = rospy.get_param("~origin_mm_xyzyaw", None)
        self.origin_mm_xyzyaw = _parse_xyzyaw_param(raw_origin_mm)

        rospy.Subscriber(self.topic_in, String, self._on_msg, queue_size=5)
        rospy.loginfo("[calib] waiting on %s for %s and %s ...", self.topic_in, self.obj3_name, self.obj4_name)

        if self.origin_mm_xyzyaw:
            x,y,z,yaw = self.origin_mm_xyzyaw
            rospy.loginfo("[calib] will write origin_mm=[%.3f, %.3f, %.3f, %.3f]", x,y,z,yaw)

        self._done = False

    def _find_obj(self, payload: Dict[str,Any], name: str) -> Optional[Dict[str,Any]]:
        for o in (payload.get("result") or {}).get("objects", []) or []:
            if str(o.get("name","")).lower() == name.lower():
                return o
        return None

    def _on_msg(self, msg: String):
        if self._done: return
        try:
            payload = json.loads(msg.data)
        except Exception as e:
            rospy.logwarn("[calib] bad JSON: %s", e); return

        o3 = self._find_obj(payload, self.obj3_name)
        o4 = self._find_obj(payload, self.obj4_name)
        if not o3 or not o4: return
        d3 = _best_det(o3); d4 = _best_det(o4)
        if not d3 or not d4: return
        if (d3.get("inliers",0) < self.min_inliers) or (d4.get("inliers",0) < self.min_inliers): return

        c3 = d3.get("center"); c4 = d4.get("center")
        if not (isinstance(c3,(list,tuple)) and isinstance(c4,(list,tuple)) and len(c3)>=2 and len(c4)>=2):
            rospy.logwarn("[calib] missing centers"); return
        origin_px = [float(c3[0]), float(c3[1])]

        sz3 = _size_px(d3, o3.get("template_size")); sz4 = _size_px(d4, o4.get("template_size"))
        if not sz3 or not sz4: rospy.logwarn("[calib] no size from quads/scales"); return
        w3,h3 = sz3; w4,h4 = sz4
        mean_w = 0.5*(w3+w4); mean_h = 0.5*(h3+h4)

        du = abs(float(c4[0]) - float(c3[0]))
        dv = abs(float(c4[1]) - float(c3[1]))
        axis = self.center_axis if self.center_axis in ("x","y") else ("x" if du >= dv else "y")

        if axis == "x":
            mm_per_px_x = self.center_sep_mm / du if du > 1e-6 else float("inf")
            mm_per_px_y = self.box_mm / mean_h
        else:
            mm_per_px_y = self.center_sep_mm / dv if dv > 1e-6 else float("inf")
            mm_per_px_x = self.box_mm / mean_w

        rospy.loginfo("[calib] centers(px): c3=(%.2f,%.2f) c4=(%.2f,%.2f)  Δu=%.2f Δv=%.2f  axis=%s",
                      c3[0],c3[1],c4[0],c4[1], du, dv, axis)
        rospy.loginfo("[calib] box(px): mean_w=%.2f mean_h=%.2f -> box_x=%.6f box_y=%.6f (mm/px)",
                      mean_w, mean_h, self.box_mm/mean_w, self.box_mm/mean_h)
        rospy.loginfo("[calib] result: mm_per_px_x=%.6f  mm_per_px_y=%.6f  origin_px=(%.1f, %.1f)",
                      mm_per_px_x, mm_per_px_y, origin_px[0], origin_px[1])

        self._write_yaml(self.yaml_in, self.yaml_out, mm_per_px_x, mm_per_px_y, origin_px, self.origin_mm_xyzyaw)
        self._done = True
        if self.origin_mm_xyzyaw:
            x,y,z,yaw = self.origin_mm_xyzyaw
            rospy.loginfo("[calib] origin_mm written: x=%.6f y=%.6f z=%.6f yaw=%.6f", x,y,z,yaw)
        rospy.loginfo("[calib] wrote %s (backup saved).", os.path.abspath(self.yaml_out))
        rospy.signal_shutdown("calibration complete")

    def _write_yaml(self, path_in: str, path_out: str, sx: float, sy: float,
                    origin_px: List[float],
                    origin_mm_xyzyaw: Optional[Tuple[float,float,float,float]]):
        with open(path_in, "r", encoding="utf-8") as f:
            y = yaml.safe_load(f) or {}
        y_out = copy.deepcopy(y) if y else {}
        y_out["mm_per_px_x"] = float(sx)
        y_out["mm_per_px_y"] = float(sy)
        y_out["origin_px"]   = [float(origin_px[0]), float(origin_px[1])]
        if origin_mm_xyzyaw:
            x,y,z,yaw = origin_mm_xyzyaw
            y_out["origin_mm"] = [float(x), float(y), float(z), float(yaw)]
        if os.path.exists(path_out):
            try: os.rename(path_out, path_out + ".bak")
            except Exception: pass
        os.makedirs(os.path.dirname(path_out) or ".", exist_ok=True)
        with open(path_out, "w", encoding="utf-8") as f:
            yaml.safe_dump(y_out, f, sort_keys=False)

if __name__ == "__main__":
    CalibNode()
    rospy.spin()
