# =====================================
# FILE: src/vgr_sdk/core/message_types.py
# =====================================
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Pose:
    x: float
    y: float
    theta_deg: float
    x_scale: float
    y_scale: float


@dataclass
class Detection:
    instance_id: int
    score: float
    inliers: int
    pose: Pose
    center: Optional[List[float]] = None     # [x,y]
    quad: Optional[List[List[float]]] = None # 4x2
    color: Optional[Dict[str, float]] = None # {bhattacharyya, correlation, deltaE}

    # Optional post-transform (world) fields the SDK may add later:
    center_world_mm: Optional[List[float]] = None        # [X,Y,Z]
    quad_world_mm: Optional[List[List[float]]] = None    # 4x3
    pose_world: Optional[Dict[str, float]] = None        # {x_mm,y_mm,z_mm,yaw_deg}


@dataclass
class ObjectDetections:
    object_id: int
    name: str
    template_size: Optional[List[float]] = None  # [w,h]
    detections: List[Detection] = field(default_factory=list)


@dataclass
class VisionResult:
    version: str
    sdk: Dict[str, Any]
    session: Dict[str, Any]
    timestamp_ms: int
    camera: Dict[str, Any]
    result: List[ObjectDetections]
    timing_ms: Dict[str, float]

    @classmethod
    def from_json(cls, payload: Dict[str, Any]) -> "VisionResult":
        """Lenient conversion from dict (no schema required)."""
        version = str(payload.get("version", "1.0"))
        sdk = dict(payload.get("sdk", {}))
        session = dict(payload.get("session", {}))
        timestamp_ms = int(payload.get("timestamp_ms", 0))
        camera = dict(payload.get("camera", {}))

        # Parse objects/detections
        result_block = payload.get("result", {}) or {}
        objects_raw = result_block.get("objects", []) or []
        objects: List[ObjectDetections] = []
        for o in objects_raw:
            obj = ObjectDetections(
                object_id=int(o.get("object_id", -1)),
                name=str(o.get("name", f"obj_{o.get('object_id', 0)}")),
                template_size=list(o.get("template_size")) if o.get("template_size") is not None else None,
                detections=[],
            )
            for d in o.get("detections", []) or []:
                p = d.get("pose", {}) or {}
                pose = Pose(
                    x=float(p.get("x", 0.0)),
                    y=float(p.get("y", 0.0)),
                    theta_deg=float(p.get("theta_deg", 0.0)),
                    x_scale=float(p.get("x_scale", 1.0)),
                    y_scale=float(p.get("y_scale", 1.0)),
                )
                det = Detection(
                    instance_id=int(d.get("instance_id", 0)),
                    score=float(d.get("score", 0.0)),
                    inliers=int(d.get("inliers", 0)),
                    pose=pose,
                    center=list(d.get("center")) if d.get("center") is not None else None,
                    quad=[list(q) for q in d.get("quad")] if d.get("quad") is not None else None,
                    color=dict(d.get("color")) if d.get("color") is not None else None,
                )
                # Pass through optional world fields (if present)
                if d.get("center_world_mm") is not None:
                    det.center_world_mm = list(d.get("center_world_mm"))
                if d.get("quad_world_mm") is not None:
                    det.quad_world_mm = [list(q) for q in d.get("quad_world_mm")]
                if d.get("pose_world") is not None:
                    det.pose_world = dict(d.get("pose_world"))
                obj.detections.append(det)
            objects.append(obj)

        timing_ms = dict(payload.get("timing_ms", {}))
        return cls(
            version=version,
            sdk=sdk,
            session=session,
            timestamp_ms=timestamp_ms,
            camera=camera,
            result=objects,
            timing_ms=timing_ms,
        )

    def flat_detections(self) -> List[Dict[str, Any]]:
        """Convenience: flatten to a list with object metadata attached."""
        flat: List[Dict[str, Any]] = []
        for obj in self.result:
            for d in obj.detections:
                flat.append({
                    "object_id": obj.object_id,
                    "name": obj.name,
                    "template_size": obj.template_size,
                    "instance_id": d.instance_id,
                    "score": d.score,
                    "inliers": d.inliers,
                    "pose": {
                        "x": d.pose.x,
                        "y": d.pose.y,
                        "theta_deg": d.pose.theta_deg,
                        "x_scale": d.pose.x_scale,
                        "y_scale": d.pose.y_scale,
                    },
                    "center": d.center,
                    "quad": d.quad,
                    "color": d.color,
                    # optional world fields:
                    "center_world_mm": d.center_world_mm,
                    "quad_world_mm": d.quad_world_mm,
                    "pose_world": d.pose_world,
                })
        return flat
