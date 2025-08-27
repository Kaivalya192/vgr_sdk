# ==================================
# FILE: src/vgr_sdk/core/transform.py
# ==================================
from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, List, Tuple, Dict, Any, Optional
import numpy as np


@dataclass
class GeomConfig:
    """
    Calibration-less mapping for a fixed Z plane.
    All angles are degrees, positions in millimeters unless stated.

    - proc_width/height: processed frame dimensions from Vision SDK.
    - mm_per_px_x/y: global scale (can be different if pixels are non-square).
    - origin_px: pixel (x,y) in processed space that maps to origin_mm in world.
    - origin_mm: world coordinates (X,Y) for origin_px.
    - invert_x / invert_y: flip axes to match robot world handedness if needed.
    - yaw_offset_deg: constant rotation to add after converting image theta to world yaw.
    - plane_z_mm: fixed Z for the pick plane.
    """
    proc_width: int
    proc_height: int
    mm_per_px_x: float
    mm_per_px_y: float
    origin_px: Tuple[float, float] = (0.0, 0.0)
    origin_mm: Tuple[float, float] = (0.0, 0.0)
    invert_x: bool = False
    invert_y: bool = False
    yaw_offset_deg: float = 0.0
    theta_sign: int = 1   # +1 if image theta maps directly to world yaw; -1 if inverted
    plane_z_mm: float = 0.0


def px_to_world_xy(xp: float, yp: float, cfg: GeomConfig) -> Tuple[float, float]:
    """Map processed pixel (xp, yp) → world (Xmm, Ymm)."""
    dx_px = xp - cfg.origin_px[0]
    dy_px = yp - cfg.origin_px[1]
    X = cfg.origin_mm[0] + ( -dx_px if cfg.invert_x else dx_px ) * cfg.mm_per_px_x
    Y = cfg.origin_mm[1] + ( -dy_px if cfg.invert_y else dy_px ) * cfg.mm_per_px_y
    return float(X), float(Y)


def extent_px_to_mm(w_px: float, h_px: float, cfg: GeomConfig) -> Tuple[float, float]:
    """Convert local width/height in pixels to millimeters."""
    return float(w_px * cfg.mm_per_px_x), float(h_px * cfg.mm_per_px_y)


def angle_img_to_world(theta_img_deg: float, cfg: GeomConfig) -> float:
    """Map image theta (deg) to world yaw (deg) with sign and offset."""
    return float(cfg.theta_sign * theta_img_deg + cfg.yaw_offset_deg)


def quad_px_to_world(quad_px: Iterable[Iterable[float]], cfg: GeomConfig) -> List[List[float]]:
    """Transform 4x2 quad points from pixels to world millimeters."""
    out: List[List[float]] = []
    for p in quad_px:
        X, Y = px_to_world_xy(float(p[0]), float(p[1]), cfg)
        out.append([X, Y, cfg.plane_z_mm])
    return out


def detection_img_to_world(det: Dict[str, Any], cfg: GeomConfig) -> Dict[str, Any]:
    """
    Convert a Vision SDK detection (image space) to world-space fields.
    Expects keys: center (optional), quad (optional), pose.x/pose.y/theta_deg present in parent caller.

    Returns a shallow copy with:
      - center_world_mm: [X, Y, Z]
      - quad_world_mm: [[...]*4]
      - pose_world: {x_mm, y_mm, z_mm, yaw_deg}
    """
    det_w = dict(det)

    # center
    cw = None
    c = det.get("center")
    if c and len(c) >= 2:
        X, Y = px_to_world_xy(float(c[0]), float(c[1]), cfg)
        cw = [X, Y, cfg.plane_z_mm]
        det_w["center_world_mm"] = cw

    # quad
    q = det.get("quad")
    if q and len(q) == 4:
        det_w["quad_world_mm"] = quad_px_to_world(q, cfg)

    # pose
    pose = det.get("pose", {}) if isinstance(det.get("pose"), dict) else {}
    xi = float(pose.get("x", 0.0))
    yi = float(pose.get("y", 0.0))
    yaw_img = float(pose.get("theta_deg", 0.0))
    Xp, Yp = px_to_world_xy(xi, yi, cfg)
    yaw_w = angle_img_to_world(yaw_img, cfg)
    det_w["pose_world"] = {"x_mm": Xp, "y_mm": Yp, "z_mm": cfg.plane_z_mm, "yaw_deg": yaw_w}

    return det_w
