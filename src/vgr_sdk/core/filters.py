# ===============================
# FILE: src/vgr_sdk/core/filters.py
# ===============================
from __future__ import annotations
from typing import List, Dict, Any, Tuple
import math


def gate_detections(
    detections: List[Dict[str, Any]],
    *,
    min_inliers: int = 10,
    min_score: float = 0.25,
) -> List[Dict[str, Any]]:
    """Return only detections that pass basic quality gates."""
    out: List[Dict[str, Any]] = []
    for d in detections:
        if int(d.get("inliers", 0)) < min_inliers:
            continue
        if float(d.get("score", 0.0)) < min_score:
            continue
        out.append(d)
    return out


def sort_by_quality(
    detections: List[Dict[str, Any]],
    *,
    inliers_weight: float = 1.0,
    score_weight: float = 0.5,
) -> List[Dict[str, Any]]:
    """Sort detections (desc) by a simple weighted metric."""
    def key(d: Dict[str, Any]) -> float:
        return inliers_weight * float(d.get("inliers", 0)) + score_weight * float(d.get("score", 0.0))
    return sorted(detections, key=key, reverse=True)


def dedupe_by_center_px(
    detections: List[Dict[str, Any]],
    *,
    min_center_dist_px: float = 40.0,
) -> List[Dict[str, Any]]:
    """Keep only detections whose centers are mutually farther than threshold (pixels)."""
    kept: List[Dict[str, Any]] = []
    for d in detections:
        c = d.get("center")
        if not c:
            kept.append(d)
            continue
        cx, cy = float(c[0]), float(c[1])
        too_close = False
        for k in kept:
            kc = k.get("center")
            if not kc:
                continue
            dx = cx - float(kc[0]); dy = cy - float(kc[1])
            if math.hypot(dx, dy) < min_center_dist_px:
                too_close = True
                break
        if not too_close:
            kept.append(d)
    return kept


def dedupe_by_center_mm(
    detections: List[Dict[str, Any]],
    *,
    min_center_dist_mm: float = 20.0,
) -> List[Dict[str, Any]]:
    """Same as above, but uses `center_world_mm` = [X,Y,Z]."""
    kept: List[Dict[str, Any]] = []
    for d in detections:
        c = d.get("center_world_mm")
        if not c:
            kept.append(d)
            continue
        cx, cy = float(c[0]), float(c[1])
        too_close = False
        for k in kept:
            kc = k.get("center_world_mm")
            if not kc:
                continue
            dx = cx - float(kc[0]); dy = cy - float(kc[1])
            if math.hypot(dx, dy) < min_center_dist_mm:
                too_close = True
                break
        if not too_close:
            kept.append(d)
    return kept
