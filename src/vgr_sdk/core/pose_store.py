# ==================================
# FILE: src/vgr_sdk/core/pose_store.py
# ==================================
from __future__ import annotations
import os
import time
import tempfile
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any
import yaml


@dataclass
class PoseEntry:
    """A named joint-space pose."""
    name: str
    joints: List[float]
    stamp: float = 0.0          # epoch seconds
    note: str = ""              # optional human note
    metadata: Dict[str, Any] = None  # optional arbitrary metadata

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # keep YAML compact
        if not d.get("note"):
            d.pop("note", None)
        if not d.get("metadata"):
            d.pop("metadata", None)
        return d


class PoseStore:
    """
    Simple YAML-backed store for recorded joint poses.

    YAML layout:
    ---
    version: 1
    poses:
      home:
        name: home
        joints: [0, 0, 0, 0, 0, 0]
        stamp: 1724720000.0
        note: "example"
    """

    def __init__(self, path: str):
        self.path = os.path.abspath(path)
        self.version = 1
        self._poses: Dict[str, PoseEntry] = {}
        # Attempt load if file exists
        if os.path.exists(self.path):
            self.load()

    # ---------- I/O ----------
    def load(self) -> None:
        with open(self.path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        self.version = int(data.get("version", 1))
        self._poses.clear()
        poses = data.get("poses", {}) or {}
        for name, pd in poses.items():
            try:
                entry = PoseEntry(
                    name=str(pd.get("name", name)),
                    joints=[float(v) for v in pd.get("joints", [])],
                    stamp=float(pd.get("stamp", 0.0)),
                    note=str(pd.get("note", "")) if pd.get("note") is not None else "",
                    metadata=dict(pd.get("metadata")) if pd.get("metadata") is not None else None,
                )
                if entry.name:
                    self._poses[entry.name] = entry
            except Exception:
                # Skip malformed entries
                continue

    def save(self) -> None:
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        data = {
            "version": self.version,
            "poses": {name: pe.to_dict() for name, pe in sorted(self._poses.items())},
        }
        tmp_dir = os.path.dirname(self.path) or "."
        with tempfile.NamedTemporaryFile("w", delete=False, dir=tmp_dir, encoding="utf-8") as tf:
            yaml.safe_dump(data, tf, sort_keys=False)
            tmp_name = tf.name
        os.replace(tmp_name, self.path)

    # ---------- CRUD ----------
    def names(self) -> List[str]:
        return sorted(self._poses.keys())

    def exists(self, name: str) -> bool:
        return name in self._poses

    def get(self, name: str) -> Optional[PoseEntry]:
        return self._poses.get(name)

    def set(self, name: str, joints: List[float], *, note: str = "", metadata: Optional[Dict[str, Any]] = None) -> PoseEntry:
        pe = PoseEntry(name=str(name), joints=list(map(float, joints)), stamp=time.time(), note=note, metadata=metadata)
        self._poses[pe.name] = pe
        return pe

    def delete(self, name: str) -> bool:
        if name in self._poses:
            del self._poses[name]
            return True
        return False

    def rename(self, old: str, new: str) -> bool:
        if old not in self._poses or not new or new in self._poses:
            return False
        pe = self._poses.pop(old)
        pe.name = new
        self._poses[new] = pe
        return True

    def to_dict(self) -> Dict[str, Any]:
        return {name: pe.to_dict() for name, pe in self._poses.items()}
