# =====================================
# FILE: src/vgr_sdk/core/vision_ingest.py
# =====================================
from __future__ import annotations
import json
import socket
import threading
from typing import Callable, Optional, Tuple, Dict, Any
import os

try:
    import simplejson as sjson  # more tolerant parser if present
except Exception:
    sjson = None

try:
    from jsonschema import Draft202012Validator  # type: ignore
except Exception:
    Draft202012Validator = None  # validation optional


class VisionJSONValidator:
    """Optional JSON Schema validator for Vision SDK payloads."""
    def __init__(self, schema_path: Optional[str] = None):
        self._validator = None
        if schema_path and Draft202012Validator:
            try:
                with open(schema_path, "r", encoding="utf-8") as f:
                    schema = json.load(f)
                self._validator = Draft202012Validator(schema)
            except Exception:
                self._validator = None

    def validate(self, payload: Dict[str, Any]) -> bool:
        if self._validator is None:
            return True
        try:
            self._validator.validate(payload)
            return True
        except Exception:
            return False


def parse_json(data: bytes) -> Optional[Dict[str, Any]]:
    """Lenient JSON parsing; returns dict or None."""
    txt = None
    try:
        txt = data.decode("utf-8")
    except Exception:
        try:
            txt = data.decode("latin-1")
        except Exception:
            return None
    # Try tolerant parser first
    if sjson:
        try:
            return sjson.loads(txt)
        except Exception:
            pass
    try:
        return json.loads(txt)
    except Exception:
        return None


class VisionUDPIngest:
    """
    Simple UDP listener for Vision SDK JSON messages.
    Standalone: no ROS deps, no project imports.

    Usage:
        ing = VisionUDPIngest(bind_ip="0.0.0.0", port=40001, schema_path="schemas/vision_result.schema.json")
        payload, addr = ing.recv_once(timeout=1.0)
        # OR
        ing.serve_forever(callback=print)  # callback(payload_dict, (ip, port))
    """
    def __init__(
        self,
        *,
        bind_ip: str = "0.0.0.0",
        port: int = 40001,
        bufsize: int = 65535,
        schema_path: Optional[str] = None,
    ):
        self.bind_ip = bind_ip
        self.port = int(port)
        self.bufsize = int(bufsize)
        self._sock: Optional[socket.socket] = None
        self._validator = VisionJSONValidator(schema_path)
        self._lock = threading.Lock()

    # ---------- socket lifecycle ----------
    def _ensure_sock(self) -> None:
        with self._lock:
            if self._sock is not None:
                return
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                # On some OSes SO_REUSEPORT helps multiple listeners; ignore if unsupported
                s.setsockopt(socket.SOL_SOCKET, 0x0F, 1)  # SO_REUSEPORT
            except Exception:
                pass
            s.bind((self.bind_ip, self.port))
            self._sock = s

    def close(self) -> None:
        with self._lock:
            if self._sock:
                try:
                    self._sock.close()
                except Exception:
                    pass
                self._sock = None

    # ---------- receiving ----------
    def recv_once(self, timeout: Optional[float] = None) -> Tuple[Optional[Dict[str, Any]], Optional[Tuple[str, int]]]:
        """
        Receive one datagram. Returns (payload_dict_or_None, (ip, port)_or_None).
        If timeout expires, returns (None, None).
        """
        self._ensure_sock()
        assert self._sock is not None
        self._sock.settimeout(timeout)
        try:
            data, addr = self._sock.recvfrom(self.bufsize)
        except socket.timeout:
            return None, None
        except Exception:
            return None, None

        payload = parse_json(data)
        if payload is None:
            return None, addr

        if not self._validator.validate(payload):
            return None, addr

        return payload, addr

    def serve_forever(self, callback: Callable[[Dict[str, Any], Tuple[str, int]], None], *, poll_timeout: float = 1.0) -> None:
        """
        Blocking loop that invokes `callback(payload, addr)` for every valid message.
        """
        self._ensure_sock()
        assert self._sock is not None
        self._sock.settimeout(poll_timeout)
        while True:
            try:
                data, addr = self._sock.recvfrom(self.bufsize)
            except socket.timeout:
                continue
            except Exception:
                break

            payload = parse_json(data)
            if payload is None:
                continue
            if not self._validator.validate(payload):
                continue

            try:
                callback(payload, addr)
            except Exception:
                # never let user callback crash the loop
                continue
