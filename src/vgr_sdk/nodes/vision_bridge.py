# =======================================
# FILE: src/vgr_sdk/nodes/vision_bridge.py
# =======================================
from __future__ import annotations
import threading
import json
import rospy
from std_msgs.msg import String

from vgr_sdk.core.vision_ingest import VisionUDPIngest


class VisionBridgeNode:
    """
    Listens for UDP JSON from the Vision app and republishes to ROS as std_msgs/String.

    ROS Params (private, ~):
      ~bind_ip (str)         : "0.0.0.0"
      ~port (int)            : 40001
      ~schema_path (str)     : path to JSON schema (optional)
      ~topic (str)           : "/vgr/vision_result"
      ~poll_timeout (float)  : 0.5 seconds
    """

    def __init__(self):
        self.bind_ip = rospy.get_param("~bind_ip", "0.0.0.0")
        self.port = int(rospy.get_param("~port", 40001))
        self.schema_path = rospy.get_param("~schema_path", "")
        self.topic = rospy.get_param("~topic", "/vgr/vision_result")
        self.poll_timeout = float(rospy.get_param("~poll_timeout", 0.5))

        self.pub = rospy.Publisher(self.topic, String, queue_size=10)
        self.ingest = VisionUDPIngest(bind_ip=self.bind_ip, port=self.port,
                                      schema_path=(self.schema_path or None))
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def start(self):
        rospy.loginfo(f"[vision_bridge] UDP {self.bind_ip}:{self.port} -> ROS {self.topic}")
        self._thread.start()

    def _loop(self):
        while not rospy.is_shutdown():
            payload, addr = self.ingest.recv_once(timeout=self.poll_timeout)
            if payload is None:
                continue
            try:
                msg = String(data=json.dumps(payload, separators=(",", ":")))
                self.pub.publish(msg)
            except Exception as e:
                rospy.logwarn(f"[vision_bridge] publish error: {e}")

    def shutdown(self):
        try:
            self.ingest.close()
        except Exception:
            pass
