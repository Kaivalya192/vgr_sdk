#!/usr/bin/env python3
# ===================================
# FILE: scripts/vision_bridge_node.py
# ===================================
# rosrun vgr_sdk vision_bridge_node.py _bind_ip:=0.0.0.0 _port:=40001 _topic:=/vgr/vision_result
import rospy
from vgr_sdk.nodes.vision_bridge import VisionBridgeNode

def main():
    rospy.init_node("vgr_vision_bridge", anonymous=False)
    node = VisionBridgeNode()
    node.start()
    rospy.on_shutdown(node.shutdown)
    rospy.spin()

if __name__ == "__main__":
    main()
