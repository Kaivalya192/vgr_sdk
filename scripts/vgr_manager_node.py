#!/usr/bin/env python3
# =================================
# FILE: scripts/vgr_manager_node.py
# =================================

# rosrun vgr_sdk vgr_manager_node.py \
#   _vision_geom:=config/vision_geom.yaml \
#   _task_config:=config/tasks/sorting.yaml \
#   _task:=sorting \
#   _sub_result_topic:=/vgr/vision_result \
#   _pub_plan_topic:=/vgr/plan

import rospy
from vgr_sdk.nodes.vgr_manager import VGRManagerNode

def main():
    rospy.init_node("vgr_manager", anonymous=False)
    node = VGRManagerNode()
    rospy.spin()

if __name__ == "__main__":
    main()
