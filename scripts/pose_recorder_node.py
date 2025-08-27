#!/usr/bin/env python3
# ====================================
# FILE: scripts/pose_recorder_node.py
# ====================================
import rospy
import yaml
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from vgr_sdk.core.pose_store import PoseStore

class PoseRecorderNode:
    """
    Subscribe to /joint_states and to a trigger topic (~save_topic, default /vgr/pose_recorder/save).
    When a String(name) arrives on the save topic, writes current joints into poses YAML as 'name'.

    Params:
      ~poses_path  : "poses/recorded_poses.yaml"
      ~save_topic  : "/vgr/pose_recorder/save"
    """
    def __init__(self):
        self.poses_path = rospy.get_param("~poses_path", "poses/recorded_poses.yaml")
        self.save_topic = rospy.get_param("~save_topic", "/vgr/pose_recorder/save")

        self.store = PoseStore(self.poses_path)
        self.current_joints = None

        rospy.Subscriber("/joint_states", JointState, self._on_js, queue_size=10)
        rospy.Subscriber(self.save_topic, String, self._on_save, queue_size=10)

        rospy.loginfo(f"[pose_recorder] listening; publish String('pose_name') on {self.save_topic} to save")

    def _on_js(self, msg: JointState):
        self.current_joints = list(msg.position)

    def _on_save(self, msg: String):
        name = msg.data.strip()
        if not name:
            rospy.logwarn("[pose_recorder] empty pose name ignored")
            return
        if self.current_joints is None:
            rospy.logwarn("[pose_recorder] no joint state yet")
            return
        self.store.set(name, self.current_joints)
        try:
            self.store.save()
            rospy.loginfo(f"[pose_recorder] saved pose '{name}' -> {self.poses_path}")
        except Exception as e:
            rospy.logerr(f"[pose_recorder] save failed: {e}")

def main():
    rospy.init_node("vgr_pose_recorder", anonymous=False)
    PoseRecorderNode()
    rospy.spin()

if __name__ == "__main__":
    main()
