#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# source /home/pikapika/Yash_WS/Addverb_Heal_and_Syncro_Hardware/devel/setup.bash 
import rospy
from addverb_cobot_msgs.msg import GraspActionGoal, ReleaseActionGoal

def publish_grasp_action_goal(grasp_force_value=100):
    """Publish a GraspActionGoal with the given grasp force."""
    pub = rospy.Publisher('/robotA/grasp_action/goal', GraspActionGoal, queue_size=10)
    rospy.sleep(0.5)  # allow publisher to connect
    msg = GraspActionGoal()
    msg.goal.grasp_force = grasp_force_value
    pub.publish(msg)
    rospy.loginfo("Published GraspActionGoal with grasp_force: %s", grasp_force_value)

def publish_release_action_goal():
    """Publish an empty ReleaseActionGoal."""
    pub = rospy.Publisher('/robotA/release_action/goal', ReleaseActionGoal, queue_size=10)
    rospy.sleep(0.5)  # allow publisher to connect
    msg = ReleaseActionGoal()
    pub.publish(msg)
    rospy.loginfo("Published ReleaseActionGoal with an empty goal.")

def main():
    rospy.init_node("grasp_release_controller")

    # Example: set this flag to True or False to control
    # grasp = True   # change to False for release
    grasp = False

    if grasp:
        publish_grasp_action_goal(60)  # set desired force here
    else:
        publish_release_action_goal()

    rospy.loginfo("Done sending action goal.")

if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
