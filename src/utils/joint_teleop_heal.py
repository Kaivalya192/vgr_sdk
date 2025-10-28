#!/usr/bin/env python3
"""
Keyboard Teleoperation for 6-DOF Robot Arm
------------------------------------------

Author: Debojit Das (2025) • debojit.das@iitgn.ac.in

Overview:
---------
This script allows the user to control each joint of a 6-DOF robot arm using keyboard keys.
Joint velocities are published to the `/velocity_controller/command` topic.

Key Mappings:
-------------
Control each joint using the following keys:

    Joint 1:    q (positive)     /     a (negative)
    Joint 2:    w (positive)     /     s (negative)
    Joint 3:    e (positive)     /     d (negative)
    Joint 4:    r (positive)     /     f (negative)
    Joint 5:    t (positive)     /     g (negative)
    Joint 6:    y (positive)     /     h (negative)

Special:
    ESC        : Exit the teleoperation session

Notes:
------
- Multiple keys can be held down to control multiple joints simultaneously.
- Commands are sent at 50 Hz for responsive control.
"""

import rospy
from pynput import keyboard
from std_msgs.msg import Float64MultiArray

# ----------------------------- Configuration -----------------------------
v = 0.1  # Velocity magnitude (rad/s)
NUM_JOINTS = 6

# Key mappings for each joint (positive/negative direction)
key_mapping = {
    'q': [v, 0, 0, 0, 0, 0],    # Joint 1 +
    'a': [-v, 0, 0, 0, 0, 0],   # Joint 1 -
    'w': [0, v, 0, 0, 0, 0],    # Joint 2 +
    's': [0, -v, 0, 0, 0, 0],   # Joint 2 -
    'e': [0, 0, v, 0, 0, 0],    # Joint 3 +
    'd': [0, 0, -v, 0, 0, 0],   # Joint 3 -
    'r': [0, 0, 0, v, 0, 0],    # Joint 4 +
    'f': [0, 0, 0, -v, 0, 0],   # Joint 4 -
    't': [0, 0, 0, 0, v, 0],    # Joint 5 +
    'g': [0, 0, 0, 0, -v, 0],   # Joint 5 -
    'y': [0, 0, 0, 0, 0, v],    # Joint 6 +
    'h': [0, 0, 0, 0, 0, -v],   # Joint 6 -
}

active_command = [0] * NUM_JOINTS
pressed_keys = set()

# --------------------------- Command Handling ---------------------------

def update_active_command():
    """Recalculates the active velocity command based on current pressed keys."""
    global active_command
    active_command = [0] * NUM_JOINTS
    for key in pressed_keys:
        key_char = None
        if isinstance(key, str):
            key_char = key
        elif hasattr(key, 'char') and key.char is not None:
            key_char = key.char
        if key_char in key_mapping:
            active_command = [sum(x) for x in zip(active_command, key_mapping[key_char])]

def on_press(key):
    """Triggered when a key is pressed."""
    global pressed_keys
    if key not in pressed_keys:
        pressed_keys.add(key)
        update_active_command()

def on_release(key):
    """Triggered when a key is released."""
    global pressed_keys
    if key in pressed_keys:
        pressed_keys.remove(key)
    update_active_command()

    if key == keyboard.Key.esc:
        rospy.loginfo("🛑 ESC pressed. Exiting teleoperation.")
        return False  # Stop the keyboard listener

# --------------------------- Main ROS Loop ---------------------------

def keyboard_control():
    rospy.init_node('keyboard_velocity_control', anonymous=True)
    pub = rospy.Publisher('/velocity_controller/command', Float64MultiArray, queue_size=10)
    rate = rospy.Rate(50)

    rospy.loginfo("🧠 Keyboard teleoperation started.")
    rospy.loginfo("🔧 Use Q/A, W/S, E/D, R/F, T/G, Y/H to move joints.")
    rospy.loginfo("🚪 Press ESC to exit.\n")

    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()

    try:
        while not rospy.is_shutdown():
            msg = Float64MultiArray(data=active_command)
            pub.publish(msg)
            print(f"🔄 Command: {active_command}", end='\r')
            rate.sleep()
    except rospy.ROSInterruptException:
        pass
    finally:
        listener.stop()
        rospy.loginfo("\n✅ Teleoperation node shut down.")

# --------------------------- Entry Point ---------------------------

if __name__ == "__main__":
    keyboard_control()
