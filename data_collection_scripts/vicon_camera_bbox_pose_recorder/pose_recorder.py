#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Image
import cv2 as cv
import torch
from std_msgs.msg import Float64MultiArray
import cv_bridge
import math
import numpy as np
import time
import matplotlib.pyplot as plt
import time
import constants
from geometry_msgs.msg import TransformStamped
from vicon_bridge.msg import Markers
import pickle
import signal
from scipy.spatial.transform import Rotation as R
import display_pose

pi_position_map = {}
omega_position_map = {}
final_map = {}

def sigint_handler(arg1,arg2):
    rospy.signal_shutdown(reason="Program Terminated")
    global pi_position_map, omega_position_map, final_map
    pi_keys = [key for key in pi_position_map]
    omega_keys = [key for key in omega_position_map]
    for pi_key in pi_keys:
        diff_pair = [(abs(omega_keys[omega_key_idx]-pi_key),omega_position_map[omega_keys[omega_key_idx]]) for omega_key_idx in range(len(omega_keys))]
        diff_pair.sort(key=lambda x:x[0])
        closest_omega_timestamp, omega_pose = diff_pair[0]
        # Get Pi Global Coordinates
        pi_pose = pi_position_map[pi_key]
        pose = getPoseArray(pi_pose,omega_pose)
        final_map[pi_key] = pose
    with open('saved_data/pose.pkl', 'wb') as f:
        pickle.dump(final_map, f)
    print("Ending Program and Saving Pose data to file.")
    exit(0)

def pi_car_callback(data):
    global pi_position_map, omega_position_map, final_map
    nanoseconds = data.header.stamp.nsecs
    seconds = data.header.stamp.secs
    rot_w = data.transform.rotation.w
    rot_x = data.transform.rotation.x
    rot_y = data.transform.rotation.y
    rot_z = data.transform.rotation.z
    pos_x = data.transform.translation.x
    pos_y = data.transform.translation.y
    pos_z = data.transform.translation.z
    # Adjust POS X,Y,and Z to be true center
    pi_offset_vectors = [constants.pi_car_center_x_offset,constants.pi_car_center_y_offset,constants.pi_car_center_z_offset]
    pi_rotation_object = R.from_quat([rot_x,rot_y,rot_z,rot_w])
    pi_offset_vectors = pi_rotation_object.apply(pi_offset_vectors)
    pos_x = pos_x + pi_offset_vectors[0]
    pos_y = pos_y + pi_offset_vectors[1]
    pos_z = pos_z + pi_offset_vectors[2]
    # Calculate Angle
    pi_rotations = R.from_quat([rot_x,rot_y,rot_z,rot_w]).as_euler('xyz', degrees=False)
    pi_yaw = pi_rotations[2]
    # Create Pose Array
    stamp_time = seconds + nanoseconds * 1e-9
    pi_position_map[stamp_time] = [pos_x,pos_y,pos_z,pi_yaw]
    # rospy.sleep(0.01)

def omega_car_callback(data):
    global pi_position_map, omega_position_map, final_map
    nanoseconds = data.header.stamp.nsecs
    seconds = data.header.stamp.secs
    rot_w = data.transform.rotation.w
    rot_x = data.transform.rotation.x
    rot_y = data.transform.rotation.y
    rot_z = data.transform.rotation.z
    pos_x = data.transform.translation.x
    pos_y = data.transform.translation.y
    pos_z = data.transform.translation.z
    # Adjust POS X,Y,and Z to be true center
    omega_offset_vectors = [constants.omega_car_center_x_offset,constants.omega_car_center_y_offset,constants.omega_car_center_z_offset]
    omega_rotation_object = R.from_quat([rot_x,rot_y,rot_z,rot_w])
    omega_offset_vectors = omega_rotation_object.apply(omega_offset_vectors)
    pos_x = pos_x + omega_offset_vectors[0]
    pos_y = pos_y + omega_offset_vectors[1]
    pos_z = pos_z + omega_offset_vectors[2]
    # Calculate Angle
    omega_rotations = R.from_quat([rot_x,rot_y,rot_z,rot_w]).as_euler('xyz', degrees=False)
    omega_yaw = omega_rotations[2]
    # Create Pose Array
    stamp_time = seconds + nanoseconds * 1e-9
    omega_position_map[stamp_time] = [pos_x,pos_y,pos_z,omega_yaw]
    # rospy.sleep(0.01)

def vicon_markers_callback(data):
    frame_number = data.frame_number
    nanoseconds = data.header.stamp.nsecs
    seconds = data.header.stamp.secs
    stamp_time = seconds + nanoseconds * 1e-9
    pi_keys = [key for key in pi_position_map]
    omega_keys = [key for key in omega_position_map]
    if(len(pi_keys)>0 and len(omega_keys)>0):
        omega_diff_pair = [(abs(omega_keys[omega_key_idx]-stamp_time),omega_position_map[omega_keys[omega_key_idx]]) for omega_key_idx in range(len(omega_keys))]
        omega_diff_pair.sort(key=lambda x:x[0])
        closest_omega_timestamp, omega_pose = omega_diff_pair[0]
        pi_diff_pair = [(abs(pi_keys[pi_key_idx]-stamp_time),pi_position_map[pi_keys[pi_key_idx]]) for pi_key_idx in range(len(pi_keys))]
        pi_diff_pair.sort(key=lambda x:x[0])
        closest_pi_timestamp, pi_pose = pi_diff_pair[0]
        pose = getPoseArray(pi_pose,omega_pose)
        display_pose.display_arrows(pose)
    rospy.sleep(0.01)

def setup():
    rospy.init_node('object_tracker')
    rospy.Subscriber(constants.car_pi_tracker,TransformStamped,pi_car_callback,queue_size=10)
    rospy.Subscriber(constants.car_omega_tracker,TransformStamped,omega_car_callback,queue_size=10)
    rospy.Subscriber(constants.vicon_marker_topic,Markers,vicon_markers_callback,queue_size=10)
    signal.signal(signal.SIGINT,sigint_handler)
    rospy.spin()

def getPoseArray(pi_pose,omega_pose):
    omega_x = omega_pose[0]
    omega_y = omega_pose[1]
    omega_z = omega_pose[2]
    omega_yaw = omega_pose[3]

    pi_x = pi_pose[0]
    pi_y = pi_pose[1]
    pi_z = pi_pose[2]
    pi_yaw = pi_pose[3]
    # Total Stuff
    # print(['%.3f' % n for n in pi_pose])
    total_yaw = constants.wrapAngle(omega_yaw - pi_yaw)
    position_vectors = [omega_pose[0] -pi_pose[0], omega_pose[1] -pi_pose[1], omega_pose[2] -pi_pose[2] ]
    # print(['%.3f' % n for n in position_vectors])
    r = R.from_euler('z',-pi_yaw+constants.pi_real_angle_vicon_diff)
    # print(position_vectors)
    local_position_vector = r.apply(position_vectors)
    local_x = local_position_vector[0]
    local_y = local_position_vector[1]
    local_z = local_position_vector[2]
    #print(['%.3f' % n for n in local_position_vector])
    # Final Pose
    pose = [local_x,local_z,local_y,total_yaw,pi_x,pi_z,pi_y,pi_yaw,omega_x,omega_z,omega_y,omega_yaw]
    return pose

if __name__ == '__main__':
    setup()
