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
from math import sin,cos
import display_pose
import os
from scipy.spatial.transform import Rotation as R
import copy

bridge = cv_bridge.CvBridge()
pose_map = {}
record_real_data = True
image_count = 849
home_data_directory = "../paper_set_august_2023"

# second_offset = 1886.4 # FIRST RECORDING FIRST FILE
# second_offset = 1498.2 # FIRST RECORDING SECOND FILE
# second_offset = 10767.5 # FIRST RECORDING THIRD FILE
# second_offset = 10885 # FIRST RECORDING FOURTH FILE
# second_offset = 10826.5 # FIRST RECORDING FIFTH FILE FIRST PART
# second_offset = 10624 # FIRST RECORDING 6th FILE

# second_offset = 2286.7 # FINAL RECORDING 1
# second_offset = 1186.7 # FINAL RECORDING 3
# second_offset = 891.7 # FINAL RECORDING 4
# second_offset = 765.35 # FINAL RECORDING 5
# second_offset = 605.85 # FINAL RECORDING 6
# second_offset = 0
# second_offset = 10623.9 # FINAL RECORDING 2-new
# second_offset = 1512.1 # BothCars Passing
second_offset = 704.25 # crossing 1
# second_offset = 1903 # Overtaking 

def image_callback(data):
    global pose_map,image_count
    start_time = time.time()
    nanoseconds = data.header.stamp.nsecs
    seconds = data.header.stamp.secs + second_offset
    stamp_time = seconds + nanoseconds * 1e-9
    cv_image = bridge.imgmsg_to_cv2(data,desired_encoding='rgb8')
    pose_keys = list(pose_map.keys())
    diff_pair = [(abs(pose_key-stamp_time),pose_map[pose_key]) for pose_key in pose_keys]
    diff_pair.sort(key=lambda x:x[0])
    pose_original = diff_pair[0][1]
    pose = copy.deepcopy(pose_original)
    display_pose.display_arrows(pose)
    position_vectors = [pose[0],pose[2],0]
    r = R.from_euler('z',constants.runtime_camera_angle_tilt)
    # print(position_vectors)
    fixed_local_pose = r.apply(position_vectors)
    pose[0] = fixed_local_pose[0]
    pose[1] = fixed_local_pose[2]
    pose[2] = fixed_local_pose[1]
    #Display Distance
    font = cv.FONT_HERSHEY_SIMPLEX
    org = (40, 40)
    fontScale = 1
    color = (255, 0, 0)
    thickness = 2
    cv_image = cv.putText(cv_image, str(pose[2]), org, font, fontScale, color, thickness, cv.LINE_AA) 
    
    # Display Points
    all_corners = get_image_projected_corners(pose[0],pose[2],pose[3])
    for corner in all_corners:
        cv_image = cv.circle(cv_image, (int(corner[0]),int(corner[1])), radius=2, color=(0, 0, 255), thickness=-1)           
    #Display Bounding Box
    min_x = constants.camera_pixel_width
    max_x = 0
    min_y = constants.camera_pixel_height
    max_y = 0
    for corner in all_corners:
        if corner[0] < min_x:
            min_x = corner[0]
        if corner[1] < min_y:
            min_y = corner[1]
        if corner[0] > max_x:
            max_x = corner[0]
        if corner[1] > max_y:
            max_y = corner[1]
    start_point = (int(min_x), int(min_y))
    end_point = (int(max_x), int(max_y))
    color = (255, 0, 0)
    thickness = 2
    cv_image = cv.rectangle(cv_image, start_point, end_point, color, thickness)
    print(pose)
    if record_real_data:
        new_dir_path = home_data_directory + '/data' + str(image_count)
        os.makedirs(new_dir_path)
        cv.imwrite(new_dir_path+"/img.png",bridge.imgmsg_to_cv2(data,desired_encoding='bgr8'))
        with open((new_dir_path + "/pose.pkl"), 'wb') as f:
            if len(all_corners) >0:
                pickle.dump([pose,1,min_x,max_x,min_y,max_y], f)
            else:
                pickle.dump([pose,1,min_x,min_x,min_y,min_y], f)
        image_count+=1
    cv.imshow("Test",cv_image)
    cv.waitKey(1)


# May need to flop x coordinates
# Assumes positive x is left side of image
def generate_corners(center_x,center_z,yaw):
    l = constants.car_length
    w = constants.car_width
    h = constants.car_height
    corners = []
    corners.append( (center_x - sin(yaw) * (l/2) + cos(yaw)*(w/2),(h/2),center_z + cos(yaw) * (l/2) + sin(yaw)*(w/2)))
    corners.append( (center_x - sin(yaw) * (-l/2) + cos(yaw)*(w/2),(h/2),center_z + cos(yaw) * (-l/2) + sin(yaw)*(w/2)))
    corners.append( (center_x - sin(yaw) * (l/2) + cos(yaw)*(-w/2),(h/2),center_z + cos(yaw) * (l/2) + sin(yaw)*(-w/2)))
    corners.append( (center_x - sin(yaw) * (-l/2) + cos(yaw)*(-w/2),(h/2),center_z + cos(yaw) * (-l/2) + sin(yaw)*(-w/2)))
    corners.append( (center_x - sin(yaw) * (l/2) + cos(yaw)*(w/2),(-h/2),center_z + cos(yaw) * (l/2) + sin(yaw)*(w/2)))
    corners.append( (center_x - sin(yaw) * (-l/2) + cos(yaw)*(w/2), (-h/2),center_z + cos(yaw) * (-l/2) + sin(yaw)*(w/2)))
    corners.append( (center_x - sin(yaw) * (l/2) + cos(yaw)*(-w/2), (-h/2),center_z + cos(yaw) * (l/2) + sin(yaw)*(-w/2)))
    corners.append( (center_x - sin(yaw) * (-l/2) + cos(yaw)*(-w/2), (-h/2),center_z + cos(yaw) * (-l/2) + sin(yaw)*(-w/2)))
    corners.append( (center_x,0,center_z))
    return corners


def translate_point_to_coordinate_frame(local_x,local_y,local_z):
    view_frame_x = local_x - constants.camera_x_offset
    view_frame_y = local_y - constants.camera_y_offset
    view_frame_z = local_z - constants.camera_z_offset
    if view_frame_z <= 0:
        return None
    pixel_x = (view_frame_x / view_frame_z) * constants.horizontal_focal_length 
    pixel_y = (view_frame_y / view_frame_z) * constants.horizontal_focal_length
    pixel_x = pixel_x + constants.horizontal_center
    pixel_y = -1 * pixel_y + constants.vertical_center
    #clip to range
    pixel_x = max(0,min(pixel_x,constants.camera_pixel_width))
    pixel_y = max(0,min(pixel_y,constants.camera_pixel_height))
    return pixel_x, pixel_y


def get_image_projected_corners(center_x,center_z,yaw):
    absolute_corners = generate_corners(center_x,center_z,yaw)
    final_list = []
    for corner in absolute_corners:
        tranformed_coordinate = translate_point_to_coordinate_frame(corner[0],corner[1],corner[2])
        if not (tranformed_coordinate is None):
            final_list.append(tranformed_coordinate)
    return final_list
def setup():
    rospy.init_node('object_tracker')
    rospy.Subscriber(constants.camera_topic, Image, image_callback)
    rospy.spin()

if __name__ == '__main__':
       # Load File
    with open('saved_data/pose.pkl', 'rb') as f:
        pose_map = pickle.load(f)
    setup()
