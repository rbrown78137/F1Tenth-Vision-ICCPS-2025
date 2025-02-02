import cv2 as cv
import numpy as np
from math import sin,cos,sqrt, pow
import constants
from scipy.spatial.transform import Rotation as R

arrow_window_width = 400
arrow_window_height = 400
direction_arrow_size = 40
scale_factor = 1/5
arrow_length = constants.car_length/2
other_arrow_length = constants.car_width /2
def display_arrows(pose_array):
    pointer_img = np.ones((arrow_window_width,arrow_window_height,3), dtype=np.uint8)*255
    #[local_x,local_y,local_z,total_yaw,pi_x,pi_y,pi_z,pi_yaw,omega_x,omega_y,omega_z,omega_yaw]
    pi_point_1, pi_point_2, pi_point_3, pi_point_4, pi_point_5, pi_point_6  = get_points(pose_array[4],pose_array[6],pose_array[7])
    # print("Pi Yaw: "+ str(pose_array[7]))
    omega_point_1, omega_point_2, omega_point_3, omega_point_4, omega_point_5, omega_point_6 = get_points(pose_array[8],pose_array[10],pose_array[11])
    
    color_thin = (255, 0, 0)
    color_long = (0, 0, 255)
    thickness = 5
    pointer_img = cv.line(pointer_img, pi_point_1, pi_point_3, color_long, thickness)
    pointer_img = cv.line(pointer_img, omega_point_1, omega_point_3, color_long, thickness)
    
    pointer_img = cv.line(pointer_img, pi_point_1, pi_point_4, color_long, thickness)
    pointer_img = cv.line(pointer_img, omega_point_1, omega_point_4, color_long, thickness)
    
    pointer_img = cv.line(pointer_img, pi_point_1, pi_point_2, color_thin, thickness)
    pointer_img = cv.line(pointer_img, omega_point_1, omega_point_2, color_thin, thickness)
    
    pointer_img = cv.line(pointer_img, pi_point_3, pi_point_5, color_thin, thickness)
    pointer_img = cv.line(pointer_img, omega_point_3, omega_point_5, color_thin, thickness)
    
    pointer_img = cv.line(pointer_img, pi_point_3, pi_point_6, color_thin, thickness)
    pointer_img = cv.line(pointer_img, omega_point_3, omega_point_6, color_thin, thickness)
    # Direction Arrow
    r_true = R.from_euler('z', pose_array[7] + constants.pi_real_angle_vicon_diff)
    r_camera = R.from_euler('z', pose_array[7] + constants.pi_real_angle_vicon_diff + constants.runtime_camera_angle_tilt)
    unit_normal = [pose_array[0],pose_array[2],0]
    # unit_normal = [0,1,0]
    #Draw Vicon perceived arrow of other car
    rotation_unit_vector = r_true.apply(unit_normal)
    direction_point = (int(pi_point_1[0] + direction_arrow_size*rotation_unit_vector[0]),int(pi_point_1[1] - direction_arrow_size* rotation_unit_vector[1]))
    pointer_img = cv.line(pointer_img, pi_point_1, direction_point, (0, 255,0), thickness)
    
    # Draw Direction Camera sees
    rotation_camera_tilt = r_camera.apply(unit_normal)
    direction_point_camera_tilt = (int(pi_point_1[0] + direction_arrow_size*rotation_camera_tilt[0]),int(pi_point_1[1] - direction_arrow_size* rotation_camera_tilt[1]))
    pointer_img = cv.line(pointer_img, pi_point_1, direction_point_camera_tilt, (255, 255,0), thickness)
    
    cv.imshow("Directions",pointer_img)
    cv.waitKey(1)

def get_points(x,z, yaw):
    first_point = (int((x)*scale_factor*arrow_window_width/2 + arrow_window_width/2),int(-1*(z)*scale_factor*arrow_window_height/2 + arrow_window_height/2))
    second_point = (int((x + arrow_length/2* -sin(yaw))*scale_factor*arrow_window_width/2 + arrow_window_width/2),int(-1*(z+ arrow_length/2* cos(yaw))*scale_factor*arrow_window_height/2 + arrow_window_height/2))
    third_point = (int((x + arrow_length* -sin(yaw))*scale_factor*arrow_window_width/2 + arrow_window_width/2),int(-1*(z+ arrow_length* cos(yaw))*scale_factor*arrow_window_height/2 + arrow_window_height/2))
    fourth_point = (int((x - arrow_length* -sin(yaw))*scale_factor*arrow_window_width/2 + arrow_window_width/2),int(-1*(z- arrow_length* cos(yaw))*scale_factor*arrow_window_height/2 + arrow_window_height/2))
    extra_corner_1 = (int((x + arrow_length* -sin(yaw) + other_arrow_length * cos(yaw))*scale_factor*arrow_window_width/2 + arrow_window_width/2),int(-1*(z+ arrow_length* cos(yaw)+ other_arrow_length * sin(yaw))*scale_factor*arrow_window_height/2 + arrow_window_height/2))
    extra_corner_2 = (int((x + arrow_length* -sin(yaw) - other_arrow_length * cos(yaw))*scale_factor*arrow_window_width/2 + arrow_window_width/2),int(-1*(z+ arrow_length* cos(yaw)- other_arrow_length * sin(yaw))*scale_factor*arrow_window_height/2 + arrow_window_height/2))
    return first_point, second_point, third_point, fourth_point, extra_corner_1,extra_corner_2
