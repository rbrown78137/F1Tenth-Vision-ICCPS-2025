horizontal_focal_length = 607.0233764648438
vertical_focal_length = 607.695068359375    
vertical_center = 239.65838623046875
horizontal_center = 317.9561462402344

camera_pixel_height = 480
camera_pixel_width = 640

# RECORDING 1 5
camera_x_offset = 0.0
camera_y_offset = 0.05
camera_z_offset = 0

pi_real_angle_vicon_diff = 0

runtime_camera_angle_tilt = 0.025 # Passing Clip
runtime_camera_angle_tilt = 0.04 # Crossing 2
# runtime_camera_angle_tilt = 0.05 # FINAL Recording 2-new
# runtime_camera_angle_tilt = 0.02 # FINAL Recording 1 to 5
vicon_marker_topic = "/vicon/markers"
car_pi_tracker = "/vicon/RedCar/RedCar"
car_omega_tracker = "/vicon/BlackCar/BlackCar"
camera_topic = "/camera/color/image_raw"

car_width = 0.38
car_length = 0.55
car_height = 0.3
# Y AND Z dimensions flipped in vicon
pi_car_center_x_offset = 0
pi_car_center_y_offset = 0
pi_car_center_z_offset = 0

# Y AND Z dimensions flipped in vicon
omega_car_center_x_offset = 0
omega_car_center_y_offset = 0 # 0
omega_car_center_z_offset = 0 # 0

import math

def wrapAngle(angle):
    while angle < 0:
        angle += 2 * math.pi
    while angle >= 2 * math.pi:
        angle -= 2 * math.pi
    return angle
