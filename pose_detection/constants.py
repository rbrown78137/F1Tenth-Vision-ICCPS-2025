import torch
import math

camera_pixel_width = 640
camera_pixel_height = 480
yolo_network_width = 416
yolo_network_height = 416
pose_network_width = 256
pose_network_height = 256

horizontal_focal_length = 607.300537109375
vertical_focal_length = 607.25244140625
vertical_center = 239.65838623046875
horizontal_center = 317.9561462402344

sampling_size = 20
device = "cuda" if torch.cuda.is_available() else "cpu"


def ray_angle(x_center):
    ray = math.atan((0.5-x_center)*camera_pixel_width/horizontal_focal_length)
    return ray


def relative_angle(x_center,ground_truth_yaw):
    return wrapAngle(ground_truth_yaw - ray_angle(x_center))


def wrapAngle(angle):
    while angle < 0:
        angle += 2 * math.pi
    while angle >= 2 * math.pi:
        angle -= 2 * math.pi
    return angle
