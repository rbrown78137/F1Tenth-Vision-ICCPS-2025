import constants
import cv2 as cv
import torch
import math

def prepare_image(cv_rgb_image, min_x, max_x, min_y, max_y):
    resized_image = cv.resize(cv_rgb_image, (constants.pose_network_width, constants.pose_network_width))
    cropped_image = cv_rgb_image[math.floor(max(0, min_y) * constants.camera_pixel_height):math.ceil(min(1, max_y) * constants.camera_pixel_height), math.floor(max(0, min_x) * constants.camera_pixel_width):math.ceil(min(1, max_x) * constants.camera_pixel_width)]
    resized_crop = cv.resize(cropped_image, (constants.pose_network_width, constants.pose_network_width))
    camera_tensor = torch.from_numpy(resized_image)
    camera_tensor = camera_tensor.to(constants.device).permute(2, 0, 1).to(torch.float).mul(1 / 256)
    crop_tensor = torch.from_numpy(resized_crop)
    crop_tensor = crop_tensor.to(constants.device).permute(2, 0, 1).to(torch.float).mul(1 / 256)
    y_ratio = 1
    object_mask_layer = torch.zeros((1, constants.pose_network_width, constants.pose_network_height), dtype=torch.float).to(constants.device)
    object_mask_layer[:, int(max(0, y_ratio * min_y) * constants.pose_network_height):int(min(1, y_ratio * max_y) * constants.pose_network_height), int(max(0, min_x) * constants.pose_network_width):int(min(1, max_x) * constants.pose_network_width)] = 1
    modified_image = torch.cat((camera_tensor, object_mask_layer, crop_tensor), dim=0)
    return modified_image
