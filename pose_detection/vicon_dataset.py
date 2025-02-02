import math
import pickle
import torch
from torch.utils.data import Dataset
import os
import cv2 as cv
import constants
import matplotlib.pyplot as plt
import random

import image_processing

# General global variables for reading data input
base_directory_for_images = '../paper_set_august_2023'
picture_width = 640
picture_height = 480
BORDER_CONSTANT = 0.1
class CustomImageDataset(Dataset):
    def __init__(self):
        random.seed(20)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Set File Count and Maps To Files
        self.possible_images = len(next(os.walk(base_directory_for_images))[1])
        self.total_images = 0
        self.image_map = {}
        self.image_cache = {}
        for index in range(self.possible_images):
            with open(base_directory_for_images + "/data" + str(index) + "/pose.pkl", 'rb') as f:
                item = pickle.load(f)
                min_x = item[2]
                max_x = item[3]
                min_y = item[4]
                max_y = item[5]
                min_x = min_x * 1 / picture_width
                max_x = max_x * 1 / picture_width
                min_y = min_y * 1 / picture_height
                max_y = max_y * 1 / picture_height
                # Check if car to close to side of car to be in camera view
                if abs(item[0][2]) < abs(item[0][0])/1.3:
                    continue
                if item[0][2] < 0.25:
                    continue
                if min_x < BORDER_CONSTANT and max_x > 1-BORDER_CONSTANT and min_y < BORDER_CONSTANT and max_y > 1-BORDER_CONSTANT:
                    continue
                if abs(float(min_x) - float(max_x)) <= 0.01 or abs(float(min_y) - float(max_y)) <= 0.01:
                    continue
                elif (max_y - min_y) / ((max_x - min_x) * 8) > 1:
                    continue
                else:
                    self.image_map[self.total_images] = index
                    self.total_images += 1

    def __len__(self):
        return self.total_images

    def __getitem__(self, idx):
        image = torch.zeros(7, constants.pose_network_width, constants.pose_network_height, dtype=torch.float).to(self.device)
        label_before_tensor = [0, 0, 0, 0]

        current_image_file_path = base_directory_for_images + "/data" + str(self.image_map[idx]) + "/img.png"
        file_image = None
        item = None
        if self.image_map[idx] in self.image_cache:
            file_image, item = self.image_cache[self.image_map[idx]]
        else:
            file_image = cv.imread(current_image_file_path, cv.IMREAD_COLOR)
            with open(base_directory_for_images + "/data" + str(self.image_map[idx]) + "/pose.pkl", 'rb') as f:
                item = pickle.load(f)
            self.image_cache[self.image_map[idx]] = file_image, item

        unmodified_camera_image = cv.cvtColor(file_image, cv.COLOR_BGR2RGB)
        min_x = item[2]
        max_x = item[3]
        min_y = item[4]
        max_y = item[5]
        min_x = min_x * 1 / picture_width
        max_x = max_x * 1 / picture_width
        min_y = min_y * 1 / picture_height
        max_y = max_y * 1 / picture_height

        # RANDOMIZE BBOX SLIGHTLY
        percent_BBOX_randomization = 0.2
        ran_min_x = min_x + (max_x-min_x) * (2*(0.5 - random.random())) * percent_BBOX_randomization
        ran_max_x = max_x + (max_x - min_x) * (2 * (0.5 - random.random())) * percent_BBOX_randomization
        ran_min_y = min_y + (max_y - min_y) * (2 * (0.5 - random.random())) * percent_BBOX_randomization
        ran_max_y = max_y + (max_y - min_y) * (2 * (0.5 - random.random())) * percent_BBOX_randomization

        # RESTRICT MODIFIED BBOX BOUNDS FROM 0 TO 1
        ran_min_x = max(0, min(1, ran_min_x))
        ran_max_x = max(0, min(1, ran_max_x))
        ran_min_y = max(0, min(1, ran_min_y))
        ran_max_y = max(0, min(1, ran_max_y))

        min_x = max(0, min(1, min_x))
        max_x = max(0, min(1, max_x))
        min_y = max(0, min(1, min_y))
        max_y = max(0, min(1, max_y))

        image = image_processing.prepare_image(unmodified_camera_image, ran_min_x, ran_max_x, ran_min_y, ran_max_y)
        pose_data = item[0]
        local_angle = constants.relative_angle((min_x + max_x) / 2, pose_data[3])
        label_before_tensor[0] = pose_data[0]
        label_before_tensor[1] = pose_data[2]
        label_before_tensor[3] = math.cos(local_angle)
        label_before_tensor[2] = -math.sin(local_angle)
        label = torch.tensor(label_before_tensor).to(self.device)
        return image, label


if __name__ == "__main__":
    dataset = CustomImageDataset()
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    for i, (images, labels) in enumerate(train_loader):
        if i>454:
            display_image = images[0][4:7]
            plt.imshow(display_image.to("cpu").permute(1, 2, 0))
            breakpoint()
