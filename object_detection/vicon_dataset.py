import pickle
import matplotlib
import torch
from torch.utils.data import Dataset
import os
import cv2 as cv
import constants
from utils import iou_width_height
import matplotlib.pyplot as plt
import image_modifications

# General global variables for reading data input
base_directory_for_images = '../paper_set_august_2023'

picture_width = 640
picture_height = 480

class CustomImageDataset(Dataset):
    def __init__(self):
        self.split_sizes = constants.split_sizes
        self.anchors = []
        for split_size in range(len(constants.anchor_boxes)):
            self.anchors = self.anchors + constants.anchor_boxes[split_size]
        self.anchors = torch.tensor(self.anchors)
        self.number_of_anchors = self.anchors.shape[0]
        self.number_of_anchors_per_scale = self.number_of_anchors // len(constants.anchor_boxes)
        self.ignore_iou_threshold = 0.5
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.cached_items = {}
        # Set File Count and Maps To Files
        self.total_images = len(next(os.walk(base_directory_for_images))[1])

    def __len__(self):
        return self.total_images

    def __getitem__(self, idx):
        # print(idx)
        if idx in self.cached_items:
            return self.cached_items[idx]
        if not idx in self.cached_items:
            image = torch.zeros(1, constants.model_image_width, constants.model_image_height, dtype=torch.float)
            # 6 = [ Prob, x, y, w, h, class]
            label = [torch.zeros((self.number_of_anchors_per_scale, scale_size, scale_size, 6)) for scale_size in self.split_sizes]

            # Get Input Image
            current_image_file_path = base_directory_for_images + "/data" + str(idx) + "/img.png"
            unmodified_camera_image = cv.cvtColor(cv.imread(current_image_file_path, cv.IMREAD_COLOR), cv.COLOR_BGR2RGB)
            # unmodified_camera_image = image_modifications.change_brightness(unmodified_camera_image, value=(-80 + random.random()*160))
            unmodified_camera_image = image_modifications.noisy("gaussian", unmodified_camera_image)
            resized_image = cv.resize(unmodified_camera_image, (constants.model_image_width, constants.model_image_height))
            camera_tensor = torch.from_numpy(resized_image)
            camera_tensor = camera_tensor.to(self.device).permute(2, 0, 1).to(torch.float).mul(1/256)
            image = camera_tensor
            debug_var = 0
            if debug_var == 1:
                plt.imshow(camera_tensor.to("cpu").permute(1, 2, 0))
            # Get Label
            # Load File
            item = None
            with open(base_directory_for_images + "/data" + str(idx) + "/pose.pkl", 'rb') as f:
                item = pickle.load(f)
            min_x = item[2]
            max_x = item[3]
            min_y = item[4]
            max_y = item[5]
            min_x = min_x * 1 / picture_width
            max_x = max_x * 1 / picture_width
            min_y = min_y * 1 / picture_height
            max_y = max_y * 1 / picture_height
            if min_x == max_x or min_y == max_y:
                self.cached_items[idx] = (image, label)
                return image, label
            if (max_y - min_y) / ((max_x - min_x)*8) > 1:
                self.cached_items[idx] = (image, label)
                return image, label
            debug_var2 = 0
            if debug_var2 == 1:
                plt.imshow(camera_tensor.to("cpu").permute(1, 2, 0))
                plt.gca().add_patch(matplotlib.patches.Rectangle((min_x*constants.model_image_width, min_y*constants.model_image_width), (max_x - min_x)*constants.model_image_width, (max_y - min_y)*constants.model_image_height, linewidth=1, edgecolor='r', facecolor='none'))
            classification = 1
            # Test Code
            if min_x == 0 and max_x == 1 and min_y == 0 and max_y ==1:
                classification = 0
            center_x = (min_x + max_x) / 2
            center_y = (min_y + max_y) / 2
            height_of_box = abs(max_y - min_y)
            width_of_box = abs(max_x - min_x)
            iou_anchors = iou_width_height(torch.tensor([width_of_box, height_of_box]), self.anchors)
            anchor_indicies = iou_anchors.argsort(descending=True, dim=0)

            has_anchor = [False] * 3
            for anchor_idx in anchor_indicies:
                anchor_idx = anchor_idx.item()
                scale_idx = anchor_idx // self.number_of_anchors_per_scale
                anchor_idx_on_scale = anchor_idx % self.number_of_anchors_per_scale
                scale = self.split_sizes[scale_idx]
                cell_row = int(center_y * scale)
                cell_col = int(center_x * scale)
                anchor_taken = label[scale_idx][anchor_idx_on_scale][cell_row][cell_col][0]
                if not anchor_taken and not has_anchor[scale_idx]:
                    x_cell = center_x * scale - cell_col
                    y_cell = center_y * scale - cell_row
                    width_cell = width_of_box * scale
                    height_cell = height_of_box * scale
                    label[scale_idx][anchor_idx_on_scale][cell_row][cell_col] = torch.tensor([1, x_cell, y_cell, width_cell, height_cell, classification])
                    has_anchor[scale_idx] = True
                elif not anchor_taken and iou_anchors[anchor_idx]>self.ignore_iou_threshold:
                    label[scale_idx][anchor_idx_on_scale,cell_row,cell_col,0] = -1
            self.cached_items[idx] = (image,label)
            return image, label


if __name__ == "__main__":
    dataset = CustomImageDataset()
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True)
    for i, (images, labels) in enumerate(train_loader):
        test = 1
