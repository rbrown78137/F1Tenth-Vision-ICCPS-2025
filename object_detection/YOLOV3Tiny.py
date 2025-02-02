import torch
import torch.nn as nn
import constants

total_number_of_anchor_boxes = len(constants.anchor_boxes[0])


class BackBone(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.layers(x)


class Block2(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.AvgPool2d(2),
            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 1024, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 256, 1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.layers(x)


class Block3(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(256, 128, 1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )

    def forward(self, x):
        return self.layers(x)


class Scale26(nn.Module):
    def __init__(self, number_of_classifications, number_of_anchor_boxes):
        super().__init__()
        self.number_of_classes = number_of_classifications
        self.number_of_anchor_boxes = number_of_anchor_boxes
        self.layers = nn.Sequential(
            nn.Conv2d(384, 256, 3,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, (number_of_classifications + 5) * self.number_of_anchor_boxes, 1),
        )

    def forward(self, x):
        return self.layers(x).reshape(x.shape[0], self.number_of_anchor_boxes, (self.number_of_classes + 5), x.shape[2], x.shape[3]).permute(0, 1, 3, 4, 2)


class Scale13(nn.Module):
    def __init__(self, number_of_classifications,number_of_anchor_boxes):
        super().__init__()
        self.number_of_classes = number_of_classifications
        self.number_of_anchor_boxes = number_of_anchor_boxes
        self.layers = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, (number_of_classifications + 5) * self.number_of_anchor_boxes, 1),
        )

    def forward(self, x):
        return self.layers(x).reshape(x.shape[0], self.number_of_anchor_boxes, (self.number_of_classes + 5), x.shape[2], x.shape[3]).permute(0, 1, 3, 4, 2)


class YOLOV3Tiny(nn.Module):
    def __init__(self, in_channels=3, number_of_classes=constants.number_of_classes,number_of_anchor_boxes = total_number_of_anchor_boxes):
        super().__init__()
        self.in_channels = in_channels
        self.number_of_classes = number_of_classes
        self.backbone = BackBone(in_channels=in_channels)
        self.block2 = Block2()
        self.block3 = Block3()
        self.scale13 = Scale13(number_of_classes, number_of_anchor_boxes)
        self.scale26 = Scale26(number_of_classes, number_of_anchor_boxes)

    def forward(self, x):
        outputs = []
        x_after_backbone = self.backbone(x)
        x_after_block2 = self.block2(x_after_backbone)
        x_after_block3 = self.block3(x_after_block2)
        input_to_scale26 = torch.cat([x_after_backbone, x_after_block3], dim=1)
        scale_13_output = self.scale13(x_after_block2)
        scale_26_output = self.scale26(input_to_scale26)
        outputs.append(scale_13_output)
        outputs.append(scale_26_output)
        return outputs

# if __name__ == "__main__":
#     num_classes = 2
#     IMAGE_SIZE = 416
#     model = YOLO(number_of_classes=num_classes)
#     x = torch.randn((2, 1, IMAGE_SIZE, IMAGE_SIZE))
#     out = model(x)
#     assert model(x)[0].shape == (2, number_of_anchor_boxes, IMAGE_SIZE // 32, IMAGE_SIZE // 32, num_classes + 5)
#     assert model(x)[1].shape == (2, number_of_anchor_boxes, IMAGE_SIZE // 16, IMAGE_SIZE // 16, num_classes + 5)
#     print("Success!")