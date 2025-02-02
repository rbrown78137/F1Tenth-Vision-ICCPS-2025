import torch
import constants
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time

device = "cuda" if torch.cuda.is_available() else "cpu"

def iou_width_height(boxes1, boxes2):
    """
    Parameters:
        boxes1 (tensor): width and height of the first bounding boxes
        boxes2 (tensor): width and height of the second bounding boxes
    Returns:
        tensor: Intersection over union of the corresponding boxes
    """
    intersection = torch.min(boxes1[..., 0], boxes2[..., 0]) * torch.min(
        boxes1[..., 1], boxes2[..., 1]
    )
    union = (
        boxes1[..., 0] * boxes1[..., 1] + boxes2[..., 0] * boxes2[..., 1] - intersection
    )
    return intersection / union

def intersection_over_union(boxes_preds, boxes_labels, box_format="midpoint"):

    if box_format == "midpoint":
        box1_x1 = boxes_preds[..., 0:1] - boxes_preds[..., 2:3] / 2
        box1_y1 = boxes_preds[..., 1:2] - boxes_preds[..., 3:4] / 2
        box1_x2 = boxes_preds[..., 0:1] + boxes_preds[..., 2:3] / 2
        box1_y2 = boxes_preds[..., 1:2] + boxes_preds[..., 3:4] / 2
        box2_x1 = boxes_labels[..., 0:1] - boxes_labels[..., 2:3] / 2
        box2_y1 = boxes_labels[..., 1:2] - boxes_labels[..., 3:4] / 2
        box2_x2 = boxes_labels[..., 0:1] + boxes_labels[..., 2:3] / 2
        box2_y2 = boxes_labels[..., 1:2] + boxes_labels[..., 3:4] / 2

    if box_format == "corners":
        box1_x1 = boxes_preds[..., 0:1]
        box1_y1 = boxes_preds[..., 1:2]
        box1_x2 = boxes_preds[..., 2:3]
        box1_y2 = boxes_preds[..., 3:4]
        box2_x1 = boxes_labels[..., 0:1]
        box2_y1 = boxes_labels[..., 1:2]
        box2_x2 = boxes_labels[..., 2:3]
        box2_y2 = boxes_labels[..., 3:4]

    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

    return intersection / (box1_area + box2_area - intersection + 1e-6)


def non_max_suppression(bboxes, iou_threshold, threshold, box_format="midpoint"):
    assert type(bboxes) == list

    bboxes = [box for box in bboxes if box[0] > threshold]
    bboxes = sorted(bboxes, key=lambda x: x[0], reverse=True)[0:50]
    bboxes_after_nms = []

    while bboxes:
        chosen_box = bboxes.pop(0)

        bboxes = [
            box
            for box in bboxes
            if box[5] != chosen_box[5]
            or intersection_over_union(
                torch.tensor(chosen_box[1:5]),
                torch.tensor(box[1:5]),
                box_format=box_format,
            )
            < iou_threshold
        ]

        bboxes_after_nms.append(chosen_box)

    return bboxes_after_nms


def get_bounding_boxes_for_prediction(prediction_tensor, batch_idx=0):
    first_split = prediction_tensor[0][batch_idx]
    second_split = prediction_tensor[1][batch_idx]
    anchors = torch.tensor(constants.anchor_boxes).to(device)
    bounding_boxes = cells_to_bboxes(first_split, anchors[0], constants.split_sizes[0]) + cells_to_bboxes(second_split, anchors[1], constants.split_sizes[1])
    return non_max_suppression(bounding_boxes,iou_threshold=0.5,threshold=0.45)


def get_label_boxes(labels, batch_index):
    bounding_boxes = []
    for S in range(2):
        for anchor in range(3):
            for row in range(len(labels[S][batch_index][anchor])):
                for col in range(len(labels[S][batch_index][anchor][row])):
                    if labels[S][batch_index][anchor][row][col][0] > 0.9:
                        bounding_boxes.append(
                            [
                                labels[S][batch_index][anchor][row][col][0],
                                (labels[S][batch_index][anchor][row][col][1] + col) / constants.split_sizes[S],
                                (labels[S][batch_index][anchor][row][col][2] + row) / constants.split_sizes[S],
                                labels[S][batch_index][anchor][row][col][3] / constants.split_sizes[S],
                                labels[S][batch_index][anchor][row][col][4] / constants.split_sizes[S],
                                labels[S][batch_index][anchor][row][col][5]
                            ]
                        )
    return non_max_suppression(bounding_boxes, threshold=.8,iou_threshold=.5)

def mean_average_precision_and_recall(predictions, labels, iou_threshold, images):
    total_precision = 0
    total_recall = 0
    precisions_counted = 0
    recall_counted = 0
    for index in range(len(labels[0])):

        predicted_bounding_boxes = get_bounding_boxes_for_prediction(predictions,batch_idx=index)

        label_bounding_boxes = get_label_boxes(labels, index)
        debug_var = 0
        if debug_var == 1:
            test = images[index]
            fig, ax = plt.subplots()
            ax.imshow(test.squeeze().to("cpu"))
            for label_generated in label_bounding_boxes:
                rect = patches.Rectangle(( (label_generated[1]-label_generated[3]/2)*constants.model_image_width, (label_generated[2]-label_generated[4]/2)*constants.model_image_height )
                                         ,label_generated[3]*constants.model_image_width, label_generated[4]*constants.model_image_height, linewidth=1, edgecolor='r', facecolor='none')
                ax.add_patch(rect)
            time.sleep(0.1)

        truePositives = 0
        falseNegative = 0
        for label_bounding_box_index in range(len(label_bounding_boxes)):
            detected = False
            for predicted_bounding_box_index in range(len(predicted_bounding_boxes)):
                if intersection_over_union(torch.tensor(label_bounding_boxes[label_bounding_box_index])[...,1:5], torch.tensor(predicted_bounding_boxes[predicted_bounding_box_index])[...,1:5]) > iou_threshold:
                    detected = True
            if detected:
                truePositives += 1
            else:
                falseNegative += 1
        if not (truePositives <= 0 and len(predicted_bounding_boxes)==0):
            total_precision += truePositives
            precisions_counted += len(predicted_bounding_boxes)
            if truePositives != len(predicted_bounding_boxes):
                debug_var = 0
            if falseNegative >0 or truePositives>0:
                total_recall += truePositives
                recall_counted += (truePositives + falseNegative)

    if precisions_counted == 0 or recall_counted == 0:
        return None, None
    return total_precision / precisions_counted, total_recall/recall_counted

def cells_to_bboxes(predictions, anchors, split_size):
    num_anchors = len(anchors)
    anchors = anchors.reshape(1, len(anchors), 1, 1, 2) * split_size
    box_predictions = predictions[..., 1:5]


    box_predictions[..., 0:2] = torch.sigmoid(box_predictions[..., 0:2])
    box_predictions[..., 2:] = torch.exp(box_predictions[..., 2:]) * anchors

    scores = torch.sigmoid(predictions[..., 0:1])
    best_class = torch.argmax(predictions[..., 5:], dim=-1).unsqueeze(-1)

    cell_indices = (
        torch.arange(split_size)
        .repeat(3, split_size, 1)
        .unsqueeze(-1)
        .to(predictions.device)
    )
    x = 1 / split_size * (box_predictions[..., 0:1] + cell_indices)
    y = 1 / split_size * (box_predictions[..., 1:2] + cell_indices.permute(0, 2, 1, 3))
    w_h = 1 / split_size * box_predictions[..., 2:4]
    converted_bboxes = torch.cat((scores, x, y, w_h, best_class), dim=-1).reshape(num_anchors * split_size * split_size, 6)
    return converted_bboxes.tolist()