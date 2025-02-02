import torch

import constants
from vicon_dataset import CustomImageDataset
from YOLOV3Tiny import YOLOV3Tiny
from loss import YoloLoss
from utils import mean_average_precision_and_recall, intersection_over_union
device = "cuda" if torch.cuda.is_available() else "cpu"

batch_size = 32
num_epochs = 1000
learning_rate = 1e-3 # was 1e-4
# weight_decay = 1e-4

classifications = constants.number_of_classes
dataset = CustomImageDataset()

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

model = YOLOV3Tiny(number_of_classes=constants.number_of_classes).to(device)
loss_function = YoloLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

highest_mean_avg_prec_saved = 0
scaled_anchors = (torch.tensor(constants.anchor_boxes) * torch.tensor(constants.split_sizes).unsqueeze(1).unsqueeze(2).repeat(1,3,2)).to(device)

for epoch in range(num_epochs):
    # Train Step
    sum_losses_over_batches = 0
    number_of_batches = 0
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        outputs = model(images)
        loss = (
            loss_function(outputs[0], labels[0].to(device), scaled_anchors[0]) +
            loss_function(outputs[1], labels[1].to(device), scaled_anchors[1])
        )
        sum_losses_over_batches += loss
        number_of_batches += 1
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    average_loss = sum_losses_over_batches / number_of_batches
    print(f"Epoch: {epoch} Loss: {average_loss}")

    # Evaluate Model After Epoch
    if epoch % 30 == 0 and epoch > 0:
        with torch.no_grad():
            mean_avg_prec_total = 0
            mean_avg_recall_total = 0
            total_counted = 0
            for i, (images, labels) in enumerate(test_loader):
                predictions = model(images.to(device))
                precision, recall = mean_average_precision_and_recall(predictions, labels, 0.6, images=images)
                if not(precision == None) and not (recall == None):
                    mean_avg_prec_total += precision
                    mean_avg_recall_total += recall
                    total_counted += 1
            if total_counted >0:
                mean_avg_prec = mean_avg_prec_total/total_counted
                mean_avg_recall = mean_avg_recall_total / total_counted
                print(f"Precision: {mean_avg_prec}")
                print(f"    Recall: {mean_avg_recall}")
                if mean_avg_prec >= 0 and mean_avg_prec >= highest_mean_avg_prec_saved:
                    highest_mean_avg_prec_saved = mean_avg_prec
                    PATH = './saved_models/train_network' + '.pth'  # + str(epoch) + '.pth'
                    torch.save(model.state_dict(), PATH)
            print(f"Precision: {0}")
            print(f"    Recall: {0}")

print('Finished Training')
