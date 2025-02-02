import torch
from torch import nn
import pickle
from vicon_dataset import CustomImageDataset
from pose_model import PoseModel
from loss import PoseEstimationLoss

device = "cuda" if torch.cuda.is_available() else "cpu"

batch_size = 64
num_epochs = 1000
learning_rate = 0.0001  # was 0.00001 for 200 epoch model # was 0.0001 before change on august 15th 2023

dataset = CustomImageDataset()

train_size = int(0.9 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size],generator=torch.Generator().manual_seed(30))
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

model = PoseModel().to(device)

loss_function = PoseEstimationLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


lowest_validation_loss = 1000

absolute_loss = nn.L1Loss()

y_error_epoch = []
y_error_ground_truth = []
y_error_internal_prediction = []
y_error_model_error = []

for epoch in range(num_epochs):
    lossForEpoch = 0
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        loss = loss_function(outputs, labels)
        lossForEpoch += loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch: {epoch} Loss: {lossForEpoch/train_size*batch_size}")
    if epoch > 0 and epoch % 5 == 0:
        with torch.no_grad():
            total_x_error = 0
            total_y_error = 0
            total_varphi_x_error = 0
            total_varphi_y_error = 0

            total_x_internal_error = 0
            total_y_internal_error = 0
            total_varphi_x_internal_error = 0
            total_varphi_y_internal_error = 0

            total_x_model_error = 0
            total_y_model_error = 0
            total_varphi_x_model_error = 0
            total_varphi_y_model_error = 0

            total_x_combined_error = 0
            total_y_combined_error = 0
            total_varphi_x_combined_error = 0
            total_varphi_y_combined_error = 0

            validation_loss_total = 0

            for i, (images, labels) in enumerate(test_loader):
                labels = labels.to(device)
                predictions = model(images.to(device))
                ground_truths = labels.to(device)
                average_error, sampling, gaussian, mix = model.sample_model(images.to(device))

                total_x_error += torch.std(
                    average_error[..., 0:1] -
                    labels[..., 0:1]
                )

                total_y_error += torch.std(
                    average_error[..., 1:2] -
                    labels[..., 1:2]
                )

                total_varphi_x_error += torch.std(
                    average_error[..., 2:3] -
                    labels[..., 2:3]
                )

                total_varphi_y_error += torch.std(
                    average_error[..., 3:4] -
                    labels[..., 3:4]
                )

                total_x_internal_error += torch.mean(gaussian[..., 0:1])
                total_y_internal_error += torch.mean(gaussian[..., 1:2])
                total_varphi_x_internal_error += torch.mean(gaussian[..., 2:3])
                total_varphi_y_internal_error += torch.mean(gaussian[..., 3:4])

                total_x_model_error += torch.mean(sampling[..., 0:1])
                total_y_model_error += torch.mean(sampling[..., 1:2])
                total_varphi_x_model_error += torch.mean(sampling[..., 2:3])
                total_varphi_y_model_error += torch.mean(sampling[..., 3:4])

                total_x_combined_error += torch.mean(mix[..., 0:1])
                total_y_combined_error += torch.mean(mix[..., 1:2])
                total_varphi_x_combined_error += torch.mean(mix[..., 2:3])
                total_varphi_y_combined_error += torch.mean(mix[..., 3:4])

                validation_loss = loss_function(predictions, ground_truths)
                validation_loss_total += validation_loss * batch_size

            average_validation_loss = validation_loss_total/ test_size

            print(f"Epoch: {epoch} Validation Loss: {average_validation_loss}")
            print(f"------: x Error: {total_x_error * batch_size / test_size}")
            print(f"------: y Error: {total_y_error * batch_size / test_size}")
            print(f"------: varphi_x Error: {total_varphi_x_error * batch_size / test_size}")
            print(f"------: varphi_y Error: {total_varphi_y_error * batch_size / test_size}")

            print(f"------: x internal Error: {total_x_internal_error * batch_size / test_size}")
            print(f"------: y internal Error: {total_y_internal_error * batch_size / test_size}")
            print(f"------: varphi_x internal Error: {total_varphi_x_internal_error * batch_size / test_size}")
            print(f"------: varphi_y internal Error: {total_varphi_y_internal_error * batch_size / test_size}")

            print(f"------: x model Error: {total_x_model_error * batch_size / test_size}")
            print(f"------: y model Error: {total_y_model_error * batch_size / test_size}")
            print(f"------: varphi_x model Error: {total_varphi_x_model_error * batch_size / test_size}")
            print(f"------: varphi_y model Error: {total_varphi_y_model_error * batch_size / test_size}")

            print(f"------: x mix Error: {total_x_combined_error * batch_size / test_size}")
            print(f"------: y mix Error: {total_y_combined_error * batch_size / test_size}")
            print(f"------: varphi_x mix Error: {total_varphi_x_combined_error * batch_size / test_size}")
            print(f"------: varphi_y mix Error: {total_varphi_y_combined_error * batch_size / test_size}")
            y_error_epoch.append(epoch)
            y_error_ground_truth.append((total_y_error * batch_size / test_size))
            y_error_internal_prediction.append((total_y_internal_error * batch_size / test_size))
            y_error_model_error.append((total_y_model_error * batch_size / test_size))
            if average_validation_loss < lowest_validation_loss:
                lowest_validation_loss = average_validation_loss
                PATH = './saved_models/yaw_network' + '.pth'  # + str(epoch) + '.pth'
                torch.save(model.state_dict(), PATH)
                print('Saved Model')
                y_error_prediction_data_over_epochs = {}
                y_error_prediction_data_over_epochs["epoch"] = y_error_epoch
                y_error_prediction_data_over_epochs["ground_truth"] = y_error_ground_truth
                y_error_prediction_data_over_epochs["internal_model_prediction"] = y_error_internal_prediction
                y_error_prediction_data_over_epochs["model_error"] = y_error_model_error
                with open('y_error_graph_data.pkl', 'wb') as f:
                    pickle.dump(y_error_prediction_data_over_epochs, f)
print('Finished Training')
