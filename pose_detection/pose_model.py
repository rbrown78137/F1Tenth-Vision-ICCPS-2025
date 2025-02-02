import torch
import torch.nn as nn
import constants

class PoseModel(nn.Module):

    def __init__(self):
        super(PoseModel, self).__init__()
        self.input_layers = 7
        self.conv_layer = nn.Sequential(
            nn.Conv2d(self.input_layers, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
        )
        self.ff_body = nn.Sequential(
            nn.Linear(in_features=256 * 16 * 16, out_features=4000),
            nn.ReLU(),
            nn.Linear(in_features=4000, out_features=1000),
            nn.ReLU(),
            nn.Linear(in_features=1000, out_features=512),
            nn.ReLU(),
        )
        self.ff_head = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_features=512, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=8),
        )

    def forward(self, input):
        output = self.conv_layer(input)
        output = self.ff_body(output)
        output = self.ff_head(output)
        return output

    def sample_model(self, input):
        conv_output = self.conv_layer(input)
        body_output = self.ff_body(conv_output)
        output_list = []
        for idx in range(constants.sampling_size):
            sample = self.ff_head(body_output).unsqueeze(0)
            output_list.append(sample)
        output_list = torch.cat(output_list, dim=0).to(constants.device)
        std_model = torch.std(output_list, dim=0).to(constants.device)
        average_model = torch.mean(output_list, dim=0).to(constants.device)
        return average_model[..., 0:4], std_model[..., 0:4], torch.abs(average_model[..., 4:8]), std_model[..., 0:4] + torch.abs(average_model[..., 4:8])


class PoseModelForConversion(nn.Module):

    def __init__(self):
        super(PoseModelForConversion, self).__init__()
        self.input_layers = 7
        self.conv_layer = nn.Sequential(
            nn.Conv2d(self.input_layers, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
        )
        self.ff_body = nn.Sequential(
            nn.Linear(in_features=256 * 16 * 16, out_features=4000),
            nn.ReLU(),
            nn.Linear(in_features=4000, out_features=1000),
            nn.ReLU(),
            nn.Linear(in_features=1000, out_features=512),
            nn.ReLU(),
        )
        self.ff_head = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_features=512, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=8),
        )

    def forward(self, input):
        conv_output = self.conv_layer(input)
        body_output = self.ff_body(conv_output)
        output_list = []
        for idx in range(constants.sampling_size):
            sample = self.ff_head(body_output).unsqueeze(0)
            output_list.append(sample)
        output_list = torch.cat(output_list, dim=0).to(constants.device)
        standard_deviation_of_distributions = torch.std(output_list, dim=0).to(constants.device)
        average_of_distributions = torch.mean(output_list, dim=0).to(constants.device)
        return average_of_distributions[..., 0:4], standard_deviation_of_distributions[..., 0:4] + torch.abs(average_of_distributions[..., 4:8])

    @torch.jit.export
    def sample_body(self, input):
        conv_output = self.conv_layer(input)
        body_output = self.ff_body(conv_output)
        return body_output

    @torch.jit.export
    def sample_head(self, input):
        output_list = []
        for idx in range(constants.sampling_size):
            sample = self.ff_head(input).unsqueeze(0)
            output_list.append(sample)
        output_list = torch.cat(output_list, dim=0).to(constants.device)
        standard_deviation_of_distributions = torch.std(output_list, dim=0).to(constants.device)
        average_of_distributions = torch.mean(output_list, dim=0).to(constants.device)
        return average_of_distributions[..., 0:4], standard_deviation_of_distributions[..., 0:4] + torch.abs(average_of_distributions[..., 4:8])
