import settings
import utils
import importlib
import numpy as np
import torch
import torch.nn as nn

class Simple_CNN(nn.Module):

    def __init__(self, config):
        super(Simple_CNN, self).__init__()

        self.output = config.action_num

        self.conv1 = nn.Conv2d(in_channels=4, out_channels=32, kernel_size=8, stride=4)
        self.bn1 = nn.BatchNorm2d(32)
        convw, convh = self.conv2d_size_calc(config.state_size_w, config.state_size_h, kernel_size=8, stride=4)

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        convw, convh = self.conv2d_size_calc(convw, convh, kernel_size=4, stride=2)

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(64)
        convw, convh = self.conv2d_size_calc(convw, convh, kernel_size=3, stride=1)

        linear_input_size = convw * convh * 64  # Last conv layer's out sizes

        # Linear Action Output layer
        self.linear1 = nn.Linear(in_features=linear_input_size, out_features=128)
        self.linear_relu = nn.LeakyReLU()  # Linear 1 activation function
        self.linear2 = nn.Linear(in_features=128, out_features=self.output)

        # Initial Parameters
        self.initial_parameters = self.state_dict()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = nn.ReLU(x)

        # x = nn.ReLU(self.bn1(self.conv1(x)))
        # x = nn.ReLU(self.bn2(self.conv2(x)))

        x = self.conv2(x)
        x = self.bn2(x)
        x = nn.ReLU(x)

        x = nn.ReLU(self.bn3(self.conv3(x)))

        x = x.view(x.size(0), -1)  # Flatten every batch

        x = self.linear_relu(self.linear1(x))
        x = self.linear2(x)  # No activation on last layer

        return x

    def conv2d_size_calc(self, w, h, kernel_size, stride):
        """
                Calculates conv layers output image sizes
        """
        next_w = (w - (kernel_size - 1) - 1) // stride + 1
        next_h = (h - (kernel_size - 1) - 1) // stride + 1
        return next_w, next_h

    def reset_parameters(self):
        self.load_state_dict(self.initial_parameters)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config = settings.config

# Constructing Environment
env = utils.create_environment(config)

# Setting The CNN Network

# Net = getattr(importlib.import_module("nets." + "Simple_Net"), config.net)
current_model = Simple_CNN(config)
current_model.to(device)
state = env.reset()

state = utils.pre_processing_img(state, config.state_size_h, config.state_size_w, config.env_type)
state = np.stack((state, state, state, state))

# Taking Action
state = torch.tensor(state, dtype=torch.float).unsqueeze(0)
with torch.no_grad():
    q_value = current_model.forward(state.to(device))
action = torch.argmax(q_value).item()
print(action)
