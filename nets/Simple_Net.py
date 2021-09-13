# LIBRARIES
import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F


# Deep Q-Network Architecture
class DQN_Vanilla(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(DQN_Vanilla, self).__init__()
        """
        Standard Deep Q-Network

        Args:
            num_inputs:     # Input Channels
            num_outputs:    # Outputs (One per action on environment)

        Architecture:
            Normal Input: 84 x 84 pixels and 4 channels
            First Layer: Convolution of (16 filters of 8 x 8) and stride: 4
                -> (16 , (20 x 20))
                -> Applies rectifier non linearity (ReLU)
            Second Layer: Convolution of (32 filters of 4 x 4) and stride: 2
                -> (32 , (9 x 9))
                -> Applies rectifier non linearity (ReLU)
            Third Layer: Convolution of (64 filters of 3 x 3) and stride: 1
                -> (64 , (7 x 7))
                -> Applies rectifier non linearity (ReLU)
            Final Layer: Fully connected 512 rectifier units
            Output Layer: Fully connected linear layer with outputs=actions
                -> ( 512 , num_outputs )
        """

        self.conv1 = nn.Conv2d(num_inputs, 32, kernel_size=8, stride=4)
        self.batch_n1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.batch_n2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=2, stride=1)
        self.batch_n3 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(7 * 7 * 64, 512)
        self.out = nn.Linear(512, num_outputs)

    def forward(self, state):
        x = state.float() / 255
        x = nn.ReLU(self.batch_n1(self.conv1(x)))
        x = nn.ReLU(self.batch_n2(self.conv2(x)))
        x = nn.ReLU(self.batch_n3(self.conv3(x)))
        x = nn.ReLU(self.fc1(x.view(x.size(0), -1)))
        return self.out(x)


class Simple_Net(nn.Module):
    def __init__(self, config):
        super(Simple_Net, self).__init__()

        # self.checkpoint_dir = chkpt_dir
        # self.checkpoint_file = os.path.join(self.checkpoint_dir, name)

        self.num_inputs = config.state_space_input
        self.num_outputs = config.action_num

        self.layers = nn.Sequential(
            nn.Linear(self.num_inputs, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, self.num_outputs)
        )

        # Initial Parameters
        self.initial_parameters = self.state_dict()

    def forward(self, x):
        return self.layers(x)

    def reset_parameters(self):
        self.load_state_dict(self.initial_parameters)

    # def save_checkpoint():
    #    print("...Saving Checkpoint in ", self.checkpoint_file)
    #    T.save(self.state_dict(), self.checkpoint_file)

    # def load_checkpoint():
    #    print("...Loading Checkpoint from ", self.checkpoint_file)
    #    self.load_state_dict(T.load(self.checkpoint_file))


# Powers of Dueling Q Network
class Dueling_Net(nn.Module):
    def __init__(self, config):
        super(Dueling_Net, self).__init__()

        self.num_inputs = config.state_space_input
        self.num_outputs = config.action_num

        # Common NN Layers
        self.feature = nn.Sequential(
            nn.Linear(self.num_inputs, 128),
            nn.ReLU()
        )

        # Advantage of each action Layers
        self.advantage = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, self.num_outputs)  # Output for each action
        )

        # State-Value Layers
        self.value = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)  # Only output a scalar
        )

        # Initial Parameters
        self.initial_parameters = self.state_dict()

    def forward(self, x):
        x = self.feature(x)  # Common Layers
        advantage = self.advantage(x)  # Advantage Stream
        value = self.value(x)  # State-Value Stream
        return value + advantage - advantage.mean()  # Aggregation

    def reset_parameters(self):
        self.load_state_dict(self.initial_parameters)


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
        self.linear2 = nn.Linear(in_features=128, out_features=self.output)

        # Initial Parameters
        self.initial_parameters = self.state_dict()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)

        x = F.relu(self.bn3(self.conv3(x)))

        x = x.view(x.size(0), -1)  # Flatten every batch

        x = F.leaky_relu(self.linear1(x))
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
