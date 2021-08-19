# LIBRARIES
import numpy as np
import torch as T
import torch.nn as nn

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

    def forward(self, x):
        return self.layers(x)

    # def save_checkpoint():
    #    print("...Saving Checkpoint in ", self.checkpoint_file)
    #    T.save(self.state_dict(), self.checkpoint_file)

    # def load_checkpoint():
    #    print("...Loading Checkpoint from ", self.checkpoint_file)
    #    self.load_state_dict(T.load(self.checkpoint_file))