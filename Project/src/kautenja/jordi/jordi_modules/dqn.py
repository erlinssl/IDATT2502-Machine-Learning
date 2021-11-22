import torch
import torch.nn as nn
import numpy as np

temp = 0


class DQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DQN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3),
            nn.ReLU()
        )

        conv_out_size = self._get_conv_out(input_shape)

        # print(conv_out_size)

        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 256),
            nn.ReLU(),
            nn.Linear(256, n_actions)
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        global temp
        temp += 1
        if temp > 7500 and temp % 1001 == 0:
            print(x)
        conv_out = self.conv(x).view(x.size()[0], -1)
        return self.fc(conv_out).double()
