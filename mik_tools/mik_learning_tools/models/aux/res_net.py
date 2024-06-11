import torch
import torch.nn as nn



class Conv2DResNetBlock(nn.Module):
    def __init__(self, num_channels=3, hidden_dim=64, activation='relu'):
        self.num_channels = num_channels
        self.hidden_dim = hidden_dim
        super().__init__()
        self.activation = self._get_activation(activation)
        self.conv_1 = nn.Conv2d(self.num_channels, self.hidden_dim, kernel_size=3, stride=1, padding=1)
        self.conv_2 = nn.Conv2d(self.hidden_dim, self.num_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        # x: (...., num_channels, w, h)
        # x_out = (..., num_channels, w, h)
        dx = self.activation(self.conv_1(x))
        dx = self.conv_2(dx)
        x_out = x + dx
        return x_out

    @classmethod
    def _get_activation(cls, activation):
        if activation == 'relu':
            return nn.ReLU()
        elif activation == 'tanh':
            return nn.Tanh()
        elif activation == 'leaky_relu':
            return nn.LeakyReLU()
        elif activation == 'sigmoid':
            return nn.Sigmoid()
        elif activation is None:
            return nn.Identity()  # no activation
        else:
            raise NotImplementedError('Activation {} not supported'.format(activation))
