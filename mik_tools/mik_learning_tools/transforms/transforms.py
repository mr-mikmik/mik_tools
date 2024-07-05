import torch
import torch.nn
import torch.nn.functional as F


class GaussianNoiseTr(object):
    def __init__(self, noise_level):
        self.noise_level = noise_level

    def __call__(self, x):
        x_tr = x + self.noise_level * 2 * (torch.randn(size=x.shape, device=x.device) - 0.5)
        return x_tr


class DilateTr(object):
    def __init__(self, kernel_size):
        self.kernel_size = kernel_size
        self.kernel = torch.ones((1, 1, self.kernel_size, self.kernel_size))

    def __call__(self, x):
        x_tr = F.conv2d(x, self.kernel.to(x.device), padding=(self.kernel_size // 2, self.kernel_size // 2))
        return x_tr


class ErodeTr(object):
    def __init__(self, kernel_size):
        self.kernel_size = kernel_size
        self.kernel = torch.ones((1, 1, self.kernel_size, self.kernel_size))

    def __call__(self, x):
        x_masked = (x > 0.5).to(torch.float32)
        x_tr = F.conv2d(x_masked, self.kernel.to(x.device), padding=(self.kernel_size // 2, self.kernel_size // 2))
        x_tr = (x_tr > self.kernel_size ** 2 * 0.5).to(torch.float32)
        return x_tr


class ToMaskTr(object):
    def __init__(self, threshold=0.5):
        self.threshold = threshold

    def __call__(self, x):
        x_tr = (x > self.threshold).to(torch.float32)
        return x_tr