import torch
import torch.nn as nn
import abc
from mik_tools.mik_learning_tools.learning_tools.batched_decorators import batched_1d_operation, batched_img_operation, fake_batched_operation


class NormalizerBase(nn.Module):
    def __init__(self):
        super().__init__()
        self.batch_norm = self._get_batch_norm()
        self.batched_operation = self._get_batched_operation()

    def forward(self, x):
        """Normalize while update the normalization weights"""
        out = self.batched_operation(self.batch_norm, x)
        return out

    def normalize(self, x):
        """Just normalize, no weight update here"""
        out = self.batched_operation(self._normalize, x)
        return out

    def denormalize(self, x):
        """Undo the normalization, no weight update here"""
        out = self.batched_operation(self._denormalize, x)
        return out

    def get_parameters(self):
        parameters = {
        'mean': self.batch_norm._buffers['running_mean'],
        'var': self.batch_norm._buffers['running_var'],
        'eps': self.batch_norm.eps,
        'gamma': self.batch_norm._parameters['weight'],
        'beta': self.batch_norm._parameters['bias'],
                }
        return parameters


    @abc.abstractmethod
    def _get_batch_norm(self):
        pass

    @abc.abstractmethod
    def _get_batched_operation(self):
        pass

    @abc.abstractmethod
    def _normalize(self, x):
        pass

    @abc.abstractmethod
    def _denormalize(self, x):
        pass


class FakeNormalizer(NormalizerBase):
    def _get_batch_norm(self):
        return self._fake_operation

    def _get_batched_operation(self):
        return fake_batched_operation

    def _normalize(self, x):
        return x

    def _denormalize(self, x):
        return x

    def _fake_operation(self, x):
        return x


class Normalizer1d(NormalizerBase):
    def __init__(self, number_of_features):
        self.number_of_features = number_of_features
        super().__init__()

    def _get_batch_norm(self):
        batch_norm = nn.BatchNorm1d(self.number_of_features)
        # freeze the parameters of the batch norm since we do not want them to be modified.
        for param in batch_norm.parameters():
            param.requires_grad = False
        return batch_norm

    def _get_batched_operation(self):
        return batched_1d_operation

    def _normalize(self, x):
        mean = self.batch_norm._buffers['running_mean']
        var = self.batch_norm._buffers['running_var']
        eps = self.batch_norm.eps
        gamma = self.batch_norm._parameters['weight']
        beta = self.batch_norm._parameters['bias']
        x_norm = ((x - mean) / torch.sqrt(var + eps)) * gamma + beta
        return x_norm

    def _denormalize(self, x_norm):
        mean = self.batch_norm._buffers['running_mean']
        var = self.batch_norm._buffers['running_var']
        eps = self.batch_norm.eps
        gamma = self.batch_norm._parameters['weight']
        beta = self.batch_norm._parameters['bias']
        x = (x_norm - beta)/gamma*torch.sqrt(var + eps)+mean
        return x


class ImageNormalizer(NormalizerBase):
    """
    Expects elements to be normalized to be of shape (...., num_channels, h, w)
    Normalizes the channels independently. (all h,w are normalized with the same value per channel)
    """

    def __init__(self, number_of_channels):
        self.number_of_channels = number_of_channels
        super().__init__()

    def _get_batch_norm(self):
        batch_norm = nn.BatchNorm2d(self.number_of_channels)
        # freeze the parameters of the batch norm since we do not want them to be modified.
        for param in batch_norm.parameters():
            param.requires_grad = False
        return batch_norm

    def _get_batched_operation(self):
        return batched_img_operation

    def _normalize(self, imprint):
        mean = self.batch_norm._buffers['running_mean']
        var = self.batch_norm._buffers['running_var']
        eps = self.batch_norm.eps
        gamma = self.batch_norm._parameters['weight']
        beta = self.batch_norm._parameters['bias']
        norm_imprint_r = (imprint.swapaxes(1, 3) - mean)/torch.sqrt(var + eps)*gamma + beta
        norm_imprint = norm_imprint_r.swapaxes(1, 3)    # swap axes back
        return norm_imprint

    def _denormalize(self, norm_imprint):
        mean = self.batch_norm._buffers['running_mean']
        var = self.batch_norm._buffers['running_var']
        eps = self.batch_norm.eps
        gamma = self.batch_norm._parameters['weight']
        beta = self.batch_norm._parameters['bias']
        imprint_r = (norm_imprint.swapaxes(1, 3) - beta)/gamma*torch.sqrt(var + eps)+mean
        imprint = imprint_r.swapaxes(1, 3) # swap axes back
        return imprint



# TESTING:


def test_normalizer_1d():
    num_feats = 5
    x = torch.randn((10, num_feats))

    normalizer = Normalizer1d(num_feats)
    import pdb; pdb.set_trace()
    x_norm = normalizer.normalize(x)
    x_denorm = normalizer.denormalize(x_norm)


def test_img_normalization():
    x_img = torch.cat([
        torch.cat([torch.ones((1, 3, 4, 2)), torch.zeros((1, 3, 4, 2))], dim=-1),
        torch.cat([torch.zeros((1, 3, 4, 2)), torch.ones((1, 3, 4, 2))], dim=-1),
        ], dim=0)
    x_img[:,1,:,:] *= 2
    x_img[:,2,:,:] *= 3
    normalizer = ImageNormalizer(3)
    # estimate the parameters
    for i in range(100):
        _ = normalizer(x_img)
    # normalize
    x_norm = normalizer.normalize(x_img)
    x_denorm = normalizer.denormalize(x_norm)
    import pdb; pdb.set_trace()
    params = normalizer.get_parameters()


if __name__ == '__main__':
    # test_normalizer_1d()
    test_img_normalization()


