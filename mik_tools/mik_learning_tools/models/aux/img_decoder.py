import numpy as np
import torch
import torch.nn as nn

from mik_tools.mik_learning_tools.models.aux import FCModule, FakeModel


class ImageDecoder(nn.Module):
    """
    Module composed by FC layers and 2D Inverse Convolutions (Transposed Conv)
    """
    def __init__(self, output_size, latent_size, num_convs=3, conv_h_sizes=None, ks=4, stride=1, padding=0, dilation=1, num_fcs=2, pooling_size=1, fc_hidden_size=50, activation='relu', out_linear=False):
        super().__init__()
        self.output_size = output_size # (C_out, W_out, H_out)
        self.latent_size = latent_size
        self.out_linear = out_linear
        self.num_convs, self.hidden_dims = self._get_convs_h_sizes(num_convs, conv_h_sizes)
        self.ks = self._get_conv_property(ks)
        self.stride = self._get_conv_property(stride)
        self.padding = self._get_conv_property(padding)
        self.dilation = self._get_conv_property(dilation)
        self.pooling_size = self._get_conv_property(pooling_size)
        self.num_fcs = num_fcs
        self.fc_hidden_size = fc_hidden_size
        self.act = self._get_activation(activation) # only used in conv
        self.conv_decoder, self.conv_in_size, self.img_conv_sizes = self._get_conv_decoder()
        self.fc_decoder = self._get_fc_decoder()
        self.fc_out = self._get_fc_out()

    def forward(self, z):
        in_size = z.shape  # shape (Batch_size, ..., Latent_in)
        z = torch.flatten(z, start_dim=0, end_dim=-2)  # (BatchExtendedSize, Latent_in)
        conv_in = self.fc_decoder(z) # adjust the shape for the convolutions
        conv_in = conv_in.view(z.size()[:-1] + tuple(self.conv_in_size))  # shape (BatchExtendedSize, C_in, H_in, W_in)
        conv_out = self.conv_decoder(conv_in) # shape (BatchExtendedSize, C_out, H_out, W_out)
        img_out = self.fc_out(conv_out)
        img_out = img_out.reshape(*in_size[:-1], *img_out.shape[-3:])
        return img_out

    def _get_convs_h_sizes(self, num_convs, conv_h_sizes):
        if conv_h_sizes is None:
            hidden_dims = [self.output_size[-3]]*num_convs + [self.output_size[-3]]
        else:
            hidden_dims = conv_h_sizes + [self.output_size[-3]]
            num_convs = len(conv_h_sizes)
        return num_convs, hidden_dims

    def _get_conv_property(self, ks):
        if type(ks) in [int]:
            # single ks, we need to extend it ot num_convs
            ks = np.array([ks]*self.num_convs)
        elif type(ks) in [np.ndarray, torch.Tensor]:
            pass
        elif type(ks) in [list, tuple]:
            ks = np.asarray(ks)
        else:
            raise NotImplementedError(f'Option to set the conv property with {ks} is not available yet. Please, check the requirements at img_encoder.py')
        assert len(
            ks) == self.num_convs, f'We must have the same args as num_convs. len(arg)={len(ks)}, num_convs={self.num_convs}'
        return ks

    def _get_conv_decoder(self):
        conv_modules = []
        sizes = [self.output_size[-2:]]
        for i in reversed(range(len(self.hidden_dims)-1)):
            h_dim = self.hidden_dims[i]
            out_dim = self.hidden_dims[i+1]
            ks = self.ks[i]
            stride = self.stride[i]
            padding = self.padding[i]
            dilation = self.dilation[i]
            pooling_size_i = self.pooling_size[i]
            _size_up = (sizes[-1] - 1 + 2 * padding - dilation * (ks - 1))
            in_size_i = np.floor(_size_up / stride + 1).astype(np.int64)
            expected_out_size = (in_size_i - 1)*stride - 2*padding + dilation*(ks - 1) + 1
            out_padding = sizes[-1][-2:] - expected_out_size
            conv_i = nn.ConvTranspose2d(in_channels=h_dim, out_channels=out_dim, kernel_size=ks, padding=padding, dilation=dilation, stride=stride, output_padding=out_padding)
            conv_modules.append(conv_i)
            sizes.append(in_size_i)
            if i < len(self.hidden_dims)-2:
                conv_modules.append(self.act)
        conv_modules.reverse()
        if len(conv_modules) > 0:
            conv_encoder = nn.Sequential(*conv_modules)
        else:
            conv_encoder = nn.Identity() # no operation needed since there are no convolutions
        # compute the tensor sizes:
        sizes.reverse()
        conv_img_in_size_wh = sizes[0]
        conv_img_in_size = np.insert(conv_img_in_size_wh, 0, self.hidden_dims[0]) # ( C_in, H_in, W_in)
        return conv_encoder, conv_img_in_size, sizes

    def _get_fc_decoder(self):
        fc_out_size = int(np.prod(self.conv_in_size))
        sizes = [self.latent_size] + [self.fc_hidden_size]*(self.num_fcs-1) + [fc_out_size]
        fc_encoder = FCModule(sizes, activation='relu')
        return fc_encoder

    def _get_fc_out(self):
        if self.out_linear:
            fc_out = ImgLinearLayer(self.img_conv_sizes[-1])
        else:
            fc_out = FakeModel() # no operation
        return fc_out

    def _get_activation(self, activation):
        if activation == 'relu':
            return nn.ReLU()
        elif activation == 'tanh':
            return nn.Tanh()
        elif activation == 'leaky_relu':
            return nn.LeakyReLU()
        else:
            raise NotImplementedError('Activation {} not supported'.format(activation))


class ImgLinearLayer(nn.Module):
    def __init__(self, img_shape):
        super().__init__()
        self.img_shape = img_shape
        wh = int(np.prod(self.img_shape[-2:]).astype(np.int64))
        self.fc_layer = nn.Linear(wh, wh)

    def forward(self, img):
        # img shape: (..., num_channels, w, h)
        img_shape = img.shape
        img = torch.flatten(img, start_dim=-2) # (..., num_channels, w*h)
        img_out = self.fc_layer(img)
        img_out = img_out.reshape(img_shape)
        return img_out
