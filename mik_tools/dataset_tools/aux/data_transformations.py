import torch


class TensorTypeTr(object):

    def __init__(self, dtype=None):
        self.dtype = dtype

    def __call__(self, sample):
        sample = sample.copy()
        if self.dtype is not None:
            for k, v in sample.items():
                try:
                    sample[k] = torch.tensor(v, dtype=self.dtype)
                except TypeError as e:
                    sample[k] = v
                except ValueError as e:
                    sample[k] = v
        return sample

