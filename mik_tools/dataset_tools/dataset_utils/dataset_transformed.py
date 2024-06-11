from torch.utils.data import Dataset
from collections.abc import Iterable


class DatasetTransformed(Dataset):
    def __init__(self, dataset, transforms=(), tr_inverse=False):
        self.dataset = dataset
        self.tr_inverse = tr_inverse
        if transforms is not None:
            if not isinstance(transforms, Iterable):
                self.transforms = [transforms]
            else:
                self.transforms = transforms

    def __getattr__(self, attr):
        self_attr_name = '_{}_'.format(self.__class__.__name__)
        # print('Getting the attribute for: {}'.format(self_attr_name))
        if self_attr_name in attr:
            attr_name = attr.split(self_attr_name)[1]
            return getattr(self, attr_name)
        elif attr in self.__dict__:
            return getattr(self, attr)
        return getattr(self.dataset, attr)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx] # untransformed sample (original)
        # apply tranformations:
        for tr_i in self.transforms:
            if not self.tr_inverse:
                sample = tr_i(sample)
            else:
                sample = tr_i.inverse(sample)
        return sample


def transform_dataset(dataset, transforms, tr_inverse=False):
    transformed_dataset = DatasetTransformed(dataset, transforms, tr_inverse=tr_inverse)
    return transformed_dataset

