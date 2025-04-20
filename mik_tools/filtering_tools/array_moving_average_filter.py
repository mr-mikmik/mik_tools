import numpy as np


class ArrayMovingAverageFilter(object):
    """
    This is a buffer that stores the last N depth images, and applies a filter to smooth the depth images

    """
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.buffer = None # (W, ...) where W is the number of images in the buffer, and ... are the dimensions of the  image
        self.tail = 0 # this keeps track of the location where the next image will be stored

    def __len__(self):
        if self.buffer is None:
            return 0
        else:
            return len(self.buffer)

    @property
    def head(self):
        head_indx = (self.tail-1) % self.buffer_size
        return head_indx

    def reset(self):
        self.buffer = None
        self.tail = 0

    def add_array(self, ar):
        """
        :param ar: numpy array of shape (...)
        :return: None
        """
        if self.buffer is None or len(self.buffer) == 0 or self.buffer_size == 1:
            # initialize the buffer and add a dimension to it
            self.buffer = np.expand_dims(ar, axis=0)
        elif len(self.buffer) < self.buffer_size:
            self.buffer = np.concatenate([self.buffer, np.expand_dims(ar, axis=0)], axis=0)
        else:
            # the buffer is full, we just need to replace the tail
            self.buffer[self.tail] = ar
        self.move_tail()

    def get_average(self):
        avg_ar = np.mean(self.buffer, axis=0)
        return avg_ar

    def move_tail(self):
        self.tail = (self.tail + 1) % self.buffer_size

    def get_head(self):
        return self.buffer[self.head]

    def get_tail(self):
        return self.buffer[self.tail]


def test_filter():
    # TEST THE FILTER
    avg_filter = ArrayMovingAverageFilter(5)
    for i in range(8):
        ar_i = i * np.ones((2, 3, 4))
        avg_filter.add_array(ar_i)
        print('\n\nIteration: {}'.format(i))
        print('Filter length:', len(avg_filter))
        print(avg_filter.get_average())
        print(avg_filter.buffer[:, 0, 0, 0])


if __name__ == '__main__':
    test_filter()