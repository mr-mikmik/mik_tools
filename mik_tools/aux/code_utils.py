import numpy as np
import torch
from typing import Tuple, Union, List, Optional, Callable
import threading
import time


def einsum(einum_keys:str, x1:Union[np.ndarray, torch.Tensor], *args) -> Union[np.ndarray, torch.Tensor]:
    if isinstance(x1, torch.Tensor):
        return torch.einsum(einum_keys, x1, *args)
    elif isinstance(x1, np.ndarray):
        return np.einsum(einum_keys, x1, *args)
    else:
        raise NotImplementedError


class Rate(object):
    """
    Rate controller for maintaining a fixed loop rate using ROS time.
    """

    def __init__(self, hz: float, sleep_fn:Optional[Callable]=None):
        """
        Initialize the Rate object.

        Args:
            hz (float): Frequency in Hz for sleeping.
            reset (bool): If True, resets the timer when ROS time moves backwards. Defaults to False.
        """
        self.last_time = self._get_time()
        self._hz = None
        self.period = None
        self.hz = hz
        self.sleep_fn = sleep_fn if sleep_fn is not None else time.sleep

    @property
    def hz(self) -> float:
        """
        Get the current frequency in Hz.

        Returns:
            float: Current frequency in Hz.
        """
        return self._hz
    
    @hz.setter
    def hz(self, hz: float) -> None:
        self._hz = hz
        self.period = 1.0 / hz if hz > 0 else float('inf')

    def _get_time(self) -> float:
        """
        Return the current ROS time.

        Returns:
            float: Current ROS time.
        """
        return time.time()

    def _remaining(self, curr_time: float) -> float:
        """
        Compute the remaining time to sleep based on current time.

        Args:
            curr_time (float): Current ROS time.

        Returns:
            rospy.rostime.Duration: Duration to sleep.
        """
        if self.last_time > curr_time:
            # we have a time jump backwards, reset the timer
            self.last_time = curr_time

        elapsed = curr_time - self.last_time
        remaining_sleep_time = self.period - elapsed
        return remaining_sleep_time

    def remaining(self) -> float:
        """
        Get the remaining sleep time.

        Returns:
            rospy.rostime.Duration: Duration to sleep.
        """
        curr_time = self._get_time()
        return self._remaining(curr_time)
    
    def _sleep(self, duration: float) -> None:
        """
        Sleep for the specified duration.

        Args:
            duration (float): Duration to sleep in seconds.
        """
        if duration > 0:
            self.sleep_fn(duration)

    def sleep(self):
        """
        Sleep for the remaining time to maintain the loop rate.
        """
        curr_time = self._get_time()
        remaining_time = self._remaining(curr_time)
        self._sleep(remaining_time)
        self.last_time = self.last_time + self.period
        # detect time jumping forwards, as well as loops that are inherently too slow
        if curr_time - self.last_time > self.period * 2:
            self.last_time = curr_time
