import numpy as np
import transformations

from mik_tools.mik_tf_tools.tf_tools import average_poses
from collections import deque


class PoseAverager(object):
    def __init__(self, buffer_size=10):
        self.buffer_size = buffer_size
        self._poses = deque(maxlen=self.buffer_size)

    def add_pose(self, pose):
        self._poses.append(pose)

    def get_average_pose(self):
        poses = np.stack(self._poses,axis=0)
        return average_poses(poses)



# DEBUG:
if __name__ == '__main__':
    from mmint_tools import tr
    pose_averager = PoseAverager(buffer_size=10)
    regular_pose = np.array([0, 0, 0, 0, 0, 0, 1])
    for i in range(30):
        random_prob = np.random.random()
        if random_prob < 0.1:
            random_pos_i = np.random.uniform(-1, 1, size=3)
            random_axis = np.random.uniform(-1, 1, size=3)
            random_angle = np.random.uniform(-np.pi, np.pi)
            random_quat_i = tr.quaternion_about_axis(random_angle, random_axis)
            random_pose_i = np.concatenate([random_pos_i, random_quat_i])
            pose_i = random_pose_i
        else:
            pose_i = regular_pose
        pose_averager.add_pose(pose_i)
        print(f'Average pose {i+1}: {pose_averager.get_average_pose()}')

