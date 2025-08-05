import numpy as np
from mik_tools import tr, pose_to_matrix


rpys = np.random.uniform(-np.pi, np.pi, (300, 3))

poss = np.random.rand(300, 3) * 0.01

poses = np.concatenate([poss, rpys], axis=1)

x_X_of = pose_to_matrix(poses)

quats = tr.quaternion_from_euler(rpys)

