import numpy as np


def project_to_box_space(space, value):
    value_projected = value
    is_lower = value < space.low
    is_higher = value > space.high
    if np.any(is_lower):
        value_projected[is_lower] = space.low[is_lower]
    elif np.all(is_higher):
        value_projected[is_higher] = space.high[is_higher]
    return value_projected


def project_to_discrete_space(space, value):
    n = space.n
    start = space.start
    discrete_values = (start + np.arange(n)).astype(np.int64).reshape(-1,1)
    distances = np.linalg.norm(discrete_values-value, axis=1)
    min_index = np.argmin(distances)
    return discrete_values.flatten()[min_index].item()
