
WRENCH_COLUMNS = ['wrench.force.x', 'wrench.force.y', 'wrench.force.z', 'wrench.torque.x',
                               'wrench.torque.y', 'wrench.torque.z']


# Wrench data processing ------------------------------------------
def process_raw_wrench_data(wrench_df, frame_id=None, wrench_columns=None):
    if wrench_columns is None:
        wrench_columns = WRENCH_COLUMNS
    frame_ids = wrench_df['header.frame_id'].values
    if frame_id is None:
        # load all the data
        wrench = wrench_df[wrench_columns].values
    elif frame_id in frame_ids:
        # return only the wrench for the given frame id
        wrench = wrench_df[wrench_df['header.frame_id'] == frame_id][wrench_columns].values
    else:
        # frame not found
        print('No frame named {} found. Available frames: {}'.format(frame_id, frame_ids))
        wrench = None
    return wrench