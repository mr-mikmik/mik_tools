import os


def change_scene_name(data_path, old_scene_name, new_scene_name, idxs_to_update):
    # modify the data legend:
    # TODO: update the scene names on the data legend to update the scene names.
    # modify the files
    old_path = os.path.join(data_path, old_scene_name)
    for root, subdirs, files in os.walk(old_path):
        if not subdirs:
            # we are at the last directory
            new_path = root.replace(old_scene_name, new_scene_name)
            for file in files:
                indx = int(file.split('_')[-1].split('.')[0]) # usally files are 'prefix_000000.xxx'
                if indx in idxs_to_update:
                    old_file_path = os.path.join(root, file)
                    new_file_path = os.path.join(new_path, file)
                    os.rename(old_file_path, new_file_path) # move the file to the new scene directory


