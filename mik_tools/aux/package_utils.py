import os
import yaml

package_name = 'mik_tools'
PACKAGE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)).split(f'/{package_name}/')[0], package_name)
MESHES_PATH = os.path.join(PACKAGE_PATH, 'meshes')
CONFIG_PATH = os.path.join(PACKAGE_PATH, 'config')


def get_mesh_dir_path(collision=False, visual=False):
    if collision:
        mesh_dir_path = os.path.join(MESHES_PATH, 'collision')
    elif visual:
        mesh_dir_path = os.path.join(MESHES_PATH, 'visual')
    else:
        mesh_dir_path = MESHES_PATH
    return mesh_dir_path


def get_dataset_path(dataset_name=None):
    dataset_default_path = '~/Datasets'
    dataset_path = os.path.expanduser(dataset_default_path)
    if dataset_name is not None:
        dataset_path = os.path.join(dataset_path, dataset_name)
    return dataset_path