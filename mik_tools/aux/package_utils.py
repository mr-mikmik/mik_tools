import os
import yaml

package_name = 'mik_tools'
PACKAGE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)).split(f'/{package_name}/')[0], package_name)
project_path = PACKAGE_PATH
MESHES_PATH = os.path.join(PACKAGE_PATH, 'meshes')
CONFIG_PATH = os.path.join(PACKAGE_PATH, 'config')
MODELS_PATH = os.path.join(PACKAGE_PATH, 'models')


def get_mesh_dir_path(collision=False, visual=False, test=False):
    if collision:
        mesh_dir_path = os.path.join(MESHES_PATH, 'collision')
    elif visual:
        mesh_dir_path = os.path.join(MESHES_PATH, 'visual')
    elif test:
        mesh_dir_path = os.path.join(MESHES_PATH, 'test_meshes')
    else:
        mesh_dir_path = MESHES_PATH
    return mesh_dir_path


def get_dataset_path(dataset_name=None):
    dataset_default_path = '~/Datasets'
    dataset_path = os.path.expanduser(dataset_default_path)
    if dataset_name is not None:
        dataset_path = os.path.join(dataset_path, dataset_name)
    return dataset_path


def get_logging_path(log_name=None):
    log_default_path = '~/lightning_logs'
    log_path = os.path.expanduser(log_default_path)
    if log_name is not None:
        log_path = os.path.join(log_path, log_name)
    return log_path


def get_mesh_path(mesh_name):
    mesh_dir_path = MESHES_PATH
    mesh_path = os.path.join(mesh_dir_path, mesh_name)
    return mesh_path


def get_test_mesh_path(mesh_name):
    mesh_dir_path = get_mesh_dir_path(test=True)
    mesh_path = os.path.join(mesh_dir_path, mesh_name)
    return mesh_path