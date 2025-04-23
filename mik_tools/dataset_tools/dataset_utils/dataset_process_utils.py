import numpy as np
import os
from numpy import *

from mik_tools.aux.package_utils import get_dataset_path


def get_data_path(data_name):
    data_path = None
    real_data_name = None
    if data_name.startswith("/"):
        # we provide the absolute path
        data_path = data_name
        real_data_name = data_path.split('/')[-1]
    elif data_name.startswith('~'):
        # we provide the absolute path
        data_path = os.path.expanduser(data_name)
        real_data_name = data_path.split('/')[-1]
    elif '/' in data_name:
        # we provide the relative path
        data_path = os.path.join(os.getcwd(), data_name)
        real_data_name = data_path.split('/')[-1]
    else:
        # we just provide the data name
        # project_path = self._get_project_path()
        # data_path = os.path.join(project_path, 'data', data_name)
        data_path = get_dataset_path(data_name)
        real_data_name = data_name
    return data_path, real_data_name


def process_list_array(list_array_raw):
    if type(list_array_raw) is str:
        _list_array_raw = list_array_raw[1:-1] # remove list "'" and "'"
        list_elem = _list_array_raw.split(' ')
        list_out = [float(x) for x in list_elem if x!= '']
    elif list_array_raw is None:
        list_out = []
    elif np.isnan(list_array_raw):
        list_out = []
    else:
        raise ValueError(f'Unsupported type {type(list_array_raw)}')
    return np.array(list_out)


def process_str_list(str_list, separator=', '):
    if type(str_list) is str:
        _str_list = str_list[1:-1]  # remove "[" and "]"
        list_elem = _str_list.split(separator)
        list_out = [float(x) for x in list_elem if x != '']
    elif str_list is None:
        list_out = []
    elif np.isnan(str_list):
        list_out = []
    else:
        raise ValueError(f'Unsupported type {type(str_list)}')
    return np.array(list_out)


def process_str_list_of_str(str_list):
    _str_list = str_list[1:-1]  # remove "[" and "]"
    list_elem = _str_list.split(', ')
    list_out = [x[1:-1] for x in list_elem if x != '']  # remove the '' around each string
    return list_out


def process_dict_array(dict_array_raw):
    if type(dict_array_raw) is str:
        dict_out = eval(dict_array_raw)
    else:
        raise NotImplementedError('Not implemented yet for type {}'.format(type(dict_array_raw)))
    return dict_out
