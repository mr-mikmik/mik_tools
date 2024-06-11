import abc
import csv
import os
import sys
import time
import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

project_path = os.path.join(os.path.dirname(os.path.abspath(__file__)).split('/mik_tools')[0], 'mik_tools')
package_path = project_path
sys.path.append(project_path)


class DataCollectorBase(abc.ABC):

    def __init__(self, data_path=None):
        self.data_path, self.filename = self._get_data_path(data_path)
        self.datalegend_path = os.path.join(self.data_path, '{}_DataLegend.csv'.format(self.filename))
        self.dataparams_path = os.path.join(self.data_path, '{}_params.yaml'.format(self.filename))
        self.datacollection_params_yaml_path = os.path.join(self.data_path, 'datacollection_params.yaml')
        self.datacollection_params = self._init_datacollection_params()
        self.data_stats = {'collected': 0, 'to_collect': 0}
        self._load()

    def _init_datacollection_params(self):
        init_datacollection_params = {'filecode': 0}
        return init_datacollection_params

    @property
    def filecode(self):
        return self.datacollection_params['filecode']

    @filecode.setter
    def filecode(self, value):
        self.datacollection_params['filecode'] = value

    def _load(self):
        # Load or Create the data infrastructure if it does not exist
        if not os.path.isdir(self.data_path):
            os.makedirs(self.data_path)
        if not os.path.isfile(self.datalegend_path):
            legend_df = pd.DataFrame(columns=self._get_legend_column_names())
            legend_df.to_csv(self.datalegend_path, index=False)
            self._save_data_params()
            self._save_filecode_pickle()
        else:
            self._load_filecode_pickle()
            self._compare_current_params_with_saved()

    def _filter_params_to_save(self, params_dict):
        params_to_save = {}
        types_to_save = [str, int, float, type(None), tuple, list, bool]
        for k, v in params_dict.items():
            if type(v) is dict:
                params_to_save[k] = self._filter_params_to_save(v)
            elif type(v) == np.ndarray:
                params_to_save[k] = v.tolist()
            elif type(v) in [list, tuple]:
                listed_params = []
                for v_i in v:
                    filtered_dict_i = self._filter_params_to_save({'mik': v_i})
                    listed_params += list(filtered_dict_i.values())
                params_to_save[k] = listed_params
            elif type(v) in types_to_save:
                params_to_save[k] = v
            elif type(v) in [np.float64, np.float32, np.int64, np.int32, np.uint8, np.int8]:
                params_to_save[k] = v.item()
            else:
                pass
                # params_to_save[k] = str(v)

        return params_to_save

    def _save_data_params(self):
        # save the data parameters as yaml
        params_dict = self._get_params_to_save()
        # save
        params_to_save = self._filter_params_to_save(params_dict)
        with open(self.dataparams_path, 'w') as f:
            yaml.dump(params_to_save, f)

    def _load_data_params(self):
        with open(self.dataparams_path, 'r') as f:
            data_params = yaml.load(f, Loader=yaml.FullLoader)
        return data_params

    def _get_params_to_save(self):
        params = self.__dict__
        params_to_save = {}
        params_to_save.update(params)
        return params_to_save

    def _compare_current_params_with_saved(self):
        loaded_params = self._load_data_params()
        current_params = self._filter_params_to_save(self._get_params_to_save())
        except_params_keys = ['filecode']
        missmatched_params = self.__compare_params(current_params, loaded_params, except_params_keys=except_params_keys)
        if len(missmatched_params) > 0:
            print('\n\n ------ Found Mismatched Params ------ ')
            for k, vals in missmatched_params.items():
                current_v, loaded_v = vals
                print('Param mismatch {} --- || Current: {} | Loaded: {}'.format(k, current_v, loaded_v))
            print('\n')

    def __compare_params(self, param_a, param_b, key=None, except_params_keys=()):
        # param_a: typically current_params
        # param_b: typically loaded_params
        missmatched_params = {}
        if type(param_b) is dict and type(param_a) is dict:
            for k, v_a in param_a.items():
                if not k in param_b:
                    v_b = None # equivalent: missmatched_params[key] = (v_a, None) # case where b does not have a parameter that a has
                else:
                    v_b = param_b[k]
                if k not in except_params_keys:
                    if key is None:
                        new_k = k
                    else:
                        new_k = '{}_{}'.format(key, k)
                    missmatched_params_i = self.__compare_params(v_a, v_b, key=new_k, except_params_keys=except_params_keys)
                    missmatched_params.update(missmatched_params_i)
        else:
            if not param_a == param_b:
                if key is not None:
                    missmatched_params[key] = (param_a, param_b)
        return missmatched_params

    def _save_filecode_pickle(self):
        # TODO: Update the method name, since we do not longer pickle, yaml instead
        self._save_datacollection_params()

    def _load_filecode_pickle(self):
        # TODO: Update the method name, since we do not longer pickle, yaml instead
        self._load_datacollection_params()

    def _save_datacollection_params(self):
        with open(self.datacollection_params_yaml_path, 'w') as f:
            yaml.dump(self.datacollection_params, f)

    def _load_datacollection_params(self):
        with open(self.datacollection_params_yaml_path, 'r') as f:
            datacollection_params = yaml.load(f, Loader=yaml.FullLoader)
        self.datacollection_params = datacollection_params

    def get_new_filecode(self, update_pickle=False):
        self.filecode = self.filecode + 1
        if update_pickle:
            self._save_filecode_pickle()
        return self.filecode

    @abc.abstractmethod
    def _get_legend_column_names(self):
        """
        Return a list containing the column names of the datalegend
        Returns:
        """
        pass

    @abc.abstractmethod
    def _get_legend_lines(self, data_params):
        """
        Return a list containing the values to log inot the data legend for the data sample with file code filecode
        Args:
            data_params: <dict> containg parameters of the collected data
        Returns:
        """
        pass

    @abc.abstractmethod
    def _collect_data_sample(self, params=None):
        """
        Collect and save data to the designed path in self.data_path
        Args:
            params:
        Returns: <dict> containing the parameters of the collected sample
        """
        pass

    def _get_data_path(self, data_path=None):
        if data_path is None:
            # Get some default directory based on the current working directory
            data_path = os.path.join(package_path, 'saved_data')
        else:
            if data_path.startswith("/"):
                data_path = data_path  # we provide the full path (absolute)
            else:
                exec_path = os.getcwd()
                data_path = os.path.join(exec_path, data_path)  # we save the data on the path specified (relative)
        filename = data_path.split('/')[-1]
        return data_path, filename

    def collect_data(self, num_data):
        # Display basic information
        print('_____________________________')
        print(' Data collection has started!')
        print('  - The data will be saved at {}'.format(self.data_path))

        # Collect data
        pbar = tqdm(range(num_data), desc='Data Collected: ')
        num_data_collected = 1
        self.data_stats['to_collect'] = num_data
        self.data_stats['collected'] = 0
        for i in pbar:
            pbar.set_postfix({'Filecode': self.filecode})
            self.data_stats['collected'] = i
            # Save data
            sample_params = self._collect_data_sample()
            # Log data sample info to data legend
            legend_lines_vals = self._get_legend_lines(sample_params)
            num_data_collected = len(legend_lines_vals)
            # ensures full numpy array printed without ellipsis
            np.set_printoptions(threshold=sys.maxsize)
            with open(self.datalegend_path, 'a+') as csv_file:
                csv_file_writer = csv.writer(csv_file)
                for line_val in legend_lines_vals:
                    csv_file_writer.writerow(line_val)
            csv_file.close() # make sure it is closed
            # Update the filecode
            self._save_filecode_pickle()
            time.sleep(0.5)
            self.data_stats['collected'] += 1
