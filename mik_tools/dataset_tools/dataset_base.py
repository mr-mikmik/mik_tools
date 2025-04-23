import numpy as np
import os
import pandas as pd
from torch.utils.data import Dataset
import abc
import matplotlib.pyplot as plt
import torch
import shutil
from tqdm import tqdm
import copy

from mik_tools.dataset_tools.aux.data_transformations import TensorTypeTr
from mik_tools.aux.package_utils import PACKAGE_PATH, get_dataset_path
import mik_tools.dataset_tools.dataset_utils.dataset_process_utils as dataset_process_utils


class DatasetBase(Dataset, abc.ABC):

    def __init__(self, data_name, transformation=None, dtype=None, load_cache=True, contribute_mode=False, device='cpu', clean_if_error=True, process=True):
        """
        :param data_name: name or path to where the data can be found.
        :param transformation: transformaiton or list of transformations to apply to the data
        :param dtype: Type of the processed data. If None, the default type is preserved
        :param load_cache: <bool> If false, we will not use the saved processed data and instead it will reprocess it on the fly. If True and no process data is found, it processes the data and then uses it.
        :param contribute_mode: <bool> if True, we will only process a sample if it is not found. If a sample is found then it will be loaded.
        :param clean_if_error: <bool> if True, we will delete all the processed data if an error occurs when it is being processed.
        """
        self.dtype = dtype
        self.device = device
        self.transformation = transformation
        self.load_cache = load_cache
        self.contribute_mode = contribute_mode
        self.clean_if_error = clean_if_error
        self.data_path, self.data_name = self._get_data_path(data_name)
        self.processed_data_path = self._get_processed_data_path()
        self.sample_codes = self._get_sample_codes() # contains the codes for each sample in the dataset. It is used to know the length of the dataset
        self.item_indxs = self._get_item_indxs() # this is used for slicing purposes
        self.tensor_type_tr = TensorTypeTr(dtype=self.dtype)
        super().__init__()
        self.means, self.stds = None, None
        self.maxs, self.mins = None, None
        self._pre_process_hook()
        if process:
            self.process()

    # ========================================================================================================
    #       IMPLEMENT/OVERWRITE FUNCTIONS:
    # ---------------------------

    @abc.abstractmethod
    def _get_sample_codes(self):
        """
        Return a list containing the data filecodes.
        Overwrite the function in case the data needs to be filtered.
        By default we load all the filecodes we find in the datalegend
        :return:
        """
        pass

    @abc.abstractmethod
    def _get_sample(self, sample_code):
        """
        Retruns the sample corresponding to the filecode fc
        :param sample_code:
        :return:
        """
        pass

    @abc.abstractmethod
    def get_name(self):
        pass

    @property
    def name(self):
        """
        Returns an unique identifier of the dataset
        :return:
        """
        return self.get_name()


    # ========================================================================================================
    #       BASIC FUNCTIONS:
    # ---------------------------

    def __len__(self):
        dataset_len = len(self.item_indxs)
        return dataset_len

    def __getitem__(self, item):
        if isinstance(item, slice):
            # item is a slice object containting (start, stop, step)
            sliced_dataset = self._slice_dataset(item)
            return sliced_dataset
        elif isinstance(item, list) or isinstance(item, np.ndarray):
            # item is a list or array of indexes
            sliced_dataset = self._subsample_dataset(item)
            return sliced_dataset
        else:
            item_i = self._get_data_item(item)
        # else:
        #     raise NotImplementedError(f'__getitem__ called with unsupported item {item} of type {type(item)}')
        return item_i

    def __iter__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)

    def _get_data_item(self, item):
        item_indx = self.item_indxs[item]
        save_path_i = self._get_sample_path(item_indx)
        item_i = None
        # the following loop is used to avoid the error: "UnpicklingError: A load persistent id instruction was encountered,
        # but no persistent_load function was specified. " We try to load the data multiple times until it works.
        for i in range(1000):
            try:
                item_i = torch.load(save_path_i, map_location=self.device)
            except:
                continue
            if item_i is not None:
                break
        if item_i is None:
            raise RuntimeError(f'We were not able to load {save_path_i}')
        return item_i

    def _slice_dataset(self, item):
        """
        Returns a new dataset where each sample is a sliding window of the original dataset
        :param window_size:
        :param step_size:
        :return:
        """
        sliced_dataset = self.copy()  # copy the dataset
        sliced_dataset.item_indxs = self.item_indxs[item]
        if self.sample_codes is None:
            sliced_dataset.sample_codes = self.sample_codes
        else:
            sliced_dataset.sample_codes = self.sample_codes[item]
        # sliced_dataset.
        return sliced_dataset

    def _subsample_dataset(self, item):
        """
        Returns a new dataset where each sample is a sliding window of the original dataset
        :param item: <list or numpy array> list of indexes to subsample
        :return:
        """
        subsampled_dataset = self.copy()
        # subsample the sample_codes and item_indxs
        subsampled_dataset.item_indxs = [self.item_indxs[i] for i in item]
        if self.sample_codes is None:
            subsampled_dataset.sample_codes = None
        else:
            subsampled_dataset.sample_codes = [self.sample_codes[i] for i in item]
        return subsampled_dataset

    def copy(self):
        copied_dataset = copy.deepcopy(self)
        return copied_dataset

    def _pre_process_hook(self):
        pass

    def _process_hook(self):
        pass

    def process(self):
        if not os.path.exists(self.processed_data_path) or not self.load_cache or self.contribute_mode:
            print('Processing the data and saving it to {}'.format(self.processed_data_path))
            os.makedirs(self.processed_data_path, exist_ok=True)
            # process the data and save it as .pt files
            self._process_hook()
            pbar = tqdm(range(self.__len__()), desc='Processing data')
            for i in pbar:
                requires_processing_i = True
                expected_sample_path_i = self._get_sample_path(i)
                if self.contribute_mode and os.path.exists(expected_sample_path_i):
                    # in contribute_mode, we will only process a sample if it does not exists.
                    requires_processing_i = False

                if requires_processing_i:
                    try:
                        self._process_sample_i(i)
                    except:
                        if self.clean_if_error:
                            print('Removing the directory since the processing was stopped due to an error.')
                            # remove the data directory since some error has occurred
                            shutil.rmtree(self.processed_data_path)
                        raise
            print('Data processed')
            
    def _process_sample_i(self, indx):
        sample_code_i = self.sample_codes[indx]
        sample_i = self._get_processed_sample(sample_code_i)
        sample_i = self._tr_sample(sample_i)
        self._save_sample(sample_i, indx)

    def _save_sample(self, sample, indx):
        save_path_i = self._get_sample_path(indx)
        torch.save(sample, save_path_i)

    def _get_processed_sample(self, sample_code):
        # each data sample is composed by:
        #   'mask': object contact mask
        #   'y': resulting delta motion

        sample = self._get_sample(sample_code)
        if self.transformation is not None:
            if type(self.transformation) is list:
                for tr in self.transformation:
                    sample = tr(sample)
            else:
                sample = self.transformation(sample)
        return sample

    def _get_sample_path(self, indx):
        sample_save_path = os.path.join(self.processed_data_path, 'data_{}.pt'.format(indx))
        return sample_save_path

    def _get_item_indxs(self):
        if not os.path.exists(self.processed_data_path) or not self.load_cache or self.contribute_mode:
            item_indxs = np.arange(len(self.sample_codes))
        else:
            # load the indexes from the processed data folder
            item_indxs = self._get_processed_sample_ids()
        return item_indxs

    def _get_processed_sample_ids(self):
        # list all files in self.processed_data_path
        processed_sample_ids = []
        for file in os.listdir(self.processed_data_path):
            if file.endswith(".pt"):
                processed_sample_ids.append(int(file.split('_')[-1].split('.')[0])) # data files are named data_0.pt, data_1.pt, etc
        # sort the list in order
        processed_sample_ids.sort()
        return processed_sample_ids

    def _tr_sample(self, sample_i):
        sample_i = self.tensor_type_tr(sample_i)
        return sample_i

    def _get_processed_data_path(self):
        return os.path.join(self.data_path, 'processed_data', self.name)

    def _get_project_path(self):
        """
        Returns the path to the main project directory. Used for finding the default data path
        :return:
        """
        return PACKAGE_PATH

    def get_sizes(self):
        sample_test = self.__getitem__(0)
        sizes = {}
        for item, value in sample_test.items():
            if not hasattr(value, 'shape') or len(value.shape) == 0:
                sizes[item] = int(1)
            else:
                size_i = value.shape
                # sizes[item] = value.shape[-1]  # Old way, only last value
                if len(size_i) == 1:
                    sizes[item] = value.shape[-1]
                else:
                    sizes[item] = np.asarray(value.shape)
        return sizes

    def invert(self, sample):
        # apply the inverse transformations
        if self.transformation is not None:
            if type(self.transformation) is list:
                for tr in reversed(self.transformation):
                    sample = tr.inverse(sample)
            else:
                sample = self.transformation.inverse(sample)
        return sample

    # ========================================================================================================
    #       AUXILIARY FUNCTIONS:
    # ---------------------------

    def _get_data_path(self, data_name):
        data_path, real_data_name = dataset_process_utils.get_data_path(data_name)
        return data_path, real_data_name

    def _process_list_array(self, list_array_raw):
        list_out = dataset_process_utils.process_list_array(list_array_raw)
        return list_out

    def _process_str_list(self, str_list, separator=', '):
        list_out = dataset_process_utils.process_str_list(str_list, separator=separator)
        return list_out

    def _process_str_list_of_str(self, str_list):
        list_out = dataset_process_utils.process_str_list_of_str(str_list)
        return list_out

    def _process_dict_array(self, dict_array_raw):
        dict_out = dataset_process_utils.process_dict_array(dict_array_raw)
        return dict_out

    def _pack_all_samples(self):
        sample_test = self.__getitem__(0)
        samples = {}
        for key in sample_test.keys():
            samples[key] = []
        for i in range(self.__len__()):
            sample_i = self.__getitem__(i)
            for key in samples.keys():
                samples[key].append(sample_i[key].detach().cpu().numpy())
        for key, value in samples.items():
            samples[key] = np.stack(value, axis=0)
        return samples

    # ========================================================================================================
    #       NORMALIZATION AND STANDARDIZATION FUNCTIONS:
    # ---------------------------------------------------

    def _get_storage_path(self):
        # Override this function in case we want to modify the path
        return os.path.join(self.data_path, 'stand_const', self.name)

    def save_standardization_constants(self):
        save_path = self._get_storage_path()
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        np.save(os.path.join(save_path, 'means.npy'), self.means)
        np.save(os.path.join(save_path, 'stds.npy'), self.stds)
        np.save(os.path.join(save_path, 'maxs.npy'), self.maxs)
        np.save(os.path.join(save_path, 'mins.npy'), self.mins)

    def load_standardization_constants(self):
        load_path = self._get_storage_path()
        if not os.path.isfile(os.path.join(load_path, 'means.npy')):
            # there are no normalization constants saved, so compute them
            self._compute_standardization_constants()
        self.means = np.load(os.path.join(load_path, 'means.npy')).item()
        self.stds = np.load(os.path.join(load_path, 'stds.npy')).item()
        self.maxs = np.load(os.path.join(load_path, 'maxs.npy')).item()
        self.mins = np.load(os.path.join(load_path, 'mins.npy')).item()
        print('Normalization constants loaded.')
        return self.means, self.stds

    def _compute_standardization_constants(self):
        sample_test = self.__getitem__(0)
        means = {}
        stds = {}
        maxs = {}
        mins = {}
        print('Computing the dataset standardization constants, please wait')

        samples = self._pack_all_samples()
        for key in samples.keys():
            samples_i = samples[key]
            mean_i = np.mean(samples_i, axis=0)
            std_i = np.std(samples_i, axis=0)
            max_i = np.max(samples_i, axis=0)
            min_i = np.min(samples_i, axis=0)
            # Remove noise from std. If it is less than 5 orders of magnitude than the mean, set it to 0
            _mean_std_orders_magnitude = np.abs(np.divide(mean_i, std_i, out=np.zeros_like(mean_i), where=std_i != 0))
            mean_std_orders_magnitude = np.log10(_mean_std_orders_magnitude,
                                                 out=np.zeros_like(_mean_std_orders_magnitude),
                                                 where=_mean_std_orders_magnitude != 0)
            std_i[np.where(mean_std_orders_magnitude > 5)] = 0

            means[key] = mean_i
            stds[key] = std_i
            maxs[key] = max_i
            mins[key] = min_i
        print('\t -- Done!')
        self.means = means
        self.stds = stds
        self.maxs = maxs
        self.mins = mins
        self.save_standardization_constants()
        return means, stds

    # ========================================================================================================
    #       DATA DEBUGGING FUNCTIONS:
    # ----------------------------------

    def get_histograms(self, num_bins=100):
        histogram_path = os.path.join(self.data_path, 'data_stats', 'histograms', self.name)
        print('\n -- Getting Histograms --')
        print('Packing all samples, please wait...')
        samples = self._pack_all_samples()
        print('Computing histograms')
        for key, sample_i in samples.items():
            hist_key_path = os.path.join(histogram_path, key)
            if not os.path.isdir(hist_key_path):
                os.makedirs(hist_key_path)
            num_features = sample_i.shape[-1]
            for feat_indx in range(num_features):
                data_i = sample_i[:, feat_indx]
                fig = plt.figure()
                plt.hist(data_i, color='blue', edgecolor='black',
                     bins=num_bins)
                plt.title('Dataset {} - {} feature {}'.format(self.name, key, feat_indx))
                plt.xlabel('{} feature {}'.format(key, feat_indx))
                plt.ylabel('Counts')
                plt.savefig(os.path.join(hist_key_path, '{}_feature_{}_hist.png'.format(key, feat_indx)))
                plt.close()
            # Save also the statistics:
            mean_i = np.mean(sample_i, axis=0)
            std_i = np.std(sample_i, axis=0)
            max_i = np.max(sample_i, axis=0)
            min_i = np.min(sample_i, axis=0)
            q_1 = np.quantile(sample_i, q=0.25, axis=0)
            q_2 = np.quantile(sample_i, q=0.5,  axis=0)
            q_3 = np.quantile(sample_i, q=0.75, axis=0)

            _mean_std_orders_magnitude = np.abs(np.divide(mean_i, std_i, out=np.zeros_like(mean_i), where=std_i != 0))
            mean_std_orders_magnitude = np.log10(_mean_std_orders_magnitude,
                                                 out=np.zeros_like(_mean_std_orders_magnitude),
                                                 where=_mean_std_orders_magnitude != 0)
            std_i[np.where(mean_std_orders_magnitude > 5)] = 0

            feature_indxs = np.arange(num_features,dtype=np.int)
            stats_data = np.stack([feature_indxs, mean_i, std_i, max_i, min_i, q_1, q_2, q_3]).T

            df = pd.DataFrame(data=stats_data, columns=['Feat Indx', 'Mean', 'Std', 'Max', 'Min', 'Q1 (25%)', 'Q2 (50%)', 'Q3 (75%)'])
            df.to_csv(os.path.join(hist_key_path, '{}_stats.csv'.format(key)), index=False)
        print('Histograms saved!')

    def get_scatterplots(self, num_samples=250):
        scatterplot_path = os.path.join(self.data_path, 'data_stats', 'scatter_plots', self.name)
        print('\n -- Getting Scatterplots --')
        print('Packing all samples, please wait...')
        samples = self._pack_all_samples()
        print('Computing scatterplots')
        for key_i, sample_i in samples.items():
            hist_key_path = os.path.join(scatterplot_path, key_i)
            if not os.path.isdir(hist_key_path):
                os.makedirs(hist_key_path)
            num_features_i = sample_i.shape[-1]
            for feat_indx_i in range(num_features_i):
                data_i = sample_i[:num_samples, feat_indx_i]
                for key_j, sample_j in samples.items():
                    num_features_j = sample_j.shape[-1]
                    for feat_indx_j in range(num_features_j):
                        data_j = sample_j[:num_samples, feat_indx_j]
                        fig = plt.figure()
                        plt.scatter(data_i, data_j, marker='.', color='blue')
                        plt.title('Dataset {} - {} feature {} vs {} feature {}'.format(self.name, key_i, feat_indx_i, key_j, feat_indx_j))
                        plt.xlabel('{} feature {}'.format(key_i, feat_indx_i))
                        plt.ylabel('{} feature {}'.format(key_j, feat_indx_j))
                        plt.savefig(os.path.join(hist_key_path, '{}_feature_{}_vs_{}_feature_{}_scatter.png'.format(key_i, feat_indx_i, key_j, feat_indx_j)))
                        plt.close()

        print('Scatterplots saved!')

    def detect_outliers(self):
        """
        Finds values that deviate to much from the norm
        :return:
        """
        # We may need to change the name of the this function
        outliers = {} # this will hold all the file codes that result outside the range
        sample_test = self.__getitem__(0)
        keys = sample_test.keys()
        if self.means is None:
            self._compute_standardization_constants()
        upp_vals, low_vals = {}, {}
        for key in keys:
            upp_vals[key] = self.means[key] + 4 * self.stds[key]
            low_vals[key] = self.means[key] - 4 * self.stds[key]

        for i in range(self.__len__()):
            sample_i = self.__getitem__(i)
            fc = self.sample_codes[i]
            for key in keys:
                out_range = np.logical_or(sample_i[key] >= upp_vals[key], sample_i[key]<=low_vals[key])
                if out_range.any():
                    # Mark it as outlier
                    if fc in outliers:
                        outliers[fc].append((key, list(np.where(out_range)[0])))
                    else:
                        outliers[fc] = [(key, list(np.where(out_range)[0]))]
        return outliers

