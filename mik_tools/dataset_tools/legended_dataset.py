import pandas as pd
import os
import yaml

from mik_tools.dataset_tools.dataset_base import DatasetBase
import mik_tools.data_utils.loading_utils as load_utils


class LegendedDataset(DatasetBase):

    def __init__(self, data_name, **kwargs):
        self.data_path, self.data_name = self._get_data_path(data_name)
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f'We have not found the dataset named {self.data_name} with path {self.data_path}')
        self.dl = self._get_datalegend()
        super().__init__(data_name, **kwargs)

    # ========================================================================================================
    #       IMPLEMENT/OVERWRITE FUNCTIONS:
    # ---------------------------

    def _get_datalegend(self):
        datalegend = load_utils.load_datalegend(self.data_path)
        datalegend = self._filter_datalegend(datalegend)
        # reset the index from the filtering, so we can use .iloc to access the samples
        # NOTE: Ideally, we should not need to use this and use .loc instead. But for backward compatibility, we will reset the index.
        datalegend = datalegend.reset_index(drop=True)
        return datalegend

    def _filter_datalegend(self, datalegend):
        # Extend this method if you want to filter the datalegend
        return datalegend

    def _get_sample_codes(self):
        """
        Return a list containing the data filecodes.
        Overwrite the function in case the data needs to be filtered.
        By default we load all the filecodes we find in the datalegend
        :return:
        """
        return self.dl['FileCode'].to_numpy()

    def load_params(self):
        params_path = os.path.join(self.data_path, f'{self.data_name}_params.yaml')
        with open(params_path) as f:
            params = yaml.safe_load(f)
        return params

