import os
from mik_tools.dataset_tools.dataset_base import DatasetBase


class ProcessedDataset(DatasetBase):
    """
    This dataset loads data from an already processed dataset.
    """
    def __init__(self, *args, processed_dataset_name=None, **kwargs):
        """
        Initialize the processed dataset.
        :param args:
        :param processed_dataset_name: <str> Name of the processed dataset to load. If None, the first found will be used.
        :param kwargs:
        """
        self.processed_dataset_name = processed_dataset_name
        override_kwargs = {
            'contribute_mode': False, # We force to not contribute,
            'load_cache': True, # We force to load from cache,
        }
        kwargs.update(override_kwargs)
        super().__init__(*args, **kwargs)
        # Assert that the data path exists
        assert os.path.exists(self.processed_data_path) , f"Processed data path does not exist: {self.data_path}"

    def _get_sample_codes(self):
        return None

    def _get_sample(self, sample_code):
        return None

    def get_name(self):
        processed_dataset_name = self.processed_dataset_name
        processed_data_path = os.path.join(self.data_path, 'processed_data')
        found_processed_dataset_names = os.listdir(processed_data_path)
        # filter out files that start with .
        found_processed_dataset_names = [f for f in found_processed_dataset_names if not f.startswith('.')]
        if processed_dataset_name is None:
            # check the processed directory and get the first directory name
            processed_dataset_name = found_processed_dataset_names[0]
            print(f'No processed dataset name provided, using the first found: {processed_dataset_name} -- all found options: {found_processed_dataset_names}')
        assert processed_dataset_name in found_processed_dataset_names, f"Processed dataset name not found: {processed_dataset_name} -- available options: {found_processed_dataset_names}"
        return processed_dataset_name