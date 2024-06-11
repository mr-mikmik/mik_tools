import copy

from mik_tools.dataset_tools.data_collection.env_data_collector import EnvDataCollector


class ReferencedEnvDataCollector(EnvDataCollector):
    """
    Extends EnvDataCollector to record the reference state
    """
    def __init__(self, *args, **kwargs):
        self.reference_fc = None
        self.ref_obs = None
        self.ref_obs_timestamp = None
        super().__init__(*args, **kwargs)

    def _record_reference_state(self):
        self.ref_obs_timestamp = copy.deepcopy(self.env.ref_obs_timestamp)
        if self.env.ref_obs is not None:
            # self.ref_obs = self.env.ref_obs # This copies the reference observation -- BAD!
            self.ref_obs = self.env.ref_obs.copy()
        else:
            self.ref_obs = None
        self.reference_fc = self.get_new_filecode()
        if self.ref_obs is not None:
            try:
                self.env.ref_obs.modify_data_params(self.data_save_params)
                self.env.ref_obs.save_fc(self.reference_fc)
            except AttributeError as e:
                raise AttributeError('The reference observation does not have the method modify_data_params. '
                                     'Please, make sure that the observation is wrapped into a self-saving class. \n\n '
                                     'If you are using an environment, make sure that you have the option wrap_data=True')

    def _get_legend_column_names(self):
        legend_column_names = super()._get_legend_column_names()
        legend_column_names.insert(3, 'ReferenceStateFC')
        return legend_column_names

    def _get_legend_lines(self, data_params):
        legend_lines = super()._get_legend_lines(data_params)
        for i, line in enumerate(legend_lines):
            line.insert(3, self.reference_fc)
        return legend_lines

    def _collect_data_sample(self, params=None):
        # check the timestamp
        if self.ref_obs_timestamp is None or self.ref_obs_timestamp != self.env.ref_obs_timestamp:
            # record the reference
            self._record_reference_state()
        sample_params = super()._collect_data_sample(params=params)
        return sample_params

    def _collect_data_sample_simple(self, params=None):
        if self.env_needs_reset:
            self._reset_env()
            self.prev_obs = None
            self.env_needs_reset = False
        sample_params = super()._collect_data_sample_simple(params=params)
        info = sample_params['info']
        planning_success = True
        execution_success = True
        if 'planning_success' in info and 'execution_success' in info:
            planning_success = info['planning_success']
            execution_success = info['execution_success']
        if (sample_params['done'] or not sample_params[
            'valid'] or not planning_success or not execution_success) and (
                self.data_stats['collected'] + 1 < self.data_stats['to_collect']):
            self.env_needs_reset = True
        return sample_params
