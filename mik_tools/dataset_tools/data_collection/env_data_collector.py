from collections import OrderedDict

import numpy as np
from gym import spaces
from tqdm import tqdm

from mik_tools.dataset_tools.data_collection.data_collector_base import DataCollectorBase
from mik_tools.env_tools.controllers import RandomController
from mik_tools.recording_utils.data_recording_wrappers import get_not_self_saved_keys


class EnvDataCollector(DataCollectorBase):

    def __init__(self, env, *args, scene_name='env_data_collection', manual_actions=False, reuse_prev_observation=False,
                 trajectory_length=1, reset_trajectories=False, controller=None, save_trajectory_on_the_fly=False, additional_info={}, **kwargs):
        """
        Class used to collect data from an environment. It performs steps and collects the observations and actions.
        :param env:
        :param args:
        :param scene_name:
        :param manual_actions:
        :param reuse_prev_observation: <bool>
            * if False: it will save a new observation as the s_t
            * if True: it will reuse the previous observation s_{t+1} as the initial observation
        :param kwargs:
        """
        self.env = env
        self.controller = self._get_controller(controller)
        self.scene_name = scene_name
        self.manual_actions = manual_actions
        self.reuse_prev_observation = reuse_prev_observation
        self.trajectory_length = trajectory_length
        self.reset_trajectories = reset_trajectories
        self.save_trajectory_on_the_fly = save_trajectory_on_the_fly
        self.additional_info = additional_info
        self.env_needs_reset = False
        self.prev_obs = None # stores the previous observation
        self.prev_obs_fc = None
        self.prev_info = None
        super().__init__(*args, **kwargs)
        self.data_save_params = {'save_path': self.data_path, 'scene_name': self.scene_name}

    @property
    def trajectory_indx(self):
        return self.datacollection_params['trajectory_indx']

    @trajectory_indx.setter
    def trajectory_indx(self, value):
        self.datacollection_params['trajectory_indx'] = value

    def get_new_trajectory_indx(self, update=False):
        self.trajectory_indx = self.trajectory_indx + 1
        if update:
            self._save_datacollection_params()
        return self.trajectory_indx

    def _init_datacollection_params(self):
        init_datacollection_params = super()._init_datacollection_params()
        init_datacollection_params['trajectory_indx'] = 0
        return init_datacollection_params

    def _get_controller(self, controller):
        if controller is None:
            controller = RandomController(env=self.env)
        return controller

    def _get_trajectory_length(self):
        return self.trajectory_length

    def _get_action_from_input(self):
        print('Please, provide the action values:')
        action_space = self.env.action_space
        action = OrderedDict()
        for a_i, as_i in action_space.spaces.items():  # Assume it is a Dict space from gym
            i = 0
            for i in range(20):
                # TODO: Extend for other types of envs!!!!
                if isinstance(as_i, spaces.Discrete):
                    raw_action_input = input('\tAction {} -- (Discrete between 0 and {},): '.format(a_i, as_i.n - 1))
                    try:
                        action_i = int(raw_action_input)
                    except ValueError:
                        print('\t Value incorrect! -- please, enter a valid value')
                    else:
                        if 0 <= action_i < as_i.n:
                            action[a_i] = action_i
                            break
                        else:
                            print('\t Value incorrect! -- please, enter a valid value')
                else:
                    raise NotImplementedError('We do not have that space implemented yet!')
            if i >= 19:
                raise ValueError('Input provided is not good')
        print('Done, Thanks')
        return action

    def _get_params_to_save(self):
        params_to_save = super()._get_params_to_save()
        # add the env parameters
        env_params = self.env.__dict__
        env_params['env_name'] = self.env.get_name()
        env_params['env_str'] = str(self.env)
        # params_to_save.update(env_params) # append the params along
        params_to_save['env'] = env_params # concatenate the parameters
        # add the controller paramseters
        controller_params = self.controller.get_controller_params()
        controller_params['controller_name'] = self.controller.name
        controller_params['controller_str'] = str(self.controller)
        params_to_save['controller'] = controller_params
        return params_to_save

    def _get_observation_keys_to_save(self, obs):
        keys_to_save = []
        not_self_saved_keys = get_not_self_saved_keys(obs)
        for key in not_self_saved_keys:
            obs_k = obs[key]
            if type(obs_k) in [np.ndarray]:
                if np.prod(obs_k.shape) < 20:
                    # we do not save arrays larger than 20 elements
                    keys_to_save.append(key)
            else:
                keys_to_save.append(key)
        return keys_to_save

    def _get_legend_column_names(self):
        # We record:
        action_space_names = ['action_{}'.format(k) for k in self.env.action_space.keys()]
        test_obs = self.env.get_observation()
        obs_keys_to_save = self._get_observation_keys_to_save(test_obs)
        init_obs_keys_to_save = ['{}_init'.format(k) for k in obs_keys_to_save]
        final_obs_keys_to_save = ['{}_final'.format(k) for k in obs_keys_to_save]
        column_names = ['Scene', 'TrajectoryIndex', 'TrajectoryStep', 'InitialStateFC', 'FinalStateFC', 'Controller', 'Reward', 'Done'] + action_space_names + ['Info'] + init_obs_keys_to_save + final_obs_keys_to_save + list(self.additional_info.keys())
        return column_names

    def _get_legend_lines(self, data_params):
        legend_lines = []
        num_samples = len(data_params)
        for i, data_params_i in enumerate(data_params):
            scene_i = self.scene_name
            trajectory_indx_i = data_params_i['trajectory_indx']
            trajectory_step_i = data_params_i['trajectory_step']
            init_fc_i = data_params_i['init_fc']
            final_fc_i = data_params_i['final_fc']
            reward_i = data_params_i['reward']
            done_i = data_params_i['done']
            info_i = data_params_i['info']
            action_values = list(data_params_i['action'].values())
            init_obs = data_params_i['init_obs']
            final_obs = data_params_i['obs']
            if init_obs is None:
                print('Init Obs is None')
                return []
            if final_obs is None:
                print('Final Obs is None')
                return [] # skip this data (not valid data)
            init_obs_keys_to_save = self._get_observation_keys_to_save(init_obs)
            final_obs_keys_to_save = self._get_observation_keys_to_save(final_obs)
            init_obs_to_save = [init_obs[k] for k in init_obs_keys_to_save] # we add the observation part that is not self saved to the datalegend
            final_obs_to_save = [final_obs[k] for k in final_obs_keys_to_save] # we add the observation part that is not self saved to the datalegend
            line_i = [scene_i, trajectory_indx_i, trajectory_step_i, init_fc_i, final_fc_i, self.controller.name, reward_i, done_i] + action_values + [info_i] + init_obs_to_save + final_obs_to_save + list(self.additional_info.values())
            legend_lines.append(line_i)
        return legend_lines

    def collect_data(self, num_data):
        self._reset_env() # Force to reset before starting a new data collection
        self.prev_obs = None
        # self.env.initialize()
        return super().collect_data(num_data)

    def _reset_env(self):
        out = self.controller.reset_env()
        return out

    def _collect_data_sample(self, params=None):
        if self.env_needs_reset:
            self._reset_env()
            self.prev_obs = None
            self.env_needs_reset = False
        trajectory_length_i = self._get_trajectory_length()
        sample_params = []
        trajectory_index = self.get_new_trajectory_indx()
        if trajectory_length_i > 1:
            iter_set = tqdm(range(trajectory_length_i), leave=False, desc='   Trajectory: ')
            iter_set.set_postfix({'TrajectoryIndex': trajectory_index})
        else:
            iter_set = range(trajectory_length_i)
        for sample_indx in iter_set:
            sample_params_i = self._collect_data_sample_simple(params=params)
            sample_params_i['trajectory_indx'] = trajectory_index
            sample_params_i['trajectory_step'] = sample_indx
            sample_params.append(sample_params_i)
            if sample_params_i['done']:
                break
            if self.save_trajectory_on_the_fly:
                self._log_data_on_the_fly(sample_params_i)

        if self.reset_trajectories and self.data_stats['collected'] < self.data_stats['to_collect'] - 1:
            self.env_needs_reset = True
        return sample_params

    def _log_data(self, sample_params):
        if not self.save_trajectory_on_the_fly:
            super()._log_data(sample_params)

    def _log_data_on_the_fly(self, sample_params):
        super()._log_data([sample_params])

    def _get_action(self, obs, info=None):
        action = self.controller.control(obs, info=info)
        valid = self.controller.is_action_valid(action)
        return action, valid

    def _collect_data_sample_simple(self, params=None):
        # Get initial observation
        if self.reuse_prev_observation and self.prev_obs is not None:
            init_fc = self.prev_obs_fc
            init_obs = self.prev_obs
            info = self.prev_info
        else:
            init_fc = self.get_new_filecode()
            init_obs = self.env.get_observation()
            try:
                init_obs.modify_data_params(self.data_save_params)
            except AttributeError as e:
                raise AttributeError('The observation init_obs does not have the method modify_data_params. '
                                     'Please, make sure that the observation is wrapped into a self-saving class. \n\n '
                                     'If you are using an environment, make sure that you have the option wrap_data=True')

            init_obs.save_fc(init_fc)
            info = None

        final_fc = self.get_new_filecode()

        obs_columns = list(init_obs.keys())
        # Get action
        if self.manual_actions:
            action = self._get_action_from_input()
            action = self.env._tr_action_space(action)
            valid = self.env.is_valid_action(action)
        else:
            action, valid = self._get_action(init_obs, info=info)
        if not valid:
            print('Action: {} -- NOT VALID!'.format(action))
            observation = init_obs # this is None to not save the trajectory
            reward = 0
            done = True
            info = {}
        else:
            # get one step sample:
            observation, reward, done, info = self.env.step(action)
            try:
                observation.modify_data_params(self.data_save_params)
                observation.save_fc(final_fc)
            except AttributeError as e:
                raise AttributeError('The observation init_obs does not have the method modify_data_params. '
                                     'Please, make sure that the observation is wrapped into a self-saving class. \n\n '
                                     'If you are using an environment, make sure that you have the option wrap_data=True')
            # try saving controller info
            try:
                self.controller.save_controller_info(init_fc, self.data_save_params)
            except AttributeError as e:
                import pdb; pdb.set_trace()
                raise AttributeError('The controller does not have the method save_controller_info. ')
            self.prev_obs = observation
            self.prev_obs_fc = final_fc
            self.prev_info = info
        sample_params = {
            'init_fc': init_fc,
            'final_fc': final_fc,
            'init_obs': init_obs,
            'obs': observation,
            'action': action,
            'reward': reward,
            'done': done,
            'info': info,
            'valid': valid,
            }
        return sample_params
