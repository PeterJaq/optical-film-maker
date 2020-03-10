from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import tensorflow as tf
import numpy as np

from tf_agents.environments import py_environment
from tf_agents.environments import utils
from tf_agents.specs import array_spec
from tf_agents.environments import wrappers
#from tf_agents.environments import suite_gymbush
from agents.tf_agents.trajectories import time_step as ts

tf.compat.v1.enable_v2_behavior()

from common.DataLoader import MaterialLoader
from common.TransferMatrix import OpticalModeling
from common.Config import FilmConfig
from common.utils.FilmLoss import film_loss
from common.utils.FilmTarget import film_target, film_weight

class FilmEnvironment(py_environment.PyEnvironment):

    def __init__(self, 
                 config_path):
        super().__init__()

        self.fmConf = FilmConfig(config_path=config_path)

        self.opticalModel = OpticalModeling(Materials = self.fmConf.materials,
                                            WLstep    = self.fmConf.WLstep,
                                            WLrange   = self.fmConf.WLrange)

        self.target = film_target(self.fmConf.targets, 
                                  self.fmConf.WLstep,
                                  self.fmConf.WLrange) 
                                  
        self.weight = film_weight(self.fmConf.weights, 
                                  self.fmConf.WLstep,
                                  self.fmConf.WLrange)

        self.round_threshold = self.fmConf.round_threshold
        self.end_threshold   = self.fmConf.end_threshold

        self.init_state = self.fmConf.init_state

        self.action_list = [1, 0.1, 0.01, -1, -0.1, -0.01]
        self._state      = self.init_state

        len_state = len(self._state) - 2

        self.round           = 0
        self.round_threshold = 100

        self.pre_observation = 0
        self._action_len      = len(self.action_list) * len_state

        self._action_spec = array_spec.BoundedArraySpec(
                shape   = (),
                dtype   = np.int64,
                minimum = 0,
                maximum = 18,
                name    = 'action')
            
        self._observation_spec = array_spec.BoundedArraySpec(     
                shape   = (1,),
                dtype   = np.float32,
                minimum = 0,
                maximum = 1000,
                name    = 'observation')

    def _reset(self):
        self._state         = self.init_state
        self._episode_ended = False
        self.round          = 0

        self.opticalModel.RunSim(self._state)

        # 计算observation
        observation = self.opticalModel.simulation_result
        observation = film_loss(aim          = self.target, 
                                weight       = self.weight,
                                observation  = observation,
                                average      = True)

        return ts.restart(observation = np.array([observation], dtype=np.float32))

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def current_time_step(self):
        return self._current_time_step

    def _step(self, action):
        """Apply action and return new time_step."""

        if self._episode_ended:
            return self.reset()

        # 计算action
        action_num     = int(action % 6)
        action_layer   = int(action / 6)

        self._state[action_layer] += self.action_list[action_num]
        self.opticalModel.RunSim(self._state)

        # 计算observation
        observation = self.opticalModel.simulation_result
        observation = film_loss(aim          = self.target, 
                                weight       = self.weight,
                                observation  = observation,
                                average      = True)

        # 退出条件
        if observation <= self.end_threshold:
            return ts.termination(observation = np.array([observation], dtype=np.float32),
                        reward      = 1)
        elif self.round >= self.round_threshold:
            return ts.termination(observation = np.array([observation], dtype=np.float32),
                        reward      = -1)
        
        # 更新条件
        elif observation > self.pre_observation:
            self.pre_observation = observation
            self.round = 0
            return ts.transition(observation = np.array([observation], dtype=np.float32),
                                 reward      = 0.1,
                                 discount    = 1.0)
        else:
            self.round += 1
            return ts.transition(observation = np.array([observation], dtype=np.float32),
                                 reward      = -0.1,
                                 discount    = 1.0)
