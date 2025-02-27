from typing import Dict, SupportsFloat, Tuple
from gymnasium import spaces
from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvWrapper, VecEnvStepReturn, VecEnvObs

from torch import nn
import numpy as np
import torch
import albumentations as A

import gymnasium as gym

from stable_baselines3.common.vec_env.subproc_vec_env import _stack_obs
from stable_baselines3.common.utils import obs_as_tensor

from ..gym_env.aug import test_trans_obs_space

class ObservationWrapper(VecEnvWrapper):
    '''
    TODO: 不考虑元组与字典观测
    '''

    def observation_trnas(self, obs: VecEnvObs) -> VecEnvObs:
        '''
        传入经过堆叠的观测 obs
        '''
        raise NotImplementedError()

    def step_wait(self) -> VecEnvStepReturn:
        obs, reward, done, info = self.venv.step_wait()
        assert isinstance(obs, np.ndarray)

        for i in range(self.num_envs):
            if "terminal_observation" in info[i]:
                single_obs = _stack_obs([info[i]["terminal_observation"]], self.observation_space)
                trans_obs = self.observation_trnas(single_obs)

                if isinstance(self.observation_space, gym.spaces.Dict) and isinstance(trans_obs, Dict):
                    info[i]["terminal_observation"] = {key: trans_obs[key][0] for key in trans_obs.keys()}
                elif isinstance(self.observation_space, gym.spaces.Tuple) and isinstance(trans_obs, Tuple):
                    info[i]["terminal_observation"] = (trans_obs[i][0] for i in range(len(trans_obs)))
                elif isinstance(trans_obs, np.ndarray):
                    info[i]["terminal_observation"] = trans_obs[0]

        return self.observation_trnas(obs), reward, done, info

    def reset(self) -> VecEnvObs:
        obs = self.venv.reset()
        return self.observation_trnas(obs)

class NetFeatObsWrapper(ObservationWrapper):
    def __init__(
            self, 
            venv: VecEnv, 
            net: nn.Module,
            feat_dim: int,
        ):

        super().__init__(
            venv, 
            spaces.Box(-np.inf, np.inf, (feat_dim,), np.float32), 
            venv.action_space
        )
        self.net = net
        self.net.train(False)

        self.device = next(net.parameters()).device

    def observation_trnas(self, obs: VecEnvObs):
        '''
        不支持 Tuple 观测
        '''
        obs_th = obs_as_tensor(obs, self.device) # type: ignore
        feat = self.net.forward(obs_th).detach().cpu().numpy()

        return feat

class AugTransWrapper(ObservationWrapper):
    def __init__(
            self, 
            venv: VecEnv, 
            origin_observation_space: spaces.Box,
            trans: A.BasicTransform,
        ):

        self.trans = trans

        super().__init__(
            venv, 
            test_trans_obs_space(origin_observation_space, trans),
            venv.action_space
        )

    def observation_trnas(self, obs: VecEnvObs):
        '''
        仅支持图片观测
        '''
        assert isinstance(obs, np.ndarray), "仅支持图片观测"
        return np.stack([self.trans(image = single_obs)["image"] for single_obs in obs], 0)

    # def step_wait(self) -> VecEnvStepReturn:
    #     obs, reward, done, info = self.venv.step_wait()
    #     for i in range(len(info)):
    #         if "terminal_observation" in info[i]:
    #             # print(f"Aug: {info[i]['terminal_observation']}")
    #             single_obs = info[i]["terminal_observation"][np.newaxis, ...]
    #             feat = self.obs_trans(single_obs)[0]
    #             info[i]["terminal_observation"] = feat

    #     return self.obs_trans(obs), reward, done, info

    # def reset(self) -> VecEnvObs:
    #     return self.obs_trans(self.venv.reset())
    