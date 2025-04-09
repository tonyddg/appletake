from typing import Callable, Optional, Union

from torch import nn
import torch

import numpy as np
import gymnasium as gym

from stable_baselines3.common.utils import obs_as_tensor
from stable_baselines3.common import type_aliases

class ModelPredictor:
    def __init__(
            self,
            net: nn.Module,
            post_decorator: Optional[Callable[[np.ndarray], np.ndarray]] = None
        ) -> None:
        self.net = net
        self.net.train(False)

        self.device = next(net.parameters()).device
        self.post_decorator = post_decorator

    def predict(
        self,
        observation: Union[np.ndarray, dict[str, np.ndarray]],
        state: Optional[tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> tuple[np.ndarray, Optional[tuple[np.ndarray, ...]]]:
        """
        Get the policy action from an observation (and optional hidden state).
        Includes sugar-coating to handle different observations (e.g. normalizing images).

        :param observation: the input observation
        :param state: The last hidden states (can be None, used in recurrent policies)
        :param episode_start: The last masks (can be None, used in recurrent policies)
            this correspond to beginning of episodes,
            where the hidden states of the RNN must be reset.
        :param deterministic: Whether or not to return deterministic actions.
        :return: the model's action and the next hidden state
            (used in recurrent policies)
        """

        obs_th = obs_as_tensor(observation, self.device)
        with torch.no_grad():
            origin_act = self.net.forward(obs_th).detach().cpu().numpy()

        if self.post_decorator is not None:
            process_act = []
            for i in range(origin_act.shape[0]):
                process_act.append(self.post_decorator(origin_act[i]))
            act = np.array(process_act)
        else:
            act = origin_act
        return act, None

class PolicyDecorator:
    def __init__(
            self,
            policy: "type_aliases.PolicyPredictor",
            post_decorator: Callable[[np.ndarray], np.ndarray]
        ) -> None:
        self.policy = policy
        self.post_decorator = post_decorator

    def predict(
        self,
        observation: Union[np.ndarray, dict[str, np.ndarray]],
        state: Optional[tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> tuple[np.ndarray, Optional[tuple[np.ndarray, ...]]]:
        """
        Get the policy action from an observation (and optional hidden state).
        Includes sugar-coating to handle different observations (e.g. normalizing images).

        :param observation: the input observation
        :param state: The last hidden states (can be None, used in recurrent policies)
        :param episode_start: The last masks (can be None, used in recurrent policies)
            this correspond to beginning of episodes,
            where the hidden states of the RNN must be reset.
        :param deterministic: Whether or not to return deterministic actions.
        :return: the model's action and the next hidden state
            (used in recurrent policies)
        """

        origin_act, _ = self.policy.predict(
            observation,
            state,
            episode_start,
            deterministic
        )

        process_act = []
        for i in range(origin_act.shape[0]):
            process_act.append(self.post_decorator(origin_act[i]))
        act = np.array(process_act)

        return act, None

class RandomPolicy:
    def __init__(
            self,
            act_space: gym.spaces.Space,
            num_envs: int
        ) -> None:
        self.act_space = act_space
        self.num_envs = num_envs

    def predict(
        self,
        observation: Union[np.ndarray, dict[str, np.ndarray]],
        state: Optional[tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> tuple[np.ndarray, Optional[tuple[np.ndarray, ...]]]:
        """
        Get the policy action from an observation (and optional hidden state).
        Includes sugar-coating to handle different observations (e.g. normalizing images).

        :param observation: the input observation
        :param state: The last hidden states (can be None, used in recurrent policies)
        :param episode_start: The last masks (can be None, used in recurrent policies)
            this correspond to beginning of episodes,
            where the hidden states of the RNN must be reset.
        :param deterministic: Whether or not to return deterministic actions.
        :return: the model's action and the next hidden state
            (used in recurrent policies)
        """

        return np.stack([
            self.act_space.sample()
        for _ in range(self.num_envs)]), None
        
