# 获取 Actor-Critic 算法的价值判断
from stable_baselines3 import TD3, DDPG, SAC
from stable_baselines3.common.vec_env.vec_normalize import VecNormalize
from stable_baselines3.common import type_aliases

from typing import Union, Dict, Optional
import numpy as np
import torch as th

CriticAvailableModel = Union[SAC, TD3, DDPG]

def is_model_critic_available(model: "type_aliases.PolicyPredictor"):
    return isinstance(model, CriticAvailableModel)

def get_critic(
        model: CriticAvailableModel, 
        observation: Union[np.ndarray, Dict[str, np.ndarray]]
    ):
    '''
    参考 BasePolicy.predict, 用于 DDPG/TD3/SAC 等使用了 ContinuousCritic 的算法
    '''

    #开启测试模式
    model.policy.set_training_mode(False)
    # 检查观测是否合法
    if isinstance(observation, tuple) and len(observation) == 2 and isinstance(observation[1], dict): # type: ignore
        raise ValueError(
            "You have passed a tuple to the predict() function instead of a Numpy array or a Dict. "
            "You are probably mixing Gym API with SB3 VecEnv API: `obs, info = env.reset()` (Gym) "
            "vs `obs = vec_env.reset()` (SB3 VecEnv). "
            "See related issue https://github.com/DLR-RM/stable-baselines3/issues/1694 "
            "and documentation for more information: https://stable-baselines3.readthedocs.io/en/master/guide/vec_envs.html#vecenv-api-vs-gym-api"
        )
    obs_tensor, _ = model.policy.obs_to_tensor(observation)

    with th.no_grad():
        actions = model.policy._predict(obs_tensor, deterministic = True)
        critic = model.policy.critic.q1_forward(obs_tensor, actions) # type: ignore
    return critic.cpu().numpy()

    # # Convert to numpy, and reshape to the original action shape
    # actions = actions.cpu().numpy().reshape((-1, *model.policy.action_space.shape))  # type: ignore[misc, assignment]

    pass
