import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from stable_baselines3.sac import SAC

from src.hppo.hppo import HybridPPO
from src.hppo.hybrid_policy import HybridActorCriticPolicy

from src.gym_env.gamble_env.gamble_env import ZeroControl

from gymnasium import spaces
from stable_baselines3.common.evaluation import evaluate_policy

import numpy as np

# 正确反应环境状态的奖励非常重要

if __name__ == "__main__":

    env = ZeroControl(use_approach_reward = False)
    model = HybridPPO(
        HybridActorCriticPolicy, env, verbose = 1, policy_kwargs = {
            "net_arch": dict(pi=[256, 256], vf=[256, 256])
        }
    )
    # env = ZeroControl(is_param_action = False, use_approach_reward = False)
    # model = SAC("MlpPolicy", env, verbose = 1)

    # p = HybridActorCriticPolicy(
    #     spaces.Box(-1, 1, (10,)),
    #     spaces.Tuple((
    #         spaces.Discrete(2),
    #         spaces.Box(-1, 1, (5,))
    #     )),
    #     lambda t: 0.1
    # )
    # print(p.predict(np.random.random((3, 2))))
    
    model.learn(int(2e5), progress_bar = True)
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100)
    print(f"mean_reward: {mean_reward}, std_reward: {std_reward}")

    pass

    # em = ExpManager(
    #     load_exp_config(
    #         "app/example/conf/pendulum_finetuning.yaml",
    #         is_resolve = False
    #     ),
    #     sample_param,
    #     exp_name_suffix = None,
    #     opt_save_model = True
    # )
    # em.optimize(20)
