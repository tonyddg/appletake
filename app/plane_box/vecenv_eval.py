import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np
from pathlib import Path

from src.pr.safe_pyrep import SafePyRep
from src.sb3.exp_manager import config_to_env, load_exp_config
from src.sb3.eval_record import eval_record_to_file
from src.sb3.model_predictor import ModelPredictor, RandomPolicy

from src.pr.safe_pyrep import SafePyRep
from src.conio.key_listen import ListenKeyPress, press_key_to_continue

from src.gym_env.plane_box.plane_box import PlaneBoxEnvTest

from finetuning.arg_dict import exp_exec_arg_dict, exp_replace_arg_dict
from stable_baselines3 import SAC, TD3

from src.net.efficient_net import EfficientNetV1WithHead
import torch

import time

def eval3(OBJECT: str = "corner", ENV_TYPE: str = "ext", ENV_SET: str = "hard"):
    # OBJECT = "corner"
    # ENV_TYPE = "ext"
    # ENV_SET = "hard"

    RL_POLICY = SAC.load(f"runs/done/{OBJECT}_ply/sac_hard_tunedres/best_model.zip")
    # RL_POLICY = SAC.load(f"runs/plane_box/corner_finetuning/ext_sac_wrong_hard/2025_03_11_00_41_53/best_model.zip")

    ENV_CONF_PATH = Path(f"app/plane_box/conf/{OBJECT}_{ENV_SET}_{ENV_TYPE}_env.yaml")
    BASE_CONF_PATH = Path(f"app/plane_box/conf/base_hard_env.yaml")

    effnet_b0 = EfficientNetV1WithHead(
        6, 1, 1, 1
    )
    effnet_b0.load_state_dict(torch.load(f"runs/done/{OBJECT}_{ENV_SET}_ext/best.pth"))
    effnet_b0.to(device = "cuda")
    effnet_b0.train(False)

    # 随机策略
    with SafePyRep(f"scene/plane_box/{OBJECT}_vec4_test2.ttt", True) as pr:

        env_conf = load_exp_config(BASE_CONF_PATH, ENV_CONF_PATH, is_resolve = True)
        env_conf.eval_env.wrapper = None
        eval_env = config_to_env(env_conf.eval_env, exp_replace_arg_dict({"pr": pr}), exp_exec_arg_dict())

        print("eval start")

        # 测试 Optuna 保存的模型中, 性能下降的原因
        eval_record_to_file(
            RandomPolicy(eval_env.action_space, 2),
            eval_env, 
            f"runs/plane_box/{OBJECT}_{ENV_TYPE}_eval", 
            save_name_prefix = "random_",
            n_eval_episodes = 50,
            num_record_episodes = 5,
            fps = 5
        )

        print("eval done")

    # 深度学习策略
    with SafePyRep(f"scene/plane_box/{OBJECT}_vec4_test2.ttt", True) as pr:

        env_conf = load_exp_config(BASE_CONF_PATH, ENV_CONF_PATH, is_resolve = True)
        env_conf.eval_env.wrapper = None
        eval_env = config_to_env(env_conf.eval_env, exp_replace_arg_dict({"pr": pr}), exp_exec_arg_dict())

        print("eval start")

        # 测试 Optuna 保存的模型中, 性能下降的原因
        eval_record_to_file(
            ModelPredictor(effnet_b0), 
            eval_env, 
            f"runs/plane_box/{OBJECT}_{ENV_TYPE}_eval", 
            save_name_prefix = "ml_",
            n_eval_episodes = 50,
            num_record_episodes = 5,
            fps = 5
        )

        print("eval done")

    # 强化学习策略
    with SafePyRep(f"scene/plane_box/{OBJECT}_vec4_test2.ttt", True) as pr:

        env_conf = load_exp_config(BASE_CONF_PATH, ENV_CONF_PATH, is_resolve = True)
        eval_env = config_to_env(env_conf.eval_env, exp_replace_arg_dict({"pr": pr}), exp_exec_arg_dict())

        print("eval start")

        # 测试 Optuna 保存的模型中, 性能下降的原因
        eval_record_to_file(
            RL_POLICY,
            eval_env, 
            f"runs/plane_box/{OBJECT}_{ENV_TYPE}_eval", 
            save_name_prefix = "sac_",
            n_eval_episodes = 50,
            num_record_episodes = 5,
            fps = 5
        )

        print("eval done")

if __name__ == "__main__":
    # eval3("corner", "ext", "hard")
    eval3("paralle", "ext", "hard")
    # eval3("three", "ext", "hard")
