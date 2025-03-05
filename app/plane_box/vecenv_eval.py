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
from stable_baselines3 import SAC

from src.net.efficient_net import EfficientNetV1WithHead
import torch

import time

if __name__ == "__main__":

    OBJECT = "corner"
    ENV_TYPE = "normal_ext"

    ENV_CONF_PATH = Path(f"app/plane_box/conf/{OBJECT}_{ENV_TYPE}_env.yaml")

    with SafePyRep(f"scene/plane_box/{OBJECT}_vec4_test2.ttt", True) as pr:

        env_conf = load_exp_config(ENV_CONF_PATH)
        env_conf.eval_env.wrapper = None

        eval_env = config_to_env(env_conf.eval_env, exp_replace_arg_dict({"pr": pr}), exp_exec_arg_dict())

        effnet_b0 = EfficientNetV1WithHead(
            6, 1, 1, 1
        )
        effnet_b0.load_state_dict(torch.load("runs/plane_box/corner_ext/2025_03_03_20_40_58/best.pth"))
        effnet_b0.to(device = "cuda")
        effnet_b0.train(False)

        print("eval start")

        avg_red, std_red = eval_record_to_file(
            ModelPredictor(effnet_b0), 
            # RandomPolicy(eval_env.action_space, 2),
            # SAC.load("runs/plane_box/corner_ply/ext_sac_2025_03_04_18_23_19/best_model.zip"),
            eval_env, 
            "runs/plane_box/corner_eval", 
            n_eval_episodes = 50,
            num_record_episodes = 5,
            fps = 5
        )

        print(f"avg: {avg_red}, std: {std_red}")
