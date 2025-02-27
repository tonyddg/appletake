import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np
from pathlib import Path

from src.pr.safe_pyrep import SafePyRep
from src.conio.key_listen import press_key_to_continue
from src.sb3.exp_manager import config_to_env, load_exp_config

from finetuning.arg_dict import exp_exec_arg_dict, exp_replace_arg_dict

if __name__ == "__main__":

    W = 256
    YOF = 80

    ENV_CONF_PATH = Path("app/plane_box/conf/corner_normal_vec_env.yaml")

    with SafePyRep("scene/plane_box/corner_vec4.ttt", True) as pr:

        env_conf = load_exp_config(ENV_CONF_PATH)
        train_env = config_to_env(env_conf.train_env, exp_replace_arg_dict({"pr": pr}), None)
        eval_env = config_to_env(env_conf.eval_env, exp_replace_arg_dict({"pr": pr}), None)

        print("done")
        press_key_to_continue(idle_run = pr.step)
