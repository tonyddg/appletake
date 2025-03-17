import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np
from pathlib import Path

from src.pr.safe_pyrep import SafePyRep
from src.sb3.exp_manager import config_to_env, load_exp_config

from src.pr.safe_pyrep import SafePyRep
from src.conio.key_listen import ListenKeyPress, press_key_to_continue

from src.gym_env.plane_box.plane_box import PlaneBoxEnvTest

from finetuning.arg_dict import exp_exec_arg_dict, exp_replace_arg_dict
from stable_baselines3 import TD3

import time

if __name__ == "__main__":

    OBJECT = "three"
    ENV_TYPE = "hard"
    OBS_TYPE = "ext"

    with SafePyRep(f"scene/plane_box/{OBJECT}_vec4_test2.ttt", False) as pr:

        env_conf = load_exp_config(
            f"app/plane_box/conf/base_{ENV_TYPE}_env.yaml",
            f"app/plane_box/conf/{OBJECT}_{ENV_TYPE}_{OBS_TYPE}_env.yaml",
            # "app/plane_box/finetuning/ext_sac_raw.yaml",
            is_resolve = True)
        env_conf.train_env.kwargs.env_action_noise = None

        # train_env = config_to_env(env_conf.train_env, exp_replace_arg_dict({"pr": pr}), exp_exec_arg_dict())
        # eval_env = config_to_env(env_conf.eval_env, exp_replace_arg_dict({"pr": pr}), exp_exec_arg_dict())

        tester = PlaneBoxEnvTest(
            lambda: config_to_env(env_conf.eval_env, exp_replace_arg_dict({"pr": pr}), exp_exec_arg_dict()), 0, None
        )
        key_cb = tester.get_base_key_dict()

        key_cb.update({
            27: lambda: True,
        })

        step_cnt = 0
        start_time = time.perf_counter()
        with ListenKeyPress(key_cb) as handler:
            while True:
                is_exit = handler()
                if is_exit == True:
                    break
                
                pr.step()
                step_cnt += 1

        use_time = time.perf_counter() - start_time
        print(f"sim speed: {use_time / step_cnt :.3e} s/step")
        press_key_to_continue(idle_run = pr.step)
