import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from pathlib import Path

from src.pr.safe_pyrep import SafePyRep
from src.sb3.exp_manager import config_to_env, load_exp_config

from src.pr.safe_pyrep import SafePyRep
from src.conio.key_listen import ListenKeyPress, press_key_to_continue

from src.gym_env.plane_box.plane_box import PlaneBoxEnvTest

from conf.arg_dict import exp_exec_arg_dict, exp_replace_arg_dict
from stable_baselines3 import PPO

import time

# TODO: 改进对齐判断代码
# TODO: 回访队列保存
# TODO: 加大距离容错，增加角度检查

if __name__ == "__main__":

    ENV_TYPE = "three"
    CNF_VARY = "normal"
    OBS_TYPE = "ext"
    PAHASE_TYPE = "test"

    with SafePyRep(f"scene/plane_box/base_vec6.ttt", False) as pr:

        env_conf = load_exp_config(
            f"app/plane_box/conf/base_{CNF_VARY}_env.yaml",
            f"app/plane_box/conf/{ENV_TYPE}_{OBS_TYPE}_env_{PAHASE_TYPE}.yaml",
            # "app/plane_box/finetuning/ext_sac_raw.yaml",
            is_resolve = True)
        # env_conf = load_exp_config(
        #     f"app/plane_box/conf/tnued_three_hppo_default.yaml",
        #     # "app/plane_box/finetuning/ext_sac_raw.yaml",
        #     is_resolve = True)

        # env_conf.train_env.kwargs.env_action_noise = None
        env_conf.train_env.kwargs.subenv_range = [0, 4]
        # env_conf.train_env.kwargs.act_is_standard_parameter_space = True
        # env_conf.train_env.kwargs.env_tolerance_offset = 0.005

        # train_env = config_to_env(env_conf.train_env, exp_replace_arg_dict({"pr": pr}), exp_exec_arg_dict())
        # eval_env = config_to_env(env_conf.eval_env, exp_replace_arg_dict({"pr": pr}), exp_exec_arg_dict())

        tester = PlaneBoxEnvTest(
            lambda: config_to_env(env_conf.train_env, exp_replace_arg_dict({"pr": pr}), exp_exec_arg_dict()), 0, None, 3,
            move_rate = 1
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
