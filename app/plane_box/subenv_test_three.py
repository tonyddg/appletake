import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np

from src.gym_env.plane_box.plane_box import PlaneBoxSubenvTest, RewardLinearDistance
from src.gym_env.plane_box.corner import CornerSubEnv
from src.gym_env.plane_box.paralle import ParalleSubEnv
from src.gym_env.plane_box.three import ThreeSubEnv

from src.pr.safe_pyrep import SafePyRep
from src.conio.key_listen import ListenKeyPress, press_key_to_continue

import albumentations as A
from finetuning.arg_dict import exp_exec_arg_dict, exp_replace_arg_dict

# TODO: 向动作加入噪声

if __name__ == "__main__":

    with SafePyRep("scene/plane_box/three.ttt") as pr:
        env = ThreeSubEnv(
            name_suffix = "", 
            obs_trans = exp_replace_arg_dict()["three_ext_trans"],
            obs_source = "depth",
            env_init_box_pos_range = (
                np.array([-40, -40, 10, -5, -5, -10], dtype = np.float32),
                np.array([40, 40, 40, 5, 5, 10], dtype = np.float32)
            ),
            env_init_vis_pos_range = (
                np.array([-10, -10, -10, -5, -5, -10], dtype = np.float32),
                np.array([10, 10, 10, 5, 5, 10], dtype = np.float32)
            ),
            env_action_noise = np.array([0.5, 0.5, 0.5, 0.1, 0.1, 0.1], dtype = np.float32),
            env_vis_persp_deg_disturb = 1,
            env_movbox_height_offset_range = (-0.03, 0.02),
            env_movebox_center_err = (
                np.array([-5, -5, -5, -2, -2, -5], dtype = np.float32),
                np.array([5, 5, 5, 2, 2, 5], dtype = np.float32)
            ),
            
            env_tolerance_offset = -0.005,

            # Checker 长度为 50mm, 因此仅当箱子与垛盘间隙为 51~100 mm (1~50) 时可通过检查 
            env_test_in = 0.050,

            three_fixbox_size_range = (
                np.array([0.12, 0.12, 0.15], dtype = np.float32),
                np.array([0.30, 0.30, 0.20], dtype = np.float32)
            )
        )
        tester = PlaneBoxSubenvTest(
            pr, env, 5, reward_fn = RewardLinearDistance()
        )
        key_cb = tester.get_base_key_dict()

        key_cb.update({
            27: lambda: True,

            '9': lambda: tester.try_init_pose(np.array([-9, -1, 1, 0, 0, 0], dtype = np.float32)),
            '0': lambda: tester.try_init_pose(np.array([-1, -9, 49, 0, 0, 0], dtype = np.float32)),

            '=': lambda: tester.try_init_pose(np.array([0.0, 0.0, 0.0], dtype = np.float32))
        })

        with ListenKeyPress(key_cb) as handler:
            while True:
                is_exit = handler()
                if is_exit == True:
                    break
                
                pr.step()
        tester.env._close_env()

        press_key_to_continue(idle_run = pr.step)
