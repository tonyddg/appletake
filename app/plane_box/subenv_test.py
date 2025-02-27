import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np
# from src.gym_env.obs import ColorObsA

from src.gym_env.plane_box.plane_box import PlaneBoxSubenvBase, PlaneBoxSubenvTest, RewardLinearDistance

from src.gym_env.plane_box.paralle import ParalleSubEnv
from src.gym_env.plane_box.corner import CornerSubEnv
from src.gym_env.plane_box.three import ThreeSubEnv

from src.pr.safe_pyrep import SafePyRep
from src.conio.key_listen import ListenKeyPress, press_key_to_continue

import albumentations as A
from albumentations.core.composition import TransformType
from src.gym_env.aug import get_coppeliasim_depth_normalize, get_crop_resize, get_depth_thresh_normalize


# TODO: 向动作加入噪声

if __name__ == "__main__":

    W = 224
    YOF = 120

    with SafePyRep("scene/box_plane/corner.ttt") as pr:
        env = CornerSubEnv(
            name_suffix = "", 
            obs_trans = A.Compose([
                get_crop_resize(
                    (1280 - W) // 2, (720 - W) // 2 + YOF, (1280 + W) // 2, (720 + W) // 2 + YOF, 224, 224
                ),
                get_coppeliasim_depth_normalize()
            ]),
            obs_source = "depth",
            env_object_kwargs = None,
            env_init_box_pos_range = (
                np.array([-50, -50, 0, 0, 0, -5], dtype = np.float32),
                np.array([50, 50, 50, 0, 0, 5], dtype = np.float32)
            ),
            env_init_vis_pos_range = (
                np.array([-10, -10, -10, -1, -1, -1], dtype = np.float32),
                np.array([10, 10, 10, 1, 1, 1], dtype = np.float32)
            ),
            env_action_noise = np.array([1, 1, 1, 0.1, 0.1, 0.1], dtype = np.float32),
            # Checker 长度为 50mm, 因此仅当箱子与垛盘间隙为 51~100 mm (1~50) 时可通过检查 
            env_test_in = 0.050
        )
        tester = PlaneBoxSubenvTest(
            pr, env, 10, reward_fn = RewardLinearDistance()
        )
        key_cb = tester.get_base_key_dict()

        key_cb.update({
            27: lambda: True,
            # '6': lambda: tester.env._set_move_box_abs_pose(np.array([-50, -50, 50, 0, 0, -5], dtype = np.float32)),
            # '7': lambda: tester.env._set_move_box_abs_pose(np.array([50, 50, 0, 0, 0, 5], dtype = np.float32)),

            '9': lambda: tester.env._set_move_box_abs_pose(np.array([-9, -1, 1, 0, 0, 0], dtype = np.float32)),
            '0': lambda: tester.env._set_move_box_abs_pose(np.array([-1, -9, 49, 0, 0, 0], dtype = np.float32)),
        })

        with ListenKeyPress(key_cb) as handler:
            while True:
                is_exit = handler()
                if is_exit == True:
                    break
                
                pr.step()

        press_key_to_continue(idle_run = pr.step)
