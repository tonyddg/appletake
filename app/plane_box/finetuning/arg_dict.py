import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

import albumentations as A
from src.gym_env.aug import get_coppeliasim_depth_normalize, get_crop_resize, get_depth_aug, get_hwc2chw
from stable_baselines3.common.noise import NormalActionNoise
import numpy as np

from src.gym_env.plane_box.plane_box import RewardLinearDistance
from src.net.efficient_net import EfficientNetV1WithHead
from src.utility import dict_factory

import torch

def make_backbone(pth_path: str):
    effnet_b0 = EfficientNetV1WithHead(
        6, 1, 1, 1
    )
    effnet_b0.load_state_dict(torch.load(pth_path))
    effnet_b0.to(device = "cuda")
    effnet_b0.train(False)
    return effnet_b0.backbone

T_SIZE = 224

CORNER_W = 336
CORNER_YOF = 60
CORNER_XOF = 60

PARALLE_W = 336
PARALLE_YOF = 60
PARALLE_XOF = 60

THREE_W = 336
THREE_YOF = 60
THREE_XOF = 60

exp_replace_arg_dict = dict_factory({
    'corner_vec_trans': A.Compose([
        get_crop_resize(
            (1280 - CORNER_W) // 2 + CORNER_XOF, (720 - CORNER_W) // 2 + CORNER_YOF, (1280 + CORNER_W) // 2 + CORNER_XOF, (720 + CORNER_W) // 2 + CORNER_YOF, T_SIZE, T_SIZE
        ),
        get_coppeliasim_depth_normalize(),
        get_hwc2chw(True)
    ]),
    'corner_ext_trans': A.Compose([
        get_crop_resize(
            (1280 - CORNER_W) // 2 + CORNER_XOF, (720 - CORNER_W) // 2 + CORNER_YOF, (1280 + CORNER_W) // 2 + CORNER_XOF, (720 + CORNER_W) // 2 + CORNER_YOF, T_SIZE, T_SIZE
        ),
        get_coppeliasim_depth_normalize(),
        get_depth_aug(0.9, 1),
        A.Normalize((0.536), (0.28), 1),
        get_hwc2chw(False)
    ]),
    'corner_ext_train_trans': A.Compose([
        get_crop_resize(
            (1280 - CORNER_W) // 2 + CORNER_XOF, (720 - CORNER_W) // 2 + CORNER_YOF, (1280 + CORNER_W) // 2 + CORNER_XOF, (720 + CORNER_W) // 2 + CORNER_YOF, T_SIZE, T_SIZE
        ),
        get_coppeliasim_depth_normalize(),
        get_depth_aug(0.8, 2),
        A.Normalize((0.536), (0.28), 1),
        get_hwc2chw(False)
    ]),

    'paralle_vec_trans': A.Compose([
        get_crop_resize(
            (1280 - PARALLE_W) // 2 + PARALLE_XOF, (720 - PARALLE_W) // 2 + PARALLE_YOF, (1280 + PARALLE_W) // 2 + PARALLE_XOF, (720 + PARALLE_W) // 2 + PARALLE_YOF, T_SIZE, T_SIZE
        ),
        get_coppeliasim_depth_normalize(),
        get_hwc2chw(False)
    ]),
    'paralle_ext_trans': A.Compose([
        get_crop_resize(
            (1280 - PARALLE_W) // 2 + PARALLE_XOF, (720 - PARALLE_W) // 2 + PARALLE_YOF, (1280 + PARALLE_W) // 2 + PARALLE_XOF, (720 + PARALLE_W) // 2 + PARALLE_YOF, T_SIZE, T_SIZE
        ),
        get_coppeliasim_depth_normalize(),
        get_depth_aug(0.9, 1),
        A.Normalize((0.67), (0.27), 1),
        get_hwc2chw(False)
    ]),
    'paralle_ext_train_trans': A.Compose([
        get_crop_resize(
            (1280 - PARALLE_W) // 2 + PARALLE_XOF, (720 - PARALLE_W) // 2 + PARALLE_YOF, (1280 + PARALLE_W) // 2 + PARALLE_XOF, (720 + PARALLE_W) // 2 + PARALLE_YOF, T_SIZE, T_SIZE
        ),
        get_coppeliasim_depth_normalize(),
        get_depth_aug(0.8, 2),
        A.Normalize((0.67), (0.27), 1),
        get_hwc2chw(False)
    ]),

    'three_vec_trans': A.Compose([
        get_crop_resize(
            (1280 - THREE_W) // 2 + THREE_XOF, (720 - THREE_W) // 2 + THREE_YOF, (1280 + THREE_W) // 2 + THREE_XOF, (720 + THREE_W) // 2 + THREE_YOF, T_SIZE, T_SIZE
        ),
        get_coppeliasim_depth_normalize(),
        get_hwc2chw(False)
    ]),
    'three_ext_trans': A.Compose([
        get_crop_resize(
            (1280 - THREE_W) // 2 + THREE_XOF, (720 - THREE_W) // 2 + THREE_YOF, (1280 + THREE_W) // 2 + THREE_XOF, (720 + THREE_W) // 2 + THREE_YOF, T_SIZE, T_SIZE
        ),
        get_coppeliasim_depth_normalize(),
        get_depth_aug(0.9, 1),
        A.Normalize((0.38), (0.3), 1),
        get_hwc2chw(False)
    ]),
    'three_ext_train_trans': A.Compose([
        get_crop_resize(
            (1280 - THREE_W) // 2 + THREE_XOF, (720 - THREE_W) // 2 + THREE_YOF, (1280 + THREE_W) // 2 + THREE_XOF, (720 + THREE_W) // 2 + THREE_YOF, T_SIZE, T_SIZE
        ),
        get_coppeliasim_depth_normalize(),
        get_depth_aug(0.8, 2),
        A.Normalize((0.38), (0.3), 1),
        get_hwc2chw(False)
    ]),

    "reward_dist_fn": RewardLinearDistance()
})

exp_exec_arg_dict = dict_factory({
    'normal_noise': lambda kwargs: NormalActionNoise(np.zeros(6), kwargs["sigma"] * np.ones(6)),
    "make_backbone": lambda kwargs: make_backbone(kwargs["path"])
})
