import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

import albumentations as A
from src.gym_env.aug import get_coppeliasim_depth_normalize, get_crop_resize, get_depth_aug, get_hwc2chw
from stable_baselines3.common.noise import NormalActionNoise
import numpy as np

from src.net.efficient_net import EfficientNetWithHead
from src.utility import dict_factory

import torch

effnet_b0 = EfficientNetWithHead(
    6, 1, 1, 1
)
effnet_b0.load_state_dict(torch.load("runs/train_net/socket_plug/best.pth"))
effnet_b0.to(device = "cuda")
effnet_b0.train(False)
effnet_b0_backbone = effnet_b0.backbone

W = 224
YOF = 120

exp_replace_arg_dict = dict_factory({
    'vec_trans': A.Compose([
        get_crop_resize(
            (1280 - W) // 2, (720 - W) // 2 + YOF, (1280 + W) // 2, (720 + W) // 2 + YOF, 96, 96
        ),
        get_coppeliasim_depth_normalize(),
        get_hwc2chw(True)
    ]),
    'ext_trans': A.Compose([
        get_crop_resize(
            (1280 - W) // 2, (720 - W) // 2 + YOF, (1280 + W) // 2, (720 + W) // 2 + YOF, 224, 224
        ),
        get_coppeliasim_depth_normalize(),
        get_depth_aug(0.2, 1),
        get_hwc2chw(False)
    ]),
    'backbone': effnet_b0_backbone,
})

exp_exec_arg_dict = dict_factory({
    'normal_noise': lambda kwargs: NormalActionNoise(np.zeros(6), kwargs["sigma"] * np.ones(6))
})
