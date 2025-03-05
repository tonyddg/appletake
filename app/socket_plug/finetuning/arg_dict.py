import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

import albumentations as A
from src.gym_env.aug import get_crop_resize, get_depth_aug, get_hwc2chw
from stable_baselines3.common.noise import NormalActionNoise
import numpy as np

from src.net.efficient_net import EfficientNetV1WithHead
from src.utility import dict_factory

import torch

effnet_b0 = EfficientNetV1WithHead(
    6, 1, 1, 1
)
effnet_b0.load_state_dict(torch.load("runs/socket_plug/ext/2025_02_27_12_34_23/best.pth"))
effnet_b0.to(device = "cuda")
effnet_b0.train(False)
effnet_b0_backbone = effnet_b0.backbone

exp_replace_arg_dict = dict_factory({
    'vec_trans': A.Compose([
        get_crop_resize(resize_height = 128, resize_width = 128),
        get_hwc2chw(True)
    ]),
    'ext_trans': A.Compose([
        get_crop_resize(),
        get_depth_aug(1, 1),
        A.Normalize((0.538), (0.287), 1),
        get_hwc2chw(False)
    ]),
    'backbone': effnet_b0_backbone,
})

exp_exec_arg_dict = dict_factory({
    'normal_noise': lambda kwargs: NormalActionNoise(np.zeros(6), kwargs["sigma"] * np.ones(6))
})
