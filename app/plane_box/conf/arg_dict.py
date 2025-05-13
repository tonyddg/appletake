import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

import albumentations as A
from src.gym_env.aug import get_coppeliasim_depth_normalize, get_crop_resize, get_depth_aug_single_view, get_depth_aug, get_hwc2chw, multi_crop_stack
from stable_baselines3.common.noise import NormalActionNoise
import numpy as np

from src.gym_env.plane_box.reward import RewardLinearDistance, RewardLinearDistanceAndAlignQuality, RewardApproachAndAlignQualityAndTimeout, RewardPassiveTimeout

from src.net.net_abc import BackboneWithHead
from src.net.efficient_net import EfficientNetV1Backbone
from src.net.multi_view_net import SeperateLateFuseNet

from src.utility import dict_factory
from src.sb3.lr_schedule import MaintainCosineLR

import torch
from torch import nn

# 不分离通道
def make_effv1_backbone(pth_path: str, channels: int = 3):
    net = BackboneWithHead(
        EfficientNetV1Backbone(
            channels, 1, 1, 0.2
        ), 6, 0.2
    )
    net.load_state_dict(torch.load(pth_path))
    net.to(device = "cuda")
    net.train(False)
    return net.backbone


# 不分离通道
def make_effv1_direct(pth_path: str, channels: int = 3):
    net = BackboneWithHead(
        EfficientNetV1Backbone(
            channels, 1, 1, 0.2
        ), 6, 0.2
    )
    net.load_state_dict(torch.load(pth_path))
    net.to(device = "cuda")
    net.train(False)
    return net

# # 基于 SEBlock 的中期融合
# def make_seblend_backbone(pth_path: str, fuse_stage: int = 3):
#     net = BackboneWithHead(
#         SEBlendEffNet(
#             3, fuse_stage, 1, 1, 0.2
#         ), 6, 0.2
#     )
#     net.load_state_dict(torch.load(pth_path))
#     net.to(device = "cuda")
#     net.train(False)
#     return net.backbone

# 直接合并特征
def make_featcat_backbone(pth_path: str):
    net = BackboneWithHead(
        SeperateLateFuseNet(
            3, lambda: EfficientNetV1Backbone(1, 1, 1, 0.2)
        ), 6, 0.2
    )
    net.load_state_dict(torch.load(pth_path))
    net.to(device = "cuda")
    net.train(False)
    return net.backbone

T_SIZE = 224
V_SIZE = 128

WINDOW_W = 336

CENTER_XOF = 0
CENTER_YOF = 60

LEFT_XOF = -265
LEFT_YOF = -135

RIGHT_XOF = 265
RIGHT_YOF = -135

#############

T_SIZE_B2 = 288
V_SIZE_B2 = 128

WINDOW_W_B2 = 346

CENTER_XOF_B2 = 0
CENTER_YOF_B2 = 60

LEFT_XOF_B2 = -270
LEFT_YOF_B2 = -150

RIGHT_XOF_B2 = 270
RIGHT_YOF_B2 = -150

#############

SINGLE_H = 560
SINGLE_W = 880
SINGLE_Y_OFFSET = -40

SINGLE_OUT_H = 224
SINGLE_OUT_W = 352

exp_replace_arg_dict = dict_factory({

    'train_trans': A.Compose([
        get_coppeliasim_depth_normalize(),
        multi_crop_stack(
            [(1280 - WINDOW_W) // 2 + CENTER_XOF, (1280 - WINDOW_W) // 2 + LEFT_XOF, (1280 - WINDOW_W) // 2 + RIGHT_XOF], 
            [(720 - WINDOW_W) // 2 + CENTER_YOF, (720 - WINDOW_W) // 2 + LEFT_YOF, (720 - WINDOW_W) // 2 + RIGHT_YOF], 
            [(1280 + WINDOW_W) // 2 + CENTER_XOF, (1280 + WINDOW_W) // 2 + LEFT_XOF, (1280 + WINDOW_W) // 2 + RIGHT_XOF], 
            [(720 + WINDOW_W) // 2 + CENTER_YOF, (720 + WINDOW_W) // 2 + LEFT_YOF, (720 + WINDOW_W) // 2 + RIGHT_YOF], 
            T_SIZE, T_SIZE
        ),
        get_depth_aug(0.8, 1.5, is_train = True),
        get_hwc2chw(False)
    ]),
    # 'eval_trans': A.Compose([
    #     get_coppeliasim_depth_normalize(),
    #     multi_crop_stack(
    #         [(1280 - WINDOW_W) // 2 + CENTER_XOF, (1280 - WINDOW_W) // 2 + LEFT_XOF, (1280 - WINDOW_W) // 2 + RIGHT_XOF], 
    #         [(720 - WINDOW_W) // 2 + CENTER_YOF, (720 - WINDOW_W) // 2 + LEFT_YOF, (720 - WINDOW_W) // 2 + RIGHT_YOF], 
    #         [(1280 + WINDOW_W) // 2 + CENTER_XOF, (1280 + WINDOW_W) // 2 + LEFT_XOF, (1280 + WINDOW_W) // 2 + RIGHT_XOF], 
    #         [(720 + WINDOW_W) // 2 + CENTER_YOF, (720 + WINDOW_W) // 2 + LEFT_YOF, (720 + WINDOW_W) // 2 + RIGHT_YOF], 
    #         T_SIZE, T_SIZE
    #     ),
    #     get_depth_aug(0.9, 1, is_train = False),
    #     get_hwc2chw(False)
    # ]),

    'ext_trans': A.Compose([
        get_coppeliasim_depth_normalize(),
        multi_crop_stack(
            [(1280 - WINDOW_W) // 2 + CENTER_XOF, (1280 - WINDOW_W) // 2 + LEFT_XOF, (1280 - WINDOW_W) // 2 + RIGHT_XOF], 
            [(720 - WINDOW_W) // 2 + CENTER_YOF, (720 - WINDOW_W) // 2 + LEFT_YOF, (720 - WINDOW_W) // 2 + RIGHT_YOF], 
            [(1280 + WINDOW_W) // 2 + CENTER_XOF, (1280 + WINDOW_W) // 2 + LEFT_XOF, (1280 + WINDOW_W) // 2 + RIGHT_XOF], 
            [(720 + WINDOW_W) // 2 + CENTER_YOF, (720 + WINDOW_W) // 2 + LEFT_YOF, (720 + WINDOW_W) // 2 + RIGHT_YOF], 
            T_SIZE, T_SIZE
        ),
        get_depth_aug(0.9, 1, is_train = False),
        get_hwc2chw(False)
    ]),

    'vec_trans': A.Compose([
        get_coppeliasim_depth_normalize(),
        multi_crop_stack(
            [(1280 - WINDOW_W) // 2 + CENTER_XOF, (1280 - WINDOW_W) // 2 + LEFT_XOF, (1280 - WINDOW_W) // 2 + RIGHT_XOF], 
            [(720 - WINDOW_W) // 2 + CENTER_YOF, (720 - WINDOW_W) // 2 + LEFT_YOF, (720 - WINDOW_W) // 2 + RIGHT_YOF], 
            [(1280 + WINDOW_W) // 2 + CENTER_XOF, (1280 + WINDOW_W) // 2 + LEFT_XOF, (1280 + WINDOW_W) // 2 + RIGHT_XOF], 
            [(720 + WINDOW_W) // 2 + CENTER_YOF, (720 + WINDOW_W) // 2 + LEFT_YOF, (720 + WINDOW_W) // 2 + RIGHT_YOF], 
            T_SIZE, T_SIZE
        ),
        get_depth_aug(0.9, 1, is_train = False),
        get_hwc2chw(True)
    ]),


    'ext_trans_noaug': A.Compose([
        get_coppeliasim_depth_normalize(),
        # get_depth_train_aug(0.95, 1.5),
        multi_crop_stack(
            [(1280 - WINDOW_W) // 2 + CENTER_XOF, (1280 - WINDOW_W) // 2 + LEFT_XOF, (1280 - WINDOW_W) // 2 + RIGHT_XOF], 
            [(720 - WINDOW_W) // 2 + CENTER_YOF, (720 - WINDOW_W) // 2 + LEFT_YOF, (720 - WINDOW_W) // 2 + RIGHT_YOF], 
            [(1280 + WINDOW_W) // 2 + CENTER_XOF, (1280 + WINDOW_W) // 2 + LEFT_XOF, (1280 + WINDOW_W) // 2 + RIGHT_XOF], 
            [(720 + WINDOW_W) // 2 + CENTER_YOF, (720 + WINDOW_W) // 2 + LEFT_YOF, (720 + WINDOW_W) // 2 + RIGHT_YOF], 
            T_SIZE, T_SIZE
        ),
        get_hwc2chw(False)
    ]),

    # 'train_trans_b2': A.Compose([
    #     get_coppeliasim_depth_normalize(),
    #     multi_crop_stack(
    #         [(1280 - WINDOW_W_B2) // 2 + CENTER_XOF_B2, (1280 - WINDOW_W_B2) // 2 + LEFT_XOF_B2, (1280 - WINDOW_W_B2) // 2 + RIGHT_XOF_B2], 
    #         [(720 - WINDOW_W_B2) // 2 + CENTER_YOF_B2, (720 - WINDOW_W_B2) // 2 + LEFT_YOF_B2, (720 - WINDOW_W_B2) // 2 + RIGHT_YOF_B2], 
    #         [(1280 + WINDOW_W_B2) // 2 + CENTER_XOF_B2, (1280 + WINDOW_W_B2) // 2 + LEFT_XOF_B2, (1280 + WINDOW_W_B2) // 2 + RIGHT_XOF_B2], 
    #         [(720 + WINDOW_W_B2) // 2 + CENTER_YOF_B2, (720 + WINDOW_W_B2) // 2 + LEFT_YOF_B2, (720 + WINDOW_W_B2) // 2 + RIGHT_YOF_B2], 
    #         T_SIZE_B2, T_SIZE_B2
    #     ),
    #     get_depth_aug(0.8, 2.0, is_train = True),
    #     get_hwc2chw(False)
    # ]),
    # 'ext_trans_b2': A.Compose([
    #     get_coppeliasim_depth_normalize(),
    #     multi_crop_stack(
    #         [(1280 - WINDOW_W_B2) // 2 + CENTER_XOF_B2, (1280 - WINDOW_W_B2) // 2 + LEFT_XOF_B2, (1280 - WINDOW_W_B2) // 2 + RIGHT_XOF_B2], 
    #         [(720 - WINDOW_W_B2) // 2 + CENTER_YOF_B2, (720 - WINDOW_W_B2) // 2 + LEFT_YOF_B2, (720 - WINDOW_W_B2) // 2 + RIGHT_YOF_B2], 
    #         [(1280 + WINDOW_W_B2) // 2 + CENTER_XOF_B2, (1280 + WINDOW_W_B2) // 2 + LEFT_XOF_B2, (1280 + WINDOW_W_B2) // 2 + RIGHT_XOF_B2], 
    #         [(720 + WINDOW_W_B2) // 2 + CENTER_YOF_B2, (720 + WINDOW_W_B2) // 2 + LEFT_YOF_B2, (720 + WINDOW_W_B2) // 2 + RIGHT_YOF_B2], 
    #         T_SIZE_B2, T_SIZE_B2
    #     ),
    #     get_depth_aug(0.9, 1.2, is_train = False),
    #     get_hwc2chw(False)
    # ]),

    # 'train_trans_corner': A.Compose([
    #     get_coppeliasim_depth_normalize(),
    #     get_crop_resize(
    #         (1280 - 720) // 2, #  + CENTER_XOF,
    #         (720 - 720) // 2 , # + CENTER_YOF, 
    #         (1280 + 720) // 2, #  + CENTER_XOF,
    #         (720 + 720) // 2 , # + CENTER_YOF, 
    #         T_SIZE, T_SIZE
    #     ),
    #     get_depth_aug_single_view(0.8, 1.5, is_train = True),
    #     get_hwc2chw(False)
    # ]),
    # 'ext_trans_corner': A.Compose([
    #     get_coppeliasim_depth_normalize(),
    #     get_crop_resize(
    #         (1280 - 720) // 2, #  + CENTER_XOF,
    #         (720 - 720) // 2 , # + CENTER_YOF, 
    #         (1280 + 720) // 2, #  + CENTER_XOF,
    #         (720 + 720) // 2 , # + CENTER_YOF, 
    #         T_SIZE, T_SIZE
    #     ),
    #     get_depth_aug_single_view(0.9, 1, is_train = False),
    #     get_hwc2chw(False)
    # ]),

    'train_trans_single': A.Compose([
        get_coppeliasim_depth_normalize(),
        get_crop_resize(
            (1280 - SINGLE_W) // 2, #  + CENTER_XOF,
            (720 - SINGLE_H) // 2 + SINGLE_Y_OFFSET, # + CENTER_YOF, 
            (1280 + SINGLE_W) // 2, #  + CENTER_XOF,
            (720 + SINGLE_H) // 2 + SINGLE_Y_OFFSET, # + CENTER_YOF, 
            SINGLE_OUT_H, SINGLE_OUT_W
        ),
        get_depth_aug_single_view(0.8, 1.8, is_train = True),
        get_hwc2chw(False)
    ]),
    'ext_trans_single': A.Compose([
        get_coppeliasim_depth_normalize(),
        get_crop_resize(
            (1280 - SINGLE_W) // 2, #  + CENTER_XOF,
            (720 - SINGLE_H) // 2 + SINGLE_Y_OFFSET, # + CENTER_YOF, 
            (1280 + SINGLE_W) // 2, #  + CENTER_XOF,
            (720 + SINGLE_H) // 2 + SINGLE_Y_OFFSET, # + CENTER_YOF, 
            SINGLE_OUT_H, SINGLE_OUT_W
        ),
        get_depth_aug_single_view(0.9, 1.2, is_train = False),
        get_hwc2chw(False)
    ]),

    ###

    # 'corner_vec_trans': A.Compose([
    #     get_crop_resize(
    #         (1280 - CORNER_W) // 2 + CORNER_XOF, (720 - CORNER_W) // 2 + CORNER_YOF, (1280 + CORNER_W) // 2 + CORNER_XOF, (720 + CORNER_W) // 2 + CORNER_YOF, T_SIZE, T_SIZE
    #     ),
    #     get_coppeliasim_depth_normalize(),
    #     get_hwc2chw(True)
    # ]),
    # 'corner_ext_trans': A.Compose([
    #     get_crop_resize(
    #         (1280 - CORNER_W) // 2 + CORNER_XOF, (720 - CORNER_W) // 2 + CORNER_YOF, (1280 + CORNER_W) // 2 + CORNER_XOF, (720 + CORNER_W) // 2 + CORNER_YOF, T_SIZE, T_SIZE
    #     ),
    #     get_coppeliasim_depth_normalize(),
    #     get_depth_aug(0.9, 1),
    #     A.Normalize((0.54), (0.276), 1),
    #     get_hwc2chw(False)
    # ]),
    # 'corner_train_trans': A.Compose([
    #     get_coppeliasim_depth_normalize(),
    #     get_depth_aug(0.8, 2),
    #     multi_crop_stack(
    #         [(1280 - CORNER_W) // 2 + CENTER_XOF, (1280 - CORNER_W) // 2 + LEFT_XOF, (1280 - CORNER_W) // 2 + RIGHT_XOF], 
    #         [(720 - CORNER_W) // 2 + CENTER_YOF, (720 - CORNER_W) // 2 + LEFT_YOF, (720 - CORNER_W) // 2 + RIGHT_YOF], 
    #         [(1280 + CORNER_W) // 2 + CENTER_XOF, (1280 + CORNER_W) // 2 + LEFT_XOF, (1280 + CORNER_W) // 2 + RIGHT_XOF], 
    #         [(720 + CORNER_W) // 2 + CENTER_YOF, (720 + CORNER_W) // 2 + LEFT_YOF, (720 + CORNER_W) // 2 + RIGHT_YOF], 
    #         T_SIZE, T_SIZE
    #     ),
    #     # A.Normalize((0.54), (0.276), 1),
    #     get_hwc2chw(False)
    # ]),

    # 'paralle_vec_trans': A.Compose([
    #     get_crop_resize(
    #         (1280 - PARALLE_W) // 2 + PARALLE_XOF, (720 - PARALLE_W) // 2 + PARALLE_YOF, (1280 + PARALLE_W) // 2 + PARALLE_XOF, (720 + PARALLE_W) // 2 + PARALLE_YOF, T_SIZE, T_SIZE
    #     ),
    #     get_coppeliasim_depth_normalize(),
    #     get_hwc2chw(False)
    # ]),
    # 'paralle_ext_trans': A.Compose([
    #     get_crop_resize(
    #         (1280 - PARALLE_W) // 2 + PARALLE_XOF, (720 - PARALLE_W) // 2 + PARALLE_YOF, (1280 + PARALLE_W) // 2 + PARALLE_XOF, (720 + PARALLE_W) // 2 + PARALLE_YOF, T_SIZE, T_SIZE
    #     ),
    #     get_coppeliasim_depth_normalize(),
    #     get_depth_aug(0.9, 1),
    #     A.Normalize((0.57), (0.30), 1),
    #     get_hwc2chw(False)
    # ]),
    # 'paralle_ext_train_trans': A.Compose([
    #     get_crop_resize(
    #         (1280 - PARALLE_W) // 2 + PARALLE_XOF, (720 - PARALLE_W) // 2 + PARALLE_YOF, (1280 + PARALLE_W) // 2 + PARALLE_XOF, (720 + PARALLE_W) // 2 + PARALLE_YOF, T_SIZE, T_SIZE
    #     ),
    #     get_coppeliasim_depth_normalize(),
    #     get_depth_aug(0.8, 2),
    #     A.Normalize((0.57), (0.30), 1),
    #     get_hwc2chw(False)
    # ]),

    # 'three_vec_trans': A.Compose([
    #     get_crop_resize(
    #         (1280 - THREE_W) // 2 + THREE_XOF, (720 - THREE_W) // 2 + THREE_YOF, (1280 + THREE_W) // 2 + THREE_XOF, (720 + THREE_W) // 2 + THREE_YOF, T_SIZE, T_SIZE
    #     ),
    #     get_coppeliasim_depth_normalize(),
    #     get_hwc2chw(False)
    # ]),
    # 'three_ext_trans': A.Compose([
    #     get_crop_resize(
    #         (1280 - THREE_W) // 2 + THREE_XOF, (720 - THREE_W) // 2 + THREE_YOF, (1280 + THREE_W) // 2 + THREE_XOF, (720 + THREE_W) // 2 + THREE_YOF, T_SIZE, T_SIZE
    #     ),
    #     get_coppeliasim_depth_normalize(),
    #     get_depth_aug(0.9, 1),
    #     A.Normalize((0.38), (0.3), 1),
    #     get_hwc2chw(False)
    # ]),
    # 'three_ext_train_trans': A.Compose([
    #     get_crop_resize(
    #         (1280 - THREE_W) // 2 + THREE_XOF, (720 - THREE_W) // 2 + THREE_YOF, (1280 + THREE_W) // 2 + THREE_XOF, (720 + THREE_W) // 2 + THREE_YOF, T_SIZE, T_SIZE
    #     ),
    #     get_coppeliasim_depth_normalize(),
    #     get_depth_aug(0.8, 2),
    #     A.Normalize((0.38), (0.3), 1),
    #     get_hwc2chw(False)
    # ]),
})

exp_exec_arg_dict = dict_factory({
    'normal_noise': lambda kwargs: NormalActionNoise(np.zeros(6), kwargs["sigma"] * np.ones(6)),
    'decision_normal_noise': lambda kwargs: NormalActionNoise(np.zeros(8), np.hstack([kwargs["pos_sigma"] * np.ones(3), kwargs["rot_sigma"] * np.ones(3), kwargs["des_sigma"] * np.ones(2)])),
    
    "make_effv1_backbone": lambda kwargs: make_effv1_backbone(
        kwargs["path"], channels = kwargs.get("channels", 3)
    ),
    "make_effv1_direct": lambda kwargs: make_effv1_direct(
        kwargs["path"], channels = kwargs.get("channels", 3)
    ),
    "make_featcat_backbone": lambda kwargs: make_featcat_backbone(kwargs["path"]),

    "reward_dist_fn": lambda kwargs: RewardLinearDistance(kwargs["max_pos_dis_mm"], kwargs["max_rot_dis_deg"]),
    "reward_align_fn": lambda kwargs: RewardLinearDistanceAndAlignQuality(
        kwargs["max_pos_dis_mm"], 
        kwargs["max_rot_dis_deg"],
        align_fail_panelty = kwargs["align_fail_panelty"],
        max_align_pos_dis_mm = kwargs["max_align_pos_dis_mm"],
        max_align_rot_dis_deg = kwargs["max_align_rot_dis_deg"],
        is_attract_to_center = kwargs["is_attract_to_center"],
        is_square_attract = kwargs["is_square_attract"]
    ),
    "reward_approach_fn": lambda kwargs: RewardApproachAndAlignQualityAndTimeout(
        timeout_panelty = kwargs["timeout_panelty"],
        align_fail_panelty = kwargs["align_fail_panelty"],

        success_reward = kwargs["success_reward"],
        max_align_reward = kwargs["max_align_reward"],
        max_align_pos_dis_mm = kwargs["max_align_pos_dis_mm"],
        max_align_rot_dis_deg = kwargs["max_align_rot_dis_deg"],

        max_move_reward = kwargs["max_approach_reward"],
        max_move_pos_dis_mm = kwargs["max_approach_pos_dis_mm"],
        max_move_rot_dis_deg = kwargs["max_approach_rot_dis_deg"],

        is_attract_to_center = kwargs["is_attract_to_center"],
    ),
    "reward_passive_fn": lambda kwargs: RewardPassiveTimeout(
        align_fail_panelty = kwargs["align_fail_panelty"],
        timeout_panelty = kwargs["timeout_panelty"],

        max_align_reward = kwargs["max_align_reward"],
        max_align_pos_dis_mm = kwargs["max_align_pos_dis_mm"],
        max_align_rot_dis_deg = kwargs["max_align_rot_dis_deg"],

        max_approach_reward = kwargs["max_approach_reward"],
        max_approach_pos_dis_mm = kwargs["max_approach_pos_dis_mm"],
        max_approach_rot_dis_deg = kwargs["max_approach_rot_dis_deg"],

        max_move_reward = kwargs["max_move_reward"],
        max_move_pos_dis_mm = kwargs["max_move_pos_dis_mm"],
        max_move_rot_dis_deg = kwargs["max_move_rot_dis_deg"],

        is_attract_to_center = kwargs["is_attract_to_center"],
        is_square_align_reward = kwargs["is_square_align_reward"],

        time_panelty = kwargs.get("time_panelty", -0.1)
    ),
    "maintain_cosine_lr": lambda kwargs: MaintainCosineLR(
        kwargs["lr"], kwargs["maintain_ratio"], kwargs["adjust_ratio"], kwargs["eta_min_rate"]
    )
})
