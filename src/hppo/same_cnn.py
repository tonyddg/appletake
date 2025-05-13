from typing import List, Optional, Union

import gymnasium as gym
import torch as th
from gymnasium import spaces
from torch import nn

from stable_baselines3.common.preprocessing import get_flattened_obs_dim, is_image_space
from stable_baselines3.common.type_aliases import TensorDict
from stable_baselines3.common.utils import get_device
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from ..net.net_abc import BackboneModule, BackboneWithHead

class ConvLNReLU(nn.Module):
    def __init__(
            self,
            in_width: int,
            in_channel: int,
            out_channel: int,
            kernel_size: int = 3,
            stride: int = 1,
            groups: int = 1,
            is_use_active_fn: bool = True
        ) -> None:
        super().__init__()

        padding = (kernel_size - 1) // 2
        self.net = nn.Sequential(
            nn.Conv2d(
                in_channel, out_channel, kernel_size,
                stride, padding, 
                groups = groups, bias = False
            ),
            nn.LayerNorm([out_channel, in_width // stride, in_width // stride])
        )

        if is_use_active_fn:
            self.net.append(nn.ReLU(True))
    
    def forward(self, X):
        return self.net(X)

class StageCNN(BaseFeaturesExtractor, BackboneModule):

    def __init__(
        self,
        observation_space: gym.Space,
        # cnn_stages: int,
        # init_feat: int = 32,
        # 10,348,166
        stage_feats: List[int] = [32, 80, 192, 320, 640], # [32, 80, 192, 320, 1280],
        stage_depth: List[int] = [2, 2, 2, 2],
        normalized_image: bool = False,
    ) -> None:

        assert isinstance(observation_space, spaces.Box), (
            "StageCNN must be used with a gym.spaces.Box ",
            f"observation space, not {observation_space}",
        )
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        assert is_image_space(observation_space, check_channels=False, normalized_image=normalized_image), (
            "You should use StageCNN "
            f"only with images not with {observation_space}\n"
            "(you are probably using `CnnPolicy` instead of `MlpPolicy` or `MultiInputPolicy`)\n"
            "If you are using a custom environment,\n"
            "please check it using our env checker:\n"
            "https://stable-baselines3.readthedocs.io/en/master/common/env_checker.html.\n"
            "If you are using `VecNormalize` or already normalized channel-first images "
            "you should pass `normalize_images=False`: \n"
            "https://stable-baselines3.readthedocs.io/en/master/guide/custom_env.html"
        )
        
        features_dim = stage_feats[-1]
        super().__init__(observation_space, features_dim)

        n_input_channels = observation_space.shape[0]
        n_input_width = observation_space.shape[1]

        # self.head = nn.Sequential(
        #     nn.Conv2d(n_input_channels, stage_feats[0], 8, 4, 3),
        #     nn.LayerNorm([stage_feats[0], n_input_width // 2, n_input_width // 2]), nn.ReLU(True),
        #     # nn.MaxPool2d(3, 2, 1)
        # )
        self.head = ConvLNReLU(
            n_input_width, n_input_channels, stage_feats[0], 8, 4
        )

        # n_input_channels = init_feat
        n_input_width = n_input_width // 4

        backbone_layers = []
        for i in range(len(stage_feats) - 1):
            for j in range(stage_depth[i]):
                if j == 0:
                    backbone_layers.append(
                        ConvLNReLU(n_input_width, stage_feats[i], stage_feats[i + 1], stride = 2)
                    )
                else:
                    backbone_layers.append(
                        ConvLNReLU(n_input_width // 2, stage_feats[i + 1], stage_feats[i + 1])
                    )
            n_input_width = n_input_width // 2
            # n_input_channels = n_input_channels * 2
        self.backbone = nn.Sequential(*backbone_layers)

        self.stem = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        X = self.head(observations)
        X = self.backbone(X)
        return self.stem(X)
    
    def get_out_feat_size(self):
        return self.features_dim
