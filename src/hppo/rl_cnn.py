from typing import List
import torch
from torch import nn
import torchsummary

import gymnasium as gym
import torch as th
from gymnasium import spaces
from torch import nn

from stable_baselines3.common.preprocessing import get_flattened_obs_dim, is_image_space
from stable_baselines3.common.type_aliases import TensorDict
from stable_baselines3.common.utils import get_device
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class ResidualBlock(nn.Module):
    def __init__(
            self,
            num_channels: int
        ):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(num_channels, num_channels, 3, 1, (3 - 1) // 2),
            nn.ReLU(True),
            nn.Conv2d(num_channels, num_channels, 3, 1, (3 - 1) // 2),
        )
        pass

    def forward(self, X):
        return self.backbone(X) + X

class NetStage(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int
        ):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, (3 - 1) // 2),
            nn.MaxPool2d(3, 2, (3 - 1) // 2),
            ResidualBlock(out_channels),
            ResidualBlock(out_channels)
        )

    def forward(self, X):
        return self.backbone(X)

class RlCNN(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: gym.Space,
        # in_channels: int,
        channel_list: List[int],
        is_full_cnn: bool,
        features_dim: int,
        normalized_image: bool = False,
    ) -> None:
        
        assert isinstance(observation_space, spaces.Box), (
            "NatureCNN must be used with a gym.spaces.Box ",
            f"observation space, not {observation_space}",
        )
        features_dim = features_dim
        super().__init__(observation_space, features_dim)
        assert is_image_space(observation_space, check_channels=False, normalized_image=normalized_image), (
            "You should use NatureCNN "
            f"only with images not with {observation_space}\n"
            "(you are probably using `CnnPolicy` instead of `MlpPolicy` or `MultiInputPolicy`)\n"
            "If you are using a custom environment,\n"
            "please check it using our env checker:\n"
            "https://stable-baselines3.readthedocs.io/en/master/common/env_checker.html.\n"
            "If you are using `VecNormalize` or already normalized channel-first images "
            "you should pass `normalize_images=False`: \n"
            "https://stable-baselines3.readthedocs.io/en/master/guide/custom_env.html"
        )

        self.out_feat = features_dim

        # 根据最小维度确定通道数
        in_channels = min(*observation_space.shape)
        backbone_layers = []
        for stage_channel in channel_list:
            backbone_layers.append(
                NetStage(
                    in_channels,
                    stage_channel
                )
            )
            in_channels = stage_channel
        self.backbone = nn.Sequential(*backbone_layers)

        if not is_full_cnn:
            with torch.no_grad():
                backbone_shape = self.backbone(torch.rand(observation_space.shape).unsqueeze(0)).shape
                flatten_size = backbone_shape[1] * backbone_shape[2] * backbone_shape[3]

            self.head = nn.Sequential(
                nn.ReLU(True),
                nn.Flatten(),
                nn.Linear(flatten_size, features_dim)
            )
        else:
            self.head = nn.Sequential(
                nn.ReLU(True),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(in_channels, features_dim)
            )

    def forward(self, X):
        X = self.backbone(X)
        return self.head(X)

# if __name__ == "__main__":
#     net = BackboneWithHead(
#         RlCNN(
#             (3, 224, 224),
#             3, [16, 32, 64, 128, 320], True, 1280
#         ), 6, 0.2
#     )
#     net.review((3, 224, 224))
