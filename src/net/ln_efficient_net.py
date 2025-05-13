from math import ceil
from typing import Any, Callable, List, Literal, Optional, Protocol, Union
from torch import nn

import torch
import torchsummary

from net_abc import BackboneModule, BackboneWithHead

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

class StageCNN(BackboneModule):

    def __init__(
        self,
        observation_space: tuple,
        # cnn_stages: int,
        # init_feat: int = 32,
        stage_feats: List[int] = [64, 192, 320, 640], # [32, 80, 192, 320, 1280],
        stage_depth: List[int] = [2, 2, 2],
        normalized_image: bool = False,
    ) -> None:
        
        super().__init__()
        self.features_dim = stage_feats[-1] # feat_dim
        
        n_input_channels = observation_space[0]
        n_input_width = observation_space[1]

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
            # nn.Flatten(),
            # nn.Linear(n_flatten, feat_dim),
            # nn.ReLU(True)
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        X = self.head(observations)
        X = self.backbone(X)
        return self.stem(X)
    
    def get_out_feat_size(self):
        return self.features_dim

if __name__ == "__main__":
    net = BackboneWithHead(
        StageCNN(
            (3, 224, 224)
        ), 6, 0.2
    )
    net.review((3, 224, 224))
