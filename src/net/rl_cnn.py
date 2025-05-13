from typing import List
import torch
from torch import nn
import torchsummary

from net_abc import BackboneModule, BackboneWithHead

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

class RlCNN(BackboneModule):
    def __init__(
        self,
        obs_shape: tuple,
        in_channels: int,
        channel_list: List[int],
        is_full_cnn: bool,
        out_feat: int
    ) -> None:
        super().__init__()

        self.out_feat = out_feat

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
                backbone_shape = self.backbone(torch.rand(obs_shape).unsqueeze(0)).shape
                flatten_size = backbone_shape[1] * backbone_shape[2] * backbone_shape[3]

            self.head = nn.Sequential(
                nn.ReLU(True),
                nn.Flatten(),
                nn.Linear(flatten_size, out_feat)
            )
        else:
            self.head = nn.Sequential(
                nn.ReLU(True),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(in_channels, out_feat)
            )

    def forward(self, X):
        X = self.backbone(X)
        return self.head(X)

    def get_out_feat_size(self):
        return self.out_feat

if __name__ == "__main__":
    net = BackboneWithHead(
        RlCNN(
            (3, 224, 224),
            3, [16, 32, 64, 128, 256], True, 1280
        ), 1000, 0.2
    )
    net.review((3, 224, 224))
