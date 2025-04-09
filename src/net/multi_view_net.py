from math import ceil
from typing import Callable
import torch
from torch import nn

from .net_abc import BackboneModule, BackboneWithHead
from .efficient_net import _make_divisible, ConvBNSiLU, MBConvStage, MBConvBlock, SEBlock, EfficientNetV1Backbone

class SeperateLateFuseNet(BackboneModule):
    def __init__(
        self,
        num_seperate_img: int,
        make_origin_backbone: Callable[[], BackboneModule],
        expand_rate: float = 1,
        fuse_dropout: float = 0.5
    ):
        super().__init__()
        self.num_seperate_img = num_seperate_img
        self.backbone_list = nn.ModuleList([
            make_origin_backbone() for _ in range(num_seperate_img)
        ])

        self.out_feat_size = int(expand_rate * self.backbone_list[0].get_out_feat_size()) # type: ignore
        self.fuse_layer = nn.Sequential(
            nn.Dropout(fuse_dropout, True),
            nn.Linear(
                num_seperate_img * self.backbone_list[0].get_out_feat_size(), # type: ignore
                self.out_feat_size
            ),
            nn.SiLU(True), 
        )

        nn.init.normal_(self.fuse_layer[1].weight, 0, 0.01) # type: ignore
        nn.init.zeros_(self.fuse_layer[1].bias) # type: ignore

    def get_out_feat_size(self):
        return self.out_feat_size

    def forward(self, X):
        concat_feat = torch.concat(
            [self.backbone_list[i](torch.unsqueeze(X[:, i, :, :], 1)) for i in range(X.shape[1])], dim = 1            
        )
        return self.fuse_layer(concat_feat)

if __name__ == "__main__":
    # net = BackboneWithHead(
    #     SEBlendEffNet(
    #         3, 3, 1, 1, 0.2
    #     ), 1000, 0.2
    # )
    net = BackboneWithHead(
        SeperateLateFuseNet(
            3, lambda: EfficientNetV1Backbone(1, 1, 1, 0.2)
        ), 6, 0.2
    )
    net.review((3, 224, 224))
