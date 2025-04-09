from typing import Tuple
import torch
from torch import nn
import torchsummary
from ptflops import get_model_complexity_info
from abc import ABCMeta, abstractmethod

class BackboneModule(nn.Module, metaclass = ABCMeta):
    @abstractmethod
    def get_out_feat_size(self) -> int:
        raise NotImplementedError()

class BackboneWithHead(nn.Module):
    def __init__(self,
                 backbone: BackboneModule,
                 num_classes: int,
                 head_drop_out: float = 0.2,
                 ):
        super().__init__()
        self.backbone = backbone
        self.cls_head = nn.Sequential(
            nn.Dropout(head_drop_out, True),
            nn.Linear(
                self.backbone.get_out_feat_size(), num_classes
            )
        )

        nn.init.normal_(self.cls_head[1].weight, 0, 0.01) # type: ignore
        nn.init.zeros_(self.cls_head[1].bias) # type: ignore

    def forward(self, X):
        X = self.backbone(X)
        # print(X.shape)
        return self.cls_head(X)

    def review(self, in_size: Tuple[int, int, int]):
        device = str(next(self.parameters()).device)
        torchsummary.summary(self, in_size, device = device)

        macs, params = get_model_complexity_info(self, in_size, as_strings = True, print_per_layer_stat = False)

        print(f"模型 FLOPs: {macs}")
        print(f"模型参数量: {params}")