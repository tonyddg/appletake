from math import ceil
from typing import Any, Callable, List, Literal, Optional, Protocol, Union
from torch import nn

import torch
import torchsummary

from .net_abc import BackboneModule, BackboneWithHead

def _make_divisible(ch: float, divisor: int = 8, min_ch: Optional[int] = None):
    '''
    保证通道数可以被 8 整除  

    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    '''
    if min_ch is None:
        min_ch = divisor
    
    new_ch = max(
        min_ch, 
        int(ch + divisor / 2) // divisor * divisor
    )
    # Make sure that round down does not go down by more than 10%.
    if new_ch < 0.9 * ch:
        new_ch += divisor
    return new_ch

def drop_path(x, drop_prob: Optional[float] = None, training: bool = False):
    '''
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    "Deep Networks with Stochastic Depth", https://arxiv.org/pdf/1603.09382.pdf

    It can be seen here:
    https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/drop.py#L140
    '''
    if drop_prob is None or not training:     
        return x
    
    keep_prob = 1 - drop_prob       # 保留的比率
    # 将单个 Batch 中随机几个样本置 0, 似乎不如 DropOut ?
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # 利用 floor 判断是否大于 1 实现二值化

    # 保证均值不变
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):
    def __init__(self, drop_prob: Optional[float] = None) -> None:
        super().__init__()
        self.drop_prob = drop_prob
    
    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

class ConvBNSiLU(nn.Module):
    def __init__(
            self,
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
            nn.BatchNorm2d(out_channel)
        )

        if is_use_active_fn:
            self.net.append(nn.SiLU(True))
    
    def forward(self, X):
        return self.net(X)

class SEBlock(nn.Module):
    def __init__(
            self,
            in_channel: int,
            base_channel: int,
            se_factor: float
        ) -> None:
        '''
        squeeze excitation 模块
        * 类似自注意力机制, 通过 squeeze_excitation_path 计算特征中各个通道的权重
        * squeeze_excitation_path 中首先使用 AdaptiveAvgPool2d 将各个通道特征变为标量 (squeeze) 
        * 使用 1x1 卷积 (类似全连接) 混合不同通道特征, 先缩小特征数再扩张为权重 (excitation) 
        * 扩张为权重时, 使用了 Sigmoid 调整权重之和为 1, 不会影响特征的均值
        '''
        super().__init__()

        # 原因未知, squeeze_channel 基于 MBConv 的 in_channel 而不是 SE 块的实际 in_channel
        squeeze_channel = max(int(base_channel * se_factor), 1)
        self.squeeze_excitation_path = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            # 1x1 卷积与全连接在数学上等价, 不使用偏置, 防止破坏通道均衡性
            nn.Conv2d(
                in_channel, squeeze_channel, (1, 1), bias = False
            ),
            nn.SiLU(True),
            nn.Conv2d(
                squeeze_channel, in_channel, (1, 1), bias = False
            ),
            nn.Sigmoid()
        )
    
    def forward(self, X):
        scale = self.squeeze_excitation_path(X)
        return scale * X
    
class MBConvBlock(nn.Module):
    def __init__(
            self,
            kernel: int,             # 3 or 5     两种可能
            in_channel: int,         # 输入MBConv的channel数
            out_channel: int,        # MBConv输出的channel数
            expanded_ratio: int,     # 1 or 6     变胖倍数
            stride: int,             # 1 or 2
            drop_rate: float,        # MBConv中的随机失活比例
        ) -> None:
        super().__init__()

        in_channel = in_channel
        out_channel = out_channel
        exp_channel = in_channel * expanded_ratio
        # 仅当 stride == 1 且输入输出通道相同时使用残差连接
        self.is_use_res_shortcut_path = ((stride == 1) and (in_channel == out_channel))

        self.main_path = nn.Sequential()

        # 1x1 升维卷积 (尽在需要提高维度时使用)
        if expanded_ratio != 1:
            self.main_path.append(
                ConvBNSiLU(
                    in_channel, exp_channel, 1
                ),
            )
        # kxk 逐层卷积
        self.main_path.append(
            ConvBNSiLU(
                exp_channel, exp_channel, kernel, stride, groups = exp_channel
            ),
        )
        # SEBlock, se_ratio 固定为 0.25
        self.main_path.append(
            SEBlock(
                exp_channel, in_channel, 0.25
            ),
        )
        # 1x1 映射降维卷积 (不使用激活函数)
        self.main_path.append(
            ConvBNSiLU(
                exp_channel, out_channel, 1, is_use_active_fn = False
            ),
        )
        # 通过 DropPath类 实现 MBConv 中的 dropout (仅在残差连接时使用)
        if self.is_use_res_shortcut_path and drop_rate > 0:
            self.main_path.append(DropPath(drop_rate))  

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        if self.is_use_res_shortcut_path:
            return self.main_path(X) + X
        else:
            return self.main_path(X)

class FusedMBConvBlock(nn.Module):
    def __init__(
            self,
            kernel: int,             # 3      (保持接口同一)
            in_channel: int,         # 输入MBConv的channel数
            out_channel: int,        # MBConv输出的channel数
            expanded_ratio: int,     # 1 or 6     变胖倍数
            stride: int,             # 1 or 2
            drop_rate: float,        # MBConv中的随机失活比例
        ) -> None:
        '''
        EfficientNetV2 使用的 FusedMBConvBlock
        虽然提高了参数量, 但可以解决 MBConvBlock 在特征尺寸较大时逐层卷积与 SEBlock 计算慢的问题
        '''
        super().__init__()

        in_channel = in_channel
        out_channel = out_channel
        exp_channel = in_channel * expanded_ratio
        # 仅当 stride == 1 且输入输出通道相同时使用残差连接
        self.is_use_res_shortcut_path = ((stride == 1) and (in_channel == out_channel))

        self.main_path = nn.Sequential()

        # 3x3 普通卷积代替升维与逐层卷积
        self.main_path.append(
            ConvBNSiLU(
                in_channel, exp_channel, kernel, stride
            ),
        )
        # 1x1 映射降维卷积 (膨胀系数为 1 时不再降维)
        if expanded_ratio != 1:
            self.main_path.append(
                ConvBNSiLU(
                    exp_channel, out_channel, 1, is_use_active_fn = False
                ),
            )
        # 通过 DropPath类 实现 MBConv 中的 dropout (仅在残差连接时使用)
        if self.is_use_res_shortcut_path and drop_rate > 0:
            self.main_path.append(DropPath(drop_rate))  

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        if self.is_use_res_shortcut_path:
            return self.main_path(X) + X
        else:
            return self.main_path(X)

class MBConvStage(nn.Module):
    def __init__(
            self,
            mbconv_block_type: Union[type[MBConvBlock], type[FusedMBConvBlock]],
            kernal: int, 
            in_channel: int,
            out_channel: int, 
            expand_ratio: int, 
            stride: int, 
            repeate: int,
            base_drop_rate: float,
            start_drop_rate: float
        ) -> None:
        super().__init__()

        self.stage = nn.Sequential()
        for i in range(repeate):
            if i == 0:
                self.stage.append(mbconv_block_type(
                    kernal, in_channel, out_channel, expand_ratio, stride, start_drop_rate
                ))
            else:
                self.stage.append(mbconv_block_type(
                    kernal, out_channel, out_channel, expand_ratio, 1, start_drop_rate
                ))       

            start_drop_rate += base_drop_rate

    def forward(self, X):
        res = self.stage(X)
        # print(res.shape)
        return res

class EfficientNetV1Backbone(BackboneModule):
    def __init__(self,
                 in_channel: int,
                 width_coefficient: float,          # 宽度倍率因子 (特征通道数)
                 depth_coefficient: float,          # 深度倍率因子 (Stage 深度)
                 drop_connect_rate: float = 0.2,    # 控制 SE 模块里的 dropout
                 ):
        super().__init__()

        stem_out_channel = 32
        head_out_feat = 1280
        # kernal, out_channel, expand_ratio, stride, repeate
        stage_cnf = [[3, 16 , 1, 1, 1],
                     [3, 24 , 6, 2, 2],
                     [5, 40 , 6, 2, 2],
                     [3, 80 , 6, 2, 3],
                     [5, 112, 6, 1, 3],
                     [5, 192, 6, 2, 4],
                     [3, 320, 6, 1, 1]]
        
        # 基于两个因子调整基础配置
        _num_blocks = 0
        for i in range(len(stage_cnf)):
            stage_cnf[i][1] = _make_divisible(stage_cnf[i][1] * width_coefficient, 8)
            stage_cnf[i][4] = int(ceil(stage_cnf[i][4] * depth_coefficient))
            _num_blocks += stage_cnf[i][4]

        base_drop_rate: float = drop_connect_rate / _num_blocks
        start_drop_rate: float = 0

        _stage_out_channel = _make_divisible(stem_out_channel * width_coefficient)
        self.stem = ConvBNSiLU(
            in_channel, _stage_out_channel,
            kernel_size = 3, stride = 2
        )

        self.body = nn.Sequential()
        for cnf in stage_cnf:
            self.body.append(
                MBConvStage(
                    MBConvBlock, cnf[0], _stage_out_channel, cnf[1], cnf[2], cnf[3], cnf[4], base_drop_rate, start_drop_rate
                )
            )
            # drop out 根据块位置逐步增加
            start_drop_rate += base_drop_rate * cnf[4]
            _stage_out_channel = cnf[1]

        self.out_feat_size = _make_divisible(head_out_feat * width_coefficient)
        self.head = nn.Sequential(
            ConvBNSiLU(
                _stage_out_channel, self.out_feat_size ,
                1, 1
            ),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode = "fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, X):
        X = self.stem(X)
        X = self.body(X)
        return self.head(X)
    
    def get_out_feat_size(self):
        return self.out_feat_size 

EFFNET_V2_STAGE_CNF = {
    # is_fused, out_channel, expand_ratio, stride, repeate
    "s": { # 384
            "stage_cnf" : [[FusedMBConvBlock, 24 , 1, 1, 2],
                           [FusedMBConvBlock, 48 , 4, 2, 4],
                           [FusedMBConvBlock, 64 , 4, 2, 4],
                           [MBConvBlock     , 128, 4, 2, 6],
                           [MBConvBlock     , 160, 6, 1, 9],
                           [MBConvBlock     , 256, 6, 2, 15]],
            "stem_out_ch": 24,
            "drop_connect_rate": 0.2
        },

    "m": { # 384
            "stage_cnf" : [[FusedMBConvBlock, 24 , 1, 1, 3],
                           [FusedMBConvBlock, 48 , 4, 2, 5],
                           [FusedMBConvBlock, 80 , 4, 2, 5],
                           [MBConvBlock     , 160, 4, 2, 7],
                           [MBConvBlock     , 176, 6, 1, 14],
                           [MBConvBlock     , 304, 6, 2, 18],
                           [MBConvBlock     , 512, 6, 1, 5]],
            "stem_out_ch": 24,
            "drop_connect_rate": 0.3
        },

    "l": { # 384
            "stage_cnf" : [[FusedMBConvBlock, 32 , 1, 1, 4],
                           [FusedMBConvBlock, 64 , 4, 2, 7],
                           [FusedMBConvBlock, 96 , 4, 2, 7],
                           [MBConvBlock     , 192, 4, 2, 10],
                           [MBConvBlock     , 224, 6, 1, 19],
                           [MBConvBlock     , 384, 6, 2, 25],
                           [MBConvBlock     , 640, 6, 1, 7]],
            "stem_out_ch": 32,
            "drop_connect_rate": 0.4,
        },

    "xl":{ # 384
            "stage_cnf" : [[FusedMBConvBlock, 32 , 1, 1, 4],
                           [FusedMBConvBlock, 64 , 4, 2, 7],
                           [FusedMBConvBlock, 96 , 4, 2, 7],
                           [MBConvBlock     , 192, 4, 2, 10],
                           [MBConvBlock     , 224, 6, 1, 19],
                           [MBConvBlock     , 384, 6, 2, 25],
                           [MBConvBlock     , 640, 6, 1, 7]],
            "stem_out_ch": 32,
            "drop_connect_rate": 0.4
        },
}

class EfficientNetV2Backbone(BackboneModule):
    def __init__(self,
                 in_channel: int,
                 stage_cnf_name: Literal['s', 'm', 'l', 'xl'],
                #  stem_out_ch: int,
                #  drop_connect_rate: float = 0.2,    # 控制 SE 模块里的 dropout
                 ):
        super().__init__()

        stage_cnf = EFFNET_V2_STAGE_CNF[stage_cnf_name]["stage_cnf"]
        stem_out_ch = EFFNET_V2_STAGE_CNF[stage_cnf_name]["stem_out_ch"]
        drop_connect_rate = EFFNET_V2_STAGE_CNF[stage_cnf_name]["drop_connect_rate"]

        stem_out_channel = stem_out_ch
        head_out_feat = 1280

        # 基于两个因子调整基础配置
        _num_blocks = 0
        for i in range(len(stage_cnf)):
            stage_cnf[i][1] = stage_cnf[i][1]
            stage_cnf[i][4] = stage_cnf[i][4]
            _num_blocks += stage_cnf[i][4]

        base_drop_rate: float = drop_connect_rate / _num_blocks
        start_drop_rate: float = 0

        _stage_out_channel = stem_out_channel
        self.stem = ConvBNSiLU(
            in_channel, _stage_out_channel,
            kernel_size = 3, stride = 2
        )

        self.body = nn.Sequential()
        for cnf in stage_cnf:
            self.body.append(
                MBConvStage(
                    cnf[0], 3, _stage_out_channel, cnf[1], cnf[2], cnf[3], cnf[4], base_drop_rate, start_drop_rate
                )
            )
            # drop out 根据块位置逐步增加
            start_drop_rate += base_drop_rate * cnf[4]
            _stage_out_channel = cnf[1]

        self.out_feat_size = head_out_feat
        self.head = nn.Sequential(
            ConvBNSiLU(
                _stage_out_channel, self.out_feat_size ,
                1, 1
            ),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode = "fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, X):
        X = self.stem(X)
        X = self.body(X)
        return self.head(X)
    
    def get_out_feat_size(self):
        return self.out_feat_size 

if __name__ == "__main__":
    # net = BackboneWithHead(
    #     EfficientNetV2Backbone(
    #         3, **EFFNET_V2_STAGE_CNF["s"] # type: ignore
    #     ), 1000, 0.2
    # )
    net = BackboneWithHead(
        EfficientNetV1Backbone(
            3, 1, 1, 0.2
        ), 6, 0.2
    )
    # net = BackboneWithHead(
    #     EfficientNetV1Backbone(
    #         3, 1.1, 1.2, 0.2
    #     ), 6, 0.3
    # )
    net.review((3, 224, 224))
