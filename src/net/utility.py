import math
import os
from typing import Any, Callable, Dict, Literal, Optional, Union

from pathlib import Path

import torch
from torch import nn
from torch import optim

from torch.utils.data import DataLoader

from tqdm import tqdm
from dataclasses import dataclass, field

import numpy as np
import matplotlib.pyplot as plt

from ..utility import get_file_time_str

def init_weights(m: nn.Module):
    '''
    使用 Xavier 方法初始化网络参数, 防止训练不收敛
    '''
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        nn.init.xavier_uniform_(m.weight)

def cls_acc_success_fn(y_predict: torch.Tensor, y_target: torch.Tensor):
    with torch.no_grad():
        return torch.mean((torch.argmax(y_predict, 1) == y_target).type(torch.float32)).item()

def reg_mse_success_fn(y_predict: torch.Tensor, y_target: torch.Tensor):
    with torch.no_grad():
        return -nn.functional.mse_loss(y_predict, y_target).item()

def get_report_dict(is_train: bool, i: int, accurate_loss: float, accurate_acc: Optional[float]):
    epoch_type = "train" if is_train else "eval"
    
    report = {
        f"{epoch_type} loss": f"{accurate_loss / (i + 1):.3f}",
    }
    if accurate_acc is not None:
        report.update({
            f"{epoch_type} acc": f"{accurate_acc / (i + 1):.3f}"
        })
    return report

def WarmUpCosineLR(optimizer: optim.Optimizer, warmup_epoch: int, T_max: int, eta_min_rate: float):
    def get_init_lr_times(epoch: int):
        if epoch < warmup_epoch:
            return epoch / warmup_epoch
        else:
            epoch -= warmup_epoch
            return eta_min_rate + 0.5 * (1 - eta_min_rate) * (1 + math.cos((epoch / T_max) * torch.pi))

    return optim.lr_scheduler.LambdaLR(optimizer, get_init_lr_times)

class ModelTeacher:

    @dataclass
    class AdvanceConfig:
        weight_decay: float = 5e-6
        momentum: float = 0.99
        is_use_adam: bool = True
        # stepLR 小周期简单分类
        # WarmUp + CosineAnnealing 大周期复杂回归 (自实现)
        # CosineAnnealingWarmRestarts 噪音复杂回归
        schedule_type: Optional[Literal["step", "warm_cos", "restart_cos"]] = None
        schedule_kwargs: Optional[Dict[str, Any]] = None

        def __post_init__(self):
            if self.schedule_kwargs is None and self.schedule_type is not None:
                if self.schedule_type == "step":
                    self.schedule_kwargs = {
                    "step_size": 5,
                    "gamma": 0.9
                }
                else:
                    # 每个重启周期长度为 10，20, 40, ... 
                    self.schedule_kwargs = {
                    "T_0": 10,
                    "T_mult": 2
                }

    def __init__(
            self, 
            net: nn.Module,
            lr: float, 
            train_data: DataLoader,
            test_data: DataLoader,
            save_root: Union[Path, str],

            # is_cls: bool = True,
            loss_factory: Callable[[], nn.modules.loss._Loss] = nn.CrossEntropyLoss,
            # 传入批次, 输出平均, 越大越好, 传入 None 时以 -loss 代替
            # success_fn(y_predict: torch.Tensor, y_target: torch.Tensor) 
            success_fn: Optional[Callable[[torch.Tensor, torch.Tensor], float]] = None,
            # 取 init_weight = True 使用 Xavier 方法初始化网络参数, 取字符串则使用加载参数
            init_weight: Union[str, bool] = False,
            advance_config: Optional[AdvanceConfig] = None
        ):
        
        self.net = net
        if isinstance(init_weight, str):
            self.net.load_state_dict(torch.load(init_weight))
        elif init_weight:
            self.net.apply(init_weights)

        self.net.to("cuda")

        save_root = Path(save_root)
        self.save_dir = save_root.joinpath(get_file_time_str())
        os.makedirs(self.save_dir)

        if advance_config == None:
            self.cfg = ModelTeacher.AdvanceConfig()
        else:
            self.cfg = advance_config

        self.train_data = train_data
        self.test_data = test_data

        self.success_fn = success_fn
        self.loss = loss_factory()

        if self.cfg.is_use_adam:
            self.optimizer = optim.Adam(
                net.parameters(), 
                lr = lr,
                weight_decay = self.cfg.weight_decay
            )
        else:
            self.optimizer = optim.SGD(
                net.parameters(), 
                lr = lr, 
                momentum = self.cfg.momentum,
                weight_decay = self.cfg.weight_decay
            )
        
        if self.cfg.schedule_type == "step":
            if self.cfg.schedule_kwargs is None:
                self.schedule = optim.lr_scheduler.StepLR(self.optimizer, step_size = 5, gamma = 0.9)
            else:
                self.schedule = optim.lr_scheduler.StepLR(self.optimizer, **self.cfg.schedule_kwargs)
        elif self.cfg.schedule_type == "restart_cos":
            if self.cfg.schedule_kwargs is None:
                self.schedule = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                    self.optimizer, T_0 = 10, T_mult = 2, eta_min = 0.0
                )
            else:
                self.schedule = optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, **self.cfg.schedule_kwargs)
        elif self.cfg.schedule_type == "warm_cos":
            if self.cfg.schedule_kwargs is None:
                self.schedule = WarmUpCosineLR(self.optimizer, warmup_epoch = 10, T_max = 50, eta_min_rate = 0.01)
            else:
                self.schedule = WarmUpCosineLR(self.optimizer, **self.cfg.schedule_kwargs)
        else:
            self.schedule = None

    def train_epoch(self, epoches: int):
        '''
        完成一个 Epoch 的训练
        '''
        self.net.train()

        total_batches = len(self.train_data)
        accurate_acc = 0
        accurate_loss = 0

        with tqdm(total = total_batches, desc = f"Epoches {epoches} Train") as pbar:
            for i, (X, y_target) in enumerate(self.train_data):
                X = X.to("cuda")
                y_target = y_target.to("cuda")
                
                y_predict = self.net(X)
                l = self.loss(y_predict, y_target)
                
                self.optimizer.zero_grad()
                l.backward()
                self.optimizer.step()

                with torch.no_grad():
                    if self.success_fn is not None:
                        accurate_acc += self.success_fn(y_predict, y_target)
                    accurate_loss += l.item()
                
                if i % 10 == 0 or (i + 1) == total_batches:
                    if self.success_fn:
                        pbar.set_postfix(get_report_dict(True, i, accurate_loss, accurate_acc))
                    else:
                        pbar.set_postfix(get_report_dict(True, i, accurate_loss, accurate_acc))

                pbar.update(1)

        if self.schedule is not None:
            self.schedule.step()

        return (accurate_acc / total_batches, accurate_loss / total_batches)

    def val_epoch(self, epoches: int):
        '''
        完成一个 Epoch 的验证
        '''
        self.net.eval()

        total_batches = len(self.test_data)
        accurate_acc = 0
        accurate_loss = 0

        with torch.no_grad():
            with tqdm(total = total_batches, desc = f"Epoches {epoches} Test") as pbar:
                for i, (X, y_target) in enumerate(self.test_data):
                    X = X.to("cuda")
                    y_target = y_target.to("cuda")
                    
                    y_predict = self.net(X)
                    l = self.loss(y_predict, y_target)

                    if self.success_fn is not None:
                        accurate_acc += self.success_fn(y_predict, y_target)
                    accurate_loss += l.item()
                
                    if i % 10 == 0 or (i + 1) == total_batches:
                        if self.success_fn:
                            pbar.set_postfix(get_report_dict(False, i, accurate_loss, accurate_acc))
                        else:
                            pbar.set_postfix(get_report_dict(False, i, accurate_loss, accurate_acc))

                    pbar.update(1)

        return (accurate_acc / total_batches, accurate_loss / total_batches)

    def train(self, num_epoch):
        
        train_acc = np.zeros(num_epoch)
        test_acc = np.zeros(num_epoch)
        train_loss = np.zeros(num_epoch)
        test_loss = np.zeros(num_epoch)

        self.best_res = -np.inf
        effect_idx = 0

        try:
            for i in range(num_epoch):
                train_acc[i], train_loss[i] = self.train_epoch(i)
                test_acc[i], test_loss[i] = self.val_epoch(i)

                cur_res = -test_loss[i]
                if self.success_fn is not None:
                    cur_res = test_acc[i]

                if self.best_res < cur_res:
                    print("Save best param")
                    self.best_res = cur_res
                    torch.save(self.net.state_dict(), self.save_dir.joinpath(f'best.pth').as_posix())

                effect_idx += 1

        except KeyboardInterrupt:
            pass

        if effect_idx > 0:
            train_acc = train_acc[:effect_idx]
            test_acc = test_acc[:effect_idx]
            train_loss = train_loss[:effect_idx]
            test_loss = test_loss[:effect_idx]

            if self.success_fn is not None:
                fig, axes = plt.subplot_mosaic('AB')
            else:
                fig, axes = plt.subplot_mosaic('B')

            axes["B"].plot(train_loss)
            axes["B"].plot(test_loss)
            axes["B"].legend(["Train Loss", "Test Loss"])
            if self.success_fn is not None:
                axes["B"].set_title(f"last test loss: {test_loss[-1]:.3e}")
            else:
                axes["B"].set_title(f"best test loss: {-self.best_res:.3e}")
            
            if self.success_fn is not None:
                axes["A"].plot(train_acc)
                axes["A"].plot(test_acc)
                axes["A"].legend(["Train Acc", "Test Acc"])
                axes["A"].set_title(f"best acc: {self.best_res:.3e}")

            fig.savefig(self.save_dir.joinpath(f"train.png").as_posix())
            with open(self.save_dir.joinpath(f"train_curve.npz"), "wb") as f:
                np.savez(f, train_acc = train_acc, test_acc = test_acc, train_loss = train_loss, test_loss = test_loss)
