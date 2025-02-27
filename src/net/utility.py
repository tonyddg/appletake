import os
from typing import Callable, Optional, Union

from pathlib import Path

import torch
from torch import nn
from torch import optim

from torch.utils.data import DataLoader

from tqdm import tqdm
from dataclasses import dataclass

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
    return torch.mean((torch.argmax(y_predict, 1) == y_target).type(torch.float32)).item()

def get_report_dict(i: int, accurate_loss: float, accurate_acc: Optional[float]):
    report = {
        "train loss": f"{accurate_loss / (i + 1):.3f}",
    }
    if accurate_acc is not None:
        report.update({
            "train acc": f"{accurate_acc / (i + 1):.3f}"
        })
    return report

class ModelTeacher:

    @dataclass
    class AdvanceConfig:
        weight_decay: float = 5e-6
        momentum: float = 0.99
        is_use_adam: bool = True
        use_schedule: bool = False
        lr_period: int = 5
        lr_decay: float = 0.9

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
            is_init_weight: bool = True,
            advance_config: AdvanceConfig | None = None
        ):
        
        self.net = net
        if is_init_weight:
            self.net.apply(init_weights)
        self.net.to("cuda")

        save_root = Path(save_root)
        if not save_root.exists():
            os.mkdir(save_root)
        self.save_dir = save_root.joinpath(get_file_time_str())
        os.mkdir(self.save_dir)

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
        
        if self.cfg.use_schedule:
            self.schedule = optim.lr_scheduler.StepLR(self.optimizer, self.cfg.lr_period, self.cfg.lr_decay)

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
                        pbar.set_postfix(get_report_dict(i, accurate_loss, accurate_acc))
                    else:
                        pbar.set_postfix(get_report_dict(i, accurate_loss, accurate_acc))

                pbar.update(1)

        if self.cfg.use_schedule:
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
                            pbar.set_postfix(get_report_dict(i, accurate_loss, accurate_acc))
                        else:
                            pbar.set_postfix(get_report_dict(i, accurate_loss, accurate_acc))

                    pbar.update(1)

        return (accurate_acc / total_batches, accurate_loss / total_batches)

    def train(self, num_epoch):
        train_acc = np.zeros(num_epoch)
        test_acc = np.zeros(num_epoch)
        train_loss = np.zeros(num_epoch)
        test_loss = np.zeros(num_epoch)

        self.best_res = -np.inf

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
        
        except KeyboardInterrupt:
            pass

        if self.success_fn is not None:
            fig, axes = plt.subplot_mosaic('AB')
        else:
            fig, axes = plt.subplot_mosaic('B')

        axes["B"].plot(train_loss)
        axes["B"].plot(test_loss)
        axes["B"].legend(["Train Loss", "Test Loss"])

        if self.success_fn is not None:
            axes["A"].plot(train_acc)
            axes["A"].plot(test_acc)
            axes["A"].legend(["Train Acc", "Test Acc"])

        fig.savefig(self.save_dir.joinpath(f"train.png").as_posix())
        torch.save(self.net.state_dict(), self.save_dir.joinpath(f'last.pth').as_posix())
        with open(self.save_dir.joinpath(f"train_curve.npz"), "wb") as f:
            np.savez(f, train_acc = train_acc, test_acc = test_acc, train_loss = train_loss, test_loss = test_loss)
