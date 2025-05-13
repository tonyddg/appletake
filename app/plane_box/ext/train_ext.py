import ast
import os
import sys
from typing import Dict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

from src.net.pr_task_dataset import PrEnvRenderDataset, sample_test
from src.net.utility import ModelTeacher, reg_mse_success_fn

from src.sb3.exp_manager import config_data_exec, config_to_env, load_exp_config, config_data_replace
from src.gym_env.plane_box.corner import CornerEnv
from src.gym_env.plane_box.paralle import ParalleEnv
from src.gym_env.plane_box.three import ThreeEnv

from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torch import nn
import numpy as np

from conf.arg_dict import exp_exec_arg_dict, exp_replace_arg_dict
from src.net.net_abc import BackboneWithHead
from src.net.efficient_net import EfficientNetV1Backbone
from src.net.multi_view_net import SeperateLateFuseNet

import argparse

# TODO 设计实验证明强化学习方法优于直接神经网络预测

if __name__ == "__main__":

    parser = argparse.ArgumentParser(prog = None, description = None, epilog = None)
    parser.add_argument("--env_type", default = "three", choices = ["three", "paralle", "corner"], nargs = "?")
    parser.add_argument("--env_vary", default = "train", type = str, nargs = "?")
    
    parser.add_argument("--net_type", default = "eff", choices = ["eff", "ftb", "eff_b2"], nargs = "?")
    parser.add_argument("--lr_schedule", default = "warmup", choices = ["warmup", "restart", "constant"], nargs = "?")
    
    parser.add_argument("--num_worker_train", default = 3, type = int, nargs = "?")
    parser.add_argument("--num_worker_eval", default = 3, type = int, nargs = "?")
    
    parser.add_argument("--batch_size", default = 64, type = int, nargs = "?")
    parser.add_argument("--lr", default = 1e-2, type = float, nargs = "?")
    parser.add_argument("--weight_decay", default = 5e-6, type = float, nargs = "?")
    parser.add_argument("--eta_min_rate", default = 1e-3, type = float, nargs = "?")
    parser.add_argument("--epoch_scale", default = 1, type = float, nargs = "?")
    parser.add_argument("--init_weight", default = True, nargs = "?")
    parser.add_argument("--phase_type", default = "default", nargs = "?")
    parser.add_argument("--custom_conf", default = "{}", type = str)
    parser.add_argument("--suffix_name", default = "ext", type = str)

    parser.add_argument("--in_channel", default = 3, type = int, nargs = "?")

    res = parser.parse_args()

    replace_arg_dict = exp_replace_arg_dict()
    env_type = res.env_type
    env_vary = res.env_vary

    net_type = res.net_type
    lr_schedule = res.lr_schedule

    # num_worker = res.num_worker
    num_worker_train = res.num_worker_train
    num_worker_eval = res.num_worker_eval

    batch_size = res.batch_size
    init_weight = res.init_weight
    phase_type = res.phase_type
    eta_min_rate = res.eta_min_rate
    custom_conf = dict(ast.literal_eval(res.custom_conf))
    suffix_name = res.suffix_name

    lr = res.lr
    weight_decay = res.weight_decay
    epoch_scale = res.epoch_scale

    in_channel = res.in_channel

    print(res)

    cls_dict = {
        "three": ThreeEnv,
        "corner": CornerEnv,
        "paralle": ParalleEnv
    }

    train_cfg_path = f"app/plane_box/conf/base_{env_vary}_env.yaml"
    if net_type == "eff_b2":
        train_cfg_path = "app/plane_box/conf/base_train_b2_env.yaml"

    cfg = load_exp_config(
        train_cfg_path,
        f"app/plane_box/conf/{env_type}_train_env_{phase_type}.yaml",
        is_resolve = True,
        merge_dict = custom_conf
    )
    if cfg.eval_env.get("is_base_on_train", False):
        cfg.eval_env = OmegaConf.merge(cfg.train_env, cfg.eval_env)
    cfg = OmegaConf.to_container(cfg)

    assert isinstance(cfg, Dict)
    cfg = config_data_replace(cfg, exp_replace_arg_dict())
    cfg = config_data_exec(cfg, exp_exec_arg_dict())

    train_env_kwargs = cfg["train_env"]["kwargs"] # type: ignore
    eval_env_kwargs = cfg["eval_env"]["kwargs"] # type: ignore

    train_dataset = PrEnvRenderDataset(
        cls_dict[env_type], "scene/plane_box/base_vec6.ttt", train_env_kwargs, num_epoch_data = 51200
    )
    train_dl = DataLoader(train_dataset, batch_size, num_workers = num_worker_train)

    test_dataset = PrEnvRenderDataset(
        cls_dict[env_type], "scene/plane_box/base_vec6.ttt", eval_env_kwargs, num_epoch_data = 5120
    )
    test_dl = DataLoader(test_dataset, batch_size, num_workers = num_worker_eval)

    # in_channel = 3
    # if env_type == "corner":
    #     in_channel = 1

    match net_type:
        case "eff":
            net = BackboneWithHead(
                EfficientNetV1Backbone(
                    in_channel, 1, 1, 0.2
                ), 6, 0.2
            )
        case "eff_b2":
            net = BackboneWithHead(
                EfficientNetV1Backbone(
                    in_channel, 1.1, 1.2, 0.2
                ), 6, 0.3
            )
        case "ftb":
            net = BackboneWithHead(
                SeperateLateFuseNet(
                    in_channel, lambda: EfficientNetV1Backbone(1, 1, 1, 0.2)
                ), 6, 0.2
            )
        case _:
            raise Exception()

    # 预热余弦

    match lr_schedule:
        case "warmup":
            cfg = ModelTeacher.AdvanceConfig(
                weight_decay = weight_decay,
                schedule_type = "warm_cos",
                schedule_kwargs = {
                    # 论文 https://arxiv.org/pdf/1812.01187
                    "warmup_epoch": 5, 
                    "T_max": int(50 * epoch_scale), 
                    "eta_min_rate": eta_min_rate
                },
                is_use_adam = True
            )
            epoch = cfg.schedule_kwargs["warmup_epoch"] + cfg.schedule_kwargs["T_max"] # type: ignore
        case "restart":
            cfg = ModelTeacher.AdvanceConfig(
                weight_decay = weight_decay,
                schedule_type = "restart_cos",
                schedule_kwargs = {
                    "T_0": int(10 * epoch_scale), 
                    "T_mult": 2, 
                    "eta_min": eta_min_rate
                },
                is_use_adam = True
            )
            epoch = int(10 * epoch_scale) * 7
        case "constant":
            cfg = ModelTeacher.AdvanceConfig(
                weight_decay = weight_decay,
                schedule_type = None,
                schedule_kwargs = None,
                is_use_adam = True
            )
            epoch = int(50 * epoch_scale)
        case _:
            raise Exception()

    # 实验 1
    mt = ModelTeacher(
        net, 
        lr, 
        train_dl, 
        test_dl, 
        f"./runs/plane_box/ext/{env_type}_{net_type}_{lr_schedule}_{lr:.0e}_{suffix_name}", 
        nn.SmoothL1Loss,
        reg_mse_success_fn, 
        advance_config = cfg,
        init_weight = init_weight
    )
    mt.train(epoch)
