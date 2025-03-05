import os
import sys
from typing import Dict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.net.pr_task_dataset import PrEnvRenderDataset, sample_test
from src.net.utility import ModelTeacher, reg_mse_success_fn
from src.net.efficient_net import EfficientNetV1WithHead

from src.sb3.exp_manager import config_to_env, load_exp_config, config_data_replace
from src.gym_env.plane_box.corner import CornerEnv

from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torch import nn
import numpy as np

from finetuning.arg_dict import exp_exec_arg_dict, exp_replace_arg_dict

# TODO 设计实验证明强化学习方法优于直接神经网络预测

if __name__ == "__main__":

    replace_arg_dict = exp_replace_arg_dict()

    cfg = OmegaConf.to_container(load_exp_config("app/plane_box/conf/corner_trainext_vec_env.yaml"))
    assert isinstance(cfg, Dict)
    cfg = config_data_replace(cfg, replace_arg_dict)

    train_env_kwargs = cfg["train_env"]["kwargs"] # type: ignore
    eval_env_kwargs = cfg["eval_env"]["kwargs"] # type: ignore

    train_dataset = PrEnvRenderDataset(
        CornerEnv, "scene/plane_box/corner_vec4.ttt", train_env_kwargs, num_epoch_data = 51200
    )
    train_dl = DataLoader(train_dataset, 64, num_workers = 4)

    test_dataset = PrEnvRenderDataset(
        CornerEnv, "scene/plane_box/corner_vec4.ttt", eval_env_kwargs, num_epoch_data = 10240
    )
    test_dl = DataLoader(test_dataset, 64, num_workers = 4)

    effnet_b0 = EfficientNetV1WithHead(
        6, 1, 1, 1
    )

    # sample_test(train_dataset, 100, "tmp/plane_box/corner_train_ext_sample", 4, 64)

    ## 余弦重启

    cfg = ModelTeacher.AdvanceConfig(
        schedule_type = "restart_cos",
        schedule_kwargs = {
            "T_0": 10, 
            "T_mult": 2, 
            "eta_min": 0.0
        },
        is_use_adam = True
    )

    # 实验 1
    mt = ModelTeacher(
        effnet_b0, 
        1e-3, 
        train_dl, 
        test_dl, 
        "./runs/plane_box/corner_ext", 
        nn.SmoothL1Loss,
        reg_mse_success_fn, 
        advance_config = cfg
    )
    mt.train(70)
