import os
import sys
from typing import Dict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

from src.net.pr_task_dataset import PrEnvRenderDataset, sample_test
from src.net.utility import ModelTeacher, reg_mse_success_fn

from src.sb3.exp_manager import config_to_env, load_exp_config, config_data_replace, config_data_exec
from src.gym_env.plane_box.corner import CornerEnv
from src.gym_env.plane_box.paralle import ParalleEnv
from src.gym_env.plane_box.three import ThreeEnv

from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torch import nn
import numpy as np

from conf.arg_dict import exp_replace_arg_dict, exp_exec_arg_dict

# TODO 设计实验证明强化学习方法优于直接神经网络预测

if __name__ == "__main__":

    env_type = "three"
    cnf_vary = "train"
    obs_type = "train"
    phase_type = "singleview"
    # custom_conf = {
    #     "train_env": {"kwargs": {"dataset_is_center_goal": True}}
    # }
    custom_conf = {}
    is_sample_train = True

    cls_dict = {
        "three": ThreeEnv,
        "corner": CornerEnv,
        "paralle": ParalleEnv
    }

    # cfg = load_exp_config(b
    #     f"app/plane_box/conf/base_train_env.yaml",
    #     f"app/plane_box/conf/{env_type}_train_env.yaml",
    #     is_resolve = True
    # )
    cfg = load_exp_config(
        f"app/plane_box/conf/base_{cnf_vary}_env.yaml",
        f"app/plane_box/conf/{env_type}_{obs_type}_env_{phase_type}.yaml",
        is_resolve = True,
        merge_dict = custom_conf
    )
    if cfg.eval_env.get("is_base_on_train", False):
        cfg.eval_env = OmegaConf.merge(cfg.train_env, cfg.eval_env)
    cfg = OmegaConf.to_container(cfg)

    assert isinstance(cfg, Dict)
    # cfg["train_env"]["kwargs"]["obs_trans"] = "@train_trans_b2"
    cfg = config_data_replace(cfg, exp_replace_arg_dict())
    cfg = config_data_exec(cfg, exp_exec_arg_dict())

    if is_sample_train:
        train_env_kwargs = cfg["train_env"]["kwargs"] # type: ignore
    else:
        train_env_kwargs = cfg["eval_env"]["kwargs"] # type: ignore

    # train_env_kwargs = cfg["train_env"]["kwargs"] # type: ignore
    # eval_env_kwargs = cfg["eval_env"]["kwargs"] # type: ignore

    train_dataset = PrEnvRenderDataset(
        cls_dict[env_type], "scene/plane_box/base_vec6.ttt", train_env_kwargs, num_epoch_data = 51200 # type: ignore
    )
    sample_test(train_dataset, 50, f"tmp/plane_box/{env_type}_train_ext_sample", 4, 64)
