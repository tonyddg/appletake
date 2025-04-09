import argparse
import os
import sys

import numpy as np
from omegaconf import OmegaConf
import torch


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.gym_env.utility import rot_to_rotvec

from src.net.efficient_net import EfficientNetV1Backbone
from src.net.multi_view_net import SeperateLateFuseNet
from src.net.net_abc import BackboneWithHead

from src.gym_env.plane_box.plane_box import PlaneBoxEnvActionType
from src.gym_env.plane_box.utility import planebox_eval
from src.sb3.exp_manager import config_to_env, train_model, load_exp_config
from src.sb3.model_predictor import ModelPredictor, PolicyDecorator
from src.pr.safe_pyrep import SafePyRep
from conf.arg_dict import exp_replace_arg_dict, exp_exec_arg_dict
from stable_baselines3 import SAC, TD3

from src.hppo.utility import seperate_action_from_numpy
from src.hppo.hppo import HybridPPO
from src.hppo.hybrid_policy import HybridActorCriticPolicy

def passive_act_decorater(act: np.ndarray):
    deg_diff, _ = rot_to_rotvec(act[3:])
    pos_diff = np.linalg.norm(act[:3])

    # print(f"deg_diff: {deg_diff}, pos_diff: {pos_diff}")

    if deg_diff < 1 and pos_diff < 1:
        act = np.hstack([act, [0, 1]], dtype = np.float32)
    else:
        act = np.hstack([act, [1, 0]], dtype = np.float32)
    act = np.hstack([act, [1, 0]], dtype = np.float32)
    return act

def hppo_act_decorater(act: np.ndarray):
    discrete_action, continue_action = seperate_action_from_numpy(act)
    act = np.zeros(8)
    act[:6] = continue_action
    if discrete_action == 0:
        act[6] = 1
    else:
        act[7] = 1

    return act

with SafePyRep("scene/plane_box/base_vec6.ttt", True) as pr:

    parser = argparse.ArgumentParser(prog = None, description = None, epilog = None)
    parser.add_argument("--env_type", default = "three", choices = ["three", "paralle", "corner"], nargs = "?")
    parser.add_argument("--obs_type", default = "ext", choices = ["ext", "vec"], nargs = "?")
    parser.add_argument("--eval_name", default = None, nargs = "?")
    parser.add_argument("--alg_type", default = "sac", choices = ["sac", "td3", "hppo", "eff", "ftb"], nargs = "?")
    parser.add_argument("--num_record_eps", default = 5, type = int, nargs = "?")
    # parser.add_argument("--is_parameter_space", action = "store_true")
    parser.add_argument("--phase_type", default = "default", nargs = "?")
    parser.add_argument("--act_type", default = "DESICION_PDDPG", type = PlaneBoxEnvActionType, nargs = "?")
    parser.add_argument("model_path")
    res = parser.parse_args()
    print(f"receive arguments: {res}")

    env_type = res.env_type
    obs_type = res.obs_type
    eval_name = res.eval_name
    model_path = res.model_path
    alg_type = res.alg_type
    num_record_eps = res.num_record_eps
    # is_parameter_space = res.is_parameter_space
    phase_type = res.phase_type
    act_type = PlaneBoxEnvActionType(res.act_type)

    cfg = load_exp_config(
        f"app/plane_box/conf/base_eval_env.yaml",
        f"app/plane_box/conf/{env_type}_{obs_type}_env_{phase_type}.yaml",
        is_resolve = True
    )
    if cfg.eval_env.get("is_base_on_train", False):
        cfg.eval_env = OmegaConf.merge(cfg.train_env, cfg.eval_env)

    if alg_type == "sac":
        rl_policy = SAC.load(model_path)

    elif alg_type == "td3":
        rl_policy = TD3.load(model_path)

        # if is_passive_policy:
        #     rl_policy = PolicyDecorator(rl_policy, act_decorater)

    elif alg_type == "hppo":
        rl_policy = HybridPPO.load(model_path)

    else:
        if alg_type == "eff":
            net = BackboneWithHead(
                EfficientNetV1Backbone(
                    3, 1, 1, 0.2
                ), 6, 0.2
            )
        elif alg_type == "ftb":
            net = BackboneWithHead(
                SeperateLateFuseNet(
                    3, lambda: EfficientNetV1Backbone(1, 1, 1, 0.2)
                ), 6, 0.2
            )
        else:
            raise Exception("未知算法")

        net.load_state_dict(torch.load(model_path))
        net.train(False)

        cfg.eval_env.wrapper = None

        rl_policy = ModelPredictor(net, None)

    if act_type is PlaneBoxEnvActionType.PASSIVE_ALWAYS or act_type is PlaneBoxEnvActionType.PASSIVE_END:
        rl_policy = PolicyDecorator(rl_policy, passive_act_decorater)
    elif act_type is PlaneBoxEnvActionType.DESICION_HPPO:
        # rl_policy = PolicyDecorator(rl_policy, hppo_act_decorater)
        if isinstance(rl_policy, HybridPPO):
            rl_policy.policy.use_hddpg_like_predict_mode(True)
        else:
            rl_policy = PolicyDecorator(rl_policy, hppo_act_decorater)

    cfg.eval_env.kwargs.subenv_range = [0, 6]
    # cfg.eval_env.kwargs.env_max_step = 2
    # cfg.eval_env.kwargs.obs_trans = "@ext_trans_noaug"
    if eval_name is None:
        eval_name = f"{alg_type}_"

    eval_env = config_to_env(cfg.eval_env, exp_replace_arg_dict({"pr": pr}), exp_exec_arg_dict())

    print("eval start")

    # 测试 Optuna 保存的模型中, 性能下降的原因
    planebox_eval(
        rl_policy,
        eval_env,  # type: ignore
        f"runs/plane_box/eval/{obs_type}_{env_type}_{phase_type}", 
        save_name_prefix = eval_name,
        num_record_episodes = num_record_eps,
        n_eval_episodes = 100,
        is_parameter_space = True
    )

    print("eval done")