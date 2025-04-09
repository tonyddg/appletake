import argparse
import os
import sys
from typing import Optional
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.sb3.exp_manager import train_model, load_exp_config
from src.pr.safe_pyrep import SafePyRep, PyRep
from conf.arg_dict import exp_replace_arg_dict, exp_exec_arg_dict

if __name__ == "__main__":

    parser = argparse.ArgumentParser(prog = None, description = None, epilog = None)
    parser.add_argument("--env_type", default = "three", choices = ["three", "paralle", "corner"], nargs = "?")
    parser.add_argument("--obs_type", default = "ext", choices = ["ext", "vec"], nargs = "?")
    parser.add_argument("--phase_type", default = "default", nargs = "?")
    parser.add_argument("--is_use_cfg_only", action = "store_true")
    parser.add_argument("--not_need_pr", action = "store_true")
    parser.add_argument("tuned_cfg")
    res = parser.parse_args()
    print(res)

    env_type = res.env_type
    obs_type = res.obs_type
    phase_type = res.phase_type
    tuned_cfg = res.tuned_cfg
    is_use_cfg_only = res.is_use_cfg_only
    not_need_pr = res.not_need_pr

    if is_use_cfg_only:
        cfg = load_exp_config(
            tuned_cfg,
            is_resolve = True
        )
    else:
        cfg = load_exp_config(
            f"app/plane_box/conf/base_normal_env.yaml",
            f"app/plane_box/conf/{env_type}_{obs_type}_env_{phase_type}.yaml",
            tuned_cfg,
            is_resolve = True
        )

    def train(pr: Optional[PyRep]):
        if pr is not None:
            addition_dict = {
                "pr": pr
            }
        else:
            addition_dict = {}

        train_model(
            cfg,
            verbose = 1,
            exp_root = f"runs/plane_box/ply/{env_type}_{obs_type}_{phase_type}",
            exp_replace_arg_dict = exp_replace_arg_dict(addition_dict),
            exp_exec_arg_dict = exp_exec_arg_dict(),
            trial_time_ratio = None
        )

    if not_need_pr:
        train(None)
    else:
        with SafePyRep("scene/plane_box/base_vec6.ttt", True) as pr:
            train(pr)

    # with SafePyRep("scene/plane_box/base_vec6.ttt", True) as pr:
    #     parser = argparse.ArgumentParser(prog = None, description = None, epilog = None)
    #     parser.add_argument("--env_type", default = "three", choices = ["three", "paralle", "corner"], nargs = "?")
    #     parser.add_argument("--obs_type", default = "ext", choices = ["ext", "vec"], nargs = "?")
    #     parser.add_argument("--phase_type", default = "default", nargs = "?")
    #     parser.add_argument("--is_use_cfg_only", action = "store_true")
    #     parser.add_argument("tuned_cfg")
    #     res = parser.parse_args()
    #     print(res)

    #     env_type = res.env_type
    #     obs_type = res.obs_type
    #     phase_type = res.phase_type
    #     tuned_cfg = res.tuned_cfg
    #     is_use_cfg_only = res.is_use_cfg_only

    #     if is_use_cfg_only:
    #         cfg = load_exp_config(
    #             tuned_cfg,
    #             is_resolve = True
    #         )
    #     else:
    #         cfg = load_exp_config(
    #             f"app/plane_box/conf/base_normal_env.yaml",
    #             f"app/plane_box/conf/{env_type}_{obs_type}_env_{phase_type}.yaml",
    #             tuned_cfg,
    #             is_resolve = True
    #         )

    #     train_model(
    #         cfg,
    #         verbose = 1,
    #         exp_root = f"runs/plane_box/ply/{env_type}_{obs_type}_{phase_type}",
    #         exp_replace_arg_dict = exp_replace_arg_dict({
    #             "pr": pr
    #         }),
    #         exp_exec_arg_dict = exp_exec_arg_dict(),
    #         trial_time_ratio = None
    #     )
