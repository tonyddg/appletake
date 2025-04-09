import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
from src.sb3.exp_manager import ExpManager, Trial, load_exp_config
from src.pr.safe_pyrep import SafePyRep
from conf.arg_dict import exp_replace_arg_dict, exp_exec_arg_dict
from sample_param import SAMPLE_SAC_DICT, SAMPLE_TD3_DICT

if __name__ == "__main__":
    with SafePyRep("scene/plane_box/base_vec6.ttt", True) as pr:

        parser = argparse.ArgumentParser(prog = None, description = None, epilog = None)
        parser.add_argument("--env_type", default = "three", choices = ["three", "paralle", "corner"], nargs = "?")
        parser.add_argument("--phase_type", default = "default", nargs = "?")
        parser.add_argument("--obs_type", default = "ext", choices = ["ext", "vec", "ftb"], nargs = "?")
        parser.add_argument("--alg_type", default = "sac", choices = ["sac", "td3"], nargs = "?")
        # parser.add_argument("--is_tuned", action = "store_true")
        parser.add_argument("--n_trials", default = 20, type = int, nargs = "?")
        parser.add_argument("--tuned_recipe", default = "raw", nargs = "?")
        res = parser.parse_args()
        print(f"receive arguments: {res}")

        env_type = res.env_type
        phase_type = res.phase_type
        obs_type = res.obs_type
        alg_type = res.alg_type
        n_trials = res.n_trials
        tuned_recipe = res.tuned_recipe

        # tuned_suffix = "raw"
        # if is_tuned:
        #     tuned_suffix = "tuned"
        # sample_func_name = tuned_suffix + "_" + phase_type

        if alg_type == "sac":
            sample_func = SAMPLE_SAC_DICT[tuned_recipe]
        elif alg_type == "td3":
            sample_func = SAMPLE_TD3_DICT[tuned_recipe]
        else:
            raise Exception()

        em = ExpManager(
            load_exp_config(
                f"app/plane_box/conf/base_normal_env.yaml",
                f"app/plane_box/conf/{env_type}_{obs_type}_env_{phase_type}.yaml",
                f"app/plane_box/conf/finetuning_{alg_type}_base.yaml",
                merge_dict = {
                    "trial": {"meta_exp_root": f"runs/plane_box/finetuning/{env_type}_{phase_type}_finetuning/{alg_type}_{obs_type}_{tuned_recipe}"}
                },
                is_resolve = False
            ),
            sample_func,
            exp_name_suffix = None,
            exp_replace_arg_dict = exp_replace_arg_dict({
                "pr": pr,
            }),
            exp_exec_arg_dict = exp_exec_arg_dict(),

            opt_save_model = True
        )
        em.optimize(n_trials)
