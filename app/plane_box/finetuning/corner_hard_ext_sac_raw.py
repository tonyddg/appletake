import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

from src.sb3.exp_manager import ExpManager, Trial, load_exp_config
from src.pr.safe_pyrep import SafePyRep
from arg_dict import exp_replace_arg_dict, exp_exec_arg_dict, make_backbone
from sample_param import sample_param_fine_hard_ext_sac

if __name__ == "__main__":
    with SafePyRep("scene/plane_box/corner_vec4_test2.ttt", True) as pr:

        em = ExpManager(
            load_exp_config(
                "app/plane_box/conf/base_hard_env.yaml",
                "app/plane_box/conf/corner_hard_ext_env.yaml",
                "app/plane_box/finetuning/ext_sac_raw.yaml",
                merge_dict = {
                    "trial": {"meta_exp_root": "runs/plane_box/corner_finetuning/ext_sac_hard_fine"}
                },
                is_resolve = False
            ),
            sample_param_fine_hard_ext_sac,
            exp_name_suffix = None,
            exp_replace_arg_dict = exp_replace_arg_dict({
                "pr": pr,
            }),
            exp_exec_arg_dict = exp_exec_arg_dict(),

            opt_save_model = True
        )
        em.optimize(20)
