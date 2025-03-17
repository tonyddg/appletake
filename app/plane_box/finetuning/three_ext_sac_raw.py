import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

from src.sb3.exp_manager import ExpManager, Trial, load_exp_config
from src.pr.safe_pyrep import SafePyRep
from arg_dict import exp_replace_arg_dict, exp_exec_arg_dict, make_backbone
from sample_param import sample_param_ext_sac_fine

if __name__ == "__main__":
    with SafePyRep("scene/plane_box/three_vec4_test2.ttt", True) as pr:

        em = ExpManager(
            load_exp_config(
                "app/plane_box/conf/three_normal_ext_env.yaml",
                "app/plane_box/finetuning/ext_sac_raw.yaml",
                merge_dict = {
                    "trial": {"meta_exp_root": "runs/plane_box/three_finetuning/ext_sac"}
                },
                is_resolve = False
            ),
            sample_param_ext_sac_fine,
            exp_name_suffix = None,
            exp_replace_arg_dict = exp_replace_arg_dict({
                "pr": pr,
            }),
            exp_exec_arg_dict = exp_exec_arg_dict(),

            opt_save_model = True
        )
        em.optimize(50)
