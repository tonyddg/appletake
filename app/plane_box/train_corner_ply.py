import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.sb3.exp_manager import train_model, load_exp_config
from src.pr.safe_pyrep import SafePyRep
from app.plane_box.finetuning.arg_dict import exp_replace_arg_dict, exp_exec_arg_dict

if __name__ == "__main__":
    with SafePyRep("scene/plane_box/corner_vec4_test2.ttt", True) as pr:

        train_model(
            load_exp_config(
                "runs/plane_box/corner_finetuning/ext_sac/2025_03_04_15_50_07/config.yaml",
            ),
            verbose = 1,
            exp_root = "runs/plane_box/corner_ply/ext_sac",
            exp_replace_arg_dict = exp_replace_arg_dict({
                "pr": pr
            }),
            exp_exec_arg_dict = exp_exec_arg_dict(),
            trial_time_ratio = None
        )
