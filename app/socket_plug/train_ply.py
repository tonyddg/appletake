import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.sb3.exp_manager import train_model, load_exp_config
from src.pr.safe_pyrep import SafePyRep
from app.socket_plug.finetuning.arg_dict import exp_replace_arg_dict, exp_exec_arg_dict

if __name__ == "__main__":
    with SafePyRep("scene/socket_plug/socket_plug_vec4_tol5.ttt", True) as pr:

        train_model(
            load_exp_config(
                "app/socket_plug/conf/ext_td3_tuned.yaml",
            ),
            verbose = 1,
            exp_root = "runs/socket_plug/train/ext_td3",
            exp_replace_arg_dict = exp_replace_arg_dict({
                "pr": pr
            }),
            exp_exec_arg_dict = exp_exec_arg_dict()
        )
