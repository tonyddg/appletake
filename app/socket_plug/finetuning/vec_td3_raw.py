import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

from src.sb3.exp_manager import ExpManager, Trial, load_exp_config
from src.pr.safe_pyrep import SafePyRep
from arg_dict import exp_replace_arg_dict, exp_exec_arg_dict

# TODO: 图像加噪音, 记录最终测试结果, 结果不可能是 1, 找 bug
# TODO: 保存最好的 n 个结果
# TODO: 随机策略测试效果
# TODO: 开启节约内存选项 optimize_memory_usage
# TODO: 课程学习
# 在终端中训练

def sample_param(trial: Trial):
    params = {
        "batch_size": trial.suggest_categorical("batch_size", [64, 256, 1024]),
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True),
        "buffer_size": trial.suggest_categorical("buffer_size", [int(5e3), int(1e4), int(2e4)]),
        "learning_starts": trial.suggest_categorical("learning_starts", [0, 1000, 10000]),
        "train_freq": trial.suggest_categorical("train_freq", [1, 4, 16]),

        "tau": trial.suggest_categorical("tau", [0.001, 0.01, 0.1]),
        "net_arch_type": trial.suggest_categorical("net_arch", ["small", "medium", "big"]),

        "noise_std": trial.suggest_float("noise_std", 0, 1),
    }

    params["gradient_steps"] = params["train_freq"] * trial.suggest_categorical("gradient_steps_ratio", [1, 2, 4])
    params["net_arch"] = { # type: ignore
        "small": [128, 128],
        "medium": [256, 256],
        "big": [512, 512]
    }[params["net_arch_type"]]

    return params

if __name__ == "__main__":
    with SafePyRep("scene/socket_plug/socket_plug_vec4_tol5.ttt", True) as pr:

        em = ExpManager(
            load_exp_config(
                "app/socket_plug/conf/normal_vec_env.yaml",
                "app/socket_plug/finetuning/vec_td3_raw.yaml"
            ),
            sample_param,
            exp_name_suffix = None,
            exp_replace_arg_dict = exp_replace_arg_dict({
                "pr": pr
            }),
            exp_exec_arg_dict = exp_exec_arg_dict()
        )
        em.optimize(50)
