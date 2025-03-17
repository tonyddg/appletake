import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.sb3.exp_manager import ExpManager, Trial, load_exp_config

def sample_param(trial: Trial):
    params = {
        "batch_size": trial.suggest_categorical("batch_size", [64, 256, 1024]),
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True),
        "buffer_size": trial.suggest_categorical("buffer_size", [int(5e3), int(1e4), int(2e4)]),
        "learning_starts": trial.suggest_categorical("learning_starts", [0, 1000, 5000]),
        "train_freq": trial.suggest_categorical("train_freq", [1, 4, 16]),

        "tau": trial.suggest_categorical("tau", [0.001, 0.01, 0.1]),
        "net_arch_type": trial.suggest_categorical("net_arch", ["small", "medium", "big"]),
    }

    params["gradient_steps"] = params["train_freq"] * trial.suggest_categorical("gradient_steps_ratio", [1, 2, 4])
    params["net_arch"] = { # type: ignore
        "small": [128],
        "medium": [256],
        "big": [512]
    }[params["net_arch_type"]]

    return params

if __name__ == "__main__":

    em = ExpManager(
        load_exp_config(
            "app/example/conf/pendulum_finetuning.yaml",
            is_resolve = False
        ),
        sample_param,
        exp_name_suffix = None,
        opt_save_model = True
    )
    em.optimize(20)
