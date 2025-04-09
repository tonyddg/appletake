from optuna import Trial

def sample_param_ext_sac_raw(trial: Trial):
    params = {
        "batch_size": trial.suggest_categorical("batch_size", [256, 512, 1024]),
        "learning_rate": trial.suggest_float("learning_rate", 5e-4, 1e-2, log=True),
        "buffer_size": trial.suggest_categorical("buffer_size", [int(2**13), int(2**14), int(2**15)]),

        "net_arch_type": trial.suggest_categorical("net_arch", ["small", "medium", "big"]),

        "gamma": trial.suggest_categorical("gamma", [0.9, 0.99, 1.0]),
        "tau": trial.suggest_categorical("tau", [0.1, 0.01, 0.001]),
        "learning_starts": 0,

        "gradient_steps": trial.suggest_categorical("gradient_steps", [4, 8, 16]),
    }

    params["net_arch"] = { # type: ignore
        "small": [256],
        "medium": [512],
        "big": [256, 256]
    }[params["net_arch_type"]]

    return params

def sample_param_ext_td3_raw(trial: Trial):
    params = {
        "batch_size": trial.suggest_categorical("batch_size", [256, 512, 1024]),
        "learning_rate": trial.suggest_float("learning_rate", 5e-4, 1e-2, log=True),
        "buffer_size": trial.suggest_categorical("buffer_size", [int(2**13), int(2**14), int(2**15)]),

        "net_arch_type": trial.suggest_categorical("net_arch", ["small", "medium", "big"]),

        "gamma": trial.suggest_categorical("gamma", [0.9, 0.99, 1.0]),
        "tau": trial.suggest_categorical("tau", [0.1, 0.01, 0.001]),
        "learning_starts": 0,

        "act_sigma": trial.suggest_float("act_sigma", 0, 2),
        "des_sigma": trial.suggest_float("des_sigma", 0, 1),

        "gradient_steps": trial.suggest_categorical("gradient_steps", [4, 8, 16]),
    }

    params["pos_sigma"] = params["act_sigma"]
    params["rot_sigma"] = float(params["act_sigma"]) * 0.2
    params["des_sigma"] = params["des_sigma"]

    params["net_arch"] = { # type: ignore
        "small": [256],
        "medium": [512],
        "big": [256, 256]
    }[params["net_arch_type"]]

    return params

SAMPLE_SAC_DICT = {
    "raw": sample_param_ext_sac_raw,
}

SAMPLE_TD3_DICT = {
    "raw": sample_param_ext_td3_raw,
}
