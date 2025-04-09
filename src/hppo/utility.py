import numpy as np
import torch as torch

def concat_numpy_action_for_step(discrete_action: np.ndarray, continue_action: np.ndarray):
    return np.concat([discrete_action[:, np.newaxis], continue_action], 1)

def concat_tensor_action_for_step(discrete_action: torch.Tensor, continue_action: torch.Tensor):
    return torch.concat([torch.unsqueeze(discrete_action, 1), continue_action], 1)

def seperate_action_from_numpy(concat_action: np.ndarray):
    if len(concat_action.shape) == 2:
        discrete_action = np.astype(concat_action[:, 0], np.int32)
        continue_action = concat_action[:, 1:]
        return (discrete_action, continue_action)
    else:
        discrete_action = np.astype(concat_action[0], np.int32)
        continue_action = concat_action[1:]
        return (discrete_action, continue_action)
