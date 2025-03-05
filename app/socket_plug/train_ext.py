import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.net.pr_task_dataset import PrEnvRenderDataset
from src.net.utility import ModelTeacher
from src.net.efficient_net import EfficientNetV1WithHead

from src.gym_env.socket_plug.socket_plug_vec import SocketPlugVecEnv
from src.gym_env.aug import get_crop_resize, get_depth_aug, get_hwc2chw
import albumentations as A

from torch.utils.data import DataLoader
from torch import nn
import numpy as np

if __name__ == "__main__":
    train_env_kwargs = {    
        "subenv_mid_name": '#',
        "subenv_range": (0, 5),
        
        "env_init_pos_min": np.array([-15, -15, 0, -5, -5, -15], dtype = np.float32),
        "env_init_pos_max": np.array([15, 15, 20, 5, 5, 15], dtype = np.float32),
        "env_max_steps": 20,

        "obs_type": 'depth',
        "obs_process": A.Compose([
            get_crop_resize(),
            get_depth_aug(0.8, 2),
            A.Normalize((0.538), (0.287), 1),
            get_hwc2chw(False)
        ]),
        "obs_depth_range": (0, 0.010),
    }
    
    eval_env_kwargs = {    
        "subenv_mid_name": '#',
        "subenv_range": (0, 5),
        
        "env_init_pos_min": np.array([-15, -15, 0, -5, -5, -15], dtype = np.float32),
        "env_init_pos_max": np.array([15, 15, 20, 5, 5, 15], dtype = np.float32),
        "env_max_steps": 20,

        "obs_type": 'depth',
        "obs_process": A.Compose([
            get_crop_resize(),
            get_depth_aug(0.2, 1),
            A.Normalize((0.538), (0.287), 1),
            get_hwc2chw(False)
        ]),
        "obs_depth_range": (0, 0.010),
    }

    train_dataset = PrEnvRenderDataset(
        SocketPlugVecEnv, "scene/socket_plug/socket_plug_vec4_tol5.ttt", train_env_kwargs, num_epoch_data = 51200
    )
    train_dl = DataLoader(train_dataset, 64, num_workers = 2)

    test_dataset = PrEnvRenderDataset(
        SocketPlugVecEnv, "scene/socket_plug/socket_plug_vec4_tol5.ttt", eval_env_kwargs, num_epoch_data = 10240
    )
    test_dl = DataLoader(test_dataset, 64, num_workers = 2)

    effnet_b0 = EfficientNetV1WithHead(
        6, 1, 1, 1
    )

    cfg = ModelTeacher.AdvanceConfig(
        schedule_type = True,
        is_use_adam = False
    )

    mt = ModelTeacher(
        effnet_b0, 
        2e-3, 
        train_dl, 
        test_dl, 
        "./runs/socket_plug/ext", 
        nn.MSELoss, 
        None, 
        advance_config = cfg
    )
    mt.train(30)