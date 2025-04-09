import os
from pathlib import Path
import signal
from typing import Any, Dict, Iterator, List, Optional, Union
import numpy as np
import torch
from torch.utils.data import IterableDataset
from pyrep import PyRep
from torch.utils.data import DataLoader

from matplotlib import pyplot as plt

import time

import atexit
from abc import abstractmethod, ABCMeta

from src.utility import get_file_time_str


class PrTaskVecEnvForDatasetInterface(metaclass = ABCMeta):
    '''
    基于 Pyrep 的并行环境接口类
    '''

    @abstractmethod
    def __init__(self, env_pr: PyRep, *args, **kwargs) -> None:
        pass

    @abstractmethod
    def dataset_get_task_label(self) -> List[np.ndarray]:
        '''
        获取最佳任务目标 (对于每个子环境)

        在对齐任务中为当前物体相对最佳位置的 x, y, z, a, b, c 齐次变换  
        get_rel_pose(subenv.plug.get_pose(None), subenv.plug_best_pose)
        '''
        pass

    @abstractmethod
    def dataset_get_list_obs(self) -> List[np.ndarray]:
        '''
        获取环境观测, 不用堆叠, 而是 List 形式
        '''
        pass

    @abstractmethod
    def dataset_reinit(self) -> None:
        '''
        环境初始化
        '''
        pass

    @abstractmethod
    def dataset_num_envs(self) -> int:
        '''
        并行环境数量
        '''
        pass

class PrEnvRenderDataset(IterableDataset):
    def __init__(
            self, 
            pr_env_type: type["PrTaskVecEnvForDatasetInterface"],
            scene_path: str, 
            env_kwargs: Dict[str, Any], 
            num_epoch_data: int,
        ) -> None:
        '''
        从观测中采样

        使用 get_obs_cv, 即保证观测为 hxw float 格式, 而不是 sb3 使用的 1xhxw uint8 格式
        '''
        self.pr_env_type = pr_env_type
        self.scene_path = scene_path
        self.env_kwargs = env_kwargs

        self.num_epoch_data = num_epoch_data

    def __iter__(self) -> Iterator:
        return PrEnvRenderIter(self.pr_env_type, self.scene_path, self.env_kwargs, self.num_epoch_data)

    def __len__(self):
        return self.num_epoch_data

class PrEnvRenderIter(Iterator):
    def __init__(
            self, 
            pr_env_type: type["PrTaskVecEnvForDatasetInterface"],
            scene_path: str, 
            env_kwargs: Dict[str, Any],
            num_epoch_data: int,
        ) -> None:
        self.pr = PyRep()
        self.pr.launch(scene_path, True)
        self.pr.start()

        env_kwargs.update(env_pr = self.pr)
        self.env = pr_env_type(
            **env_kwargs
        )

        work_info = torch.utils.data.get_worker_info()
        if work_info is not None and work_info.num_workers > 1:
            num_workers = work_info.num_workers
        else:
            num_workers = 1

        self.is_close = False
        self.num_epoch_data = num_epoch_data // num_workers
        self.iter_epoch_data = 0

        signal.signal(signal.SIGINT, self._sig_exit_handler)
        signal.signal(signal.SIGTERM, self._sig_exit_handler)
        atexit.register(self.close)

        self.sample_data()

    def close(self):
        if not self.is_close:
            self.is_close = True
            try:
                print("Close Pyrep Start")
                self.pr.shutdown()
                print("Close Pyrep Over")
            except Exception as e:
                print(f"Error during cleanup: {str(e)}")

    def _sig_exit_handler(self, sig, frame):
        print("Close By Handler")
        self.close()

    def __del__(self):
        print("Close By Del")
        self.close()

    def sample_data(self):
        self.env.dataset_reinit()
        self.cache_obs = self.env.dataset_get_list_obs()
        self.cache_label = self.env.dataset_get_task_label()
        self.cur_idx = 0

    def __next__(self):
        if self.cur_idx >= self.env.dataset_num_envs():
            self.sample_data()
        
        self.cur_idx += 1
        self.iter_epoch_data += 1

        if self.iter_epoch_data >= self.num_epoch_data:
            raise StopIteration()

        obs = self.cache_obs[self.cur_idx - 1]
        label = self.cache_label[self.cur_idx - 1]

        return obs, label
    
def sample_test(dataset: PrEnvRenderDataset, sample_times: int, sample_savepath: Optional[Union[Path, str]], num_workers: int = 2, batchsize: int = 64, alpha: float = 0.98, is_seprate_channel: bool = True):
    
    dl = DataLoader(dataset, batchsize, num_workers = num_workers)
    
    moving_obs_mean, moving_obs_std = 0, 0
    moving_label_mean, moving_label_std = 0, 0

    last_obs = None
    last_label = None

    print("Sample Start")
    t_start = time.perf_counter()
    for i, (obs, label) in enumerate(dl):
        batch_obs_std, batch_obs_mean = torch.std_mean(obs)
        batch_label_std, batch_label_mean = torch.std_mean(label, 0)

        moving_obs_mean = moving_obs_mean * alpha + batch_obs_mean * (1 - alpha)
        moving_obs_std = moving_obs_std * alpha + batch_obs_std * (1 - alpha)
        moving_label_mean = moving_label_mean * alpha + batch_label_mean * (1 - alpha)
        moving_label_std = moving_label_std * alpha + batch_label_std * (1 - alpha)

        if i >= sample_times:
            last_obs = obs
            last_label = label
            break

    avg_sample_times = (time.perf_counter() - t_start) / sample_times
    print("Sample Over")

    if sample_savepath is not None:
        assert isinstance(last_obs, torch.Tensor)
        assert isinstance(last_label, torch.Tensor)

        fig, axes = plt.subplot_mosaic([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        fig.set_layout_engine("compressed")
        fig.set_size_inches(32, 24)

        obs_c = last_obs.shape[1]

        for i, axe in enumerate(axes.values()):
            if is_seprate_channel:
                img = torch.squeeze(last_obs[i // obs_c, i % obs_c]).cpu().numpy()
                axe.set_title(f"label: {str(last_label[i // obs_c])}")
            else:
                img = torch.permute(last_obs[i], (1, 2, 0)).cpu().numpy()
                axe.set_title(f"label: {str(last_label[i])}")
            axe.plot
            axe.imshow(img)
            axe.set_xticks([])
            axe.set_yticks([])
        
        sample_savepath = Path(sample_savepath)
        if not sample_savepath.exists():
            os.makedirs(sample_savepath)
        fig.savefig(sample_savepath.joinpath(get_file_time_str() + ".png").as_posix())
    
    print(f"sample speed: {avg_sample_times}s/batch")
    print(f"obs mean: {moving_obs_mean}, obs std: {moving_obs_std}")
    print(f"label mean: {moving_label_mean}, label std: {moving_label_std}")