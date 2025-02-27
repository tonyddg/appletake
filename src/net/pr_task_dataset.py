import signal
from typing import Any, Dict, Iterator, List
import numpy as np
import torch
from torch.utils.data import IterableDataset
from pyrep import PyRep

import atexit

from abc import abstractmethod, ABCMeta

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
        return SocketPlugRenderIter(self.pr_env_type, self.scene_path, self.env_kwargs, self.num_epoch_data)

    def __len__(self):
        return self.num_epoch_data

class SocketPlugRenderIter(Iterator):
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

        self.num_epoch_data = num_epoch_data // num_workers
        self.iter_epoch_data = 0

        signal.signal(signal.SIGINT, self._sig_exit_handler)
        signal.signal(signal.SIGTERM, self._sig_exit_handler)
        atexit.register(self.close)
        self.is_close = False

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