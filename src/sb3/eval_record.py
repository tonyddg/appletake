from typing import Union, Literal, List, Dict, Any, Callable, Optional
from pathlib import Path
import os
from tqdm import tqdm
import numpy as np

from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvWrapper
from stable_baselines3.common import type_aliases
from stable_baselines3.common.evaluation import evaluate_policy

from torch.utils.tensorboard.writer import SummaryWriter
import torch

from ..analysis_log import TickAnalysis
from .get_critic import get_critic, is_model_critic_available
import warnings

# TODO: 通过 callback 创建环境, 保证每次测试环境相同
# TODO: 回报折扣因子

# 记录数据标志
from enum import Flag, auto
class RecordFlag(Flag):
    VEDIO = auto()
    REWARD = auto()
    CRITIC = auto()

    ALL = VEDIO | REWARD | CRITIC

    NORMAL = VEDIO | REWARD

class RecordBuf:
    def __init__(self, num_envs: int, record_flag: RecordFlag, test_flag: RecordFlag):
        if test_flag & record_flag:
            self.venv_buf = [[] for _ in range(num_envs)]
            self._is_available = True
        else:
            self._is_available = False

    def is_available(self):
        return self._is_available

    def append(self, i: int, obj: Any):
        if self.is_available():
            self.venv_buf[i].append(obj)
        else:
            raise Exception(f"尝试向无效的记录缓冲区插入数据 {obj}")
    
    def get_res(self, i: int, to_numpy: bool = True):
        if self.is_available() and len(self.venv_buf[i]) > 0:
            if to_numpy:
                res = np.array(self.venv_buf[i])
            else:
                res = self.venv_buf[i]
            self.venv_buf[i] = []

            return res
        else:
            return None

def eval_record(
    model: "type_aliases.PolicyPredictor",
    env: Union[VecEnv, VecEnvWrapper],

    vedio_record_callback: Callable[[int, Optional[List[np.ndarray]], Optional[np.ndarray], Optional[np.ndarray]], None],
    num_record_episodes: Optional[int] = None,
    record_flag: RecordFlag = RecordFlag.ALL,

    verbose: int = 0,
    n_eval_episodes: int = 10,
    deterministic: bool = True,
    **kwargs
):
    '''
    对 evaluate_policy 的包裹, 实现记录视频文件功能
    * `model, env, n_eval_episodes, deterministic, **kwargs` 函数 `evaluate_policy` 的原始参数
    * `vedio_record_callback` 视频记录回调, 参数为 (num_ep, vedio, reward_list) -> None
            * `num_ep` Episode 编号
            * `vedio` 视频 (T, C, H, W) 列表
            * `reward_list` 各个时刻获得奖励的列表
    * `num_record_episodes` 记录视频数
    '''

    num_envs = env.num_envs
    # 按环境独立记录每帧画面

    if not is_model_critic_available(model):
        warnings.warn("给定的模型不支持获取 Critic", UserWarning)
        record_flag = record_flag & (~RecordFlag.CRITIC)

    vedio_buf = RecordBuf(num_envs, RecordFlag.VEDIO, record_flag)
    reward_buf = RecordBuf(num_envs, RecordFlag.REWARD, record_flag)
    critic_buf = RecordBuf(num_envs, RecordFlag.CRITIC, record_flag)

    if num_record_episodes == None:
        num_record_episodes = n_eval_episodes

    num_episode = 0

    if verbose > 0:
        pbar = tqdm(total = num_record_episodes)
        pbar.set_description_str("Model Evaling")
    else:
        pbar = None

    def eval_record_callback(_locals: Dict[str, Any], _globals: Dict[str, Any]):
        nonlocal num_episode

        if num_episode < num_record_episodes:

            i = _locals["i"]
            done = _locals["done"]

            if vedio_buf.is_available():
                # done 时, 当前状态为新环境, 尝试插入黑色图片
                if done:
                    if len(vedio_buf.venv_buf[i]) > 0:
                        last_img = vedio_buf.venv_buf[i][0]
                        vedio_buf.append(i, np.zeros(last_img.shape, dtype = last_img.dtype))
                else:
                    vedio_buf.append(i, env.get_images()[i])
            if reward_buf.is_available():
                reward_buf.append(i, _locals["reward"])
            if critic_buf.is_available():
                critic_buf.append(i, get_critic(model, _locals["observations"][i])) # type: ignore

            if done:
                with TickAnalysis(f"记录第 {num_episode} 个视频", __name__):
                    vedio_record_callback(
                        num_episode, 
                        vedio_buf.get_res(i), #type:ignore 
                        reward_buf.get_res(i), #type:ignore 
                        critic_buf.get_res(i) #type:ignore 
                    )

                    num_episode += 1

                    if pbar != None:
                        pbar.update()

    eval_res = evaluate_policy(model, env, n_eval_episodes, deterministic, callback = eval_record_callback, **kwargs)
    
    if pbar != None:
        pbar.close()

    return eval_res

def eval_record_to_mp4_file(
    model: "type_aliases.PolicyPredictor",
    env: Union[VecEnv, VecEnvWrapper],

    vedio_fold: Union[str, Path],
    vedio_prefix: str = "",
    fps: int = 25,

    num_record_episodes: Optional[int] = None,
    verbose: int = 0,
    n_eval_episodes: int = 10,
    deterministic: bool = True,
    **kwargs
):
    '''
    将过程记录为视频
    '''
    try:
        from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
    except ImportError as e:  # pragma: no cover
        raise Exception("MoviePy is not installed") from e

    if isinstance(vedio_fold, str):
        vedio_fold = Path(vedio_fold)
    if not vedio_fold.is_dir():
        os.mkdir(vedio_fold)
    vedio_name_pattern = str(vedio_fold.joinpath(vedio_prefix + "{}.mp4"))
    
    def mp4_callback(num_ep, vedio, *args):
        clip = ImageSequenceClip(sequence = vedio, fps = fps)
        clip.write_videofile(vedio_name_pattern.format(num_ep), logger = None)
    
    return eval_record(
        model, env, mp4_callback, num_record_episodes, RecordFlag.VEDIO, verbose, n_eval_episodes, deterministic, **kwargs
    )

def eval_record_to_tensorbard(
    model: "type_aliases.PolicyPredictor",
    env: Union[VecEnv, VecEnvWrapper],

    tb_writer: SummaryWriter,
    root_tag: str = "eval",
    record_flag: RecordFlag = RecordFlag.NORMAL,
    fps: int = 25,
    return_gamma: float = 1,

    num_record_episodes: Optional[int] = None,
    verbose: int = 0,
    n_eval_episodes: int = 10,
    deterministic: bool = True,
    **kwargs
):
    '''
    将过程记录为 Tensorboard 视频
    '''
    def tb_callback(num_ep, vedio, reward_list, critic_list):
        if vedio is not None:
            tb_writer.add_video(
                root_tag + "/episode_" + str(num_ep) + "/vedio", 
                torch.tensor(np.asarray(vedio)).unsqueeze(0).permute((0, 1, 4, 2, 3)), 
                fps = fps
            )
        
        if reward_list is not None:
            return_tag = root_tag + "/episode_" + str(num_ep) + "/discount_return"
            
            total_length = len(reward_list)
            ep_return = 0
            ep_return_list = np.zeros(total_length)

            for t in range(total_length):
                ep_return = reward_list[-1 - t] + ep_return * return_gamma
                ep_return_list[total_length - t - 1] = ep_return
                # tb_writer.add_scalar(return_tag, ep_return, total_length - t - 1)

            for t in range(total_length):
                tb_writer.add_scalar(return_tag, ep_return_list[t], t)

        if critic_list is not None:
            critic_tag = root_tag + "/episode_" + str(num_ep) + "/critic"
            for ep, ep_critic in enumerate(critic_list):
                tb_writer.add_scalar(critic_tag, ep_critic, ep)

    return eval_record(
        model, env, tb_callback, num_record_episodes, record_flag, verbose, n_eval_episodes, deterministic, **kwargs
    )
