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

from matplotlib import pyplot as plt
from matplotlib.axes import Axes

from ..utility import get_file_time_str
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

    SUCCESS = auto()
    # 提供相关回调函数参数后自动开启
    CUSTOM = auto()

    ALL = VEDIO | REWARD | CRITIC | SUCCESS

    NORMAL = VEDIO | REWARD | SUCCESS

class RecordBuf:
    def __init__(self, num_envs: int, record_flag: Optional[RecordFlag], test_flag: Optional[RecordFlag]):
        if (record_flag is None or test_flag is None) or (test_flag & record_flag):
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

    vedio_record_callback: Callable[[int, Optional[List[np.ndarray]], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]], None],
    num_record_episodes: Optional[int] = None,
    record_flag: RecordFlag = RecordFlag.ALL,

    verbose: int = 0,
    n_eval_episodes: int = 10,
    deterministic: bool = True,

    # fun(i, _local, _global) -> np.ndarray
    custom_record_callback: Optional[Callable[[int, Dict, Dict], np.ndarray]] = None,
    addition_callback: Optional[Callable[[Dict, Dict], None]] = None,
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

    if RecordFlag.CRITIC in record_flag and not is_model_critic_available(model):
        warnings.warn("给定的模型不支持获取 Critic", UserWarning)
        record_flag = record_flag & (~RecordFlag.CRITIC)

    if RecordFlag.CUSTOM in record_flag and custom_record_callback is None:
        warnings.warn("没有提供自定义记录回调参数 custom_record_callback", UserWarning)
        record_flag = record_flag & (~RecordFlag.CUSTOM)
    # 如果提供了回调函数自动开启
    if custom_record_callback is not None:
        record_flag = record_flag | (RecordFlag.CUSTOM)

    vedio_buf = RecordBuf(num_envs, RecordFlag.VEDIO, record_flag)
    reward_buf = RecordBuf(num_envs, RecordFlag.REWARD, record_flag)
    critic_buf = RecordBuf(num_envs, RecordFlag.CRITIC, record_flag)
    custom_buf = RecordBuf(num_envs, RecordFlag.CUSTOM, record_flag)

    if num_record_episodes == None:
        num_record_episodes = n_eval_episodes

    num_episode = 0

    if verbose > 0:
        pbar = tqdm(total = num_record_episodes)
        pbar.set_description_str("Model Evaling")
    else:
        pbar = None
    
    success_evals = 0

    def eval_record_callback(_locals: Dict[str, Any], _globals: Dict[str, Any]):
        nonlocal num_episode
        nonlocal success_evals
        nonlocal record_flag

        i = _locals["i"]
        done = _locals["done"]

        # 关于成功率的部分独立记录
        if done:
            if RecordFlag.SUCCESS in record_flag:
                if "is_success" in _locals["info"]:
                    if _locals["info"]["is_success"]:
                        success_evals += 1
                else:
                    warnings.warn("Info 中没有键 is_success, 无法记录成功率")
                    record_flag = record_flag & (~RecordFlag.SUCCESS)

        if num_episode < num_record_episodes:

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
            if custom_buf.is_available():
                if custom_record_callback is not None:
                    custom_buf.append(i, custom_record_callback(i, _locals, _globals))

            if done:
                vedio_record_callback(
                    num_episode, 
                    vedio_buf.get_res(i), #type:ignore 
                    reward_buf.get_res(i), #type:ignore 
                    critic_buf.get_res(i), #type:ignore
                    custom_buf.get_res(i), #type:ignore
                )

                num_episode += 1

                if pbar != None:
                    pbar.update()

        if addition_callback is not None:
            addition_callback(_locals, _globals)

    eval_res = evaluate_policy(model, env, n_eval_episodes, deterministic, callback = eval_record_callback, **kwargs)
    
    if pbar != None:
        pbar.close()

    success_rate = None
    if RecordFlag.SUCCESS in record_flag:
        success_rate = float(success_evals / n_eval_episodes)

    return eval_res, success_rate

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

    custom_record_callback: Optional[Callable[[int, Dict, Dict], np.ndarray]] = None,
    # fun(num_ep, custom_list, tb_writer)
    custom_tb_callback: Optional[Callable[[int, np.ndarray, SummaryWriter], None]] = None,
    addition_callback: Optional[Callable[[Dict, Dict], None]] = None,

    **kwargs
):
    '''
    将过程记录为 Tensorboard 视频
    '''
    def tb_callback(num_ep, vedio, reward_list, critic_list, custom_list):
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

        if custom_list is not None and custom_tb_callback is not None:
            custom_tb_callback(num_ep, custom_list, tb_writer)

    return eval_record(
        model, env, tb_callback, num_record_episodes, record_flag, verbose, n_eval_episodes, deterministic, custom_record_callback = custom_record_callback, addition_callback = addition_callback, **kwargs
    )

def eval_record_to_file(
    model: "type_aliases.PolicyPredictor",
    env: Union[VecEnv, VecEnvWrapper],

    save_root: Union[Path, str],
    save_name_prefix: Optional[str] = None,
    use_timestamp: bool = True,
    is_save_return_plot: bool = True,

    record_flag: RecordFlag = RecordFlag.NORMAL,
    fps: int = 25,
    return_gamma: float = 1,

    num_record_episodes: Optional[int] = None,
    verbose: int = 0,
    n_eval_episodes: int = 10,
    deterministic: bool = True,

    custom_record_callback: Optional[Callable[[int, Dict, Dict], np.ndarray]] = None,
    custom_axe_callback: Optional[Callable[[int, np.ndarray, Axes], None]] = None,
    addition_callback: Optional[Callable[[Dict, Dict], None]] = None,
    
    **kwargs
):
    '''
    将过程记录为文件
    '''
    # if RecordFlag.VEDIO in record_flag:
    try:
        from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
    except ImportError as e:  # pragma: no cover
        raise Exception("MoviePy is not installed") from e

    save_root = Path(save_root)
    if use_timestamp:
        dir_name = get_file_time_str()
        if save_name_prefix is not None:
            dir_name = save_name_prefix + dir_name
            
        save_root = save_root.joinpath(dir_name)
    
    if not save_root.exists():
        os.makedirs(save_root)
    vedio_name_pattern = save_root.joinpath("vedio_{}.gif").as_posix()
    reward_plot_name_pattern = save_root.joinpath("reward_{}.png").as_posix()
    custom_plot_name_pattern = save_root.joinpath("custom_{}.png").as_posix()

    reward_buf = []
    critic_buf = []
    custom_buf = []

    def tb_callback(num_ep, vedio, reward_list, critic_list, custom_list):
        if vedio is not None:
            # 拆分为 List
            clip = ImageSequenceClip(sequence = [(img_slice * 255).astype(np.uint8) for img_slice in vedio], fps = fps)
            clip.write_gif(vedio_name_pattern.format(num_ep), fps = fps, logger = None)

        if reward_list is not None:
            total_length = len(reward_list)
            ep_return = 0
            ep_return_list = np.zeros(total_length)

            for t in range(total_length):
                ep_return = reward_list[-1 - t] + ep_return * return_gamma
                ep_return_list[total_length - t - 1] = ep_return

            reward_buf.append(ep_return_list)

            if is_save_return_plot:
                fig, axe = plt.subplots()
                axe.plot(ep_return_list)
                axe.set_xlabel("Time Step")
                axe.set_ylabel("Return")
                axe.set_title(f"Eval {num_ep} return-time")
                fig.savefig(reward_plot_name_pattern.format(num_ep))

                plt.close(fig)

        if critic_list is not None:
            critic_buf.append(np.asarray(critic_list))

        if custom_list is not None:
            if is_save_return_plot and custom_axe_callback is not None:
                fig, axe = plt.subplots()
                custom_axe_callback(num_ep, np.asarray(custom_list), axe)
                fig.savefig(custom_plot_name_pattern.format(num_ep))

                plt.close(fig)

            custom_buf.append(np.asarray(custom_list))

    origin_res, success_rate = eval_record(
        model, env, tb_callback, num_record_episodes, record_flag, verbose, n_eval_episodes, deterministic, addition_callback = addition_callback, custom_record_callback = custom_record_callback, return_episode_rewards = True, **kwargs
    )

    assert isinstance(origin_res[0], list) and isinstance(origin_res[1], list), "保证参数 return_episode_rewards 为 True"
    return_list = np.asarray(origin_res[0])
    lenth_list = np.asarray(origin_res[1])

    if len(reward_buf) > 0:
        with open(save_root.joinpath("reward.npz"), 'wb') as f:
            np.savez(
                f, 
                *reward_buf
            )

    if len(critic_buf) > 0:
        with open(save_root.joinpath("critic.npz"), 'wb') as f:
            np.savez(
                f, 
                *critic_buf
            )

    if len(custom_buf) > 0:
        with open(save_root.joinpath("custom.npz"), 'wb') as f:
            np.savez(
                f, 
                *custom_buf
            )

    if is_save_return_plot:

        fig, axe = plt.subplots()
        axe.hist(return_list)

        title_str = f"ra: {np.mean(return_list):.3f} rs: {np.std(return_list):.3f} al: {np.mean(lenth_list):.3f} "
        if success_rate is not None:
            title_str += f"sr: {success_rate:.3f}"

        axe.set_title(title_str)
        fig.savefig(reward_plot_name_pattern.format("summary"))

        plt.close(fig)

    return origin_res, save_root