import os
from typing import Optional, Union, Callable, Dict, Any, List
from pathlib import Path
from optuna import Trial
from torch.utils.tensorboard.writer import SummaryWriter
import numpy as np
import warnings

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvWrapper
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm

from .eval_record import eval_record_to_tensorbard, RecordFlag, evaluate_policy

# TODO: 区分训练环境与测试环境
# TODO: 折扣回报率
# TODO: Optuna 筛选模型
# TODO: 保存成功率

class TensorboardEvalCallback(BaseCallback):
    def __init__(
            self, 
            tb_writer: Optional[SummaryWriter],

            eval_freq: int = 500,
            n_eval_episodes: int = 10,
            deterministic: bool = True,
            return_gamma: float = 1,
            eval_env: Optional[Union[VecEnv, VecEnvWrapper]] = None,
            
            tb_eval_tag_prefix: str = "trail_",
            tb_eval_summary_root: str = "summary",
            tb_record_flag: RecordFlag = RecordFlag.NORMAL,

            tb_record_last_only: bool = False,

            # # "train/critic_loss"
            # tb_log_critic_loss_name: Optional[str] = None,
            # # "train/actor_loss"
            # tb_log_actor_loss_name: Optional[str] = None,
            
            # 从 stable-baseline 的 logger 中记录数据
            # 键: tb 标签, 值: logger 路径
            tb_rollout_log_record_dict: Optional[dict[str, str]] = None,

            save_last_model: bool = True,
            save_best_model: bool = True,
            save_replay_buff: bool = True,
            save_data: Optional[Union[str, Path]] = "data.npz",
            save_root: Optional[Union[str, Path]] = None,

            num_record_episodes: Optional[int] = 2,
            fps: int = 25,
            verbose = 0,

            **eval_kwargs
        ):
        '''
        基于 Tensorboard 记录训练评估结果 (视频与性能) 同时保存最优与最新模型
        '''
        super().__init__(verbose)

        self.eval_freq = eval_freq

        ### 评估函数相关属性
        self.deterministic = deterministic
        self.num_record_episodes = num_record_episodes
        self.n_eval_episodes = n_eval_episodes
        self.eval_kwargs = eval_kwargs
        self.return_gamma = return_gamma
        if eval_env == None:
            warnings.warn("正在使用训练环境评估模型", UserWarning)
            self.eval_env = None
        else:
            self.eval_env = eval_env

        ### Tensorboard 相关属性
        self.tb_eval_summary_root = tb_eval_summary_root
        # 使用平均的方式记录结果
        self.tb_eval_return_avg_tag = self.tb_eval_summary_root + "/eval_return_avg"
        # 使用直方图的方式记录结果
        self.tb_eval_return_hist_tag = self.tb_eval_summary_root + "/eval_return_hist"
        # 使用长度记录结果
        self.tb_eval_length_avg_tag = self.tb_eval_summary_root + "/eval_length_avg"
        # 使用成功率记录结果
        self.tb_eval_success_rate_tag = self.tb_eval_summary_root + "/eval_success_rate"
        
        # 记录损失
        # self.tb_log_critic_loss_name = tb_log_critic_loss_name
        # self.tb_log_actor_loss_name = tb_log_actor_loss_name
        # self.tb_update_actor_loss_tag = tb_eval_summary_root + "/update_actor_loss"
        # self.tb_update_critic_loss_tag = tb_eval_summary_root + "/update_critic_loss"
        self.tb_rollout_log_record_dict = tb_rollout_log_record_dict

        # 记录各次实验结果
        self.tb_eval_tag_prefix = tb_eval_tag_prefix
        self.tb_record_flag = tb_record_flag
        self.fps = fps
        self.tb_record_last_only = tb_record_last_only

        self.best_result = -np.inf
        self.last_result = -np.inf
        self.eval_times = 0
        self.total_timesteps = 0

        # 数据保存相关设置
        if save_root is not None:
            self.save_last_model = save_last_model
            self.save_best_model = save_best_model
            self.save_replay_buff = save_replay_buff
            self.save_data = save_data

            if not isinstance(save_root, Path):
                self.save_root = Path(save_root)
            else:
                self.save_root = save_root
            if not self.save_root.exists():
                os.mkdir(self.save_root)
        else:
            self.save_last_model = False
            self.save_best_model = False
            self.save_replay_buff = False
            self.save_data = None

        if self.save_best_model:
            self.best_model_save_path = self.save_root.joinpath("best_model")
        if self.save_last_model:
            self.last_model_save_path = self.save_root.joinpath("last_model")
        if save_replay_buff:
            self.buff_save_path = self.save_root.joinpath("replay_buff")
            # if isinstance(self.model, OffPolicyAlgorithm):
            #     self.buff_save_path = self.save_root.joinpath("replay_buff")
            # else:
            #     warnings.warn("非 Off Policy 策略没有回放队列")
            #     self.save_replay_buff = False
        if self.save_data is not None:
            self.data_save_path = self.save_root.joinpath(self.save_data)
            self.episode_return_hist_buf = []
            self.episode_length_hist_buf = []

        if tb_writer is not None:
            self.tb_writer = tb_writer
        elif self.save_root is not None:
            self.tb_writer = SummaryWriter(log_dir = self.save_root.as_posix())
        else:
            raise Exception("没有提供 tb_writer 对象或保存路径")

    def _on_training_start(self) -> None:
        if self.eval_env == None:
            self.eval_env = self.training_env
        self.total_timesteps = self.locals["total_timesteps"]
        return 

    def _eval_model(self, is_last_eval: bool):

        self.eval_times += 1

        # 评估结果

        if self.tb_record_last_only and not is_last_eval:
            (episode_return_list, episode_length_list), success_rate = eval_record_to_tensorbard(
                model = self.model,
                env = self.eval_env, # type: ignore[_on_training_start]
                tb_writer = self.tb_writer,
                # 单次评估按次数划分
                root_tag = self.tb_eval_tag_prefix + str(self.eval_times),
                record_flag = self.tb_record_flag,
                fps = self.fps,
                num_record_episodes = 0, # 启用跳过最后一帧的模式时, 即设置为不记录任何 episode
                return_gamma = self.return_gamma,
                verbose = 0,
                n_eval_episodes = self.n_eval_episodes,
                deterministic = self.deterministic,
                # 记录奖励总结直方图
                return_episode_rewards = True,
                **self.eval_kwargs
            )
        
        else:
            (episode_return_list, episode_length_list), success_rate = eval_record_to_tensorbard(
                model = self.model,
                env = self.eval_env, # type: ignore[_on_training_start]
                tb_writer = self.tb_writer,
                # 单次评估按次数划分
                root_tag = self.tb_eval_tag_prefix + str(self.eval_times),
                record_flag = self.tb_record_flag,
                fps = self.fps,
                num_record_episodes = self.num_record_episodes,
                return_gamma = self.return_gamma,
                verbose = 0,
                n_eval_episodes = self.n_eval_episodes,
                deterministic = self.deterministic,
                # 记录奖励总结直方图
                return_episode_rewards = True,
                **self.eval_kwargs
            )
        assert isinstance(episode_return_list, List), f"episode_return_list 的类型应为 List, 得到 {type(episode_return_list)}"

        # 评估性能

        episode_return_list = np.asarray(episode_return_list)
        self.last_result = np.mean(episode_return_list)

        # 绘制性能总结  
        self.tb_writer.add_histogram(
            self.tb_eval_return_hist_tag,
            episode_return_list,
            self.num_timesteps,
        )
        self.tb_writer.add_scalar(
            self.tb_eval_return_avg_tag,
            self.last_result,
            self.num_timesteps,
        )
        self.tb_writer.add_scalar(
            self.tb_eval_length_avg_tag,
            np.mean(episode_length_list),
            self.num_timesteps,
        )
        if success_rate is not None:
            self.tb_writer.add_scalar(
                self.tb_eval_success_rate_tag,
                success_rate,
                self.num_timesteps,
            )

        # 保存模型
        if self.best_result < self.last_result:

            self.best_result = self.last_result
            
            if self.save_best_model:
                self.model.save(self.best_model_save_path)

        if self.save_last_model:
            self.model.save(self.last_model_save_path)
        if self.save_replay_buff:
            if isinstance(self.model, OffPolicyAlgorithm):
                self.model.save_replay_buffer(self.buff_save_path)
            else:
                warnings.warn("非 Off Policy 策略没有回放队列")
                self.save_replay_buff = False

        if self.save_data is not None:
            self.episode_return_hist_buf.append(episode_return_list)
            self.episode_length_hist_buf.append(episode_length_list)

    def _need_eval(self) -> tuple[bool, bool]:
        # 每个间隔或到达最后一步时进行测试
        is_last_eval = (self.num_timesteps == self.total_timesteps)
        # 不用求余, 并行环境下 num_timesteps 一次增加 num_envs 个
        eval_step = self.eval_freq * (self.eval_times + 1)

        return self.num_timesteps >= eval_step or is_last_eval, is_last_eval

    def _on_step(self) -> bool:
        # 每个间隔或到达最后一步时进行测试
        need_eval, is_last_eval = self._need_eval()
        if need_eval:
            self._eval_model(is_last_eval)
        
        return True
    
    def _on_training_end(self) -> None:
        if self.save_data:
            with open(self.data_save_path, 'wb') as f:
                np.savez(
                    f, 
                    episode_return_hist = np.stack(self.episode_return_hist_buf),
                    episode_length_hist = np.stack(self.episode_length_hist_buf)
                )
        self.tb_writer.close()

    def _on_rollout_start(self) -> None:

        if self.tb_rollout_log_record_dict is None:
            return

        iterations = self.num_timesteps # self.model.logger.name_to_value.get("train/n_updates", 0)
        for log_tag, log_path in self.tb_rollout_log_record_dict.items():
            val = self.model.logger.name_to_value.get(log_path, 0)
            self.tb_writer.add_scalar(
                self.tb_eval_summary_root + '/' + log_tag,
                val,
                iterations,
            )

        # if self.tb_log_critic_loss_name is not None:
        #     critic_loss = self.model.logger.name_to_value.get(self.tb_log_critic_loss_name, 0)
        #     self.tb_writer.add_scalar(
        #         self.tb_update_critic_loss_tag,
        #         critic_loss,
        #         updates,
        #     )

        # if self.tb_log_actor_loss_name is not None:
        #     actor_loss = self.model.logger.name_to_value.get(self.tb_log_actor_loss_name, 0)
        #     self.tb_writer.add_scalar(
        #         self.tb_update_actor_loss_tag,
        #         actor_loss,
        #         updates,
        #     )

class TensorboardEpReturnCallback(BaseCallback):
    def __init__(
            self, 
            tb_writer: SummaryWriter,
            tb_return_summary_root: str = "summary",

            verbose: int = 0
        ):
        super().__init__(verbose)

        self.tb_writer = tb_writer
        self.tb_return_summary_tag = tb_return_summary_root + "/train_ep_return"

        self._env_returns = np.zeros(1, np.float32)
        self._episode_count = 0
    
    def _on_training_start(self):
        self._env_returns = np.zeros(self.training_env.num_envs, np.float32)

    def _on_step(self) -> bool:
        rewards = self.locals["rewards"]
        dones = self.locals["dones"]

        self._env_returns += rewards
        for i in range(len(dones)):
            if dones[i]:
                episode_return = self._env_returns[i]
                self.tb_writer.add_scalar(self.tb_return_summary_tag, episode_return, self._episode_count)
                self._env_returns[i] = 0
                self._episode_count += 1

        return True

class OptunaEvalCallback(TensorboardEvalCallback):
    def __init__(
        self, 

        optuna_trial: Trial,
        optuna_is_use_pruner: bool,

        tb_writer: SummaryWriter,

        eval_freq: int = 500,
        n_eval_episodes: int = 10,
        deterministic: bool = True,
        return_gamma: float = 1,
        eval_env: Optional[Union[VecEnv, VecEnvWrapper]] = None,
        
        tb_eval_tag_prefix: str = "trail_",
        tb_eval_summary_root: str = "summary",
        tb_record_flag: RecordFlag = RecordFlag.NORMAL,
        tb_record_last_only: bool = False,

        # tb_log_critic_loss_name: Optional[str] = None,
        # tb_log_actor_loss_name: Optional[str] = None,
        tb_rollout_log_record_dict: Optional[dict[str, str]] = None,

        save_last_model: bool = True,
        save_best_model: bool = True,
        save_replay_buff: bool = True,
        save_data: Optional[Union[str, Path]] = "data.npz",
        save_root: Optional[Union[str, Path]] = None,

        num_record_episodes: Optional[int] = 2,
        fps: int = 25,
        verbose = 0,

        **eval_kwargs
    ):
        super().__init__(
            tb_writer, eval_freq, n_eval_episodes, deterministic, return_gamma, eval_env, tb_eval_tag_prefix, tb_eval_summary_root, tb_record_flag, tb_record_last_only, tb_rollout_log_record_dict, save_last_model, save_best_model, save_replay_buff, save_data, save_root, num_record_episodes, fps, verbose, **eval_kwargs)
        self.optuna_trial = optuna_trial
        self.optuna_is_use_pruner = optuna_is_use_pruner

    def _on_step(self) -> bool:
        # 需要在验证前判断, 否则步数将加一, 导致判断出错
        need_eval, _ = super()._need_eval()
        super()._on_step()
        if need_eval:
            if self.optuna_is_use_pruner:
                self.optuna_trial.report(float(self.last_result), self.eval_times)
                self._is_prune = self.optuna_trial.should_prune()
                return not self._is_prune
            else:
                return True
        else:
            return True

    def is_prune(self):
        return self._is_prune

    def get_last_result(self):
        return float(self.last_result)
