from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvWrapper

from pathlib import Path
from typing import Optional, Union
import numpy as np
import warnings

from optuna.trial import Trial
from torch.utils.tensorboard.writer import SummaryWriter

from .eval_record import eval_record_to_tensorbard, RecordFlag

# class OptunaFinetuningCallback(BaseCallback):
#     def __init__(
#             self, 
#             # model_save_root: Union[str, Path],

#             optuna_trial: Trial,
#             optuna_is_use_pruner: bool,

#             tb_writer: SummaryWriter,
#             tb_tag_root: str,
#             tb_record_episodes: int = 2,
#             tb_record_flag: RecordFlag = RecordFlag.NORMAL,
#             tb_fps: int = 20,

#             eval_freq: int = 500,
#             eval_num_episodes: int = 10,
#             eval_deterministic: bool = True,
#             eval_env: Optional[Union[VecEnv, VecEnvWrapper]] = None,

#             verbose = 0,
#         ):
#         '''
#         用于 Optuna 超参数搜索的 Callback
#         '''
#         super().__init__(verbose)

#         # # 模型保存路径
#         # if not isinstance(model_save_root, Path):
#         #     self.model_save_path = Path(model_save_root)
#         # else:
#         #     self.model_save_path = model_save_root
#         # self.model_best_save_path = self.model_save_path.joinpath("best_model")
#         # self.model_last_save_path = self.model_save_path.joinpath("last_model")

#         # 模型评估相关参数
#         self.eval_freq = eval_freq
#         self.eval_num_episodes = eval_num_episodes
#         self.eval_deterministic = eval_deterministic
#         if eval_env == None:
#             warnings.warn("正在使用训练环境评估模型", UserWarning)
#             self.eval_env = None
#         else:
#             self.eval_env = eval_env
#         # Optuna 参数
#         self.optuna_trail = optuna_trial
#         self.optuna_is_use_pruner = optuna_is_use_pruner
#         # TB 参数
#         self.tb_writer = tb_writer

#         self.tb_hist_tag = tb_tag_root + "/hist"
#         self.tb_tag_root = tb_tag_root

#         self.tb_record_episodes = tb_record_episodes
#         self.tb_record_flag = tb_record_flag
#         self.tb_fps = tb_fps

#         # 记录参数
#         self.best_result = -np.inf
#         self.eval_times = 0
#         self.last_result = 0.0
#         self._is_prune = False

#     def _on_training_start(self) -> None:
#         if self.eval_env == None:
#             self.eval_env = self.training_env
#         return 

#     def _eval_model(self, is_last_eval: bool = False):
#         self.eval_times += 1
#         print(f"第 {self.eval_times} 次性能测试开始")

#         if is_last_eval:
#             trail_return_list, _ = eval_record_to_tensorbard(
#                 self.model,
#                 self.eval_env, # type: ignore[_on_training_start]
#                 tb_writer = self.tb_writer,
#                 root_tag = self.tb_tag_root,
#                 record_flag = self.tb_record_flag,
#                 num_record_episodes = self.tb_record_episodes,
#                 n_eval_episodes = self.eval_num_episodes,
#                 deterministic = self.eval_deterministic,

#                 return_episode_rewards = True
#             )
#         else:
#             trail_return_list, _ = evaluate_policy(
#                 self.model,
#                 self.eval_env, # type: ignore[_on_training_start]
#                 self.eval_num_episodes,
#                 self.eval_deterministic,
#                 return_episode_rewards = True
#             )
#         assert isinstance(trail_return_list, list), "请勿关闭 return_episode_rewards"
#         trail_return_list = np.array(trail_return_list)

#         self.tb_writer.add_histogram(
#             self.tb_hist_tag, trail_return_list, self.num_timesteps
#         )

#         # self.model.save(self.model_last_save_path)
#         # if trail_result > self.best_result: # type: ignore[float]
#         #     self.model.save(self.model_best_save_path)
#         #     self.best_result = trail_result

#         self.last_result = float(np.mean(trail_return_list))
#         return self.last_result

#     def _on_step(self) -> bool:

#         total_timesteps = self.locals["total_timesteps"]

#         # 每个间隔或到达最后一步时进行测试
#         is_last_eval = (self.num_timesteps == total_timesteps)
#         # 不用求余, 并行环境下 num_timesteps 一次增加 num_envs 个
#         eval_step = self.eval_freq * (self.eval_times + 1)

#         if self.num_timesteps >= eval_step or is_last_eval:
#             trail_result = self._eval_model(is_last_eval)

#             if self.optuna_is_use_pruner:
#                 self.optuna_trail.report(trail_result, self.eval_times)
#                 self._is_prune = self.optuna_trail.should_prune()
#                 return not self._is_prune
#             else:
#                 return True
#         else:
#             return True
        
#     def is_prune(self):
#         return self._is_prune

#     def get_last_result(self):
#         return self.last_result