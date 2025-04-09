import os
from pathlib import Path
from typing import Any, Dict, Optional, Union
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
import numpy as np
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common import type_aliases

from ...utility import get_file_time_str
from ...sb3.eval_record import RecordFlag, eval_record_to_file
from .plane_box import PlaneBoxEnv

def planebox_eval(
    model: "type_aliases.PolicyPredictor",
    env: PlaneBoxEnv,
    
    save_root: Union[Path, str],
    save_name_prefix: Optional[str] = None,
    use_timestamp: bool = True,
    is_save_return_plot: bool = True,

    record_flag: RecordFlag = RecordFlag.NORMAL,
    fps: int = 25,

    num_record_episodes: int = 5,
    n_eval_episodes: int = 10,
    deterministic: bool = True,

    is_parameter_space: bool = True,
    **kwargs
):
    
    # save_root = Path(save_root)
    # if use_timestamp:
    #     dir_name = get_file_time_str()
    #     if save_name_prefix is not None:
    #         dir_name = save_name_prefix + dir_name
            
    #     save_root = save_root.joinpath(dir_name)
    # if not save_root.exists():
    #     os.makedirs(save_root)
    # plot_name_pattern = save_root.joinpath("{}.png").as_posix()

    pos_diff_list = []
    rot_diff_list = []

    done_continue_critic_list = []
    done_done_critic_list = []
    done_success_list = []
    done_pos_act_scale_list = []
    done_rot_act_scale_list = []

    def eval_record_callback(_locals: Dict[str, Any], _globals: Dict[str, Any]):
        nonlocal pos_diff_list
        nonlocal rot_diff_list

        i = _locals["i"]
        acts = _locals["actions"][i]

        done = _locals["done"]
        info = _locals["info"]

        if done:
            pos_diff_list.append(info["pos_diff"])
            rot_diff_list.append(info["rot_diff"])

            if is_parameter_space:
                done_continue_critic_list.append(acts[6])
                done_done_critic_list.append(acts[7])
                done_pos_act_scale_list.append(np.linalg.norm(acts[:3]))
                done_rot_act_scale_list.append(np.linalg.norm(acts[3:6]))

            done_success_list.append(info["is_success"])
    
    def custom_reward_callback(i: int, _locals: Dict[str, Any], _globals: Dict[str, Any]):
        return _locals["actions"][i, 6:]

    def custom_plot_callback(num_ep: int, data: np.ndarray, axe: Axes):
        # print(f"{data.shape}")
        axe.plot(data[:, 0])
        axe.plot(data[:, 1])
        axe.set_xlabel("Time Step")
        axe.set_ylabel("Action Critic")
        axe.legend(["Continue", "Done"])
        axe.set_title(f"Eval {num_ep} action-time")

    if is_parameter_space:
        (ep_reward_list, ep_length_list), save_root = eval_record_to_file(
            model, 
            env, 

            save_root = save_root,
            save_name_prefix = save_name_prefix,
            use_timestamp = use_timestamp,
            is_save_return_plot = is_save_return_plot,

            record_flag = record_flag,
            fps = fps,

            num_record_episodes = num_record_episodes,
            n_eval_episodes = n_eval_episodes, deterministic = deterministic, 
            custom_axe_callback = custom_plot_callback,
            custom_record_callback = custom_reward_callback,
            addition_callback = eval_record_callback, 
            **kwargs
        )
    else:
        (ep_reward_list, ep_length_list), save_root = eval_record_to_file(
            model, 
            env, 

            save_root = save_root,
            save_name_prefix = save_name_prefix,
            use_timestamp = use_timestamp,
            is_save_return_plot = is_save_return_plot,

            record_flag = record_flag,
            fps = fps,

            num_record_episodes = num_record_episodes,
            n_eval_episodes = n_eval_episodes, deterministic = deterministic, 
            custom_axe_callback = None,
            custom_record_callback = None,
            addition_callback = eval_record_callback, 
            **kwargs
        )

    plot_name_pattern = save_root.joinpath("{}.png").as_posix()

    pos_diff_array = np.array(pos_diff_list)
    rot_diff_array = np.array(rot_diff_list)

    # assert isinstance(ep_reward_list, list) and isinstance(ep_length_list, list), "return_episode_rewards 需要为 True"

    # ep_reward_array = np.asarray(ep_reward_list)
    # ep_length_array = np.asarray(ep_length_list)

    with open(save_root.joinpath("diff.npz"), 'wb') as f:
        np.savez(
            f, 
            pos_diff = pos_diff_array,
            rot_diff = rot_diff_array,
            # ep_reward = ep_reward_array,
            # ep_length = ep_length_array
        )
    if is_save_return_plot:

        fig, axes = plt.subplot_mosaic([[0, 1]])
        fig.set_layout_engine("compressed")
        axes[0].hist(pos_diff_array)
        axes[1].hist(rot_diff_array)

        axes[0].set_title(f"pos a:{np.mean(pos_diff_array):.2e}, s:{np.std(pos_diff_array):.2e}")
        axes[1].set_title(f"rot a:{np.mean(rot_diff_array):.2e}, s:{np.std(rot_diff_array):.2e}")

        fig.savefig(plot_name_pattern.format("pos_diff"))
        plt.close(fig)

        ########

        if is_parameter_space:
            done_continue_critic_array = np.asarray(done_continue_critic_list)
            done_done_critic_array = np.asarray(done_done_critic_list)
            done_success_array = np.asarray(done_success_list)
            done_pos_act_scale_array = np.asarray(done_pos_act_scale_list)
            done_rot_act_scale_array = np.asarray(done_rot_act_scale_list)

            # done_softmax_critic_array = np.exp(done_done_critic_array) / (np.exp(done_done_critic_array) + np.exp(done_continue_critic_list))

            fig, axes = plt.subplot_mosaic([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
            fig.set_layout_engine("compressed")

            axes[0].scatter(done_continue_critic_array, pos_diff_array)
            axes[1].scatter(done_done_critic_array, pos_diff_array)
            axes[2].scatter(done_pos_act_scale_array, pos_diff_array)

            axes[3].scatter(done_continue_critic_array, rot_diff_array)
            axes[4].scatter(done_done_critic_array, rot_diff_array)
            axes[5].scatter(done_rot_act_scale_array, rot_diff_array)

            axes[6].scatter(done_continue_critic_array, done_success_array)
            axes[7].scatter(done_done_critic_array, done_success_array)
            axes[8].scatter(done_pos_act_scale_array + done_rot_act_scale_array, done_success_array)

            axes[0].set_title("continue")
            axes[1].set_title("done")
            axes[2].set_title("act scale")

            fig.savefig(plot_name_pattern.format("pos_diff_compare"))
            plt.close(fig)

    return ep_reward_list, ep_length_list, pos_diff_list, rot_diff_list
