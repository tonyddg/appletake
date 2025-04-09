from typing import Any, Callable, Dict, List, Literal, Optional, Protocol, Sequence, Tuple, Union
from abc import ABCMeta, abstractmethod

import numpy as np

from .plane_box import PlaneBoxSubenvBase, RewardFnABC

# reward = reward_fn(subenv, is_alignments, is_collisions)

def clip_ratio(obj: float, fraction: float):
    return max((fraction - obj) / fraction, 0)

# (self, subenv: PlaneBoxSubenvBase, is_alignment: bool, is_collision: bool, is_execute_align: bool) -> float:
# RewardFnType = Union[Callable[[PlaneBoxSubenvBase, bool, bool, bool, bool], float], RewardFnABC]
# RewardFnType = Union[Callable[[PlaneBoxSubenvBase, bool, bool, bool, bool], float], RewardFnABC]

class RewardSpare(RewardFnABC):
    def __init__(
            self, 
            time_panelty: float = -0.1, 
            colision_panelty: float = -1, 
            align_fail_panelty: float = -1,
            success_reward: float = 1
        ) -> None:
        self.time_panelty = time_panelty
        self.colision_panelty = colision_panelty
        self.align_fail_panelty = align_fail_panelty
        self.success_reward = success_reward

    def __call__(self, subenv: PlaneBoxSubenvBase, is_alignment: bool, is_collision: bool, is_execute_align: bool, is_timeout: bool) -> float:
        if is_collision:
            return self.colision_panelty
        # 主动插入中, 如果不进行插入决策则永远为未对齐状态
        elif is_alignment:
            return self.success_reward
        elif is_execute_align:
            return self.align_fail_panelty
        else:
            return self.time_panelty

class RewardLinearDistance(RewardFnABC):
    def __init__(
            self, 
            # 用于基本范围
            max_pos_dis_mm: float = 40,
            max_rot_dis_deg: float = 8,
            time_panelty: float = -0.1, 
            colision_panelty: float = -1, 
            align_fail_panelty: float = -1,
            success_reward: float = 1
        ) -> None:
        '''
        基于相对 best pose 给出奖励

        使用轴角对表示角度误差
        '''
        self.max_pos_dis = max_pos_dis_mm * 1e-3
        self.max_rot_dis = float(max_rot_dis_deg)
        self.time_panelty = time_panelty
        self.colision_panelty = colision_panelty
        self.align_fail_panelty = align_fail_panelty
        self.success_reward = success_reward

    def __call__(self, subenv: PlaneBoxSubenvBase, is_alignment: bool, is_collision: bool, is_execute_align: bool, is_timeout: bool) -> float:
        if is_collision:
            return self.colision_panelty
        elif is_alignment:
            return self.success_reward
        elif is_execute_align:
            return self.align_fail_panelty
        else:
            # # cur_pose = subenv.anchor_move_object.get_pose(subenv.anchor_core_object)
            # pose_diff = subenv.move_box.get_pose(subenv.anchor_best_align_movebox_pose)

            # # xyz 三方向均计入惩罚
            # diff_len = np.linalg.norm(pose_diff[:3], 2)
            # # diff_rad = get_quat_diff_rad(cur_pose[3:], subenv.best_pose_move_anchor_relcore[3:])
            # diff_deg, _ = quat_to_rotvec(pose_diff[3:])

            diff_len, diff_deg = subenv.get_dis_norm_to_best()

            reach_xy = max(float(self.max_pos_dis - diff_len), 0.0) / self.max_pos_dis
            reach_ang = max(self.max_rot_dis - diff_deg, 0) / self.max_rot_dis
            reach_reward = (reach_xy + reach_ang) / 2

            return (1 - reach_reward) * self.time_panelty

class RewardLinearDistanceAndAlignQuality(RewardFnABC):
    def __init__(
            self, 
            # 用于基本范围
            max_pos_dis_mm: float = 40,
            max_rot_dis_deg: float = 8,

            time_panelty: float = -0.1, 
            colision_panelty: float = -1, 
            align_fail_panelty: float = -1,
            
            success_reward: float = 1,
            max_align_pos_dis_mm: float = 18,
            max_align_rot_dis_deg: float = 2,

            is_attract_to_center: bool = False,
            is_square_attract: bool = False,

        ) -> None:
        '''
        基于相对 best pose 给出奖励

        使用轴角对表示角度误差
        '''
        self.max_pos_dis = max_pos_dis_mm * 1e-3
        self.max_rot_dis = float(max_rot_dis_deg)
        self.time_panelty = time_panelty
        self.colision_panelty = colision_panelty
        self.align_fail_panelty = align_fail_panelty

        self.success_reward = success_reward
        self.max_align_pos_dis = max_align_pos_dis_mm * 1e-3
        self.max_align_rot_dis = float(max_align_rot_dis_deg)

        self.is_attract_to_center = is_attract_to_center
        self.is_square_attract = is_square_attract

    def __call__(self, subenv: PlaneBoxSubenvBase, is_alignment: bool, is_collision: bool, is_execute_align: bool, is_timeout: bool) -> float:
        if is_collision:
            return self.colision_panelty
        elif is_alignment:
            diff_len, diff_deg = subenv.get_dis_norm_to_best(self.is_attract_to_center)

            if self.is_square_attract:
                reach_len = max(float(self.max_align_pos_dis - diff_len), 0.0) ** 2 / (self.max_align_pos_dis) ** 2
                reach_ang = max((self.max_align_rot_dis - diff_deg), 0) ** 2 / (self.max_align_rot_dis) ** 2
            else:
                reach_len = max(float(self.max_align_pos_dis - diff_len), 0.0) / (self.max_align_pos_dis)
                reach_ang = max((self.max_align_rot_dis - diff_deg), 0) / (self.max_align_rot_dis)

            reach_reward = (reach_len + reach_ang) / 2

            return (1 + reach_reward) * self.success_reward

        elif is_execute_align:
            return self.align_fail_panelty
        else:
            diff_len, diff_deg = subenv.get_dis_norm_to_best(self.is_attract_to_center)

            reach_len = max(float(self.max_pos_dis - diff_len), 0.0) / self.max_pos_dis
            reach_ang = max(self.max_rot_dis - diff_deg, 0) / self.max_rot_dis
            reach_reward = (reach_len + reach_ang) / 2

            return (1 - reach_reward) * self.time_panelty

class RewardApproachAndAlignQualityAndTimeout(RewardFnABC):
    def __init__(
            self, 
            # 用于基本范围
            time_panelty: float = -0.1, 
            colision_panelty: float = -1, 
            align_fail_panelty: float = -1,
            timeout_panelty: float = -1,

            success_reward: float = 1,
            max_align_reward: float = 5,
            max_move_reward: float = 0.1,

            max_move_pos_dis_mm: float = 8,
            max_move_rot_dis_deg: float = 2,

            max_align_pos_dis_mm: float = 8,
            max_align_rot_dis_deg: float = 2,

            is_attract_to_center: bool = False,
        ) -> None:
        '''
        基于相对 best pose 给出奖励

        使用轴角对表示角度误差
        '''
        self.time_panelty = time_panelty
        self.colision_panelty = colision_panelty
        self.align_fail_panelty = align_fail_panelty
        self.timeout_panelty = timeout_panelty

        self.success_reward = success_reward
        self.max_align_reward = max_align_reward
        self.max_align_pos_dis = max_align_pos_dis_mm * 1e-3
        self.max_align_rot_dis = float(max_align_rot_dis_deg)

        self.max_move_reward = max_move_reward
        self.max_move_pos_dis = max_move_pos_dis_mm * 1e-3
        self.max_move_rot_dis = float(max_move_rot_dis_deg)

        self.is_attract_to_center = is_attract_to_center

    def reset(self, subenv: PlaneBoxSubenvBase):
        pos_dis, rot_dis = subenv.get_dis_norm_to_best(self.is_attract_to_center)

        subenv.episode_info["last_pos_dis"] = pos_dis
        subenv.episode_info["last_rot_dis"] = rot_dis

    def __call__(self, subenv: PlaneBoxSubenvBase, is_alignment: bool, is_collision: bool, is_execute_align: bool, is_timeout: bool) -> float:
        if is_collision:
            return self.colision_panelty
        
        elif is_timeout:
            return self.timeout_panelty
        
        elif is_alignment:
            pos_dis, rot_dis = subenv.get_dis_norm_to_best(self.is_attract_to_center)

            pos_reach_ratio = clip_ratio(float(pos_dis), self.max_align_pos_dis) # max(float(self.max_align_pos_dis - pos_dis), 0.0) / (self.max_align_pos_dis)
            rot_reach_ratio = clip_ratio(float(rot_dis), self.max_align_rot_dis) # max((self.max_align_rot_dis - rot_dis), 0) / (self.max_align_rot_dis)

            reach_reward = (pos_reach_ratio + rot_reach_ratio) / 2

            return reach_reward * self.max_align_reward + self.success_reward

        elif is_execute_align:
            return self.align_fail_panelty
        
        else:
            pos_dis, rot_dis = subenv.get_dis_norm_to_best(self.is_attract_to_center)
            last_pos_dis = subenv.episode_info["last_pos_dis"]
            last_rot_dis = subenv.episode_info["last_rot_dis"]

            pos_approach_ratio = float(last_pos_dis - pos_dis) / self.max_move_pos_dis
            rot_approach_ratio = float(last_rot_dis - rot_dis) / self.max_move_rot_dis

            approach_reward = self.max_move_reward * (pos_approach_ratio + rot_approach_ratio) / 2

            subenv.episode_info["last_pos_dis"] = pos_dis
            subenv.episode_info["last_rot_dis"] = rot_dis

            return approach_reward + self.time_panelty

class RewardPassiveTimeout(RewardFnABC):
    def __init__(
            self, 
            # 基本惩罚
            time_panelty: float = -0.1, 
            colision_panelty: float = -1, 
            timeout_panelty: Optional[float] = None,

            is_attract_to_center: bool = False,
            is_square_align_reward: bool = False,

            # 插入判断
            align_success_reward: float = 1,
            align_fail_panelty: float = -1,
            max_align_reward: float = 4,
            
            max_align_pos_dis_mm: float = 8,
            max_align_rot_dis_deg: float = 2,

            # 接近判断
            max_approach_reward: float = 0.1,

            max_approach_pos_dis_mm: float = 40,
            max_approach_rot_dis_deg: float = 5,

            # 良好移动判断
            max_move_reward: float = 0.1,
            
            max_move_pos_dis_mm: float = 8,
            max_move_rot_dis_deg: float = 2,

        ) -> None:
        '''
        '''
        self.time_panelty = time_panelty
        self.colision_panelty = colision_panelty
        self.align_fail_panelty = align_fail_panelty
        self.timeout_panelty = timeout_panelty

        self.align_success_reward = align_success_reward
        self.max_align_reward = max_align_reward
        self.max_align_pos_dis = max_align_pos_dis_mm * 1e-3
        self.max_align_rot_dis = float(max_align_rot_dis_deg)

        self.max_approach_reward = max_approach_reward
        self.max_approach_pos_dis = max_approach_pos_dis_mm * 1e-3
        self.max_approach_rot_dis = float(max_approach_rot_dis_deg)

        self.max_move_reward = max_move_reward
        self.max_move_pos_dis = max_move_pos_dis_mm * 1e-3
        self.max_move_rot_dis = float(max_move_rot_dis_deg)

        self.is_attract_to_center = is_attract_to_center
        self.is_square_align_reward = is_square_align_reward

    def before_action(self, subenv: PlaneBoxSubenvBase):
        pos_dis, rot_dis = subenv.get_dis_norm_to_best(self.is_attract_to_center)

        subenv.episode_info["last_pos_dis"] = pos_dis
        subenv.episode_info["last_rot_dis"] = rot_dis

    def __call__(self, subenv: PlaneBoxSubenvBase, is_alignment: bool, is_collision: bool, is_execute_align: bool, is_timeout: bool) -> float:
        if is_collision:
            return self.colision_panelty

        elif is_alignment:
            pos_dis, rot_dis = subenv.get_dis_norm_to_best(self.is_attract_to_center)

            pos_align_ratio = clip_ratio(float(pos_dis), self.max_align_pos_dis) # max(float(self.max_align_pos_dis - pos_dis), 0.0) / (self.max_align_pos_dis)
            rot_align_ratio = clip_ratio(float(rot_dis), self.max_align_rot_dis) # max((self.max_align_rot_dis - rot_dis), 0) / (self.max_align_rot_dis)
            
            if self.is_square_align_reward:
                align_reward = (pos_align_ratio ** 2 + rot_align_ratio ** 2) / 2
            else:
                align_reward = (pos_align_ratio + rot_align_ratio) / 2

            return align_reward * self.max_align_reward + self.align_success_reward
        
        elif is_timeout and self.timeout_panelty is not None:
            return self.timeout_panelty
        
        # elif is_execute_align:
        #     return self.align_fail_panelty
        
        else:
            pos_dis, rot_dis = subenv.get_dis_norm_to_best(self.is_attract_to_center)
            last_pos_dis = subenv.episode_info["last_pos_dis"]
            last_rot_dis = subenv.episode_info["last_rot_dis"]

            pos_move_ratio = float(np.clip(last_pos_dis - pos_dis / self.max_move_pos_dis, -1, 1))
            rot_move_ratio = float(np.clip(last_rot_dis - rot_dis / self.max_move_rot_dis, -1, 1))
            move_reward = self.max_move_reward * (pos_move_ratio + rot_move_ratio) / 2

            pos_approach_ratio = clip_ratio(float(pos_dis), self.max_approach_pos_dis) # max(float(self.max_align_pos_dis - pos_dis), 0.0) / (self.max_align_pos_dis)
            rot_approach_ratio = clip_ratio(float(rot_dis), self.max_approach_rot_dis) # max((self.max_align_rot_dis - rot_dis), 0) / (self.max_align_rot_dis)
            approach_reward = self.max_approach_reward * (pos_approach_ratio + rot_approach_ratio) / 2

            totla_reward = move_reward + approach_reward + self.time_panelty
            if is_execute_align:
                totla_reward = approach_reward + self.time_panelty + self.align_fail_panelty
            
            return totla_reward
