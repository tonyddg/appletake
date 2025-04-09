from pyrep import PyRep
from pyrep.objects.shape import Shape

from typing import Any, Dict, Literal, Optional, Sequence, Tuple, Union
import numpy as np
from albumentations.core.composition import TransformType

from ..utility import sample_float, set_pose6_by_self

from .plane_box import PlaneBoxEnv, RewardFnType, PlaneBoxSubenvBase

# TODO: 地面不平整扰动

class CornerSubEnv(PlaneBoxSubenvBase):

    def __init__(
        self, 
        name_suffix: str,

        obs_trans: Optional[TransformType],
        obs_source: Literal["color", "depth"],

        env_action_noise: Optional[np.ndarray] = None,
        env_init_box_pos_range: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        env_init_vis_pos_range: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        env_vis_persp_deg_disturb: Optional[float] = None,
        env_movbox_size_range: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        env_movebox_center_err: Optional[Tuple[np.ndarray, np.ndarray]] = None,
 
        # env_is_complexity_progression: bool = False,
        # env_minium_ratio: float = 1,
        env_random_sigma: Optional[float] = None,

        env_tolerance_offset: float = 0,
        env_center_adjust: float = 0,

        env_align_deg_check: float = 1.2,
        env_max_step: int = 20,

        corner_plane_height_offset_range: Optional[Tuple[float, float]] = None,
        **subenv_kwargs: Any,
    ) -> None:

        self.plane = Shape("Plane" + name_suffix)
        # 对于环境问题的补救
        self.plane.set_dynamic(False)
        self.plane.set_respondable(False)

        super().__init__(
            name_suffix = name_suffix, 
            obs_trans = obs_trans, obs_source = obs_source, env_object = self.plane, 
            env_action_noise = env_action_noise, env_init_box_pos_range = env_init_box_pos_range, env_init_vis_pos_range = env_init_vis_pos_range, env_vis_persp_deg_disturb = env_vis_persp_deg_disturb,
            env_movbox_size_range = env_movbox_size_range,
            env_movebox_center_err = env_movebox_center_err,
            # env_is_complexity_progression = env_is_complexity_progression, 
            # env_minium_ratio = env_minium_ratio,
            env_random_sigma = env_random_sigma,
            env_tolerance_offset = env_tolerance_offset, env_center_adjust = env_center_adjust,
            env_align_deg_check = env_align_deg_check, env_max_step = env_max_step,
            **subenv_kwargs
        )

        self.corner_plane_height_range = corner_plane_height_offset_range

        self.origin_pose_plane_relenv = self.plane.get_pose(self.anchor_env_object)
        self.plane_origin_height = self.plane.get_bounding_box()[5] - self.plane.get_bounding_box()[4]
        self._last_scale_plane = None

    def _to_origin_state(self):
        super()._to_origin_state()
        if self._last_scale_plane is not None:
            self.plane.scale_object(1, 1, 1 / self._last_scale_plane)
            self._last_scale_plane = None
        self.plane.set_pose(self.origin_pose_plane_relenv, self.anchor_env_object)

    def _set_plane_height_offset(self, offset: float):

        self._last_scale_plane = offset / self.plane_origin_height + 1
        self.plane.scale_object(1, 1, self._last_scale_plane)

        # 保持地盘与地面接触的高度偏移
        set_pose6_by_self(self.plane, np.array([0, 0, offset / 2 * 1e3]))
        
        # set_pose6_by_self(self.anchor_core_object, np.array([0, 0, offset]) * 1e3)
        set_pose6_by_self(self.anchor_core_object, np.array([0, 0, offset]) * 1e3)
        # 将间隙控制在对齐边缘两侧而不是单侧
        set_pose6_by_self(self.anchor_test_object, np.array([self.env_tolerance_offset / 2, self.env_tolerance_offset / 2, 0]) * 1e3)
        set_pose6_by_self(self.anchor_center_align_movebox_pose, np.array([self.env_tolerance_offset / 2, self.env_tolerance_offset / 2, 0]) * 1e3)
        # set_pose6_by_self(self.anchor_core_object, np.array([0, 0, offset]) * 1e3)

    def _set_init_plane_height_scale(self):
        if self.corner_plane_height_range is not None:
            self._set_plane_height_offset(sample_float(self.corner_plane_height_range[0], self.corner_plane_height_range[1]))

    def _set_max_init(self, direct: Literal[0, 1] = 0):
        '''
        测试, 用于使用最大扰动初始化环境
        '''
        super()._set_max_init(direct)
        if self.corner_plane_height_range is not None:
            self._set_plane_height_offset(self.corner_plane_height_range[direct])
        
    def reset(self):
        '''
        重新初始化
        '''
        super().reset()
        self._set_init_plane_height_scale()

class CornerEnv(PlaneBoxEnv):

    def __init__(
        self,
        
        env_pr: PyRep,

        subenv_mid_name: str,
        subenv_range: Tuple[int, int], # 作为 range 参数的 start 与 stop

        obs_trans: Optional[TransformType],
        obs_source: Literal["color", "depth"],

        env_reward_fn: RewardFnType,

        env_tolerance_offset: float = 0,
        env_center_adjust: float = 0,

        env_align_deg_check: float = 1.2,
        env_max_step: int = 20,
        # 直接加在原始动作上, 不转换
        env_action_noise: Optional[np.ndarray] = None,
        env_init_box_pos_range: Optional[Tuple[Union[Sequence[float], np.ndarray], Union[Sequence[float], np.ndarray]]] = None,
        env_init_vis_pos_range: Optional[Tuple[Union[Sequence[float], np.ndarray], Union[Sequence[float], np.ndarray]]] = None,
        
        env_vis_persp_deg_disturb: Optional[float] = None,
        env_movbox_size_range: Optional[Tuple[Union[Sequence[float], np.ndarray], Union[Sequence[float], np.ndarray]]] = None,
 
        env_movebox_center_err: Optional[Tuple[Union[Sequence[float], np.ndarray], Union[Sequence[float], np.ndarray]]] = None,

        # env_is_complexity_progression: bool = False,
        # env_minium_ratio: float = 1,
        # progress_end_timestep_ratio: Optional[float] = None,
        # train_total_timestep: Optional[int] = None,
        env_random_sigma: Optional[float] = None,

        # 单位 mm, deg
        act_unit: Sequence[float] = (5, 5, 5, 1, 1, 1),
        # act_is_passive_align: bool = True,
        # # 使用标准参数化动作空间
        # act_is_standard_parameter_space: bool = False,
        act_type: str = "PASSIVE_ALWAYS",
        # 时间长度是否为无限长
        env_is_unlimit_time: bool = True,
        env_is_terminate_when_insert: bool = False,

        corner_plane_height_offset_range: Optional[Tuple[float, float]] = None,
    ) -> None:

        super().__init__(
            env_pr = env_pr, 
            subenv_mid_name = subenv_mid_name, subenv_range = subenv_range, subenv_make_fn = CornerSubEnv, 
            obs_trans = obs_trans, obs_source = obs_source, 
            env_reward_fn = env_reward_fn, env_tolerance_offset = env_tolerance_offset, env_center_adjust = env_center_adjust,
            env_align_deg_check = env_align_deg_check, env_max_step = env_max_step, 
            env_action_noise = env_action_noise, env_init_box_pos_range = env_init_box_pos_range, env_init_vis_pos_range = env_init_vis_pos_range, 
            env_vis_persp_deg_disturb = env_vis_persp_deg_disturb, env_movbox_size_range = env_movbox_size_range,
            env_movebox_center_err = env_movebox_center_err,
            # env_is_complexity_progression = env_is_complexity_progression, 
            # env_minium_ratio = env_minium_ratio,
            # progress_end_timestep_ratio = progress_end_timestep_ratio,
            env_random_sigma = env_random_sigma,
            act_unit = act_unit, # train_total_timestep = train_total_timestep,
            corner_plane_height_offset_range = corner_plane_height_offset_range,
            act_type = act_type,
            env_is_unlimit_time = env_is_unlimit_time, env_is_terminate_when_insert = env_is_terminate_when_insert
        )
