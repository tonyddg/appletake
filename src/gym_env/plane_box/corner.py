from pyrep import PyRep
from pyrep.objects.shape import Shape

from typing import Any, Dict, Literal, Optional, Sequence, Tuple, Union
import numpy as np
from albumentations.core.composition import TransformType

from ..utility import sample_float

from .plane_box import PlaneBoxEnv, RewardFnType, PlaneBoxSubenvBase

# TODO: 相机观测角扰动
# TODO: 已放置箱子的位置尺寸扰动
# TODO: 环境中其他物体扰动

class CornerSubEnv(PlaneBoxSubenvBase):

    def __init__(
        self, 
        name_suffix: str,

        obs_trans: TransformType,
        obs_source: Literal["color", "depth"],

        env_object_kwargs: Optional[Dict[str, Any]] = None,

        env_action_noise: Optional[np.ndarray] = None,
        env_init_box_pos_range: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        env_init_vis_pos_range: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        env_vis_persp_deg_disturb: Optional[float] = None,
        # 原始高度 0.05, 使用原始单位 m
        # corner_plane_height_offset_range: Optional[Tuple[float, float]] = None,
 
        env_tolerance_offset: float = 0,
        env_test_in: float = 0.05,
        env_max_step: int = 20
    ) -> None:
        self.plane = Shape("Plane" + name_suffix)

        super().__init__(
            name_suffix, obs_trans, obs_source, self.plane, 
            env_action_noise, env_init_box_pos_range, env_init_vis_pos_range, env_vis_persp_deg_disturb,
            env_tolerance_offset, env_test_in, env_max_step
        )

        self.corner_plane_height_range: Optional[Tuple[float, float]] = env_object_kwargs["corner_plane_height_offset_range"] # type: ignore
        assert isinstance(self.corner_plane_height_range, Tuple)
        
        self.origin_pose_plane_relenv = self.plane.get_pose(self.anchor_env_object)
        self.plane_origin_height = self.plane.get_bounding_box()[5] - self.plane.get_bounding_box()[4]
        self._last_scale = None

    def _set_plane_height_offset(self, offset: float):
        if self._last_scale is not None:
            self.plane.scale_object(1, 1, 1 / self._last_scale)
        self.plane.set_pose(self.origin_pose_plane_relenv, self.anchor_env_object)
        
        self._last_scale = offset / self.plane_origin_height + 1
        self.plane.scale_object(1, 1, self._last_scale)

        # 保持地盘与地面接触的高度偏移
        self.plane.set_position([0, 0, offset / 2], self.plane)
        self._set_core_object_anchor_pose(np.array([0, 0, offset * 1e3]))

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

        obs_trans: TransformType,
        obs_source: Literal["color", "depth"],

        env_reward_fn: RewardFnType,

        env_tolerance_offset: float = 0,
        env_test_in: float = 0.08,
        env_max_step: int = 20,
        # 直接加在原始动作上, 不转换
        env_action_noise: Optional[np.ndarray] = None,
        env_init_box_pos_range: Optional[Tuple[Union[Sequence[float], np.ndarray], Union[Sequence[float], np.ndarray]]] = None,
        env_init_vis_pos_range: Optional[Tuple[Union[Sequence[float], np.ndarray], Union[Sequence[float], np.ndarray]]] = None,
        # 单位 mm, deg
        act_unit: Sequence[float] = (5, 5, 5, 1, 1, 1),

        corner_plane_height_offset_range: Optional[Tuple[float, float]] = None,
    ) -> None:
        super().__init__(
            env_pr, subenv_mid_name, subenv_range, CornerSubEnv, obs_trans, obs_source, 
            env_reward_fn, env_tolerance_offset, env_test_in, env_max_step, 
            env_action_noise, env_init_box_pos_range, env_init_vis_pos_range, act_unit,
            corner_plane_height_offset_range = corner_plane_height_offset_range
        )
