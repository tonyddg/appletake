from pyrep import PyRep
from pyrep.objects.shape import Shape

from typing import Any, Dict, Literal, Optional, Sequence, Tuple, Union
import numpy as np
from albumentations.core.composition import TransformType

from ..utility import sample_vec, set_pose6_by_self

from .plane_box import PlaneBoxEnv, RewardFnType, PlaneBoxSubenvBase, GroupEnvObject

# TODO: 已放置箱子的位置尺寸扰动
# TODO: 环境中其他物体扰动

class ParalleSubEnv(PlaneBoxSubenvBase):

    def __init__(
        self, 
        name_suffix: str,

        obs_trans: TransformType,
        obs_source: Literal["color", "depth"],

        env_action_noise: Optional[np.ndarray] = None,
        env_init_box_pos_range: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        env_init_vis_pos_range: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        env_vis_persp_deg_disturb: Optional[float] = None,
        env_movbox_height_offset_range: Optional[Tuple[float, float]] = None,
        env_movebox_center_err: Optional[Tuple[np.ndarray, np.ndarray]] = None,
 
        env_is_complexity_progression: bool = False,
        env_minium_ratio: float = 1,

        env_tolerance_offset: float = 0,
        env_test_in: float = 0.05,
        env_max_step: int = 20,

        # 对齐纸箱的尺寸偏移范围 (原始尺寸 0.15 x 0.15 x 0.15)
        paralle_fixbox_size_offset_range: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        **subenv_kwargs: Any,
    ) -> None:
        
        self.plane = Shape("Plane" + name_suffix)
        # 对于环境问题的补救
        self.plane.set_dynamic(False)
        self.plane.set_respondable(False)

        self.fixbox = Shape("FixBox" + name_suffix)

        super().__init__(
            name_suffix = name_suffix, 
            obs_trans = obs_trans, obs_source = obs_source, env_object = GroupEnvObject([self.plane, self.fixbox]), 
            env_action_noise = env_action_noise, env_init_box_pos_range = env_init_box_pos_range, env_init_vis_pos_range = env_init_vis_pos_range, env_vis_persp_deg_disturb = env_vis_persp_deg_disturb,
            env_movbox_height_offset_range = env_movbox_height_offset_range,
            env_movebox_center_err = env_movebox_center_err,
            env_is_complexity_progression = env_is_complexity_progression, 
            env_minium_ratio = env_minium_ratio,
            env_tolerance_offset = env_tolerance_offset, env_test_in = env_test_in, env_max_step = env_max_step,
            **subenv_kwargs
        )

        self.paralle_fixbox_size_offset_range = paralle_fixbox_size_offset_range
        self.origin_pose_fixbox_relenv = self.fixbox.get_pose(self.anchor_env_object)

        fix_box_bounding_box = self.fixbox.get_bounding_box()
        self.fixbox_origin_size = np.array(
            [fix_box_bounding_box[2 * i + 1] - fix_box_bounding_box[2 * i] for i in range(3)]
        )
        self._last_scale_fixbox = None

    def _to_origin_state(self):
        super()._to_origin_state()
        if self._last_scale_fixbox is not None:
            self.fixbox.scale_object(1 / self._last_scale_fixbox[0], 1 / self._last_scale_fixbox[1], 1 / self._last_scale_fixbox[2])
            self._last_scale_fixbox = None
        self.fixbox.set_pose(self.origin_pose_fixbox_relenv, self.anchor_env_object)

    def _set_fixbox_size_offset(self, size_offset: np.ndarray):
        self._last_scale_fixbox = size_offset / self.fixbox_origin_size + 1
        self.fixbox.scale_object(self._last_scale_fixbox[0], self._last_scale_fixbox[1], self._last_scale_fixbox[2])

        # 保持地盘与地面接触的高度偏移
        set_pose6_by_self(self.fixbox, np.array([-size_offset[0] / 2, -size_offset[1] / 2, size_offset[2] / 2]) * 1e3)
        set_pose6_by_self(self.anchor_core_object, np.array([0, -size_offset[1], size_offset[2]]) * 1e3)

    def _set_init_fixbox_scale(self):
        if self.paralle_fixbox_size_offset_range is not None:
            self._set_fixbox_size_offset(sample_vec(self.paralle_fixbox_size_offset_range[0], self.paralle_fixbox_size_offset_range[1]))
    
    def _set_max_init(self, direct: Literal[0, 1] = 0):
        '''
        测试, 用于使用最大扰动初始化环境
        '''
        super()._set_max_init(direct)
        if self.paralle_fixbox_size_offset_range is not None:
            self._set_fixbox_size_offset(self.paralle_fixbox_size_offset_range[direct])

    def reset(self):
        '''
        重新初始化
        '''
        super().reset()
        self._set_init_fixbox_scale()

class ParalleEnv(PlaneBoxEnv):

    def __init__(
        self,
        
        env_pr: PyRep,

        subenv_mid_name: str,
        subenv_range: Tuple[int, int], # 作为 range 参数的 start 与 stop

        obs_trans: TransformType,
        obs_source: Literal["color", "depth"],

        env_reward_fn: RewardFnType,

        env_tolerance_offset: float = 0,
        env_test_in: float = 0.05,
        env_max_step: int = 20,
        # 直接加在原始动作上, 不转换
        env_action_noise: Optional[np.ndarray] = None,
        env_init_box_pos_range: Optional[Tuple[Union[Sequence[float], np.ndarray], Union[Sequence[float], np.ndarray]]] = None,
        env_init_vis_pos_range: Optional[Tuple[Union[Sequence[float], np.ndarray], Union[Sequence[float], np.ndarray]]] = None,
        
        env_vis_persp_deg_disturb: Optional[float] = None,
        env_movbox_height_offset_range: Optional[Tuple[float, float]] = None,
 
        env_movebox_center_err: Optional[Tuple[Union[Sequence[float], np.ndarray], Union[Sequence[float], np.ndarray]]] = None,

        env_is_complexity_progression: bool = False,
        env_minium_ratio: float = 1,

        # 单位 mm, deg
        act_unit: Sequence[float] = (5, 5, 5, 1, 1, 1),

        train_total_timestep: Optional[int] = None,

        paralle_fixbox_size_offset_range: Optional[Tuple[Union[Sequence[float], np.ndarray], Union[Sequence[float], np.ndarray]]] = None,
    ) -> None:
        
        if paralle_fixbox_size_offset_range is not None:
            _paralle_fixbox_size_offset_range = (
                np.asarray(paralle_fixbox_size_offset_range[0], np.float32),
                np.asarray(paralle_fixbox_size_offset_range[1], np.float32),
            )
        else:
            _paralle_fixbox_size_offset_range = None

        super().__init__(
            env_pr = env_pr, 
            subenv_mid_name = subenv_mid_name, subenv_range = subenv_range, subenv_make_fn = ParalleSubEnv, 
            obs_trans = obs_trans, obs_source = obs_source, 
            env_reward_fn = env_reward_fn, env_tolerance_offset = env_tolerance_offset, env_test_in = env_test_in, env_max_step = env_max_step, 
            env_action_noise = env_action_noise, env_init_box_pos_range = env_init_box_pos_range, env_init_vis_pos_range = env_init_vis_pos_range, 
            env_vis_persp_deg_disturb = env_vis_persp_deg_disturb, env_movbox_height_offset_range = env_movbox_height_offset_range,
            env_movebox_center_err = env_movebox_center_err,
            env_is_complexity_progression = env_is_complexity_progression, 
            env_minium_ratio = env_minium_ratio,
            act_unit = act_unit, train_total_timestep = train_total_timestep,
            paralle_fixbox_size_offset_range = _paralle_fixbox_size_offset_range
        )

