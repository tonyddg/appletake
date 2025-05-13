import warnings
from pyrep import PyRep
from pyrep.objects.shape import Shape, PrimitiveShape
from pyrep.objects.dummy import Dummy
from pyrep.objects.object import Object

from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Union
import numpy as np
from albumentations.core.composition import TransformType

from ..utility import sample_float, set_pose6_by_self
from ...pr.shape_size_setter import create_fixbox
from .plane_box import PlaneBoxEnv, RewardFnType, PlaneBoxSubenvBase, EnvObjectsBase

# TODO: 地面不平整扰动
# 1. 完全平整
# 2. 三块/两块/一块

class PlaneAndThree(EnvObjectsBase):
    def __init__(self, plane: Shape, three_group: Sequence[Optional[Shape]]) -> None:
        self.plane = plane
        self.three_group = three_group

    def check_collision(self, obj: Object) -> bool:
        for env_obj in self.three_group:
            if env_obj is not None:
                if env_obj.check_collision(obj):
                    return True
        return self.plane.check_collision(obj)

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
        # 是否将运动坐标系设置在拐角
        env_is_corner_move: bool = False,

        debug_center_check: bool = False,

        # corner_plane_height_offset_range: Optional[Tuple[float, float]] = None,

        # 三个已放置纸箱的尺寸范围 (单位 m)
        corner_fixbox_size_range: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        # 纸箱间间隙尺寸范围
        corner_box_gap_range: Optional[Tuple[float, float]] = None,
        # 出现额外纸箱的概率
        # corner_extra_fixbox_prob: float = 0.5,

        **subenv_kwargs: Any,
    ) -> None:

        self.plane = Shape("Plane" + name_suffix)
        # 对于环境问题的补救
        self.plane.set_dynamic(False)
        self.plane.set_respondable(False)

        self.corner = Dummy("CornerPosition" + name_suffix)
        self.fixbox_list: List[Optional[Shape]] = [
            Shape.create(PrimitiveShape.CUBOID, [0.001, 0.001, 0.001])
        for i in range(3)]
        # self.fixbox_list.append(None)
        # self.fixbox_list.append(None)

        super().__init__(
            name_suffix = name_suffix, 
            obs_trans = obs_trans, obs_source = obs_source, env_object = PlaneAndThree(self.plane, self.fixbox_list), 
            env_action_noise = env_action_noise, env_init_box_pos_range = env_init_box_pos_range, env_init_vis_pos_range = env_init_vis_pos_range, env_vis_persp_deg_disturb = env_vis_persp_deg_disturb,
            env_movbox_size_range = env_movbox_size_range,
            env_movebox_center_err = env_movebox_center_err,
            # env_is_complexity_progression = env_is_complexity_progression, 
            # env_minium_ratio = env_minium_ratio,
            env_random_sigma = env_random_sigma,
            env_is_corner_move = env_is_corner_move,
            env_tolerance_offset = env_tolerance_offset, env_center_adjust = env_center_adjust,
            env_align_deg_check = env_align_deg_check, env_max_step = env_max_step,
            debug_center_check = debug_center_check,
            **subenv_kwargs
        )

        # self.corner_plane_height_range = corner_plane_height_offset_range

        # self.origin_pose_plane_relenv = self.plane.get_pose(self.anchor_env_object)
        # self.plane_origin_height = self.plane.get_bounding_box()[5] - self.plane.get_bounding_box()[4]
        # self._last_scale_plane = None

        # print(corner_fixbox_size_range)
        self.corner_fixbox_size_range = corner_fixbox_size_range
        # 纸箱间间隙尺寸范围
        self.corner_box_gap_range = corner_box_gap_range
        # 出现额外纸箱的概率
        # self.corner_extra_fixbox_prob = corner_extra_fixbox_prob,

    def _to_origin_state(self):
        super()._to_origin_state()
        # if self._last_scale_plane is not None:
        #     self.plane.scale_object(1, 1, 1 / self._last_scale_plane)
        #     self._last_scale_plane = None
        # self.plane.set_pose(self.origin_pose_plane_relenv, self.anchor_env_object)
        for i in range(len(self.fixbox_list)):
            fixbox = self.fixbox_list[i]
            if fixbox is not None:
                if fixbox.exists(fixbox.get_name()):
                    fixbox.remove()
                self.fixbox_list[i] = None

    def _set_fixbox_scale(self, fixbox_size: np.ndarray, box_gap: np.ndarray):
        '''
        * `fixbox_size` 已做水平与高度的对齐处理
        * `box_gap` 0, 1 普通间隙; 2, 3 垂直间隙 (可取正负)
        '''

        moveobx_size = self.movebox_size_setter.get_cur_bbox()

        # self._last_scale_plane = offset / self.plane_origin_height + 1
        # self.plane.scale_object(1, 1, self._last_scale_plane)

        # # 保持地盘与地面接触的高度偏移
        # set_pose6_by_self(self.plane, np.array([0, 0, offset / 2 * 1e3]))
        
        # # set_pose6_by_self(self.anchor_core_object, np.array([0, 0, offset]) * 1e3)
        # set_pose6_by_self(self.anchor_core_object, np.array([0, 0, offset]) * 1e3)
        # # 将间隙控制在对齐边缘两侧而不是单侧
        # set_pose6_by_self(self.anchor_test_object, np.array([self.env_tolerance_offset / 2, self.env_tolerance_offset / 2, 0]) * 1e3)
        # set_pose6_by_self(self.anchor_center_align_movebox_pose, np.array([self.env_tolerance_offset / 2, self.env_tolerance_offset / 2, 0]) * 1e3)
        # # set_pose6_by_self(self.anchor_core_object, np.array([0, 0, offset]) * 1e3)

        # 0 仅垛盘, 1 仅对角, 2 对角与上侧 -y, 3 对角与右侧 -x, 4 三箱
        box_case = np.random.choice(5)

        if box_case == 0:
            height_offset = 0
            # set_pose6_by_self(self.anchor_test_object, np.array([self.env_tolerance_offset / 2, self.env_tolerance_offset / 2, 0]) * 1e3)
            # set_pose6_by_self(self.anchor_center_align_movebox_pose, np.array([self.env_tolerance_offset / 2, self.env_tolerance_offset / 2, 0]) * 1e3)
        else:
            self.fixbox_list[0] = create_fixbox(fixbox_size[0], self.corner)
            height_offset = fixbox_size[0, 2]

            if box_case == 2 or box_case == 4:
                # if fixbox_size[0, 1] + box_gap[0] < moveobx_size[1]:
                #     fixbox_size[1, 2] = fixbox_size[0, 2]
                fixbox_size[1, 2] = fixbox_size[0, 2]

                self.fixbox_list[1] = create_fixbox(fixbox_size[1], self.corner)
                self.fixbox_list[1].set_position([box_gap[2], -fixbox_size[0, 1] - box_gap[0], 0], self.fixbox_list[1])

            if box_case == 3 or box_case == 4:
                # if fixbox_size[0, 0] + box_gap[3] < moveobx_size[0]:
                #     fixbox_size[2, 2] = fixbox_size[0, 2]
                fixbox_size[2, 2] = fixbox_size[0, 2]

                self.fixbox_list[2] = create_fixbox(fixbox_size[2], self.corner)
                self.fixbox_list[2].set_position([-fixbox_size[0, 0] - box_gap[1], box_gap[3], 0], self.fixbox_list[2])
        
        set_pose6_by_self(self.anchor_core_object, np.array([0, 0, height_offset]) * 1e3)
        set_pose6_by_self(self.anchor_test_object, np.array([self.env_tolerance_offset / 2, self.env_tolerance_offset / 2, 0]) * 1e3)
        set_pose6_by_self(self.anchor_center_align_movebox_pose, np.array([self.env_tolerance_offset / 2, self.env_tolerance_offset / 2, 0]) * 1e3)

    def _set_init_plane_height_scale(self):
        # if self.corner_plane_height_range is not None:
        #     self._set_fixbox_scale(sample_float(self.corner_plane_height_range[0], self.corner_plane_height_range[1]))
        if self.corner_fixbox_size_range is None:
            warnings.warn(f"没有提供平面对齐环境的关键参数 {self.corner_fixbox_size_range}")
            return

        # 基础尺寸
        fixbox_size = np.array([[
            self.sample_float(self.corner_fixbox_size_range[0][i], self.corner_fixbox_size_range[1][i])
            for i in range(3)
        ] for _ in range(5)], np.float32)

        if self.corner_box_gap_range is not None:
            fixbox_gap_size = np.array([
                self.sample_float(self.corner_box_gap_range[0], self.corner_box_gap_range[1])
            for _ in range(4)], np.float32)
            
            for i in range(2, 4):
                if np.random.choice(2) == 0:
                    fixbox_gap_size[i] *= -1
        else:
            fixbox_gap_size = np.zeros(4)

        # # 间隙尺寸
        # if self.three_is_zero_middle:
        #     random_pm = np.random.choice([0, -1, 1], 4)
        # else:
        #     random_pm = np.random.choice([-1, 1], 4)

        fixbox_size[1, 0] = fixbox_size[0, 0]
        fixbox_size[2, 1] = fixbox_size[0, 1]

        # fixbox_size[1, 2] = fixbox_size[0, 2]
        # fixbox_size[2, 2] = fixbox_size[0, 2]

        self._set_fixbox_scale(fixbox_size, fixbox_gap_size)

    def _set_max_init(self, direct: Literal[0, 1] = 0):
        '''
        测试, 用于使用最大扰动初始化环境
        '''
        super()._set_max_init(direct)
        if self.corner_fixbox_size_range is None:
            warnings.warn(f"没有提供平面对齐环境的关键参数 {self.corner_fixbox_size_range}")
            return

        if self.corner_box_gap_range is not None:
            fixbox_gap_size = np.ones(4, np.float32) * self.corner_box_gap_range[direct]
            
            for i in range(2, 4):
                if np.random.choice(2) == 0:
                    fixbox_gap_size[i] *= -1
        else:
            fixbox_gap_size = np.zeros(4)

        # 基础尺寸
        fixbox_size = np.array([
            self.corner_fixbox_size_range[int(np.random.random() > 0.5)],
            self.corner_fixbox_size_range[int(np.random.random() > 0.5)],
            self.corner_fixbox_size_range[int(np.random.random() > 0.5)],
        ])
        fixbox_size[1, 0] = fixbox_size[0, 0]
        fixbox_size[2, 1] = fixbox_size[0, 1]
        # fixbox_size[1, 2] = fixbox_size[0, 2]
        # fixbox_size[2, 2] = fixbox_size[0, 2]

        self._set_fixbox_scale(fixbox_size, fixbox_gap_size)

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
        # 是否将运动坐标系设置在拐角
        env_is_corner_move: bool = False,

        # 单位 mm, deg
        act_unit: Sequence[float] = (5, 5, 5, 1, 1, 1),
        # act_is_passive_align: bool = True,
        # # 使用标准参数化动作空间
        # act_is_standard_parameter_space: bool = False,
        act_type: str = "PASSIVE_ALWAYS",
        # 时间长度是否为无限长
        env_is_unlimit_time: bool = True,
        env_is_terminate_when_insert: bool = False,
        dataset_is_center_goal: bool = False,
        debug_center_check: bool = False,
        debug_close_pr: bool = False,

        # corner_plane_height_offset_range: Optional[Tuple[float, float]] = None,
        corner_fixbox_size_range: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        # 纸箱间间隙尺寸范围
        corner_box_gap_range: Optional[Tuple[float, float]] = None,
    ) -> None:
        
        if corner_fixbox_size_range is not None:
            _corner_fixbox_size_range = (
                np.asarray(corner_fixbox_size_range[0], np.float32),
                np.asarray(corner_fixbox_size_range[1], np.float32),
            )
        else:
            _corner_fixbox_size_range = None
        # print(_corner_fixbox_size_range)

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
            env_is_corner_move = env_is_corner_move,
            act_unit = act_unit, # train_total_timestep = train_total_timestep,
            dataset_is_center_goal = dataset_is_center_goal,
            debug_center_check = debug_center_check,
            debug_close_pr = debug_close_pr,

            corner_fixbox_size_range = _corner_fixbox_size_range,
            corner_box_gap_range = corner_box_gap_range,
            act_type = act_type,
            env_is_unlimit_time = env_is_unlimit_time, env_is_terminate_when_insert = env_is_terminate_when_insert
        )
