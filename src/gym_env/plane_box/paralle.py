import warnings
from pyrep import PyRep
from pyrep.objects.shape import Shape, PrimitiveShape
from pyrep.objects.object import Object
from pyrep.objects.dummy import Dummy

from typing import Any, Dict, Literal, Optional, Sequence, Tuple, Union
import numpy as np
from albumentations.core.composition import TransformType

from ..utility import sample_float, sample_vec, set_pose6_by_self
from ...pr.shape_size_setter import ShapeSizeSetter, create_fixbox
from .plane_box import EnvObjectsBase, PlaneBoxEnv, RewardFnType, PlaneBoxSubenvBase, GroupEnvObject

# TODO: 已放置箱子的位置尺寸扰动
# TODO: 环境中其他物体扰动

class PlaneAndFixbox(EnvObjectsBase):
    def __init__(self, plane: Shape, fixboxA: Object, fixboxB: Optional[Object]) -> None:
        self.plane = plane
        self.fixboxA = fixboxA
        self.fixboxB = fixboxB

    def check_collision(self, obj: Object) -> bool:
        if self.fixboxA.check_collision(obj):
            return True
        if self.plane.check_collision(obj):
            return True
        if self.fixboxB is not None:
            return self.fixboxB.check_collision(obj)
        else:
            return False

class ParalleSubEnv(PlaneBoxSubenvBase):

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

        # 对齐纸箱的尺寸偏移范围 (原始尺寸 0.15 x 0.15 x 0.15)
        paralle_fixbox_size_range: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        # 额外纸箱对齐变动比率
        paralle_size_disturb_range: Optional[Tuple[float, float]] = None,
        # 出现额外纸箱的概率
        paralle_extra_fixbox_prob: float = 0.5,
        # 是否出现长度变动为 0 的中间态
        paralle_is_zero_middle: bool = False,
        
        # 纸箱间间隙尺寸范围
        paralle_box_gap_range: Optional[Tuple[float, float]] = None,

        **subenv_kwargs: Any,
    ) -> None:
        
        self.plane = Shape("Plane" + name_suffix)
        # 对于环境问题的补救
        self.plane.set_dynamic(False)
        self.plane.set_respondable(False)

        self.fixbox_env_object = PlaneAndFixbox(self.plane, Shape.create(PrimitiveShape.CUBOID, [0.001, 0.001, 0.001]), None)
        self.corner = Dummy("CornerPosition" + name_suffix)

        super().__init__(
            name_suffix = name_suffix, 
            obs_trans = obs_trans, obs_source = obs_source, env_object = self.fixbox_env_object, 
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

        self.paralle_fixbox_size_range = paralle_fixbox_size_range
        self.paralle_size_disturb_range = paralle_size_disturb_range
        self.paralle_extra_fixbox_prob = paralle_extra_fixbox_prob
        self.paralle_box_gap_range = paralle_box_gap_range
        self.paralle_is_zero_middle = paralle_is_zero_middle

    def _to_origin_state(self):
        super()._to_origin_state()
        if self.fixbox_env_object.fixboxA.exists(self.fixbox_env_object.fixboxA.get_name()):
            self.fixbox_env_object.fixboxA.remove()
        if self.fixbox_env_object.fixboxB is not None:
            if self.fixbox_env_object.fixboxB.exists(self.fixbox_env_object.fixboxB.get_name()):
                self.fixbox_env_object.fixboxB.remove()
            self.fixbox_env_object.fixboxB = None

    def _set_fixbox_scale(self, fixbox_size: np.ndarray, disturb_size: np.ndarray, box_gap: float):
        # A 对角, B 额外

        movebox_size = self.movebox_size_setter.get_cur_bbox()

        self.fixbox_env_object.fixboxA = create_fixbox(fixbox_size[0], self.corner)

        # 向上平行对齐
        is_upper = bool(np.random.random() > 0.5)
        # 使用 B
        is_useB = bool(np.random.random() < self.paralle_extra_fixbox_prob)
        align_x = 0
        align_y = 0
        align_z = 0

        test_offset_x = 0 # self.env_center_adjust
        test_offset_y = 0 # self.env_center_adjust

        if not is_useB:
            align_z = fixbox_size[0, 2]
            # 存在两种情况, 平行纸箱在视野左侧与右侧
            if is_upper:
                # 将间隙控制在对齐边缘两侧而不是单侧
                align_x = fixbox_size[0, 0]

                test_offset_y = self.env_tolerance_offset / 2
                align_y = 0
            else:
                align_y = fixbox_size[0, 1]

                test_offset_x = self.env_tolerance_offset / 2
                align_x = 0
        else:
            # align_z = max(fixbox_size[0, 2], fixbox_size[1, 2])
            # 计算对齐偏移
            if is_upper:
                # 限制对齐长度范围
                fixbox_size[1, 0] = fixbox_size[0, 0] + disturb_size
                # fixbox A 短于 movebox 时 fixbox B 突出将改变对齐
                if movebox_size[1] > (fixbox_size[0, 1] + box_gap):
                    align_x = max(fixbox_size[0, 0], fixbox_size[1, 0])
                    align_z = max(fixbox_size[0, 2], fixbox_size[1, 2])
                else:
                    align_x = fixbox_size[0, 0]
                    align_z = fixbox_size[0, 2]
                test_offset_y = self.env_tolerance_offset / 2
                align_y = 0
            else:
                fixbox_size[1, 1] = fixbox_size[0, 1] + disturb_size
                if movebox_size[0] > (fixbox_size[0, 0] + box_gap):
                    align_y = max(fixbox_size[0, 1], fixbox_size[1, 1])
                    align_z = max(fixbox_size[0, 2], fixbox_size[1, 2])
                else:
                    align_y = fixbox_size[0, 1]
                    align_z = fixbox_size[0, 2]
                test_offset_x = self.env_tolerance_offset / 2
                align_x = 0

            self.fixbox_env_object.fixboxB = create_fixbox(fixbox_size[1], self.corner)

            # 移动 B 到对齐位置
            if is_upper:
                self.fixbox_env_object.fixboxB.set_position([0, -fixbox_size[0, 1] - box_gap, 0], self.fixbox_env_object.fixboxB)
            else:
                self.fixbox_env_object.fixboxB.set_position([-fixbox_size[0, 0] - box_gap, 0, 0], self.fixbox_env_object.fixboxB)

        set_pose6_by_self(self.anchor_core_object, np.array([-align_x, -align_y, align_z]) * 1e3)
        # 将间隙控制在对齐边缘两侧而不是单侧
        set_pose6_by_self(self.anchor_test_object, np.array([test_offset_x, test_offset_y, 0]) * 1e3)
        set_pose6_by_self(self.anchor_center_align_movebox_pose, np.array([test_offset_x, test_offset_y, 0]) * 1e3)

        if is_upper:
            set_pose6_by_self(self.anchor_center_align_movebox_pose, np.array([self.env_center_adjust, 0, 0]) * 1e3)
        else:
            set_pose6_by_self(self.anchor_center_align_movebox_pose, np.array([0, self.env_center_adjust, 0]) * 1e3)

    def _set_init_fixbox_scale(self):
        if self.paralle_fixbox_size_range is None or self.paralle_size_disturb_range is None:
            warnings.warn(f"没有提供单边环境的关键参数 {self.paralle_fixbox_size_range} 或 {self.paralle_size_disturb_range}")
            return
        
        fixbox_size = np.array([[
            self.sample_float(self.paralle_fixbox_size_range[0][i], self.paralle_fixbox_size_range[1][i])
            for i in range(3)
        ] for _ in range(2)], np.float32)

        if self.paralle_is_zero_middle:
            disturb_size = np.random.choice([0, -1, 1], 1) * self.sample_float(self.paralle_size_disturb_range[0], self.paralle_size_disturb_range[1])
        else:
            disturb_size = np.random.choice([-1, 1], 1) * self.sample_float(self.paralle_size_disturb_range[0], self.paralle_size_disturb_range[1])
        
        if self.paralle_box_gap_range is not None:
            box_gap = self.sample_float(self.paralle_box_gap_range[0], self.paralle_box_gap_range[1])
        else:
            box_gap = 0

        self._set_fixbox_scale(fixbox_size, disturb_size, box_gap)

    def _set_max_init(self, direct: Literal[0, 1] = 0):
        '''
        测试, 用于使用最大扰动初始化环境
        '''
        super()._set_max_init(direct)
        if self.paralle_fixbox_size_range is not None and self.paralle_size_disturb_range is not None:

            disturb_size = np.random.choice([0, -1, 1], 1) * self.sample_float(self.paralle_size_disturb_range[0], self.paralle_size_disturb_range[1])

            if self.paralle_box_gap_range is not None:
                box_gap = self.paralle_box_gap_range[direct]
            else:
                box_gap = 0

            self._set_fixbox_scale(
                np.array([
                    self.paralle_fixbox_size_range[int(np.random.random() > 0.5)],
                    self.paralle_fixbox_size_range[int(np.random.random() > 0.5)],
                ]), disturb_size, box_gap
            )

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

        paralle_fixbox_size_range: Optional[Tuple[Union[Sequence[float], np.ndarray], Union[Sequence[float], np.ndarray]]] = None,
        # 额外纸箱对齐变动比率
        paralle_size_disturb_range: Optional[Tuple[float, float]] = None,
        # 出现额外纸箱的概率
        paralle_extra_fixbox_prob: float = 0.5,
        # 是否出现长度变动为 0 的中间态
        paralle_is_zero_middle: bool = False,
        
        # 纸箱间间隙尺寸范围
        paralle_box_gap_range: Optional[Tuple[float, float]] = None,


    ) -> None:
        
        if paralle_fixbox_size_range is not None:
            _paralle_fixbox_size_range = (
                np.asarray(paralle_fixbox_size_range[0], np.float32),
                np.asarray(paralle_fixbox_size_range[1], np.float32),
            )
        else:
            _paralle_fixbox_size_range = None

        super().__init__(
            env_pr = env_pr, 
            subenv_mid_name = subenv_mid_name, subenv_range = subenv_range, subenv_make_fn = ParalleSubEnv, 
            obs_trans = obs_trans, obs_source = obs_source, 
            env_reward_fn = env_reward_fn, env_tolerance_offset = env_tolerance_offset, env_center_adjust = env_center_adjust,
            env_align_deg_check = env_align_deg_check, env_max_step = env_max_step, 
            env_action_noise = env_action_noise, env_init_box_pos_range = env_init_box_pos_range, env_init_vis_pos_range = env_init_vis_pos_range, 
            env_vis_persp_deg_disturb = env_vis_persp_deg_disturb, env_movbox_size_range = env_movbox_size_range,
            env_movebox_center_err = env_movebox_center_err,
            # env_is_complexity_progression = env_is_complexity_progression, 
            # env_minium_ratio = env_minium_ratio,
            # progress_end_timestep_ratio = progress_end_timestep_ratio,
            # train_total_timestep = train_total_timestep,
            env_random_sigma = env_random_sigma,
            act_unit = act_unit, act_type = act_type, 
            paralle_fixbox_size_range = _paralle_fixbox_size_range,
            paralle_size_disturb_range = paralle_size_disturb_range,
            paralle_extra_fixbox_prob = paralle_extra_fixbox_prob,
            paralle_box_gap_range = paralle_box_gap_range,
            paralle_is_zero_middle = paralle_is_zero_middle,
            env_is_unlimit_time = env_is_unlimit_time, env_is_terminate_when_insert = env_is_terminate_when_insert
        )

