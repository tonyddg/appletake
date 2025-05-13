import warnings
from pyrep import PyRep
from pyrep.objects.shape import Shape, PrimitiveShape
from pyrep.objects.dummy import Dummy
from pyrep.objects.object import Object

from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Union
import numpy as np
from albumentations.core.composition import TransformType

from ...gym_env.utility import set_pose6_by_self
from ...pr.shape_size_setter import create_fixbox
from .plane_box import EnvObjectsBase, PlaneBoxEnv, RewardFnType, PlaneBoxSubenvBase
from ..aabb import try_max_in, Rect

# 寻找纸箱放置位置的算法
def fixbox_to_fix_rect_list(box_group_list: Sequence[Optional[Shape]], box_size_list: np.ndarray, corner: Dummy):
    '''
    注意 size 的单位应为 m
    '''
    fix_rect_list = []
    for box_handle, box_size in zip(box_group_list, box_size_list):
        if box_handle is None or (box_size == 0).all():
            continue
        cx, cy, _ = box_handle.get_position(corner)
        rect = Rect.from_center_wh(cx, cy, box_size[0], box_size[1])
        fix_rect_list.append(rect)
    return fix_rect_list

def movebox_to_try_rect(init_x: float, init_y: float, movebox_size: np.ndarray):
    return Rect.from_center_wh(
        init_x, init_y, movebox_size[0], movebox_size[1]
    )

def try_align_xy(box_group_list: Sequence[Optional[Shape]], box_size_list: np.ndarray, movebox_size: np.ndarray, corner: Dummy, init_x: float, init_y: float):
    fix_rect_list = fixbox_to_fix_rect_list(box_group_list, box_size_list, corner)
    try_rect = movebox_to_try_rect(init_x, init_y, movebox_size)
    return try_max_in(fix_rect_list, try_rect)

# TODO: 已放置箱子的位置尺寸扰动
# TODO: 环境中其他物体扰动

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

class ThreeSubEnv(PlaneBoxSubenvBase):

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

        # debug, 用于调整最佳插入位置为无间隙区域以用于测试卷积神经网络的性能 (需要根据不同的场景具体实现)
        debug_center_check: bool = False,
        # debug, 用于模拟重心偏移下的误差分布 (用于训练特征提取器)
        debug_is_fake_center_err: bool = False,

        # 三个已放置纸箱的尺寸范围 (单位 m)
        three_fixbox_size_range: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        # 纸箱对齐侧长度变动比率
        # three_size_disturb: float = 0.1,
        three_size_disturb_range: Optional[Tuple[float, float]] = None,
        # 出现额外纸箱的概率
        three_extra_fixbox_prob: float = 0.5,
        # 是否出现长度变动为 0 的中间态
        three_is_zero_middle: bool = False,
        
        # 纸箱间间隙尺寸范围
        three_box_gap_range: Optional[Tuple[float, float]] = None,

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
        self.fixbox_list.append(None)
        self.fixbox_list.append(None)

        super().__init__(
            name_suffix = name_suffix, 
            obs_trans = obs_trans, obs_source = obs_source, env_object = PlaneAndThree(self.plane, self.fixbox_list), 
            env_action_noise = env_action_noise, env_init_box_pos_range = env_init_box_pos_range, env_init_vis_pos_range = env_init_vis_pos_range, env_vis_persp_deg_disturb = env_vis_persp_deg_disturb,
            env_movbox_size_range = env_movbox_size_range,
            env_movebox_center_err = env_movebox_center_err,
            # env_is_complexity_progression = env_is_complexity_progression, 
            # env_minium_ratio = env_minium_ratio,
            env_random_sigma = env_random_sigma,
            env_tolerance_offset = env_tolerance_offset, env_center_adjust = env_center_adjust,
            env_align_deg_check = env_align_deg_check, env_max_step = env_max_step,
            env_is_corner_move = env_is_corner_move,
            debug_center_check = debug_center_check,
            debug_is_fake_center_err = debug_is_fake_center_err,
            **subenv_kwargs
        )

        self.three_fixbox_size_range = three_fixbox_size_range
        self.three_size_disturb_range = three_size_disturb_range
        self.three_extra_fixbox_prob = three_extra_fixbox_prob
        self.three_box_gap_range = three_box_gap_range
        self.three_is_zero_middle = three_is_zero_middle
    
    def _to_origin_state(self):
        super()._to_origin_state()
        for i in range(len(self.fixbox_list)):
            fixbox = self.fixbox_list[i]
            if fixbox is not None:
                if fixbox.exists(fixbox.get_name()):
                    fixbox.remove()
                self.fixbox_list[i] = None

    def _set_fixbox_scale(self, fixbox_size: np.ndarray, box_gap: np.ndarray):
        # A 对角, B 上侧 -y, C 右侧 -x, D 上侧额外, E 右侧额外
        # set_type = bool(np.random.random() > 0.5)
        # 基于对角纸箱在最小长度的 ±10% 范围内扰动, 防止不可能的极端情况

        movebox_size = self.movebox_size_setter.get_cur_bbox()

        # fixbox_size[1, 0] = fixbox_size[0, 0] * (1 + sample_float(-self.three_size_disturb, self.three_size_disturb))
        # fixbox_size[1, 1] = fixbox_size[0, 1] * (1 + sample_float(-0.1, 0.1))

        # fixbox_size[2, 0] = fixbox_size[0, 0] * (1 + sample_float(-0.1, 0.1))
        # fixbox_size[2, 1] = fixbox_size[0, 1] * (1 + sample_float(-self.three_size_disturb, self.three_size_disturb))

        for i in range(3):
            self.fixbox_list[i] = create_fixbox(fixbox_size[i], self.corner)

        # 外延距离
        x_outline = 0
        y_outline = 0
        # 原始对齐距离
        x_foralign = 0
        y_foralign = 0

        # 上侧对齐长, 右侧对齐短
        if (fixbox_size[0, 0] <= fixbox_size[1, 0] and fixbox_size[0, 1] >= fixbox_size[2, 1]) or (fixbox_size[0, 0] >= fixbox_size[1, 0] and fixbox_size[0, 1] <= fixbox_size[2, 1]):
            self.fixbox_list[1].set_position([0, -fixbox_size[0, 1] - box_gap[1], 0], self.fixbox_list[1])
            self.fixbox_list[2].set_position([-fixbox_size[0, 0] - box_gap[2], 0, 0], self.fixbox_list[2])
            
            x_outline = fixbox_size[0, 0] + fixbox_size[2, 0] + box_gap[2]
            y_outline = fixbox_size[0, 1] + fixbox_size[1, 1] + box_gap[1]

            y_foralign = y_outline - fixbox_size[2, 1]
            x_foralign = x_outline - fixbox_size[1, 0]

            # set_pose6_by_self(self.anchor_core_object, np.array([
            #     -fixbox_size[1, 0], 
            #     -fixbox_size[2, 1], 
            #     np.max(fixbox_size[:, 2])
            # ]) * 1e3)

        # 两侧长
        elif fixbox_size[0, 0] < fixbox_size[1, 0] and fixbox_size[0, 1] < fixbox_size[2, 1]:
            # 让空隙尽量小
            if fixbox_size[1, 0] - fixbox_size[0, 0] > fixbox_size[2, 1] - fixbox_size[0, 1]:
                # 与右侧对齐
                self.fixbox_list[1].set_position([0, -fixbox_size[2, 1] - box_gap[1], 0], self.fixbox_list[1])
                self.fixbox_list[2].set_position([-fixbox_size[0, 0] - box_gap[2], 0, 0], self.fixbox_list[2])
            
                x_outline = fixbox_size[0, 0] + fixbox_size[2, 0] + box_gap[2]
                y_outline = fixbox_size[2, 1] + fixbox_size[1, 1] + box_gap[1]

                y_foralign = y_outline - fixbox_size[2, 1]
                x_foralign = x_outline - fixbox_size[1, 0]
            else:
                self.fixbox_list[1].set_position([0, -fixbox_size[0, 1] - box_gap[1], 0], self.fixbox_list[1])
                self.fixbox_list[2].set_position([-fixbox_size[1, 0] - box_gap[2], 0, 0], self.fixbox_list[2])
            
                x_outline = fixbox_size[1, 0] + fixbox_size[2, 0] + box_gap[2]
                y_outline = fixbox_size[0, 1] + fixbox_size[1, 1] + box_gap[1]

                y_foralign = y_outline - fixbox_size[2, 1]
                x_foralign = x_outline - fixbox_size[1, 0]

            # set_pose6_by_self(self.anchor_core_object, np.array([
            #     -fixbox_size[1, 0], 
            #     -fixbox_size[2, 1], 
            #     np.max(fixbox_size[:, 2])
            # ]) * 1e3)

        # 两侧短
        elif fixbox_size[0, 0] > fixbox_size[1, 0] and fixbox_size[0, 1] > fixbox_size[2, 1]:
            # 两侧对齐
            self.fixbox_list[1].set_position([0, -fixbox_size[0, 1] - box_gap[1], 0], self.fixbox_list[1])
            self.fixbox_list[2].set_position([-fixbox_size[0, 0] - box_gap[2], 0, 0], self.fixbox_list[2])
            x_outline = fixbox_size[0, 0] + fixbox_size[2, 0] + box_gap[2]
            y_outline = fixbox_size[0, 1] + fixbox_size[1, 1] + box_gap[1]

            # 当对齐区域小于 5mm, 优先与 1 对齐为目标
            if (fixbox_size[0, 0] - fixbox_size[1, 0] > fixbox_size[0, 1] - fixbox_size[2, 1]) or (max(fixbox_size[0, 0] - fixbox_size[1, 0], fixbox_size[0, 1] - fixbox_size[2, 1]) < 5 * 1e-3):

                y_foralign = y_outline - fixbox_size[0, 1]
                x_foralign = x_outline - fixbox_size[1, 0]

            else:
                y_foralign = y_outline - fixbox_size[2, 1]
                x_foralign = x_outline - fixbox_size[0, 0]

        # 是否生成额外箱子
        # 上侧额外判定
        is_3_available = bool(np.random.random() < self.three_extra_fixbox_prob)
        if is_3_available:
            # fixbox_size[3, 0] = fixbox_size[0, 0] * (1 + sample_float(-self.three_size_disturb, self.three_size_disturb))
            self.fixbox_list[3] = create_fixbox(fixbox_size[3], self.corner)
            self.fixbox_list[3].set_position([0, -y_outline - box_gap[3], 0], self.fixbox_list[3])
            y_foralign += box_gap[3]
        # else:
        # 当 1 足够长或 1 比 3 凸出, 3 对对齐没有影响
        if not is_3_available or (y_foralign > movebox_size[1]) or (fixbox_size[1, 0] > fixbox_size[3, 0]):
            # 不生成相当于尺寸为 0
            fixbox_size[3] = np.zeros(3)
            is_3_available = False

        is_4_available = bool(np.random.random() < self.three_extra_fixbox_prob)
        if is_4_available:
            # fixbox_size[4, 1] = fixbox_size[0, 1] * (1 + sample_float(-self.three_size_disturb, self.three_size_disturb))
            self.fixbox_list[4] = create_fixbox(fixbox_size[4], self.corner)
            self.fixbox_list[4].set_position([-x_outline - box_gap[4], 0, 0], self.fixbox_list[4])
            x_foralign += box_gap[4]
        # else:
        # 当 2 足够长, 4 对对齐没有影响
        if not is_4_available or (x_foralign > movebox_size[0]) or (fixbox_size[2, 1] > fixbox_size[4, 1]):
            # 不生成相当于尺寸为 0
            fixbox_size[4] = np.zeros(3)
            is_4_available = False

        align_rect = try_align_xy(
            self.fixbox_list, 
            fixbox_size, 
            movebox_size, 
            self.corner,
            -(fixbox_size[0, 0] + fixbox_size[2, 0] + fixbox_size[4, 0] + movebox_size[0] / 2 - 0.001),
            -(fixbox_size[0, 1] + fixbox_size[1, 1] + fixbox_size[3, 1] + movebox_size[1] / 2 - 0.001))
        set_pose6_by_self(self.anchor_core_object, np.array([
            align_rect.maxX, 
            align_rect.maxY, 
            np.max(fixbox_size[:, 2])
        ]) * 1e3)
        # print(self.env_center_adjust)
        self.zoffset = np.max(fixbox_size[:, 2])

        if not self.debug_center_check:
            set_pose6_by_self(self.anchor_center_align_movebox_pose, np.array([
                self.env_center_adjust, 
                self.env_center_adjust, 
                0
            ]) * 1e3)
        else:
            set_pose6_by_self(self.anchor_test_object, np.array([self.env_tolerance_offset / 2, self.env_tolerance_offset / 2, 0]) * 1e3)

        # debug
        # set_pose6_by_self(self.anchor_test_object, np.array([self.env_tolerance_offset / 2, self.env_tolerance_offset / 2, 0]) * 1e3)

    def _set_init_fixbox_scale(self):

        if self.three_fixbox_size_range is None or self.three_size_disturb_range is None:
            warnings.warn(f"没有提供两边对齐环境的关键参数 {self.three_fixbox_size_range} 或 {self.three_size_disturb_range}")
            return
        
        # 基础尺寸
        fixbox_size = np.array([[
            self.sample_float(self.three_fixbox_size_range[0][i], self.three_fixbox_size_range[1][i])
            for i in range(3)
        ] for _ in range(5)], np.float32)

        if self.three_box_gap_range is not None:
            fixbox_gap_size = np.array([
                self.sample_float(self.three_box_gap_range[0], self.three_box_gap_range[1])
            for _ in range(5)], np.float32)
            fixbox_gap_size[0] = 0
        else:
            fixbox_gap_size = np.zeros(5)

        # 间隙尺寸
        if self.three_is_zero_middle:
            random_pm = np.random.choice([0, -1, 1], 4)
        else:
            random_pm = np.random.choice([-1, 1], 4)

        fixbox_size[1, 0] = fixbox_size[0, 0] + random_pm[0] * self.sample_float(self.three_size_disturb_range[0], self.three_size_disturb_range[1])
        fixbox_size[2, 1] = fixbox_size[0, 1] + random_pm[1] * self.sample_float(self.three_size_disturb_range[0], self.three_size_disturb_range[1])
        fixbox_size[3, 0] = fixbox_size[1, 0] + random_pm[2] * self.sample_float(self.three_size_disturb_range[0], self.three_size_disturb_range[1])
        fixbox_size[4, 1] = fixbox_size[2, 1] + random_pm[3] * self.sample_float(self.three_size_disturb_range[0], self.three_size_disturb_range[1])

        self._set_fixbox_scale(fixbox_size, fixbox_gap_size)

    def _set_max_init(self, direct: Literal[0, 1] = 0):
        '''
        测试, 用于使用最大扰动初始化环境
        '''
        super()._set_max_init(direct)
        if self.three_fixbox_size_range is not None and self.three_size_disturb_range is not None:

            if self.three_box_gap_range is not None:
                fixbox_gap_size = np.ones(5, np.float32) * self.three_box_gap_range[direct]
                fixbox_gap_size[0] = 0
            else:
                fixbox_gap_size = np.zeros(5)

            # 基础尺寸
            fixbox_size = np.array([
                self.three_fixbox_size_range[int(np.random.random() > 0.5)],
                self.three_fixbox_size_range[int(np.random.random() > 0.5)],
                self.three_fixbox_size_range[int(np.random.random() > 0.5)],
                self.three_fixbox_size_range[int(np.random.random() > 0.5)],
                self.three_fixbox_size_range[int(np.random.random() > 0.5)]
            ])
            random_pm = np.astype(np.random.random(4) > 0.5, np.float32) * 2 - 1
            fixbox_size[1, 0] = fixbox_size[0, 0] + random_pm[0] * self.sample_float(self.three_size_disturb_range[0], self.three_size_disturb_range[1])
            fixbox_size[2, 1] = fixbox_size[0, 1] + random_pm[1] * self.sample_float(self.three_size_disturb_range[0], self.three_size_disturb_range[1])
            fixbox_size[3, 0] = fixbox_size[0, 0] + random_pm[2] * self.sample_float(self.three_size_disturb_range[0], self.three_size_disturb_range[1])
            fixbox_size[4, 1] = fixbox_size[0, 1] + random_pm[3] * self.sample_float(self.three_size_disturb_range[0], self.three_size_disturb_range[1])

            self._set_fixbox_scale(fixbox_size, fixbox_gap_size)

    def reset(self):
        '''
        重新初始化
        '''
        super().reset()
        self._set_init_fixbox_scale()

class ThreeEnv(PlaneBoxEnv):

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
        # 是否将运动坐标系设置在拐角
        env_is_corner_move: bool = False,

        dataset_is_center_goal: bool = False,
        # debug, 用于调整最佳插入位置为无间隙区域以用于测试卷积神经网络的性能 (需要根据不同的场景具体实现)
        debug_center_check: bool = False,
        debug_close_pr: bool = False,
        # debug, 用于模拟重心偏移下的误差分布 (用于训练特征提取器)
        debug_is_fake_center_err: bool = False,

        three_fixbox_size_range: Optional[Tuple[Union[Sequence[float], np.ndarray], Union[Sequence[float], np.ndarray]]] = None,
        # 纸箱对齐侧长度变动比率
        three_size_disturb_range: Optional[Tuple[float, float]] = None,
        # 出现额外纸箱的概率
        three_extra_fixbox_prob: float = 0.5,

        # 是否出现长度变动为 0 的中间态
        three_is_zero_middle: bool = False,
        
        # 纸箱间间隙尺寸范围
        three_box_gap_range: Optional[Tuple[float, float]] = None,

    ) -> None:
        if three_fixbox_size_range is not None:
            _three_fixbox_size_range = (
                np.asarray(three_fixbox_size_range[0], np.float32),
                np.asarray(three_fixbox_size_range[1], np.float32),
            )
        else:
            _three_fixbox_size_range = None

        super().__init__(
            env_pr = env_pr, 
            subenv_mid_name = subenv_mid_name, subenv_range = subenv_range, subenv_make_fn = ThreeSubEnv, 
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
            debug_center_check = debug_center_check,
            debug_close_pr = debug_close_pr,
            debug_is_fake_center_err = debug_is_fake_center_err,
            dataset_is_center_goal = dataset_is_center_goal,
            three_fixbox_size_range = _three_fixbox_size_range,
            three_size_disturb_range = three_size_disturb_range,
            three_extra_fixbox_prob = three_extra_fixbox_prob,
            three_box_gap_range = three_box_gap_range,
            three_is_zero_middle = three_is_zero_middle,
            act_type = act_type,
            env_is_unlimit_time = env_is_unlimit_time, env_is_terminate_when_insert = env_is_terminate_when_insert
        )
