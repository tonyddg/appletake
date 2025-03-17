from pyrep import PyRep
from pyrep.objects.shape import Shape, PrimitiveShape
from pyrep.objects.dummy import Dummy
from pyrep.objects.object import Object

from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Union
import numpy as np
from albumentations.core.composition import TransformType

from ...gym_env.utility import set_pose6_by_self

from .plane_box import EnvObjectsBase, PlaneBoxEnv, RewardFnType, PlaneBoxSubenvBase

# TODO: 已放置箱子的位置尺寸扰动
# TODO: 环境中其他物体扰动

class PlaneAndThree(EnvObjectsBase):
    def __init__(self, plane: Shape, three_group: Sequence[Shape]) -> None:
        self.plane = plane
        self.three_group = three_group

    def check_collision(self, obj: Object) -> bool:
        for env_obj in self.three_group:
            if env_obj.check_collision(obj):
                return True
        return self.plane.check_collision(obj)

class ThreeSubEnv(PlaneBoxSubenvBase):

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

        # 三个已放置纸箱的尺寸范围 (单位 m)
        three_fixbox_size_range: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        **subenv_kwargs: Any,
    ) -> None:
        
        self.plane = Shape("Plane" + name_suffix)
        # 对于环境问题的补救
        self.plane.set_dynamic(False)
        self.plane.set_respondable(False)

        self.corner = Dummy("CornerPosition" + name_suffix)
        self.fixbox_list: List[Shape] = [
            Shape.create(PrimitiveShape.CUBOID, [0.001, 0.001, 0.001])
        for i in range(3)]

        super().__init__(
            name_suffix = name_suffix, 
            obs_trans = obs_trans, obs_source = obs_source, env_object = PlaneAndThree(self.plane, self.fixbox_list), 
            env_action_noise = env_action_noise, env_init_box_pos_range = env_init_box_pos_range, env_init_vis_pos_range = env_init_vis_pos_range, env_vis_persp_deg_disturb = env_vis_persp_deg_disturb,
            env_movbox_height_offset_range = env_movbox_height_offset_range,
            env_movebox_center_err = env_movebox_center_err,
            env_is_complexity_progression = env_is_complexity_progression, 
            env_minium_ratio = env_minium_ratio,
            env_tolerance_offset = env_tolerance_offset, env_test_in = env_test_in, env_max_step = env_max_step,
            **subenv_kwargs
        )

        self.three_fixbox_size_range = three_fixbox_size_range
    
    def _to_origin_state(self):
        super()._to_origin_state()
        for i in range(len(self.fixbox_list)):
            if self.fixbox_list[i].exists(self.fixbox_list[i].get_name()):
                self.fixbox_list[i].remove()

    def _set_fixbox_scale(self, fixbox_size: np.ndarray):
        # A 对角, B 左侧, C 上侧
        set_type = bool(np.random.random() > 0.5)
        if set_type: # 左侧对齐
            fixbox_size[1, 0] = fixbox_size[0, 0]
        else:
            fixbox_size[2, 1] = fixbox_size[0, 1]

        for i in range(len(self.fixbox_list)):
            self.fixbox_list[i] = Shape.create(PrimitiveShape.CUBOID, [float(fixbox_size[i, k]) for k in range(3)])
            self.fixbox_list[i].set_dynamic(False)
            self.fixbox_list[i].set_respondable(False)
            self.fixbox_list[i].set_collidable(True)
            self.fixbox_list[i].set_renderable(True)

            self.fixbox_list[i].set_color([float(k == i) for k in range(3)])

            self.fixbox_list[i].set_position([-fixbox_size[i, 0] / 2, -fixbox_size[i, 1] / 2, fixbox_size[i, 2] / 2], self.corner)

        self.fixbox_list[1].set_position([0, -fixbox_size[0, 1], 0], self.fixbox_list[1])
        self.fixbox_list[2].set_position([-fixbox_size[0, 0], 0, 0], self.fixbox_list[2])

        if set_type: # 左侧对齐
            set_pose6_by_self(self.anchor_core_object, np.array([
                -fixbox_size[0, 0], 
                -fixbox_size[2, 1], 
                np.max(fixbox_size[:, 2])
            ]) * 1e3)
        else:
            set_pose6_by_self(self.anchor_core_object, np.array([
                -fixbox_size[1, 0], 
                -fixbox_size[0, 1], 
                np.max(fixbox_size[:, 2])
            ]) * 1e3)

    def _set_init_fixbox_scale(self):

        if self.three_fixbox_size_range is None:
            return
        
        fixbox_size = np.array([[
            self.sample_float(self.three_fixbox_size_range[0][i], self.three_fixbox_size_range[1][i])
            for i in range(3)
        ] for _ in range(3)], np.float32)

        self._set_fixbox_scale(fixbox_size)

    def _set_max_init(self, direct: Literal[0, 1] = 0):
        '''
        测试, 用于使用最大扰动初始化环境
        '''
        super()._set_max_init(direct)
        if self.three_fixbox_size_range is not None:
            self._set_fixbox_scale(
                np.array([
                    self.three_fixbox_size_range[int(np.random.random() > 0.5)],
                    self.three_fixbox_size_range[int(np.random.random() > 0.5)],
                    self.three_fixbox_size_range[int(np.random.random() > 0.5)]
                ])
            )

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

        three_fixbox_size_range: Optional[Tuple[Union[Sequence[float], np.ndarray], Union[Sequence[float], np.ndarray]]] = None,
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
            env_reward_fn = env_reward_fn, env_tolerance_offset = env_tolerance_offset, env_test_in = env_test_in, env_max_step = env_max_step, 
            env_action_noise = env_action_noise, env_init_box_pos_range = env_init_box_pos_range, env_init_vis_pos_range = env_init_vis_pos_range, 
            env_vis_persp_deg_disturb = env_vis_persp_deg_disturb, env_movbox_height_offset_range = env_movbox_height_offset_range,
            env_movebox_center_err = env_movebox_center_err,
            env_is_complexity_progression = env_is_complexity_progression, 
            env_minium_ratio = env_minium_ratio,
            act_unit = act_unit, train_total_timestep = train_total_timestep,
            three_fixbox_size_range = _three_fixbox_size_range,
        )
