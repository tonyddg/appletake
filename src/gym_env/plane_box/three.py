from pyrep import PyRep
from pyrep.objects.shape import Shape

from typing import Any, Dict, Literal, Optional, Sequence, Tuple, Union
import numpy as np
from albumentations.core.composition import TransformType

from .plane_box import PlaneBoxEnv, RewardFnType, PlaneBoxSubenvBase, GroupEnvObject

# TODO: 已放置箱子的位置尺寸扰动
# TODO: 环境中其他物体扰动

class ThreeSubEnv(PlaneBoxSubenvBase):

    def __init__(
        self, 
        name_suffix: str,

        obs_trans: TransformType,
        obs_source: Literal["color", "depth"],

        env_object_kwargs: Optional[Dict[str, Any]] = None,

        env_action_noise: Optional[np.ndarray] = None,
        env_init_box_pos_range: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        env_init_vis_pos_range: Optional[Tuple[np.ndarray, np.ndarray]] = None,
 
        env_tolerance_offset: float = 0,
        env_test_in: float = 0.05,
        env_max_step: int = 20
    ) -> None:
        
        self.plane = Shape("Plane" + name_suffix)
        self.fixbox_a = Shape("FixBoxA" + name_suffix)
        self.fixbox_b = Shape("FixBoxB" + name_suffix)
        self.fixbox_c = Shape("FixBoxC" + name_suffix)

        super().__init__(
            name_suffix, obs_trans, obs_source, GroupEnvObject([self.plane, self.fixbox_a, self.fixbox_b, self.fixbox_c]), env_action_noise, env_init_box_pos_range, env_init_vis_pos_range,
            env_tolerance_offset, env_test_in, env_max_step
        )

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
        env_test_in: float = 0.08,
        env_max_step: int = 20,
        # 直接加在原始动作上, 不转换
        env_action_noise: Optional[np.ndarray] = None,
        env_init_box_pos_range: Optional[Tuple[Union[Sequence[float], np.ndarray], Union[Sequence[float], np.ndarray]]] = None,
        env_init_vis_pos_range: Optional[Tuple[Union[Sequence[float], np.ndarray], Union[Sequence[float], np.ndarray]]] = None,
        # 单位 mm, deg
        act_unit: Sequence[float] = (5, 5, 5, 1, 1, 1),
    ) -> None:
        super().__init__(
            env_pr, subenv_mid_name, subenv_range, ThreeSubEnv, None, obs_trans, obs_source, 
            env_reward_fn, env_tolerance_offset, env_test_in, env_max_step, 
            env_action_noise, env_init_box_pos_range, env_init_vis_pos_range, act_unit
        )
