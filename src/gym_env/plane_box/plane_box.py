from pyrep import PyRep
from pyrep.objects.shape import Shape
from pyrep.objects.vision_sensor import VisionSensor
from pyrep.objects.object import Object
from pyrep.objects.dummy import Dummy

from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvObs, VecEnvStepReturn
from stable_baselines3.common import type_aliases

from typing import Any, Callable, Dict, List, Literal, Optional, Protocol, Sequence, Tuple, Union
from dataclasses import dataclass
import gymnasium as gym
import numpy as np
from matplotlib import pyplot as plt
from abc import ABCMeta, abstractmethod
from pprint import pp as pprint

from ...net.pr_task_dataset import PrTaskVecEnvForDatasetInterface

from ..utility import get_quat_diff_rad, get_rel_pose, mrad_to_mmdeg, sample_float, set_pose6_by_self, sample_vec, DIRECT2INDEX
from ...utility import get_file_time_str
from ...conio.key_listen import KEY_CB_DICT_TYPE

from albumentations.core.composition import TransformType
from ..aug import test_trans_obs_space

class EnvObjectsBase(metaclass = ABCMeta):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def check_collision(self, obj: Object) -> bool:
        raise NotImplementedError()

class GroupEnvObject(EnvObjectsBase):
    def __init__(self, group_objects: Sequence[Object]) -> None:
        self.group_objects = group_objects

    def check_collision(self, obj: Object) -> bool:
        for env_obj in self.group_objects:
            if env_obj.check_collision(obj):
                return True
        return False

@dataclass
class TestWall(EnvObjectsBase):
    test_a: Shape
    test_b: Shape
    test_c: Shape
    test_d: Shape

    def offset_tolerance(self, offset: float):
        '''
        认为 B, D 为可动的边界, 且自身坐标系的 +z 方向为增大容差的方向  
        默认容差为 1cm  
        offset 单位为 m  
        需要调用 pr.step 使更改生效

        测试与验证共用环境时, 要小心出错
        '''
        self.test_b.set_position(np.array([0, 0, offset]), self.test_b)
        self.test_d.set_position(np.array([0, 0, offset]), self.test_d)
    
    def check_collision(self, obj: Object):
        return self.test_a.check_collision(obj) or\
                self.test_b.check_collision(obj) or\
                self.test_c.check_collision(obj) or\
                self.test_d.check_collision(obj)

class PlaneBoxSubenvBase(metaclass = ABCMeta):

    def __init__(
        self, 
        name_suffix: str,

        obs_trans: TransformType,
        obs_source: Literal["color", "depth"],

        env_object: Union[EnvObjectsBase, Object],

        # 基础环境噪音
        # 动作噪音
        env_action_noise: Optional[np.ndarray] = None,
        # 纸箱初始位置噪音
        env_init_box_pos_range: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        # 相机相对位置噪音
        env_init_vis_pos_range: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        # 相机观测角噪音 (Coppeliasim 为最大边 FOV)
        env_vis_persp_deg_disturb: Optional[float] = None,
        # 纸箱高度变化 (基于 m 单位)
        env_movbox_height_offset_range: Optional[Tuple[float, float]] = None,
 
        env_tolerance_offset: float = 0,
        # Checker 长度为 50mm, 因此仅当箱子与垛盘间隙为 51~100 mm (相对原始位置偏移 1~50 mm) 时可通过检查 
        env_test_in: float = 0.05,
        env_max_step: int = 20,

        # 基于 [px, py, pz] 单位 m 的最佳位置偏移 (直接相加), (考虑 get_coppeliasim_depth_normalize, 原始间隙为 1cm)
        # env_move_box_best_position_offset: np.ndarray = np.array([-0.005, -0.005, 0.025])
        **subenv_kwargs: Any,
    ):
        '''
        底层构建方法
        '''

        assert len(subenv_kwargs) == 0, f"有尚未提取的参数: {subenv_kwargs}"

        # 环境物体锚点
        self.anchor_env_object = Dummy("EnvObject" + name_suffix)
        # 核心物体锚点
        self.anchor_core_object = Dummy("CoreObject" + name_suffix)
        # 运动物体锚点
        self.anchor_move_object = Dummy("MoveObject" + name_suffix)
        # 检测区域锚点
        self.anchor_test_object = Dummy("TestObject" + name_suffix)

        self.move_box: Shape = Shape("MoveBox" + name_suffix)
        self.env_object: Union[EnvObjectsBase, Object] = env_object

        self.vis_anchor: Dummy = Dummy("VisAnchor" + name_suffix)
        self.color_camera: VisionSensor= VisionSensor("ColorCamera" + name_suffix)
        self.depth_camera: VisionSensor= VisionSensor("DepthCamera" + name_suffix)

        self.obs_trans: TransformType = obs_trans
        self.obs_source: Literal["color", "depth"] = obs_source

        self.test_wall = TestWall(
            Shape("TestA" + name_suffix),
            Shape("TestB" + name_suffix),
            Shape("TestC" + name_suffix),
            Shape("TestD" + name_suffix),
        )
        self.test_wall.offset_tolerance(env_tolerance_offset)
        self.test_check = Shape("TestCheck" + name_suffix)

        ###

        self.env_test_in: float = env_test_in
        self.env_max_step: int = env_max_step
        self.env_action_noise: Optional[np.ndarray] = env_action_noise

        ### 记录位置关系的初始状态

        # 核心锚点 (移动物体与检测区域) -> None
        self.origin_pose_core_anchor_relnone = self.anchor_core_object.get_pose(None)

        # 移动锚点 (盒子与相机) -> 核心锚点
        self.origin_pose_move_anchor_relcore = self.anchor_move_object.get_pose(self.anchor_core_object)
        # 移动盒子
        self.origin_pose_move_box_relmove: np.ndarray = self.move_box.get_pose(self.anchor_move_object)
        # 相机
        self.origin_pose_vis_anchor_relmove: np.ndarray = self.vis_anchor.get_pose(self.anchor_move_object)

        # 检测锚点 (检测区域) -> 核心锚点
        self.origin_pose_test_anchor_relcore = self.anchor_test_object.get_pose(self.anchor_core_object)

        ###

        self.env_init_box_pos_range = env_init_box_pos_range
        self.env_init_vis_pos_range = env_init_vis_pos_range

        # 箱子的最佳位置, 用于奖励函数与神经网络构建等
        self.best_pose_move_anchor_relcore = np.copy(self.origin_pose_move_anchor_relcore)

        # 根据容差自动计算
        xy_best_position_offset = (env_tolerance_offset + 0.01) / 2
        env_move_box_best_position_offset: np.ndarray = np.array([-xy_best_position_offset, -xy_best_position_offset, 0.025])

        self.best_pose_move_anchor_relcore[:3] += env_move_box_best_position_offset
        self.env_move_box_best_position_offset = env_move_box_best_position_offset

        ###

        self.env_vis_fov_disturb = env_vis_persp_deg_disturb
        # 相机初始视场角
        self.vis_origin_persp_deg: float = 0
        if self.obs_source == "color":
            self.vis_origin_persp_deg = self.color_camera.get_perspective_angle()
        else:
            self.vis_origin_persp_deg = self.depth_camera.get_perspective_angle()

        ###

        self.env_movbox_height_offset_range = env_movbox_height_offset_range
        self.movebox_origin_height = self.move_box.get_bounding_box()[5] - self.move_box.get_bounding_box()[4]
        self._last_scale_movebox = None

        ###

        self._cur_position_move_box_relcore: Optional[np.ndarray] = None
        self._subenv_step: int = 0
        self._last_render: Optional[np.ndarray] = None

    ###

    def get_obs(self) -> np.ndarray:
        if self.obs_source == "color":
            return self.obs_trans(image = self.color_camera.capture_rgb())["image"]
        else:
            # 通过自定义转换进行标准化
            return self.obs_trans(image = self.depth_camera.capture_depth())["image"]

    def get_space(self):
        if self.obs_source == "color":
            return test_trans_obs_space(gym.spaces.Box(0, 1, [1080, 1920, 3]), self.obs_trans)
        else:
            return test_trans_obs_space(gym.spaces.Box(0, 1, [720, 1080]), self.obs_trans)

    def render(self):
        if self._last_render is None:
            self._last_render = self.color_camera.capture_rgb()

        return self._last_render

    ###

    def is_collision_with_env(self):
        return self.env_object.check_collision(self.move_box)
    
    def pre_alignment_detect(
            self
        ):
        self._cur_position_move_box_relcore = self.move_box.get_position(self.anchor_core_object) 
        # 执行插入检测前的准备
        self.move_box.set_position([0, 0, -self.env_test_in], self.move_box)

    def post_alignment_detect(self):
        assert self._cur_position_move_box_relcore is not None, "请先执行 pre_alignment_detect"

        is_collision = self.test_wall.check_collision(self.move_box)
        is_alignment = self.move_box.check_collision(self.test_check)

        self.move_box.set_position(self._cur_position_move_box_relcore, self.anchor_core_object)
        # 碰撞到检测区域, 并且没有碰撞到插座区域
        return (not is_collision) and is_alignment

    def tack_action(self, action: np.ndarray):

        if self.env_action_noise is not None:
            action += self.env_action_noise * (2 * np.random.rand(*action.shape) - 1)
        set_pose6_by_self(self.anchor_move_object, action)

        self._subenv_step += 1
        self._last_render = None

    def is_truncated(self):
        return self._subenv_step >= self.env_max_step

    ###

    def _to_origin_state(self):
        '''
        返回原始状态
        '''

        # 核心锚点 (移动物体与检测区域) -> None
        self.anchor_core_object.set_pose(self.origin_pose_core_anchor_relnone, None)
        
        # 移动锚点 (盒子与相机) -> 核心锚点
        self.anchor_move_object.set_pose(self.origin_pose_move_anchor_relcore, self.anchor_core_object)
        # 移动盒子
        self.move_box.set_pose(self.origin_pose_move_box_relmove, self.anchor_move_object)
        # 相机
        self.vis_anchor.set_pose(self.origin_pose_vis_anchor_relmove, self.anchor_move_object)

        # 检测锚点 (检测区域) -> 核心锚点
        self.anchor_test_object.set_pose(self.origin_pose_test_anchor_relcore, self.anchor_core_object)

        if self._last_scale_movebox is not None:
            self.move_box.scale_object(1, 1, 1 / self._last_scale_movebox)
            self._last_scale_movebox = None

    def _set_vis_fov_offset(self, offset: float):
        operate_vis = self.color_camera if self.obs_source == "color" else self.depth_camera
        operate_vis.set_perspective_angle(self.vis_origin_persp_deg + offset)

    def _set_movebox_height_offset(self, offset: float):
        
        self._last_scale_movebox = offset / self.movebox_origin_height + 1
        self.move_box.scale_object(1, 1, self._last_scale_movebox)

        # 保持盒子底面与检测区的接触
        set_pose6_by_self(self.vis_anchor, np.array([0, 0, offset / 2]) * 1e3)
        set_pose6_by_self(self.anchor_move_object, np.array([0, 0, offset / 2]) * 1e3)

    def _set_init_move_anchor_pos(
            self,
        ):
        '''
        设置箱子初始位置 (基于最佳位置随机初始化)
        '''
        if self.env_init_box_pos_range is not None:
            init_pose = sample_vec(self.env_init_box_pos_range[0], self.env_init_box_pos_range[1])
            set_pose6_by_self(self.anchor_move_object, init_pose)

    def _set_init_vis_pos(
            self,
        ):
        '''
        设置相机初始位置
        '''
        if self.env_init_vis_pos_range is not None:
            init_pose = sample_vec(self.env_init_vis_pos_range[0], self.env_init_vis_pos_range[1])
            set_pose6_by_self(self.vis_anchor, init_pose)

    def _set_init_vis_fov(
            self
        ):
        '''
        设置相机初始视场角
        '''
        if self.env_vis_fov_disturb:
            self._set_vis_fov_offset(self.env_vis_fov_disturb * float(np.random.random(1)))

    def _set_init_move_box_height(
            self,
        ):
        '''
        设置箱子初始高度
        '''
        # 从均匀分布随机采样 (可改为正态分布 ?)
        if self.env_movbox_height_offset_range is not None:
            init_offset = sample_float(self.env_movbox_height_offset_range[0], self.env_movbox_height_offset_range[1])
            self._set_movebox_height_offset(init_offset)

    def _set_max_init(self, direct: Literal[0, 1] = 0):
        '''
        测试, 用于使用最大扰动初始化环境
        '''
        self._to_origin_state()

        if self.env_init_box_pos_range is not None:
            set_pose6_by_self(self.anchor_move_object, self.env_init_box_pos_range[direct])
        if self.env_init_vis_pos_range is not None:
            set_pose6_by_self(self.vis_anchor, self.env_init_vis_pos_range[direct])
        if self.env_vis_fov_disturb is not None:
            if direct == 0:
                self._set_vis_fov_offset(-self.env_vis_fov_disturb)
            else:
                self._set_vis_fov_offset(self.env_vis_fov_disturb)
        if self.env_movbox_height_offset_range is not None:
            self._set_movebox_height_offset(self.env_movbox_height_offset_range[direct])

    def reset(self):
        '''
        重新初始化
        '''

        self._subenv_step = 0
        self._last_render = None

        self._to_origin_state()

        self._set_init_vis_pos()
        self._set_init_vis_fov()
        self._set_init_move_box_height()
        self._set_init_move_anchor_pos()

    ###

    def _step_env_check(self, pr: PyRep):
        '''
        返回 is_alignments, is_collisions
        '''

        is_collision = self.is_collision_with_env()

        self.pre_alignment_detect()
        pr.step()
        is_alignment = self.post_alignment_detect()
        pr.step()

        return (is_alignment, is_collision)

    def _step_take_action(self, action: np.ndarray, pr: PyRep):
        self.tack_action(action)
        pr.step()

        return self.is_truncated()

    def _step(self, action: np.ndarray, pr: PyRep):
        '''
        返回 is_truncated, is_alignments, is_collisions
        '''
        is_truncated = self._step_take_action(action, pr)
        is_alignments, is_collisions = self._step_env_check(pr)

        if is_truncated or is_alignments or is_collisions:
            self.reset()
        return is_truncated, is_alignments, is_collisions
    
#     raise NotImplementedError

# reward = reward_fn(subenv, is_alignments, is_collisions)
class RewardFnABC(metaclass = ABCMeta):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def __call__(self, subenv: PlaneBoxSubenvBase, is_alignment: bool, is_collision: bool) -> float:
        raise NotImplementedError()

# (self, subenv: PlaneBoxSubenvBase, is_alignment: bool, is_collision: bool) -> float:
RewardFnType = Union[Callable[[PlaneBoxSubenvBase, bool, bool], float], RewardFnABC]

class PlaneBoxSubenvInitProtocol(Protocol):
    def __call__(
        self, 
        name_suffix: str,

        obs_trans: TransformType,
        obs_source: Literal["color", "depth"],

        env_action_noise: Optional[np.ndarray] = None,
        env_init_box_pos_range: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        env_init_vis_pos_range: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        env_vis_persp_deg_disturb: Optional[float] = None,
        env_movbox_height_offset_range: Optional[Tuple[float, float]] = None,

        env_tolerance_offset: float = 0,
        env_test_in: float = 0.05,
        env_max_step: int = 20,

        **subenv_kwargs: Any
    ) -> PlaneBoxSubenvBase: 
        ...

class RewardSpare(RewardFnABC):
    def __init__(
            self, 
            time_panelty: float = -0.1, 
            colision_panelty: float = -1, 
            success_reward: float = 1
        ) -> None:
        self.time_panelty = time_panelty
        self.colision_panelty = colision_panelty
        self.success_reward = success_reward

    def __call__(self, subenv: PlaneBoxSubenvBase, is_alignment: bool, is_collision: bool) -> float:
        if is_collision:
            return self.colision_panelty
        elif is_alignment:
            return self.success_reward
        else:
            return self.time_panelty

class RewardLinearDistance(RewardFnABC):
    def __init__(
            self, 
            max_pos_dis_mm: float = 40,
            max_rot_dis_deg: float = 10,
            time_panelty: float = -0.1, 
            colision_panelty: float = -1, 
            success_reward: float = 1
        ) -> None:
        '''
        基于相对 best pose 给出奖励
        '''
        self.max_pos_dis = max_pos_dis_mm * 1e-3
        self.max_rot_dis = float(np.deg2rad(max_rot_dis_deg))
        self.time_panelty = time_panelty
        self.colision_panelty = colision_panelty
        self.success_reward = success_reward

    def __call__(self, subenv: PlaneBoxSubenvBase, is_alignment: bool, is_collision: bool) -> float:
        if is_collision:
            return self.colision_panelty
        elif is_alignment:
            return self.success_reward
        else:
            cur_pose = subenv.anchor_move_object.get_pose(subenv.anchor_core_object)

            diff_xy = np.linalg.norm(cur_pose[:2] - subenv.best_pose_move_anchor_relcore[:2], 2)
            diff_rad = get_quat_diff_rad(cur_pose[3:], subenv.best_pose_move_anchor_relcore[3:])

            reach_xy = max(float(self.max_pos_dis - diff_xy), 0.0) / self.max_pos_dis
            reach_ang = max(self.max_rot_dis - diff_rad, 0) / self.max_rot_dis
            reach_reward = (reach_xy + reach_ang) / 2

            return (1 - reach_reward) * self.time_panelty

class PlaneBoxSubenvTest:
    def __init__(
            self,
            pr: PyRep,
            env: PlaneBoxSubenvBase,
            pos_unit: float = 10, 
            rot_unit: float = 1, 
            reward_fn: RewardFnType = lambda *args: 0
        ) -> None:
        self.pr = pr
        self.env = env
        self.pos_unit, self.rot_unit = pos_unit, rot_unit
        self.reward_fn = reward_fn
    
        self.fig, self.axe = plt.subplots()

    def unit_step(self, direct: Literal['x', 'y', 'z'], move_type: Literal['pos', 'rot'], rate: float = 1):
        action = np.zeros(6)
        if move_type == 'pos':
            action[DIRECT2INDEX[direct]] = rate * self.pos_unit
        else:
            action[DIRECT2INDEX[direct] + 3] = rate * self.rot_unit

        is_truncated, is_alignments, is_collisions = self.env._step(action, self.pr)
        print(f'''
is_truncated: {is_truncated}, is_alignments: {is_alignments}, is_collisions: {is_collisions}
step: {self.env._subenv_step}, reward: {self.reward_fn(self.env, is_alignments, is_collisions)}
''')

    def check(self):
        is_alignment, is_collision = self.env._step_env_check(self.pr)
        print(f"is_alignment: {is_alignment}, is_collision: {is_collision}")

    def save_obs(self):
        self.axe.clear()
        obs = self.env.get_obs()
        print(f"obs_mean: {np.mean(obs)}, min: {np.min(obs)}, max: {np.max(obs)}")
        self.axe.imshow(np.squeeze(obs))
        self.fig.savefig(f"tmp/plane_box/{get_file_time_str()}_obs.png")

    def save_render(self):
        self.axe.clear()
        self.axe.imshow(np.squeeze(self.env.render()))
        self.fig.savefig(f"tmp/plane_box/{get_file_time_str()}_render.png")

    def try_init_pose(self, mmdeg_pose: np.ndarray):
        self.env.anchor_move_object.set_pose(self.env.origin_pose_move_anchor_relcore, self.env.anchor_core_object)
        set_pose6_by_self(self.env.anchor_move_object, mmdeg_pose)

    def get_base_key_dict(self) -> KEY_CB_DICT_TYPE:
        return {
            'a': lambda: self.unit_step('y', 'pos', -1),
            'd': lambda: self.unit_step('y', 'pos', 1),
            'w': lambda: self.unit_step('x', 'pos', 1),
            's': lambda: self.unit_step('x', 'pos', -1),
            'q': lambda: self.unit_step('z', 'pos', -1),
            'e': lambda: self.unit_step('z', 'pos', 1),

            'i': lambda: self.unit_step('x', 'rot', 1),
            'k': lambda: self.unit_step('x', 'rot', -1),
            'l': lambda: self.unit_step('y', 'rot', 1),
            'j': lambda: self.unit_step('y', 'rot', -1),
            'o': lambda: self.unit_step('z', 'rot', 1),
            'u': lambda: self.unit_step('z', 'rot', -1),

            '1': lambda: self.save_obs(),
            '2': lambda: print(self.env.get_obs().shape),
            '3': lambda: self.save_render(),

            '4': lambda: self.check(),
            '5': lambda: self.env.test_wall.offset_tolerance(0.001),

            # 反向最远位置
            '6': lambda: self.env._set_max_init(0),
            # 正向最远位置
            '7': lambda: self.env._set_max_init(1),
            # 最佳位置
            '8': lambda: self.try_init_pose(mrad_to_mmdeg(np.array([
                    self.env.best_pose_move_anchor_relcore[0], 
                    self.env.best_pose_move_anchor_relcore[1], 
                    self.env.best_pose_move_anchor_relcore[2], 
                    0, 0, 0
                ], dtype = np.float32))),

            '`': lambda: self.env.reset()
        }

class PlaneBoxEnv(VecEnv, PrTaskVecEnvForDatasetInterface):
    def __init__(
            self,
            
            env_pr: PyRep,

            subenv_mid_name: str,
            subenv_range: Tuple[int, int], # 作为 range 参数的 start 与 stop

            subenv_make_fn: PlaneBoxSubenvInitProtocol,
            
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

            # 单位 mm, deg
            act_unit: Sequence[float] = (5, 5, 5, 1, 1, 1),

            **subenv_env_object_kwargs: Any,
        ):

        self.act_unit = np.asarray(act_unit)
        if env_init_box_pos_range is not None:
            env_init_box_pos_range_ = (
                np.asarray(env_init_box_pos_range[0], np.float32), 
                np.asarray(env_init_box_pos_range[1], np.float32)
            )
        else:
            env_init_box_pos_range_ = None
        if env_init_vis_pos_range is not None:
            env_init_vis_pos_range_ = (
                np.asarray(env_init_vis_pos_range[0], np.float32), 
                np.asarray(env_init_vis_pos_range[1], np.float32)
            )
        else:
            env_init_vis_pos_range_ = None

        self.subenv_list = [
            subenv_make_fn(
                name_suffix = subenv_mid_name + str(i),
                obs_trans = obs_trans, obs_source = obs_source, 

                env_action_noise = env_action_noise, 
                env_init_box_pos_range = env_init_box_pos_range_, 
                env_init_vis_pos_range = env_init_vis_pos_range_, 
                env_vis_persp_deg_disturb = env_vis_persp_deg_disturb,
                env_movbox_height_offset_range = env_movbox_height_offset_range,

                env_tolerance_offset = env_tolerance_offset,
                env_test_in = env_test_in, env_max_step = env_max_step,

                **subenv_env_object_kwargs
            )
        for i in range(subenv_range[0], subenv_range[1])]

        self.render_mode = "rgb_array"
        super().__init__(
            len(self.subenv_list), 
            self.subenv_list[0].get_space(),
            gym.spaces.Box(-self.act_unit, self.act_unit, [6,], dtype = np.float32)
        )

        self.env_reward_fn = env_reward_fn
        self.env_pr = env_pr
        self._last_action_list = None

    def _get_obs(self):

        obs_list = [
            subenv.get_obs() 
        for subenv in self.subenv_list]
        return np.stack(obs_list, 0)

    def _reinit(self):
        for subenv in self.subenv_list:
            subenv.reset()
        self.env_pr.step()

    def reset(self) -> VecEnvObs:
        self._reinit()
        return self._get_obs()

    def _step_take_action(self):
        assert isinstance(self._last_action_list, np.ndarray), "需要先调用 step_async"
        # 执行动作 
        truncateds = np.zeros(self.num_envs, dtype = np.bool_)
        for i, (action, subenv) in enumerate(zip(self._last_action_list, self.subenv_list)):
            subenv.tack_action(action)
            truncateds[i] = subenv.is_truncated()
        
        self.env_pr.step()

        return truncateds

    def _step_env_check(self):
        '''
        返回 is_alignments, is_collisions
        '''
        is_collisions, is_alignments = (
            np.zeros(self.num_envs, dtype = np.bool_),
            np.zeros(self.num_envs, dtype = np.bool_),
        )

        # 碰撞检测
        for i, subenv in enumerate(self.subenv_list):
            is_collisions[i] = subenv.is_collision_with_env()
            if not is_collisions[i]:
                subenv.pre_alignment_detect()

        self.env_pr.step()

        for i, subenv in enumerate(self.subenv_list):
            if not is_collisions[i]:
                is_alignments[i] = subenv.post_alignment_detect()
        
        # 还原现场
        self.env_pr.step()

        return is_alignments, is_collisions

    def _step_post_info(self, truncateds: np.ndarray, is_collisions: np.ndarray, is_alignments: np.ndarray):
        '''
        obs, rewards, dones, infos
        '''
        infos: List[Dict] = [{} for i in range(self.num_envs)]

        # 完成插入或碰撞时结束环境
        terminateds = is_alignments | is_collisions
        dones = terminateds | truncateds

        # 结算奖励与后处理

        rewards = [self.env_reward_fn(subenv, is_alignment, is_collision) for subenv, is_collision, is_alignment in zip(self.subenv_list, is_collisions, is_alignments)]
        rewards = np.array(rewards)

        for i, subenv in enumerate(self.subenv_list):
            # 将对齐与碰撞信息写入 infos
            infos[i]["PlaneBox.is_alignment"] = is_alignments[i]
            infos[i]["PlaneBox.is_collision"] = is_collisions[i]

            if dones[i]:
                infos[i]["TimeLimit.truncated"] = truncateds[i] and not terminateds[i]
                infos[i]["terminal_observation"] = subenv.get_obs()

                subenv.reset()

        # 执行完成新场景初始化
        self.env_pr.step()

        return (
            self._get_obs(),
            rewards,
            dones,
            infos
        )

    def step_async(self, actions: np.ndarray) -> None:
        self._last_action_list = actions
        self.render_img_list = None

    def step_wait(self) -> VecEnvStepReturn:
        truncateds = self._step_take_action()
        is_alignments, is_collisions = self._step_env_check()
        return self._step_post_info(
            truncateds, is_collisions, is_alignments
        )

    def get_images(self):
        return [env.render() for env in self.subenv_list]

    def close(self) -> None:
        # 在环境关闭时还原到原始状态
        for subenv in self.subenv_list:
            subenv._to_origin_state()

    ### 未实现的虚函数

    def set_attr(self, attr_name, value, indices=None):
        raise NotImplementedError()

    def get_attr(self, attr_name, indices = None) -> list[Any]:
        if indices is None:
            indices = slice(None)
            num_indices = self.num_envs
        else:
            num_indices = len(indices) # type: ignore
        attr_val = getattr(self, attr_name)

        return [attr_val] * num_indices

    def env_method(self, method_name: str, *method_args, indices=None, **method_kwargs):
        raise NotImplementedError()

    def env_is_wrapped(self, wrapper_class, indices=None):
        # For compatibility with eval and monitor helpers
        return [False]

    ### 仿真数据集接口 (PrTaskVecEnv)
    def dataset_get_task_label(self) -> List[np.ndarray]:
        '''
        获取最佳任务目标 (对于每个子环境)

        在对齐任务中为当前物体相对最佳位置的 x, y, z, a, b, c 齐次变换  
        get_rel_pose(subenv.plug.get_pose(None), subenv.plug_best_pose)
        '''

        return [
            # 转为 float32 用于训练
            np.asarray(
                mrad_to_mmdeg(
                    get_rel_pose(subenv.anchor_move_object.get_pose(subenv.anchor_core_object), subenv.best_pose_move_anchor_relcore)
                ), np.float32)
            for subenv in self.subenv_list
        ]

    def dataset_get_list_obs(self) -> List[np.ndarray]:
        '''
        获取环境观测, 不用堆叠, 而是 List 形式
        '''
        return [
            subenv.get_obs()
            for subenv in self.subenv_list
        ]

    def dataset_reinit(self) -> None:
        '''
        环境初始化
        '''
        self._reinit()

    def dataset_num_envs(self):
        return self.num_envs

class PlaneBoxEnvTest:
    def __init__(
            self,
            plane_box_env: VecEnv,
            watch_idx: int,
            model: Optional["type_aliases.PolicyPredictor"]
        ) -> None:

        assert isinstance(plane_box_env.unwrapped, PlaneBoxEnv), "不支持其他类型的环境"
        self.plane_box_env: PlaneBoxEnv = plane_box_env.unwrapped
        
        self.wrapped_env = plane_box_env

        self.watch_idx = watch_idx
        self.model = model

        self.fig, self.axe = plt.subplots()

        self.last_wrapped_obs = self.wrapped_env.reset()

    def step(self, action: np.ndarray, is_print: bool = True):
        (self.last_wrapped_obs, rewards, dones, infos) = self.wrapped_env.step(action)

        if is_print:
            
            print_info = {
                "is_done": dones[self.watch_idx],
                "is_alignment": infos[self.watch_idx]["PlaneBox.is_alignment"],
                "is_collision": infos[self.watch_idx]["PlaneBox.is_collision"],
                "cur_step": self.plane_box_env.subenv_list[self.watch_idx]._subenv_step,
                "cur_reward": rewards[self.watch_idx]
            }

            pprint(print_info, width = 20)

    def reset(self):
        self.last_wrapped_obs = self.wrapped_env.reset()

    def unit_step(self, direct: Literal['x', 'y', 'z'], move_type: Literal['pos', 'rot'], rate: float = 1):
        action = np.zeros((self.wrapped_env.num_envs, 6))
        if move_type == 'pos':
            unit_idx = DIRECT2INDEX[direct]
        else:
            unit_idx = DIRECT2INDEX[direct] + 3

        action[self.watch_idx, unit_idx] = rate * self.plane_box_env.act_unit[unit_idx]
        self.step(action, True)

    def check(self):
        is_alignments, is_collisions = self.plane_box_env._step_env_check()
        is_alignment = is_alignments[self.watch_idx]
        is_collision = is_collisions[self.watch_idx]

        print(f"is_alignment: {is_alignment}, is_collision: {is_collision}")

    def save_obs(self):
        self.axe.clear()
        obs = self.plane_box_env.subenv_list[self.watch_idx].get_obs()
        print(f"obs_mean: {np.mean(obs)}, min: {np.min(obs)}, max: {np.max(obs)}")
        self.axe.imshow(np.squeeze(obs))
        self.fig.savefig(f"tmp/plane_box/{get_file_time_str()}_obs.png")

    def save_render(self):
        self.axe.clear()
        self.axe.imshow(np.squeeze(self.plane_box_env.subenv_list[self.watch_idx].render()))
        self.fig.savefig(f"tmp/plane_box/{get_file_time_str()}_render.png")

    def take_model_action(self):
        if self.model is not None:
            model_action, _ = self.model.predict(self.last_wrapped_obs, deterministic = True) # type: ignore
            self.step(model_action, True)
        else:
            print("No Module Loaded")

    def try_init_pose(self, mmdeg_pose: np.ndarray):
        env = self.plane_box_env.subenv_list[self.watch_idx]

        env.anchor_move_object.set_pose(env.origin_pose_move_anchor_relcore, env.anchor_core_object)
        set_pose6_by_self(env.anchor_move_object, mmdeg_pose)

        self.step(np.zeros((self.wrapped_env.num_envs, 6)))

    def to_best_init(self):
        env = self.plane_box_env.subenv_list[self.watch_idx]
        env.anchor_move_object.set_pose(self.plane_box_env.subenv_list[self.watch_idx].best_pose_move_anchor_relcore, env.anchor_core_object)
        self.step(np.zeros((self.wrapped_env.num_envs, 6)))

    def try_max_state(self, direction: Literal[0, 1]):
        env = self.plane_box_env.subenv_list[self.watch_idx]
        env._set_max_init(direction)
        self.step(np.zeros((self.wrapped_env.num_envs, 6)))

    def get_base_key_dict(self) -> KEY_CB_DICT_TYPE:
        return {
            'a': lambda: self.unit_step('y', 'pos', -1),
            'd': lambda: self.unit_step('y', 'pos', 1),
            'w': lambda: self.unit_step('x', 'pos', 1),
            's': lambda: self.unit_step('x', 'pos', -1),
            'q': lambda: self.unit_step('z', 'pos', -1),
            'e': lambda: self.unit_step('z', 'pos', 1),

            'i': lambda: self.unit_step('x', 'rot', 1),
            'k': lambda: self.unit_step('x', 'rot', -1),
            'l': lambda: self.unit_step('y', 'rot', 1),
            'j': lambda: self.unit_step('y', 'rot', -1),
            'o': lambda: self.unit_step('z', 'rot', 1),
            'u': lambda: self.unit_step('z', 'rot', -1),

            '1': lambda: self.save_obs(),
            '2': lambda: self.save_render(),

            # 最佳位置
            '3': lambda: self.to_best_init(),

            # 极端位置
            '4': lambda: self.try_max_state(0),
            '5': lambda: self.try_max_state(1),

            # 初始化测试
            '6': lambda: self.wrapped_env.close(),
            '7': lambda: self.plane_box_env.close(),

            '=': lambda: self.take_model_action(),
            '`': lambda: self.reset()
        }
