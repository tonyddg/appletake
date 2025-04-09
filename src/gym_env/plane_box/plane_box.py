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
from warnings import warn

from ...net.pr_task_dataset import PrTaskVecEnvForDatasetInterface

from ...pr.shape_size_setter import ShapeSizeSetter
from ..utility import get_rel_pose, mrad_to_mmdeg, set_pose6_by_self, DIRECT2INDEX, tuple_seq_asnumpy, rot_to_rotvec, DistSampler # , progressive_sample_float, progressive_sample_vec
from ...utility import get_file_time_str
from ...conio.key_listen import KEY_CB_DICT_TYPE
# from .reward import RewardSpare

from albumentations.core.composition import TransformType
from ..aug import test_trans_obs_space

from ...hppo.utility import seperate_action_from_numpy

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

    # init_offset_x: float = 0
    # init_offset_y: float = 0

    cur_offset_x: float = 0
    cur_offset_y: float = 0

    # def __post_init__(self):
    #     self.init_offset_x = self.test_c.get_position(self.test_a)[2]
    #     self.init_offset_y = self.test_d.get_position(self.test_b)[2]

    def set_offset(self, xoffset: float, yoffset: float):
        '''
        认为 C (x 方向), D (y 方向) 为可动的边界, 且自身坐标系的 +z 方向为增大容差的方向  
        初始无容差
        offset 单位为 m  
        需要调用 pr.step 使更改生效
        '''
        self.test_c.set_position([0, 0, xoffset], self.test_c)
        self.test_d.set_position([0, 0, yoffset], self.test_d)

        self.cur_offset_x += xoffset
        self.cur_offset_y += yoffset

    def to_origin(self):
        self.set_offset(-self.cur_offset_x, -self.cur_offset_y)

    def check_collision(self, obj: Object):
        return self.test_a.check_collision(obj) or\
                self.test_b.check_collision(obj) or\
                self.test_c.check_collision(obj) or\
                self.test_d.check_collision(obj)

class PlaneBoxSubenvBase(metaclass = ABCMeta):

    def __init__(
        self, 
        name_suffix: str,

        obs_trans: Optional[TransformType],
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
        env_movbox_size_range: Optional[Tuple[np.ndarray, np.ndarray]] = None,

        # 纸箱中心位置估计误差 (与相机误差同源, 可能需要 FrameStack 解决?)
        env_movebox_center_err: Optional[Tuple[np.ndarray, np.ndarray]] = None,
 
        # 是否使用环境复杂性递增
        # env_is_complexity_progression: bool = False,
        # env_minium_ratio: float = 1,

        # 环境随机性参数 (正态分布采样时归一化的 3 sigma 大小, 取 None 表示随机采样)
        env_random_sigma: Optional[float] = None,

        env_tolerance_offset: float = 0,
        # 相对于原始的中心位置, 向角点偏移距离 (单位), 根据不同类型的环境施加效果
        env_center_adjust: float = 0,

        env_max_step: int = 20,
        # 额外的角度检查
        env_align_deg_check: float = 1.2,

        # 是否主动进行插入决策
        # action_is_insert_decision: bool = False,

        # 基于 [px, py, pz] 单位 m 的对齐中心位置偏移 (直接相加), (考虑 get_coppeliasim_depth_normalize, 原始间隙为 1cm)
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

        self.obs_trans: Optional[TransformType] = obs_trans
        self.obs_source: Literal["color", "depth"] = obs_source

        self.test_wall = TestWall(
            Shape("TestA" + name_suffix),
            Shape("TestB" + name_suffix),
            Shape("TestC" + name_suffix),
            Shape("TestD" + name_suffix),
        )

        # 不能在创建环境时修改环境
        self.env_tolerance_offset = env_tolerance_offset
        self.env_center_adjust = env_center_adjust
        self.test_check = Shape("TestCheck" + name_suffix)

        ###

        self.env_align_deg_check: float = env_align_deg_check
        self.env_max_step: int = env_max_step
        self.env_action_noise: Optional[np.ndarray] = env_action_noise

        ### 记录位置关系的初始状态

        # 核心锚点 (移动物体与检测区域) -> None
        self.origin_pose_core_anchor_relnone = self.anchor_core_object.get_pose(None)
        self.origin_pose_env_anchor_relnone = self.anchor_env_object.get_pose(None)

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

        # # 箱子的对齐中心位置, 用于奖励函数与神经网络构建等
        # self.best_pose_move_anchor_relcore = np.copy(self.origin_pose_move_anchor_relcore)

        # # 根据容差自动计算 (要求初始位姿的坐标系与锚点坐标系平行)
        # xy_best_position_offset = (env_tolerance_offset + 0.01) / 2
        # env_move_box_best_position_offset: np.ndarray = np.array([-xy_best_position_offset, -xy_best_position_offset, 0.025])

        # self.best_pose_move_anchor_relcore[:3] += env_move_box_best_position_offset

        # 使用锚点代替对齐中心位置的具体坐标
        self.anchor_center_align_movebox_pose = Dummy.create()
        self.anchor_center_align_movebox_pose.set_parent(self.anchor_core_object)
        self.anchor_center_align_movebox_pose.set_pose(self.origin_pose_move_anchor_relcore, self.anchor_core_object)
        self.anchor_center_align_movebox_pose.set_name("InsertMoveboxPose" + name_suffix)

        # 使用锚点代替最佳对齐位置的具体坐标 (区别在于是否根据间隙调整位置)
        self.anchor_best_align_movebox_pose = Dummy.create()
        self.anchor_best_align_movebox_pose.set_parent(self.anchor_core_object)
        self.anchor_best_align_movebox_pose.set_pose(self.origin_pose_move_anchor_relcore, self.anchor_core_object)
        self.anchor_best_align_movebox_pose.set_name("BestMoveboxPose" + name_suffix)

        # 两个锚点的原始位置相同
        self.origin_pose_align_anchor_relcore = self.anchor_center_align_movebox_pose.get_pose(self.anchor_core_object)

        ###

        self.env_vis_fov_disturb = env_vis_persp_deg_disturb
        # 相机初始视场角
        self.vis_origin_persp_deg: float = 0
        if self.obs_source == "color":
            self.vis_origin_persp_deg = self.color_camera.get_perspective_angle()
        else:
            self.vis_origin_persp_deg = self.depth_camera.get_perspective_angle()

        ###

        self.env_movbox_size_range = env_movbox_size_range
        self.movebox_size_setter = ShapeSizeSetter(self.move_box)
        # self.movebox_origin_height = self.move_box.get_bounding_box()[5] - self.move_box.get_bounding_box()[4]
        # self._last_scale_movebox = None

        ###

        self.env_movebox_center_err = env_movebox_center_err
        # self._init_best_pose_move_anchor_relcore = np.copy(self.best_pose_move_anchor_relcore)

        ###

        # self.env_is_complexity_progression = env_is_complexity_progression
        # self.env_minium_ratio = env_minium_ratio
        # self.env_random_sigma = env_random_sigma
        self.dist_sampler = DistSampler(env_random_sigma)

        # 使用 0, 1 的浮点数表示训练进度 (初始状态下认为训练进度为 1)
        self.env_train_progress = 1

        ###

        self._cur_position_move_box_relcore: Optional[np.ndarray] = None
        self._subenv_step: int = 0
        # self._last_render: Optional[np.ndarray] = None

        ### 
        # 每个 episode 刷新的信息字典
        self.episode_info = {}
    ###

    def get_obs(self) -> np.ndarray:
        if self.obs_source == "color":
            image = self.color_camera.capture_rgb()
        else:
            image = self.depth_camera.capture_depth()

        if self.obs_trans is None:
            return image
        else:
            return self.obs_trans(image = image)["image"]

    def get_space(self):
        if self.obs_source == "color":
            origin_space = gym.spaces.Box(0, 1, [1080, 1920, 3])
        else:
            origin_space = gym.spaces.Box(0, 1, [720, 1280])

        if self.obs_trans is None:
            return origin_space
        else:
            return test_trans_obs_space(origin_space, self.obs_trans)

    def render(self):
        return self.color_camera.capture_rgb()
        # if self._last_render is None:
        #     self._last_render = self.color_camera.capture_rgb()

        # return self._last_render.copy()

    ###

    def is_collision_with_env(self):
        return self.env_object.check_collision(self.move_box)
    
    # def pre_alignment_detect(
    #         self
    #     ):
    #     self._cur_position_move_box_relcore = self.move_box.get_position(self.anchor_core_object) 
    #     # 执行插入检测前的准备
    #     self.move_box.set_position([0, 0, -self.env_test_in], self.move_box)

    # def post_alignment_detect(self):
    #     assert self._cur_position_move_box_relcore is not None, "请先执行 pre_alignment_detect"

    #     is_collision = self.test_wall.check_collision(self.move_box)
    #     is_alignment = self.move_box.check_collision(self.test_check)

    #     self.move_box.set_position(self._cur_position_move_box_relcore, self.anchor_core_object)
    #     # 碰撞到检测区域, 并且没有碰撞到插座区域
    #     # print(f"collision wall: {is_collision}")
    #     # print(f"alignment check: {is_alignment}")
    #     return (not is_collision) and is_alignment

    def alignment_detect(self):
        # assert self._cur_position_move_box_relcore is not None, "请先执行 pre_alignment_detect"

        is_collision = self.test_wall.check_collision(self.move_box)
        is_alignment = self.move_box.check_collision(self.test_check)

        # print(f"align is_collision {is_collision}")
        # print(f"align is_alignment {is_alignment}")

        _, diff_deg = self.get_dis_norm_to_best()
        if diff_deg > self.env_align_deg_check:
            is_alignment = False

        # self.move_box.set_position(self._cur_position_move_box_relcore, self.anchor_core_object)
        # 碰撞到检测区域, 并且没有碰撞到插座区域
        # print(f"collision wall: {is_collision}")
        # print(f"alignment check: {is_alignment}")
        return (not is_collision) and is_alignment

    def tack_action(self, action: np.ndarray):

        # if self.env_action_noise is not None:
        #     action += self.env_action_noise * (2 * np.random.rand(*action.shape) - 1)
        set_pose6_by_self(self.anchor_move_object, action)
        # self._last_render = None

    def is_truncated(self):
        self._subenv_step += 1
        return self._subenv_step >= self.env_max_step

    def apply_noisy(self):
        if self.env_action_noise is not None:
            noisy = self.sample_vec(-self.env_action_noise, self.env_action_noise)
            set_pose6_by_self(self.anchor_move_object, noisy)

    ###

    def _close_env(self):
        '''
        close 时调用
        还原原始状态
        删除初始化环境时创建的锚点
        '''
        self._to_origin_state()
        self.anchor_center_align_movebox_pose.remove()
        self.anchor_best_align_movebox_pose.remove()

    def _to_origin_state(self):
        '''
        返回原始状态
        '''

        # 核心锚点 (移动物体与检测区域) -> None
        self.anchor_core_object.set_pose(self.origin_pose_core_anchor_relnone, None)
        # 环境锚点
        self.anchor_env_object.set_pose(self.origin_pose_env_anchor_relnone, None)

        # 移动锚点 (盒子与相机) -> 核心锚点
        self.anchor_move_object.set_pose(self.origin_pose_move_anchor_relcore, self.anchor_core_object)
        # 移动盒子
        self.move_box.set_pose(self.origin_pose_move_box_relmove, self.anchor_move_object)
        # 相机
        self.vis_anchor.set_pose(self.origin_pose_vis_anchor_relmove, self.anchor_move_object)

        # 检测锚点 (检测区域) -> 核心锚点
        self.anchor_test_object.set_pose(self.origin_pose_test_anchor_relcore, self.anchor_core_object)

        # 修正对齐中心位置
        self.anchor_center_align_movebox_pose.set_pose(self.origin_pose_align_anchor_relcore, self.anchor_core_object)
        self.anchor_best_align_movebox_pose.set_pose(self.origin_pose_align_anchor_relcore, self.anchor_core_object)
        
        # self.best_pose_move_anchor_relcore = self._init_best_pose_move_anchor_relcore

        self.movebox_size_setter.to_origin_size()
        # if self._last_scale_movebox is not None:
        #     self.move_box.scale_object(1, 1, 1 / self._last_scale_movebox)
        #     self._last_scale_movebox = None

        # 修正探测区偏移
        self.test_wall.to_origin()

    def _set_vis_fov_offset(self, offset: float):
        operate_vis = self.color_camera if self.obs_source == "color" else self.depth_camera
        operate_vis.set_perspective_angle(self.vis_origin_persp_deg + offset)

    def _set_movebox_size(self, movebox_size: np.ndarray):
        
        size_diff = self.movebox_size_setter.set_size(movebox_size)
        # offset_vec = np.array([size_diff[0] / 2, size_diff[1] / 2, size_diff[2] / 2]) * 1e3

        # 保持盒子底面与检测区的接触
        set_pose6_by_self(self.vis_anchor, np.array([size_diff[0] / 2, size_diff[1] / 2, size_diff[2] / 2]) * 1e3)
        set_pose6_by_self(self.anchor_move_object, np.array([-size_diff[0] / 2, -size_diff[1] / 2, size_diff[2] / 2]) * 1e3)
        # 调整接触区域
        self.test_wall.set_offset(size_diff[0], size_diff[1])
        # 修正对齐中心位置
        set_pose6_by_self(self.anchor_center_align_movebox_pose, np.array([-size_diff[0] / 2, -size_diff[1] / 2, size_diff[2] / 2]) * 1e3)
        set_pose6_by_self(self.anchor_best_align_movebox_pose, np.array([-size_diff[0] / 2, -size_diff[1] / 2, size_diff[2] / 2]) * 1e3)

    def _set_movebox_center_offset(self, offset: np.ndarray):
        set_pose6_by_self(self.move_box, offset)

    def _set_init_move_anchor_pos(
            self,
        ):
        '''
        设置箱子初始位置 (基于对齐中心位置随机初始化)
        '''
        if self.env_init_box_pos_range is not None:
            init_pose = self.sample_vec(self.env_init_box_pos_range[0], self.env_init_box_pos_range[1])
            set_pose6_by_self(self.anchor_move_object, init_pose)

    def _set_init_vis_pos(
            self,
        ):
        '''
        设置相机初始位置
        '''
        if self.env_init_vis_pos_range is not None:
            init_pose = self.sample_vec(self.env_init_vis_pos_range[0], self.env_init_vis_pos_range[1])
            set_pose6_by_self(self.vis_anchor, init_pose)

    def _set_init_vis_fov(
            self
        ):
        '''
        设置相机初始视场角
        '''
        if self.env_vis_fov_disturb:
            self._set_vis_fov_offset(self.sample_float(-self.env_vis_fov_disturb, self.env_vis_fov_disturb))

    def _set_init_move_box_size(
            self,
        ):
        '''
        设置箱子初始高度
        '''
        # 从均匀分布随机采样
        if self.env_movbox_size_range is not None:
            init_offset = self.sample_vec(self.env_movbox_size_range[0], self.env_movbox_size_range[1])
            # print(self.env_movbox_size_range)
            # print(init_offset)
            self._set_movebox_size(init_offset)

    def _set_init_movebox_center_offset(self):
        if self.env_movebox_center_err is not None:
            init_offset = self.sample_vec(self.env_movebox_center_err[0], self.env_movebox_center_err[1])
            self._set_movebox_center_offset(init_offset)

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
        if self.env_movbox_size_range is not None:
            self._set_movebox_size(self.env_movbox_size_range[direct])
        if self.env_movebox_center_err is not None:
            self._set_movebox_center_offset(self.env_movebox_center_err[direct])

        self.test_wall.set_offset(self.env_tolerance_offset, self.env_tolerance_offset)
        # 根据偏移调整对齐中心位置
        set_pose6_by_self(self.anchor_center_align_movebox_pose, np.array([self.env_tolerance_offset, self.env_tolerance_offset, 0]) * 1e3 / 2 * -1)
        
    def reset(self):
        '''
        重新初始化
        '''

        self._subenv_step = 0
        # self._last_render = None

        self._to_origin_state()

        self._set_init_vis_pos()
        self._set_init_vis_fov()
        self._set_init_move_box_size()
        self._set_init_move_anchor_pos()
        self._set_init_movebox_center_offset()

        # 在 to_origin 初始化环境
        # 在 reset 应用更改
        # 重置容许区域偏置
        self.test_wall.set_offset(self.env_tolerance_offset, self.env_tolerance_offset)
        # 根据偏移调整对齐中心位置
        set_pose6_by_self(self.anchor_center_align_movebox_pose, np.array([self.env_tolerance_offset, self.env_tolerance_offset, 0]) * 1e3 / 2 * -1)
    
        self.episode_info = {}

    ###

    def _step_env_check(self):
        '''
        返回 is_alignments, is_collisions
        '''

        is_collision = self.is_collision_with_env()
        is_alignment = self.alignment_detect()
        # self.pre_alignment_detect()
        # pr.step()
        # is_alignment = self.post_alignment_detect()
        # pr.step()

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
        is_alignments, is_collisions = self._step_env_check()

        if is_truncated or is_alignments or is_collisions:
            self.reset()
        return is_truncated, is_alignments, is_collisions

    ###

    def sample_vec(self, min_vec: np.ndarray, max_vec: np.ndarray):
        return self.dist_sampler.sample_vec(min_vec, max_vec)

    def sample_float(self, min_side: float, max_side: float):
        return self.dist_sampler.sample_float(min_side, max_side)

    ### 实用函数

    def get_dis_norm_to_best(self, is_center: bool = False):
        '''
        获取距离最佳对齐区的距离差 (m) 与位姿差 (deg)
        '''
        # cur_pose = subenv.anchor_move_object.get_pose(subenv.anchor_core_object)
        if is_center:
            pose_diff = self.move_box.get_pose(self.anchor_center_align_movebox_pose)
        else:
            pose_diff = self.move_box.get_pose(self.anchor_best_align_movebox_pose)

        # xyz 三方向均计入惩罚
        diff_len = np.linalg.norm(pose_diff[:3], 2)
        # diff_rad = get_quat_diff_rad(cur_pose[3:], subenv.best_pose_move_anchor_relcore[3:])
        diff_deg, _ = rot_to_rotvec(pose_diff[3:])
        return diff_len, diff_deg

class RewardFnABC(metaclass = ABCMeta):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def __call__(self, subenv: PlaneBoxSubenvBase, is_alignment: bool, is_collision: bool, is_execute_align: bool, is_timeout: bool) -> float:
        raise NotImplementedError()

    def reset(self, subenv: PlaneBoxSubenvBase):
        pass

    def before_action(self, subenv: PlaneBoxSubenvBase):
        pass

RewardFnType = RewardFnABC

class PlaneBoxSubenvTest:
    def __init__(
            self,
            pr: PyRep,
            env: PlaneBoxSubenvBase,
            pos_unit: float = 10, 
            rot_unit: float = 1, 
            reward_fn: Optional[RewardFnType] = None
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
        if self.reward_fn is not None:
            print(f'''
    is_truncated: {is_truncated}, is_alignments: {is_alignments}, is_collisions: {is_collisions}
    step: {self.env._subenv_step}, reward: {self.reward_fn(self.env, is_alignments, is_collisions, False, False)}
    ''')
        else:
            print(f'''
    is_truncated: {is_truncated}, is_alignments: {is_alignments}, is_collisions: {is_collisions}
    step: {self.env._subenv_step}
    ''')

    def check(self):
        is_alignment, is_collision = self.env._step_env_check()
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

    def to_center_pose(self):
        pose_diff = self.env.anchor_center_align_movebox_pose.get_pose(self.env.move_box)
        self.env.anchor_move_object.set_pose(pose_diff, self.env.anchor_move_object)
        # 可能是精度不足导致第一次移动仍与对齐中心位置存在误差
        # pose_diff = self.env.anchor_best_movebox_pose.get_pose(self.env.move_box)
        # self.env.anchor_move_object.set_pose(pose_diff, self.env.anchor_move_object)

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
            '5': lambda: self.env._to_origin_state(),

            # 反向最远位置
            '6': lambda: self.env._set_max_init(0),
            # 正向最远位置
            '7': lambda: self.env._set_max_init(1),
            # 对齐中心位置
            '8': lambda: self.to_center_pose(),

            '`': lambda: self.env.reset()
        }

class PlaneBoxSubenvInitProtocol(Protocol):
    def __call__(
        self, 
        name_suffix: str,

        obs_trans: Optional[TransformType],
        obs_source: Literal["color", "depth"],

        env_action_noise: Optional[np.ndarray] = None,
        env_init_box_pos_range: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        env_init_vis_pos_range: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        env_vis_persp_deg_disturb: Optional[float] = None,
        env_movbox_size_range: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        # 纸箱中心位置估计误差 (与相机误差同源, 可能需要 FrameStack 解决?)
        env_movebox_center_err: Optional[Tuple[np.ndarray, np.ndarray]] = None,
 
        # 是否使用环境复杂性递增
        # env_is_complexity_progression: bool = False,
        # env_minium_ratio: float = 1,
        # 环境随机性参数 (正态分布采样时归一化的 3 sigma 大小, 取 None 表示随机采样)
        env_random_sigma: Optional[float] = None,

        env_tolerance_offset: float = 0,
        env_center_adjust: float = 0,

        env_align_deg_check: float = 1.2,
        env_max_step: int = 20,

        # **subenv_kwargs: Any
    ) -> PlaneBoxSubenvBase: 
        ...

from enum import Enum, auto
class PlaneBoxEnvActionType(Enum):
    # 被动插入, 总是检测
    PASSIVE_ALWAYS = "PASSIVE_ALWAYS"
    # 被动插入, 超时时检测
    PASSIVE_END = "PASSIVE_END"
    # 主动插入, pddpg 范式, (act, continue, done)
    DESICION_PDDPG = "DESICION_PDDPG"
    # 主动插入, hppo 范式 (continue/done, act)
    DESICION_HPPO = "DESICION_HPPO"

    def get_empty_action(self, batch_size: Optional[int] = None):
        match self:
            case PlaneBoxEnvActionType.PASSIVE_ALWAYS | PlaneBoxEnvActionType.PASSIVE_END:
                if batch_size is None:
                    return np.zeros(6)
                else:
                    return np.zeros((batch_size, 6))
            case PlaneBoxEnvActionType.DESICION_PDDPG:
                if batch_size is None:
                    res = np.zeros(8)
                    res[-2] = 1
                    return res
                else:
                    res = np.zeros((batch_size, 8))
                    res[:, -2] = 1
                    return res
            case PlaneBoxEnvActionType.DESICION_HPPO:
                if batch_size is None:
                    return np.zeros(7)
                else:
                    return np.zeros((batch_size, 7))
            case _:
                raise Exception()

    def analys_action(self, actions: np.ndarray, is_timeouts: Optional[np.ndarray] = None):
        move_actions = None
        is_need_aligns = None
        is_need_turminate = None

        match self:
            case PlaneBoxEnvActionType.PASSIVE_ALWAYS:
                move_actions = actions
                is_need_aligns = np.ones((actions.shape[0]), dtype = np.bool_)
                is_need_turminate = np.zeros((actions.shape[0]), dtype = np.bool_)
            case PlaneBoxEnvActionType.PASSIVE_END:
                move_actions = actions
                if is_timeouts is None:
                    is_need_aligns = np.zeros((actions.shape[0]), dtype = np.bool_)
                    is_need_turminate = np.zeros((actions.shape[0]), dtype = np.bool_)
                else:
                    is_need_aligns = is_timeouts
                    is_need_turminate = is_timeouts
            case PlaneBoxEnvActionType.DESICION_PDDPG:
                move_actions = actions[:, :6]
                is_need_aligns = actions[:, 6] < actions[:, 7]
                is_need_turminate = is_need_aligns
            case PlaneBoxEnvActionType.DESICION_HPPO:
                discrete_action, continue_action = seperate_action_from_numpy(actions)
                # print(f"discrete_action {discrete_action}")
                # print(f"continue_action {continue_action}")
                
                move_actions = continue_action
                # 离散动作取 1 时表示插入
                is_need_aligns = discrete_action == 1
                is_need_turminate = is_need_aligns
            case _:
                raise Exception()
        
        return move_actions, is_need_aligns, is_need_turminate

    def set_action(self, origin_actions: np.ndarray, idx: int, action: np.ndarray, is_done: bool = False):
        match self:
            case PlaneBoxEnvActionType.PASSIVE_ALWAYS | PlaneBoxEnvActionType.PASSIVE_END:
                origin_actions[idx] = action
                return origin_actions
            case PlaneBoxEnvActionType.DESICION_PDDPG:
                origin_actions[idx, :6] = action
                if is_done:
                    origin_actions[idx, 6] = 0
                    origin_actions[idx, 7] = 1
                else:
                    origin_actions[idx, 6] = 1
                    origin_actions[idx, 7] = 0
            case PlaneBoxEnvActionType.DESICION_HPPO:
                origin_actions[idx, 1:] = action
                if is_done:
                    origin_actions[idx, 0] = 1
                else:
                    origin_actions[idx, 0] = 0
            case _:
                raise Exception()
        
        return origin_actions
    
    def get_space(self, act_unit: np.ndarray):
        match self:
            case PlaneBoxEnvActionType.PASSIVE_ALWAYS | PlaneBoxEnvActionType.PASSIVE_END:
                return gym.spaces.Box(-act_unit, act_unit, act_unit.shape, dtype = np.float32)
            case PlaneBoxEnvActionType.DESICION_PDDPG:
                act_unit = np.hstack((act_unit, np.ones(2)), dtype = np.float32)
                return gym.spaces.Box(-act_unit, act_unit, act_unit.shape, dtype = np.float32)
            case PlaneBoxEnvActionType.DESICION_HPPO:
                return gym.spaces.Tuple((
                    gym.spaces.Discrete(2),
                    gym.spaces.Box(
                        -act_unit, act_unit, act_unit.shape
                    )
                ))
            case _:
                raise Exception()

class PlaneBoxEnv(VecEnv, PrTaskVecEnvForDatasetInterface):
    def __init__(
            self,
            
            env_pr: PyRep,

            subenv_mid_name: str,
            subenv_range: Tuple[int, int], # 作为 range 参数的 start 与 stop

            subenv_make_fn: PlaneBoxSubenvInitProtocol,
            
            obs_trans: Optional[TransformType],
            obs_source: Literal["color", "depth"],

            env_reward_fn: RewardFnType,

            env_tolerance_offset: float = 0,
            env_center_adjust: float = 0,

            env_align_deg_check: float = 1.2,
            env_max_step: int = 20,
            # 直接加在原始动作上, 不转换
            env_action_noise: Optional[Union[Sequence[float], np.ndarray]] = None,
            env_init_box_pos_range: Optional[Tuple[Union[Sequence[float], np.ndarray], Union[Sequence[float], np.ndarray]]] = None,
            env_init_vis_pos_range: Optional[Tuple[Union[Sequence[float], np.ndarray], Union[Sequence[float], np.ndarray]]] = None,
            env_vis_persp_deg_disturb: Optional[float] = None,
            env_movbox_size_range: Optional[Tuple[Union[Sequence[float], np.ndarray], Union[Sequence[float], np.ndarray]]] = None,
            # 纸箱中心位置估计误差 (与相机误差同源, 可能需要 FrameStack 解决?)
            env_movebox_center_err: Optional[Tuple[Union[Sequence[float], np.ndarray], Union[Sequence[float], np.ndarray]]] = None,
     
            # 是否使用环境复杂性递增
            # env_is_complexity_progression: bool = False,
            # env_minium_ratio: float = 1,
            # train_total_timestep: Optional[int] = None,
            # progress_end_timestep_ratio: Optional[float] = None,
            # 环境随机性参数 (正态分布采样时归一化的 3 sigma 大小, 取 None 表示随机采样)
            env_random_sigma: Optional[float] = None,

            # 时间长度是否为无限长
            env_is_unlimit_time: bool = True,
            env_is_terminate_when_insert: bool = False,

            # 单位 mm, deg
            act_unit: Sequence[float] = (5, 5, 5, 1, 1, 1),

            # 使用中心位置而不是最佳位置作为训练目标
            dataset_is_center_goal: bool = False,
            # 使用标准参数化动作空间
            # act_is_standard_parameter_space: bool = False,
            # act_is_passive_align: bool = True,
            act_type: Union[PlaneBoxEnvActionType, str] = PlaneBoxEnvActionType.PASSIVE_ALWAYS,

            **subenv_env_object_kwargs: Any,
        ):

        self.act_unit = np.asarray(act_unit)
        # self.act_is_passive_align = act_is_passive_align
        # self.act_is_standard_parameter_space = act_is_standard_parameter_space
        if isinstance(act_type, str):
            act_type = PlaneBoxEnvActionType(act_type)
        self.act_type = act_type

        # PASSIVE_END 将强制触发结束
        self.env_is_unlimit_time = env_is_unlimit_time or (self.act_type is PlaneBoxEnvActionType.PASSIVE_END)
        # 是否在插入后立即结束环境
        self.env_is_terminate_when_insert = env_is_terminate_when_insert

        env_init_box_pos_range_ = tuple_seq_asnumpy(env_init_box_pos_range)
        env_init_vis_pos_range_ = tuple_seq_asnumpy(env_init_vis_pos_range)
        env_movebox_center_err_ = tuple_seq_asnumpy(env_movebox_center_err)
        env_movbox_size_range_ = tuple_seq_asnumpy(env_movbox_size_range)

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

                env_action_noise = np.asarray(env_action_noise), 
                env_init_box_pos_range = env_init_box_pos_range_, 
                env_init_vis_pos_range = env_init_vis_pos_range_, 
                env_vis_persp_deg_disturb = env_vis_persp_deg_disturb,
                env_movbox_size_range = env_movbox_size_range_,
                env_movebox_center_err = env_movebox_center_err_,

                # env_is_complexity_progression = env_is_complexity_progression,
                # env_minium_ratio = env_minium_ratio,
                # 环境随机性参数 (正态分布采样时归一化的 3 sigma 大小, 取 None 表示随机采样)
                env_random_sigma = env_random_sigma,

                env_tolerance_offset = env_tolerance_offset,
                env_center_adjust = env_center_adjust,
                env_align_deg_check = env_align_deg_check, env_max_step = env_max_step,

                **subenv_env_object_kwargs
            )
        for i in range(subenv_range[0], subenv_range[1])]

        # if not act_is_passive_align:
        #     # 补充两个虚拟连续动作, (不插入 6, 插入 7)
        #     self.act_unit = np.hstack((self.act_unit, np.ones(2)), dtype = np.float32)

        self.render_mode = "rgb_array"
        super().__init__(
            len(self.subenv_list), 
            self.subenv_list[0].get_space(),
            self.act_type.get_space(self.act_unit)
        )

        # if train_total_timestep is not None:
        #     if progress_end_timestep_ratio is not None:
        #         self.progress_stop_timestep = train_total_timestep * progress_end_timestep_ratio
        #     else:
        #         self.progress_stop_timestep = train_total_timestep

        # self.cur_timestep = 0

        self.env_reward_fn = env_reward_fn
        self.env_pr = env_pr
        self._last_action_list = None

        self.dataset_is_center_goal = dataset_is_center_goal

    def _get_obs(self):

        obs_list = [
            subenv.get_obs() 
        for subenv in self.subenv_list]
        return np.stack(obs_list, 0)

    def _reinit(self):
        for subenv in self.subenv_list:
            subenv.reset()
            self.env_reward_fn.reset(subenv)

        self.env_pr.step()

        # 随机生成错误的环境时, 强制重新初始化
        try_times = 0
        while True:
            has_error_env = False
            for subenv in self.subenv_list:
                if subenv.is_collision_with_env():
                    subenv.reset()
                    has_error_env = True
            if has_error_env:
                if try_times >= 1:
                    print(f"第 {try_times} 次尝试重新初始化生成错误的环境, 强制重新初始化")
                try_times += 1
                self.env_pr.step()
            else:
                break

    def reset(self) -> VecEnvObs:
        self._reinit()
        return self._get_obs()

    def _step_take_action(self):
        assert isinstance(self._last_action_list, np.ndarray), "需要先调用 step_async"
        # 执行动作 
        # truncateds = np.zeros(self.num_envs, dtype = np.bool_)
        for i, (action, subenv) in enumerate(zip(self._last_action_list, self.subenv_list)):
            # 让奖励函数记住执行动作前的状态
            self.env_reward_fn.before_action(subenv)
            # 主动决策中，在插入时不移动
            if self._need_terminate[i] == False:
                subenv.tack_action(action)
            # truncateds[i] = subenv.is_truncated()
        
        self.env_pr.step()
        # return truncateds

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

            if not is_collisions[i] and self._need_align[i]:
                is_alignments[i] = subenv.alignment_detect()

        return is_alignments, is_collisions

    def _step_post_info(self, truncateds: np.ndarray, is_collisions: np.ndarray, is_alignments: np.ndarray):
        '''
        obs, rewards, dones, infos
        '''
        infos: List[Dict] = [{} for i in range(self.num_envs)]

        # 完成插入或碰撞时结束环境
        terminateds = is_alignments | is_collisions #| self._need_terminate
        if self.env_is_terminate_when_insert:
            terminateds = terminateds | self._need_terminate
        if not self.env_is_unlimit_time:
            terminateds = terminateds | truncateds

        dones = terminateds | truncateds

        # 结算奖励与后处理

        # if self.act_is_passive_align:
        #     rewards = [self.env_reward_fn(subenv, is_alignment, is_collision, False, truncated) for subenv, is_collision, is_alignment, truncated in zip(self.subenv_list, is_collisions, is_alignments, truncateds)]
        # else:
        rewards = [self.env_reward_fn(subenv, is_alignment, is_collision, bool(is_execute_align), truncated) for subenv, is_collision, is_alignment, is_execute_align, truncated in zip(self.subenv_list, is_collisions, is_alignments, self._need_align, truncateds)]
        rewards = np.array(rewards)

        for i, subenv in enumerate(self.subenv_list):
            # 将对齐与碰撞信息写入 infos
            infos[i]["PlaneBox.is_alignment"] = is_alignments[i]
            infos[i]["PlaneBox.is_collision"] = is_collisions[i]

            # 在奖励结算后添加噪音
            subenv.apply_noisy()

            if dones[i]:
                infos[i]["TimeLimit.truncated"] = truncateds[i] and not terminateds[i]
                infos[i]["terminal_observation"] = subenv.get_obs()

                # 记录成功率
                infos[i]["is_success"] = is_alignments[i] and not is_collisions[i]

                # 记录最终偏差
                pos_diff, rot_diff = self.subenv_list[i].get_dis_norm_to_best(True)
                infos[i]["pos_diff"] = pos_diff
                infos[i]["rot_diff"] = rot_diff

                # if self.progress_stop_timestep != None:
                #     subenv.set_train_progress(
                #         min(self.cur_timestep / self.progress_stop_timestep, 1)
                #     )
                # else:
                #     subenv.set_train_progress(
                #         1
                #     )
                #     if subenv.env_is_complexity_progression:
                #         warn(f"使用 complexity_progression 但没有给出总步长")
                subenv.reset()
                self.env_reward_fn.reset(subenv)

        # 执行完成新场景初始化
        self.env_pr.step()

        return (
            self._get_obs(),
            rewards,
            dones,
            infos
        )

    def step_async(self, actions: np.ndarray) -> None:
        # print(actions)
        # if self.act_is_passive_align:
        #     self._last_action_list = actions
        #     self._need_align = np.ones(self.num_envs, np.bool_)
        # else:
        #     if self.act_is_standard_parameter_space:
        #         discrete_action, continue_action = seperate_action_from_numpy(actions)
        #         self._last_action_list = continue_action
        #         # 离散动作取 1 时表示插入
        #         self._need_align = discrete_action == 1
        #     else:
        #         self._last_action_list = actions[:, :6]
        #         self._need_align = actions[:, 6] < actions[:, 7]
        self._subenv_truncateds = np.array([
            subenv.is_truncated() for subenv in self.subenv_list
        ], dtype = np.bool_)   
        self._last_action_list, self._need_align, self._need_terminate = self.act_type.analys_action(actions, self._subenv_truncateds)
        # print(self._need_align)
        self.render_img_list = None

    def step_wait(self) -> VecEnvStepReturn:

        # self.cur_timestep += self.num_envs

        self._step_take_action()
        is_alignments, is_collisions = self._step_env_check()
        return self._step_post_info(
            self._subenv_truncateds, is_collisions, is_alignments
        )

    def get_images(self):
        return [env.render() for env in self.subenv_list]

    def close(self) -> None:
        # 在环境关闭时还原到原始状态
        for subenv in self.subenv_list:
            # 还原原始状态
            # 删除初始化环境时创建的锚点
            subenv._close_env()

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

        在对齐任务中为当前物体相对对齐中心位置的 x, y, z, a, b, c 齐次变换  
        get_rel_pose(subenv.plug.get_pose(None), subenv.plug_best_pose)
        '''

        if self.dataset_is_center_goal:
            return [
                # 转为 float32 用于训练
                np.asarray(
                    mrad_to_mmdeg(
                        get_rel_pose(subenv.move_box.get_pose(subenv.anchor_core_object), subenv.anchor_center_align_movebox_pose.get_pose(subenv.anchor_core_object))
                    ), np.float32)
                for subenv in self.subenv_list
            ]
        else:
            return [
                # 转为 float32 用于训练
                np.asarray(
                    mrad_to_mmdeg(
                        get_rel_pose(subenv.move_box.get_pose(subenv.anchor_core_object), subenv.anchor_best_align_movebox_pose.get_pose(subenv.anchor_core_object))
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
            # plane_box_env: VecEnv,
            env_maker: Callable[[], VecEnv],
            watch_idx: int,
            model: Optional["type_aliases.PolicyPredictor"],
            obs_channel: int = 1,
            move_rate: float = 1
        ) -> None:

        plane_box_env = env_maker()
        self.env_maker = env_maker

        assert isinstance(plane_box_env.unwrapped, PlaneBoxEnv), "不支持其他类型的环境"
        self.plane_box_env: PlaneBoxEnv = plane_box_env.unwrapped
        
        self.wrapped_env = plane_box_env

        self.watch_idx = watch_idx
        self.model = model

        if obs_channel == 1:
            self.fig, self.axes = plt.subplots()
        else:
            self.fig, self.axes = plt.subplot_mosaic([list(range(obs_channel))])

        self.last_wrapped_obs = self.wrapped_env.reset()
        # if not self.plane_box_env.act_is_passive_align:
        #     if self.plane_box_env.act_is_standard_parameter_space:
        #         self.act_length = 7
        #     else:
        #         self.act_length = 8
        # else:
        #     self.act_length = 6

        self.move_rate = move_rate

    def reinit_test(self):
        self.wrapped_env.close()

        plane_box_env = self.env_maker()
        assert isinstance(plane_box_env.unwrapped, PlaneBoxEnv), "不支持其他类型的环境"
        self.plane_box_env = plane_box_env.unwrapped
        self.wrapped_env = plane_box_env
        self.last_wrapped_obs = self.wrapped_env.reset()

    def get_empty_action(self):
        # act = np.zeros((self.wrapped_env.num_envs, self.act_length))
        # if not self.plane_box_env.act_is_passive_align:
        #     if self.plane_box_env.act_is_standard_parameter_space:
        #         act[0] = 0
        #     else:
        #         act[:, 6] = 1
        return self.plane_box_env.act_type.get_empty_action(self.wrapped_env.num_envs)

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
        action = np.zeros((6))
        if move_type == 'pos':
            unit_idx = DIRECT2INDEX[direct]
        else:
            unit_idx = DIRECT2INDEX[direct] + 3

        action[unit_idx] = rate * self.plane_box_env.act_unit[unit_idx]
        # if not self.plane_box_env.act_is_passive_align:
        #     if self.plane_box_env.act_is_standard_parameter_space:
        #         action = np.hstack((np.zeros((self.plane_box_env.num_envs, 1)), action))
        #     else:    
        #         action = np.hstack((action, np.ones((self.plane_box_env.num_envs, 1)), np.zeros((self.plane_box_env.num_envs, 1))))
        cmd_actions = self.get_empty_action()
        cmd_actions = self.plane_box_env.act_type.set_action(cmd_actions, self.watch_idx, action)

        self.step(cmd_actions, True)

    def check(self):
        is_alignments, is_collisions = self.plane_box_env._step_env_check()
        is_alignment = is_alignments[self.watch_idx]
        is_collision = is_collisions[self.watch_idx]

        print(f"is_alignment: {is_alignment}, is_collision: {is_collision}")

    def save_obs(self):
        
        obs = self.plane_box_env.subenv_list[self.watch_idx].get_obs()
        print(f"obs_mean: {np.mean(obs)}, min: {np.min(obs)}, max: {np.max(obs)}")

        if isinstance(self.axes, dict):
            for i, axe in enumerate(self.axes.values()):
                axe.clear()
                axe.imshow(obs[i])
                axe.set_xticks([])
                axe.set_yticks([])
            
        else:
            self.axes.clear()
            self.axes.imshow(np.squeeze(obs))
            self.axes.set_xticks([])
            self.axes.set_yticks([])
 
        self.fig.savefig(f"tmp/plane_box/{get_file_time_str()}_obs.png")

        with open(f"tmp/plane_box/{get_file_time_str()}_obs_data.npz", 'wb') as f:
            np.savez(
                f, 
                obs = obs
            )

    def save_origin_obs(self):
        
        if self.plane_box_env.subenv_list[self.watch_idx].obs_source == "color":
            obs = self.plane_box_env.subenv_list[self.watch_idx].color_camera.capture_rgb()
        else:
            obs = self.plane_box_env.subenv_list[self.watch_idx].depth_camera.capture_depth()

        print(f"obs_mean: {np.mean(obs)}, min: {np.min(obs)}, max: {np.max(obs)}")

        if isinstance(self.axes, dict):
            for i, axe in enumerate(self.axes.values()):
                axe.clear()
                axe.imshow(obs)
                axe.set_xticks([])
                axe.set_yticks([])
                break

        else:
            self.axes.clear()
            self.axes.imshow(np.squeeze(obs))
            self.axes.set_xticks([])
            self.axes.set_yticks([])

        self.fig.savefig(f"tmp/plane_box/{get_file_time_str()}_origin_obs.png")
            
        with open(f"tmp/plane_box/{get_file_time_str()}_origin_obs_data.npz", 'wb') as f:
            np.savez(
                f, 
                obs = obs
        )

    # def save_render(self):
    #     self.axe[0].clear()
    #     self.axe[0].imshow(np.squeeze(self.plane_box_env.subenv_list[self.watch_idx].render()))
    #     self.fig.savefig(f"tmp/plane_box/{get_file_time_str()}_render.png")

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

        self.step(self.get_empty_action())

    def to_center_init(self):
        env = self.plane_box_env.subenv_list[self.watch_idx]

        pose_diff = env.anchor_center_align_movebox_pose.get_pose(env.move_box)
        env.anchor_move_object.set_pose(pose_diff, env.anchor_move_object)

        # print(env.anchor_best_movebox_pose.get_pose(env.anchor_env_object))
        # print(env.movebox_size_setter.get_cur_bbox())

        self.step(self.get_empty_action())

    def try_max_state(self, direction: Literal[0, 1]):
        env = self.plane_box_env.subenv_list[self.watch_idx]
        env._set_max_init(direction)
        self.step(self.get_empty_action())

    def execute_align(self):
        cmd_actions = self.get_empty_action()
        cmd_actions = self.plane_box_env.act_type.set_action(cmd_actions, self.watch_idx, np.zeros(6), True)
        # print(self.plane_box_env.act_type)
        self.step(cmd_actions)
        # self.plane_box_env
        # if not self.plane_box_env.act_is_passive_align:
        #     if self.plane_box_env.act_is_standard_parameter_space:
        #         act = self.get_empty_action()
        #         act[self.watch_idx, 0] = 1
        #         self.step(act)
        #     else:
        #         act = self.get_empty_action()
        #         act[self.watch_idx, 7] = 1
        #         act[self.watch_idx, 6] = 0
        #         self.step(act)
        # else:
        #     print(f"Useless because in passive Mode {self.plane_box_env.act_is_passive_align}")

    def get_base_key_dict(self) -> KEY_CB_DICT_TYPE:
        return {
            'a': lambda: self.unit_step('y', 'pos', -self.move_rate),
            'd': lambda: self.unit_step('y', 'pos', self.move_rate),
            'w': lambda: self.unit_step('x', 'pos', self.move_rate),
            's': lambda: self.unit_step('x', 'pos', -self.move_rate),
            'q': lambda: self.unit_step('z', 'pos', -self.move_rate),
            'e': lambda: self.unit_step('z', 'pos', self.move_rate),

            'i': lambda: self.unit_step('x', 'rot', self.move_rate),
            'k': lambda: self.unit_step('x', 'rot', -self.move_rate),
            'l': lambda: self.unit_step('y', 'rot', self.move_rate),
            'j': lambda: self.unit_step('y', 'rot', -self.move_rate),
            'o': lambda: self.unit_step('z', 'rot', self.move_rate),
            'u': lambda: self.unit_step('z', 'rot', -self.move_rate),

            '1': lambda: self.save_obs(),
            # '2': lambda: self.save_render(),

            '2': self.execute_align,

            # 对齐中心位置
            '3': lambda: self.to_center_init(),

            # 极端位置
            '4': lambda: self.try_max_state(0),
            '5': lambda: self.try_max_state(1),

            # 初始化测试
            '6': lambda: self.reinit_test(),

            # '8': lambda: self.plane_box_env.subenv_list[self.watch_idx].set_train_progress(0.2),
            # '9': lambda: self.plane_box_env.subenv_list[self.watch_idx].set_train_progress(0.9),

            '9': self.save_origin_obs,

            '=': lambda: self.take_model_action(),
            '`': lambda: self.reset()
        }
