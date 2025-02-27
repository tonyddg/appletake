from pyrep import PyRep
from pyrep.objects.shape import Shape
from pyrep.objects.vision_sensor import VisionSensor
from pyrep.objects.object import Object
from pyrep.objects.dummy import Dummy

from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvObs, VecEnvStepReturn

from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Tuple, Union
from dataclasses import dataclass
import gymnasium as gym
import numpy as np
from matplotlib import pyplot as plt
from abc import ABCMeta, abstractmethod

from ..utility import get_quat_diff_rad, set_pose6_by_self, sample_vec, DIRECT2INDEX
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
 
        env_tolerance_offset: float = 0,
        # Checker 长度为 50mm, 因此仅当箱子与垛盘间隙为 51~100 mm (相对原始位置偏移 1~50 mm) 时可通过检查 
        env_test_in: float = 0.05,
        env_max_step: int = 20,

        # 基于 [px, py, pz] 单位 m 的最佳位置偏移 (直接相加), (考虑 get_coppeliasim_depth_normalize, 原始间隙为 1cm)
        env_move_box_best_position_offset: np.ndarray = np.array([-0.005, -0.005, 0.025])
    ):
        '''
        底层构建方法
        '''
        self.move_box: Shape = Shape("MoveBox" + name_suffix)
        self.env_object: Union[EnvObjectsBase, Object] = env_object

        self.vis_anchor: Dummy = Dummy("VisAnchor" + name_suffix)
        self.color_camera: VisionSensor= VisionSensor("ColorCamera" + name_suffix)
        self.depth_camera: VisionSensor= VisionSensor("DepthCamera" + name_suffix)
        self.env_vis_fov_disturb = env_vis_persp_deg_disturb

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

        ### 添加扰动前保留初始状态
        # 相对于绝对坐标系 (距离目标位置 10cm)
        self.move_box_origin_pose: np.ndarray = self.move_box.get_pose(None)
        # 认为两个相机位于同一位置, 相对于盒子
        self.vis_anchor_origin_relpose: np.ndarray = self.vis_anchor.get_pose(self.move_box)
        # 相机初始视场角
        self.vis_origin_persp_deg: float = 0
        if self.obs_source == "color":
            self.vis_origin_persp_deg = self.color_camera.get_perspective_angle()
        else:
            self.vis_origin_persp_deg = self.depth_camera.get_perspective_angle()

        # 箱子的最佳位置, 用于奖励函数与神经网络构建等
        self.move_box_best_pose = np.copy(self.move_box_origin_pose)
        self.move_box_best_pose[:3] += env_move_box_best_position_offset
        # 转换为 mm 单位
        self.env_move_box_best_position_offset_mm = env_move_box_best_position_offset * 1e3

        self.env_test_in: float = env_test_in
        self.env_max_step: int = env_max_step
        self.env_action_noise: Optional[np.ndarray] = env_action_noise
        self.env_init_box_pos_range: Optional[Tuple[np.ndarray, np.ndarray]] = env_init_box_pos_range
        self.env_init_vis_pos_range: Optional[Tuple[np.ndarray, np.ndarray]] = env_init_vis_pos_range

        self._cur_move_box_position: Optional[np.ndarray] = None
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
        self._cur_move_box_position = self.move_box.get_position(None) 
        # 执行插入检测前的准备
        self.move_box.set_position([0, 0, -self.env_test_in], self.move_box)

    def post_alignment_detect(self):
        assert self._cur_move_box_position is not None, "请先执行 pre_alignment_detect"

        is_collision = self.test_wall.check_collision(self.move_box)
        is_alignment = self.move_box.check_collision(self.test_check)

        self.move_box.set_position(self._cur_move_box_position, None)
        # 碰撞到检测区域, 并且没有碰撞到插座区域
        return (not is_collision) and is_alignment

    def tack_action(self, action: np.ndarray):

        if self.env_action_noise is not None:
            action += self.env_action_noise * (2 * np.random.rand(*action.shape) - 1)
        set_pose6_by_self(self.move_box, action)

        self._subenv_step += 1
        self._last_render = None

    def is_truncated(self):
        return self._subenv_step >= self.env_max_step

    ###

    def _set_move_box_abs_pose(self, pose: np.ndarray):
        self.move_box.set_pose(self.move_box_origin_pose, None)
        set_pose6_by_self(self.move_box, pose)

    def _set_vis_rel_pose(self, pose: np.ndarray):
        self.vis_anchor.set_pose(self.vis_anchor_origin_relpose, self.move_box)
        set_pose6_by_self(self.vis_anchor, pose)

    def _set_vis_fov_offset(self, offset: float):
        operate_vis = self.color_camera if self.obs_source == "color" else self.depth_camera
        operate_vis.set_perspective_angle(self.vis_origin_persp_deg + offset)

    def _set_init_move_box_pos(
            self,
        ):
        '''
        设置箱子初始位置
        '''
        # 从均匀分布随机采样 (可改为正态分布 ?)
        if self.env_init_box_pos_range is not None:
            init_pose = sample_vec(self.env_init_box_pos_range[0], self.env_init_box_pos_range[1])
            self._set_move_box_abs_pose(init_pose)

    def _set_init_vis_pos(
            self,
        ):
        '''
        设置相机初始位置
        '''
        # 从均匀分布随机采样 (可改为正态分布 ?)
        if self.env_init_vis_pos_range is not None:
            init_pose = sample_vec(self.env_init_vis_pos_range[0], self.env_init_vis_pos_range[1])
            self._set_vis_rel_pose(init_pose)

    def _set_init_vis_fov(
            self
        ):
        '''
        设置相机初始视场角
        '''
        if self.env_vis_fov_disturb:
            self._set_vis_fov_offset(self.env_vis_fov_disturb * float(np.random.random(1)))

    def _set_max_init(self, direct: Literal[0, 1] = 0):
        '''
        测试, 用于使用最大扰动初始化环境
        '''
        if self.env_init_box_pos_range is not None:
            self._set_move_box_abs_pose(self.env_init_box_pos_range[direct])
        if self.env_init_vis_pos_range is not None:
            self._set_vis_rel_pose(self.env_init_vis_pos_range[direct])
        if self.env_vis_fov_disturb is not None:
            if direct == 0:
                self._set_vis_fov_offset(-self.env_vis_fov_disturb)
            else:
                self._set_vis_fov_offset(self.env_vis_fov_disturb)

    def reset(self):
        '''
        重新初始化
        '''

        self._subenv_step = 0
        self._last_render = None

        self._set_init_move_box_pos()
        self._set_init_vis_pos()
        self._set_init_vis_fov()

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
PlaneBoxSubenvMakeFnType = Callable[[
            str, TransformType, Literal["color", "depth"], Optional[Dict[str, Any]],
            Optional[np.ndarray], Optional[Tuple[np.ndarray, np.ndarray]], Optional[Tuple[np.ndarray, np.ndarray]],
            float, float, int
    ], PlaneBoxSubenvBase]

# reward = reward_fn(subenv, is_alignments, is_collisions)
class RewardFnABC(metaclass = ABCMeta):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def __call__(self, subenv: PlaneBoxSubenvBase, is_alignment: bool, is_collision: bool) -> float:
        raise NotImplementedError()

RewardFnType = Union[Callable[[PlaneBoxSubenvBase, bool, bool], float], RewardFnABC]

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
            max_pos_dis_mm: float = 80,
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
            cur_pose = subenv.move_box.get_pose(None)

            diff_xy = np.linalg.norm(cur_pose[:2] - subenv.move_box_best_pose[:2], 2)
            diff_rad = get_quat_diff_rad(cur_pose[3:], subenv.move_box_best_pose[3:])

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
            '8': lambda: self.env._set_move_box_abs_pose(np.array([
                    self.env.env_move_box_best_position_offset_mm[0], 
                    self.env.env_move_box_best_position_offset_mm[1], 
                    self.env.env_move_box_best_position_offset_mm[2], 
                    0, 0, 0
                ], dtype = np.float32)),

            '`': lambda: self.env.reset()
        }

class PlaneBoxEnv(VecEnv):
    def __init__(
            self,
            
            env_pr: PyRep,

            subenv_mid_name: str,
            subenv_range: Tuple[int, int], # 作为 range 参数的 start 与 stop

            subenv_make_fn: PlaneBoxSubenvMakeFnType,
            subenv_env_object_kwargs: Optional[Dict[str, Any]],

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
        ):

        act_unit_ = np.asarray(act_unit)
        if env_init_box_pos_range is not None:
            env_init_box_pos_range_ = (
                np.asarray(env_init_box_pos_range[0]), 
                np.asarray(env_init_box_pos_range[1])
            )
        else:
            env_init_box_pos_range_ = None
        if env_init_vis_pos_range is not None:
            env_init_vis_pos_range_ = (
                np.asarray(env_init_vis_pos_range[0]), 
                np.asarray(env_init_vis_pos_range[1])
            )
        else:
            env_init_vis_pos_range_ = None

        self.subenv_list = [
            subenv_make_fn(
                subenv_mid_name + str(i),
                obs_trans, obs_source, 
                subenv_env_object_kwargs,
                env_action_noise, env_init_box_pos_range_, env_init_vis_pos_range_, 
                env_tolerance_offset,
                env_test_in, env_max_step
            ) 
        for i in range(subenv_range[0], subenv_range[1])]

        self.render_mode = "rgb_array"
        super().__init__(
            len(self.subenv_list), 
            self.subenv_list[0].get_space(),
            gym.spaces.Box(-act_unit_, act_unit_, [6,], dtype = np.float32)
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
        infos: List[Dict] = [{} for i in range(self.num_envs)]

        # 完成插入或碰撞时结束环境
        terminateds = is_alignments | is_collisions
        dones = terminateds | truncateds

        # 结算奖励与后处理

        rewards = [self.env_reward_fn(subenv, is_collision, is_alignment) for subenv, is_collision, is_alignment in zip(self.subenv_list, is_collisions, is_alignments)]
        rewards = np.array(rewards)

        for i, subenv in enumerate(self.subenv_list):
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
        pass

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
