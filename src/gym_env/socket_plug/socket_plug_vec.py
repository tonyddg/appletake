from pyrep import PyRep
from pyrep.objects.shape import Shape
from pyrep.objects.vision_sensor import VisionSensor

from typing import Dict, List, Literal, Tuple, Optional, Any, SupportsFloat, Sequence, Union
import numpy as np
from matplotlib import pyplot as plt

import gymnasium as gym
from dataclasses import dataclass

from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvObs, VecEnvStepReturn

from ...conio.key_listen import KEY_CB_DICT_TYPE
from ...utility import get_file_time_str
from ..utility import get_quat_diff_rad, get_rel_pose, set_pose6_by_self
from ..utility import depth_normalize, mrad_to_mmdeg
from ...sb3.vec_wrap import test_trans_obs_space

from albumentations.core.composition import TransformType
from ...net.pr_task_dataset import PrTaskVecEnvForDatasetInterface

@dataclass
class SocketPlugSubenv:
    # 物体对象
    plug: Shape
    socket: Shape
    test_area: Shape
    color_camera: VisionSensor
    depth_camera: VisionSensor

    obs_process: TransformType
    action_noise: Optional[np.ndarray]

    # 子环境参数
    plug_origin_pose: np.ndarray
    plug_best_pose: np.ndarray
    # subenv_max_step: int
    subenv_step: int = 0

    @classmethod
    def make(cls, name_suffix: str, obs_process: TransformType, action_noise: Optional[np.ndarray]):
        plug = Shape("Plug" + name_suffix)
        socket = Shape("Socket" + name_suffix)
        test_area = Shape("TestArea" + name_suffix)
        color_camera = VisionSensor("ColorCamera" + name_suffix)
        depth_camera = VisionSensor("DepthCamera" + name_suffix)

        plug_origin_pose = plug.get_pose()
        plug_best_pose = np.array(plug_origin_pose)
        plug_best_pose[2] += 5e-3

        return SocketPlugSubenv(
            plug, socket, test_area, color_camera, depth_camera, obs_process, action_noise, plug.get_pose(), plug_best_pose
        )

    def get_obs(
            self, 
            obs_type: Literal["color", "depth"],
            obs_depth_range: Optional[Tuple[float, float]] = None,
        ):
        if obs_type == "color":
            obs = self.color_camera.capture_rgb()
        else:
            obs = self.depth_camera.capture_depth()
            if obs_depth_range is not None:
                obs = depth_normalize(obs, obs_depth_range[0], obs_depth_range[1])
        return self.obs_process(image = obs)["image"]

    def get_obs_space(self, obs_type: Literal["color", "depth"]):
        if obs_type == "color":
            return test_trans_obs_space(
                gym.spaces.Box(0, 1, (512, 512, 3)),
                self.obs_process)
        else:
            return test_trans_obs_space(
                gym.spaces.Box(0, 1, (512, 512)),
                self.obs_process)

    def is_collision_with_socket(self):
        return self.plug.check_collision(self.socket)

    def pre_alignment_detect(
            self,
            alignment_test_in: float
        ):
        '''
        * 返回 False 即检测失败
        * 返回 True 初步检测成功, 等待 pr.step() 完成插入测试
        '''
        cur_position = self.plug.get_position(None) 
        # 执行插入检测前的准备
        self.plug.set_position([0, 0, -alignment_test_in], self.plug)
        return cur_position

    def post_alignment_detect(self, cur_position: np.ndarray):
        is_collision = self.plug.check_collision(self.socket)
        is_alignment = self.plug.check_collision(self.test_area)
        self.plug.set_position(cur_position, None)
        # 碰撞到检测区域, 并且没有碰撞到插座区域
        return (not is_collision) and is_alignment

    def tack_action(self, action: np.ndarray):
        self.subenv_step += 1

        if self.action_noise is not None:
            action += self.action_noise * (2 * np.random.rand(*self.action_noise.shape) - 1)

        set_pose6_by_self(self.plug, action)

    def is_truncated(self, subenv_max_step: int):
        return self.subenv_step >= subenv_max_step

    ###
    def _set_abs_pose(self, pose: np.ndarray):
        self.plug.set_pose(self.plug_origin_pose, None)
        set_pose6_by_self(self.plug, pose)
            
    def _set_init_pos(
            self,
            init_pos_min: np.ndarray,
            init_pos_max: np.ndarray,
        ):
        '''
        设置初始位置
        '''
        # 从均匀分布随机采样 (可改为正态分布 ?)
        init_pose = np.random.random(6)
        init_pose = init_pose * (init_pos_max - init_pos_min) + init_pos_min

        self._set_abs_pose(init_pose)

    def reinit(
            self,
            init_pos_min: np.ndarray,
            init_pos_max: np.ndarray,
        ):
        '''
        重新初始化
        '''
        self._set_init_pos(
            init_pos_min, init_pos_max
        )
        self.subenv_step = 0

    ###
    def render(self):
        return self.color_camera.capture_rgb()

def reward_func1(subenv: SocketPlugSubenv, is_collision: bool, is_alignment: bool):
    if is_collision:
        return -1
    if is_alignment:
        return 1
    else:
        return -0.1

def reward_func2(subenv: SocketPlugSubenv, is_collision: bool, is_alignment: bool):
    if is_collision:
        return -1
    if is_alignment:
        return 1
    else:
        cur_pose = subenv.plug.get_pose(None)

        diff_xy = cur_pose[:2] - subenv.plug_origin_pose[:2]
        diff_ang = np.rad2deg(get_quat_diff_rad(cur_pose[3:], subenv.plug_origin_pose[3:]))

        reach_xy = max(10 - np.linalg.norm(diff_xy, 2) * 1e3, 0) / 10 # type: ignore
        reach_ang = max((10 - diff_ang), 0) / 10 # type: ignore
        reach_reward = (reach_xy + reach_ang) / 2

        # print(f"reach_xy: {reach_xy}, reach_ang: {reach_ang}")

        return (reach_reward - 1) / 10

class SocketPlugVecEnv(VecEnv, PrTaskVecEnvForDatasetInterface):
    def __init__(
            self,
            
            env_pr: PyRep,

            subenv_mid_name: str,
            subenv_range: Tuple[int, int], # 作为 range 参数的 start 与 stop

            env_init_pos_min: Union[Sequence[float], np.ndarray],
            env_init_pos_max: Union[Sequence[float], np.ndarray],
            env_max_steps: int,

            obs_type: Literal["color", "depth"],
            obs_process: TransformType,

            env_action_noise: Optional[Union[Sequence[float], np.ndarray]] = None,
            obs_depth_range: Optional[Tuple[float, float]] = None,
            env_alignment_test_in: float = 0.030,
            env_reward_scale: float = 1,
        ):
        pass

        subenv_list = [
            SocketPlugSubenv.make(subenv_mid_name + str(i), obs_process, np.asarray(env_action_noise, np.float32)) 
        for i in range(subenv_range[0], subenv_range[1])]

        # VecEnv 初始化
        self.obs_type: Literal["color", "depth"] = obs_type
        self.obs_depth_range = obs_depth_range
        # 以图像为观测 (使用 channel-first)

        self.render_mode = "rgb_array"
        super().__init__(
            len(subenv_list), 
            subenv_list[0].get_obs_space(obs_type),
            gym.spaces.Box(-1, 1, [6,])
        )

        self.subenv_list = subenv_list

        # 环境相关参数初始化
        self.env_pr = env_pr
        self.env_alignment_test_in = env_alignment_test_in
        self.env_step_penalty = 1.0 / env_max_steps
        self.env_max_steps = env_max_steps
        self.env_reward_scale = env_reward_scale
        if isinstance(env_init_pos_min, np.ndarray):
            self.env_init_pos_min = env_init_pos_min
        else:
            self.env_init_pos_min = np.array(env_init_pos_min, np.float32)

        if isinstance(env_init_pos_max, np.ndarray):
            self.env_init_pos_max = env_init_pos_max
        else:
            self.env_init_pos_max = np.array(env_init_pos_max, np.float32)

        self.render_img_list = None

    def _get_obs(self):

        obs_list = [
            subenv.get_obs(self.obs_type, self.obs_depth_range) 
        for subenv in self.subenv_list]
        return np.stack(obs_list, 0)

    def _reinit(self):
        for subenv in self.subenv_list:
            subenv.reinit(
                self.env_init_pos_min, self.env_init_pos_max
            )
        self.env_pr.step()

    def reset(self) -> VecEnvObs:

        self.render_img_list = None
        self._reinit()

        return self._get_obs()

    def _step_take_action(self):
        assert isinstance(self._last_action_list, np.ndarray), "需要先调用 step_async"
        # 执行动作 
        truncateds = np.zeros(self.num_envs, dtype = np.bool_)
        for i, (action, subenv) in enumerate(zip(self._last_action_list, self.subenv_list)):
            subenv.tack_action(action)
            truncateds[i] = subenv.is_truncated(self.env_max_steps)
        
        self.env_pr.step()

        return truncateds

    def _step_env_check(self):
        '''
        返回 is_alignments, is_collisions
        '''
        is_collisions, is_alignments, cur_position_list = (
            np.zeros(self.num_envs, dtype = np.bool_),
            np.zeros(self.num_envs, dtype = np.bool_),
            np.zeros((self.num_envs, 3), dtype = np.float32),
        )

        # 碰撞检测
        for i, subenv in enumerate(self.subenv_list):
            is_collisions[i] = subenv.is_collision_with_socket()
            if not is_collisions[i]:
                cur_position_list[i] = subenv.pre_alignment_detect(
                    self.env_alignment_test_in
                )

        self.env_pr.step()

        for i, subenv in enumerate(self.subenv_list):
            if not is_collisions[i]:
                is_alignments[i] = subenv.post_alignment_detect(
                    cur_position_list[i]
                )
        
        # 还原现场
        self.env_pr.step()

        return is_alignments, is_collisions

    def _step_post_info(self, truncateds: np.ndarray, is_collisions: np.ndarray, is_alignments: np.ndarray):
        infos: List[Dict] = [{} for i in range(self.num_envs)]

        # 完成插入或碰撞时结束环境
        terminateds = is_alignments | is_collisions
        dones = terminateds | truncateds

        # 结算奖励与后处理

        rewards = [reward_func2(subenv, is_collision, is_alignment) for subenv, is_collision, is_alignment in zip(self.subenv_list, is_collisions, is_alignments)]
        rewards = np.array(rewards) * self.env_reward_scale

        for i, subenv in enumerate(self.subenv_list):
            if dones[i]:
                infos[i]["TimeLimit.truncated"] = truncateds[i] and not terminateds[i]
                
                infos[i]["terminal_observation"] = subenv.get_obs(
                    self.obs_type, self.obs_depth_range
                )

                subenv.reinit(
                    self.env_init_pos_min, self.env_init_pos_max
                )

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
        if not isinstance(self.render_img_list, list):
            self.render_img_list = [env.render() for env in self.subenv_list]

        return self.render_img_list

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
    
    ### 仿真数据集接口 (PrTaskVecEnv)
    def dataset_get_task_label(self) -> List[np.ndarray]:
        '''
        获取最佳任务目标 (对于每个子环境)

        在对齐任务中为当前物体相对最佳位置的 x, y, z, a, b, c 齐次变换  
        get_rel_pose(subenv.plug.get_pose(None), subenv.plug_best_pose)
        '''
        
        return [
            # 转为 float32 用于训练
            np.asarray(mrad_to_mmdeg(get_rel_pose(subenv.plug.get_pose(None), subenv.plug_best_pose)), np.float32)
            for subenv in self.subenv_list
        ]

    def dataset_get_list_obs(self) -> List[np.ndarray]:
        '''
        获取环境观测, 不用堆叠, 而是 List 形式
        '''
        return [
            subenv.get_obs(self.obs_type, self.obs_depth_range)
            for subenv in self.subenv_list
        ]

    def dataset_reinit(self) -> None:
        '''
        环境初始化
        '''
        self._reinit()

    def dataset_num_envs(self):
        return self.num_envs

DIRECT2INDEX = {
    'x': 0, 'y': 1, 'z': 2
}

class SocketPlugTester:
    def __init__(
            self, 
            env: SocketPlugVecEnv,
            env_idx: int,
            pos_unit: float = 5, 
            rot_unit: float = 1, 
        ) -> None:
        self.env = env
        self.env_idx = env_idx
        self.pos_unit, self.rot_unit = pos_unit, rot_unit
        self.last_result: Optional[Tuple[Any, SupportsFloat, bool, dict[str, Any]]] = None

        self.fig, self.axe = plt.subplots()

    def _cache_result(self, res: VecEnvStepReturn):
        '''
        TODO: 无法处理字典观测
        '''
        observations, rewards, dones, infos = res
        self.last_result = (observations[self.env_idx], rewards[self.env_idx], dones[self.env_idx], infos[self.env_idx]) # type: ignore

    def print_info(self):
        if self.last_result is not None:
            print(f'''
step: {self.env.subenv_list[self.env_idx].subenv_step}
reward: {self.last_result[1]},
done: {self.last_result[2]},
info: {self.last_result[3]},
            ''')
        else:
            print("Info Not Available")

    def save_obs(self):
        self.axe.clear()
        self.axe.imshow(np.squeeze(self.env._get_obs()[self.env_idx]), vmin = 0, vmax = 255)
        self.fig.savefig(f"tmp/socket_plug/{get_file_time_str()}.png")

    def check(self):
        is_alignments, is_collisions = self.env._step_env_check()
        print(f"is_alignment: {is_alignments[self.env_idx]}, is_collision: {is_collisions[self.env_idx]}")

    def pos_move(self, direct: Literal['x', 'y', 'z'], rate: float = 1):
        action = np.zeros((self.env.num_envs, 6))
        action[self.env_idx, DIRECT2INDEX[direct]] = rate * self.pos_unit
        self._cache_result(self.env.step(action))

    def rot_move(self, direct: Literal['x', 'y', 'z'], rate: float = 1):
        action = np.zeros((self.env.num_envs, 6))
        action[self.env_idx, DIRECT2INDEX[direct] + 3] = rate * self.rot_unit
        self._cache_result(self.env.step(action))

    def reset(self):
        self.env.reset()

    def set_abs_pose(self, pose: np.ndarray):
        self.env.subenv_list[self.env_idx]._set_abs_pose(pose)

    def get_base_key_dict(self) -> KEY_CB_DICT_TYPE:
        return {
            'a': lambda: self.pos_move('y', -1),
            'd': lambda: self.pos_move('y', 1),
            'w': lambda: self.pos_move('x', 1),
            's': lambda: self.pos_move('x', -1),
            'q': lambda: self.pos_move('z', -1),
            'e': lambda: self.pos_move('z', 1),

            'i': lambda: self.rot_move('x', 1),
            'k': lambda: self.rot_move('x', -1),
            'l': lambda: self.rot_move('y', 1),
            'j': lambda: self.rot_move('y', -1),
            'o': lambda: self.rot_move('z', 1),
            'u': lambda: self.rot_move('z', -1),

            '1': lambda: self.check(),
            '2': lambda: self.print_info(),
            '3': lambda: self.save_obs(),
            '`': lambda: self.reset()
        }
