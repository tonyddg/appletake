import multiprocessing as mp
import multiprocessing.connection

import warnings
from collections.abc import Sequence
from typing import Any, Callable, Optional, Union

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from stable_baselines3.common.vec_env import VecEnv, VecEnvWrapper

from stable_baselines3.common.vec_env.base_vec_env import (
    CloudpickleWrapper,
    VecEnv,
    VecEnvIndices,
    VecEnvObs,
    VecEnvStepReturn,
)
from stable_baselines3.common.vec_env.patch_gym import _patch_env


def _worker(
    remote: multiprocessing.connection.Connection,
    parent_remote: multiprocessing.connection.Connection,
    env_fn_wrapper: CloudpickleWrapper,
) -> None:
    # Import here to avoid a circular import
    from stable_baselines3.common.env_util import is_wrapped

    parent_remote.close()
    # env = _patch_env(env_fn_wrapper.var())
    env = env_fn_wrapper.var()
    if not (isinstance(env, VecEnv) or isinstance(env, VecEnvWrapper)):
        raise Exception("Invalid Env Type")
    
    # reset_info: Optional[dict[str, Any]] = {}
    while True:
        try:
            cmd, data = remote.recv()
            if cmd == "step":
                observation, reward, done, info = env.step(data)
                # convert to SB3 VecEnv api
                # done = terminated or truncated
                # info["TimeLimit.truncated"] = truncated and not terminated
                # if done:
                #     # save final observation where user can get it, then reset
                #     info["terminal_observation"] = observation
                #     observation, reset_info = env.reset()
                # remote.send((observation, reward, done, info, reset_info))
                remote.send((observation, reward, done, info))
            elif cmd == "reset":
                # maybe_options = {"options": data[1]} if data[1] else {}
                # observation, reset_info = env.reset(seed=data[0], **maybe_options)
                # remote.send((observation, reset_info))
                observation = env.reset()
                remote.send(observation)
            elif cmd == "render":
                remote.send(env.render())
            elif cmd == "close":
                env.close()
                remote.close()
                break
            elif cmd == "get_spaces":
                remote.send((env.observation_space, env.action_space))
            elif cmd == "num_envs":
                remote.send(env.num_envs)
            elif cmd == "get_images":
                remote.send(env.get_images())
            # elif cmd == "env_method":
            #     method = env.get_wrapper_attr(data[0])
            #     remote.send(method(*data[1], **data[2]))
            # elif cmd == "get_attr":
            #     remote.send(env.get_wrapper_attr(data))
            # elif cmd == "set_attr":
            #     remote.send(setattr(env, data[0], data[1]))  # type: ignore[func-returns-value]
            # elif cmd == "is_wrapped":
            #     remote.send(is_wrapped(env, data))
            else:
                raise NotImplementedError(f"`{cmd}` is not implemented in the worker")
        except EOFError:
            break


class StackVecEnv(VecEnv):
    """
    Creates a multiprocess vectorized wrapper for multiple environments, distributing each environment to its own
    process, allowing significant speed up when the environment is computationally complex.

    For performance reasons, if your environment is not IO bound, the number of environments should not exceed the
    number of logical cores on your CPU.

    .. warning::

        Only 'forkserver' and 'spawn' start methods are thread-safe,
        which is important when TensorFlow sessions or other non thread-safe
        libraries are used in the parent (see issue #217). However, compared to
        'fork' they incur a small start-up cost and have restrictions on
        global variables. With those methods, users must wrap the code in an
        ``if __name__ == "__main__":`` block.
        For more information, see the multiprocessing documentation.

    :param env_fns: Environments to run in subprocesses
    :param start_method: method used to start the subprocesses.
           Must be one of the methods returned by multiprocessing.get_all_start_methods().
           Defaults to 'forkserver' on available platforms, and 'spawn' otherwise.
    """

    def __init__(self, env_fns: list[Callable[[], Union[VecEnv, VecEnvWrapper]]], start_method: Optional[str] = None):
        self.waiting = False
        self.closed = False
        n_envs = len(env_fns)

        if start_method is None:
            # Fork is not a thread safe method (see issue #217)
            # but is more user friendly (does not require to wrap the code in
            # a `if __name__ == "__main__":`)
            forkserver_available = "forkserver" in mp.get_all_start_methods()
            start_method = "forkserver" if forkserver_available else "spawn"
        ctx = mp.get_context(start_method)

        self.remotes, self.work_remotes = zip(*[ctx.Pipe() for _ in range(n_envs)])
        self.processes = []
        for work_remote, remote, env_fn in zip(self.work_remotes, self.remotes, env_fns):
            args = (work_remote, remote, CloudpickleWrapper(env_fn))
            # daemon=True: if the main process crashes, we should not cause things to hang
            process = ctx.Process(target=_worker, args=args, daemon=True)  # type: ignore[attr-defined]
            process.start()
            self.processes.append(process)
            work_remote.close()

        self.remotes[0].send(("get_spaces", None))
        observation_space, action_space = self.remotes[0].recv()

        self.remote_num_envs_list = []
        self.num_envs = 0
        for remote in self.remotes:
            remote.send(("num_envs", None))
            remote_num_envs = remote.recv()
            self.num_envs += remote_num_envs
            self.remote_num_envs_list.append(remote_num_envs)

        # self.num_sub_env_list = [
        #     remote.send(("get_spaces", None)) for remote in self.remotes
        # ]

        # super().__init__(len(env_fns), observation_space, action_space)
        self.render_mode = "rgb_array"
        super().__init__(self.num_envs, observation_space, action_space)

    def step_async(self, actions: np.ndarray) -> None:
        action_pointer = 0
        for remote, remote_num_envs in zip(self.remotes, self.remote_num_envs_list):
            remote.send(("step", actions[action_pointer : action_pointer + remote_num_envs]))
            action_pointer += remote_num_envs
        self.waiting = True

    def step_wait(self) -> VecEnvStepReturn:
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        # obs, rews, dones, infos, self.reset_infos = zip(*results)  # type: ignore[assignment]
        obs, rews, dones, stack_infos = zip(*results)  # type: ignore[assignment]
        
        infos = []
        for remote_infos in stack_infos:
            for single_info in remote_infos:
                infos.append(single_info)

        return _concat_obs(obs, self.observation_space), np.concat(rews), np.concat(dones), infos  # type: ignore[return-value]

    def reset(self) -> VecEnvObs:
        for env_idx, remote in enumerate(self.remotes):
            remote.send(("reset", (self._seeds[env_idx], self._options[env_idx])))
        results = [remote.recv() for remote in self.remotes]
        # obs, self.reset_infos = zip(*results)  # type: ignore[assignment]
        obs = results  # type: ignore[assignment]
        # Seeds and options are only used once
        self._reset_seeds()
        self._reset_options()
        return _concat_obs(obs, self.observation_space)

    def close(self) -> None:
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(("close", None))
        for process in self.processes:
            process.join()
        self.closed = True

    def get_images(self) -> Sequence[Optional[np.ndarray]]:
        # if self.render_mode != "rgb_array":
        #     warnings.warn(
        #         f"The render mode is {self.render_mode}, but this method assumes it is `rgb_array` to obtain images."
        #     )
        #     return [None for _ in self.remotes]
        # for pipe in self.remotes:
        #     # gather render return from subprocesses
        #     pipe.send(("render", None))
        # outputs = [pipe.recv() for pipe in self.remotes]
        # return outputs
        for pipe in self.remotes:
            # gather render return from subprocesses
            pipe.send(("get_images", None))
        # outputs = [pipe.recv() for pipe in self.remotes]
        outputs = []
        for pipe in self.remotes:
            for single_image in pipe.recv():
                outputs.append(single_image)
        return outputs

    # def get_attr(self, attr_name: str, indices: VecEnvIndices = None) -> list[Any]:
    #     """Return attribute from vectorized environment (see base class)."""
    #     # target_remotes = self._get_target_remotes(indices)
    #     # for remote in target_remotes:
    #     #     remote.send(("get_attr", attr_name))
    #     # return [remote.recv() for remote in target_remotes]
    #     raise NotImplementedError()
    def get_attr(self, attr_name, indices = None) -> list[Any]:
        if indices is None:
            indices = slice(None)
            num_indices = self.num_envs
        else:
            num_indices = len(indices) # type: ignore
        attr_val = getattr(self, attr_name)

        return [attr_val] * num_indices

    def set_attr(self, attr_name: str, value: Any, indices: VecEnvIndices = None) -> None:
        """Set attribute inside vectorized environments (see base class)."""
        # target_remotes = self._get_target_remotes(indices)
        # for remote in target_remotes:
        #     remote.send(("set_attr", (attr_name, value)))
        # for remote in target_remotes:
        #     remote.recv()
        raise NotImplementedError()

    def env_method(self, method_name: str, *method_args, indices: VecEnvIndices = None, **method_kwargs) -> list[Any]:
        """Call instance methods of vectorized environments."""
        # target_remotes = self._get_target_remotes(indices)
        # for remote in target_remotes:
        #     remote.send(("env_method", (method_name, method_args, method_kwargs)))
        # return [remote.recv() for remote in target_remotes]
        raise NotImplementedError()

    # def env_is_wrapped(self, wrapper_class: type[gym.Wrapper], indices: VecEnvIndices = None) -> list[bool]:
    #     """Check if worker environments are wrapped with a given wrapper"""
    #     # target_remotes = self._get_target_remotes(indices)
    #     # for remote in target_remotes:
    #     #     remote.send(("is_wrapped", wrapper_class))
    #     # return [remote.recv() for remote in target_remotes]
    #     raise NotImplementedError()
    def env_is_wrapped(self, wrapper_class, indices=None):
        # For compatibility with eval and monitor helpers
        return [False]

    def _get_target_remotes(self, indices: VecEnvIndices) -> list[Any]:
        """
        Get the connection object needed to communicate with the wanted
        envs that are in subprocesses.

        :param indices: refers to indices of envs.
        :return: Connection object to communicate between processes.
        """
        # indices = self._get_indices(indices)
        # return [self.remotes[i] for i in indices]
        raise NotImplementedError()

def _concat_obs(obs_list: Union[list[VecEnvObs], tuple[VecEnvObs]], space: spaces.Space) -> VecEnvObs:
    """
    Stack observations (convert from a list of single env obs to a stack of obs),
    depending on the observation space.

    :param obs: observations.
                A list or tuple of observations, one per environment.
                Each environment observation may be a NumPy array, or a dict or tuple of NumPy arrays.
    :return: Concatenated observations.
            A NumPy array or a dict or tuple of stacked numpy arrays.
            Each NumPy array has the environment index as its first axis.
    """
    assert isinstance(obs_list, (list, tuple)), "expected list or tuple of observations per environment"
    assert len(obs_list) > 0, "need observations from at least one environment"

    if isinstance(space, spaces.Dict):
        assert isinstance(space.spaces, dict), "Dict space must have ordered subspaces"
        assert isinstance(obs_list[0], dict), "non-dict observation for environment with Dict observation space"
        return {key: np.concat([single_obs[key] for single_obs in obs_list]) for key in space.spaces.keys()}  # type: ignore[call-overload]
    elif isinstance(space, spaces.Tuple):
        assert isinstance(obs_list[0], tuple), "non-tuple observation for environment with Tuple observation space"
        obs_len = len(space.spaces)
        return tuple(np.concat([single_obs[i] for single_obs in obs_list]) for i in range(obs_len))  # type: ignore[index]
    else:
        return np.concat(obs_list)  # type: ignore[arg-type]
