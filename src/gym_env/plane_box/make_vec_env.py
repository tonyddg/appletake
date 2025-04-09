from ...pr.pr_env_maker import single_pr_env_maker, PyRep
from .corner import CornerEnv
from .three import ThreeEnv
from .paralle import ParalleEnv

from ...sb3.stack_vec_env import StackVecEnv

cls_dict = {
    "three": ThreeEnv,
    "corner": CornerEnv,
    "paralle": ParalleEnv
}

def make_plane_box_vec_env(env_type: str, scene_file: str, kwargs: dict, num_envs: int):
    def pr_leak_env_maker(pr: PyRep):
        return cls_dict[env_type](env_pr = pr, **kwargs)
    def env_maker():
        return lambda: single_pr_env_maker(scene_file, pr_leak_env_maker)

    env_fns = [
        env_maker() for _ in range(num_envs)
    ]
    return StackVecEnv(env_fns)
