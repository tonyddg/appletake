from typing import Callable, Union
from ..sb3.exp_manager import config_to_env, load_exp_config
from pyrep import PyRep
import signal
import atexit
from stable_baselines3.common.vec_env import VecEnv, VecEnvWrapper

def single_pr_env_maker(scene_file: str, partial_make_fn: Callable[[PyRep], Union[VecEnv, VecEnvWrapper]]):

    pr = PyRep()
    pr.launch(scene_file, True)
    pr.start()
    is_close = False
    atexit_register = None

    def exit_pr(*args):
        nonlocal is_close

        if is_close:
            return

        pr.stop()
        pr.shutdown()
        is_close = True
        if atexit_register is not None:
            atexit.unregister(atexit_register)

    atexit_register = atexit.register(exit_pr)
    signal.signal(signal.SIGTERM, exit_pr)
    signal.signal(signal.SIGINT, exit_pr)

    return partial_make_fn(pr)
