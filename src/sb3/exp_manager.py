import logging.config
from typing import List, Sequence, Tuple, Union, Callable, Dict, Any, Optional
from dataclasses import dataclass, field
import warnings
from pathlib import Path
from torch.utils.tensorboard.writer import SummaryWriter
import os
import yaml

from optuna.trial import Trial
import optuna

from omegaconf import ListConfig, OmegaConf
from omegaconf.dictconfig import DictConfig

from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvWrapper
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import CallbackList

from .tensorboard_callback import OptunaEvalCallback, TensorboardEpReturnCallback, TensorboardEvalCallback
from ..utility import get_file_time_str
from .eval_record import RecordFlag

import importlib

# 用于导入模块
LegalModelType = BaseAlgorithm
LegalEnvType = Union[VecEnv, VecEnvWrapper]
LegalWrapperType = VecEnvWrapper

@dataclass
class TrialArgs:
    '''
    训练参数, 理论上均不可变
    '''
    total_timesteps: int = 10000

    meta_object_path: Optional[str] = None
    meta_manual_seed: int = 0
    meta_exp_root: str = "runs"
    meta_exp_name: str = "trial"

    eval_freq: int = int(5e3)
    eval_num_episodes: int = 10
    eval_deterministic: bool = True
    # 不可变, 仅在启用 pruner 时生效
    eval_pruner_warmup_ratio: float = 0.4
 
    tb_record_episodes: int = 2
    tb_fps: int = 20
    tb_record_flag: RecordFlag = RecordFlag["NORMAL"]

@dataclass
class ObjectiveArgs:
    model: LegalModelType
    train_env: LegalEnvType
    eval_env: LegalEnvType
    trial_args: TrialArgs

    def close(self):
        self.train_env.close()
        self.eval_env.close()

def config_data_replace(data: Union[Dict, List, str], replace_dict: Dict[str, Any]):
    '''
    字典替换值, 基本格式

    key: @val
    
    替换为
    
    key: replace_dict[val]
    '''
    if isinstance(data, Dict):
        for k in list(data.keys()):
            data[k] = config_data_replace(data[k], replace_dict)
        return data
    
    # 处理列表类型（可变，直接修改）
    if isinstance(data, List):
        for i in range(len(data)):
            data[i] = config_data_replace(data[i], replace_dict)
        return data

    # 处理字符串类型
    if isinstance(data, str):
        if data[0] == "@" and data[1:] in replace_dict:
                return replace_dict[data[1:]]   
        return data

    # 其他类型保持原样
    return data

def config_data_exec(data: Union[Dict, List, str], exec_dict: Dict[str, Callable[[Dict], Any]]):
    '''
    函数调用键, 基本格式

    @key:
        type: "..."
        kwargs:
            ...

    替换为
    
    `key: exec_dict[type](kwargs)`
    '''
    if isinstance(data, Dict):
        for k in list(data.keys()):
            if isinstance(k, str) and k[0] == "@" and\
               "type" in data[k] and "kwargs" in data[k] and\
               data[k]["type"] in exec_dict:
                data.update({k[1:]: exec_dict[data[k]["type"]](data[k]["kwargs"])})
                del data[k]
            else:
                data[k] = config_data_exec(data[k], exec_dict)
        return data
    
    # 处理列表类型（可变，直接修改）
    if isinstance(data, List):
        for i in range(len(data)):
            data[i] = config_data_exec(data[i], exec_dict)
        return data

    # 其他类型保持原样
    return data

def call_type(type_str: str, kwargs: dict, replace_arg_dict: Optional[Dict[str, Any]], exec_arg_dict: Optional[Dict[str, Callable[[Any], Any]]]):
    if replace_arg_dict is not None:
        kwargs = config_data_replace(kwargs, replace_arg_dict) # type: ignore
    if exec_arg_dict is not None:
        kwargs = config_data_exec(kwargs, exec_arg_dict) # type: ignore

    lib, fun = type_str.split(":")
    if lib != "":
        lib_module = importlib.import_module(lib)
        return eval(f"lib_module.{fun}(**kwargs)")
    else:
        return eval(f"{fun}(**kwargs)")

def config_to_wrapper(wrapper_cfg: DictConfig, wrap_env: LegalEnvType, replace_arg_dict: Optional[Dict[str, Any]], exec_arg_dict: Optional[Dict[str, Callable[[Any], Any]]]):
    kwargs = OmegaConf.to_container(wrapper_cfg.kwargs, resolve = True)
    assert isinstance(kwargs, dict)
    kwargs.update(venv = wrap_env)
    wrap_env = call_type(wrapper_cfg.type, kwargs, replace_arg_dict, exec_arg_dict)
    # env = eval(f"{env_cfg.type}(**kwargs)")
    assert isinstance(wrap_env, LegalWrapperType)
    return wrap_env

def config_to_env(env_cfg: DictConfig, replace_arg_dict: Optional[Dict[str, Any]], exec_arg_dict: Optional[Dict[str, Callable[[Any], Any]]]):
    kwargs = OmegaConf.to_container(env_cfg.kwargs, resolve = True)
    assert isinstance(kwargs, dict)
    env = call_type(env_cfg.type, kwargs, replace_arg_dict, exec_arg_dict)
    # env = eval(f"{env_cfg.type}(**kwargs)")
    assert isinstance(env, LegalEnvType)

    # 自动添加环境包裹器
    wrapper_cfg_list = env_cfg.get("wrapper", None)

    if isinstance(wrapper_cfg_list, ListConfig):
        for wrapper_cfg in wrapper_cfg_list:
            env = config_to_wrapper(wrapper_cfg, env, replace_arg_dict, exec_arg_dict)

    return env

def config_to_model(model_cfg: DictConfig, train_env: LegalEnvType, replace_arg_dict: Optional[Dict[str, Any]], exec_arg_dict: Optional[Dict[str, Callable[[Any], Any]]]):
    # TODO: 自定义策略
    # kwargs = OmegaConf.to_container(model_cfg.kwargs, resolve = True)
    # model = eval(f"{model_cfg.type}(env = train_env, **kwargs)")

    kwargs = OmegaConf.to_container(model_cfg.kwargs, resolve = True)
    assert isinstance(kwargs, dict)
    kwargs.update(env = train_env)
    model = call_type(model_cfg.type, kwargs, replace_arg_dict, exec_arg_dict)

    assert isinstance(model, LegalModelType)

    return model

def config_to_trialargs(trial_cfg: DictConfig):
    trial_args = OmegaConf.structured(TrialArgs)
    return OmegaConf.merge(trial_args, trial_cfg)

def parse_exp_config(cfg: DictConfig, replace_arg_dict: Optional[Dict[str, Any]], exec_arg_dict: Optional[Dict[str, Callable[[Any], Any]]]):
    train_env = config_to_env(cfg.train_env, replace_arg_dict, exec_arg_dict)
    model = config_to_model(cfg.model, train_env, replace_arg_dict, exec_arg_dict)
    trial_args = config_to_trialargs(cfg.trial)

    if cfg.eval_env == None:
        warnings.warn("警告: 复制训练环境作为测试环境", UserWarning)
        eval_env = config_to_env(cfg.train_env, replace_arg_dict, exec_arg_dict)
    else:
        eval_env = config_to_env(cfg.eval_env, replace_arg_dict, exec_arg_dict)

    return ObjectiveArgs(model, train_env, eval_env, trial_args) # type: ignore

def load_exp_config(*cfg_file_list: Union[str, Path], merge_dict: Dict[str, Any] = {}):
    merge_cfg = OmegaConf.create(merge_dict)
    for cfg_file in cfg_file_list:
        with open(cfg_file) as f:
            cfg = OmegaConf.load(f)
        merge_cfg = OmegaConf.merge(merge_cfg, cfg)

    assert isinstance(merge_cfg, DictConfig), "配置文件格式不正确"
    return merge_cfg

def hanle_learn_LiveError():
    '''
    处理 learn 函数意外退出, 进度条没有关闭导致的错误  
    参考 <https://github.com/freqtrade/freqtrade/issues/8959#issuecomment-2194346579>
    '''
    #%% Trouver des objets (fermer les pbars buguées)
    import gc

    # On cherche tous les objets dont le nom du type contient tqdm
    tqdm_objects = [obj for obj in gc.get_objects() if 'tqdm' in type(obj).__name__]

    # On ferme ceux qu'on veut
    for tqdm_object in tqdm_objects:
        if 'tqdm_rich' in type(tqdm_object).__name__:
            tqdm_object.close()

OPTUNA_LOGGER_CONFIG_PATH = "src/sb3/exp_manger_log_config.yaml"
OPTUNA_LOGGER_NAME = "optuna_logger"

class ExpManager:
    def __init__(
            self,
            exp_conf: Union[str, Path, DictConfig],
            sample_func: Callable[[Trial, ], Dict[str, Any]],
            exp_name_suffix: Optional[str] = None,

            is_use_pruner: bool = True,

            opt_save_conf: bool = True,
            opt_save_model: bool = False,
            opt_save_data: bool = True,

            opt_record_train_return: bool = True,

            exp_replace_arg_dict: Optional[Dict[str, Any]] = None,
            exp_exec_arg_dict: Optional[Dict[str, Callable[[Any], Any]]] = None
        ) -> None:
        '''
        ## 基本配置编写  

        必要的键 
        * `train_env` 训练环境
            * `type` 环境类型, 使用 `<模块名>:<类 / 函数名>` 表示
            * `kwargs` 环境参数, 使用字典表示
            * `wrapper` 环境包裹器 (可选)
        * `model` 模型
            * `type` 模型类型
            * `kwargs` 模型参数
        * `eval_env` 测试环境 (可选)
            * `type` 环境类型
            * `kwargs` 环境参数
            * `wrapper` 环境包裹器 (可选)
        * `trial` 训练参数
            * 参见类 `TrialArgs`

        ## 搜索参数表示

        搜索参数使用 `${sample.<参数名>}` 表示, 其中参数名为函数 `sample_func` 返回字典的键值

        ## 魔法参数值表示

        函数调用键, 基本格式

        @key:
            type: "..."
            kwargs:
                ...

        替换为
        
        `key: exec_dict[type](kwargs)`

        ---

        字典替换值, 基本格式

        key: @val
        
        替换为
        
        `key: replace_dict[val]`

        '''
        self.sample_func = sample_func
        self.is_use_pruner = is_use_pruner
        if isinstance(exp_conf, DictConfig):
            self.exp_conf = exp_conf
        else:
            self.exp_conf = load_exp_config(exp_conf)

        self.exp_name = str(self.exp_conf.trial.get("meta_exp_name", "trial"))
        if exp_name_suffix is not None:
            self.exp_name = self.exp_name + exp_name_suffix

        self.exp_replace_arg_dict = exp_replace_arg_dict
        self.exp_exec_arg_dict = exp_exec_arg_dict

        self.exp_root = Path(str(self.exp_conf.trial.get("meta_exp_root", "runs")))
        if not self.exp_root.exists():
            os.mkdir(self.exp_root)

        # 创建 optuna 训练记录
        self.exp_db_path = f"sqlite:///{self.exp_root.as_posix()}/optuna.db"
        eval_times = (self.exp_conf.trial.total_timesteps // self.exp_conf.trial.eval_freq) + 1
        # print(eval_times)
        # print(self.exp_conf.trial.eval_freq)

        if self.is_use_pruner:
            # 默认使用
            pruner = optuna.pruners.MedianPruner(
                    n_warmup_steps = int(self.exp_conf.trial.eval_pruner_warmup_ratio * eval_times)
                )
        else:
            # 参考 https://github.com/DLR-RM/rl-baselines3-zoo/blob/6cac9487f17dbd00568693366615d510a320d4e7/utils/exp_manager.py#L581
            # 将 n_warmup_steps 设置为最大 steps 保证可以因其他原因 (训练异常) 使用中断
            pruner = optuna.pruners.MedianPruner(
                    n_warmup_steps = eval_times
                )
        self.study = optuna.create_study(
            storage = self.exp_db_path,
            study_name = self.exp_name,
            direction = "maximize",
            load_if_exists = True,
            pruner = pruner
        )

        # # 创建 tensorboard 训练记录
        # self.tb_writer = SummaryWriter(self.exp_runs_root.as_posix())

        # 设置 Optuna 日志
        with open(OPTUNA_LOGGER_CONFIG_PATH, encoding = 'utf-8') as f:
            # 使用 pyyaml 模块加载配置, 加载器选择 FullLoader 或 SafeLoader
            conf = yaml.load(f, yaml.FullLoader)
        conf["handlers"]["file"]["filename"] = self.exp_root.joinpath("optuna.log").as_posix()
        optuna.logging.disable_default_handler()
        logging.config.dictConfig(conf)

        self.opt_save_conf = opt_save_conf
        self.opt_save_model = opt_save_model
        self.opt_save_data = opt_save_data
        self.opt_record_train_return = opt_record_train_return

    def _objective(self, trial: Trial):
        
        sample = self.sample_func(trial)
        sample = OmegaConf.create(sample)

        # 设置随机种子
        set_random_seed(int(self.exp_conf.trial.meta_manual_seed), True)

        # 创建实验所需的环境与模型
        cur_exp_conf = self.exp_conf.copy()
        cur_exp_conf.update(sample = sample)
        OmegaConf.resolve(cur_exp_conf)

        objective_data = parse_exp_config(cur_exp_conf, self.exp_replace_arg_dict, self.exp_exec_arg_dict)

        # 创建目录并保存配置
        object_root = self.exp_root.joinpath(get_file_time_str())
        cur_exp_conf.trial.meta_object_path = object_root.as_posix()
        os.mkdir(object_root)
        with open(object_root.joinpath("config.yaml"), 'w') as f:
            OmegaConf.resolve(cur_exp_conf)
            # 最终配置不保留 sample
            del cur_exp_conf["sample"]
            OmegaConf.save(cur_exp_conf, f)
        # 在目录下创建 tb_writer
        tb_writer = SummaryWriter(object_root.as_posix())

        learn_callback = OptunaEvalCallback(
            trial, 
            optuna_is_use_pruner = self.is_use_pruner,

            tb_writer = tb_writer,

            eval_freq = objective_data.trial_args.eval_freq,
            eval_env = objective_data.eval_env,
            n_eval_episodes = objective_data.trial_args.eval_num_episodes,

            tb_record_flag = objective_data.trial_args.tb_record_flag,
            fps = objective_data.trial_args.tb_fps,

            num_record_episodes = objective_data.trial_args.tb_record_episodes,
            deterministic = objective_data.trial_args.eval_deterministic,

            save_root = object_root.as_posix(),
            save_model = self.opt_save_model,
            save_data = self.opt_save_data,
        )
        if self.opt_record_train_return:
            use_callback = CallbackList([
                learn_callback,
                TensorboardEpReturnCallback(learn_callback.tb_writer)
            ])
        else:
            use_callback = learn_callback

        try:
            # 开始训练
            objective_data.model.learn(
                objective_data.trial_args.total_timesteps, use_callback, progress_bar = True
            )
        except (AssertionError, ValueError) as e:
            print(e)
            print("学习率可能过大, 导致出现 nan 参数 https://github.com/DLR-RM/rl-baselines3-zoo/issues/156#issuecomment-910097343")
            
            hanle_learn_LiveError()
            objective_data.close()
            raise optuna.TrialPruned()

        objective_data.close()
        if learn_callback.is_prune():
            raise optuna.TrialPruned()
        else:
            return learn_callback.get_last_result()

    def optimize(self, n_trials: int = 100):

        self.study.optimize(
            self._objective, n_trials
        )

def train_model(
        exp_conf: Union[str, Path, DictConfig],
        exp_root: Union[Path, str] = "runs",

        verbose: Optional[int] = None,
        add_time_stamp: bool = True,

        opt_save_opt: bool = True,
        opt_save_data: bool = True,
        opt_record_train_return: bool = True,

        exp_replace_arg_dict: Optional[Dict[str, Any]] = None,
        exp_exec_arg_dict: Optional[Dict[str, Callable[[Any], Any]]] = None
    ) -> None:

    exp_root = Path(exp_root)
    if add_time_stamp:
        exp_root = Path(exp_root.as_posix() + '_' + get_file_time_str())
    if not exp_root.exists():
        os.mkdir(exp_root)

    if isinstance(exp_conf, DictConfig):
        exp_conf = exp_conf
    else:
        exp_conf = load_exp_config(exp_conf)

    if verbose is not None:
        exp_conf.merge_with({"model": { "kwargs": { "verbose": verbose}}})

    if opt_save_opt:
        with open(exp_root.joinpath("config.yaml"), 'w') as f:
            OmegaConf.save(exp_conf, f)

    # 设置随机种子
    set_random_seed(int(exp_conf.trial.meta_manual_seed), True)
    # 创建实验所需的环境与模型
    objective_data = parse_exp_config(exp_conf, exp_replace_arg_dict, exp_exec_arg_dict)

    learn_callback = TensorboardEvalCallback(
        tb_writer = None,

        eval_freq = objective_data.trial_args.eval_freq,
        eval_env = objective_data.eval_env,
        n_eval_episodes = objective_data.trial_args.eval_num_episodes,

        tb_record_flag = objective_data.trial_args.tb_record_flag,
        fps = objective_data.trial_args.tb_fps,

        num_record_episodes = objective_data.trial_args.tb_record_episodes,
        deterministic = objective_data.trial_args.eval_deterministic,

        save_root = exp_root,
        save_model = True,
        save_data = opt_save_data,
    )
    if opt_record_train_return:
        use_callback = CallbackList([
            learn_callback,
            TensorboardEpReturnCallback(learn_callback.tb_writer)
        ])
    else:
        use_callback = learn_callback

    objective_data.model.learn(
        objective_data.trial_args.total_timesteps, use_callback, progress_bar = True
    )
