import yaml
import time
import warnings
from pathlib import Path

from typing import Union

import logging
import logging.config

# 将模块名作为根记录器, 避免与其他日志记录器冲突
BASELOG_DOMAIN = "analysis_log"
ANALYSIS_LOGLEVEL = logging.INFO
INDENT_BASE = "  "

ANALYSIS_LOGGER = logging.getLogger(BASELOG_DOMAIN)
GLOBAL_INDENT_LEVEL = 0
IS_INIT = False

# TODO: 运行概况与总结的 logger

def initLogger(model_name: str | None = None, log_root: Union[Path, str] = "./"):

    global IS_INIT

    if not IS_INIT:

        if isinstance(log_root, str):
            log_root = Path(log_root)
        time_str = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime(time.time()))
        log_name = f"analysis_{time_str}.log"
        log_path = str(log_root.joinpath(log_name))

        with open("./src/analysis_log_config.yaml", encoding = 'utf-8') as f:
            # 使用 pyyaml 模块加载配置, 加载器选择 FullLoader 或 SafeLoader
            conf = yaml.load(f, yaml.FullLoader)

            # 在线修改属性, 使用程序运行时间作为日志名
            conf["handlers"]["analysis_hdl"]["filename"] = log_path

            # 针对特定模块的 logger
            if model_name is not None:
                conf["loggers"].update({f"{BASELOG_DOMAIN}.{model_name}" : conf["loggers"].pop("analysis_log")})

            # 在配置文件中仅设置根记录器, 其下的子记录器的内容能自动传递到根记录器
            logging.config.dictConfig(conf)
        IS_INIT = True

    else:
        warnings.warn("日志记录器已经初始化", UserWarning)

def getLogger(model_name: str | None = None):
    if model_name != None: 
        return logging.getLogger(f"{BASELOG_DOMAIN}.{model_name}")
    else:
        return logging.getLogger(BASELOG_DOMAIN)

def disableAnalysis(model_name: str | None = None):
    logger = getLogger(model_name)
    logger.setLevel(logging.CRITICAL)

class TickAnalysis:
    def __init__(self, action_name: str, model_name: str) -> None:
        self.logger = getLogger(model_name)
        self.start_tick = 0
        self.is_enable = self.logger.isEnabledFor(ANALYSIS_LOGLEVEL)
        self.action_name = action_name

    def __enter__(self):
        global GLOBAL_INDENT_LEVEL
        self.indent_level = GLOBAL_INDENT_LEVEL
        GLOBAL_INDENT_LEVEL += 1

        if self.is_enable:
            self.logger.log(ANALYSIS_LOGLEVEL, f"动作 {self.action_name} 开始执行", stacklevel = 2, extra = {"indent_str": self.indent_level * INDENT_BASE})
            self.start_tick = time.perf_counter()
        return TickLogger(self.logger, self.indent_level)
    
    def __exit__(self, exc_type, exc_value, traceback):
        if self.is_enable:
            use_time = time.perf_counter() - self.start_tick
            self.logger.log(ANALYSIS_LOGLEVEL, f"动作 {self.action_name} 执行结束耗时: {use_time:.3f}s", stacklevel = 2, extra = {"indent_str": self.indent_level * INDENT_BASE})
        
        global GLOBAL_INDENT_LEVEL
        GLOBAL_INDENT_LEVEL -= 1
        return False

class TickLogger:
    def __init__(self, logger: logging.Logger, indent_level: int) -> None:
        self.logger = logger
        self.indent_level = indent_level + 1
    
    def log(self, level: int, msg: str):
        self.logger.log(level, msg, stacklevel = 2, extra = {"indent_str": self.indent_level * INDENT_BASE})