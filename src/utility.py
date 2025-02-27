import time
from typing import Any, Dict, Optional

def get_file_time_str():
    return time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime(time.time()))

def dict_factory(base_dict: Dict[str, Any]):
    def fn(update_dict: Optional[Dict[str, Any]] = None):
        if update_dict is not None:
            base_dict.update(update_dict)
        return base_dict
    return fn
