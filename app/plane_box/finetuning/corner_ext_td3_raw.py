import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

from src.sb3.exp_manager import ExpManager, Trial, load_exp_config
from src.pr.safe_pyrep import SafePyRep
from arg_dict import exp_replace_arg_dict, exp_exec_arg_dict, make_backbone
from sample_param import sample_param_ext_td3

# TODO: 图像加噪音, 记录最终测试结果, 结果不可能是 1, 找 bug
# TODO: 保存最好的 n 个结果
# TODO: 随机策略测试效果
# TODO: 开启节约内存选项 optimize_memory_usage
# TODO: 课程学习
# 在终端中训练

if __name__ == "__main__":
    with SafePyRep("scene/plane_box/corner_vec4_test2.ttt", True) as pr:

        em = ExpManager(
            load_exp_config(
                "app/plane_box/conf/corner_normal_ext_env.yaml",
                "app/plane_box/finetuning/ext_td3_raw.yaml",
                merge_dict = {
                    "trial": {"meta_exp_root": "runs/plane_box/corner_finetuning/ext_td3"}
                }
            ),
            sample_param_ext_td3,
            exp_name_suffix = None,
            exp_replace_arg_dict = exp_replace_arg_dict({
                "pr": pr,
            }),
            exp_exec_arg_dict = exp_exec_arg_dict()
        )
        em.optimize(50)
