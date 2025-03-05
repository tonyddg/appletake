# PyRep (不使用 ~/.bashrc, 防止污染系统环境变量, 运行 Python 可使用同目录下的 py.sh)
export COPPELIASIM_ROOT=~/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$COPPELIASIM_ROOT
export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT

# 启动虚拟环境
source .venv/bin/activate