from typing import Optional, Sequence, Tuple, Union
from warnings import warn
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.stats import truncnorm

from pyrep.objects.object import Object

DIRECT2INDEX = {
    'x': 0, 'y': 1, 'z': 2
}

def quat_to_rotvec(q: np.ndarray, is_deg: bool = True) -> Tuple[float, np.ndarray]:
    '''
    将 quat 转为轴角对, 返回值 (rot, vec), 单位 deg
    '''
    rotvec = R.from_quat(q).as_rotvec(is_deg)
    rot = float(np.linalg.norm(rotvec, 2))
    vec = rotvec / rot
    return (rot, vec)

def get_quat_diff_rad(q1: np.ndarray, q2: np.ndarray):
    '''
    x, y, z, w
    '''
    mat1 = R.from_quat(q1).as_matrix()
    mat2 = R.from_quat(q2).as_matrix()

    rot = np.linalg.matmul(mat1.T, mat2)
    theta = np.arccos(1 - 0.5 * np.linalg.trace(np.identity(3) - rot))
    return float(theta)

def get_rel_pose(a1: np.ndarray, a2: np.ndarray):
    '''
    * [px, py, pz, qx, qy, qz, qw] -> [px, py, pz, rx, ry, rz] (以 p1 为坐标系基底, xyz 欧拉角, 弧度制)
    * mix_mode: 将 Ta-1 Tb 变为 Ta Tb
    '''
    tw1 = np.identity(4)
    tw1[0:3, 0:3] = R.from_quat(a1[3:]).as_matrix()
    tw1[0:3, 3] = a1[:3]

    tw2 = np.identity(4)
    tw2[0:3, 0:3] = R.from_quat(a2[3:]).as_matrix()
    tw2[0:3, 3] = a2[:3]

    t_rel = np.linalg.matmul(np.linalg.inv(tw1), tw2)

    p_rel = t_rel[0:3, 3]
    r_rel = R.from_matrix(t_rel[0:3, 0:3]).as_euler("zyx", False)
    # 返回欧拉角为 z, y, x 轴的转角, 调整为 x, y, z
    r_rel = np.array([r_rel[2], r_rel[1], r_rel[0]])
    return np.concat([p_rel, r_rel], axis = 0)

def get_trans_res(a1: np.ndarray, a2: np.ndarray):
    pass

def eular_to_quat(pos_eular: np.ndarray, is_deg: bool) -> np.ndarray:
    '''
    将 x, y, z, a, b, c 规范 (欧拉角, deg) 的位置转换为 px, py, pz, qx, qy, qz, qw
    '''
    res = np.zeros(7)
    res[:3] = pos_eular[:3]
    res[3:] = R.from_euler("zyx", pos_eular[3:], is_deg).as_quat()
    return res

def mmdeg_to_mrad(pose: np.ndarray):
    # 防止原地修改导致的错误
    pose = pose.copy()
    pose[:3] *= 1e-3
    pose[3:6] = np.deg2rad(pose[3:6])
    return pose

def mrad_to_mmdeg(pose: np.ndarray):
    # 防止原地修改导致的错误
    pose = pose.copy()
    pose[:3] *= 1e3
    pose[3:6] = np.rad2deg(pose[3:6])
    return pose

def set_pose6_by_self(obj: Object, pose: np.ndarray):
    '''
    使用 [x, y, z, a, b, c] 的向量设置物体相对自身坐标系的新位置
    距离单位为 mm, 角度单位为 deg
    '''
    assert pose.shape[0] == 6 or pose.shape[0] == 3, "错误的变换表示"

    pose = mmdeg_to_mrad(pose)

    obj.set_position(pose[:3], obj)
    if pose.shape[0] == 6:
        obj.set_orientation(pose[3:], obj)

def sample_vec(min_vec: np.ndarray, max_vec: np.ndarray, ratio: float = 1, min_ratio: float = 0.1):
    if (max_vec < min_vec).any():
        warn(f"{max_vec} and {min_vec} is not strict", UserWarning)

    ratio = max(ratio, min_ratio)
    rnd_base = 2 * np.asarray(np.random.random(max_vec.shape), np.float32) - 1
    rnd_res = rnd_base * (max_vec - min_vec) * ratio / 2 + (max_vec + min_vec) / 2

    return rnd_res

def sample_float(min_side: float, max_side: float, ratio: float = 1, min_ratio: float = 0.1):
    if max_side < min_side:
        warn(f"{max_side} and {min_side} is not strict", UserWarning)

    ratio = max(ratio, min_ratio)
    rnd_base = 2 * float(np.random.random(1)) - 1
    rnd_res = rnd_base * (max_side - min_side) * ratio / 2 + (max_side + min_side) / 2

    return rnd_res

def depth_normalize(img: np.ndarray, z_min: float, z_far: float):
    '''
    标准化深度图 (正确方法还需要查文献)
    '''
    img[img > z_far] = z_far
    img[img < z_min] = z_min
    return (img - z_min) / (z_far - z_min)

def tuple_seq_asnumpy(obj: Optional[Tuple[Union[Sequence[float], np.ndarray], Union[Sequence[float], np.ndarray]]]):
    if obj is not None:
        obj = (
            np.asarray(obj[0], np.float32), 
            np.asarray(obj[1], np.float32)
        )
    else:
        obj = None
    return obj

def truncated_normal(mu: float, sigma: float, a: float, b: float):
    # 标准化截断区间
    a_norm = (a - mu) / sigma
    b_norm = (b - mu) / sigma
    
    # 创建截断正态分布对象
    dist = truncnorm(a = a_norm, b = b_norm, loc = mu, scale = sigma)
    
    # 生成随机样本
    return dist

TRUNCATED_NORMAL_DISP_LIST = [
    truncated_normal(0, 2 / 18 * (i + 1), -1, 1)
    for i in range(9)
]

def progressive_sample_base(ratio: float, size: int = 1):
    
    base = int(np.floor(10 * ratio))
    rnd = 0

    if base >= 9:
        rnd = np.random.random(size) * 2 - 1
    else:
        rnd = TRUNCATED_NORMAL_DISP_LIST[base].rvs(size)

    if size == 1:
        return float(rnd)
    else:
        return np.asarray(rnd)

def progressive_sample_vec(min_vec: np.ndarray, max_vec: np.ndarray, ratio: float = 1, min_ratio: float = 0.1):
    if (max_vec < min_vec).any():
        warn(f"{max_vec} and {min_vec} is not strict", UserWarning)

    ratio = max(ratio, min_ratio)
    rnd_base = progressive_sample_base(ratio, max_vec.shape)
    rnd_res = rnd_base * (max_vec - min_vec) / 2 + (max_vec + min_vec) / 2

    return rnd_res

def progressive_sample_float(min_side: float, max_side: float, ratio: float = 1, min_ratio: float = 0.1):
    if max_side < min_side:
        warn(f"{max_side} and {min_side} is not strict", UserWarning)

    ratio = max(ratio, min_ratio)
    rnd_base = progressive_sample_base(ratio, 1)
    rnd_res = rnd_base * (max_side - min_side) / 2 + (max_side + min_side) / 2

    return rnd_res
