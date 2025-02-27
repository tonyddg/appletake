from warnings import warn
import numpy as np
from scipy.spatial.transform import Rotation as R

from pyrep.objects.object import Object

DIRECT2INDEX = {
    'x': 0, 'y': 1, 'z': 2
}

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
    [px, py, pz, qx, qy, qz, qw] -> [px, py, pz, rx, ry, rz] (以 p1 为坐标系基底, xyz 欧拉角, 弧度制)
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
    pose = mmdeg_to_mrad(pose)
    obj.set_position(pose[:3], obj)
    obj.set_orientation(pose[3:6], obj)

def sample_vec(min_vec: np.ndarray, max_vec: np.ndarray):
    if (max_vec < min_vec).any():
        warn(f"{max_vec} and {min_vec} is not strict", UserWarning)
    res = np.random.random(max_vec.shape)
    return res * (max_vec - min_vec) + min_vec

def depth_normalize(img: np.ndarray, z_min: float, z_far: float):
    '''
    标准化深度图 (正确方法还需要查文献)
    '''
    img[img > z_far] = z_far
    img[img < z_min] = z_min
    return (img - z_min) / (z_far - z_min)
