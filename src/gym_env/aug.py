from functools import partial
from typing import Union
import cv2
import numpy as np
import gymnasium as gym
import albumentations as A
from albumentations.core.composition import TransformType

# Pytorch: CxHxW (float), SB3: CxHxW (uint8)
# Albumentations, Opencv: HxWxC

def test_trans_obs_space(
    origin_observation_space: gym.spaces.Box,
    trans: TransformType
):
    test_input = origin_observation_space.sample()
    test_res = np.array(trans(image = test_input)["image"])
    if test_res.dtype == np.float32:
        obs_space = gym.spaces.Box(0, 1, test_res.shape, np.float32)
    elif test_res.dtype == np.uint8:
        obs_space = gym.spaces.Box(0, 255, test_res.shape, np.uint8)
    else:
        raise Exception("Unsupport Type")
    return obs_space

def get_crop_resize(
    x_min: int = 128, 
    y_min: int = 128, 
    x_max: int = 384, 
    y_max: int = 384,
    resize_height: int = 224,
    resize_width: int = 224,
):
    return A.Sequential([
        A.Crop(x_min, y_min, x_max, y_max),
        A.Resize(resize_height, resize_width, cv2.INTER_AREA)
    ], p = 1)

def get_depth_aug(noise_p: float, noise_scale: float = 1):
    return A.Sequential([

        # 模拟深度估计误差
        A.MotionBlur(
            int(3 * noise_scale), p = noise_p
        ),
        A.GaussNoise(
            (0.02 * noise_scale, 0.05 * noise_scale),
            noise_scale_factor = 1,
            p = noise_p
        ),

        # 模拟无法填充的空洞
        A.CoarseDropout(
            num_holes_range = (1, int(5 * noise_scale)),
            hole_height_range = (0.05 / noise_scale, 0.2 * noise_scale),
            hole_width_range = (0.05 / noise_scale, 0.2 * noise_scale),
            p = noise_p
        ),

        # 传感器噪声
        A.SaltAndPepper((0.0005 * noise_scale, 0.001 * noise_scale), p = noise_p),

        # 模拟小空洞填充
        A.Morphological((5, 5), p = 1),
        A.MedianBlur(3, p = noise_p),
    ], p = 1)

def _hwc2chw(img: np.ndarray, to_sb3: bool = False, **param):
    if len(img.shape) == 3:
        img = np.transpose(img, (2, 0, 1))
    elif len(img.shape) == 2:
        img = img[np.newaxis, ...]
    else:
        raise Exception(f"只能处理图像, 无法处理 {img.shape}")
    
    if to_sb3 and img.dtype != np.uint8:
        img = np.asarray(img * 255, np.uint8)

    return img

def get_hwc2chw(to_sb3: bool = False):
    '''
    将 openCV 图片 (HxWxC, HxW) 转为 sb3 或 torch 格式 (CxHxW, uint8, float), 同时保持为 Numpy 数组
    '''
    fn = partial(_hwc2chw, to_sb3 = to_sb3)
    return A.Lambda(
        image = fn
    )

# def get_depth_standard():
#     return A.Sequential([
#         # 标准化
#         A.Normalize((0.538), (0.287), 1),
#         # 转为 Torch 格式
#         A.Lambda(image = _hwc2chw)
#     ], p = 1)

# def _gray_to_sb3(img: np.ndarray, **param):
#     if img.dtype != np.uint8:
#         img = np.asarray(img * 255, np.uint8)
#     if len(img.shape) == 2:
#         img = img[np.newaxis, :, :]
#     return img

# def get_gray_to_sb3():
#     '''
#     将灰度图 (float, HxW) 转为 SB3 格式 (uint8, 1xHxW)
#     '''
#     return A.Lambda(
#         image = _gray_to_sb3
#     )

# def _depth_norm()
    
def _depth_norm2real(img, z_min: float, z_far: float, **params):
    '''
    将标准化深度还原为真实深度
    '''
    return img * (z_far - z_min) + z_min

def _depth_thresh_normalize(img: np.ndarray, z_min: float, z_far: float, **params):
    '''
    深度图阈值标准化 (超出范围使用阈值代替)
    '''
    img[img > z_far] = z_far
    img[img < z_min] = z_min
    return (img - z_min) / (z_far - z_min)

def get_depth_thresh_normalize(z_min: float, z_far: float):
    fn = partial(_depth_thresh_normalize, z_min = z_min, z_far = z_far)
    return A.Sequential([
        A.Lambda(
            image = fn
        )
    ], p = 1)

def get_coppeliasim_depth_normalize(vis_z_min: float = 1e-4, vis_z_far: float = 5e-1, new_z_min: float = 2.5e-1, new_z_far: float = 5e-1):
    '''
    将 Coppeliasim 已经标准化的深度图重映射到新的范围 (防止 Coppeliasim 中 z_min 过大导致的穿透问题)
    '''
    fn1 = partial(_depth_norm2real, z_min = vis_z_min, z_far = vis_z_far)
    fn2 = partial(_depth_thresh_normalize, z_min = new_z_min, z_far = new_z_far)

    return A.Sequential([
        A.Lambda(
            image = fn1
        ),
        A.Lambda(
            image = fn2
        )
    ], p = 1)
