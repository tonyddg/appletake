import copy
from typing import List
import numpy as np
from dataclasses import dataclass

# x 水平向左, y 水平向下

@dataclass
class Rect:
    minX: float
    minY: float
    maxX: float
    maxY: float

    @classmethod
    def from_center_wh(cls, cx: float, cy: float, w: float, h: float):
        minX = cx - w / 2
        maxX = minX + w
        minY = cy - h / 2
        maxY = minY + h
        return Rect(minX, minY, maxX, maxY)
    
    def align_to_maxX(self, maxX):
        w = self.maxX - self.minX
        step = maxX - self.maxX

        self.maxX = maxX
        self.minX = self.maxX - w

        return step

    def align_to_maxY(self, maxY):
        h = self.maxY - self.minY
        step = maxY - self.maxY

        self.maxY = maxY
        self.minY = self.maxY - h

        return step
    
    def get_cxy(self):
        return ((self.maxX - self.minX) / 2, (self.maxY - self.minY) / 2)

def try_x_move(fix_rect_list: List[Rect], try_rect: Rect, direct: bool = True):
    '''
    仅实现向 x 正方向搜索, 没有阻挡时返回 inf
    '''
    select_minX = np.inf
    # 筛选出向 X 正方向运动能碰到的第一个矩形距离
    for test_rect in fix_rect_list:
        if try_rect.maxX <= test_rect.minX and\
           try_rect.maxY > test_rect.minY and\
           try_rect.minY < test_rect.maxY:
            
            if test_rect.minX < select_minX:
                select_minX = test_rect.minX
    return select_minX

def try_y_move(fix_rect_list: List[Rect], try_rect: Rect, direct: bool = True):
    '''
    仅实现向 y 正方向搜索, 没有阻挡时返回 inf
    '''
    select_minY = np.inf
    # 筛选出向 Y 正方向运动能碰到的第一个矩形距离
    for test_rect in fix_rect_list:
        if try_rect.maxY <= test_rect.minY and\
           try_rect.maxX > test_rect.minX and\
           try_rect.minX < test_rect.maxX:
            
            if test_rect.minY < select_minY:
                select_minY = test_rect.minY
    return select_minY

def try_max_in(fix_rect_list: List[Rect], try_rect: Rect):

    try_rect_x_first = copy.copy(try_rect)
    while True:
        move_align_x = try_x_move(fix_rect_list, try_rect_x_first)
        if move_align_x == np.inf:
            raise Exception(f"X 方向不被遮挡, fix_rect_list: {str(fix_rect_list)}, try_rect: {str(try_rect_x_first)}")
        x_step = try_rect_x_first.align_to_maxX(move_align_x)

        move_align_y = try_y_move(fix_rect_list, try_rect_x_first)
        if move_align_y == np.inf:
            raise Exception(f"Y 方向不被遮挡, fix_rect_list: {str(fix_rect_list)}, try_rect: {str(try_rect_x_first)}")
        y_step = try_rect_x_first.align_to_maxY(move_align_y)

        if x_step == 0 and y_step == 0:
            break
    
    try_rect_y_first = copy.copy(try_rect)
    while True:
        move_align_y = try_y_move(fix_rect_list, try_rect_y_first)
        if move_align_y == np.inf:
            raise Exception(f"Y 方向不被遮挡, fix_rect_list: {str(fix_rect_list)}, try_rect: {str(try_rect_y_first)}")
        y_step = try_rect_y_first.align_to_maxY(move_align_y)

        move_align_x = try_x_move(fix_rect_list, try_rect_y_first)
        if move_align_x == np.inf:
            raise Exception(f"X 方向不被遮挡, fix_rect_list: {str(fix_rect_list)}, try_rect: {str(try_rect_y_first)}")
        x_step = try_rect_y_first.align_to_maxX(move_align_x)

        if x_step == 0 and y_step == 0:
            break
    
    # 尝试从 x 开始与从 y 开始, 保留移动 L1 距离最远的作为结果
    len_x = try_rect_x_first.maxX + try_rect_x_first.maxY
    len_y = try_rect_y_first.maxX + try_rect_y_first.maxY
    if len_x > len_y:
        return try_rect_x_first
    else:
        return try_rect_y_first
        

    # # 返回最小空隙位置的 X, Y 坐标
    # return try_rect
