from typing import List, Optional, Union
import numpy as np
from pyrep.objects.shape import Shape
from pyrep.objects.dummy import Dummy
from pyrep.objects.shape import Shape, PrimitiveShape

class ShapeSizeSetter:
    def __init__(
            self,
            set_shape: Shape,
            shape_origin_size: Optional[np.ndarray] = None
        ) -> None:
        self.op_shape = set_shape

        bounding_box = set_shape.get_bounding_box()
        if shape_origin_size is None:
            self.shape_origin_size = np.array(
                [bounding_box[2 * i + 1] - bounding_box[2 * i] for i in range(3)]
            )
        else:
            self.shape_origin_size = shape_origin_size

    def get_cur_bbox(self):
        bounding_box = self.op_shape.get_bounding_box()
        return np.array(
            [bounding_box[2 * i + 1] - bounding_box[2 * i] for i in range(3)]
        )

    def set_size(self, new_bbox: Union[List[Optional[float]], np.ndarray]):
        '''
        * `new_bbox` 新 bbox 大小, 传入 None 保持原大小
        * 返回值为新旧大小的差异, 单位均为 m
        '''
        size_diff = np.zeros(3)
        new_scale = np.ones(3)

        for i, (new_bbox_size, origin_size) in enumerate(zip(new_bbox, self.get_cur_bbox())):
            if new_bbox_size is not None:
                size_diff[i] = new_bbox_size - origin_size
                new_scale[i] = new_bbox_size / origin_size

        self.op_shape.scale_object(*new_scale)
        return size_diff

    def to_origin_size(self):
        self.set_size(list(self.shape_origin_size))

def create_fixbox(box_size: np.ndarray, corner_dummy: Dummy):
    res = Shape.create(PrimitiveShape.CUBOID, [float(box_size[k]) for k in range(3)])
    res.set_dynamic(False)
    res.set_respondable(False)
    res.set_collidable(True)
    res.set_renderable(True)

    color = np.random.random(3)
    res.set_color([color[0], color[1], color[2]])
    # X, Y 向外
    res.set_position([-box_size[0] / 2, -box_size[1] / 2, box_size[2] / 2], corner_dummy)
    return res