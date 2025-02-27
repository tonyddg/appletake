from typing import Optional, Literal
from pyrep.objects.object import Object
from ..conio.key_listen import KEY_CB_DICT_TYPE

DIRECT2INDEX = {
    'x': 0, 'y': 1, 'z': 2
}

class ObjectController:
    def __init__(
            self, 
            obj: Object, 
            pos_unit: float = 0.005, 
            rot_unit: float = 0.01, 
            rel_obj: Optional[Object] = None,
        ) -> None:
        self.obj = obj
        self.pos_unit, self.rot_unit = pos_unit, rot_unit
        self.rel_obj = rel_obj

    def pos_move(self, direct: Literal['x', 'y', 'z'], rate: float = 1):
        pos = self.obj.get_position(self.rel_obj)
        pos[DIRECT2INDEX[direct]] += rate * self.pos_unit
        self.obj.set_position(pos, self.rel_obj)

    def rot_move(self, direct: Literal['x', 'y', 'z'], rate: float = 1):
        rot = self.obj.get_orientation(self.rel_obj)
        rot[DIRECT2INDEX[direct]] += rate * self.rot_unit
        self.obj.set_orientation(rot, self.rel_obj)

    def get_key_dict(self) -> KEY_CB_DICT_TYPE:
        return {
            'a': lambda: self.pos_move('y', -1),
            'd': lambda: self.pos_move('y', 1),
            'w': lambda: self.pos_move('x', 1),
            's': lambda: self.pos_move('x', -1),
            'q': lambda: self.pos_move('z', -1),
            'e': lambda: self.pos_move('z', 1),

            'i': lambda: self.rot_move('x', 1),
            'k': lambda: self.rot_move('x', -1),
            'l': lambda: self.rot_move('y', 1),
            'j': lambda: self.rot_move('y', -1),
            'o': lambda: self.rot_move('z', 1),
            'u': lambda: self.rot_move('z', -1),
        }
