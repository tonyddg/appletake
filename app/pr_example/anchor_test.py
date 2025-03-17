import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from pyrep import PyRep
from pyrep.objects.shape import Shape, PrimitiveShape
from pyrep.objects.dummy import Dummy

import numpy as np
from src.conio.key_listen import ListenKeyPress, press_key_to_continue

pr = PyRep()
# Launch the application with a scene file in headless mode
pr.launch('', headless=False) 
pr.start()  # Start the simulation


cube1 = Shape.create(PrimitiveShape.CUBOID, [0.1, 0.2, 0.3], position = [0, 0, 1])
cube1.set_dynamic(False)
cube1.set_transparency(0.5)

cube_static = Shape.create(PrimitiveShape.CUBOID, [0.1, 0.1, 0.1], position = [0, 0, 0])
cube_static.set_dynamic(False)

center_dummy = Dummy.create()
center_dummy.set_position([0, 0, 1])

corner_dummy = Dummy.create()
corner_dummy.set_position([0.05, 0.1, 0.15], cube1)
center_dummy.set_orientation([0, np.deg2rad(90), 0], None)

# center_dummy.set_parent(cube1)
# corner_dummy.set_parent(cube1)

# set_orientation 效果
# 1. 旋转 rel 的坐标系
# 2. 将被旋转物体的坐标系与 rel 的方向对齐 (位置不变)

def rotate_corner():
    corner_dummy.set_orientation([0, 0, 0], None)
    cube1.set_parent(corner_dummy)
    corner_dummy.set_orientation([0, 0, np.deg2rad(30)], None)

def rotate_center():
    center_dummy.set_orientation([0, 0, 0], None)
    cube1.set_parent(center_dummy)
    center_dummy.set_orientation([0, 0, np.deg2rad(30)], None)

def rotate_self():
    cube1.set_orientation([0, 0, 0], None)
    cube1.set_position([0, 0, 1], None)

key_cb_dict = {
    27: lambda: True,
    'a': rotate_center,
    's': rotate_corner,
    'd': rotate_self
}

with ListenKeyPress(key_cb_dict) as handle:
    while True:
        pr.step()

        if handle():
            print("强制退出")
            break

press_key_to_continue(idle_run = pr.step)
pr.stop()  # Stop the simulation
pr.shutdown()  # Close the application
