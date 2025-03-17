import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from pyrep import PyRep
from pyrep.objects.shape import Shape, PrimitiveShape

import numpy as np
from src.conio.key_listen import ListenKeyPress, ListenSingleKeyPress, press_key_to_continue

pr = PyRep()
# Launch the application with a scene file in headless mode
pr.launch('', headless=False) 
pr.start()  # Start the simulation

cube3 = Shape.create(PrimitiveShape.CUBOID, [0.1, 0.1, 0.1], position = [1, 1, 1])
cube2 = Shape.create(PrimitiveShape.CUBOID, [0.1, 0.1, 0.1], position = [0, 1, 1])
cube1 = Shape.create(PrimitiveShape.CUBOID, [0.1, 0.1, 0.1], position = [0, 0, 1])

cube1.set_dynamic(False)
cube2.set_dynamic(False)
cube3.set_dynamic(False)

origin_pose_cube1_relnone = cube1.get_position(None)
origin_pose_cube2_relcube1 = cube2.get_position(cube1)
origin_pose_cube3_relcube2 = cube3.get_position(cube2)

def to_origin():
    cube1.set_position(origin_pose_cube1_relnone, None)
    cube2.set_position(origin_pose_cube2_relcube1, cube1)
    cube3.set_position(origin_pose_cube3_relcube2, cube2)

# PyRep 中, 一个 step 内的位置设置将按相对位置关系解析, 相对于 None 的最先解析
# 因此一个 step 中 set_position 指定的相对位置关系总能得到满足

def move_forward():
    cube1.set_position(origin_pose_cube1_relnone + np.array([0.1, 0, 0]), None)
    cube2.set_position(origin_pose_cube2_relcube1 + np.array([0, 0.1, 0]), cube1)
    cube3.set_position(origin_pose_cube3_relcube2 + np.array([0, 0, 0.1]), cube2)

def move_inverse():
    cube3.set_position(origin_pose_cube3_relcube2 + np.array([0, 0, 0.1]), cube2)
    cube2.set_position(origin_pose_cube2_relcube1 + np.array([0, 0.1, 0]), cube1)
    cube1.set_position(origin_pose_cube1_relnone + np.array([0.1, 0, 0]), None)

def move_multi():
    # 对同一个物体多次调用 set_position 将叠加
    cube1.set_position(np.array([0.1, 0, 0]), cube1)
    cube1.set_position(np.array([0.1, 0, 0]), cube1)

key_cb_dict = {
    27: lambda: True,
    'a': lambda: to_origin(),
    's': lambda: move_forward(),
    'd': lambda: move_forward(),
    'f': lambda: move_multi(),
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
