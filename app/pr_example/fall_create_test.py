import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from pyrep import PyRep
from pyrep.robots.arms.ur3 import UR3
from pyrep.objects.shape import Shape, PrimitiveShape
from pyrep.objects.vision_sensor import VisionSensor
from pyrep.objects.camera import Camera

import numpy as np
from src.conio.key_listen import ListenSingleKeyPress, press_key_to_continue

pr = PyRep()
# Launch the application with a scene file in headless mode
pr.launch('scene/pr_example/fall_create_test.ttt', headless=False) 
pr.start()  # Start the simulation

falling_cube = Shape.create(PrimitiveShape.CUBOID, [0.1, 0.1, 0.1], position = [0, 0, 1])
stand_cube = Shape("Cuboid")

is_collision = False

with ListenSingleKeyPress(27) as is_esc_press:
    while not is_collision:
        # 需要在计算模块中开启 Enable all collision detections
        is_collision = falling_cube.check_collision(stand_cube)
        cube_distance = falling_cube.check_distance(stand_cube)
        print(cube_distance)
        pr.step()

        if is_esc_press:
            print("强制退出")
            break

press_key_to_continue(idle_run = pr.step)
pr.stop()  # Stop the simulation
pr.shutdown()  # Close the application
