import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from pyrep import PyRep
from pyrep.robots.arms.ur3 import UR3
from pyrep.objects.shape import Shape
from pyrep.objects.vision_sensor import VisionSensor

from src.pr.object_controller import ObjectController
from src.conio.key_listen import ListenKeyPress, press_key_to_continue

from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
import numpy as np

pr = PyRep()
# Launch the application with a scene file in headless mode
pr.launch('scene/pr_example/two_cube.ttt', headless=False) 
pr.start()  # Start the simulation

cubic_blue = Shape("blue")
cubic_red = Shape("red")
camera = VisionSensor("kinect_rgb")

def get_distance():
    return cubic_blue.check_distance(cubic_red)

ob_cubic = ObjectController(cubic_blue)
key_cb_dict = ob_cubic.get_key_dict()
key_cb_dict.update({
    27: lambda: True,
    '1': lambda: print(get_distance())
})

vedio_list = []

with ListenKeyPress(key_cb_dict) as key_handler:
    while True:
        pr.step()
        vedio_list.append((camera.capture_rgb() * 255).astype(np.uint8))
        
        is_esc_press = key_handler()
        if is_esc_press:
            break

press_key_to_continue(idle_run = pr.step)
pr.stop()  # Stop the simulation
pr.shutdown()  # Close the application

clip = ImageSequenceClip(sequence = vedio_list, fps = 24)
clip.write_videofile("vedio.mp4", logger = None)