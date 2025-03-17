import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from pyrep import PyRep
from pyrep.objects.shape import Shape
from pyrep.robots.end_effectors.suction_cup import SuctionCup
from pyrep.misc.signals import IntegerSignal

from src.pr.object_controller import ObjectController
from src.conio.key_listen import ListenKeyPress, press_key_to_continue

import numpy as np

pr = PyRep()
# Launch the application with a scene file in headless mode
pr.launch('scene/pr_example/suction_test.ttt', headless=False) 
pr.start()  # Start the simulation

cubic = Shape("Cuboid")
suction = Shape("suctionPad")
signal_isSuctionActivate = IntegerSignal("suctionActivate")

ob_cubic = ObjectController(suction)
key_cb_dict = ob_cubic.get_key_dict()
key_cb_dict.update({
    27: lambda: True,
    '1': lambda: signal_isSuctionActivate.set(1),
    '2': lambda: signal_isSuctionActivate.set(0),
})

vedio_list = []

with ListenKeyPress(key_cb_dict) as key_handler:
    while True:
        pr.step()

        is_esc_press = key_handler()
        if is_esc_press:
            break

press_key_to_continue(idle_run = pr.step)
pr.stop()  # Stop the simulation
pr.shutdown()  # Close the application
