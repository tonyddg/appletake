from pyrep import PyRep
import signal

pr = PyRep()
pr.launch('scene/pr_example/collision_example.ttt') 
pr.start()  # Start the simulation

is_running = True
def sig_handle(sig, frame):
    global is_running
    is_running = False
    print(f"Force Exit For Signal {sig}")

signal.signal(signal.SIGINT, sig_handle)
signal.signal(signal.SIGTERM, sig_handle)

while is_running:
    pr.step()

pr.stop()  # Stop the simulation
pr.shutdown()  # Close the application