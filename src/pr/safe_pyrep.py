from pyrep import PyRep
import signal

class SafePyRep:
    def __init__(self, scene_file: str = "", headless: bool = False,) -> None:
        self.pr = PyRep()
        self.pr.launch(scene_file, headless)
        self.pr.start()

    def __enter__(self):
        signal.signal(signal.SIGINT, self.sig_int_handler)
        return self.pr
    
    def __exit__(self, *args):
        self.pr.stop()
        self.pr.shutdown()

    def sig_int_handler(self, *args):
        raise KeyboardInterrupt()
