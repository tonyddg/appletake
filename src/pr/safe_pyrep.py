from pyrep import PyRep
import signal
import atexit

class SafePyRep:
    def __init__(self, scene_file: str = "", headless: bool = False,) -> None:
        self.pr = PyRep()
        self.pr.launch(scene_file, headless)
        self.pr.start()
        self.is_close = False

    def __enter__(self):
        signal.signal(signal.SIGINT, self.sig_int_handler)
        self.atexit_register = atexit.register(self.__exit__)
        return self.pr
    
    def __exit__(self, *args):
        if self.is_close:
            return

        self.pr.stop()
        self.pr.shutdown()
        atexit.unregister(self.atexit_register)
        self.is_close = True

    def sig_int_handler(self, *args):
        raise KeyboardInterrupt()
