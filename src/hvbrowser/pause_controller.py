import threading
from time import sleep


class PauseController:
    def __init__(self):
        self.pause_event = threading.Event()
        self.quit_event = threading.Event()
        self.listener_thread = threading.Thread(target=self.input_listener, daemon=True)
        self.listener_thread.start()

    def input_listener(self):
        while not self.quit_event.is_set():
            cmd = input().strip().lower()
            if cmd == "pause":
                print("Paused. Type 'continue' to resume or 'quit' to exit.")
                self.pause_event.set()
                while True:
                    cmd2 = input().strip().lower()
                    if cmd2 == "continue":
                        self.pause_event.clear()
                        print("Resumed.")
                        break
                    elif cmd2 == "quit":
                        self.quit_event.set()
                        print("Exiting.")
                        break

    def pauseable(self, func):
        def wrapper(*args, **kwargs):
            while not self.quit_event.is_set():
                if self.pause_event.is_set():
                    sleep(0.5)
                    continue
                return func(*args, **kwargs)

        return wrapper
