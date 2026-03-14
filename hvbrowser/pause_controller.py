import threading
from time import sleep


class PauseController:
    def __init__(self) -> None:
        self.pause_event = threading.Event()
        self.quit_event = threading.Event()
        self.listener_thread = threading.Thread(target=self.input_listener, daemon=True)
        self.listener_thread.start()

    def input_listener(self) -> None:
        while not self.quit_event.is_set():
            cmd = input().strip().lower()
            if cmd == "pause":
                print("Paused. Type 'continue' to resume or 'quit' to exit.")
                self.pause_event.set()
                while True:
                    cmd2 = input().strip().lower()
                    match cmd2:
                        case "continue":
                            self.pause_event.clear()
                            print("Resumed.")
                            break
                        case "quit":
                            self.quit_event.set()
                            print("Exiting.")
                            break

    def wait_if_paused(self) -> None:
        while self.pause_event.is_set() and not self.quit_event.is_set():
            sleep(0.5)
