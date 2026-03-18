import threading
import tkinter as tk
from time import sleep


class ControlPanel:
    def __init__(self) -> None:
        self._toggles: dict[str, tk.BooleanVar] = {}
        self._skill_vars: dict[str, tk.BooleanVar] = {}
        self.pause_event = threading.Event()
        self.quit_event = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._ready = threading.Event()
        self._thread.start()
        self._ready.wait()

    def _run(self) -> None:
        self._root = tk.Tk()
        self._root.title("Battle Control Panel")
        self._root.resizable(False, False)
        self._root.attributes("-topmost", True)

        btn_frame = tk.Frame(self._root)
        btn_frame.pack(padx=10, pady=5)

        self._pause_btn = tk.Button(btn_frame, text="Pause", command=self._toggle_pause)
        self._pause_btn.pack(side="left", padx=5)

        tk.Button(btn_frame, text="Quit", command=self._quit).pack(side="left", padx=5)

        self._skill_container = tk.Frame(self._root)
        self._skill_container.pack(padx=10, pady=5, fill="x")

        self._ready.set()
        self._root.mainloop()

    def _toggle_pause(self) -> None:
        if self.pause_event.is_set():
            self.pause_event.clear()
            self._pause_btn.config(text="Pause")
        else:
            self.pause_event.set()
            self._pause_btn.config(text="Resume")

    def _quit(self) -> None:
        self.quit_event.set()
        self.pause_event.clear()

    def register_toggle(self, name: str, default: bool = False) -> None:
        """Register a toggle and add a checkbox to the panel."""

        def _add() -> None:
            var = tk.BooleanVar(value=default)
            self._toggles[name] = var
            cb = tk.Checkbutton(self._root, text=name, variable=var)
            cb.pack(anchor="w", padx=10, pady=2)

        self._root.after(0, _add)

    def get_toggle(self, name: str) -> bool:
        var = self._toggles.get(name)
        if var is None:
            return False
        return var.get()

    def set_skills(
        self,
        skill_groups: dict[str, list[str]],
        forbidden: list[str],
    ) -> None:
        """Set skill groups with forbidden skills checked."""

        def _add() -> None:
            for widget in self._skill_container.winfo_children():
                widget.destroy()
            self._skill_vars.clear()
            for group_name, skills in skill_groups.items():
                frame = tk.LabelFrame(self._skill_container, text=group_name)
                frame.pack(padx=5, pady=3, fill="x")
                for skill in skills:
                    var = tk.BooleanVar(value=skill in forbidden)
                    self._skill_vars[skill] = var
                    cb = tk.Checkbutton(frame, text=skill, variable=var)
                    cb.pack(anchor="w", padx=5, pady=1)

        self._root.after(0, _add)

    def get_forbidden_skills(self) -> list[str]:
        return [name for name, var in self._skill_vars.items() if var.get()]

    def wait_if_paused(self) -> None:
        while self.pause_event.is_set() and not self.quit_event.is_set():
            sleep(0.5)

    def destroy(self) -> None:
        self._root.after(0, self._root.destroy)
