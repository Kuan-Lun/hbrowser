from __future__ import annotations

import multiprocessing
import tkinter as tk
from multiprocessing import Queue
from time import sleep
from typing import Any


def _run_gui(
    is_isekai: bool,
    pause_flag: Any,
    toggle_dict: Any,
    skill_dict: Any,
    cmd_queue: Queue[tuple[str, Any]],
    ready_event: Any,
) -> None:
    """Entry point for the GUI subprocess. Runs on its own main thread."""
    root = tk.Tk()
    mode = "Isekai" if is_isekai else "Persistent"
    root.title(f"Battle Control Panel ({mode})")
    root.minsize(width=300, height=0)

    btn_frame = tk.Frame(root)
    btn_frame.pack(padx=10, pady=5)

    pause_btn = tk.Button(btn_frame, text="Pause")
    pause_btn.pack(side="left", padx=5)

    skill_container = tk.Frame(root)
    skill_container.pack(padx=10, pady=5, fill="x")

    local_toggles: dict[str, tk.BooleanVar] = {}
    local_skills: dict[str, tk.BooleanVar] = {}

    def toggle_pause() -> None:
        if pause_flag.is_set():
            pause_flag.clear()
            pause_btn.config(text="Pause")
        else:
            pause_flag.set()
            pause_btn.config(text="Resume")

    pause_btn.config(command=toggle_pause)

    def sync_to_shared() -> None:
        """Periodically sync local tk vars to shared dicts."""
        for name, var in local_toggles.items():
            toggle_dict[name] = var.get()
        for name, var in local_skills.items():
            skill_dict[name] = var.get()
        root.after(200, sync_to_shared)

    def poll_commands() -> None:
        """Process commands from the main process."""
        try:
            while not cmd_queue.empty():
                cmd, args = cmd_queue.get_nowait()
                if cmd == "register_toggle":
                    name, default = args
                    var = tk.BooleanVar(value=default)
                    local_toggles[name] = var
                    toggle_dict[name] = default
                    cb = tk.Checkbutton(root, text=name, variable=var)
                    cb.pack(anchor="w", padx=10, pady=2)
                elif cmd == "set_skills":
                    skill_groups, forbidden = args
                    for widget in skill_container.winfo_children():
                        widget.destroy()
                    local_skills.clear()
                    for group_name, skills in skill_groups.items():
                        frame = tk.LabelFrame(skill_container, text=group_name)
                        frame.pack(padx=5, pady=3, fill="x")
                        for skill in skills:
                            val = skill not in forbidden
                            var = tk.BooleanVar(value=val)
                            local_skills[skill] = var
                            skill_dict[skill] = val
                            cb = tk.Checkbutton(frame, text=skill, variable=var)
                            cb.pack(anchor="w", padx=5, pady=1)
                elif cmd == "destroy":
                    root.destroy()
                    return
        except Exception:
            pass
        root.after(100, poll_commands)

    root.after(100, poll_commands)
    root.after(200, sync_to_shared)
    ready_event.set()
    root.mainloop()


class ControlPanel:
    def __init__(self, is_isekai: bool = False) -> None:
        manager = multiprocessing.Manager()
        self._pause_flag = manager.Event()
        self._toggle_dict = manager.dict()
        self._skill_dict = manager.dict()
        self._cmd_queue = manager.Queue()
        ready_event = manager.Event()

        self._process = multiprocessing.Process(
            target=_run_gui,
            args=(
                is_isekai,
                self._pause_flag,
                self._toggle_dict,
                self._skill_dict,
                self._cmd_queue,
                ready_event,
            ),
            daemon=True,
        )
        self._process.start()
        ready_event.wait()

    def register_toggle(self, name: str, default: bool = False) -> None:
        self._cmd_queue.put(("register_toggle", (name, default)))

    def get_toggle(self, name: str) -> bool:
        return bool(self._toggle_dict.get(name, False))

    def set_skills(
        self,
        skill_groups: dict[str, list[str]],
        forbidden: list[str],
    ) -> None:
        # Clear stale skill state before sending new layout
        self._skill_dict.clear()
        self._cmd_queue.put(("set_skills", (skill_groups, forbidden)))

    def get_forbidden_skills(self) -> list[str]:
        return [name for name, val in self._skill_dict.items() if not val]

    def wait_if_paused(self) -> None:
        while self._pause_flag.is_set():
            sleep(0.5)

    def destroy(self) -> None:
        self._cmd_queue.put(("destroy", None))
        self._process.join(timeout=3)
