from __future__ import annotations

import asyncio
import multiprocessing
import tkinter as tk
from abc import ABC, abstractmethod
from multiprocessing import Queue
from typing import Any


def _run_gui(
    pause_flag: Any,
    toggle_dict: Any,
    skill_dict: Any,
    cmd_queue: Queue[tuple[str, Any]],
    ready_event: Any,
) -> None:
    """Entry point for the GUI subprocess. Runs on its own main thread."""
    root = tk.Tk()
    root.title("Battle Control Panel")
    root.minsize(width=300, height=0)

    btn_frame = tk.Frame(root)
    btn_frame.pack(padx=10, pady=5)

    pause_btn = tk.Button(btn_frame, text="Pause")
    pause_btn.pack(side="left", padx=5)

    skill_container = tk.Frame(root)
    skill_container.pack(padx=10, pady=5, fill="x")

    local_toggles: dict[str, tk.BooleanVar] = {}
    local_skills: dict[str, tk.BooleanVar] = {}
    toggle_frame: tk.LabelFrame | None = None
    skill_group_count = 0

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
        nonlocal toggle_frame, skill_group_count
        while not cmd_queue.empty():
            cmd, args = cmd_queue.get_nowait()
            match cmd:
                case "register_toggle":
                    name, label, default = args
                    if toggle_frame is None:
                        toggle_frame = tk.LabelFrame(
                            skill_container, text="Auto Next Battle"
                        )
                        toggle_frame.grid(
                            row=0,
                            column=skill_group_count,
                            padx=5,
                            pady=3,
                            sticky="nsew",
                        )
                        skill_container.columnconfigure(skill_group_count, weight=1)
                    var = tk.BooleanVar(value=default)
                    local_toggles[name] = var
                    toggle_dict[name] = default
                    cb = tk.Checkbutton(toggle_frame, text=label, variable=var)
                    cb.pack(anchor="w", padx=5, pady=1)
                case "set_skills":
                    skill_groups, forbidden = args
                    for widget in skill_container.winfo_children():
                        if widget is not toggle_frame:
                            widget.destroy()
                    local_skills.clear()
                    for col, (group_name, skills) in enumerate(skill_groups.items()):
                        frame = tk.LabelFrame(skill_container, text=group_name)
                        frame.grid(row=0, column=col, padx=5, pady=3, sticky="nsew")
                        for skill in skills:
                            val = skill not in forbidden
                            var = tk.BooleanVar(value=val)
                            local_skills[skill] = var
                            skill_dict[skill] = val
                            cb = tk.Checkbutton(frame, text=skill, variable=var)
                            cb.pack(anchor="w", padx=5, pady=1)
                    skill_group_count = len(skill_groups)
                    for col in range(skill_group_count):
                        skill_container.columnconfigure(col, weight=1)
                    if toggle_frame is not None:
                        toggle_frame.grid(row=0, column=skill_group_count)
                        skill_container.columnconfigure(skill_group_count, weight=1)
                case "set_title":
                    root.title(args)
                case "destroy":
                    root.destroy()
                    return
        root.after(100, poll_commands)

    root.after(100, poll_commands)
    root.after(200, sync_to_shared)
    ready_event.set()
    root.mainloop()


class BaseControlPanel(ABC):
    """戰鬥控制面板介面，讓 GUI 版（ControlPanel）跟無 GUI 版（NullControlPanel）
    可以互相替換（Liskov substitution），呼叫端不需要關心目前是不是 headless。
    """

    @abstractmethod
    def set_title(self, title: str) -> None: ...

    @abstractmethod
    def register_toggle(self, name: str, label: str, default: bool = False) -> None: ...

    @abstractmethod
    def get_toggle(self, name: str) -> bool: ...

    @abstractmethod
    def set_skills(
        self, skill_groups: dict[str, list[str]], forbidden: list[str]
    ) -> None: ...

    @abstractmethod
    def get_forbidden_skills(self) -> list[str]: ...

    @abstractmethod
    async def wait_if_paused(self) -> None: ...

    @abstractmethod
    def destroy(self) -> None: ...


class ControlPanel(BaseControlPanel):
    def __init__(self) -> None:
        manager = multiprocessing.Manager()
        self._pause_flag = manager.Event()
        self._toggle_dict = manager.dict()
        self._skill_dict = manager.dict()
        self._cmd_queue = manager.Queue()
        ready_event = manager.Event()

        self._process = multiprocessing.Process(
            target=_run_gui,
            args=(
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

    def set_title(self, title: str) -> None:
        self._cmd_queue.put(("set_title", title))

    def register_toggle(self, name: str, label: str, default: bool = False) -> None:
        self._cmd_queue.put(("register_toggle", (name, label, default)))

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

    async def wait_if_paused(self) -> None:
        # 用 asyncio.sleep 而非 time.sleep：暫停期間絕對不能擋住 event loop，
        # 否則 CDP websocket 連線在暫停期間完全沒有機會處理任何背景事件，
        # 恢復時很容易直接拿到 ConnectionClosed。
        while self._pause_flag.is_set():
            await asyncio.sleep(0.5)

    def destroy(self) -> None:
        self._cmd_queue.put(("destroy", None))
        self._process.join(timeout=3)


class NullControlPanel(BaseControlPanel):
    """headless 模式使用：不開 GUI 視窗，僅在記憶體中保存設定。"""

    def __init__(self) -> None:
        self._toggles: dict[str, bool] = {}
        self._forbidden_skills: list[str] = []

    def set_title(self, title: str) -> None:
        pass

    def register_toggle(self, name: str, label: str, default: bool = False) -> None:
        self._toggles[name] = default

    def get_toggle(self, name: str) -> bool:
        return self._toggles.get(name, False)

    def set_skills(
        self, skill_groups: dict[str, list[str]], forbidden: list[str]
    ) -> None:
        self._forbidden_skills = list(forbidden)

    def get_forbidden_skills(self) -> list[str]:
        return list(self._forbidden_skills)

    async def wait_if_paused(self) -> None:
        return

    def destroy(self) -> None:
        pass
