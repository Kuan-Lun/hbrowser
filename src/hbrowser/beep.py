__all__ = ["beep_os_independent"]

import sys
import os


def beep_os_independent():
    if sys.platform.startswith("linux") or sys.platform == "darwin":
        # 对于 Linux 和 macOS
        os.system('echo -n "\a"')  # 这会在 shell 中使用 echo 命令来发出提示音
    elif sys.platform == "win32":
        # 对于 Windows
        import winsound
        import time

        winsound.MessageBeep()  # 使用 winsound 模块发出默认提示音
        time.sleep(10)
