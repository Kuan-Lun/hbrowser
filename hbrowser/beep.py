__all__ = ["beep_os_independent"]

import os
import sys


def beep_os_independent() -> None:
    """跨平台提示音。

    - Windows: 使用 winsound.MessageBeep。
    - macOS: 使用 say 命令播放语音提示，更稳定。
    - 其他平台（含 Linux）: 输出 ASCII 铃声 "\a" 到标准输出。

    不使用第三方套件，全部为内置或系统自带能力。
    """

    try:
        if sys.platform == "win32":
            # Windows 原生 API
            import winsound  # type: ignore

            winsound.MessageBeep()
            return

        if sys.platform == "darwin":
            # macOS: 使用 AppleScript 的系统蜂鸣，更稳定
            # 若 osascript 不可用，则回退到 ASCII 铃声
            exit_code = os.system('say -v Alex "Warning"')
            if exit_code == 0:
                return
            # 回退：继续走到通用分支

        # Linux 及通用回退：输出 ASCII 铃声
        try:
            # 直接写入标准输出，避免依赖 shell 行为（echo -e/-n 差异）
            sys.stdout.write("\a")
            sys.stdout.flush()
        except Exception:
            # 最后再退回到一个尽量简单的方式
            os.system('printf "\a" >/dev/null 2>&1')
    except Exception:
        # 忽略所有异常，避免因提示音影响主流程
        pass
