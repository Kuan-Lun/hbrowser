__all__ = ["beep_os_independent"]

import subprocess
import sys

_BEEP_TIMEOUT_SECONDS = 5.0


def beep_os_independent() -> None:
    """跨平台提示音。

    - Windows: 使用 winsound.MessageBeep。
    - macOS: 使用 say 命令播放语音提示，更稳定。
    - 其他平台（含 Linux）: 输出 ASCII 铃声 "\a" 到标准输出。

    不使用第三方套件，全部为内置或系统自带能力。
    """

    if sys.platform == "win32":
        try:
            # Windows 原生 API
            import winsound  # type: ignore

            winsound.MessageBeep()
        except Exception:
            pass
        return

    if sys.platform == "darwin":
        # macOS: 使用 say 命令播放语音提示，更稳定
        # 若 say 不可用、失敗或逾時，則回退到 ASCII 鈴聲。
        try:
            result = subprocess.run(
                ["say", "-v", "Alex", "Warning"],
                capture_output=True,
                timeout=_BEEP_TIMEOUT_SECONDS,
            )
            if result.returncode == 0:
                return
        except Exception:
            pass

    # Linux 及通用回退：输出 ASCII 铃声
    try:
        # 直接写入标准输出，避免依赖 shell 行为（echo -e/-n 差異）
        sys.stdout.write("\a")
        sys.stdout.flush()
    except Exception:
        # 最后再退回到一个尽量简单的方式
        try:
            subprocess.run(
                ["printf", "\a"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=_BEEP_TIMEOUT_SECONDS,
            )
        except Exception:
            # 忽略所有異常，避免因提示音影響主流程
            pass
