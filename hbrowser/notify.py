__all__ = ["notify"]

import subprocess
import sys

from .beep import beep_os_independent

_NOTIFICATION_TIMEOUT_SECONDS = 5.0


def notify(title: str, message: str) -> None:
    """跨平台系統通知。

    - macOS: 使用 osascript 發送原生通知。
    - Linux: 使用 notify-send (libnotify)。
    - Windows 10+: 使用 PowerShell WinRT Toast Notification。
    - 所有平台失敗時 fallback 到 beep。
    """
    try:
        match sys.platform:
            case "darwin":
                command = [
                    "osascript",
                    "-e",
                    f'display notification "{message}" with title "{title}"',
                ]
            case "linux":
                command = ["notify-send", title, message]
            case "win32":
                ps_script = (
                    "[Windows.UI.Notifications.ToastNotificationManager,"
                    " Windows.UI.Notifications,"
                    " ContentType=WindowsRuntime] | Out-Null;"
                    "[Windows.Data.Xml.Dom.XmlDocument,"
                    " Windows.Data.Xml.Dom.XmlDocument,"
                    " ContentType=WindowsRuntime] | Out-Null;"
                    "$t = [Windows.UI.Notifications.ToastNotificationManager]"
                    "::GetTemplateContent("
                    "[Windows.UI.Notifications.ToastTemplateType]"
                    "::ToastText02);"
                    "$t.GetElementsByTagName('text').Item(0)"
                    f".AppendChild($t.CreateTextNode('{title}'));"
                    "$t.GetElementsByTagName('text').Item(1)"
                    f".AppendChild($t.CreateTextNode('{message}'));"
                    "[Windows.UI.Notifications.ToastNotificationManager]"
                    "::CreateToastNotifier('HBrowser').Show("
                    "[Windows.UI.Notifications.ToastNotification]"
                    "::new($t))"
                )
                command = ["powershell", "-Command", ps_script]
            case _:
                beep_os_independent()
                return

        subprocess.run(
            command,
            check=True,
            capture_output=True,
            timeout=_NOTIFICATION_TIMEOUT_SECONDS,
        )
        return
    except Exception:
        pass

    # Fallback: beep
    beep_os_independent()
