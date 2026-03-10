__all__ = ["notify"]

import subprocess
import sys

from .beep import beep_os_independent


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
                subprocess.run(
                    [
                        "osascript",
                        "-e",
                        f'display notification "{message}" with title "{title}"',
                    ],
                    check=True,
                    capture_output=True,
                )
            case "linux":
                subprocess.run(
                    ["notify-send", title, message],
                    check=True,
                    capture_output=True,
                )
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
                subprocess.run(
                    ["powershell", "-Command", ps_script],
                    check=True,
                    capture_output=True,
                )
        return
    except Exception:
        pass

    # Fallback: beep
    beep_os_independent()
