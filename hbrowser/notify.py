__all__ = ["notify"]

import json
import subprocess
import sys

from .beep import beep_os_independent
from .gallery.utils._child_environment import (
    environment_without_logging_capabilities,
)

_NOTIFICATION_TIMEOUT_SECONDS = 5.0
_APPLESCRIPT_NOTIFICATION = """on run argv
    display notification (item 2 of argv) with title (item 1 of argv)
end run
"""
_POWERSHELL_NOTIFICATION = """
$ErrorActionPreference = 'Stop';
$notification = ConvertFrom-Json -InputObject ([Console]::In.ReadToEnd());
[Windows.UI.Notifications.ToastNotificationManager,
 Windows.UI.Notifications, ContentType=WindowsRuntime] | Out-Null;
[Windows.Data.Xml.Dom.XmlDocument,
 Windows.Data.Xml.Dom.XmlDocument, ContentType=WindowsRuntime] | Out-Null;
$t = [Windows.UI.Notifications.ToastNotificationManager]::GetTemplateContent(
    [Windows.UI.Notifications.ToastTemplateType]::ToastText02);
$t.GetElementsByTagName('text').Item(0).AppendChild(
    $t.CreateTextNode($notification.title)) | Out-Null;
$t.GetElementsByTagName('text').Item(1).AppendChild(
    $t.CreateTextNode($notification.message)) | Out-Null;
[Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier(
    'HBrowser').Show([Windows.UI.Notifications.ToastNotification]::new($t));
"""


def notify(title: str, message: str) -> None:
    """跨平台系統通知。

    - macOS: 使用 osascript 發送原生通知。
    - Linux: 使用 notify-send (libnotify)。
    - Windows 10+: 使用 PowerShell WinRT Toast Notification。
    - 所有平台失敗時 fallback 到 beep。
    """
    try:
        payload: bytes | None = None
        match sys.platform:
            case "darwin":
                command = [
                    "osascript",
                    "-e",
                    _APPLESCRIPT_NOTIFICATION,
                    "--",
                    title,
                    message,
                ]
            case "linux":
                command = ["notify-send", "--", title, message]
            case "win32":
                # ASCII JSON also works with Windows PowerShell's legacy stdin
                # encoding; notification data never becomes PowerShell source.
                payload = json.dumps({"title": title, "message": message}).encode(
                    "ascii"
                )
                command = [
                    "powershell",
                    "-NoProfile",
                    "-NonInteractive",
                    "-Command",
                    _POWERSHELL_NOTIFICATION,
                ]
            case _:
                beep_os_independent()
                return

        subprocess.run(
            command,
            check=True,
            capture_output=True,
            input=payload,
            env=environment_without_logging_capabilities(),
            timeout=_NOTIFICATION_TIMEOUT_SECONDS,
        )
        return
    except Exception:
        pass

    # Fallback: beep
    beep_os_independent()
