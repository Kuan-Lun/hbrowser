from __future__ import annotations

import importlib
import io
import json
import subprocess
import unittest
from unittest.mock import Mock, patch

beep_module = importlib.import_module("hbrowser.beep")
notify_module = importlib.import_module("hbrowser.notify")
child_environment_module = importlib.import_module(
    "hbrowser.gallery.utils._child_environment"
)


class NotificationTests(unittest.TestCase):
    def test_linux_notification_has_a_timeout(self) -> None:
        with (
            patch.object(notify_module.sys, "platform", "linux"),
            patch.object(notify_module.subprocess, "run") as run,
            patch.object(
                notify_module,
                "environment_without_logging_capabilities",
                return_value={"PATH": "/commands"},
            ),
            patch.object(notify_module, "beep_os_independent") as beep,
        ):
            notify_module.notify("title", "message")

        run.assert_called_once_with(
            ["notify-send", "--", "title", "message"],
            check=True,
            capture_output=True,
            input=None,
            env={"PATH": "/commands"},
            timeout=notify_module._NOTIFICATION_TIMEOUT_SECONDS,
        )
        beep.assert_not_called()

    def test_macos_content_is_passed_as_arguments_not_applescript(self) -> None:
        title = '-title " & do shell script "unexpected" & "'
        message = "message\\\n新訊息 ' \""
        with (
            patch.object(notify_module.sys, "platform", "darwin"),
            patch.object(notify_module.subprocess, "run") as run,
            patch.object(notify_module, "beep_os_independent") as beep,
        ):
            notify_module.notify(title, message)
        command = run.call_args.args[0]
        self.assertEqual(command[:2], ["osascript", "-e"])
        self.assertEqual(command[3:], ["--", title, message])
        self.assertNotIn(title, command[2])
        self.assertNotIn(message, command[2])
        self.assertIn("item 1 of argv", command[2])
        self.assertIn("item 2 of argv", command[2])
        beep.assert_not_called()

    def test_windows_content_is_json_data_not_powershell(self) -> None:
        title = "'); throw 'unexpected'; #"
        message = "新訊息\n' \" ` $($env:PATH)"
        with (
            patch.object(notify_module.sys, "platform", "win32"),
            patch.object(notify_module.subprocess, "run") as run,
            patch.object(notify_module, "beep_os_independent") as beep,
        ):
            notify_module.notify(title, message)
        command = run.call_args.args[0]
        self.assertEqual(
            command[:4], ["powershell", "-NoProfile", "-NonInteractive", "-Command"]
        )
        self.assertNotIn(title, command[4])
        self.assertNotIn(message, command[4])
        self.assertIn("ConvertFrom-Json", command[4])
        self.assertIn("[Console]::In.ReadToEnd()", command[4])
        payload = run.call_args.kwargs["input"]
        self.assertTrue(payload.isascii())
        self.assertEqual(json.loads(payload), {"title": title, "message": message})
        beep.assert_not_called()

    def test_timed_out_notification_falls_back_to_beep(self) -> None:
        timeout = subprocess.TimeoutExpired("notify-send", 5)
        with (
            patch.object(notify_module.sys, "platform", "linux"),
            patch.object(notify_module.subprocess, "run", side_effect=timeout),
            patch.object(notify_module, "beep_os_independent") as beep,
        ):
            notify_module.notify("title", "message")

        beep.assert_called_once_with()

    def test_unsupported_platform_falls_back_to_beep(self) -> None:
        with (
            patch.object(notify_module.sys, "platform", "unsupported"),
            patch.object(notify_module.subprocess, "run") as run,
            patch.object(notify_module, "beep_os_independent") as beep,
        ):
            notify_module.notify("title", "message")

        run.assert_not_called()
        beep.assert_called_once_with()


class BeepTests(unittest.TestCase):
    def test_external_command_environment_strips_only_log_ownership(self) -> None:
        environment = {
            "DBUS_SESSION_BUS_ADDRESS": "preserved",
            "hbrowser_log_dir": "removed-case-insensitively",
            "HBROWSER_PROCESS_LOG_FILE": "removed",
            "HBROWSER_LOG_FORWARD_ENDPOINT": "removed",
            "HBROWSER_LOG_FORWARD_TOKEN": "removed",
        }

        sanitized = child_environment_module.environment_without_logging_capabilities(
            environment
        )

        self.assertEqual(sanitized, {"DBUS_SESSION_BUS_ADDRESS": "preserved"})

    def test_macos_say_has_a_timeout(self) -> None:
        completed = Mock(returncode=0)
        with (
            patch.object(beep_module.sys, "platform", "darwin"),
            patch.object(beep_module.subprocess, "run", return_value=completed) as run,
            patch.object(
                beep_module,
                "environment_without_logging_capabilities",
                return_value={"PATH": "/commands"},
            ),
        ):
            beep_module.beep_os_independent()

        run.assert_called_once_with(
            ["say", "-v", "Alex", "Warning"],
            capture_output=True,
            env={"PATH": "/commands"},
            timeout=beep_module._BEEP_TIMEOUT_SECONDS,
        )

    def test_timed_out_macos_say_falls_back_to_ascii_bell(self) -> None:
        output = io.StringIO()
        timeout = subprocess.TimeoutExpired("say", 5)
        with (
            patch.object(beep_module.sys, "platform", "darwin"),
            patch.object(beep_module.subprocess, "run", side_effect=timeout),
            patch.object(beep_module.sys, "stdout", output),
        ):
            beep_module.beep_os_independent()

        self.assertEqual(output.getvalue(), "\a")
