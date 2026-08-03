from __future__ import annotations

import importlib
import io
import subprocess
import unittest
from unittest.mock import Mock, patch

beep_module = importlib.import_module("hbrowser.beep")
notify_module = importlib.import_module("hbrowser.notify")


class NotificationTests(unittest.TestCase):
    def test_linux_notification_has_a_timeout(self) -> None:
        with (
            patch.object(notify_module.sys, "platform", "linux"),
            patch.object(notify_module.subprocess, "run") as run,
            patch.object(notify_module, "beep_os_independent") as beep,
        ):
            notify_module.notify("title", "message")

        run.assert_called_once_with(
            ["notify-send", "title", "message"],
            check=True,
            capture_output=True,
            timeout=notify_module._NOTIFICATION_TIMEOUT_SECONDS,
        )
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
    def test_macos_say_has_a_timeout(self) -> None:
        completed = Mock(returncode=0)
        with (
            patch.object(beep_module.sys, "platform", "darwin"),
            patch.object(beep_module.subprocess, "run", return_value=completed) as run,
        ):
            beep_module.beep_os_independent()

        run.assert_called_once_with(
            ["say", "-v", "Alex", "Warning"],
            capture_output=True,
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
