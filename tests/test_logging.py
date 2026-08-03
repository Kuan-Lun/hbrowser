from __future__ import annotations

import io
import logging
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from hbrowser.gallery.utils.log import get_log_dir, setup_logger


class LoggerSetupTests(unittest.TestCase):
    def test_logger_keeps_stdout_handler_and_non_propagating_contract(self) -> None:
        logger_name = f"hbrowser.tests.compatibility.{id(self)}"
        logger = logging.getLogger(logger_name)
        original_handlers = logger.handlers[:]
        original_level = logger.level
        original_propagate = logger.propagate
        output = io.StringIO()
        try:
            logger.handlers = []
            with (
                patch("hbrowser.gallery.utils.log.sys.stdout", output),
                patch.dict(
                    "os.environ",
                    {"HBROWSER_LOG_LEVEL": "DEBUG"},
                    clear=False,
                ),
            ):
                configured = setup_logger(logger_name)
                configured.debug("compatibility record")
                configured_handlers = configured.handlers[:]
                configured_level = configured.level
                handler_level = configured.handlers[0].level
                configured_propagate = configured.propagate
        finally:
            logger.handlers = original_handlers
            logger.setLevel(original_level)
            logger.propagate = original_propagate

        self.assertIs(configured, logger)
        self.assertEqual(len(configured_handlers), 1)
        self.assertIsInstance(configured_handlers[0], logging.StreamHandler)
        self.assertEqual(configured_level, logging.DEBUG)
        self.assertEqual(handler_level, logging.DEBUG)
        self.assertFalse(configured_propagate)
        self.assertIn("compatibility record", output.getvalue())

    def test_repeated_setup_does_not_duplicate_handlers(self) -> None:
        logger_name = f"hbrowser.tests.idempotent.{id(self)}"
        logger = logging.getLogger(logger_name)
        original_handlers = logger.handlers[:]
        try:
            logger.handlers = []
            first = setup_logger(logger_name)
            handler = first.handlers[0]
            second = setup_logger(logger_name)
            configured_handlers = second.handlers[:]
        finally:
            logger.handlers = original_handlers

        self.assertIs(second, first)
        self.assertEqual(configured_handlers, [handler])


class LogDirectoryTests(unittest.TestCase):
    def test_environment_override_creates_nested_absolute_directory(self) -> None:
        with TemporaryDirectory() as directory_name:
            configured = Path(directory_name) / "nested" / "diagnostics"
            with patch.dict(
                "os.environ",
                {"HBROWSER_LOG_DIR": str(configured)},
                clear=False,
            ):
                result = get_log_dir()

            self.assertEqual(result, configured.resolve())
            self.assertTrue(result.is_dir())

    def test_default_directory_stays_next_to_main_script(self) -> None:
        with TemporaryDirectory() as directory_name:
            script = Path(directory_name) / "application" / "run.py"
            with (
                patch.dict("os.environ", {}, clear=False),
                patch("sys.argv", [str(script)]),
                patch.dict("os.environ", {"HBROWSER_LOG_DIR": ""}, clear=False),
            ):
                result = get_log_dir()

            self.assertEqual(result, script.parent.resolve() / "log")
            self.assertTrue(result.is_dir())
