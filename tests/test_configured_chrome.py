import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import ANY, AsyncMock, Mock, patch

from hbrowser.gallery.browser import factory
from hbrowser.gallery.browser.chrome_manager import ChromePaths
from hbrowser.gallery.browser.process import OwnedProcess
from hbrowser.gallery.driver_base import Driver

_ENVIRONMENT_VARIABLE = "HBROWSER_CHROME_EXECUTABLE"


def _make_executable(root: Path, name: str = "chrome") -> Path:
    executable = root / name
    executable.touch()
    executable.chmod(0o700)
    return executable


def _make_symlink(test: unittest.TestCase, link: Path, target: Path) -> Path:
    try:
        link.symlink_to(target)
    except NotImplementedError, OSError:
        test.skipTest("symbolic links are unavailable")
    return link


class ConfiguredChromeValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.enterContext(patch.dict(os.environ, {}, clear=True))
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))

    def _assert_rejected(self, value: str) -> None:
        with (
            patch.dict(os.environ, {_ENVIRONMENT_VARIABLE: value}),
            self.assertRaisesRegex(ValueError, _ENVIRONMENT_VARIABLE) as raised,
        ):
            factory._configured_chrome_executable()
        if value.strip():
            self.assertNotIn(value, str(raised.exception))

    def test_unset_uses_the_managed_installer(self) -> None:
        self.assertIsNone(factory._configured_chrome_executable())

    def test_empty_or_whitespace_is_rejected(self) -> None:
        for value in ("", " ", "\t\n"):
            with self.subTest(value=repr(value)):
                self._assert_rejected(value)

    def test_relative_path_is_rejected(self) -> None:
        self._assert_rejected("private-browser/chrome")

    def test_missing_file_is_rejected_without_disclosing_its_path(self) -> None:
        self._assert_rejected(str(self.root / "private-missing-chrome"))

    def test_directory_is_rejected(self) -> None:
        self._assert_rejected(str(self.root))

    def test_non_executable_file_is_rejected(self) -> None:
        executable = _make_executable(self.root)
        executable.chmod(0o600)
        # Windows does not implement POSIX mode-bit executable access checks.
        with patch("hbrowser.gallery.browser.factory.os.access", return_value=False):
            self._assert_rejected(str(executable))

    def test_broken_symlink_is_rejected(self) -> None:
        link = _make_symlink(self, self.root / "chrome-link", self.root / "missing")
        self._assert_rejected(str(link))

    def test_executable_path_keeps_its_original_spelling(self) -> None:
        executable = _make_executable(self.root, "Chrome with spaces")
        configured = f"{executable.parent}{os.sep}.{os.sep}{executable.name}"
        with patch.dict(os.environ, {_ENVIRONMENT_VARIABLE: configured}):
            self.assertEqual(factory._configured_chrome_executable(), configured)

    def test_symlink_to_executable_keeps_the_link_path(self) -> None:
        executable = _make_executable(self.root)
        link = _make_symlink(self, self.root / "chrome-link", executable)
        with patch.dict(os.environ, {_ENVIRONMENT_VARIABLE: str(link)}):
            self.assertEqual(factory._configured_chrome_executable(), str(link))

    def test_configuration_is_read_again_for_each_startup(self) -> None:
        first = _make_executable(self.root, "first")
        second = _make_executable(self.root, "second")
        for executable in (first, second):
            with patch.dict(os.environ, {_ENVIRONMENT_VARIABLE: str(executable)}):
                self.assertEqual(
                    factory._configured_chrome_executable(), str(executable)
                )


class _TestDriver(Driver):
    def _setname(self) -> str:
        return "E-Hentai"


class ConfiguredChromeStartupTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.enterContext(patch.dict(os.environ, {}, clear=True))
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.executable = _make_executable(self.root)
        self.profile = self.root / "profile"
        self.profile.mkdir()
        self.extension = self.root / "proxy-extension"
        self.extension.mkdir()
        self.owner = Mock(spec=OwnedProcess)
        self.owner.poll.return_value = None
        self.browser = SimpleNamespace(
            connection=None,
            targets=[],
            start=AsyncMock(),
            stop=AsyncMock(),
        )
        self.browser.start.return_value = self.browser
        self.page = object()
        self.use_tor = self.enterContext(
            patch.object(factory, "should_use_tor", return_value=False)
        )
        self.start_tor = self.enterContext(
            patch.object(factory, "start_tor_with_retry")
        )
        self.proxy = self.enterContext(
            patch.object(factory, "configure_proxy", return_value=None)
        )
        self.install = self.enterContext(
            patch.object(
                factory,
                "_install_chrome_in_owned_worker",
                new=AsyncMock(
                    return_value=ChromePaths(chrome=str(self.executable), version="1")
                ),
            )
        )
        self.autodiscover = self.enterContext(
            patch(
                "zendriver.core.config.find_executable",
                side_effect=AssertionError(
                    "explicit executable must bypass host search"
                ),
            )
        )
        self.staging = self.enterContext(
            patch.object(factory, "create_chrome_install_staging_root")
        )
        self.worker = self.enterContext(patch.object(factory, "start_owned_process"))
        self.make_profile = self.enterContext(
            patch(
                "hbrowser.gallery.browser.factory.tempfile.mkdtemp",
                return_value=str(self.profile),
            )
        )
        self.launch = self.enterContext(
            patch.object(
                factory, "start_owned_browser_process", return_value=self.owner
            )
        )
        self.enterContext(
            patch.object(
                factory,
                "_wait_for_devtools_active_port_async",
                new=AsyncMock(return_value=43123),
            )
        )
        self.client = self.enterContext(
            patch.object(factory, "_OwnedZendriverBrowser", return_value=self.browser)
        )
        self.register = self.enterContext(
            patch.object(factory, "_register_browser_atexit")
        )
        self.main_tab = self.enterContext(
            patch.object(
                factory, "_wait_for_main_tab", new=AsyncMock(return_value=self.page)
            )
        )
        self.setup = self.enterContext(
            patch.object(factory, "_post_create_setup", new=AsyncMock())
        )
        self.janitor = self.enterContext(
            patch.object(factory, "start_zendriver_mapper_janitor")
        )

    def _assert_no_installer(self) -> None:
        self.install.assert_not_awaited()
        self.staging.assert_not_called()
        self.worker.assert_not_called()
        self.autodiscover.assert_not_called()

    def _assert_no_startup_side_effects(self) -> None:
        self.use_tor.assert_not_called()
        self.start_tor.assert_not_called()
        self.proxy.assert_not_called()
        self.make_profile.assert_not_called()
        self.launch.assert_not_called()
        self._assert_no_installer()

    def _assert_owner_shutdown(self) -> None:
        self.browser.stop.assert_awaited_once_with()
        self.owner.shutdown.assert_called_once_with(
            graceful_timeout=factory._BROWSER_PROCESS_NATURAL_EXIT_SECONDS,
            terminate_timeout=factory._BROWSER_PROCESS_TERMINATE_WAIT_SECONDS,
            kill_timeout=factory._BROWSER_PROCESS_KILL_WAIT_SECONDS,
            cleanup_timeout=factory._BROWSER_PRIVATE_RELEASE_TIMEOUT_SECONDS,
            deadline=ANY,
        )
        self.assertIs(
            getattr(self.browser, factory._BROWSER_PROCESS_OWNER_ATTRIBUTE), self.owner
        )

    async def _assert_configured_success(self, *, headless: bool, proxy: bool) -> None:
        os.environ[_ENVIRONMENT_VARIABLE] = str(self.executable)
        self.proxy.return_value = str(self.extension) if proxy else None

        result = await factory.create_browser(headless=headless)

        self.assertEqual(result, (self.browser, self.page))
        self._assert_no_installer()
        self.launch.assert_called_once()
        executable, parameters = self.launch.call_args.args
        self.assertEqual(executable, self.executable)
        config = self.client.call_args.args[0]
        self.assertEqual(config.browser_executable_path, str(self.executable))
        self.assertEqual(config.user_data_dir, str(self.profile))
        self.assertEqual(config.headless, headless)
        self.assertFalse(config.sandbox)
        self.assertIn("--remote-debugging-port=0", parameters)
        self.assertEqual(parameters[-1], "about:blank")
        cleanup_paths = (
            (str(self.profile), str(self.extension)) if proxy else (str(self.profile),)
        )
        self.assertEqual(self.launch.call_args.kwargs["cleanup_paths"], cleanup_paths)
        self.assertNotIn(str(self.executable), cleanup_paths)
        if proxy:
            self.assertIn(f"--load-extension={self.extension}", parameters)
        else:
            self.assertIn("--disable-extensions", parameters)
        self.register.assert_called_once_with(self.browser)
        self.setup.assert_awaited_once_with(
            self.browser, self.page, False, deadline=ANY
        )
        self.janitor.assert_called_once_with(self.browser)

        await factory.stop_browser(self.browser)

        self._assert_owner_shutdown()
        self.assertTrue(self.executable.is_file())

    async def test_configured_headless_browser_retains_owned_launch_and_cleanup(
        self,
    ) -> None:
        await self._assert_configured_success(headless=True, proxy=False)

    async def test_configured_windowed_browser_retains_proxy_extension_ownership(
        self,
    ) -> None:
        await self._assert_configured_success(headless=False, proxy=True)

    async def test_invalid_configuration_fails_before_any_startup_side_effect(
        self,
    ) -> None:
        for configured in ("", "relative-chrome", str(self.root / "missing")):
            with (
                self.subTest(configured=configured),
                patch.dict(os.environ, {_ENVIRONMENT_VARIABLE: configured}),
                self.assertRaisesRegex(ValueError, _ENVIRONMENT_VARIABLE),
            ):
                await factory.create_browser()
        self._assert_no_startup_side_effects()

    async def test_driver_rejects_invalid_configuration_before_ownership(self) -> None:
        os.environ[_ENVIRONMENT_VARIABLE] = str(self.root / "missing")
        driver = _TestDriver()
        driver.logger = Mock()
        with (
            patch.object(driver, "login", new=AsyncMock()) as login,
            patch.object(driver, "gohomepage", new=AsyncMock()) as homepage,
            self.assertRaisesRegex(ValueError, _ENVIRONMENT_VARIABLE),
        ):
            await driver.__aenter__()

        self.assertFalse(driver.is_browser_bound)
        self.assertFalse(driver.owns_browser)
        login.assert_not_awaited()
        homepage.assert_not_awaited()
        self._assert_no_startup_side_effects()

    async def test_unset_configuration_still_uses_owned_installer(self) -> None:
        result = await factory.create_browser()

        self.assertEqual(result, (self.browser, self.page))
        self.install.assert_awaited_once()
        self.autodiscover.assert_not_called()
        self.launch.assert_called_once()
        self.assertEqual(self.launch.call_args.args[0], self.executable)
        self.assertEqual(
            self.client.call_args.args[0].browser_executable_path, str(self.executable)
        )
        await factory.stop_browser(self.browser)
        self._assert_owner_shutdown()

    async def test_configured_browser_start_failure_reaps_the_same_owner(self) -> None:
        os.environ[_ENVIRONMENT_VARIABLE] = str(self.executable)
        failure = RuntimeError("connection failed")
        self.browser.start.side_effect = failure

        with self.assertRaises(RuntimeError) as raised:
            await factory.create_browser()

        self.assertIs(raised.exception, failure)
        self._assert_no_installer()
        self.launch.assert_called_once()
        self._assert_owner_shutdown()
        self.main_tab.assert_not_awaited()
        self.setup.assert_not_awaited()
        self.janitor.assert_not_called()
        self.assertTrue(self.executable.is_file())
