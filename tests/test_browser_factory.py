import unittest
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock, Mock, patch

import zendriver as zd
from zendriver import cdp

from hbrowser.gallery.browser import factory
from hbrowser.gallery.browser.mapper import start_zendriver_mapper_janitor


def _tab(target_id: str = "page-1", *, type_: str = "page") -> zd.Tab:
    target = cdp.target.TargetInfo(
        target_id=cdp.target.TargetID(target_id),
        type_=type_,
        title="",
        url="about:blank",
        attached=False,
        can_access_opener=False,
    )
    return zd.Tab(f"ws://127.0.0.1/devtools/page/{target_id}", target)


class _Process:
    def __init__(self, returncode: int | None = None) -> None:
        self.returncode = returncode

    def poll(self) -> int | None:
        return self.returncode


class _Target:
    type_ = "page"
    url = "about:blank"


class _Browser:
    def __init__(
        self,
        main_tab: zd.Tab | None,
        tabs: list[object] | None = None,
        returncode: int | None = None,
    ) -> None:
        self._main_tab = main_tab
        self.tabs = tabs or []
        self.targets = self.tabs
        self._process = _Process(returncode)

    @property
    def main_tab(self) -> zd.Tab | None:
        return self._main_tab

    @property
    def stopped(self) -> bool:
        return self._process.poll() is not None


class WaitForMainTabTests(unittest.IsolatedAsyncioTestCase):
    async def test_returns_immediate_main_tab_without_sleeping(self) -> None:
        page = _tab()
        browser = _Browser(page)

        with patch(
            "hbrowser.gallery.browser.factory.asyncio.sleep",
            new=AsyncMock(),
        ) as sleep:
            result = await factory._wait_for_main_tab(cast(zd.Browser, browser))

        self.assertIs(result, page)
        sleep.assert_not_awaited()

    async def test_waits_for_delayed_target_created_event(self) -> None:
        page = _tab()
        browser = _Browser(None, tabs=[_Target()])

        async def expose_tab(_: float) -> None:
            browser.targets.append(page)

        logger = Mock()
        with (
            patch(
                "hbrowser.gallery.browser.factory.asyncio.sleep",
                new=AsyncMock(side_effect=expose_tab),
            ) as sleep,
            patch.object(factory, "logger", logger),
        ):
            result = await factory._wait_for_main_tab(cast(zd.Browser, browser))

        self.assertIs(result, page)
        sleep.assert_awaited_once()
        logger.warning.assert_called_once()
        logger.debug.assert_called_once_with(
            "Browser main tab became available after startup delay"
        )
        logger.info.assert_not_called()

    async def test_recovers_tab_when_main_tab_is_blocked_by_connection(self) -> None:
        page = _tab()
        browser = _Browser(None, tabs=[_Target(), page])

        result = await factory._wait_for_main_tab(cast(zd.Browser, browser))

        self.assertIs(result, page)

    async def test_timeout_includes_browser_process_and_target_state(self) -> None:
        browser = _Browser(None, tabs=[_Target()])

        with self.assertRaises(RuntimeError) as raised:
            await factory._wait_for_main_tab(
                cast(zd.Browser, browser),
                timeout=0,
            )

        message = str(raised.exception)
        self.assertIn("within 0.0 seconds", message)
        self.assertIn("stopped=False", message)
        self.assertIn("process_returncode=None", message)
        self.assertIn("_Target(type='page', url='about:blank')", message)

    async def test_fails_immediately_when_browser_process_exits(self) -> None:
        page = _tab()
        browser = _Browser(page, tabs=[page], returncode=17)

        with (
            patch(
                "hbrowser.gallery.browser.factory.asyncio.sleep",
                new=AsyncMock(),
            ) as sleep,
            self.assertRaisesRegex(
                RuntimeError,
                "stopped=True, process_returncode=17",
            ),
        ):
            await factory._wait_for_main_tab(cast(zd.Browser, browser))

        sleep.assert_not_awaited()

    async def test_rejects_non_page_tab_returned_by_main_tab(self) -> None:
        worker = _tab("worker-1", type_="service_worker")
        browser = _Browser(worker, tabs=[worker])

        with self.assertRaises(RuntimeError):
            await factory._wait_for_main_tab(
                cast(zd.Browser, browser),
                timeout=0,
            )


class CreateBrowserCleanupTests(unittest.IsolatedAsyncioTestCase):
    async def test_success_starts_mapper_janitor_after_post_setup(self) -> None:
        page = _tab()
        browser = _Browser(page, tabs=[page])
        events: list[str] = []

        async def post_setup(*_: object) -> None:
            events.append("post-setup")

        def start_janitor(_: object) -> None:
            events.append("start-janitor")

        with (
            patch.object(factory, "should_use_tor", return_value=False),
            patch.object(factory, "configure_proxy", return_value=None),
            patch.object(factory, "ensure_chrome_installed") as ensure_chrome,
            patch.object(factory, "_build_config", return_value=object()),
            patch.object(
                zd,
                "start",
                new=AsyncMock(return_value=browser),
            ),
            patch.object(
                factory,
                "_post_create_setup",
                new=AsyncMock(side_effect=post_setup),
            ),
            patch.object(
                factory,
                "start_zendriver_mapper_janitor",
                side_effect=start_janitor,
            ) as start_mapper_janitor,
        ):
            ensure_chrome.return_value.chrome = "/test/chrome"
            result = await factory.create_browser()

        self.assertEqual(result, (browser, page))
        self.assertEqual(events, ["post-setup", "start-janitor"])
        start_mapper_janitor.assert_called_once_with(browser)

    async def test_main_tab_failure_stops_browser_without_post_setup(self) -> None:
        browser = _Browser(None)
        failure = RuntimeError("main tab unavailable")

        with (
            patch.object(factory, "should_use_tor", return_value=False),
            patch.object(factory, "configure_proxy", return_value=None),
            patch.object(factory, "ensure_chrome_installed") as ensure_chrome,
            patch.object(factory, "_build_config", return_value=object()),
            patch.object(
                zd,
                "start",
                new=AsyncMock(return_value=browser),
            ),
            patch.object(
                factory,
                "_wait_for_main_tab",
                new=AsyncMock(side_effect=failure),
            ),
            patch.object(
                factory,
                "_post_create_setup",
                new=AsyncMock(),
            ) as post_setup,
            patch.object(
                factory,
                "stop_browser",
                new=AsyncMock(),
            ) as stop_browser,
            patch.object(
                factory,
                "start_zendriver_mapper_janitor",
            ) as start_mapper_janitor,
            self.assertRaisesRegex(RuntimeError, "main tab unavailable"),
        ):
            ensure_chrome.return_value.chrome = "/test/chrome"
            await factory.create_browser()

        stop_browser.assert_awaited_once_with(browser)
        post_setup.assert_not_awaited()
        start_mapper_janitor.assert_not_called()

    async def test_stop_waits_for_mapper_janitor_before_browser(self) -> None:
        browser = SimpleNamespace(
            connection=None,
            targets=[],
            _tor_process=None,
            stop=AsyncMock(),
        )
        events: list[str] = []
        janitor = start_zendriver_mapper_janitor(browser)

        async def stop_process() -> None:
            self.assertTrue(janitor.done())
            events.append("stop-browser")

        browser.stop.side_effect = stop_process
        await factory.stop_browser(browser)

        self.assertEqual(events, ["stop-browser"])
        self.assertTrue(janitor.cancelled())
        browser.stop.assert_awaited_once_with()

    async def test_browser_stop_failure_still_finishes_mapper_janitor(self) -> None:
        browser = SimpleNamespace(
            connection=None,
            targets=[],
            _tor_process=None,
            stop=AsyncMock(side_effect=RuntimeError("browser stop failed")),
        )
        janitor = start_zendriver_mapper_janitor(browser)

        with self.assertRaisesRegex(RuntimeError, "browser stop failed"):
            await factory.stop_browser(browser)

        self.assertTrue(janitor.done())
        self.assertTrue(janitor.cancelled())

    async def test_janitor_stop_failure_still_stops_browser(self) -> None:
        browser = Mock()
        browser._tor_process = None
        browser.stop = AsyncMock()
        failure = RuntimeError("janitor stop failed")

        with (
            patch.object(
                factory,
                "stop_zendriver_mapper_janitor",
                new=AsyncMock(side_effect=failure),
            ),
            self.assertRaisesRegex(RuntimeError, "janitor stop failed"),
        ):
            await factory.stop_browser(browser)

        browser.stop.assert_awaited_once_with()
