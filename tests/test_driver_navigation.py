import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from hbrowser import BrowserMutationOutcomeUnknownError
from hbrowser.gallery.browser import FlareSolverrResult
from hbrowser.gallery.driver_base import Driver
from hbrowser.gallery.forums_auth import ForumsAuthState
from hbrowser.gallery.utils import (
    ZendriverOperationTimeout,
)
from hbrowser.gallery.utils.protocol import ZendriverOwnerRetiredError


class _TestDriver(Driver):
    def _setname(self) -> str:
        return "E-Hentai"


class DriverHomepageNavigationTests(unittest.IsolatedAsyncioTestCase):
    def _driver(self, current_url: str) -> tuple[_TestDriver, Mock, AsyncMock]:
        driver = _TestDriver()
        driver.page = SimpleNamespace(
            evaluate=AsyncMock(return_value=current_url),
        )
        get = AsyncMock()
        logger = Mock()
        driver.get = get  # type: ignore[method-assign]
        driver.logger = logger
        return driver, logger, get

    async def test_navigation_is_debug_with_source_target_and_force_context(
        self,
    ) -> None:
        current_url = "https://e-hentai.org/?page=maintenance"
        driver, logger, get = self._driver(current_url)

        await driver.gohomepage(force=True)

        logger.debug.assert_called_once_with(
            "Navigate to homepage: source=%s://%s route=%s has_query=%s "
            "target=%s://%s route=%s has_query=%s forced=%s",
            "https",
            "e-hentai.org",
            "root",
            True,
            "https",
            "e-hentai.org",
            "root",
            False,
            True,
        )
        logger.info.assert_not_called()
        get.assert_awaited_once_with("https://e-hentai.org/")

    async def test_skipped_navigation_keeps_debug_context(self) -> None:
        current_url = "https://e-hentai.org"
        driver, logger, get = self._driver(current_url)

        await driver.gohomepage()

        logger.debug.assert_called_once_with(
            "Already on homepage: source=%s://%s route=%s has_query=%s "
            "target=%s://%s route=%s has_query=%s forced=%s",
            "https",
            "e-hentai.org",
            "root",
            False,
            "https",
            "e-hentai.org",
            "root",
            False,
            False,
        )
        logger.info.assert_not_called()
        get.assert_not_awaited()


class DriverTypedTimeoutPropagationTests(unittest.IsolatedAsyncioTestCase):
    def _driver(self) -> _TestDriver:
        driver = _TestDriver()
        driver.logger = Mock()
        return driver

    async def test_url_change_timeout_does_not_poll_or_sleep_again(self) -> None:
        driver = self._driver()
        timeout = ZendriverOperationTimeout(timeout_seconds=5)
        driver.page = SimpleNamespace(evaluate=AsyncMock(side_effect=timeout))

        with (
            patch(
                "hbrowser.gallery.driver_base.asyncio.sleep",
                new=AsyncMock(),
            ) as sleep,
            self.assertRaises(ZendriverOperationTimeout) as raised,
        ):
            await driver._wait_for_url_change("https://example.test/old")

        self.assertIs(raised.exception, timeout)
        driver.page.evaluate.assert_awaited_once_with("window.location.href")
        sleep.assert_not_awaited()

    async def test_get_poll_timeout_skips_final_delay(self) -> None:
        driver = self._driver()
        timeout = ZendriverOperationTimeout(timeout_seconds=5)
        driver.page = SimpleNamespace(
            evaluate=AsyncMock(
                side_effect=["https://example.test/old", timeout],
            ),
        )
        driver.myget = AsyncMock()

        with (
            patch(
                "hbrowser.gallery.driver_base.asyncio.sleep",
                new=AsyncMock(),
            ) as sleep,
            self.assertRaises(ZendriverOperationTimeout) as raised,
        ):
            await driver.get("https://example.test/new")

        self.assertIs(raised.exception, timeout)
        self.assertEqual(driver.page.evaluate.await_count, 2)
        driver.myget.assert_awaited_once_with("https://example.test/new")
        sleep.assert_not_awaited()

    async def test_same_url_page_wait_timeout_skips_final_delay(self) -> None:
        driver = self._driver()
        timeout = ZendriverOperationTimeout(timeout_seconds=5)
        driver.page = SimpleNamespace(
            evaluate=AsyncMock(return_value="https://example.test/same"),
            wait=AsyncMock(side_effect=timeout),
        )
        driver.myget = AsyncMock()

        with (
            patch(
                "hbrowser.gallery.driver_base.asyncio.sleep",
                new=AsyncMock(),
            ) as sleep,
            self.assertRaises(ZendriverOperationTimeout) as raised,
        ):
            await driver.get("https://example.test/same")

        self.assertIs(raised.exception, timeout)
        driver.page.wait.assert_awaited_once_with(1)
        sleep.assert_not_awaited()

    async def test_wait_invokes_mutation_once_and_stops_on_any_failure(self) -> None:
        driver = self._driver()
        driver.page = SimpleNamespace(
            evaluate=AsyncMock(return_value="https://example.test/old"),
            wait=AsyncMock(),
        )
        mutation = AsyncMock(side_effect=RuntimeError("outcome unknown"))

        with self.assertRaises(BrowserMutationOutcomeUnknownError) as raised:
            await driver.wait(
                mutation,
                ischangeurl=False,
                owner=driver.page,
                operation_timeout=1,
            )

        mutation.assert_awaited_once_with()
        driver.page.wait.assert_not_awaited()
        self.assertNotIn("outcome unknown", str(raised.exception))
        self.assertIsNone(raised.exception.__cause__)
        self.assertIsNone(raised.exception.__context__)

    async def test_wait_mutation_timeout_runs_no_page_probe_or_delay(self) -> None:
        driver = self._driver()
        timeout = ZendriverOperationTimeout(timeout_seconds=1)
        driver.page = SimpleNamespace(
            evaluate=AsyncMock(return_value="https://example.test/old"),
            wait=AsyncMock(),
        )
        mutation = AsyncMock(side_effect=timeout)

        with (
            patch(
                "hbrowser.gallery.driver_base.asyncio.sleep",
                new=AsyncMock(),
            ) as sleep,
            self.assertRaises(ZendriverOperationTimeout) as raised,
        ):
            await driver.wait(
                mutation,
                ischangeurl=False,
                owner=driver.page,
                operation_timeout=1,
            )

        self.assertIs(raised.exception, timeout)
        mutation.assert_awaited_once_with()
        driver.page.wait.assert_not_awaited()
        sleep.assert_not_awaited()

    async def test_login_mutation_failures_are_terminal_and_never_advance(
        self,
    ) -> None:
        stages = ("login_link", "username", "password", "submit")
        for failed_stage in stages:
            with self.subTest(failed_stage=failed_stage):
                driver = self._driver()
                driver.username = "private-user"
                driver.password = "private-password"
                driver.myget = AsyncMock()
                driver._handle_page_challenge = AsyncMock()  # type: ignore[method-assign]
                driver._wait_for_url_change = AsyncMock()  # type: ignore[method-assign]
                driver._handle_login_challenge = AsyncMock()  # type: ignore[method-assign]
                driver._verify_login_succeeded = AsyncMock()  # type: ignore[method-assign]
                driver.gohomepage = AsyncMock()  # type: ignore[method-assign]

                elements: dict[str, SimpleNamespace] = {
                    "login_link": SimpleNamespace(click=AsyncMock()),
                    "username": SimpleNamespace(send_keys=AsyncMock()),
                    "password": SimpleNamespace(send_keys=AsyncMock()),
                    "submit": SimpleNamespace(click=AsyncMock()),
                }
                mutation = (
                    elements[failed_stage].send_keys
                    if failed_stage in {"username", "password"}
                    else elements[failed_stage].click
                )
                mutation.side_effect = RuntimeError(
                    f"sensitive {failed_stage} protocol payload"
                )
                driver.page = SimpleNamespace(
                    select=AsyncMock(side_effect=elements.values()),
                    evaluate=AsyncMock(return_value="https://forums.e-hentai.org/"),
                )

                with (
                    patch(
                        "hbrowser.gallery.driver_base.detect_forums_auth_state",
                        new=AsyncMock(return_value=ForumsAuthState.GUEST),
                    ),
                    self.assertRaises(BrowserMutationOutcomeUnknownError) as raised,
                ):
                    await driver._login(None)

                mutation.assert_awaited_once()
                self.assertNotIn("sensitive", str(raised.exception))
                self.assertIsNone(raised.exception.__cause__)
                self.assertIsNone(raised.exception.__context__)
                driver._verify_login_succeeded.assert_not_awaited()
                driver.gohomepage.assert_not_awaited()

    async def test_identity_mutation_preserves_retired_error(self) -> None:
        driver = self._driver()
        retired = ZendriverOwnerRetiredError("generation retired")
        driver.page = SimpleNamespace(send=AsyncMock(side_effect=retired))
        result = FlareSolverrResult(cookies=[], user_agent="test-agent")

        with self.assertRaises(ZendriverOwnerRetiredError) as raised:
            await driver._apply_flaresolverr_result(result)

        self.assertIs(raised.exception, retired)
        driver.page.send.assert_awaited_once()
