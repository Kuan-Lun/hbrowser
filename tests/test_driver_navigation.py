import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from hbrowser import BrowserMutationOutcomeUnknownError
from hbrowser.gallery.browser import FlareSolverrResult
from hbrowser.gallery.driver_base import Driver
from hbrowser.gallery.forums_auth import ForumsAuthState
from hbrowser.gallery.utils import (
    Deadline,
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

    async def test_get_delegates_to_lifecycle_navigation_without_polling(self) -> None:
        driver = self._driver()
        driver.page = SimpleNamespace(evaluate=AsyncMock())
        driver.myget = AsyncMock()

        await driver.get("https://example.test/new")

        driver.myget.assert_awaited_once_with("https://example.test/new")
        driver.page.evaluate.assert_not_awaited()

    async def test_get_forwards_a_shared_semantic_deadline(self) -> None:
        driver = self._driver()
        driver.page = SimpleNamespace(evaluate=AsyncMock())
        driver.myget = AsyncMock()
        deadline = Deadline.after(1)

        await driver.get("https://example.test/new", deadline=deadline)

        driver.myget.assert_awaited_once_with(
            "https://example.test/new",
            deadline=deadline,
        )

    async def test_get_preserves_lifecycle_navigation_timeout(self) -> None:
        driver = self._driver()
        timeout = ZendriverOperationTimeout(timeout_seconds=5)
        driver.page = SimpleNamespace(evaluate=AsyncMock())
        driver.myget = AsyncMock(side_effect=timeout)

        with (
            patch(
                "hbrowser.gallery.driver_base.asyncio.sleep",
                new=AsyncMock(),
            ) as sleep,
            self.assertRaises(ZendriverOperationTimeout) as raised,
        ):
            await driver.get("https://example.test/new")

        self.assertIs(raised.exception, timeout)
        driver.myget.assert_awaited_once_with("https://example.test/new")
        driver.page.evaluate.assert_not_awaited()
        sleep.assert_not_awaited()

    async def test_get_same_url_does_not_add_page_wait_or_delay(self) -> None:
        driver = self._driver()
        driver.page = SimpleNamespace(
            evaluate=AsyncMock(),
            wait=AsyncMock(),
        )
        driver.myget = AsyncMock()

        with patch(
            "hbrowser.gallery.driver_base.asyncio.sleep",
            new=AsyncMock(),
        ) as sleep:
            await driver.get("https://example.test/same")

        driver.myget.assert_awaited_once_with("https://example.test/same")
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
