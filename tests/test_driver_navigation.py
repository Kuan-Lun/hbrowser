import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

from hbrowser.gallery.driver_base import Driver


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
