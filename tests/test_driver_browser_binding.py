from __future__ import annotations

import unittest
from unittest.mock import AsyncMock, Mock, patch

from hbrowser import DriverBrowserBindingError
from hbrowser.gallery.driver_base import Driver


class _TestDriver(Driver):
    def _setname(self) -> str:
        return "E-Hentai"


class DriverBrowserBindingTests(unittest.IsolatedAsyncioTestCase):
    def test_new_driver_is_unbound_and_does_not_own_a_browser(self) -> None:
        driver = _TestDriver()

        self.assertFalse(driver.is_browser_bound)
        self.assertFalse(driver.owns_browser)

    def test_binding_sets_fixed_objects_and_builds_page_navigator(self) -> None:
        driver = _TestDriver()
        browser = object()
        page = object()
        navigator = AsyncMock()

        with patch(
            "hbrowser.gallery.driver_base.handle_ban_decorator",
            return_value=navigator,
        ) as decorate:
            driver.bind_existing_browser(browser, page)

        self.assertIs(driver.browser, browser)
        self.assertIs(driver.page, page)
        self.assertIs(driver.myget, navigator)
        self.assertTrue(driver.is_browser_bound)
        self.assertFalse(driver.owns_browser)
        decorate.assert_called_once_with(page)

    def test_identical_binding_is_idempotent(self) -> None:
        driver = _TestDriver()
        browser = object()
        page = object()

        with patch(
            "hbrowser.gallery.driver_base.handle_ban_decorator",
            return_value=AsyncMock(),
        ) as decorate:
            driver.bind_existing_browser(browser, page, owns_browser=True)
            original_navigator = driver.myget
            driver.bind_existing_browser(browser, page, owns_browser=True)

        self.assertIs(driver.myget, original_navigator)
        self.assertTrue(driver.owns_browser)
        decorate.assert_called_once_with(page)

    def test_rebinding_browser_page_or_ownership_is_rejected(self) -> None:
        original_browser = object()
        original_page = object()

        cases = (
            (object(), original_page, False),
            (original_browser, object(), False),
            (original_browser, original_page, True),
        )
        for browser, page, owns_browser in cases:
            with self.subTest(
                changes_browser=browser is not original_browser,
                changes_page=page is not original_page,
                changes_ownership=owns_browser,
            ):
                driver = _TestDriver()
                driver.bind_existing_browser(original_browser, original_page)

                with self.assertRaises(DriverBrowserBindingError):
                    driver.bind_existing_browser(
                        browser,
                        page,
                        owns_browser=owns_browser,
                    )

                self.assertIs(driver.browser, original_browser)
                self.assertIs(driver.page, original_page)
                self.assertFalse(driver.owns_browser)

    def test_none_browser_or_page_is_rejected_without_partial_binding(self) -> None:
        driver = _TestDriver()

        with self.assertRaisesRegex(ValueError, "browser must not be None"):
            driver.bind_existing_browser(None, object())
        with self.assertRaisesRegex(ValueError, "page must not be None"):
            driver.bind_existing_browser(object(), None)

        self.assertFalse(driver.is_browser_bound)
        self.assertIsNone(driver.browser)
        self.assertIsNone(driver.page)
        self.assertIsNone(driver.myget)

    def test_non_boolean_ownership_is_rejected(self) -> None:
        driver = _TestDriver()

        with self.assertRaisesRegex(TypeError, "owns_browser must be a bool"):
            driver.bind_existing_browser(
                object(),
                object(),
                owns_browser=1,  # type: ignore[arg-type]
            )

        self.assertFalse(driver.is_browser_bound)

    def test_legacy_partial_assignment_cannot_be_silently_replaced(self) -> None:
        driver = _TestDriver()
        driver.page = object()

        with self.assertRaises(DriverBrowserBindingError):
            driver.bind_existing_browser(object(), object())

        self.assertFalse(driver.is_browser_bound)

    async def test_init_browser_uses_an_owned_binding(self) -> None:
        driver = _TestDriver(headless=True)
        browser = object()
        page = object()
        navigator = AsyncMock()
        get = AsyncMock()
        driver.get = get  # type: ignore[method-assign]

        with (
            patch(
                "hbrowser.gallery.driver_base.create_browser",
                new=AsyncMock(return_value=(browser, page)),
            ) as create_browser,
            patch(
                "hbrowser.gallery.driver_base.handle_ban_decorator",
                return_value=navigator,
            ),
        ):
            await driver._init_browser()

        create_browser.assert_awaited_once_with(headless=True)
        self.assertIs(driver.browser, browser)
        self.assertIs(driver.page, page)
        self.assertIs(driver.myget, navigator)
        self.assertTrue(driver.owns_browser)
        get.assert_awaited_once_with(driver.url["Forums"])

    async def test_only_owner_closes_the_browser(self) -> None:
        browser = object()
        owner = _TestDriver()
        owner.bind_existing_browser(browser, object(), owns_browser=True)
        shared = _TestDriver()
        shared.bind_existing_browser(browser, object())

        with patch(
            "hbrowser.gallery.driver_base.stop_browser",
            new=AsyncMock(),
        ) as stop_browser:
            await shared.__aexit__(None, None, None)
            await owner.__aexit__(None, None, None)

        stop_browser.assert_awaited_once_with(browser)

    async def test_startup_failure_with_no_browser_has_quiet_cleanup(self) -> None:
        driver = _TestDriver()
        driver.logger = Mock()
        startup_error = RuntimeError("startup failed")

        with (
            patch.object(
                driver,
                "_init_browser",
                new=AsyncMock(side_effect=startup_error),
            ),
            patch.object(driver, "_save_page_diagnostic", new=AsyncMock()) as save,
            patch(
                "hbrowser.gallery.driver_base.stop_browser",
                new=AsyncMock(),
            ) as stop_browser,
            self.assertRaises(RuntimeError) as raised,
        ):
            await driver.__aenter__()

        self.assertIs(raised.exception, startup_error)
        save.assert_not_awaited()
        stop_browser.assert_not_awaited()
        driver.logger.info.assert_not_called()


if __name__ == "__main__":
    unittest.main()
