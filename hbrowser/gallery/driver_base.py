"""Driver 基類"""

from __future__ import annotations

import asyncio
import os
from abc import ABC, abstractmethod
from pathlib import Path
from random import random
from typing import Any, Self, cast
from urllib.parse import urlsplit

from zendriver import cdp

from ..exceptions import (
    BrowserIdentityApplyException,
    BrowserMutationOutcomeUnknownError,
    DriverBrowserBindingError,
    LoginFailedException,
)
from .browser import (
    FlareSolverrClient,
    FlareSolverrResult,
    FlareSolverrSessionScope,
    FlareSolverrSolveReceipt,
    create_browser,
    get_flaresolverr_url,
    should_use_flaresolverr,
    stop_browser,
)
from .browser.ban_handler import handle_ban_decorator
from .captcha import CaptchaDetector, LoginChallengeHandler, PageChallengeHandler
from .forums_auth import ForumsAuthState, detect_forums_auth_state
from .utils import (
    ZendriverOperationTimeout,
    get_log_dir,
    is_browser_generation_error,
    log_context,
    matchurl,
    setup_logger,
    wait_for_zendriver,
    write_page_diagnostic,
)
from .utils.mutation import wait_for_zendriver_mutation

_PAGE_DIAGNOSTIC_CAPTURE_TIMEOUT_SECONDS = 5.0
_NAVIGATION_READ_TIMEOUT_SECONDS = 5.0
_PAGE_MUTATION_TIMEOUT_SECONDS = 15.0
_IDENTITY_MUTATION_TIMEOUT_SECONDS = 15.0
_LOGIN_ELEMENT_WAIT_TIMEOUT_SECONDS = 10.0
_LOGIN_ELEMENT_WATCHDOG_TIMEOUT_SECONDS = 12.0
_SAFE_NAVIGATION_ROUTES = {
    "favorites.php": "favorites",
    "g": "gallery",
    "hentaiathome.php": "hath",
    "home.php": "home",
    "news.php": "news",
    "popular": "popular",
    "tag": "tag",
    "toplist.php": "toplist",
    "uploader": "uploader",
    "watched": "watched",
}


def _redacted_url_context(url: str) -> tuple[str, str, str, bool]:
    """Return useful navigation context without path values or query contents."""
    try:
        parsed_url = urlsplit(url)
        scheme = parsed_url.scheme or "relative"
        hostname = parsed_url.hostname or "unknown"
    except ValueError:
        return "invalid", "invalid", "invalid", False

    path_parts = [part for part in parsed_url.path.split("/") if part]
    if not path_parts:
        route = "root"
    else:
        route = _SAFE_NAVIGATION_ROUTES.get(path_parts[0].casefold(), "other")
    return scheme, hostname, route, bool(parsed_url.query)


class _DriverTurnstileSolver:
    """Adapt a FlareSolverr session to the login challenge handler."""

    def __init__(self, driver: Driver, session: FlareSolverrSessionScope) -> None:
        self._driver = driver
        self._session = session

    async def solve_turnstile(
        self,
        url: str,
        *,
        tabs: int,
        timeout_ms: int = 30_000,
    ) -> str | None:
        receipt = await self._session.solve_turnstile(
            url,
            tabs=tabs,
            timeout_ms=timeout_ms,
        )
        await self._driver._apply_flaresolverr_receipt(receipt)
        self._session.mark_identity_applied(receipt)
        return receipt.result.turnstile_token


class Driver(ABC):
    """
    Gallery Driver 抽象基類
    """

    @abstractmethod
    def _setname(self) -> str:
        """設定網站名稱"""
        pass

    def __init__(
        self,
        headless: bool = True,
        flaresolverr_session_attempts: int = 3,
        captcha_manual_timeout: int = 180,
        turnstile_tabs: int = 15,
    ) -> None:
        if (
            type(flaresolverr_session_attempts) is not int
            or flaresolverr_session_attempts < 1
        ):
            raise ValueError("flaresolverr_session_attempts must be a positive integer")
        if captcha_manual_timeout <= 0:
            raise ValueError("captcha_manual_timeout must be greater than zero")
        if turnstile_tabs < 1:
            raise ValueError("turnstile_tabs must be at least 1")

        def seturl() -> dict[str, str]:
            url: dict[str, str] = dict()
            url["My Home"] = "https://e-hentai.org/home.php"
            url["E-Hentai"] = "https://e-hentai.org/"
            url["ExHentai"] = "https://exhentai.org/"
            url["Forums"] = "https://forums.e-hentai.org/"
            return url

        self.logger = setup_logger(__name__)
        self.username = os.getenv("EH_USERNAME")
        self.password = os.getenv("EH_PASSWORD")
        self.url = seturl()
        self.name = self._setname()
        self.headless = headless
        self.browser: Any = None
        self.page: Any = None
        self.myget: Any = None
        self._browser_bound = False
        self._owns_browser = False
        self.flaresolverr_session_attempts = flaresolverr_session_attempts
        self.captcha_manual_timeout = captcha_manual_timeout
        self.turnstile_tabs = turnstile_tabs
        self.captcha_detector = CaptchaDetector()

    @property
    def is_browser_bound(self) -> bool:
        """Whether this driver has an immutable browser/page binding."""

        return self._browser_bound

    @property
    def owns_browser(self) -> bool:
        """Whether this driver is responsible for closing its bound browser."""

        return self._owns_browser

    def bind_existing_browser(
        self,
        browser: Any,
        page: Any,
        *,
        owns_browser: bool = False,
    ) -> None:
        """Bind this driver to one browser and page for its complete lifetime.

        A second call with the identical objects and ownership is a no-op.  A
        caller may not retarget a bound driver or transfer browser ownership;
        it must construct a new driver instead.
        """

        if browser is None:
            raise ValueError("browser must not be None")
        if page is None:
            raise ValueError("page must not be None")
        if type(owns_browser) is not bool:
            raise TypeError("owns_browser must be a bool")

        if self._browser_bound:
            if (
                self.browser is browser
                and self.page is page
                and self._owns_browser is owns_browser
            ):
                return
            raise DriverBrowserBindingError(
                "driver is already bound to a different browser, page, or "
                "ownership mode"
            )

        if self.browser is not None or self.page is not None or self.myget is not None:
            raise DriverBrowserBindingError(
                "driver browser state was assigned without bind_existing_browser"
            )

        self.browser = browser
        self.page = page
        self.myget = handle_ban_decorator(page)
        self._owns_browser = owns_browser
        self._browser_bound = True

    async def _init_browser(self) -> None:
        if self._browser_bound:
            raise DriverBrowserBindingError("driver already has a browser binding")
        browser, page = await create_browser(headless=self.headless)
        self.bind_existing_browser(browser, page, owns_browser=True)
        await self.get(self.url["Forums"])

    async def __aenter__(self) -> Self:
        try:
            await self._init_browser()
            await self.login()
            await self.gohomepage()
        except BaseException as e:
            # __aexit__ 不會在 __aenter__ 失敗時自動被呼叫，
            # 這裡手動觸發以確保瀏覽器與子程序資源不會洩漏。
            await self.__aexit__(type(e), e, e.__traceback__)
            raise
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        del exc_tb
        primary_error = exc_val if isinstance(exc_val, BaseException) else None
        diagnostic_error: BaseException | None = None
        if exc_type is not None:
            error_type = getattr(exc_type, "__name__", type(exc_val).__name__)
            cause = getattr(exc_val, "__cause__", None) or getattr(
                exc_val,
                "__context__",
                None,
            )
            self.logger.error(
                "Browser session exiting after error: error_type=%s cause_type=%s",
                error_type,
                type(cause).__name__ if cause is not None else "none",
            )
            may_capture_diagnostic = not isinstance(
                exc_val, BaseException
            ) or not is_browser_generation_error(exc_val)
            if self.page is not None and may_capture_diagnostic:
                try:
                    await self.save_page_diagnostic("driver_error")
                except BaseException as error:
                    diagnostic_error = error
        if not self._owns_browser or self.browser is None:
            if isinstance(diagnostic_error, asyncio.CancelledError):
                if primary_error is not None:
                    diagnostic_error.add_note(
                        "Browser session was already exiting after: "
                        f"{type(primary_error).__name__}"
                    )
                raise diagnostic_error
            if primary_error is not None and diagnostic_error is not None:
                primary_error.add_note(
                    "Driver diagnostic also failed: "
                    f"{type(diagnostic_error).__name__}: {diagnostic_error}"
                )
            elif diagnostic_error is not None:
                raise diagnostic_error
            return
        self.logger.info("Closing browser")
        cleanup_error: BaseException | None = None
        try:
            await stop_browser(self.browser)
        except BaseException as error:
            cleanup_error = error
            self.logger.warning(
                "Failed to close browser cleanly: error_type=%s",
                type(error).__name__,
            )

        secondary_errors = tuple(
            error for error in (diagnostic_error, cleanup_error) if error is not None
        )
        cancellation_error = next(
            (
                error
                for error in secondary_errors
                if isinstance(error, asyncio.CancelledError)
            ),
            None,
        )
        if cancellation_error is not None:
            if primary_error is not None:
                cancellation_error.add_note(
                    "Browser session was already exiting after: "
                    f"{type(primary_error).__name__}"
                )
            for secondary_error in secondary_errors:
                if secondary_error is not cancellation_error:
                    cancellation_error.add_note(
                        "Additional driver shutdown failure: "
                        f"{type(secondary_error).__name__}: {secondary_error}"
                    )
            raise cancellation_error
        if primary_error is not None:
            for secondary_error in secondary_errors:
                primary_error.add_note(
                    "Driver shutdown also failed: "
                    f"{type(secondary_error).__name__}: {secondary_error}"
                )
            return
        if secondary_errors:
            primary, *secondary = secondary_errors
            for secondary_error in secondary:
                primary.add_note(
                    "Additional driver shutdown failure: "
                    f"{type(secondary_error).__name__}: {secondary_error}"
                )
            raise primary

    @staticmethod
    def _write_page_diagnostic(kind: str, content: str) -> Path:
        return write_page_diagnostic(get_log_dir(), kind, content)

    async def save_page_diagnostic(
        self,
        kind: str,
        content: str | None = None,
    ) -> Path | None:
        """Persist a bounded, redacted HTML snapshot for page-state failures."""

        if content is None:
            try:
                if self.page is None:
                    raise RuntimeError("browser page is not available")
                content = cast(
                    str,
                    await wait_for_zendriver(
                        self.page.get_content(),
                        timeout=_PAGE_DIAGNOSTIC_CAPTURE_TIMEOUT_SECONDS,
                        owner=self.page,
                    ),
                )
            except ZendriverOperationTimeout:
                raise
            except Exception as error:
                if is_browser_generation_error(error):
                    raise
                self.logger.warning(
                    "Failed to capture %s page diagnostic: error_type=%s",
                    kind,
                    type(error).__name__,
                )
                return None

        try:
            path = await asyncio.to_thread(
                self._write_page_diagnostic,
                kind,
                content,
            )
        except Exception as error:
            self.logger.warning(
                "Failed to save %s page diagnostic: error_type=%s",
                kind,
                type(error).__name__,
            )
            return None

        self.logger.warning(
            "Page diagnostic saved: kind=%s path=%s",
            kind,
            path,
        )
        return path

    async def gohomepage(self, force: bool = False) -> None:
        url = self.url[self.name]
        current_url = await wait_for_zendriver(
            self.page.evaluate("window.location.href"),
            timeout=_NAVIGATION_READ_TIMEOUT_SECONDS,
            owner=self.page,
        )
        source_scheme, source_host, source_route, source_has_query = (
            _redacted_url_context(current_url)
        )
        target_scheme, target_host, target_route, target_has_query = (
            _redacted_url_context(url)
        )
        if force or not matchurl(current_url, url):
            self.logger.debug(
                "Navigate to homepage: source=%s://%s route=%s has_query=%s "
                "target=%s://%s route=%s has_query=%s forced=%s",
                source_scheme,
                source_host,
                source_route,
                source_has_query,
                target_scheme,
                target_host,
                target_route,
                target_has_query,
                force,
            )
            await self.get(url)
        else:
            self.logger.debug(
                "Already on homepage: source=%s://%s route=%s has_query=%s "
                "target=%s://%s route=%s has_query=%s forced=%s",
                source_scheme,
                source_host,
                source_route,
                source_has_query,
                target_scheme,
                target_host,
                target_route,
                target_has_query,
                force,
            )

    async def _wait_for_url_change(
        self, old_url: str, *, deadline_seconds: float = 10.0
    ) -> None:
        """Best-effort wait for navigation; gives up silently on timeout."""
        deadline = asyncio.get_event_loop().time() + deadline_seconds
        while True:
            try:
                current_url = await wait_for_zendriver(
                    self.page.evaluate("window.location.href"),
                    timeout=_NAVIGATION_READ_TIMEOUT_SECONDS,
                    owner=self.page,
                )
            except ZendriverOperationTimeout:
                raise
            except TimeoutError:
                return
            if current_url != old_url:
                return
            if asyncio.get_event_loop().time() >= deadline:
                return
            await asyncio.sleep(0.1)

    async def find_element_chain(self, *selectors: str) -> Any:
        element: Any = self.page
        for selector in selectors:
            element = await wait_for_zendriver(
                element.query_selector(selector),
                timeout=_NAVIGATION_READ_TIMEOUT_SECONDS,
                owner=element,
            )
        return element

    async def get(self, url: str) -> None:
        current_url = await wait_for_zendriver(
            self.page.evaluate("window.location.href"),
            timeout=_NAVIGATION_READ_TIMEOUT_SECONDS,
            owner=self.page,
        )
        scheme, hostname, route, has_query = _redacted_url_context(url)
        self.logger.debug(
            "Navigate to URL: scheme=%s host=%s route=%s has_query=%s",
            scheme,
            hostname,
            route,
            has_query,
        )
        is_new_url = not matchurl(url, current_url)
        await self.myget(url)
        if is_new_url:
            try:
                deadline = asyncio.get_event_loop().time() + 10
                while matchurl(
                    await wait_for_zendriver(
                        self.page.evaluate("window.location.href"),
                        timeout=_NAVIGATION_READ_TIMEOUT_SECONDS,
                        owner=self.page,
                    ),
                    current_url,
                ):
                    if asyncio.get_event_loop().time() >= deadline:
                        break
                    await asyncio.sleep(0.1)
            except ZendriverOperationTimeout:
                raise
            except TimeoutError:
                pass
        else:
            try:
                await wait_for_zendriver(
                    self.page.wait(1),
                    timeout=_NAVIGATION_READ_TIMEOUT_SECONDS,
                    owner=self.page,
                )
            except ZendriverOperationTimeout:
                raise
            except TimeoutError:
                pass
        await asyncio.sleep(3 * random())

    async def wait(
        self,
        fun: Any,
        ischangeurl: bool,
        sleeptime: int = -1,
        *,
        owner: Any,
        operation_timeout: float,
    ) -> None:
        """執行 async 函數並等待頁面變化。

        Args:
            fun: 要執行的 async callable
            ischangeurl: 是否等待 URL 變化
            sleeptime: 等待時間（秒），-1 表示隨機等待
            owner: 執行操作的固定 Zendriver transport
            operation_timeout: 操作 watchdog 秒數
        """
        old_url = await wait_for_zendriver(
            self.page.evaluate("window.location.href"),
            timeout=_NAVIGATION_READ_TIMEOUT_SECONDS,
            owner=self.page,
        )

        # This callable may mutate remote state. Once invoked, any failure has
        # an unknown outcome and must never cause the operation to be replayed.
        await wait_for_zendriver_mutation(
            fun(),
            timeout=operation_timeout,
            owner=owner,
            operation="Driver mutation",
        )

        if ischangeurl:
            await self._wait_for_url_change(old_url)
        else:
            try:
                await wait_for_zendriver(
                    self.page.wait(1),
                    timeout=_NAVIGATION_READ_TIMEOUT_SECONDS,
                    owner=self.page,
                )
            except ZendriverOperationTimeout:
                raise
            except TimeoutError:
                pass

        if sleeptime < 0:
            await asyncio.sleep(3 * random())
        else:
            await asyncio.sleep(sleeptime)

    async def _handle_login_challenge(
        self,
        flaresolverr_session: FlareSolverrSessionScope | None,
    ) -> None:
        automatic_solver = (
            _DriverTurnstileSolver(self, flaresolverr_session)
            if flaresolverr_session is not None
            else None
        )
        handler = LoginChallengeHandler(
            detector=self.captcha_detector,
            logger=self.logger,
            headless=self.headless,
            manual_timeout=self.captcha_manual_timeout,
            turnstile_tabs=self.turnstile_tabs,
            automatic_solver=automatic_solver,
        )
        await handler.resolve(self.page)

    async def _handle_page_challenge(
        self,
        url: str,
        flaresolverr_session: FlareSolverrSessionScope | None,
        *,
        detect_timeout: float = 3.0,
    ) -> None:
        """Resolve a page-level challenge without replacing browser or route."""
        handler = PageChallengeHandler(
            detector=self.captcha_detector,
            logger=self.logger,
            headless=self.headless,
            manual_timeout=self.captcha_manual_timeout,
            automatic_solver=flaresolverr_session,
            apply_identity=self._apply_flaresolverr_receipt,
            navigate=self.myget,
            save_diagnostic=self.save_page_diagnostic,
        )
        await handler.resolve(self.page, url, detect_timeout=detect_timeout)

    async def _apply_flaresolverr_receipt(
        self,
        receipt: FlareSolverrSolveReceipt,
    ) -> None:
        await self._apply_flaresolverr_result(receipt.result)

    async def _apply_flaresolverr_result(
        self,
        result: FlareSolverrResult,
    ) -> None:
        if result.user_agent:
            identity_failed = False
            try:
                await wait_for_zendriver_mutation(
                    self.page.send(
                        cdp.network.set_user_agent_override(
                            user_agent=result.user_agent
                        )
                    ),
                    timeout=_IDENTITY_MUTATION_TIMEOUT_SECONDS,
                    owner=self.page,
                    operation="Browser user-agent override",
                )
            except BrowserMutationOutcomeUnknownError:
                identity_failed = True
            if identity_failed:
                raise BrowserIdentityApplyException(
                    "Could not apply the FlareSolverr browser identity"
                )
        cookie_params = result.to_cdp_cloudflare_cookie_params()
        if not cookie_params:
            return
        cookie_apply_failed = False
        try:
            await wait_for_zendriver_mutation(
                self.page.send(cdp.network.set_cookies(cookie_params)),
                timeout=_IDENTITY_MUTATION_TIMEOUT_SECONDS,
                owner=self.page,
                operation="Browser cookie replacement",
            )
        except BrowserMutationOutcomeUnknownError:
            cookie_apply_failed = True
        if cookie_apply_failed:
            # CDP protocol errors may contain cookie values in their params.
            raise BrowserIdentityApplyException(
                "Could not apply the FlareSolverr Cloudflare cookies"
            )

    async def login(self) -> None:
        """Log in, sharing one FlareSolverr browser across both challenges."""
        with log_context(activity="Login"):
            flaresolverr_url = get_flaresolverr_url()
            if not flaresolverr_url or not should_use_flaresolverr():
                await self._login(None)
                return

            async with FlareSolverrClient(flaresolverr_url) as client:
                async with client.session(
                    max_attempts=self.flaresolverr_session_attempts
                ) as session:
                    await self._login(session)

    async def _login(
        self,
        flaresolverr_session: FlareSolverrSessionScope | None,
    ) -> None:
        """透過 Forums 頁面登入。

        流程：
        1. 進入 Forums 首頁（Cloudflare 驗證在此發生）
        2. 點擊 "Log In" 連結進入登入頁面
        3. 輸入帳號密碼並點擊 "Log me in"
        4. 驗證登入成功後前往主頁
        """
        self.logger.info("Signing in")

        await self.myget(self.url["Forums"])
        await self._handle_page_challenge(
            self.url["Forums"],
            flaresolverr_session,
        )

        auth_state = await detect_forums_auth_state(self.page)
        if auth_state is ForumsAuthState.AUTHENTICATED:
            self.logger.info("Already signed in")
            await self.gohomepage()
            return
        if auth_state is not ForumsAuthState.GUEST:
            raise LoginFailedException(
                "Forums page has no trustworthy guest or member login state"
            )

        self.logger.debug("Clicking 'Log In' link on Forums page")
        login_link = await wait_for_zendriver(
            self.page.select(
                "#userlinksguest a[href*='act=Login&CODE=00']",
                timeout=_LOGIN_ELEMENT_WAIT_TIMEOUT_SECONDS,
            ),
            timeout=_LOGIN_ELEMENT_WATCHDOG_TIMEOUT_SECONDS,
            owner=self.page,
        )
        old_url = await wait_for_zendriver(
            self.page.evaluate("window.location.href"),
            timeout=_NAVIGATION_READ_TIMEOUT_SECONDS,
            owner=self.page,
        )
        await wait_for_zendriver_mutation(
            login_link.click(),
            timeout=_PAGE_MUTATION_TIMEOUT_SECONDS,
            owner=login_link,
            operation="Forums login-link click",
        )
        await self._wait_for_url_change(old_url)

        username_input = await wait_for_zendriver(
            self.page.select(
                "[name='UserName']", timeout=_LOGIN_ELEMENT_WAIT_TIMEOUT_SECONDS
            ),
            timeout=_LOGIN_ELEMENT_WATCHDOG_TIMEOUT_SECONDS,
            owner=self.page,
        )
        await wait_for_zendriver_mutation(
            username_input.send_keys(self.username),
            timeout=_PAGE_MUTATION_TIMEOUT_SECONDS,
            owner=username_input,
            operation="Forums username input",
        )

        password_input = await wait_for_zendriver(
            self.page.select(
                "[name='PassWord']", timeout=_LOGIN_ELEMENT_WAIT_TIMEOUT_SECONDS
            ),
            timeout=_LOGIN_ELEMENT_WATCHDOG_TIMEOUT_SECONDS,
            owner=self.page,
        )
        await wait_for_zendriver_mutation(
            password_input.send_keys(self.password),
            timeout=_PAGE_MUTATION_TIMEOUT_SECONDS,
            owner=password_input,
            operation="Forums password input",
        )

        await self._handle_login_challenge(flaresolverr_session)

        old_url = await wait_for_zendriver(
            self.page.evaluate("window.location.href"),
            timeout=_NAVIGATION_READ_TIMEOUT_SECONDS,
            owner=self.page,
        )
        submit_button = await wait_for_zendriver(
            self.page.select(
                "input[type='submit'][value='Log me in']",
                timeout=_LOGIN_ELEMENT_WAIT_TIMEOUT_SECONDS,
            ),
            timeout=_LOGIN_ELEMENT_WATCHDOG_TIMEOUT_SECONDS,
            owner=self.page,
        )
        await wait_for_zendriver_mutation(
            submit_button.click(),
            timeout=_PAGE_MUTATION_TIMEOUT_SECONDS,
            owner=submit_button,
            operation="Forums login-form submission",
        )
        self.logger.debug("'Log me in' button clicked, waiting for redirect...")

        await self._wait_for_url_change(old_url)

        await self._verify_login_succeeded(flaresolverr_session)
        self.logger.info("Signed in")

        await self.gohomepage()

    async def _verify_login_succeeded(
        self,
        flaresolverr_session: FlareSolverrSessionScope | None,
    ) -> None:
        """確認登入是否成功：重新檢查 Forums 頁面的 guest 標記是否已消失。

        提交登入表單後可能再次跳出 Cloudflare 驗證，因此這裡會再走一次
        驗證碼偵測流程，而不是直接信任「URL 有變化」就代表登入成功。
        """
        await self.myget(self.url["Forums"])
        await self._handle_page_challenge(
            self.url["Forums"],
            flaresolverr_session,
        )

        auth_state = await detect_forums_auth_state(self.page)
        if auth_state is not ForumsAuthState.AUTHENTICATED:
            raise LoginFailedException(
                "Forums did not show an authenticated member state after login "
                f"(state: {auth_state.value})"
            )
