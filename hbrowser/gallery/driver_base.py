"""Driver 基類"""

from __future__ import annotations

import asyncio
import os
from abc import ABC, abstractmethod
from random import random
from typing import Any, Self

from zendriver import cdp

from ..exceptions import BrowserIdentityApplyException, LoginFailedException
from .browser import (
    DriverRestartRotator,
    FlareSolverrClient,
    FlareSolverrResult,
    FlareSolverrSession,
    ProxyRotator,
    create_browser,
    get_flaresolverr_url,
    should_use_flaresolverr,
    stop_browser,
)
from .browser.ban_handler import handle_ban_decorator
from .captcha import CaptchaDetector, LoginChallengeHandler
from .forums_auth import ForumsAuthState, detect_forums_auth_state
from .utils import get_log_dir, matchurl, setup_logger


class _DriverTurnstileSolver:
    """Adapt a FlareSolverr session to the login challenge handler."""

    def __init__(self, driver: Driver, session: FlareSolverrSession) -> None:
        self._driver = driver
        self._session = session

    async def solve_turnstile(
        self,
        url: str,
        *,
        tabs: int,
        timeout_ms: int = 30_000,
    ) -> str | None:
        result = await self._session.solve_turnstile(
            url,
            tabs=tabs,
            timeout_ms=timeout_ms,
        )
        await self._driver._apply_flaresolverr_result(result)
        return result.turnstile_token


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
        proxy_rotator: ProxyRotator | None = None,
        max_captcha_retries: int = 3,
        captcha_manual_timeout: int = 180,
        turnstile_tabs: int = 15,
    ) -> None:
        if max_captcha_retries < 1:
            raise ValueError("max_captcha_retries must be at least 1")
        if captcha_manual_timeout <= 0:
            raise ValueError("captcha_manual_timeout must be greater than zero")
        if turnstile_tabs < 1:
            raise ValueError("turnstile_tabs must be at least 1")

        def seturl() -> dict[str, str]:
            url: dict[str, str] = dict()
            url["My Home"] = "https://e-hentai.org/home.php"
            url["E-Hentai"] = "https://e-hentai.org/"
            url["ExHentai"] = "https://exhentai.org/"
            url["HentaiVerse"] = "https://hentaiverse.org"
            url["HentaiVerse isekai"] = "https://hentaiverse.org/isekai/"
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
        self.proxy_rotator = proxy_rotator or DriverRestartRotator()
        self.max_captcha_retries = max_captcha_retries
        self.captcha_manual_timeout = captcha_manual_timeout
        self.turnstile_tabs = turnstile_tabs
        self.captcha_detector = CaptchaDetector()

    async def _init_browser(self) -> None:
        self.browser, self.page = await create_browser(headless=self.headless)
        self.myget = handle_ban_decorator(self.page)
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
        if exc_type:
            self.logger.error(f"Exception occurred: {exc_type.__name__}: {exc_val}")
            try:
                error_file = get_log_dir() / "error.txt"
                error_file.write_text(await self.page.get_content(), errors="ignore")
                self.logger.debug(f"Error page saved to: {error_file}")
            except Exception:
                self.logger.error("Failed to save error page (browser session invalid)")
        self.logger.info("Closing browser")
        try:
            await stop_browser(self.browser)
        except Exception:
            pass

    async def gohomepage(self, force: bool = False) -> None:
        url = self.url[self.name]
        current_url = await self.page.evaluate("window.location.href")
        if force or not matchurl(current_url, url):
            self.logger.info(f"Navigate to homepage: {url}")
            await self.get(url)
        else:
            self.logger.debug("Already on homepage, no navigation needed")

    async def find_element_chain(self, *selectors: str) -> Any:
        element: Any = self.page
        for selector in selectors:
            element = await element.query_selector(selector)
        return element

    async def get(self, url: str) -> None:
        current_url = await self.page.evaluate("window.location.href")
        self.logger.debug(f"Navigate to URL: {url}")
        is_new_url = not matchurl(url, current_url)
        await self.myget(url)
        if is_new_url:
            try:
                deadline = asyncio.get_event_loop().time() + 10
                while matchurl(
                    await self.page.evaluate("window.location.href"), current_url
                ):
                    if asyncio.get_event_loop().time() >= deadline:
                        break
                    await asyncio.sleep(0.1)
            except TimeoutError:
                pass
        else:
            await self.page.wait(1)
        await asyncio.sleep(3 * random())

    async def wait(
        self,
        fun: Any,
        ischangeurl: bool,
        sleeptime: int = -1,
    ) -> None:
        """執行 async 函數並等待頁面變化。

        Args:
            fun: 要執行的 async callable
            ischangeurl: 是否等待 URL 變化
            sleeptime: 等待時間（秒），-1 表示隨機等待
        """
        old_url = await self.page.evaluate("window.location.href")

        max_retries = 3
        for attempt in range(max_retries):
            try:
                await fun()
                break
            except Exception:
                if attempt == max_retries - 1:
                    raise
                await asyncio.sleep(0.5)

        if ischangeurl:
            deadline = asyncio.get_event_loop().time() + 10
            while (
                await self.page.evaluate("window.location.href") == old_url
                and asyncio.get_event_loop().time() < deadline
            ):
                await asyncio.sleep(0.1)
        else:
            await self.page.wait(1)

        if sleeptime < 0:
            await asyncio.sleep(3 * random())
        else:
            await asyncio.sleep(sleeptime)

    async def _rotate_proxy(self) -> None:
        self.browser, self.page = await self.proxy_rotator.rotate(
            self.browser, self.headless
        )
        self.myget = handle_ban_decorator(self.page)

    async def _handle_login_challenge(
        self,
        flaresolverr_session: FlareSolverrSession | None,
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

    async def _apply_flaresolverr_result(
        self,
        result: FlareSolverrResult,
    ) -> None:
        if result.user_agent:
            try:
                await self.page.send(
                    cdp.network.set_user_agent_override(user_agent=result.user_agent)
                )
            except Exception:
                raise BrowserIdentityApplyException(
                    "Could not apply the FlareSolverr browser identity"
                ) from None
        cookie_params = result.to_cdp_cloudflare_cookie_params()
        if not cookie_params:
            return
        try:
            await self.page.send(cdp.network.set_cookies(cookie_params))
        except Exception:
            # CDP protocol errors may contain cookie values in their params.
            raise BrowserIdentityApplyException(
                "Could not apply the FlareSolverr Cloudflare cookies"
            ) from None

    async def _try_flaresolverr(
        self,
        url: str,
        flaresolverr_session: FlareSolverrSession,
    ) -> bool:
        """嘗試用 FlareSolverr 自動解決 Cloudflare managed challenge。

        Returns:
            是否成功解決（解出 cookie 並重新導航後挑戰已消失）
        """
        self.logger.info("Trying FlareSolverr to solve managed challenge")
        try:
            result = await flaresolverr_session.get(url)
        except Exception as error:
            self.logger.warning(
                "FlareSolverr managed-challenge request failed (%s)",
                type(error).__name__,
            )
            return False

        # Treat the identity transition as fail-closed. Continuing after only
        # the UA or only the cookies were applied would create a fingerprint
        # that cannot match the clearance FlareSolverr obtained.
        await self._apply_flaresolverr_result(result)

        await self.myget(url)
        det = await self.captcha_detector.detect(self.page, timeout=3.0)
        if det.kind == "none":
            self.logger.info("FlareSolverr resolved the challenge")
            return True

        self.logger.warning(
            "FlareSolverr identity did not clear the challenge; "
            "keeping that identity for manual resolution"
        )
        return False

    async def detect_and_solve_with_rotation(
        self,
        url: str,
        detect_timeout: float = 3.0,
        flaresolverr_session: FlareSolverrSession | None = None,
    ) -> None:
        """檢測 Cloudflare 驗證並等待使用者手動解決，失敗時自動輪換代理重試。

        Args:
            url: 要存取的 URL（輪換代理後需重新導航）
            detect_timeout: 驗證碼檢測超時時間（秒）

        Raises:
            Exception: 所有重試都失敗後拋出
        """
        for attempt in range(1, self.max_captcha_retries + 1):
            det = await self.captcha_detector.detect(self.page, timeout=detect_timeout)
            if det.kind == "none":
                self.logger.info("No challenge detected")
                return

            self.logger.warning(
                f"Challenge detected: {det.kind} "
                f"(attempt {attempt}/{self.max_captcha_retries})"
            )

            if det.kind == "cf_managed_challenge" and flaresolverr_session is not None:
                if await self._try_flaresolverr(url, flaresolverr_session):
                    return

            challenge_page_path = get_log_dir() / "challenge_page.html"
            challenge_page_path.write_text(
                await self.page.get_content(), errors="ignore"
            )
            self.logger.debug(f"Challenge page saved to: {challenge_page_path}")

            if self.headless:
                self.logger.warning(
                    "Headless mode active; no browser window for manual solving. "
                    f"Skipping manual wait (attempt {attempt}/{self.max_captcha_retries})"
                )
            else:
                self.logger.info(
                    "Please solve the challenge manually in the browser. "
                    f"Waiting up to {self.captcha_manual_timeout} seconds..."
                )

                start_time = asyncio.get_event_loop().time()
                while (
                    asyncio.get_event_loop().time() - start_time
                    < self.captcha_manual_timeout
                ):
                    current_det = await self.captcha_detector.detect(
                        self.page, timeout=1.0
                    )
                    if current_det.kind == "none":
                        self.logger.info("Challenge resolved successfully")
                        return
                    await asyncio.sleep(5)

                self.logger.warning(
                    f"Challenge not resolved within {self.captcha_manual_timeout}s "
                    f"(attempt {attempt}/{self.max_captcha_retries})"
                )
            if attempt < self.max_captcha_retries:
                await self._rotate_proxy()
                await self.myget(url)

        raise Exception(
            f"Failed to resolve captcha after {self.max_captcha_retries} attempts "
            f"with proxy rotation"
        )

    async def login(self) -> None:
        """Log in, sharing one FlareSolverr browser across both challenges."""
        flaresolverr_url = get_flaresolverr_url()
        if not flaresolverr_url or not should_use_flaresolverr():
            await self._login(None)
            return

        async with FlareSolverrClient(flaresolverr_url) as client:
            try:
                session = await client.create_session()
            except Exception as error:
                self.logger.warning(
                    "Could not create FlareSolverr session; "
                    "falling back to browser interaction (%s)",
                    type(error).__name__,
                )
                await self._login(None)
                return

            try:
                await self._login(session)
            finally:
                await client.destroy_session(session)

    async def _login(
        self,
        flaresolverr_session: FlareSolverrSession | None,
    ) -> None:
        """透過 Forums 頁面登入。

        流程：
        1. 進入 Forums 首頁（Cloudflare 驗證在此發生）
        2. 點擊 "Log In" 連結進入登入頁面
        3. 輸入帳號密碼並點擊 "Log me in"
        4. 驗證登入成功後前往主頁
        """
        self.logger.info("Starting login process")

        await self.myget(self.url["Forums"])
        await self.detect_and_solve_with_rotation(
            self.url["Forums"],
            flaresolverr_session=flaresolverr_session,
        )

        auth_state = await detect_forums_auth_state(self.page)
        if auth_state is ForumsAuthState.AUTHENTICATED:
            self.logger.info("Already logged in, skipping login")
            await self.gohomepage()
            return
        if auth_state is not ForumsAuthState.GUEST:
            raise LoginFailedException(
                "Forums page has no trustworthy guest or member login state"
            )

        self.logger.info("Clicking 'Log In' link on Forums page")
        login_link = await self.page.select(
            "#userlinksguest a[href*='act=Login&CODE=00']"
        )
        old_url = await self.page.evaluate("window.location.href")
        await login_link.click()
        deadline = asyncio.get_event_loop().time() + 10
        while (
            await self.page.evaluate("window.location.href") == old_url
            and asyncio.get_event_loop().time() < deadline
        ):
            await asyncio.sleep(0.1)

        await self.page.select("[name='UserName']", timeout=10)

        username_input = await self.page.select("[name='UserName']")
        await username_input.send_keys(self.username)

        password_input = await self.page.select("[name='PassWord']")
        await password_input.send_keys(self.password)

        await self._handle_login_challenge(flaresolverr_session)

        old_url = await self.page.evaluate("window.location.href")
        submit_button = await self.page.select(
            "input[type='submit'][value='Log me in']"
        )
        await submit_button.click()
        self.logger.info("'Log me in' button clicked, waiting for redirect...")

        deadline = asyncio.get_event_loop().time() + 10
        while (
            await self.page.evaluate("window.location.href") == old_url
            and asyncio.get_event_loop().time() < deadline
        ):
            await asyncio.sleep(0.1)

        await self._verify_login_succeeded(flaresolverr_session)
        self.logger.info("Login completed successfully")

        await self.gohomepage()

    async def _verify_login_succeeded(
        self,
        flaresolverr_session: FlareSolverrSession | None,
    ) -> None:
        """確認登入是否成功：重新檢查 Forums 頁面的 guest 標記是否已消失。

        提交登入表單後可能再次跳出 Cloudflare 驗證，因此這裡會再走一次
        驗證碼偵測流程，而不是直接信任「URL 有變化」就代表登入成功。
        """
        await self.myget(self.url["Forums"])
        await self.detect_and_solve_with_rotation(
            self.url["Forums"],
            flaresolverr_session=flaresolverr_session,
        )

        auth_state = await detect_forums_auth_state(self.page)
        if auth_state is not ForumsAuthState.AUTHENTICATED:
            raise LoginFailedException(
                "Forums did not show an authenticated member state after login "
                f"(state: {auth_state.value})"
            )
