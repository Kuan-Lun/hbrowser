"""Driver 基類"""

import asyncio
import os
from abc import ABC, abstractmethod
from random import random
from typing import Any

from .browser import DriverRestartRotator, ProxyRotator, create_browser, stop_browser
from .browser.ban_handler import handle_ban_decorator
from .captcha import CaptchaManager, TwoCaptchaAdapter
from .utils import get_log_dir, matchurl, setup_logger


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
    ) -> None:
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

        # 初始化驗證碼管理器
        # 使用 180 秒（3 分鐘）的等待時間，
        # 以便在非 headless 模式下��足夠時間手動解決驗證碼
        solver = TwoCaptchaAdapter(max_wait=180)
        self.captcha_manager = CaptchaManager(solver)

    async def _init_browser(self) -> None:
        """非同步初始化瀏覽器"""
        self.browser, self.page = await create_browser(headless=self.headless)
        self.myget = handle_ban_decorator(self.page)
        await self.get(self.url["Forums"])

    async def __aenter__(self) -> "Driver":
        await self._init_browser()
        await self.login()
        await self.gohomepage()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if exc_type:
            self.logger.error(f"Exception occurred: {exc_type.__name__}: {exc_val}")
            try:
                error_file = os.path.join(get_log_dir(), "error.txt")
                with open(error_file, "w", errors="ignore") as f:
                    f.write(await self.page.get_content())
                self.logger.debug(f"Error page saved to: {error_file}")
            except Exception:
                self.logger.error("Failed to save error page (browser session invalid)")
        self.logger.info("Closing browser")
        try:
            await stop_browser(self.browser)
        except Exception:
            pass

    async def gohomepage(self) -> None:
        """前往主頁"""
        url = self.url[self.name]
        current_url = await self.page.evaluate("window.location.href")
        if not matchurl(current_url, url):
            self.logger.info(f"Navigate to homepage: {url}")
            await self.get(url)
        else:
            self.logger.debug("Already on homepage, no navigation needed")

    async def find_element_chain(self, *selectors: str) -> Any:
        """通過 CSS selector 鏈逐步查找元素"""
        element: Any = self.page
        for selector in selectors:
            element = await element.query_selector(selector)
        return element

    async def get(self, url: str) -> None:
        """導航到指定 URL"""
        current_url = await self.page.evaluate("window.location.href")
        self.logger.debug(f"Navigate to URL: {url}")
        is_new_url = not matchurl(url, current_url)
        await self.myget(url)
        if is_new_url:
            # 等待 URL 變化
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
        # 隨機延遲
        await asyncio.sleep(3 * random())

    async def wait(
        self,
        fun: Any,
        ischangeurl: bool,
        sleeptime: int = -1,
    ) -> None:
        """
        執行 async 函數並等待頁面變化

        Args:
            fun: 要執行的 async callable
            ischangeurl: 是否等待 URL 變化
            sleeptime: 等待��間（秒），-1 表示隨機等待
        """
        old_url = await self.page.evaluate("window.location.href")

        # 重試機制
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
        """透過 ProxyRotator 輪換代理，替換當前 browser/page。"""
        self.browser, self.page = await self.proxy_rotator.rotate(
            self.browser, self.headless
        )
        self.myget = handle_ban_decorator(self.page)

    async def _handle_login_recaptcha(self, manual_timeout: float = 300.0) -> None:
        """處理登入表單上的 reCAPTCHA v2。

        先嘗試透過 2Captcha 自動解決，若失敗則等待使用者手動完成。

        Args:
            manual_timeout: 等待手動完成的超時時間（秒），預設 300 秒
        """
        det = await self.captcha_manager.detect(self.page, timeout=3.0)
        if det.kind != "recaptcha_v2":
            return

        self.logger.info("reCAPTCHA v2 detected on login form")

        # 嘗試透過 2Captcha 自動解決
        try:
            result = await self.captcha_manager.solve(det, self.page)
            if result:
                self.logger.info("reCAPTCHA v2 solved automatically")
                return
        except Exception:
            self.logger.debug("Auto-solve failed, will wait for manual completion")

        # 檢查 token 是否已就緒
        token = await self.page.evaluate(
            "(() => {"
            "var el = document.getElementById('g-recaptcha-response');"
            "return el ? el.value : '';"
            "})()"
        )
        if token:
            self.logger.info("reCAPTCHA token already present")
            return

        # 等待使用者手動完成 reCAPTCHA
        self.logger.info(
            "Please complete the reCAPTCHA manually in the browser. "
            f"Waiting up to {manual_timeout:.0f} seconds..."
        )
        deadline = asyncio.get_event_loop().time() + manual_timeout
        while asyncio.get_event_loop().time() < deadline:
            result = await self.page.evaluate(
                "(() => {"
                "var el = document.getElementById('g-recaptcha-response');"
                "return el && el.value.length > 0;"
                "})()"
            )
            if result:
                self.logger.info("reCAPTCHA completed by user")
                return
            await asyncio.sleep(1)
        self.logger.warning(
            "reCAPTCHA manual completion timed out, "
            "proceeding with login attempt anyway"
        )

    async def detect_and_solve_with_rotation(
        self,
        url: str,
        detect_timeout: float = 3.0,
    ) -> None:
        """檢測並解決 Cloudflare 驗證，失敗時自動輪換代理重試。

        可在任何需要通過 Cloudflare 驗證的地方呼叫此方法。

        Args:
            url: 要存取的 URL（輪換代理後��重新導航）
            detect_timeout: 驗證碼檢測超時時間（秒）

        Raises:
            Exception: 所有重試都失敗後拋出
        """
        for attempt in range(1, self.max_captcha_retries + 1):
            det = await self.captcha_manager.detect(self.page, timeout=detect_timeout)
            if det.kind == "none":
                self.logger.info("No challenge detected")
                return

            self.logger.warning(
                f"Challenge detected: {det.kind} "
                f"(attempt {attempt}/{self.max_captcha_retries})"
            )

            # 儲存驗證頁面供除錯
            challenge_page_path = os.path.join(get_log_dir(), "challenge_page.html")
            with open(challenge_page_path, "w", errors="ignore") as f:
                f.write(await self.page.get_content())
            self.logger.debug(f"Challenge page saved to: {challenge_page_path}")

            self.logger.info(f"Attempting to solve {det.kind} challenge...")
            try:
                success = await self.captcha_manager.solve(det, self.page)
            except Exception:
                success = False

            if success:
                self.logger.info("Challenge solved successfully")
                return

            # 解決失敗，輪換代理重試
            self.logger.warning(
                f"Failed to solve challenge "
                f"(attempt {attempt}/{self.max_captcha_retries})"
            )
            if attempt < self.max_captcha_retries:
                await self._rotate_proxy()
                await self.myget(url)

        raise Exception(
            f"Failed to solve captcha after {self.max_captcha_retries} attempts "
            f"with proxy rotation"
        )

    async def login(self) -> None:
        """
        登入流程

        透過 Forums 頁面登入：
        1. 進入 Forums 首頁（Cloudflare 驗��在此發生）
        2. 點擊 "Log In" 連結進入登入頁面
        3. 輸入帳號密碼並點擊 "Log me in"
        4. 驗證登入成功後前往主頁
        """
        self.logger.info("Starting login process")

        # 進入 Forums 首頁
        await self.myget(self.url["Forums"])

        # 檢測並解決 Cloudflare 驗證（失敗時自動輪換代理重試）
        await self.detect_and_solve_with_rotation(self.url["Forums"])

        # 檢查是否已登入（已登入時不會出現 userlinksguest）
        guest_elements = await self.page.query_selector_all("#userlinksguest")
        if not guest_elements:
            self.logger.info("Already logged in, skipping login")
            await self.gohomepage()
            return

        # 點擊 "Log In" 連結進入登入頁面
        self.logger.info("Clicking 'Log In' link on Forums page")
        login_link = await self.page.select(
            "#userlinksguest a[href*='act=Login&CODE=00']"
        )
        old_url = await self.page.evaluate("window.location.href")
        await login_link.click()
        # 等待 URL 變化
        deadline = asyncio.get_event_loop().time() + 10
        while (
            await self.page.evaluate("window.location.href") == old_url
            and asyncio.get_event_loop().time() < deadline
        ):
            await asyncio.sleep(0.1)

        # 等待登入表單載入
        await self.page.select("[name='UserName']", timeout=10)

        # 輸入帳號密碼
        username_input = await self.page.select("[name='UserName']")
        await username_input.send_keys(self.username)

        password_input = await self.page.select("[name='PassWord']")
        await password_input.send_keys(self.password)

        # 處理登入表單上的 reCAPTCHA v2
        await self._handle_login_recaptcha()

        # 點擊 "Log me in" 按鈕
        old_url = await self.page.evaluate("window.location.href")
        submit_button = await self.page.select(
            "input[type='submit'][value='Log me in']"
        )
        await submit_button.click()
        self.logger.info("'Log me in' button clicked, waiting for redirect...")

        # 等待登入完成（URL 變化）
        deadline = asyncio.get_event_loop().time() + 10
        while (
            await self.page.evaluate("window.location.href") == old_url
            and asyncio.get_event_loop().time() < deadline
        ):
            await asyncio.sleep(0.1)
        self.logger.info("Login completed successfully")

        await self.gohomepage()
