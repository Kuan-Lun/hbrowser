"""Driver 基類"""

import os
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from functools import partial
from random import random
from typing import Any

from selenium.common.exceptions import (
    ElementNotInteractableException,
    StaleElementReferenceException,
    TimeoutException,
    WebDriverException,
)
from selenium.webdriver.common.by import By
from selenium.webdriver.remote.webelement import WebElement
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

from .browser import DriverRestartRotator, ProxyRotator, create_driver
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

    @abstractmethod
    def _setlogin(self) -> str:
        """設定登入頁面名稱"""
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
        self.driver = create_driver(headless=headless)
        self.proxy_rotator = proxy_rotator or DriverRestartRotator()
        self.max_captcha_retries = max_captcha_retries

        # 初始化驗證碼管理器
        # 使用 180 秒（3 分鐘）的等待時間，
        # 以便在非 headless 模式下有足夠時間手動解決驗證碼
        solver = TwoCaptchaAdapter(max_wait=180)
        self.captcha_manager = CaptchaManager(solver)

        self.get(self.url["Forums"])

    def __enter__(self) -> "Driver":
        self.login()
        self.gohomepage()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if exc_type:
            self.logger.error(f"Exception occurred: {exc_type.__name__}: {exc_val}")
            try:
                error_file = os.path.join(get_log_dir(), "error.txt")
                with open(error_file, "w", errors="ignore") as f:
                    f.write(self.driver.page_source)
                self.logger.debug(f"Error page saved to: {error_file}")
            except WebDriverException:
                self.logger.error("Failed to save error page (browser session invalid)")
        self.logger.info("Closing browser driver")
        try:
            self.driver.quit()
        except WebDriverException:
            pass

    def gohomepage(self) -> None:
        """前往主頁"""
        url = self.url[self.name]
        if not matchurl(self.driver.current_url, url):
            self.logger.info(f"Navigate to homepage: {url}")
            self.get(url)
        else:
            self.logger.debug("Already on homepage, no navigation needed")

    def find_element_chain(self, *selectors: tuple[str, str]) -> WebElement:
        """通過選擇器鏈逐步查找元素，每次在前一個元素的基礎上查找下一個"""
        element: Any = self.driver
        for by, value in selectors:
            element = element.find_element(by, value)
        return element

    def get(self, url: str) -> None:
        """導航到指定 URL"""
        old_url = self.driver.current_url
        self.logger.debug(f"Navigate to URL: {url}")
        self.wait(
            fun=partial(self.driver.myget, url),
            ischangeurl=(not matchurl(url, old_url)),
        )

    def wait(
        self, fun: Callable[[], None], ischangeurl: bool, sleeptime: int = -1
    ) -> None:
        """
        執行函數並等待頁面變化

        Args:
            fun: 要執行的函數
            ischangeurl: 是否等待 URL 變化
            sleeptime: 等待時間（秒），-1 表示隨機等待
        """
        old_url = self.driver.current_url

        # 重試機制處理 StaleElementReferenceException / ElementNotInteractableException
        max_retries = 3
        for attempt in range(max_retries):
            try:
                fun()
                break
            except (StaleElementReferenceException, ElementNotInteractableException):
                if attempt == max_retries - 1:
                    raise
                # 短暫等待後重試
                time.sleep(0.5)

        try:
            match ischangeurl:
                case False:
                    self.driver.implicitly_wait(10)
                case True:
                    wait = WebDriverWait(self.driver, 10)
                    wait.until(lambda driver: driver.current_url != old_url)
                case _:
                    raise KeyError()
        except TimeoutException as e:
            raise e
        if sleeptime < 0:
            time.sleep(3 * random())
        else:
            time.sleep(sleeptime)

    def _rotate_proxy(self) -> None:
        """透過 ProxyRotator 輪換代理，替換當前 driver。"""
        self.driver = self.proxy_rotator.rotate(self.driver, self.headless)

    def detect_and_solve_with_rotation(
        self,
        url: str,
        detect_timeout: float = 3.0,
    ) -> None:
        """檢測並解決 Cloudflare 驗證，失敗時自動輪換代理重試。

        可在任何需要通過 Cloudflare 驗證的地方呼叫此方法。

        Args:
            url: 要存取的 URL（輪換代理後需重新導航）
            detect_timeout: 驗證碼檢測超時時間（秒）

        Raises:
            Exception: 所有重試都失敗後拋出
        """
        for attempt in range(1, self.max_captcha_retries + 1):
            det = self.captcha_manager.detect(self.driver, timeout=detect_timeout)
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
                f.write(self.driver.page_source)
            self.logger.debug(f"Challenge page saved to: {challenge_page_path}")

            self.logger.info(f"Attempting to solve {det.kind} challenge...")
            try:
                success = self.captcha_manager.solve(det, self.driver)
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
                self._rotate_proxy()
                self.driver.myget(url)

        raise Exception(
            f"Failed to solve captcha after {self.max_captcha_retries} attempts "
            f"with proxy rotation"
        )

    def login(self) -> None:
        """
        登入流程

        透過 Forums 頁面登入：
        1. 進入 Forums 首頁（Cloudflare 驗證在此發生）
        2. 點擊 "Log In" 連結進入登入頁面
        3. 輸入帳號密碼並點擊 "Log me in"
        4. 驗證登入成功後前往主頁
        """
        self.logger.info("Starting login process")

        # 進入 Forums 首頁
        self.driver.myget(self.url["Forums"])

        # 檢測並解決 Cloudflare 驗證（失敗時自動輪換代理重試）
        self.detect_and_solve_with_rotation(self.url["Forums"])

        # 檢查是否已登入（已登入時不會出現 userlinksguest）
        if not self.driver.find_elements(By.ID, "userlinksguest"):
            self.logger.info("Already logged in, skipping login")
            self.gohomepage()
            return

        # 點擊 "Log In" 連結進入登入頁面
        self.logger.info("Clicking 'Log In' link on Forums page")
        login_link = self.driver.find_element(
            By.CSS_SELECTOR, "#userlinksguest a[href*='act=Login&CODE=00']"
        )
        old_url = self.driver.current_url
        login_link.click()
        WebDriverWait(self.driver, 10).until(
            lambda driver: driver.current_url != old_url
        )

        # 等待登入表單載入
        WebDriverWait(self.driver, 10).until(
            EC.presence_of_element_located((By.NAME, "UserName"))
        )

        # 輸入帳號密碼
        username_input = self.driver.find_element(By.NAME, "UserName")
        username_input.send_keys(self.username)

        password_input = self.driver.find_element(By.NAME, "PassWord")
        password_input.send_keys(self.password)

        # 點擊 "Log me in" 按鈕
        old_url = self.driver.current_url
        submit_button = self.driver.find_element(
            By.CSS_SELECTOR, "input[type='submit'][value='Log me in']"
        )
        submit_button.click()
        self.logger.info("'Log me in' button clicked, waiting for redirect...")

        # 等待登入完成（URL 變化）
        WebDriverWait(self.driver, 10).until(
            lambda driver: driver.current_url != old_url
        )
        self.logger.info("Login completed successfully")

        self.gohomepage()
