"""Driver 基類"""

import os
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from functools import partial
from random import random
from typing import Any

from selenium.common.exceptions import (
    NoSuchElementException,
    StaleElementReferenceException,
    TimeoutException,
)
from selenium.webdriver.common.by import By
from selenium.webdriver.remote.webelement import WebElement
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

from .browser import create_driver
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
    ) -> None:
        def seturl() -> dict[str, str]:
            url: dict[str, str] = dict()
            url["My Home"] = "https://e-hentai.org/home.php"
            url["E-Hentai"] = "https://e-hentai.org/"
            url["ExHentai"] = "https://exhentai.org/"
            url["HentaiVerse"] = "https://hentaiverse.org"
            url["HentaiVerse isekai"] = "https://hentaiverse.org/isekai/"
            return url

        self.logger = setup_logger(__name__)
        self.username = os.getenv("EH_USERNAME")
        self.password = os.getenv("EH_PASSWORD")
        self.url = seturl()
        self.name = self._setname()
        self.driver = create_driver(headless=headless)

        # 初始化驗證碼管理器
        # 使用 180 秒（3 分鐘）的等待時間，
        # 以便在非 headless 模式下有足夠時間手動解決驗證碼
        solver = TwoCaptchaAdapter(max_wait=180)
        self.captcha_manager = CaptchaManager(solver)

        self.get(self.url["My Home"])

    def __enter__(self) -> "Driver":
        self.login()
        self.gohomepage()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if exc_type:
            self.logger.error(f"Exception occurred: {exc_type.__name__}: {exc_val}")
            error_file = os.path.join(get_log_dir(), "error.txt")
            with open(error_file, "w", errors="ignore") as f:
                f.write(self.driver.page_source)
            self.logger.debug(f"Error page saved to: {error_file}")
        self.logger.info("Closing browser driver")
        self.driver.quit()

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
        return element  # type: ignore[no-any-return]

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

        # 重試機制處理 StaleElementReferenceException
        max_retries = 3
        for attempt in range(max_retries):
            try:
                fun()
                break
            except StaleElementReferenceException:
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

    def login(self, isfirst: bool = True) -> None:
        """登入流程"""
        self.logger.info("Starting login process")
        # 打開登入網頁
        self.driver.myget(self.url["My Home"])
        try:
            self.driver.find_element(By.XPATH, "//a[contains(text(), 'Hentai@Home')]")
            iscontinue = False
        except NoSuchElementException:
            iscontinue = True
        if not iscontinue:
            self.logger.info("Already logged in, skipping login")
            return
        WebDriverWait(self.driver, 10).until(
            EC.presence_of_element_located((By.NAME, "UserName"))
        )

        if self.driver.find_elements(By.NAME, "PassWord"):
            element_present = EC.presence_of_element_located((By.NAME, "UserName"))
            WebDriverWait(self.driver, 10).until(element_present)

            # 定位用戶名輸入框並輸入用戶名
            username_input = self.driver.find_element(By.NAME, "UserName")
            username_input.send_keys(self.username)

            # 定位密碼輸入框並輸入密碼
            password_input = self.driver.find_element(By.NAME, "PassWord")
            password_input.send_keys(self.password)

            # 獲取點擊之前的 URL
            old_url = self.driver.current_url

            # 定位登入按鈕並點擊它
            login_button = self.driver.find_element(By.NAME, "ipb_login_submit")
            login_button.click()

            # 顯式等待，直到 URL 改變
            wait = WebDriverWait(self.driver, 10)
            wait.until(lambda driver: driver.current_url != old_url)
            self.logger.info("Login button clicked, checking for challenges...")

            # 檢測是否遇到 Cloudflare 驗證
            det = self.captcha_manager.detect(self.driver, timeout=3.0)

            if det.kind != "none":
                self.logger.warning(f"Challenge detected: {det.kind}")
                # 保存登入後的頁面以供調試
                login_page_path = os.path.join(get_log_dir(), "login_page.html")
                with open(login_page_path, "w", errors="ignore") as f:
                    f.write(self.driver.page_source)
                self.logger.debug(f"Login page saved to: {login_page_path}")

                # 嘗試解決驗證
                self.logger.info(f"Attempting to solve {det.kind} challenge...")
                success = self.captcha_manager.solve(det, self.driver)
                if not success:
                    self.logger.error("Failed to solve captcha challenge")
                    raise Exception("Failed to solve captcha challenge")
                self.logger.info("Challenge solved successfully")
            else:
                self.logger.info("No challenge detected, proceeding with login...")

            # 假設跳轉後的頁面有一個具有 NAME=reset_imagelimit 的元素
            element_present = EC.presence_of_element_located(
                (By.NAME, "reset_imagelimit")
            )
            self.logger.debug("Waiting for homepage to load...")
            WebDriverWait(self.driver, 10).until(element_present)
            self.logger.info("Login completed successfully")

        self.gohomepage()
