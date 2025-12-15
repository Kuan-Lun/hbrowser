__all__ = ["EHDriver", "ExHDriver", "Tag"]


import os
import re
import time
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from functools import partial
from random import random
from typing import Optional, Literal
from dataclasses import dataclass


from fake_useragent import UserAgent  # type: ignore
from twocaptcha import TwoCaptcha  # type: ignore
from h2h_galleryinfo_parser import GalleryURLParser
from selenium.common.exceptions import NoSuchElementException, TimeoutException
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.remote.webelement import WebElement
import undetected_chromedriver as uc  # type: ignore

from .exceptions import (
    ClientOfflineException,
    InsufficientFundsException,
    CaptchaAPIKeyNotSetException,
    CaptchaSolveException,
)


Kind = Literal["none", "turnstile_widget", "cf_managed_challenge"]

TURNSTILE_IFRAME_CSS = (
    "iframe[src*='challenges.cloudflare.com'][src*='/turnstile/'], "
    "iframe[src*='challenges.cloudflare.com'][src*='turnstile']"
)
SITEKEY_RE = re.compile(r"/(0x[a-zA-Z0-9]+)/")
RAY_RE = re.compile(r"Ray ID:\s*<code>\s*([0-9a-f]+)\s*</code>", re.IGNORECASE)


def get_log_dir() -> str:
    """
    獲取主腳本所在目錄下的 log 資料夾路徑，如果不存在則建立

    Returns:
        log 資料夾的絕對路徑
    """
    import sys

    # 獲取主腳本的路徑
    if hasattr(sys, 'argv') and len(sys.argv) > 0:
        main_script = sys.argv[0]
        if main_script:
            # 獲取主腳本所在的目錄
            script_dir = os.path.dirname(os.path.abspath(main_script))
        else:
            # 如果無法獲取主腳本路徑，使用當前工作目錄
            script_dir = os.getcwd()
    else:
        script_dir = os.getcwd()

    # 建立 log 資料夾路徑
    log_dir = os.path.join(script_dir, "log")

    # 如果 log 資料夾不存在，則建立
    os.makedirs(log_dir, exist_ok=True)

    return log_dir


class Tag:
    def __init__(
        self,
        filter: str,
        name: str,
        href: str,
    ) -> None:
        self.filter = filter
        self.name = name
        self.href = href

    def __repr__(self) -> str:
        itemlist = list()
        for attr_name, attr_value in self.__dict__.items():
            itemlist.append(": ".join([attr_name, attr_value]))
        return "\n".join(itemlist)

    def __str__(self) -> str:
        return ", ".join(self.__repr__().split("\n"))


@dataclass(frozen=True)
class ChallengeDetection:
    url: str
    kind: Kind
    sitekey: Optional[str] = None
    iframe_src: Optional[str] = None
    ray_id: Optional[str] = None


def matchurl(*args) -> bool:
    """
    Example:
    matchurl("https://e-hentai.org", "https://e-hentai.org/") -> True
    matchurl("https://e-hentai.org", "https://e-hentai.org") -> True
    matchurl("https://e-hentai.org", "https://exhentai.org") -> False
    matchurl("https://e-hentai.org", "https://e-hentai.org", "https://e-hentai.org") -> True
    """
    fixargs = list()
    for url in args:
        while url[-1] == "/":
            url = url[0:-1]
        fixargs.append(url)

    t = True
    for url in fixargs[1:]:
        t &= fixargs[0] == url
    return t


def find_new_window(existing_windows, driver):
    current_windows = set(driver.window_handles)
    new_windows = current_windows - existing_windows
    return next(iter(new_windows or []), None)


class DriverPass:
    def __init__(
        self,
        username: str,
        password: str,
        logcontrol=None,
        headless=True,
    ) -> None:
        self.username = username
        self.password = password
        self.logcontrol = logcontrol
        self.headless = headless

    def getdict(self) -> dict:
        vdict = dict()
        for attr_name, attr_value in self.__dict__.items():
            vdict[attr_name] = attr_value
        return vdict


def handle_ban_decorator(driver, logcontrol):  # , cookiesname):
    def sendmsg(msg: str) -> None:
        if logcontrol is not None:
            logcontrol(msg)
        else:
            print(msg)

    def banningcheck() -> None:
        def banningmsg() -> str:
            a = timedelta(seconds=wait_seconds)
            msg = f"IP banned, waiting for {a} (until {wait_until.strftime('%Y-%m-%d %H:%M:%S')}) to retry..."
            return msg

        def whilecheck() -> bool:
            return whilecheckban() or whilechecknothing()

        def whilecheckban() -> bool:
            return baningmsg in source

        def whilechecknothing() -> bool:
            return nothing == source

        source = driver.page_source
        nothing = "<html><head></head><body></body></html>"
        baningmsg = "Your IP address has been temporarily banned"
        onehour = 60 * 60

        if whilecheck():
            isfirst = True
            isnothing = nothing == source
            while whilecheck():
                sendmsg(source)
                if not isfirst:
                    sendmsg("Ban again")
                if isnothing:
                    wait_seconds = 4 * onehour
                else:
                    wait_seconds = parse_ban_time(source)
                wait_until = datetime.now() + timedelta(seconds=wait_seconds)
                sendmsg(banningmsg())

                while wait_seconds > onehour:
                    time.sleep(onehour)
                    wait_seconds -= onehour
                    sendmsg(banningmsg())
                time.sleep(wait_seconds + 15 * 60)
                wait_seconds = 0
                sendmsg("Retry")
                driver.refresh()
                source = driver.page_source
                isfirst = False
                if isnothing:
                    # Cookies.remove(cookiesname)
                    raise RuntimeError()
            sendmsg("Now is fine")
        else:
            return

    def myget(*args, **kwargs) -> None:
        driver.get(*args, **kwargs)
        banningcheck()

    return myget


def parse_ban_time(page_source: str) -> int:
    def calculate(duration_str: str) -> dict[str, int]:
        # Regular expression patterns to capture days, hours, and minutes
        patterns = {
            "days": r"(\d+) day?",
            "hours": r"(\d+) hour?",
            "minutes": r"(\d+) minute?",
        }

        # Dictionary to store the found durations
        durations = {"days": 0, "hours": 0, "minutes": 0}

        # Search for each duration in the string and update the durations dictionary
        for key, pattern in patterns.items():
            match = re.search(pattern, duration_str)
            if match:
                durations[key] = int(match.group(1))

        return durations

    # 解析被禁時間的實現這裡省略，與前面相同
    durations = calculate(page_source)
    return 60 * (
        60 * (24 * durations["days"] + durations["hours"]) + durations["minutes"]
    )


class Driver(ABC):
    def detect_challenge(self, timeout: float = 2.0) -> ChallengeDetection:
        url = self.driver.current_url
        title = (self.driver.title or "").strip()

        # (A) 先判斷是否為 Cloudflare managed challenge 整頁（你貼的就是這種）
        # 特徵：title + _cf_chl_opt + /cdn-cgi/challenge-platform
        html = self.driver.page_source or ""
        if (
            ("請稍候" in title)
            or ("Just a moment" in title)
            or ("_cf_chl_opt" in html)
            or ("/cdn-cgi/challenge-platform/" in html)
        ):
            ray = None
            m = RAY_RE.search(html)
            if m:
                ray = m.group(1)
            return ChallengeDetection(url=url, kind="cf_managed_challenge", ray_id=ray)

        # (B) 再找 Turnstile widget iframe（可抽 sitekey 的那種）
        try:
            iframe = WebDriverWait(self.driver, timeout).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, TURNSTILE_IFRAME_CSS))
            )
        except TimeoutException:
            return ChallengeDetection(url=url, kind="none")

        iframe_src = iframe.get_attribute("src") or ""
        m = SITEKEY_RE.search(iframe_src)
        sitekey = m.group(1) if m else None
        return ChallengeDetection(
            url=url, kind="turnstile_widget", sitekey=sitekey, iframe_src=iframe_src
        )

    @abstractmethod
    def _setname(self) -> str:
        pass

    @abstractmethod
    def _setlogin(self) -> str:
        pass

    def gohomepage(self) -> None:
        url = self.url[self.name]
        if not matchurl(self.driver.current_url, url):
            self.get(url)

    def find_element_chain(self, *selectors: tuple[str, str]) -> WebElement:
        """通過選擇器鏈逐步查找元素，每次在前一個元素的基礎上查找下一個"""
        element = self.driver
        for by, value in selectors:
            element = element.find_element(by, value)
        return element

    def __init__(
        self,
        username: str,
        password: str,
        # cookiesname: str,
        logcontrol=None,
        headless=True,
    ) -> None:
        def gendriver(logcontrol):
            # 設定瀏覽器參數
            options = uc.ChromeOptions()
            options.add_argument("--disable-extensions")
            if headless:
                options.add_argument("--headless=new")  # 使用新的無頭模式
            options.add_argument(
                "--no-sandbox"
            )  # 解決DevToolsActivePort文件不存在的問題
            options.add_argument("--window-size=1600,900")
            options.add_argument("--disable-dev-shm-usage")
            options.add_argument(
                "user-agent={ua}".format(ua=UserAgent()["google chrome"])
            )
            options.page_load_strategy = (
                "normal"  # 等待加载图片normal eager none </span></div>
            )

            # 使用 undetected-chromedriver 初始化 WebDriver
            driver = uc.Chrome(options=options, use_subprocess=True)

            # driver.request_interceptor = interceptor
            driver.myget = handle_ban_decorator(driver, logcontrol)  # , cookiesname)

            return driver

        def seturl() -> dict:
            url = dict()
            url["My Home"] = "https://e-hentai.org/home.php"
            url["E-Hentai"] = "https://e-hentai.org/"
            url["ExHentai"] = "https://exhentai.org/"
            url["HentaiVerse"] = "https://hentaiverse.org"
            url["HentaiVerse isekai"] = "https://hentaiverse.org/isekai/"
            return url

        self.username = username
        self.password = password
        self.url = seturl()
        self.name = self._setname()
        self.driver = gendriver(logcontrol)
        self.get(self.url["My Home"])
        # self.cookiesname = cookiesname
        # if Cookies.load(self.driver, self.cookiesname):
        #     self.get(self.url["My Home"])

    def __enter__(self):
        self.login()
        self.gohomepage()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if exc_type:
            with open(
                os.path.join(get_log_dir(), "error.txt"),
                "w",
                errors="ignore",
            ) as f:
                f.write(self.driver.page_source)
        self.driver.quit()

    def get(self, url: str) -> None:
        old_url = self.driver.current_url
        self.wait(
            fun=partial(self.driver.myget, url),
            ischangeurl=(not matchurl(url, old_url)),
        )

    def wait(self, fun, ischangeurl: bool, sleeptime: int = -1) -> None:
        old_url = self.driver.current_url
        fun()
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

    def solve_cloudflare_challenge(self, det: ChallengeDetection, max_wait: int = 60) -> None:
        """
        解決 Cloudflare 驗證挑戰

        Args:
            det: 檢測到的驗證信息
            max_wait: 最長等待時間（秒）

        Raises:
            ValueError: 當 APIKEY_2CAPTCHA 環境變數未設置
            RuntimeError: 當驗證解決失敗或超時
        """
        # 檢查 API key
        api_key = os.getenv("APIKEY_2CAPTCHA")
        if not api_key:
            raise CaptchaAPIKeyNotSetException(
                "APIKEY_2CAPTCHA environment variable is not set. "
                "Please set it using: export APIKEY_2CAPTCHA=your_api_key"
            )

        print(f"Detected {det.kind} challenge, attempting to solve...")

        try:
            solver = TwoCaptcha(api_key)

            if det.kind == "cf_managed_challenge":
                # Cloudflare managed challenge (整頁驗證)
                # 保存當前頁面以供調試
                with open(
                    os.path.join(get_log_dir(), "challenge_page.html"),
                    "w",
                    errors="ignore",
                ) as f:
                    f.write(self.driver.page_source)

                print(f"Cloudflare managed challenge detected (Ray ID: {det.ray_id})")

                # 嘗試提取 sitekey（managed challenge 也可能包含 Turnstile）
                html = self.driver.page_source
                sitekey_match = re.search(r'sitekey["\s:]+([0-9a-zA-Z_-]+)', html)

                if sitekey_match:
                    sitekey = sitekey_match.group(1)
                    print(f"Found sitekey in managed challenge: {sitekey}")

                    # 嘗試使用 2Captcha 解決 Turnstile
                    try:
                        print("Attempting to solve with 2Captcha Turnstile API...")
                        result = solver.turnstile(
                            sitekey=sitekey,
                            url=det.url,
                        )

                        token = result.get("code")
                        if token:
                            print(f"Got token from 2Captcha: {token[:50]}...")

                            # 嘗試注入 token
                            self.driver.execute_script(
                                """
                                // 方法1: 尋找並設置 turnstile response input
                                var inputs = document.querySelectorAll('input[name*="turnstile"], input[name*="cf-turnstile"]');
                                for (var i = 0; i < inputs.length; i++) {
                                    inputs[i].value = arguments[0];
                                }

                                // 方法2: 如果有 callback
                                if (window.turnstile && typeof window.turnstile.reset === 'function') {
                                    try {
                                        // 嘗試觸發驗證完成
                                        if (window.cfCallback) window.cfCallback(arguments[0]);
                                        if (window.tsCallback) window.tsCallback(arguments[0]);
                                    } catch(e) {
                                        console.log('Callback error:', e);
                                    }
                                }

                                // 方法3: 提交表單（如果存在）
                                var form = document.querySelector('form');
                                if (form) {
                                    try {
                                        form.submit();
                                    } catch(e) {
                                        console.log('Form submit error:', e);
                                    }
                                }
                                """,
                                token
                            )

                            print("Token injected, waiting for page to respond...")
                            time.sleep(5)
                    except Exception as e:
                        print(f"2Captcha solve attempt failed: {str(e)}")
                        print("Falling back to passive waiting...")

                # 輪詢檢查頁面是否已經通過驗證
                print("Monitoring page for challenge resolution...")
                start_time = time.time()
                check_interval = 5
                last_url = self.driver.current_url

                while time.time() - start_time < max_wait:
                    time.sleep(check_interval)

                    current_url = self.driver.current_url

                    # 檢查 URL 是否變化（表示可能已通過）
                    if current_url != last_url:
                        print(f"URL changed from {last_url} to {current_url}")
                        last_url = current_url

                    # 重新檢測是否還有驗證
                    current_det = self.detect_challenge(timeout=1.0)
                    if current_det.kind == "none":
                        print("Challenge resolved successfully!")
                        return

                    elapsed = int(time.time() - start_time)
                    print(f"Still waiting... ({elapsed}s/{max_wait}s)")

                # 超時仍未解決
                raise CaptchaSolveException(
                    f"Cloudflare managed challenge not resolved after {max_wait} seconds. "
                    f"Ray ID: {det.ray_id}. "
                    f"\n\nPossible solutions:"
                    f"\n1. Disable headless mode by setting headless=False"
                    f"\n2. Try running the script with a real browser window"
                    f"\n3. Use a different IP address or wait before retrying"
                    f"\n4. Cloudflare may be blocking automated access to this site"
                )

            elif det.kind == "turnstile_widget":
                # Turnstile widget (可以通過 API 解決)
                if not det.sitekey:
                    raise CaptchaSolveException("Turnstile widget detected but sitekey not found")

                print(f"Solving Turnstile with sitekey: {det.sitekey}")

                # 提交驗證任務到 2Captcha
                result = solver.turnstile(
                    sitekey=det.sitekey,
                    url=det.url,
                )

                # 獲取解決的 token
                token = result.get("code")
                if not token:
                    raise CaptchaSolveException("Failed to get token from 2Captcha response")

                print(f"Got token from 2Captcha: {token[:50]}...")

                # 將 token 注入到頁面
                # 方法1: 通過 Turnstile callback
                success = self.driver.execute_script(
                    """
                    if (window.turnstile && window.turnstile.reset) {
                        // 如果有 callback，直接調用
                        if (window.cfCallback || window.tsCallback) {
                            const callback = window.cfCallback || window.tsCallback;
                            callback(arguments[0]);
                            return true;
                        }
                    }

                    // 方法2: 設置到隱藏的表單欄位
                    const input = document.querySelector('input[name="cf-turnstile-response"]');
                    if (input) {
                        input.value = arguments[0];
                        return true;
                    }

                    return false;
                    """,
                    token
                )

                if success:
                    print("Token injected successfully, waiting for page to respond...")
                    time.sleep(3)

                    # 檢查是否通過驗證
                    current_det = self.detect_challenge(timeout=1.0)
                    if current_det.kind == "none":
                        print("Turnstile challenge resolved successfully!")
                        return
                else:
                    print("Warning: Could not inject token using standard methods")

        except (CaptchaAPIKeyNotSetException, CaptchaSolveException):
            # 直接重新拋出這些已知的異常
            raise
        except Exception as e:
            # 保存錯誤時的頁面狀態
            with open(
                os.path.join(get_log_dir(), "challenge_error.html"),
                "w",
                errors="ignore",
            ) as f:
                f.write(self.driver.page_source)

            # 包裝其他未預期的異常
            raise CaptchaSolveException(f"Failed to solve Cloudflare challenge: {str(e)}") from e

    def login(self) -> None:
        # 打開登入網頁
        self.driver.myget(self.url["My Home"])
        try:
            self.driver.find_element(By.XPATH, "//a[contains(text(), 'Hentai@Home')]")
            iscontinue = False
        except NoSuchElementException:
            iscontinue = True
        if not iscontinue:
            return
        WebDriverWait(self.driver, 10).until(
            EC.presence_of_element_located((By.NAME, "UserName"))
        )

        if self.driver.find_elements(By.NAME, "PassWord"):
            element_present = EC.presence_of_element_located((By.NAME, "UserName"))
            WebDriverWait(self.driver, 10).until(element_present)

            # 定位用戶名輸入框並輸入用戶名，替換 'your_username' 為實際的用戶名
            username_input = self.driver.find_element(
                By.NAME, "UserName"
            )  # 可能需要根據實際情況調整查找方法
            username_input.send_keys(self.username)

            # 定位密碼輸入框並輸入密碼，替換 'your_password' 為實際的密碼
            password_input = self.driver.find_element(
                By.NAME, "PassWord"
            )  # 可能需要根據實際情況調整查找方法
            password_input.send_keys(self.password)

            # 獲取點擊之前的 URL
            old_url = self.driver.current_url

            # 定位登入按鈕並點擊它
            login_button = self.driver.find_element(
                By.NAME, "ipb_login_submit"
            )  # 查找方法可能需要根據實際情況調整
            login_button.click()

            # 顯式等待，直到 URL 改變
            wait = WebDriverWait(self.driver, 10)
            wait.until(lambda driver: driver.current_url != old_url)
            print("Login button clicked, checking for challenges...")

            # 檢測是否遇到 Cloudflare 驗證
            det = self.detect_challenge(timeout=3.0)

            if det.kind != "none":
                print(f"Challenge detected: {det.kind}")
                # 保存登入後的頁面以供調試
                with open(
                    os.path.join(get_log_dir(), "login_page.html"),
                    "w",
                    errors="ignore",
                ) as f:
                    f.write(self.driver.page_source)

                # 嘗試解決驗證
                self.solve_cloudflare_challenge(det, max_wait=60)
            else:
                print("No challenge detected, proceeding...")

            # 假設跳轉後的頁面有一個具有 NAME=reset_imagelimit 的元素
            element_present = EC.presence_of_element_located(
                (By.NAME, "reset_imagelimit")
            )
            print("Waiting for homepage to load...")
            WebDriverWait(self.driver, 10).until(element_present)
            print("Login completed successfully.")
        # Cookies.save(self.driver, self.cookiesname)
        self.gohomepage()


class EHDriver(Driver):
    def _setname(self) -> str:
        return "E-Hentai"

    def _setlogin(self) -> str:
        return "My Home"

    def checkh2h(self) -> bool:
        self.get("https://e-hentai.org/hentaiathome.php")
        WebDriverWait(self.driver, 10).until(
            EC.presence_of_element_located((By.ID, "hct"))
        )
        table = self.driver.find_element(By.ID, "hct")
        headers = table.find_element(By.TAG_NAME, "tr").find_elements(By.TAG_NAME, "th")
        status_index = [
            index for index, th in enumerate(headers) if th.text == "Status"
        ][0]
        rows = table.find_elements(By.TAG_NAME, "tr")
        for row in rows[1:]:
            # 獲取每行的所有單元格
            cells = row.find_elements(By.TAG_NAME, "td")
            # 使用 'Status' 列的索引來檢查狀態
            status = cells[status_index].text
            if status.lower() == "online":
                return True
        return False

    def punchin(self) -> None:
        # 嘗試簽到
        self.get("https://e-hentai.org/news.php")

        # 刷新以免沒簽到成功
        self.wait(self.driver.refresh, ischangeurl=False)

    def search2gallery(self, url: str) -> list[GalleryURLParser]:
        if not matchurl(self.driver.current_url, url):
            self.get(url)

        input_element = self.driver.find_element(By.ID, "f_search")
        input_value = input_element.get_attribute("value")
        if input_value == "":
            raise ValueError(
                "The value in the search box is empty. I think there are TOO MANY GALLERIES."
            )

        glist = list()
        while True:
            html_content = self.driver.page_source
            pattern = r"https://exhentai.org/g/\d+/[A-Za-z0-9]+"
            glist += re.findall(pattern, html_content)
            try:
                element = self.driver.find_element(By.ID, "unext")
            except NoSuchElementException:
                break
            if element.tag_name == "a":
                self.wait(element.click, ischangeurl=True)
                element_present = EC.presence_of_element_located((By.ID, "unext"))
                WebDriverWait(self.driver, 10).until(element_present)
            else:
                break
        if len(glist) == 0:
            try:
                self.driver.find_element(
                    By.XPATH,
                    "//*[contains(text(), 'No hits found')] | //td[contains(text(), 'No unfiltered results found.')]",
                )
            except NoSuchElementException:
                raise ValueError("找出 0 個 Gallery，但頁面沒有顯示 'No hits found'。")
        glist = list(set(glist))
        glist = [GalleryURLParser(url) for url in glist]
        return glist

    def search(self, key: str, isclear: bool) -> list[GalleryURLParser]:
        def waitpage() -> None:
            element_present = EC.presence_of_element_located((By.ID, "f_search"))
            WebDriverWait(self.driver, 10).until(element_present)

        try:
            input_element = self.driver.find_element(By.ID, "f_search")
        except NoSuchElementException:
            self.gohomepage()
            waitpage()
            input_element = self.driver.find_element(By.ID, "f_search")
        if isclear:
            input_element.clear()
            time.sleep(random())
            new_value = key
        else:
            input_value = input_element.get_attribute("value")
            if key == "":
                new_value = input_value
            else:
                new_value = " " + key
        input_element.send_keys(new_value)
        time.sleep(random())

        # 全總類搜尋
        elements = self.driver.find_elements(
            By.XPATH, "//div[contains(@id, 'cat_') and @data-disabled='1']"
        )
        for element in elements:
            element.click()
            time.sleep(random())

        button = self.driver.find_elements(By.XPATH, "//tr")
        button = self.driver.find_element(
            By.XPATH, '//input[@type="submit" and @value="Search"]'
        )
        button.click()
        time.sleep(random())
        waitpage()

        input_element = self.driver.find_element(By.ID, "f_search")
        input_value = input_element.get_attribute("value")
        print("Search", input_value)

        result = self.search2gallery(self.driver.current_url)
        return result

    def download(self, gallery: GalleryURLParser) -> bool:
        def _check_ekey(driver, ekey: str):
            return EC.presence_of_element_located((By.XPATH, ekey))(
                driver
            ) or EC.visibility_of_element_located((By.XPATH, ekey))(driver)

        def check_download_success_by_element(driver):
            ekey = "//p[contains(text(), 'Downloads should start processing within a couple of minutes.')]"
            return _check_ekey(driver, ekey)

        def check_client_offline_by_element(driver):
            ekey = "//p[contains(text(), 'Your H@H client appears to be offline.')]"
            try:
                _check_ekey(driver, ekey)
            except NoSuchElementException:
                raise ClientOfflineException()

        def check_insufficient_funds_by_element(driver):
            ekey = "//p[contains(text(), 'Cannot start download: Insufficient funds')]"
            try:
                _check_ekey(driver, ekey)
            except NoSuchElementException:
                raise InsufficientFundsException()

        self.get(gallery.url)
        try:
            xpath_query_list = [
                "//p[contains(text(), 'This gallery is unavailable due to a copyright claim by Irodori Comics.')]",
                "//input[@id='f_search']",
            ]
            xpath_query = " | ".join(xpath_query_list)
            self.driver.find_element(By.XPATH, xpath_query)
            return False
        except NoSuchElementException:
            gallerywindow = self.driver.current_window_handle
            existing_windows = set(self.driver.window_handles)
            key = "//a[contains(text(), 'Archive Download')]"
            try:
                self.driver.find_element(By.XPATH, key).click()
            except NoSuchElementException:
                print("NoSuchElementException")
                self.driver.close()
                self.driver.switch_to.window(gallerywindow)
                print("Retry again.")
                return self.download(gallery)
            WebDriverWait(self.driver, 10).until(
                partial(find_new_window, existing_windows)
            )

            # 切換到新視窗
            new_window = self.driver.window_handles[-1]
            self.driver.switch_to.window(new_window)

            # 點擊 Original，開始下載。
            key = "//a[contains(text(), 'Original')]"
            element_present = EC.presence_of_element_located((By.XPATH, key))
            WebDriverWait(self.driver, 10).until(element_present)
            self.driver.find_element(By.XPATH, key).click()

            # 確認是否連接 H@H
            try:
                try:
                    WebDriverWait(self.driver, 10).until(
                        lambda driver: check_download_success_by_element(driver)
                        or check_client_offline_by_element(driver)
                        or check_insufficient_funds_by_element(driver)
                    )
                except TimeoutException:
                    if (
                        "Cannot start download: Insufficient funds"
                        in self.driver.page_source
                    ):
                        raise InsufficientFundsException()
                    else:
                        raise TimeoutException()
            except TimeoutException:
                with open(os.path.join(".", "error.txt"), "w", errors="ignore") as f:
                    f.write(self.driver.page_source)
                retrytime = 1 * 60  # 1 minute1
                print("TimeoutException")
                self.driver.close()
                self.driver.switch_to.window(gallerywindow)
                print("Retry again.")
                time.sleep(retrytime)
                return self.download(gallery)
            if len(self.driver.current_window_handle) > 1:
                self.driver.close()
                time.sleep(random())
                self.driver.switch_to.window(gallerywindow)
                time.sleep(random())
            else:
                print(
                    "Error. driver.current_window_handle: {a}".format(
                        a=self.driver.current_window_handle
                    )
                )
            return True

    def gallery2tag(self, gallery: GalleryURLParser, filter: str) -> list[Tag]:
        self.get(gallery.url)
        try:
            elements = self.driver.find_elements(
                By.XPATH, "//a[contains(@id, 'ta_{filter}')]".format(filter=filter)
            )
        except NoSuchElementException:
            return list()

        tag = list()
        for element in elements:
            tag.append(
                Tag(
                    filter=filter, name=element.text, href=element.get_attribute("href")
                )
            )
        return tag


class ExHDriver(EHDriver):
    def _setname(self) -> str:
        return "ExHentai"
