__all__ = ["EHDriver", "ExHDriver", "Tag"]


import os
import re
import time
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from functools import partial
from random import random


from fake_useragent import UserAgent  # type: ignore
from h2h_galleryinfo_parser import GalleryURLParser
from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException, TimeoutException
from selenium.webdriver.chrome.options import ChromiumOptions
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.remote.webelement import WebElement

from .exceptions import ClientOfflineException, InsufficientFundsException


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
            # 設定 ChromeDriver 的路徑
            driver_service = Service(ChromeDriverManager().install())

            # 設定瀏覽器參數
            options = ChromiumOptions()
            options.add_argument("--disable-extensions")
            if headless:
                options.add_argument("--headless")  # 無頭模式
                # options.add_argument("--disable-gpu")  # 禁用GPU加速
            options.add_argument(
                "--no-sandbox"
            )  # 解決DevToolsActivePort文件不存在的問題
            options.add_argument("--window-size=1600,900")
            options.add_argument("start-maximized")  # 最大化窗口
            options.add_argument("disable-infobars")
            options.add_argument("--disable-extensions")
            options.add_argument("--disable-dev-shm-usage")
            # options.add_argument("--incognito")  # 隐身模式
            # options.add_argument("--disable-dev-shm-usage")  # 覆蓋限制導致的問題
            # options.add_argument("--accept-lang=zh-TW")
            # options.add_argument("--lang=zh-TW")
            options.add_argument(
                "user-agent={ua}".format(ua=UserAgent()["google chrome"])
            )
            # options.add_argument(
            #     "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
            # )
            options.page_load_strategy = (
                "normal"  # 等待加载图片normal eager none </span></div>
            )

            # 初始化 WebDriver
            driver = webdriver.Chrome(service=driver_service, options=options)
            driver.execute_script(
                "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"
            )
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
                os.path.join(os.path.dirname(__file__), "error.txt"),
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
            # self.screenshot["login"].shot()

            # 假設跳轉後的頁面有一個具有 NAME=reset_imagelimit 的元素
            element_present = EC.presence_of_element_located(
                (By.NAME, "reset_imagelimit")
            )
            WebDriverWait(self.driver, 10).until(element_present)
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
