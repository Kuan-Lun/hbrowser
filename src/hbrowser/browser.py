__all__ = ["DriverPass", "EHDriver", "ExHDriver", "HVDriver"]


import os
import sys
import time
import re
from abc import ABC, abstractmethod
from functools import partial
from random import random
from datetime import datetime, timedelta


from fake_useragent import UserAgent  # type: ignore
from h2h_galleryinfo_parser import GalleryURLParser
from selenium import webdriver
from selenium.webdriver.remote.webelement import WebElement
from selenium.webdriver.chrome.options import ChromiumOptions
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains
from selenium.common.exceptions import (
    NoSuchElementException,
    JavascriptException,
    TimeoutException,
)
from webdriver_manager.chrome import ChromeDriverManager


def beep_os_independent():
    if sys.platform.startswith("linux") or sys.platform == "darwin":
        # 对于 Linux 和 macOS
        os.system('echo -n "\a"')  # 这会在 shell 中使用 echo 命令来发出提示音
    elif sys.platform == "win32":
        # 对于 Windows
        import winsound

        winsound.MessageBeep()  # 使用 winsound 模块发出默认提示音


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
            self.driver.find_element(By.XPATH, key).click()
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
            successkey = "//p[contains(text(), 'Downloads should start processing within a couple of minutes.')]"
            failkey = "//p[contains(text(), 'Your H@H client appears to be offline.')]"
            waitkey = (
                "//p[contains(text(), 'Cannot start download: Insufficient funds')]"
            )
            try:
                WebDriverWait(self.driver, 10).until(
                    lambda driver: EC.presence_of_element_located(
                        (By.XPATH, successkey)
                    )(driver)
                    or EC.presence_of_element_located((By.XPATH, failkey))(driver)
                    or EC.presence_of_element_located((By.XPATH, waitkey))(driver)
                )
            except TimeoutException:
                print("TimeoutException")
                self.driver.close()
                self.driver.switch_to.window(gallerywindow)
                print("Retry again.")
                return self.download(gallery)
            try:
                self.driver.find_element(By.XPATH, successkey)
            except NoSuchElementException:
                try:
                    self.driver.find_element(By.XPATH, failkey)
                    print("H@H client appears to be offline. Retry after 30 minutes.")
                    retrytime = 30 * 60  # 30 minutes
                except NoSuchElementException:
                    print("Insufficient funds. Retry after 4 hours.")
                    retrytime = 4 * 60 * 60  # 4 hours
                self.driver.close()
                time.sleep(random())
                self.driver.switch_to.window(gallerywindow)
                try:
                    time.sleep(retrytime)
                except KeyboardInterrupt:
                    print("Sleep is interrupted. Continue.")
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


genxpath = lambda imagepath: '//img[@src="{imagepath}"]'.format(imagepath=imagepath)
searchxpath_fun = lambda srclist: " | ".join(
    [genxpath(s + imagepath) for imagepath in srclist for s in ["", "/isekai"]]
)


class HVDriver(EHDriver):
    def _setname(self) -> str:
        return "HentaiVerse"

    def _setlogin(self) -> str:
        return "My Home"

    def goisekai(self) -> None:
        self.get(self.url["HentaiVerse isekai"])

    def battle(self) -> None:
        def ponychart() -> bool:
            try:
                self.driver.find_element(By.ID, "riddlesubmit")
            except NoSuchElementException:
                return False
            beep_os_independent()
            time.sleep(60)
            return False

        def click2newlog(element: WebElement) -> None:
            html = self.driver.find_element(By.ID, "textlog").get_attribute("outerHTML")
            actions = ActionChains(self.driver)
            actions.move_to_element(element).click().perform()
            time.sleep(0.05)
            n: float = 0
            while html == self.driver.find_element(By.ID, "textlog").get_attribute(
                "outerHTML"
            ):
                time.sleep(0.05)
                n += 0.05
                if n == 10:
                    raise TimeoutError("I don't know what happened.")

        def getrate(key: str) -> float:
            match key:
                case "HP":
                    searchxpath = searchxpath_fun(
                        ["/y/bar_bgreen.png", "/y/bar_dgreen.png"]
                    )
                    factor = 100
                case "MP":
                    searchxpath = searchxpath_fun(["/y/bar_blue.png"])
                    factor = 100
                case "SP":
                    searchxpath = searchxpath_fun(["/y/bar_red.png"])
                    factor = 100
                case "Overcharge":
                    searchxpath = searchxpath_fun(["/y/bar_orange.png"])
                    factor = 250

            img_element = self.driver.find_element(By.XPATH, searchxpath)
            style_attribute = img_element.get_attribute("style")
            width_value_match = re.search(r"width:\s*(\d+)px", style_attribute)
            if width_value_match is None:
                raise ValueError("width_value_match is None")
            width_value_match = width_value_match.group(1)  # type: ignore
            return factor * (int(width_value_match) - 1) / (414 - 1)  # type: ignore

        def clickitem(key: str) -> bool:
            try:
                element = self.driver.find_element(
                    By.XPATH,
                    searchxpath_fun(["/y/battle/items_n.png"]),
                )
                element.click()
            except NoSuchElementException:
                return False
            try:
                click2newlog(
                    self.driver.find_element(
                        By.XPATH,
                        "//div[@class=\"fc2 fal fcb\"]/div[contains(text(), '{key}')]".format(
                            key=key
                        ),
                    )
                )
                return True
            except NoSuchElementException:
                return False

        def clickskill(key: str, iswait=True) -> bool:
            def click_this_skill(skillstring: str) -> None:
                element = self.driver.find_element(By.XPATH, skillstring)
                if iswait:
                    click2newlog(element)
                else:
                    actions = ActionChains(self.driver)
                    actions.move_to_element(element).click().perform()
                    time.sleep(0.05)

            skillstring = (
                "//div[not(@style)]/div/div[contains(text(), '{key}')]".format(key=key)
            )
            try:
                click_this_skill(skillstring)
            except JavascriptException:
                element = self.driver.find_element(
                    By.XPATH, searchxpath_fun(["/y/battle/skill_n.png"])
                )
                element.click()
                try:
                    click_this_skill(skillstring)
                except JavascriptException:
                    element = self.driver.find_element(
                        By.XPATH, searchxpath_fun(["/y/battle/sbsel_spells_n.png"])
                    )
                    element.click()
                    click_this_skill(skillstring)
            except NoSuchElementException:
                return False
            return True

        def checkstat(key: str) -> bool:
            match key:
                case "HP":
                    if getrate("HP") < 50:
                        for fun in [
                            partial(clickitem, "Health Potion"),
                            partial(clickskill, "Cure"),
                            partial(clickskill, "Full-Cure"),
                            partial(clickitem, "Health Elixir"),
                            partial(clickitem, "Last Elixir"),
                        ]:
                            if getrate("HP") < 50:
                                if not fun():
                                    continue
                                return True
                    try:
                        self.driver.find_element(
                            By.XPATH, searchxpath_fun(["/y/e/healthpot.png"])
                        )
                    except NoSuchElementException:
                        return clickitem("Health Draught")

                case "MP":
                    if getrate("MP") < 50:
                        for key in ["Mana Potion", "Mana Elixir", "Last Elixir"]:
                            if clickitem(key):
                                return True
                    try:
                        self.driver.find_element(
                            By.XPATH, searchxpath_fun(["/y/e/manapot.png"])
                        )
                    except NoSuchElementException:
                        return clickitem("Mana Draught")
                case "SP":
                    if getrate("SP") < 50:
                        for key in ["Spirit Potion", "Spirit Elixir", "Last Elixir"]:
                            if clickitem(key):
                                return True
                    try:
                        self.driver.find_element(
                            By.XPATH, searchxpath_fun(["/y/e/spiritpot.png"])
                        )
                    except NoSuchElementException:
                        return clickitem("Spirit Draught")
                case "Overcharge":
                    clickspirit = partial(
                        click2newlog, self.driver.find_element(By.ID, "ckey_spirit")
                    )
                    if getrate("Overcharge") > 240 and getrate("SP") > 50:
                        try:
                            self.driver.find_element(
                                By.XPATH, searchxpath_fun(["/y/battle/spirit_a.png"])
                            )
                        except NoSuchElementException:
                            clickspirit()
                            return True
                    if countmonster() >= 5 and getrate("Overcharge") < 180:
                        try:
                            self.driver.find_element(
                                By.XPATH, searchxpath_fun(["/y/battle/spirit_a.png"])
                            )
                            clickspirit()
                            return True
                        except NoSuchElementException:
                            return False
            return False

        def nextbattle() -> bool:
            try:
                click2newlog(
                    self.driver.find_element(
                        By.XPATH,
                        searchxpath_fun(
                            [
                                "/y/battle/arenacontinue.png",
                                "/y/battle/grindfestcontinue.png",
                            ]
                        ),
                    )
                )
                return True
            except NoSuchElementException:
                return False

        def countmonster() -> int:
            count = 0
            for n in range(10):
                count += (
                    len(
                        self.driver.find_elements(
                            By.XPATH,
                            '//div[@id="mkey_{n}" and not(.//img[@src="/y/s/nbardead.png"]) and not(.//img[@src="/isekai/y/s/nbardead.png"])]'.format(
                                n=n
                            ),
                        )
                    )
                    > 0
                )
            return count

        def attack() -> bool:
            if getrate("Overcharge") > 220:
                try:
                    self.driver.find_element(
                        By.XPATH, searchxpath_fun(["/y/battle/spirit_a.png"])
                    )
                    if countmonster() >= 5:
                        clickskill("Orbital Friendship Cannon", iswait=False)
                except NoSuchElementException:
                    pass
            for n in [2, 1, 3, 5, 4, 6, 8, 7, 9, 0]:
                print(n)
                try:
                    self.driver.find_element(
                        By.XPATH,
                        '//div[@id="mkey_{n}" and not(.//img[@src="/y/s/nbardead.png"]) and not(.//img[@src="/isekai/y/s/nbardead.png"])]'.format(
                            n=n
                        ),
                    )
                    if getrate("MP") > 50:
                        try:
                            self.driver.find_element(
                                By.XPATH,
                                '//div[@id="mkey_{n}" and not(.//img[@src="/y/e/imperil.png"]) and not(.//img[@src="/isekai/y/e/imperil.png"])]'.format(
                                    n=n
                                ),
                            )
                            clickskill("Imperil", iswait=False)
                        except NoSuchElementException:
                            pass
                    click2newlog(
                        self.driver.find_element(
                            By.XPATH, '//div[@id="mkey_{n}"]'.format(n=n)
                        )
                    )
                    return True
                except NoSuchElementException:
                    pass
            return False

        while True:
            print("arenacontinue")
            if nextbattle():
                continue
            print("ponychart")
            if ponychart():
                continue
            try:
                self.driver.find_element(
                    By.XPATH, searchxpath_fun(["/y/e/channeling.png"])
                )
                clickskill("Heartseeker")
                continue
            except NoSuchElementException:
                pass
            # 開始戰鬥
            try:
                ending = self.driver.find_element(
                    By.XPATH, searchxpath_fun(["/y/battle/finishbattle.png"])
                )
                break
            except NoSuchElementException:
                iscontinue = False
                for key in ["HP", "MP", "SP", "Overcharge"]:
                    print(key)
                    iscontinue |= checkstat(key)
                    if iscontinue:
                        break
                if iscontinue:
                    continue
                try:
                    print("regen")
                    self.driver.find_element(
                        By.XPATH, searchxpath_fun(["/y/e/regen.png"])
                    )
                except NoSuchElementException:
                    clickskill("Regen")
                    continue
                try:
                    print("heartseeker")
                    self.driver.find_element(
                        By.XPATH, searchxpath_fun(["/y/e/heartseeker.png"])
                    )
                except NoSuchElementException:
                    clickskill("Heartseeker")
                    continue
                if attack():
                    continue

        actions = ActionChains(self.driver)
        actions.move_to_element(ending).click().perform()

    def loetterycheck(self) -> None:
        self.gohomepage()

        for lettory in ["Weapon Lottery", "Armor Lottery"]:
            element = dict()
            element["Bazaar"] = self.driver.find_element(By.ID, "parent_Bazaar")
            element[lettory] = self.driver.find_element(
                By.XPATH, "//div[contains(text(), '{lettory}')]".format(lettory=lettory)
            )
            actions = ActionChains(self.driver)
            self.wait(
                actions.move_to_element(element["Bazaar"])
                .move_to_element(element[lettory])
                .click()
                .perform,
                ischangeurl=False,
            )

            html_element = self.driver.find_element(
                By.XPATH, "//*[contains(text(), 'You currently have')]"
            )

            numbers: list[str] = re.findall(r"[\d,]+", html_element.text)
            currently_number = numbers[0].replace(",", "")
            html_element = self.driver.find_element(
                By.XPATH, "//*[contains(text(), 'You hold')]"
            )

            numbers = re.findall(r"[\d,]+", html_element.text)
            buy_number = numbers[0].replace(",", "")

            if int(buy_number) < 200 and int(currently_number) > (200 * 1000):
                html_element = self.driver.find_element(By.ID, "ticket_temp")
                html_element.clear()
                html_element.send_keys(200 - int(buy_number))
                self.driver.execute_script("submit_buy()")

    def monstercheck(self) -> None:
        self.gohomepage()

        # 進入 Monster Lab
        element = dict()
        element["Bazaar"] = self.driver.find_element(By.ID, "parent_Bazaar")
        element["Monster Lab"] = self.driver.find_element(
            By.XPATH, "//div[contains(text(), 'Monster Lab')]"
        )
        actions = ActionChains(self.driver)
        self.wait(
            actions.move_to_element(element["Bazaar"])
            .move_to_element(element["Monster Lab"])
            .click()
            .perform,
            ischangeurl=False,
        )

        keypair = dict()
        keypair["feed"] = "food"
        keypair["drug"] = "drugs"
        for key in keypair:
            # 嘗試找到圖片元素
            images = self.driver.find_elements(
                By.XPATH,
                searchxpath_fun(["/y/monster/{key}allmonsters.png".format(key=key)]),
            )

            # 如果存在，則執行 JavaScript
            if images:
                self.driver.execute_script(
                    "do_feed_all('{key}')".format(key=keypair[key])
                )
                self.driver.implicitly_wait(10)  # 隱式等待，最多等待10秒

    def marketcheck(self) -> None:
        def marketpage() -> None:
            # 進入 Market
            self.get("https://hentaiverse.org/?s=Bazaar&ss=mk")

        def filterpage(key: str, ischangeurl: bool) -> None:
            self.wait(
                self.driver.find_element(
                    By.XPATH, "//div[contains(text(), '{key}')]/..".format(key=key)
                ).click,
                ischangeurl=ischangeurl,
            )

        def itempage() -> bool:
            try:
                # 获取<tr>元素中第二个<td>的文本
                quantity_text = tr_element.find_element(By.XPATH, ".//td[2]").text

                # 检查数量是否非零
                iszero = quantity_text == ""
            except NoSuchElementException:
                iszero = True
            return iszero

        def resell():
            # 定位到元素
            element = self.driver.find_element(
                By.XPATH, "//td[contains(@onclick, 'autofill_from_sell_order')]"
            )

            # 獲取 onclick 屬性值
            onclick_attr = element.get_attribute("onclick")

            # 使用正則表達式從屬性值中提取數字
            match = re.search(r"autofill_from_sell_order\((\d+),0,0\)", onclick_attr)
            if match:
                number = match.group(1)
            else:
                print("未能從 onclick 屬性中提取數字")
            # 假設 driver 是你的 WebDriver 實例
            self.driver.execute_script(
                "autofill_from_sell_order({number},0,0);".format(number=number)
            )

            for id in ["sell_order_stock_field", "sellorder_update"]:
                Sell_button = self.driver.find_element(
                    By.ID, id
                )  # 查找方法可能需要根據實際情況調整
                Sell_button.click()
            self.driver.implicitly_wait(10)  # 隱式等待，最多等待10秒
            time.sleep(2 * random())

            filterpage(marketkey, ischangeurl=False)

        self.gohomepage()
        marketpage()

        # 存錢
        self.driver.find_element(
            By.XPATH, "//div[contains(text(), 'Account Balance')]"
        ).click()
        self.wait(
            self.driver.find_element(By.NAME, "account_deposit").click,
            ischangeurl=False,
        )

        marketurl = dict()
        itemlist = dict()
        # Consumables
        marketurl["Consumables"] = (
            "https://hentaiverse.org/?s=Bazaar&ss=mk&screen=browseitems&filter=co"
        )
        itemlist["Consumables"] = [
            "Energy Drink",
            "Caffeinated Candy",
            "Last Elixir",
            "Flower Vase",
            "Bubble-Gum",
        ]
        # Materials
        marketurl["Materials"] = (
            "https://hentaiverse.org/?s=Bazaar&ss=mk&screen=browseitems&filter=ma"
        )
        itemlist["Materials"] = [
            "Low-Grade Cloth",
            "Mid-Grade Cloth",
            "High-Grade Cloth",
            "Low-Grade Leather",
            "Mid-Grade Leather",
            "High-Grade Leather",
            "Low-Grade Metals",
            "Mid-Grade Metals",
            "High-Grade Metals",
            "Low-Grade Wood",
            "Mid-Grade Wood",
            "High-Grade Wood",
            "Crystallized Phazon",
            "Shade Fragment",
            "Repurposed Actuator",
            "Defense Matrix Modulator",
            "Binding of Slaughter",
            "Binding of Balance",
            "Binding of Destruction",
            "Binding of Focus",
            "Binding of Protection",
            "Binding of the Fleet",
            "Binding of the Barrier",
            "Binding of the Nimble",
            "Binding of the Elementalist",
            "Binding of the Heaven-sent",
            "Binding of the Demon-fiend",
            "Binding of the Curse-weaver",
            "Binding of the Earth-walker",
            "Binding of Surtr",
            "Binding of Niflheim",
            "Binding of Mjolnir",
            "Binding of Freyr",
            "Binding of Heimdall",
            "Binding of Fenrir",
            "Binding of Dampening",
            "Binding of Stoneskin",
            "Binding of Deflection",
            "Binding of the Fire-eater",
            "Binding of the Frost-born",
            "Binding of the Thunder-child",
            "Binding of the Wind-waker",
            "Binding of the Thrice-blessed",
            "Binding of the Spirit-ward",
            "Binding of the Ox",
            "Binding of the Raccoon",
            "Binding of the Cheetah",
            "Binding of the Turtle",
            "Binding of the Fox",
            "Binding of the Owl",
            "Binding of Warding",
            "Binding of Negation",
            "Binding of Isaac",
            "Binding of Friendship",
            "Legendary Weapon Core",
            "Legendary Staff Core",
            "Legendary Armor Core",
            "Voidseeker Shard",
            "Aether Shard",
            "Featherweight Shard",
            "Amnesia Shard",
        ]
        # Monster Items
        # marketurl['Monster Items'] = 'https://hentaiverse.org/?s=Bazaar&ss=mk&screen=browseitems&filter=mo'
        # itemidlist['Monster Items'] = [50001, 50002, 50003, 50004, 50005, 50006]#,
        #                                #50011, 50012, 50013, 50014, 50015, 50016]

        filterpage("Browse Items", ischangeurl=True)
        for marketkey in marketurl:
            filterpage(marketkey, ischangeurl=False)
            sellidx = list()
            # 使用find_elements方法获取页面上所有<tr>元素
            tr_elements = self.driver.find_elements(By.XPATH, "//tr")
            for idx, tr_element in enumerate(tr_elements[1:]):
                itemname = tr_element.find_element(By.XPATH, ".//td[1]").text
                if itemname not in itemlist[marketkey]:
                    continue
                if itempage():
                    continue
                sellidx.append(idx + 1)
            for idx in sellidx:
                tr_element = self.driver.find_element(
                    By.XPATH, "//tr[{n}]".format(n=idx + 1)
                )
                self.wait(tr_element.click, ischangeurl=False)
                resell()

        filterpage("My Sell Orders", ischangeurl=True)
        for marketkey in marketurl:
            filterpage(marketkey, ischangeurl=False)
            try:
                tr_elements = self.driver.find_elements(By.XPATH, "//tr")
                sellitemnum = len(tr_elements) - 1
                for n in range(sellitemnum):
                    tr_element = self.driver.find_element(
                        By.XPATH, "//tr[{n}]".format(n=n + 2)
                    )
                    self.wait(tr_element.click, ischangeurl=False)
                    resell()
            except NoSuchElementException:
                pass