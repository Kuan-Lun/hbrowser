"""E-Hentai Driver 實現"""

import os
import re
import time
from functools import partial
from random import random
from typing import Any

from h2h_galleryinfo_parser import GalleryURLParser
from selenium.common.exceptions import NoSuchElementException, TimeoutException
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

from ..exceptions import ClientOfflineException, InsufficientFundsException
from .driver_base import Driver
from .models import Tag
from .utils import find_new_window, matchurl


class EHDriver(Driver):
    """E-Hentai Driver"""

    def _setname(self) -> str:
        return "E-Hentai"

    def _setlogin(self) -> str:
        return "My Home"

    def checkh2h(self) -> bool:
        """檢查 H@H 客戶端是否在線"""
        self.logger.info("Checking H@H client status")
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
                self.logger.info("H@H client is online")
                return True
        self.logger.warning("H@H client is offline")
        return False

    def punchin(self) -> None:
        """簽到"""
        self.logger.info("Starting daily check-in")
        # 嘗試簽到
        self.get("https://e-hentai.org/news.php")

        # 刷新以免沒簽到成功
        self.wait(self.driver.refresh, ischangeurl=False)
        self.logger.info("Check-in completed")

    def search2gallery(self, url: str) -> list[GalleryURLParser]:
        """從搜索結果頁面提取所有 gallery URLs"""
        if not matchurl(self.driver.current_url, url):
            self.get(url)

        input_element = self.driver.find_element(By.ID, "f_search")
        input_value = input_element.get_attribute("value")
        if input_value == "":
            raise ValueError(
                "The value in the search box is empty. "
                "I think there are TOO MANY GALLERIES."
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
                xpath = (
                    "//*[contains(text(), 'No hits found')] | "
                    "//td[contains(text(), 'No unfiltered results found.')]"
                )
                self.driver.find_element(By.XPATH, xpath)
            except NoSuchElementException:
                raise ValueError("找出 0 個 Gallery，但頁面沒有顯示 'No hits found'。")
        glist = list(set(glist))
        glist = [GalleryURLParser(url) for url in glist]
        return glist

    def search(self, key: str, isclear: bool) -> list[GalleryURLParser]:
        """搜索 galleries"""

        def waitpage() -> None:
            element_present: Any = EC.presence_of_element_located((By.ID, "f_search"))
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
        self.logger.info(f"Search keyword: {input_value}")

        result = self.search2gallery(self.driver.current_url)
        self.logger.info(f"Found {len(result)} galleries")
        return result

    def download(self, gallery: GalleryURLParser) -> bool:
        """下載 gallery"""
        self.logger.info(f"Starting download for gallery: {gallery.url}")

        def _check_ekey(driver: Any, ekey: str) -> Any:
            return EC.presence_of_element_located((By.XPATH, ekey))(
                driver
            ) or EC.visibility_of_element_located((By.XPATH, ekey))(driver)

        def check_download_success_by_element(driver: Any) -> Any:
            ekey = (
                "//p[contains(text(), "
                "'Downloads should start processing within a couple of minutes.'"
                ")]"
            )
            return _check_ekey(driver, ekey)

        def check_client_offline_by_element(driver: Any) -> Any:
            ekey = "//p[contains(text(), 'Your H@H client appears to be offline.')]"
            try:
                return _check_ekey(driver, ekey)
            except NoSuchElementException:
                raise ClientOfflineException()

        def check_insufficient_funds_by_element(driver: Any) -> Any:
            ekey = "//p[contains(text(), 'Cannot start download: Insufficient funds')]"
            try:
                return _check_ekey(driver, ekey)
            except NoSuchElementException:
                raise InsufficientFundsException()

        self.get(gallery.url)
        try:
            xpath_query_list = [
                "//p[contains(text(), "
                "'This gallery is unavailable due to a copyright claim "
                "by Irodori Comics.')]",
                "//input[@id='f_search']",
            ]
            xpath_query = " | ".join(xpath_query_list)
            self.driver.find_element(By.XPATH, xpath_query)
            self.logger.warning(f"Gallery unavailable or deleted: {gallery.url}")
            return False
        except NoSuchElementException:
            gallerywindow = self.driver.current_window_handle
            existing_windows = set(self.driver.window_handles)
            key = "//a[contains(text(), 'Archive Download')]"
            try:
                self.driver.find_element(By.XPATH, key).click()
            except NoSuchElementException:
                self.logger.warning(
                    "Archive Download element not found, retrying download"
                )
                self.driver.close()
                self.driver.switch_to.window(gallerywindow)
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
                error_file = os.path.join(".", "error.txt")
                with open(error_file, "w", errors="ignore") as f:
                    f.write(self.driver.page_source)
                retrytime = 1 * 60  # 1 minute
                self.logger.warning(
                    f"Download timeout, error page saved to {error_file}, "
                    f"retrying in {retrytime}s"
                )
                self.driver.close()
                self.driver.switch_to.window(gallerywindow)
                time.sleep(retrytime)
                return self.download(gallery)
            if len(self.driver.current_window_handle) > 1:
                self.driver.close()
                time.sleep(random())
                self.driver.switch_to.window(gallerywindow)
                time.sleep(random())
            else:
                self.logger.error(
                    f"Window handle anomaly: {self.driver.current_window_handle}"
                )
            self.logger.info(f"Gallery downloaded successfully: {gallery.url}")
            return True

    def gallery2tag(self, gallery: GalleryURLParser, filter: str) -> list[Tag]:
        """從 gallery 頁面提取指定 filter 的 tags"""
        self.get(gallery.url)
        try:
            elements = self.driver.find_elements(
                By.XPATH, f"//a[contains(@id, 'ta_{filter}')]"
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
