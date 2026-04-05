"""E-Hentai Driver 實現"""

import asyncio
import os
import re
from random import random

from h2h_galleryinfo_parser import GalleryURLParser

from ..exceptions import ClientOfflineException, InsufficientFundsException
from .driver_base import Driver
from .models import Tag
from .utils import matchurl, wait_for_new_tab


class EHDriver(Driver):
    """E-Hentai Driver"""

    def _setname(self) -> str:
        return "E-Hentai"

    async def checkh2h(self) -> bool:
        """檢查 H@H 客戶端是否在線"""
        self.logger.info("Checking H@H client status")
        await self.get("https://e-hentai.org/hentaiathome.php")
        table = await self.page.select("#hct", timeout=10)
        header_row = await table.query_selector("tr")
        headers = await header_row.query_selector_all("th")
        status_index = [
            index for index, th in enumerate(headers) if th.text == "Status"
        ][0]
        rows = await table.query_selector_all("tr")
        for row in rows[1:]:
            cells = await row.query_selector_all("td")
            status = cells[status_index].text
            if status.lower() == "online":
                self.logger.info("H@H client is online")
                return True
        self.logger.warning("H@H client is offline")
        return False

    async def punchin(self) -> None:
        """簽到"""
        self.logger.info("Starting daily check-in")
        await self.get("https://e-hentai.org/news.php")

        # 刷新以免沒簽到成功
        await self.wait(self.page.reload, ischangeurl=False)
        self.logger.info("Check-in completed")

    async def search2gallery(self, url: str) -> list[GalleryURLParser]:
        """從搜索結果頁面提取所有 gallery URLs"""
        current_url = await self.page.evaluate("window.location.href")
        if not matchurl(current_url, url):
            await self.get(url)

        input_element = await self.page.select("#f_search")
        input_value = await input_element.apply("(el) => el.value || ''")
        if input_value == "":
            raise ValueError(
                "The value in the search box is empty. "
                "I think there are TOO MANY GALLERIES."
            )

        glist = list()
        while True:
            html_content = await self.page.get_content()
            pattern = r"https://exhentai.org/g/\d+/[A-Za-z0-9]+"
            glist += re.findall(pattern, html_content)
            try:
                element = await self.page.select("#unext", timeout=2)
            except TimeoutError:
                break
            if element.tag_name == "a":
                await self.wait(element.click, ischangeurl=True)
                await self.page.select("#unext", timeout=10)
            else:
                break
        if len(glist) == 0:
            try:
                xpath = (
                    "//*[contains(text(), 'No hits found')] | "
                    "//td[contains(text(), 'No unfiltered results found.')]"
                )
                results = await self.page.xpath(xpath, timeout=2)
                if not results:
                    raise ValueError(
                        "找出 0 個 Gallery，但頁面沒有顯示 'No hits found'。"
                    )
            except TimeoutError:
                raise ValueError("找出 0 個 Gallery，但頁面沒有顯示 'No hits found'。")
        glist = list(set(glist))
        glist = [GalleryURLParser(url) for url in glist]
        return glist

    async def search(self, key: str, isclear: bool) -> list[GalleryURLParser]:
        """搜索 galleries"""

        async def waitpage() -> None:
            await self.page.select("#f_search", timeout=10)

        try:
            input_element = await self.page.select("#f_search", timeout=2)
        except TimeoutError:
            await self.gohomepage()
            await waitpage()
            input_element = await self.page.select("#f_search")
        if isclear:
            await input_element.apply(
                "(el, k) => { el.value = k; el.dispatchEvent(new Event('input')); }",
                key,
            )
        else:
            if key != "":
                await input_element.apply(
                    "(el, k) => {"
                    " el.value = el.value + ' ' + k;"
                    " el.dispatchEvent(new Event('input'));"
                    "}",
                    key,
                )
        await asyncio.sleep(random())

        # 全總類搜尋
        elements = await self.page.xpath(
            "//div[contains(@id, 'cat_') and @data-disabled='1']", timeout=2
        )
        for element in elements:
            await element.click()
            await asyncio.sleep(random())

        button = await self.page.select('input[type="submit"][value="Search"]')
        await button.click()
        await asyncio.sleep(random())
        await waitpage()

        input_element = await self.page.select("#f_search")
        input_value = await input_element.apply("(el) => el.value || ''")
        self.logger.info(f"Search keyword: {input_value}")

        current_url = await self.page.evaluate("window.location.href")
        result = await self.search2gallery(current_url)
        self.logger.info(f"Found {len(result)} galleries")
        return result

    async def download(self, gallery: GalleryURLParser) -> bool:
        """下載 gallery"""
        self.logger.info(f"Starting download for gallery: {gallery.url}")

        await self.get(gallery.url)
        try:
            xpath_query_list = [
                "//p[contains(text(), "
                "'This gallery is unavailable due to a copyright claim "
                "by Irodori Comics.')]",
                "//input[@id='f_search']",
            ]
            xpath_query = " | ".join(xpath_query_list)
            results = await self.page.xpath(xpath_query, timeout=2)
            if results:
                self.logger.warning(f"Gallery unavailable or deleted: {gallery.url}")
                return False
        except TimeoutError:
            pass

        # 記錄現有 tabs
        existing_tabs = {t.target.target_id for t in self.browser.tabs}
        gallery_tab = self.page

        key_xpath = "//a[contains(text(), 'Archive Download')]"
        try:
            archive_links = await self.page.xpath(key_xpath, timeout=2)
            if archive_links:
                await archive_links[0].click()
            else:
                raise Exception("Archive Download not found")
        except Exception:
            self.logger.warning("Archive Download element not found, retrying download")
            await self.page.close()
            self.page = gallery_tab
            return await self.download(gallery)

        # 等待新 tab ���啟
        new_tab = await wait_for_new_tab(self.browser, existing_tabs)
        if not new_tab:
            self.logger.warning("No new tab opened, retrying download")
            return await self.download(gallery)

        # 切換到新 tab
        await new_tab.activate()
        self.page = new_tab

        # 點擊 Original，開始下載
        original_links = await self.page.xpath(
            "//a[contains(text(), 'Original')]", timeout=10
        )
        if original_links:
            await original_links[0].click()

        # 確認是否連接 H@H
        try:
            deadline = asyncio.get_event_loop().time() + 10
            while asyncio.get_event_loop().time() < deadline:
                html = await self.page.get_content()
                if (
                    "Downloads should start processing within a couple of minutes."
                    in html
                ):
                    break
                if "Your H@H client appears to be offline." in html:
                    raise ClientOfflineException()
                if "Cannot start download: Insufficient funds" in html:
                    raise InsufficientFundsException()
                await asyncio.sleep(0.5)
            else:
                html = await self.page.get_content()
                if "Cannot start download: Insufficient funds" in html:
                    raise InsufficientFundsException()
                raise TimeoutError()
        except TimeoutError:
            error_file = os.path.join(".", "error.txt")
            with open(error_file, "w", errors="ignore") as f:
                f.write(await self.page.get_content())
            retrytime = 1 * 60  # 1 minute
            self.logger.warning(
                f"Download timeout, error page saved to {error_file}, "
                f"retrying in {retrytime}s"
            )
            await self.page.close()
            self.page = gallery_tab
            await asyncio.sleep(retrytime)
            return await self.download(gallery)

        # 關閉下載 tab，切回 gallery tab
        if len(self.browser.tabs) > 1:
            await self.page.close()
            await asyncio.sleep(random())
            self.page = gallery_tab
            await gallery_tab.activate()
            await asyncio.sleep(random())
        else:
            self.logger.error(
                f"Tab anomaly: only {len(self.browser.tabs)} tab(s) remaining"
            )
        self.logger.info(f"Gallery downloaded successfully: {gallery.url}")
        return True

    async def gallery2tag(self, gallery: GalleryURLParser, filter: str) -> list[Tag]:
        """從 gallery 頁面提取指定 filter 的 tags"""
        await self.get(gallery.url)
        try:
            elements = await self.page.xpath(
                f"//a[contains(@id, 'ta_{filter}')]", timeout=2
            )
        except TimeoutError:
            return list()

        tag = list()
        for element in elements:
            tag.append(
                Tag(
                    filter=filter,
                    name=element.text,
                    href=element.attrs.get("href", ""),
                )
            )
        return tag
