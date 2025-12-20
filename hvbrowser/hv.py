import re
import time
from abc import ABC
from random import random
from typing import Any

from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.by import By

from hbrowser.gallery import EHDriver
from hbrowser.gallery.utils import setup_logger

logger = setup_logger(__name__)


def genxpath(imagepath: str) -> str:
    return f'//img[@src="{imagepath}"]'


def searchxpath_fun(srclist: list[Any] | tuple[Any, ...] | set[Any]) -> str:
    return " | ".join(
        [genxpath(s + imagepath) for imagepath in srclist for s in ["", "/isekai"]]
    )


class BSItems(ABC):
    def __init__(
        self,
        consumables: list[str] = list(),
        materials: list[str] = list(),
        trophies: list[str] = list(),
        artifacts: list[str] = list(),
        figures: list[str] = list(),
        monster_items: list[str] = list(),
    ) -> None:
        self.consumables = consumables
        self.materials = materials
        self.trophies = trophies
        self.artifacts = artifacts
        self.figures = figures
        self.monster_items = monster_items


class SellItems(BSItems):
    pass


class BuyItems(BSItems):
    pass


class HVDriver(EHDriver):
    def _setname(self) -> str:
        return "HentaiVerse"

    def _setlogin(self) -> str:
        return "My Home"

    def goisekai(self) -> None:
        logger.info("Navigating to HentaiVerse isekai page")
        self.get(self.url["HentaiVerse isekai"])

    def loetterycheck(self, num: int) -> None:
        logger.info(f"Checking lottery tickets (target: {num})")
        self.gohomepage()

        for lettory in ["Weapon Lottery", "Armor Lottery"]:
            logger.debug(f"Processing {lettory}")
            element: dict[str, Any] = dict()
            element["Bazaar"] = self.driver.find_element(By.ID, "parent_Bazaar")
            element[lettory] = self.driver.find_element(
                By.XPATH, f"//div[contains(text(), '{lettory}')]"
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

            logger.info(
                f"{lettory}: Currently have {currently_number} credits, "
                f"hold {buy_number} tickets"
            )

            if int(buy_number) < num and int(currently_number) > (num * 1000):
                purchase_amount = num - int(buy_number)
                logger.info(f"Purchasing {purchase_amount} tickets for {lettory}")
                html_element = self.driver.find_element(By.ID, "ticket_temp")
                html_element.clear()
                html_element.send_keys(purchase_amount)
                self.driver.execute_script("submit_buy()")
            else:
                logger.debug(
                    f"No purchase needed for {lettory} "
                    f"(tickets: {buy_number}, credits: {currently_number})"
                )

    def monstercheck(self) -> None:
        logger.info("Starting monster check")
        self.gohomepage()

        # 進入 Monster Lab
        logger.debug("Navigating to Monster Lab")
        element: dict[str, Any] = dict()
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

        keypair: dict[str, str] = dict()
        keypair["feed"] = "food"
        keypair["drug"] = "drugs"
        for key in keypair:
            # 嘗試找到圖片元素
            images = self.driver.find_elements(
                By.XPATH,
                searchxpath_fun([f"/y/monster/{key}allmonsters.png"]),
            )

            # 如果存在，則執行 JavaScript
            if images:
                logger.info(f"Feeding all monsters with {keypair[key]}")
                self.driver.execute_script(f"do_feed_all('{keypair[key]}')")
                self.driver.implicitly_wait(10)  # 隱式等待，最多等待10秒
            else:
                logger.debug(f"No feed all option available for {keypair[key]}")

    def marketcheck(self, sellitems: SellItems) -> None:
        logger.info("Starting market check for selling items")

        def marketpage() -> None:
            # 進入 Market
            logger.debug("Navigating to market page")
            self.get("https://hentaiverse.org/?s=Bazaar&ss=mk")

        def filterpage(key: str, ischangeurl: bool) -> None:
            logger.debug(f"Filtering page by: {key}")
            self.wait(
                self.driver.find_element(
                    By.XPATH, f"//div[contains(text(), '{key}')]/.."
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
            return bool(iszero)

        def resell() -> None:
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
                logger.debug(f"Re-listing sell order #{number}")
            else:
                logger.warning("Unable to extract number from onclick attribute")
                return
            # 假設 driver 是你的 WebDriver 實例
            self.driver.execute_script(f"autofill_from_sell_order({number},0,0);")

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
        logger.info("Depositing credits to account")
        self.driver.find_element(
            By.XPATH, "//div[contains(text(), 'Account Balance')]"
        ).click()
        self.wait(
            self.driver.find_element(By.NAME, "account_deposit").click,
            ischangeurl=False,
        )

        marketurl: dict[str, str] = dict()
        # Consumables
        marketurl["Consumables"] = (
            "https://hentaiverse.org/?s=Bazaar&ss=mk&screen=browseitems&filter=co"
        )
        # Materials
        marketurl["Materials"] = (
            "https://hentaiverse.org/?s=Bazaar&ss=mk&screen=browseitems&filter=ma"
        )
        # Monster Items
        marketurl["Monster Items"] = (
            "https://hentaiverse.org/?s=Bazaar&ss=mk&screen=browseitems&filter=mo"
        )

        filterpage("Browse Items", ischangeurl=True)
        for marketkey in marketurl:
            filterpage(marketkey, ischangeurl=False)
            sellidx: list[int] = list()
            # 使用find_elements方法获取页面上所有<tr>元素
            tr_elements = self.driver.find_elements(By.XPATH, "//tr")
            for idx, tr_element in enumerate(tr_elements[1:]):
                itemname = tr_element.find_element(By.XPATH, ".//td[1]").text
                thecheckitemlist: list[str]
                match marketkey:
                    case "Consumables":
                        thecheckitemlist = sellitems.consumables
                    case "Materials":
                        thecheckitemlist = sellitems.materials
                    case "Trophies":
                        thecheckitemlist = sellitems.trophies
                    case "Artifacts":
                        thecheckitemlist = sellitems.artifacts
                    case "Figures":
                        thecheckitemlist = sellitems.figures
                    case "Monster Items":
                        thecheckitemlist = sellitems.monster_items
                    case _:
                        raise KeyError()
                if itemname not in thecheckitemlist:
                    continue
                if itempage():
                    continue
                sellidx.append(idx + 1)
            logger.info(f"Found {len(sellidx)} items to sell in {marketkey}")
            for idx in sellidx:
                tr_element = self.driver.find_element(By.XPATH, f"//tr[{idx + 1}]")
                self.wait(tr_element.click, ischangeurl=False)
                resell()

        filterpage("My Sell Orders", ischangeurl=True)
        logger.info("Checking existing sell orders for re-listing")
        for marketkey in marketurl:
            filterpage(marketkey, ischangeurl=False)
            try:
                tr_elements = self.driver.find_elements(By.XPATH, "//tr")
                sellitemnum = len(tr_elements) - 1
                logger.debug(f"Found {sellitemnum} existing sell orders in {marketkey}")
                for n in range(sellitemnum):
                    tr_element = self.driver.find_element(By.XPATH, f"//tr[{n + 2}]")
                    self.wait(tr_element.click, ischangeurl=False)
                    resell()
            except NoSuchElementException:
                logger.debug(f"No existing sell orders found in {marketkey}")
                pass
        logger.info("Market check completed")
