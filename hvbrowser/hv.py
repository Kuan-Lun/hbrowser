import asyncio
import re
from abc import ABC
from random import random
from typing import Any

from hbrowser.gallery import EHDriver
from hbrowser.gallery.utils import setup_logger

logger = setup_logger(__name__)


def genxpath(imagepath: str) -> str:
    return f'//img[@src="{imagepath}"]'


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

    @property
    async def is_isekai(self) -> bool:
        url = await self.page.evaluate("window.location.href")
        return "isekai" in url

    async def _get_path_prefix(self) -> str:
        return "/isekai" if await self.is_isekai else ""

    def searchxpath(self, srclist: list[Any] | tuple[Any, ...] | set[Any]) -> str:
        return " | ".join([genxpath(imagepath) for imagepath in srclist])

    async def goisekai(self) -> None:
        logger.info("Navigating to HentaiVerse isekai page")
        await self.get(self.url["HentaiVerse isekai"])

    async def loetterycheck(self, num: int) -> None:
        logger.info(f"Checking lottery tickets (target: {num})")
        await self.gohomepage()

        for lettory in ["Weapon Lottery", "Armor Lottery"]:
            logger.debug(f"Processing {lettory}")
            bazaar = await self.page.select("#parent_Bazaar")
            lettory_xpath = f"//div[contains(text(), '{lettory}')]"
            lettory_elements = await self.page.xpath(lettory_xpath, timeout=5)
            if not lettory_elements:
                continue
            lettory_elem = lettory_elements[0]

            # Hover bazaar then click lottery
            await bazaar.mouse_move()
            await lettory_elem.mouse_move()
            await lettory_elem.mouse_click()
            await self.page.wait(1)

            currently_elements = await self.page.xpath(
                "//*[contains(text(), 'You currently have')]", timeout=5
            )
            if not currently_elements:
                continue
            numbers: list[str] = re.findall(r"[\d,]+", currently_elements[0].text)
            currently_number = numbers[0].replace(",", "")

            hold_elements = await self.page.xpath(
                "//*[contains(text(), 'You hold')]", timeout=5
            )
            if not hold_elements:
                continue
            numbers = re.findall(r"[\d,]+", hold_elements[0].text)
            buy_number = numbers[0].replace(",", "")

            logger.info(
                f"{lettory}: Currently have {currently_number} credits, "
                f"hold {buy_number} tickets"
            )

            if int(buy_number) < num and int(currently_number) > (num * 1000):
                purchase_amount = num - int(buy_number)
                logger.info(f"Purchasing {purchase_amount} tickets for {lettory}")
                html_element = await self.page.select("#ticket_temp")
                await html_element.clear_input()
                await html_element.send_keys(str(purchase_amount))
                await self.page.evaluate("submit_buy()")
            else:
                logger.debug(
                    f"No purchase needed for {lettory} "
                    f"(tickets: {buy_number}, credits: {currently_number})"
                )

    async def monstercheck(self) -> None:
        logger.info("Starting monster check")
        await self.gohomepage()

        # 進入 Monster Lab
        logger.debug("Navigating to Monster Lab")
        bazaar = await self.page.select("#parent_Bazaar")
        monster_lab_elements = await self.page.xpath(
            "//div[contains(text(), 'Monster Lab')]", timeout=5
        )
        if not monster_lab_elements:
            return
        monster_lab = monster_lab_elements[0]

        await bazaar.mouse_move()
        await monster_lab.mouse_move()
        await monster_lab.mouse_click()
        await self.page.wait(1)

        path_prefix = await self._get_path_prefix()
        keypair: dict[str, str] = dict()
        keypair["feed"] = "food"
        keypair["drug"] = "drugs"
        for key in keypair:
            # 嘗試找到圖片元素
            xpath = self.searchxpath([f"{path_prefix}/y/monster/{key}allmonsters.png"])
            images = await self.page.xpath(xpath, timeout=2)

            # 如果存在，則執行 JavaScript
            if images:
                logger.info(f"Feeding all monsters with {keypair[key]}")
                await self.page.evaluate(f"do_feed_all('{keypair[key]}')")
                await self.page.wait(10)
            else:
                logger.debug(f"No feed all option available for {keypair[key]}")

    async def marketcheck(self, sellitems: SellItems) -> None:
        logger.info("Starting market check for selling items")

        async def marketpage() -> None:
            logger.debug("Navigating to market page")
            await self.get("https://hentaiverse.org/?s=Bazaar&ss=mk")

        async def filterpage(key: str, ischangeurl: bool) -> None:
            logger.debug(f"Filtering page by: {key}")
            filter_elements = await self.page.xpath(
                f"//div[contains(text(), '{key}')]/..", timeout=5
            )
            if filter_elements:
                await self.wait(filter_elements[0].click, ischangeurl=ischangeurl)

        async def itempage(tr_element: Any) -> bool:
            try:
                td_elements = await tr_element.query_selector_all("td")
                if len(td_elements) >= 2:
                    quantity_text = td_elements[1].text
                    iszero = quantity_text == ""
                else:
                    iszero = True
            except Exception:
                iszero = True
            return bool(iszero)

        async def resell() -> None:
            # 定位到元素
            resell_elements = await self.page.xpath(
                "//td[contains(@onclick, 'autofill_from_sell_order')]", timeout=5
            )
            if not resell_elements:
                logger.warning("Unable to find sell order element")
                return
            element = resell_elements[0]

            # 獲取 onclick 屬性值
            onclick_attr = element.attrs.get("onclick", "")

            # 使用正則表達式從屬性值中提取數字
            match = re.search(r"autofill_from_sell_order\((\d+),0,0\)", onclick_attr)
            if match:
                number = match.group(1)
                logger.debug(f"Re-listing sell order #{number}")
            else:
                logger.warning("Unable to extract number from onclick attribute")
                return

            await self.page.evaluate(f"autofill_from_sell_order({number},0,0);")

            for id_val in ["sell_order_stock_field", "sellorder_update"]:
                sell_button = await self.page.select(f"#{id_val}")
                await sell_button.click()
            await self.page.wait(10)
            await asyncio.sleep(2 * random())

            await filterpage(marketkey, ischangeurl=False)

        await self.gohomepage()
        await marketpage()

        # 存錢
        logger.info("Depositing credits to account")
        deposit_elements = await self.page.xpath(
            "//div[contains(text(), 'Account Balance')]", timeout=5
        )
        if deposit_elements:
            await deposit_elements[0].click()
        deposit_button = await self.page.select("[name='account_deposit']")
        await self.wait(deposit_button.click, ischangeurl=False)

        marketurl: dict[str, str] = dict()
        marketurl["Consumables"] = (
            "https://hentaiverse.org/?s=Bazaar&ss=mk&screen=browseitems&filter=co"
        )
        marketurl["Materials"] = (
            "https://hentaiverse.org/?s=Bazaar&ss=mk&screen=browseitems&filter=ma"
        )
        marketurl["Monster Items"] = (
            "https://hentaiverse.org/?s=Bazaar&ss=mk&screen=browseitems&filter=mo"
        )

        await filterpage("Browse Items", ischangeurl=True)
        for marketkey in marketurl:
            await filterpage(marketkey, ischangeurl=False)
            sellidx: list[int] = list()
            tr_elements = await self.page.xpath("//tr", timeout=5)
            for idx, tr_element in enumerate(tr_elements[1:]):
                td_elements = await tr_element.query_selector_all("td")
                itemname = td_elements[0].text if td_elements else ""
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
                if await itempage(tr_element):
                    continue
                sellidx.append(idx + 1)
            logger.info(f"Found {len(sellidx)} items to sell in {marketkey}")
            for idx in sellidx:
                tr_xpath = f"//tr[{idx + 1}]"
                tr_results = await self.page.xpath(tr_xpath, timeout=5)
                if tr_results:
                    await self.wait(tr_results[0].click, ischangeurl=False)
                await resell()

        await filterpage("My Sell Orders", ischangeurl=True)
        logger.info("Checking existing sell orders for re-listing")
        for marketkey in marketurl:
            await filterpage(marketkey, ischangeurl=False)
            try:
                tr_elements = await self.page.xpath("//tr", timeout=5)
                sellitemnum = len(tr_elements) - 1
                logger.debug(f"Found {sellitemnum} existing sell orders in {marketkey}")
                for n in range(sellitemnum):
                    tr_xpath = f"//tr[{n + 2}]"
                    tr_results = await self.page.xpath(tr_xpath, timeout=5)
                    if tr_results:
                        await self.wait(tr_results[0].click, ischangeurl=False)
                    await resell()
            except Exception:
                logger.debug(f"No existing sell orders found in {marketkey}")
                pass
        logger.info("Market check completed")
