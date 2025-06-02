from collections import defaultdict

from selenium.webdriver.common.by import By
from selenium.webdriver.remote.webdriver import WebDriver

from .hv import HVDriver, searchxpath_fun
from .hv_battle_actionmanager import ElementActionManager


class ItemProvider:
    def __init__(self, driver: HVDriver) -> None:
        self.hvdriver: HVDriver = driver
        self._checked_items: dict[str, str] = defaultdict(lambda: "available")

    @property
    def driver(self) -> WebDriver:
        return self.hvdriver.driver

    def get_item_status(self, item: str) -> str:
        """
        回傳 'available', 'unavailable', 'not_found'
        """
        if self._checked_items[item] == "not_found":
            return "not_found"

        item_divs = self.driver.find_elements(By.XPATH, f"//div/div[text()='{item}']")
        if not item_divs:
            self._checked_items[item] = "not_found"
            return "not_found"

        for div in item_divs:
            parent = div.find_element(By.XPATH, "./ancestor::div[2]")
            if parent.get_attribute("id") and parent.get_attribute("onclick"):
                return "available"
        return "unavailable"

    def use(self, item: str) -> bool:
        if self._checked_items[item] == "not_found":
            return False

        if self.get_item_status(item) == "unavailable":
            return False

        item_menu_list = self.driver.find_elements(
            By.XPATH,
            searchxpath_fun(["/y/battle/items_n.png"]),
        )
        if not item_menu_list:
            return False
        ElementActionManager(self.hvdriver).click(item_menu_list[0])

        item_button_list = self.driver.find_elements(
            By.XPATH,
            "//div[@class=\"fc2 fal fcb\"]/div[contains(text(), '{key}')]".format(
                key=item
            ),
        )
        if not item_button_list:
            return False
        ElementActionManager(self.hvdriver).click_and_wait_log(item_button_list[0])
        return True
