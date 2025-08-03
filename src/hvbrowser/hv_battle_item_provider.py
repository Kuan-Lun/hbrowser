from collections import defaultdict

from selenium.webdriver.common.by import By
from selenium.webdriver.remote.webdriver import WebDriver, WebElement

from .hv import HVDriver, searchxpath_fun
from .hv_battle_action_manager import ElementActionManager

GEM_ITEMS = {"Mystic Gem", "Health Gem", "Mana Gem", "Spirit Gem"}


class ItemProvider:
    def __init__(self, driver: HVDriver) -> None:
        self.hvdriver: HVDriver = driver
        self._checked_items: dict[str, str] = defaultdict(lambda: "available")

    @property
    def driver(self) -> WebDriver:
        return self.hvdriver.driver

    @property
    def items_menu_web_element(self) -> WebElement:
        return self.hvdriver.driver.find_element(By.ID, "ckey_items")

    def click_items_menu(self) -> None:
        ElementActionManager(self.hvdriver).click(self.items_menu_web_element)

    def is_open_items_menu(self) -> bool:
        """
        Check if the items menu is open.
        """
        items_menum = self.items_menu_web_element.get_attribute("src") or ""
        return "items_s.png" in items_menum

    def get_pane_items(self) -> WebElement:
        return self.hvdriver.driver.find_element(By.ID, "pane_item")

    def get_item_status(self, item: str) -> str:
        """
        回傳 'available', 'unavailable', 'not_found'
        """
        if self._checked_items[item] == "not_found":
            return "not_found"

        item_divs = self.get_pane_items().find_elements(
            By.XPATH, f"//div/div[text()='{item}']"
        )
        if not item_divs:
            if item not in GEM_ITEMS:
                self._checked_items[item] = "not_found"
            return "not_found"

        for div in item_divs:
            parent = div.find_element(By.XPATH, "./ancestor::div[2]")
            if parent.get_attribute("id") and parent.get_attribute("onclick"):
                return "available"
        return "unavailable"

    def get_item_elements(self, item: str) -> list[WebElement]:
        return self.get_pane_items().find_elements(
            By.XPATH,
            "//div[@id and @onclick and div[@class='fc2 fal fcb']/div[text()='{item_name}']]".format(
                item_name=item
            ),
        )

    def use(self, item: str) -> bool:
        if self._checked_items[item] == "not_found":
            return False

        if self.get_item_status(item) == "unavailable":
            return False

        item_button_list = self.get_item_elements(item)
        if not item_button_list:
            return False

        if not self.is_open_items_menu():
            self.click_items_menu()
            item_button_list = self.get_item_elements(item)
        ElementActionManager(self.hvdriver).click_and_wait_log(item_button_list[0])
        return True
