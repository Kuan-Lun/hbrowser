from typing import Any

from selenium.webdriver.common.by import By

from .hv import HVDriver
from .hv_battle_action_manager import ElementActionManager
from .hv_battle_observer_pattern import BattleDashboard

GEM_ITEMS = {"Mystic Gem", "Health Gem", "Mana Gem", "Spirit Gem"}


class ItemProvider:
    def __init__(self, driver: HVDriver, battle_dashboard: BattleDashboard) -> None:
        self.hvdriver: HVDriver = driver
        self.battle_dashboard = battle_dashboard
        self.element_action_manager = ElementActionManager(
            self.hvdriver, battle_dashboard
        )

    @property
    def driver(self) -> Any:  # WebDriver from EHDriver is untyped
        return self.hvdriver.driver

    @property
    def items_menu_web_element(self) -> Any:  # WebElement from untyped driver
        return self.hvdriver.driver.find_element(By.ID, "ckey_items")

    def click_items_menu(self) -> None:
        # Resilient click to mitigate stale menu button
        self.element_action_manager.click_resilient(
            lambda: self.hvdriver.driver.find_element(By.ID, "ckey_items")
        )

    def is_open_items_menu(self) -> bool:
        """
        Check if the items menu is open.
        """
        items_menum = self.items_menu_web_element.get_attribute("src") or ""
        return "items_s.png" in items_menum

    def get_pane_items(self) -> Any:  # WebElement from untyped driver
        return self.hvdriver.driver.find_element(By.ID, "pane_item")

    def get_item_elements(self, item: str) -> list[Any]:
        xpath = (
            f"//div[@id and @onclick and "
            f"div[@class='fc2 fal fcb']/div[text()='{item}']]"
        )
        elements: list[Any] = self.get_pane_items().find_elements(By.XPATH, xpath)
        return elements

    def use(self, item: str) -> bool:
        if item not in self.battle_dashboard.snap.items.items:
            return False

        if not self.battle_dashboard.snap.items.items[item].available:
            return False

        item_button_list = self.get_item_elements(item)
        if not item_button_list:
            return False

        if not self.is_open_items_menu():
            self.click_items_menu()
            item_button_list = self.get_item_elements(item)
        # Use locator-based click (derive unique locator via first
        # element's id attribute if present)
        target = item_button_list[0]
        item_id = target.get_attribute("id")
        if item_id:
            self.element_action_manager.click_and_wait_log_locator(By.ID, item_id)
        else:
            raise ValueError("Item ID not found")
        return True
