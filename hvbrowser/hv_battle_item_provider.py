from typing import Any

from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

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

    def use(self, item: str) -> bool:
        if item not in self.battle_dashboard.snap.items.items:
            return False

        parsed_item = self.battle_dashboard.snap.items.items[item]
        if not parsed_item.available:
            return False

        if not parsed_item.element_id:
            return False

        if not self.is_open_items_menu():
            self.click_items_menu()
            WebDriverWait(self.driver, 2).until(
                EC.visibility_of_element_located((By.ID, "pane_item"))
            )

        self.element_action_manager.click_and_wait_log_locator(
            By.ID, parsed_item.element_id
        )
        return True
