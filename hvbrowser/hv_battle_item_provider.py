from typing import Any

from .hv import HVDriver
from .hv_battle_action_manager import ElementActionManager
from .hv_battle_observer_pattern import BattleDashboard

GEM_ITEMS = {"mystic gem", "health gem", "mana gem", "spirit gem"}


class ItemProvider:
    def __init__(self, driver: HVDriver, battle_dashboard: BattleDashboard) -> None:
        self.hvdriver: HVDriver = driver
        self.battle_dashboard = battle_dashboard
        self.element_action_manager = ElementActionManager(
            self.hvdriver, battle_dashboard
        )

    @property
    def page(self) -> Any:
        return self.hvdriver.page

    async def _get_items_menu_element(self) -> Any:
        return await self.hvdriver.page.select("#ckey_items")

    async def click_items_menu(self) -> None:
        # Resilient click to mitigate stale menu button
        await self.element_action_manager.click_resilient(
            lambda: self.hvdriver.page.select("#ckey_items")
        )

    async def is_open_items_menu(self) -> bool:
        """
        Check if the items menu is open.
        """
        items_menu = await self._get_items_menu_element()
        items_src = items_menu.attrs.get("src", "")
        return "items_s.png" in items_src

    async def use(self, item: str) -> bool:
        if item not in self.battle_dashboard.snap.items.items:
            return False

        parsed_item = self.battle_dashboard.snap.items.items[item]
        if not parsed_item.available:
            return False

        if not parsed_item.element_id:
            return False

        if not await self.is_open_items_menu():
            await self.click_items_menu()
            # Wait for items pane to become visible
            await self.hvdriver.page.select("#pane_item", timeout=2)

        await self.element_action_manager.click_and_wait_log_locator(
            f"#{parsed_item.element_id}"
        )
        return True
