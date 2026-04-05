from collections import defaultdict
from typing import Any

from .hv import HVDriver
from .hv_battle_action_manager import ElementActionManager
from .hv_battle_observer_pattern import BattleDashboard


class SkillManager:
    def __init__(
        self,
        driver: HVDriver,
        battle_dashboard: BattleDashboard,
    ) -> None:
        self.hvdriver = driver
        self.battle_dashboard = battle_dashboard
        self.element_action_manager = ElementActionManager(
            self.hvdriver, self.battle_dashboard
        )
        self.skills_cost: dict[str, int] = defaultdict(lambda: 1)

    @property
    def page(self) -> Any:
        return self.hvdriver.page

    async def _is_pane_visible(self, pane_id: str) -> bool:
        element = await self.page.select(f"#{pane_id}")
        style: str = element.attrs.get("style", "")
        return style != "display: none;"

    async def open_skills_menu(self) -> None:
        await self.element_action_manager.click_until(
            lambda: self.page.select("#ckey_skill"),
            lambda: self._is_pane_visible("pane_skill"),
        )

    async def open_spells_menu(self) -> None:
        await self.element_action_manager.click_until(
            lambda: self.page.select("#ckey_skill"),
            lambda: self._is_pane_visible("pane_magic"),
        )

    async def _click_skill(self, element_id: str, iswait: bool) -> None:
        selector = f'[id="{element_id}"]'
        if iswait:
            await self.element_action_manager.click_and_wait_log_locator(selector)
        else:
            await self.element_action_manager.click_locator(selector)

    async def cast(self, key: str, iswait: bool = True) -> bool:
        if key not in self.get_skills_and_spells():
            return False

        ability = self.get_skills_and_spells()[key]

        self.skills_cost[key] = max(
            self.get_max_skill_mp_cost_by_name(key), self.skills_cost[key]
        )

        if ability.available:
            if key in self.battle_dashboard.snap.abilities.skills:
                await self.open_skills_menu()
            if key in self.battle_dashboard.snap.abilities.spells:
                await self.open_spells_menu()
            await self._click_skill(ability.element_id, iswait)
            return True
        else:
            return False

    def get_skills_and_spells(self) -> dict[str, Any]:
        return (  # type: ignore[no-any-return]
            self.battle_dashboard.snap.abilities.skills
            | self.battle_dashboard.snap.abilities.spells
        )

    def get_max_skill_mp_cost_by_name(self, skill_name: str) -> int:
        """
        根據技能名稱（如 'Haste' 或 'Weaken'���從 HTML 片段中找出對應的數值。
        """

        if skill_name not in self.get_skills_and_spells():
            return -1  # Default cost if skill not found

        self.skills_cost[skill_name] = max(
            self.get_skills_and_spells()[skill_name].cost,
            self.skills_cost[skill_name],
        )
        return self.skills_cost[skill_name]
