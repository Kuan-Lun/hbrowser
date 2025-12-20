from collections import defaultdict
from typing import Any

from selenium.common.exceptions import StaleElementReferenceException
from selenium.webdriver.common.by import By

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
    def driver(self) -> Any:  # WebDriver from EHDriver is untyped
        return self.hvdriver.driver

    def _get_skill_xpath(self, key: str) -> str:
        return f"//div[not(@style)]/div/div[contains(text(), '{key}')]"

    def _click_skill_menu(self) -> None:
        # Use resilient locator-based click
        try:
            self.element_action_manager.click_resilient(
                lambda: self.driver.find_element(By.ID, "ckey_skill")
            )
        except StaleElementReferenceException:
            # One more direct attempt (DOM might have re-rendered menu container)
            self.element_action_manager.click_resilient(
                lambda: self.driver.find_element(By.ID, "ckey_skill")
            )

    def open_skills_menu(self) -> None:
        attempts = 0
        while True:
            style = self.driver.find_element(By.ID, "pane_skill").get_attribute("style")
            if style != "display: none;":
                break
            if attempts >= 5:  # safety cap
                break
            try:
                self._click_skill_menu()
            except StaleElementReferenceException:
                pass
            attempts += 1

    def open_spells_menu(self) -> None:
        attempts = 0
        while True:
            style = self.driver.find_element(By.ID, "pane_magic").get_attribute("style")
            if style != "display: none;":
                break
            if attempts >= 5:
                break
            try:
                self._click_skill_menu()
            except StaleElementReferenceException:
                pass
            attempts += 1

    def _click_skill(self, skill_xpath: str, iswait: bool) -> None:
        if iswait:
            # Use locator-based wait+click so stale elements are re-found
            self.element_action_manager.click_and_wait_log_locator(
                By.XPATH, skill_xpath
            )
        else:
            self.element_action_manager.click_resilient(
                lambda: self.driver.find_element(By.XPATH, skill_xpath)
            )

    def cast(self, key: str, iswait: bool = True) -> bool:
        if key not in self.get_skills_and_spells():
            return False

        self.skills_cost[key] = max(
            self.get_max_skill_mp_cost_by_name(key), self.skills_cost[key]
        )

        if self.get_skills_and_spells()[key].available:
            if key in self.battle_dashboard.snap.abilities.skills:
                self.open_skills_menu()
            if key in self.battle_dashboard.snap.abilities.spells:
                self.open_spells_menu()
            skill_xpath = self._get_skill_xpath(key)
            self._click_skill(skill_xpath, iswait)
            return True
        else:
            return False

    def get_skills_and_spells(self) -> dict[str, Any]:
        return (
            self.battle_dashboard.snap.abilities.skills
            | self.battle_dashboard.snap.abilities.spells
        )

    def get_max_skill_mp_cost_by_name(self, skill_name: str) -> int:
        """
        根據技能名稱（如 'Haste' 或 'Weaken'）從 HTML 片段中找出對應的數值。
        """

        if skill_name not in self.get_skills_and_spells():
            return -1  # Default cost if skill not found

        self.skills_cost[skill_name] = max(
            self.get_skills_and_spells()[skill_name].cost,
            self.skills_cost[skill_name],
        )
        return self.skills_cost[skill_name]
