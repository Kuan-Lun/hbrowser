import re
from collections import defaultdict

from bs4 import BeautifulSoup
from selenium.webdriver.common.by import By
from selenium.common.exceptions import (
    NoSuchElementException,
    ElementNotInteractableException,
)
from selenium.webdriver.remote.webdriver import WebDriver

from .hv import HVDriver
from .hv_battle_action_manager import ElementActionManager


class SkillManager:
    def __init__(self, driver: HVDriver) -> None:
        self.hvdriver: HVDriver = driver
        # missing_skills: list[str] = []
        # owned_skills: list[str] = []
        self._checked_skills: dict[str, str] = defaultdict(lambda: "available")
        self.skills_cost: dict[str, int] = defaultdict(lambda: 1)

    @property
    def driver(self) -> WebDriver:
        return self.hvdriver.driver

    def _get_skill_xpath(self, key: str) -> str:
        return f"//div[not(@style)]/div/div[contains(text(), '{key}')]"

    def _click_skill_menu(self):
        button = self.driver.find_element(By.ID, "ckey_skill")
        button.click()

    def _click_skill(self, skill_xpath: str, iswait: bool):
        element = self.driver.find_element(By.XPATH, skill_xpath)
        if iswait:
            ElementActionManager(self.hvdriver).click_and_wait_log(element)
        else:
            ElementActionManager(self.hvdriver).click(element)

    def cast(self, key: str, iswait=True) -> bool:
        if self._checked_skills[key] == "missing":
            return False

        self.skills_cost[key] = max(
            self.get_skill_mp_cost_by_name(key), self.skills_cost[key]
        )

        skill_xpath = self._get_skill_xpath(key)

        if key not in self._checked_skills:
            self.get_skill_status(key)

        match self._checked_skills[key]:
            case "missing":
                return False
            case "unavailable":
                return False
            case "available":
                try:
                    self._click_skill(skill_xpath, iswait)
                except ElementNotInteractableException:
                    self._click_skill_menu()
                    try:
                        self._click_skill(skill_xpath, iswait)
                    except ElementNotInteractableException:
                        self._click_skill_menu()
                        self._click_skill(skill_xpath, iswait)
                except NoSuchElementException:
                    return False
                return True
            case _:
                raise ValueError(f"Unknown skill status: {self._checked_skills[key]}")

    def get_skill_status(self, key: str) -> str:
        """
        回傳 'missing'（未擁有）、'available'（可用）、'unavailable'（不可用）
        """

        elementlist = self.driver.find_elements(By.XPATH, self._get_skill_xpath(key))
        if not elementlist:
            self._checked_skills[key] = "missing"
            return "missing"

        style = elementlist[0].get_attribute("style") or ""
        if "opacity" in style:
            opacity = float(style.split("opacity:")[1].split(";")[0])
            return "unavailable" if opacity < 1 else "available"
        return "available"

    def get_skill_mp_cost_by_name(self, skill_name: str) -> int:
        """
        根據技能名稱（如 'Haste' 或 'Weaken'）從 HTML 片段中找出對應的數值。
        """
        page_source = self.hvdriver.driver.page_source
        soup = BeautifulSoup(page_source, "html.parser")
        for div in soup.find_all("div"):
            if not hasattr(div, "get"):
                continue
            onmouseover = div.get("onmouseover", "")
            # 用正則找技能名稱和數值
            pattern = r"set_infopane_spell\('{}',.*?,.*?,\s*(\d+),".format(
                re.escape(skill_name)
            )
            match = re.search(pattern, onmouseover)
            if match:
                self.skills_cost[skill_name] = max(
                    int(match.group(1)), self.skills_cost[skill_name]
                )
        return self.skills_cost[skill_name]
