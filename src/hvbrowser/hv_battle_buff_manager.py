import re

from bs4 import BeautifulSoup
from selenium.webdriver.remote.webelement import WebElement
from selenium.webdriver.common.by import By

from .hv import HVDriver, searchxpath_fun
from .hv_battle_skill_manager import SkillManager
from .hv_battle_item_provider import ItemProvider
from .hv_battle_action_manager import ElementActionManager

ITEM_BUFFS = {
    "Health Draught",
    "Mana Draught",
    "Spirit Draught",
    "Scroll of Absorption",
}

SKILL_BUFFS = {
    "Absorb",
    "Heartseeker",
    "Regen",
    "Shadow Veil",
}

BUFF2ICONS = {
    # Item icons
    "Health Draught": {"/y/e/healthpot.png"},
    "Mana Draught": {"/y/e/manapot.png"},
    "Spirit Draught": {"/y/e/spiritpot.png"},
    # Skill icons
    "Absorb": {"/y/e/absorb.png", "/y/e/absorb_scroll.png"},
    "Heartseeker": {"/y/e/heartseeker.png"},
    "Regen": {"/y/e/regen.png"},
    "Shadow Veil": {"/y/e/shadowveil.png"},
    # Spirit icon
    "Spirit Stance": {"/y/battle/spirit_a.png"},
}


class BuffManager:
    def __init__(self, driver: HVDriver) -> None:
        self.hvdriver: HVDriver = driver
        self._item_provider: ItemProvider = ItemProvider(self.hvdriver)
        self._skill_manager: SkillManager = SkillManager(self.hvdriver)

    @property
    def driver(self) -> WebElement:
        return self.hvdriver.driver

    def get_buff_remaining_turns(self, key: str) -> int:
        """
        Get the remaining turns of the buff.
        Returns 0 if the buff is not active.
        """

        if self.has_buff(key) is False:
            return 0

        src_values = BUFF2ICONS[key]
        soup = BeautifulSoup(self.hvdriver.driver.page_source, "html.parser")
        results = {}
        for img in soup.find_all("img"):
            if not hasattr(img, "get"):
                continue
            src = img.get("src", "")
            if src in src_values:
                onmouseover = img.get("onmouseover", "")
                match = re.search(
                    r"set_infopane_effect\([^,]+,[^,]+,(\d+)\)", onmouseover
                )
                if match:
                    results[src] = int(match.group(1))
        return results.get(next(iter(src_values)), 0)

    def has_buff(self, key: str) -> bool:
        """
        Check if the buff is active.
        """
        return (
            self.driver.find_elements(By.XPATH, searchxpath_fun(BUFF2ICONS[key])) != []
        )

    def apply_buff(self, key: str, force: bool) -> bool:
        """
        Apply the buff if it is not already active.
        """
        if all([not force, self.has_buff(key)]):
            return False

        if key == "Absorb":
            if self._item_provider.use("Scroll of Absorption"):
                return True
            else:
                return self._skill_manager.cast(key)

        if key in ITEM_BUFFS:
            return self._item_provider.use(key)

        if key in SKILL_BUFFS:
            self._item_provider.use("Mystic Gem")
            return self._skill_manager.cast(key)

        if key == "Spirit Stance":
            ElementActionManager(self.hvdriver).click_and_wait_log(
                self.driver.find_element(By.ID, "ckey_spirit")
            )
            return True

        raise ValueError(f"Unknown buff key: {key}")
