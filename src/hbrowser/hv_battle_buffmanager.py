from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.remote.webelement import WebElement
from selenium.webdriver.common.by import By

from .hv_battle import HVDriver, searchxpath_fun
from .hv_battle_skillmanager import SkillManager
from .hv_battle_itemprovider import ItemProvider

ITEM_BUFFS = {
    "Health Draught",
    "Mana Draught",
    "Spirit Draught",
}

SKILL_BUFFS = {
    "Absorb",
    "Heartseeker",
    "Regen",
}

BUFF2ICON = {
    # Item icons
    "Health Draught": "/y/e/healthpot.png",
    "Mana Draught": "/y/e/manapot.png",
    "Spirit Draught": "/y/e/spiritpot.png",
    # Skill icons
    "Absorb": "/y/e/absorb.png",
    "Heartseeker": "/y/e/channeling.png",
    "Regen": "/y/e/regen.png",
}

BUFF2ICON = {v: k for k, v in BUFF2ICON.items()}


class BuffManager:
    def __init__(self, driver: HVDriver) -> None:
        self.hvdriver: HVDriver = driver

    @property
    def driver(self) -> WebElement:
        return self.hvdriver.driver

    def has_buff(self, key: str) -> bool:
        """
        Check if the buff is active.
        """
        try:
            self.driver.find_element(By.XPATH, searchxpath_fun([BUFF2ICON[key]]))
            return True
        except NoSuchElementException:
            return False

    def apply_buff(self, key: str) -> bool:
        """
        Apply the buff if it is not already active.
        """
        if self.has_buff(key):
            return False

        if key in ITEM_BUFFS:
            item_provider = ItemProvider(self.hvdriver)
            if item_provider.use(key):
                return True
            return False

        if key in SKILL_BUFFS:
            skill_manager = SkillManager(self.hvdriver)
            if skill_manager.cast(key):
                return True
            return False

        raise ValueError(f"Unknown buff key: {key}")
