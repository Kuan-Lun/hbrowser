from collections import defaultdict
from typing import Any

from selenium.webdriver.common.by import By

from .hv import HVDriver
from .hv_battle_action_manager import ElementActionManager
from .hv_battle_item_provider import ItemProvider
from .hv_battle_observer_pattern import BattleDashboard
from .hv_battle_skill_manager import SkillManager

ITEM_BUFFS = {
    "Health Draught",
    "Mana Draught",
    "Spirit Draught",
    "Scroll of Absorption",
    "Scroll of Life",
    "Scroll of Protection",
}

SKILLS_TO_CHARACTER_BUFFS = {
    "Absorb": "Absorbing Ward",
    "Scroll of Absorption": "Absorbing Ward",
    "Scroll of Protection": "Protection",
    "Scroll of Life": "Spark of Life",
    "Health Draught": "Regeneration",
    "Mana Draught": "Replenishment",
    "Spirit Draught": "Refreshment",
}

AutoCast_BUFFS = {
    "Spark of Life",
    "Spirit Shield",
    "Shadow Veil",
    "Protection",
    "Hastened",
}

SKILL_BUFFS = {
    "Absorb",
    "Heartseeker",
    "Regen",
    "Shadow Veil",
    "Spark of Life",
}

# BUFF2ICONS = {
#     # Item icons
#     "Health Draught": {"/y/e/healthpot.png"},
#     "Mana Draught": {"/y/e/manapot.png"},
#     "Spirit Draught": {"/y/e/spiritpot.png"},
#     "Scroll of Life": {"/y/e/sparklife_scroll.png"},
#     "Scroll of Protection": {"/y/e/protection_scroll.png"},
#     # Skill icons
#     "Absorb": {"/y/e/absorb.png", "/y/e/absorb_scroll.png"},
#     "Heartseeker": {"/y/e/heartseeker.png"},
#     "Regen": {"/y/e/regen.png"},
#     "Shadow Veil": {"/y/e/shadowveil.png"},
#     "Spark of Life": {"/y/e/sparklife.png", "/y/e/sparklife_scroll.png"},
#     # Spirit icon
#     "Spirit Stance": {"/y/battle/spirit_a.png"},
# }

# BUFF2ITEMS = {
#     "Absorb": ["Scroll of Absorption"],
#     "Protection": ["Scroll of Protection"],
#     "Spark of Life": ["Scroll of Life"],
# }


class BuffManager:
    def __init__(self, driver: HVDriver, battle_dashboard: BattleDashboard) -> None:
        self.hvdriver = driver
        self.battle_dashboard = battle_dashboard
        self._item_provider = ItemProvider(self.hvdriver, self.battle_dashboard)
        self._skill_manager = SkillManager(self.hvdriver, self.battle_dashboard)
        self.element_action_manager = ElementActionManager(
            self.hvdriver, self.battle_dashboard
        )
        self.skill2turn: dict[str, int] = defaultdict(lambda: 1)

    @property
    def driver(self) -> Any:  # WebDriver from EHDriver is untyped
        return self.hvdriver.driver

    def get_buff_remaining_turns(self, key: str) -> int | float:
        """
        Get the remaining turns of the buff.
        Returns 0 if the buff is not active.
        """

        if self.has_buff(key) is False:
            return 0

        remaining_turns = self.battle_dashboard.snap.player.buffs[key].remaining_turns
        turns = int(remaining_turns)
        self.skill2turn[key] = max(self.skill2turn[key], turns)
        return turns

    def _cast_skill(self, key: str) -> bool:
        iscast = self._skill_manager.cast(key)
        if iscast:
            self.get_buff_remaining_turns(key)
        return iscast

    def has_buff(self, key: str) -> bool:
        """
        Check if the buff is active.
        """
        if key not in self.battle_dashboard.snap.player.buffs:
            return False

        remaining_turns = self.battle_dashboard.snap.player.buffs[key].remaining_turns

        if key in AutoCast_BUFFS:
            return bool(float("inf") > remaining_turns >= 0)
        else:
            return bool(remaining_turns >= 0)

    def _apply_hybrid_buff(self, key: str, item_name: str) -> bool:
        """
        Apply buff that can be cast from both item and skill.
        Try item first, fallback to skill if item fails.
        """
        if self._item_provider.use(item_name):
            return True
        else:
            return self._cast_skill(key)

    def apply_buff(self, key: str, force: bool) -> bool:
        """
        Apply the buff if it is not already active.
        """
        if key in SKILLS_TO_CHARACTER_BUFFS:
            buff_key = SKILLS_TO_CHARACTER_BUFFS[key]
        else:
            buff_key = key
        if all([not force, self.has_buff(buff_key)]):
            return False

        # Special cases
        match key:
            case "Spirit Stance":
                # Use locator-based resilient click; Spirit Stance toggles instantly
                self.element_action_manager.click_and_wait_log_locator(
                    By.ID, "ckey_spirit"
                )
                return True

        if key in ITEM_BUFFS:
            return self._item_provider.use(key)

        if key in SKILL_BUFFS:
            self._item_provider.use("Mystic Gem")
            return self._cast_skill(key)

        raise ValueError(f"Unknown buff key: {key}")
