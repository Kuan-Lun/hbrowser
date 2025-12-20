import time
from collections import defaultdict
from collections.abc import Callable
from functools import partial, wraps
from random import random
from typing import Any, TypeVar

from selenium.common.exceptions import UnexpectedAlertPresentException
from selenium.webdriver.common.by import By

from hbrowser.gallery.utils import setup_logger

from .hv import HVDriver
from .hv_battle_action_manager import ElementActionManager
from .hv_battle_buff_manager import BuffManager
from .hv_battle_item_provider import ItemProvider
from .hv_battle_observer_pattern import BattleDashboard
from .hv_battle_ponychart import PonyChart
from .hv_battle_skill_manager import SkillManager
from .pause_controller import PauseController

logger = setup_logger(__name__)

_F = TypeVar("_F", bound=Callable[..., Any])

MONSTER_DEBUFF_TO_CHARACTER_SKILL = {
    "Imperiled": "Imperil",
    "Weakened": "Weaken",
    "Slowed": "Slow",
    "Asleep": "Sleep",
    "Confused": "Confuse",
    "Magically Snared": "MagNet",
    "Blinded": "Blind",
    "Vital Theft": "Drain",
    "Silenced": "Silence",
}


def retry_on_server_fail(func: _F) -> _F:
    """在出現 Server communication failed alert 時，自動刷新頁面並重試一次"""

    @wraps(func)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        try:
            return func(self, *args, **kwargs)
        except UnexpectedAlertPresentException:
            try:
                alert = self.hvdriver.driver.switch_to.alert
                text = alert.text
                alert.accept()
                if "Server communication failed" in text:
                    logger.warning(
                        "Server communication failed detected, "
                        "retrying after refresh..."
                    )
                    time.sleep(5)
                    self.hvdriver.driver.refresh()
                    return func(self, *args, **kwargs)
            except Exception as e:
                logger.error(f"Failed to handle alert or refresh: {e}")
            # 如果不是這種錯誤或重試也失敗，拋出原例外
            raise

    return wrapper  # type: ignore[return-value]


class StatThreshold:
    def __init__(
        self,
        hp: tuple[int, int],
        mp: tuple[int, int],
        sp: tuple[int, int],
        overcharge: tuple[int, int],
        countmonster: tuple[int, int],
    ) -> None:
        if len(hp) != 2:
            raise ValueError("hp should be a list with 2 elements.")

        if len(mp) != 2:
            raise ValueError("mp should be a list with 2 elements.")

        if len(sp) != 2:
            raise ValueError("sp should be a list with 2 elements.")

        if len(overcharge) != 2:
            raise ValueError("overcharge should be a list with 2 elements.")

        if len(countmonster) != 2:
            raise ValueError("countmonster should be a list with 2 elements.")

        self.hp = hp
        self.mp = mp
        self.sp = sp
        self.overcharge = overcharge
        self.countmonster = countmonster


class BattleDriver(HVDriver):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        self.battle_dashboard = BattleDashboard(self)
        self.element_action_manager = ElementActionManager(self, self.battle_dashboard)

        self.with_ofc = "isekai" not in self.driver.current_url
        self._itemprovider = ItemProvider(self, self.battle_dashboard)
        self._skillmanager = SkillManager(self, self.battle_dashboard)
        self._buffmanager = BuffManager(self, self.battle_dashboard)
        self.pausecontroller = PauseController()
        self.turn = -1
        self.round = -1
        self.pround = -1

    def clear_cache(self) -> None:
        # 重新解析戰鬥儀表板以獲取最新的怪物狀態
        self.round = self.battle_dashboard.log_entries.current_round
        self.battle_dashboard.update()

    def reset_pround(self) -> None:
        self.pround = self.round

    def set_battle_parameters(
        self, statthreshold: StatThreshold, forbidden_skills: list[str]
    ) -> None:
        self.statthreshold = statthreshold
        self.forbidden_skills = forbidden_skills

    def click_skill(self, key: str, iswait: bool = True) -> bool:
        if key in self.forbidden_skills:
            return False
        result = self._skillmanager.cast(key, iswait=iswait)
        return result

    def get_stat_percent(self, stat: str) -> float:
        match stat.lower():
            case "hp":
                value = self.battle_dashboard.snap.player.hp_percent
            case "mp":
                value = self.battle_dashboard.snap.player.mp_percent
            case "sp":
                value = self.battle_dashboard.snap.player.sp_percent
            case "overcharge":
                value = self.battle_dashboard.snap.player.overcharge_value
            case _:
                raise ValueError(f"Unknown stat: {stat}")
        return float(value)

    @property
    def new_logs(self) -> list[str]:
        new_logs = self.battle_dashboard.log_entries.current_lines
        # 固定寬度，假設最大 3 位數
        turn_str = f"Turn {self.turn:>5}"
        current = self.battle_dashboard.log_entries.current_round
        total = self.battle_dashboard.log_entries.total_round
        round_str = f"Round {current:>3} / {total:<3}"
        return [f"{turn_str} {round_str} {line}" for line in new_logs]

    def use_item(self, key: str) -> bool:
        return self._itemprovider.use(key)

    def apply_buff(self, key: str, force: bool = False) -> bool:
        if key in self.forbidden_skills:
            return False
        apply_buff = partial(self._buffmanager.apply_buff, key=key, force=force)
        if not force:
            match key:
                case "Health Draught":
                    if self.get_stat_percent("hp") < 90:
                        return apply_buff()
                    else:
                        return False
                case "Mana Draught":
                    if self.get_stat_percent("mp") < 90:
                        return apply_buff()
                    else:
                        return False
                case "Spirit Draught":
                    if self.get_stat_percent("sp") < 90:
                        return apply_buff()
                    else:
                        return False
        return apply_buff()

    def check_hp(self) -> bool:
        if self.get_stat_percent("hp") < self.statthreshold.hp[0]:
            for fun in [
                partial(self.use_item, "Health Gem"),
                partial(self.click_skill, "Full-Cure"),
                partial(self.use_item, "Health Potion"),
                partial(self.use_item, "Health Elixir"),
                partial(self.use_item, "Last Elixir"),
                partial(self.click_skill, "Cure"),
            ]:
                if fun():
                    return True

        if self.get_stat_percent("hp") < self.statthreshold.hp[1]:
            for fun in [
                partial(self.use_item, "Health Gem"),
                partial(self.click_skill, "Cure"),
                partial(self.use_item, "Health Potion"),
            ]:
                if fun():
                    return True

        return False

    def check_mp(self) -> bool:
        if self.get_stat_percent("mp") < self.statthreshold.mp[0]:
            for fun in [
                partial(self.use_item, "Mana Gem"),
                partial(self.use_item, "Mana Potion"),
                partial(self.use_item, "Mana Elixir"),
                partial(self.use_item, "Last Elixir"),
            ]:
                if fun():
                    return True

        if self.get_stat_percent("mp") < self.statthreshold.mp[1]:
            for fun in [
                partial(self.use_item, "Mana Gem"),
                partial(self.use_item, "Mana Potion"),
            ]:
                if fun():
                    return True

        return False

    def check_sp(self) -> bool:
        if self.get_stat_percent("sp") < self.statthreshold.sp[0]:
            for fun in [
                partial(self.use_item, "Spirit Gem"),
                partial(self.use_item, "Spirit Potion"),
                partial(self.use_item, "Spirit Elixir"),
                partial(self.use_item, "Last Elixir"),
            ]:
                if fun():
                    return True

        if self.get_stat_percent("sp") < self.statthreshold.sp[1]:
            for fun in [
                partial(self.use_item, "Spirit Gem"),
                partial(self.use_item, "Spirit Potion"),
            ]:
                if fun():
                    return True

        return False

    def check_overcharge(self) -> bool:
        if self._buffmanager.has_buff("Spirit Stance"):
            # If Spirit Stance is active, check if Overcharge and SP
            # are below thresholds
            if any(
                [
                    self.get_stat_percent("overcharge")
                    < self.statthreshold.overcharge[0],
                    self.get_stat_percent("sp") < self.statthreshold.sp[0],
                ]
            ):
                return self.apply_buff("Spirit Stance", force=True)

        if all(
            [
                self.get_stat_percent("overcharge") > self.statthreshold.overcharge[1],
                self.get_stat_percent("sp") > self.statthreshold.sp[0],
                not self._buffmanager.has_buff("Spirit Stance"),
            ]
        ):
            return self.apply_buff("Spirit Stance")
        return False

    def go_next_floor(self) -> bool:
        # Use locator-based click to avoid stale element caching
        if self.driver.find_elements(By.ID, "btcp"):
            self.element_action_manager.click_and_wait_log_locator(By.ID, "btcp")
            self._create_last_debuff_monster_id()
            return True
        return False

    def debuff_monster(self, debuff: str, nums: list[int]) -> bool:
        debuff_skill = MONSTER_DEBUFF_TO_CHARACTER_SKILL[debuff]
        if debuff_skill in self.forbidden_skills:
            return False

        monster_ids_with_debuff = (
            self.battle_dashboard.overview_monsters.alive_monster_with_buff.get(
                debuff, []
            )
        ) + [self.last_debuff_monster_id[debuff]]
        for num in nums:
            if num not in monster_ids_with_debuff:
                self.attack_monster_by_skill(
                    num, MONSTER_DEBUFF_TO_CHARACTER_SKILL[debuff]
                )
                self.last_debuff_monster_id[debuff] = num
                return True
        return False

    def attack_monster(self, n: int) -> bool:
        element_id = f"mkey_{n}"
        if not self.driver.find_elements(By.ID, element_id):
            return False
        self.element_action_manager.click_and_wait_log_locator(By.ID, element_id)
        return True

    def attack_monster_by_skill(self, n: int, skill_name: str) -> bool:
        self.click_skill(skill_name, iswait=False)
        return self.attack_monster(n)

    def attack(self) -> bool:
        base_monster_ids: list[int] = [1, 3, 5, 7, 9, 2, 4, 6, 8, 0]

        def monster_ids_starting_with(ids: list[int], n: int) -> list[int]:
            """Returns a list of monster IDs starting with the given number."""
            return ids[ids.index(n) :] + ids[: ids.index(n)]

        def resort_monster_alive_ids(bmlist: list[int]) -> list[int]:
            """Returns a list of monster IDs starting with the given number."""
            monster_alive_ids: list[int] = [
                id
                for id in bmlist
                if id in self.battle_dashboard.overview_monsters.alive_monster
            ]
            if len(self.battle_dashboard.overview_monsters.alive_monster):
                monster_alive_ids = monster_ids_starting_with(
                    monster_alive_ids,
                    self.battle_dashboard.overview_monsters.alive_monster[0],
                )
            for monster_name in ["Yggdrasil", "Skuld", "Urd", "Verdandi"][::-1]:
                if (
                    monster_name
                    not in self.battle_dashboard.overview_monsters.alive_monster_name
                ):
                    continue
                monster_id = self.battle_dashboard.overview_monsters.alive_monster_name[
                    monster_name
                ]
                if monster_id in monster_alive_ids:
                    monster_alive_ids = monster_ids_starting_with(
                        monster_alive_ids, monster_id
                    )
            return monster_alive_ids

        # Check if Orbital Friendship Cannon can be used
        if (
            self.with_ofc
            and self.get_stat_percent("overcharge") > 220
            and self._buffmanager.has_buff("Spirit Stance")
            and len(self.battle_dashboard.overview_monsters.alive_monster)
            >= self.statthreshold.countmonster[1]
            and "Orbital Friendship Cannon"
            in self.battle_dashboard.snap.abilities.skills
            and self.battle_dashboard.snap.abilities.skills[
                "Orbital Friendship Cannon"
            ].available
        ):
            self.attack_monster_by_skill(
                self.battle_dashboard.overview_monsters.alive_monster[0],
                "Orbital Friendship Cannon",
            )
            return True

        # Get the list of alive monster IDs
        monster_alive_ids: list[int] = resort_monster_alive_ids(base_monster_ids)

        # Get the list of monster IDs that are not debuffed with the specified debuffs
        if (
            len(monster_alive_ids) > 3
            and self.get_stat_percent("mp") > self.statthreshold.mp[1]
        ):
            for debuff in MONSTER_DEBUFF_TO_CHARACTER_SKILL:
                if debuff in ["Imperiled"]:
                    continue
                debuffed_monsters = (
                    self.battle_dashboard.overview_monsters.alive_monster_with_buff.get(
                        debuff, []
                    )
                )
                if len(monster_alive_ids) - len(debuffed_monsters) < 3:
                    continue
                if self.debuff_monster(debuff, monster_alive_ids):
                    return True

        # Get the list of monster IDs that are not debuffed with Imperil
        monster_with_imperil: list[int]
        if (
            "Imperil" not in self.forbidden_skills
            and self.get_stat_percent("mp") > self.statthreshold.mp[1]
        ):
            monster_with_imperil = (
                self.battle_dashboard.overview_monsters.alive_monster_with_buff.get(
                    "Imperiled", []
                )
            )
        else:
            monster_with_imperil = monster_alive_ids

        if monster_alive_ids:
            n = monster_alive_ids[0]
            if n in monster_with_imperil:
                if self.get_stat_percent(
                    "overcharge"
                ) > 200 and self._buffmanager.has_buff("Spirit Stance"):
                    monster_health = self.battle_dashboard.snap.monsters[n].hp_percent
                    if (
                        monster_health < 25
                        and "Merciful Blow"
                        in self.battle_dashboard.snap.abilities.skills
                        and self.battle_dashboard.snap.abilities.skills[
                            "Merciful Blow"
                        ].available
                    ):
                        self.attack_monster_by_skill(n, "Merciful Blow")
                    elif (
                        monster_health > 5
                        and "Vital Strike"
                        in self.battle_dashboard.snap.abilities.skills
                        and self.battle_dashboard.snap.abilities.skills[
                            "Vital Strike"
                        ].available
                    ):
                        self.attack_monster_by_skill(n, "Vital Strike")
                    else:
                        self.attack_monster(n)
                else:
                    self.attack_monster(n)
                self.last_debuff_monster_id["Imperiled"] = -1
            else:
                if n == self.last_debuff_monster_id["Imperiled"]:
                    if random() < 0.5:
                        self.attack_monster_by_skill(n, "Imperil")
                    else:
                        self.attack_monster(n)
                else:
                    self.attack_monster_by_skill(
                        n, MONSTER_DEBUFF_TO_CHARACTER_SKILL["Imperiled"]
                    )
                    self.last_debuff_monster_id["Imperiled"] = n
            return True
        else:
            return False

    def use_channeling(self) -> bool:
        if "Channeling" in self.battle_dashboard.snap.player.buffs:
            skill_names = ["Regen", "Heartseeker"]
            skill2remaining: dict[str, float] = dict()
            for skill_name in skill_names:
                remaining_turns = self._buffmanager.get_buff_remaining_turns(skill_name)
                refresh_turns = self._buffmanager.skill2turn[skill_name]
                skill_cost = self._skillmanager.get_max_skill_mp_cost_by_name(
                    skill_name
                )
                skill2remaining[skill_name] = (
                    (refresh_turns - remaining_turns) * refresh_turns / skill_cost
                )
            if max(skill2remaining.values()) < 0:
                return False

            to_use_skill_name = max(skill2remaining, key=lambda k: skill2remaining[k])

            self.apply_buff(to_use_skill_name, force=True)
            return True

        return False

    @retry_on_server_fail
    def battle_in_turn(self) -> bool:
        self.turn += 1
        self.clear_cache()
        # Log the current round logs
        if self.new_logs:
            for log_line in self.new_logs:
                logger.info(log_line)

        for fun in [
            self.go_next_floor,
            PonyChart(self).check,
            self.check_hp,
            self.check_mp,
            self.check_sp,
            self.check_overcharge,
            lambda: self.apply_buff("Health Draught"),
            lambda: self.apply_buff("Mana Draught"),
            lambda: self.apply_buff("Spirit Draught"),
            lambda: self.apply_buff("Regen"),
            lambda: self.apply_buff("Scroll of Life"),
            lambda: self.apply_buff("Scroll of Absorption"),
            lambda: self.apply_buff("Absorb"),
            lambda: self.apply_buff("Scroll of Protection"),
            lambda: self.apply_buff("Heartseeker"),
            self.use_channeling,
            self.attack,
        ]:
            if fun():
                return True

        return False

    def _create_last_debuff_monster_id(self) -> None:
        self.last_debuff_monster_id: dict[str, int] = defaultdict(lambda: -1)

    def battle(self) -> None:
        self._create_last_debuff_monster_id()
        # Open skill menu twice using resilient locator-based click
        # (no element caching & no log wait)
        from selenium.webdriver.common.by import (
            By as _By,
        )  # local import to avoid top-level clutter

        for _ in range(2):
            self.element_action_manager.click_resilient(
                lambda: self.driver.find_element(_By.ID, "ckey_skill")
            )
            time.sleep(0.05)

        while True:
            if not self.pausecontroller.pauseable(self.battle_in_turn)():
                break
