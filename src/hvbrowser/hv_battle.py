from functools import partial
from collections import defaultdict, deque
from random import random

from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.by import By

from .hv import HVDriver, searchxpath_fun
from .hv_battle_ponychart import PonyChart
from .hv_battle_item_provider import ItemProvider
from .hv_battle_action_manager import ElementActionManager
from .hv_battle_skill_manager import SkillManager
from .hv_battle_buff_manager import BuffManager
from .hv_battle_monster_status_manager import MonsterStatusManager
from .hv_battle_log import LogProvider
from .hv_battle_dashboard import BattleDashBoard
from .pause_controller import PauseController

monster_debuff_to_character_skill = {
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


def return_false_on_nosuch(fun):
    def wrapper(*args, **kwargs):
        try:
            return fun(*args, **kwargs)
        except NoSuchElementException:
            return False

    return wrapper


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
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.battle_dashboard = BattleDashBoard(self)
        self.element_action_manager = ElementActionManager(self, self.battle_dashboard)

        self.with_ofc = "isekai" not in self.driver.current_url
        self._logprovider = LogProvider(self, self.battle_dashboard)
        self._itemprovider = ItemProvider(self, self.battle_dashboard)
        self._skillmanager = SkillManager(self, self.battle_dashboard)
        self._buffmanager = BuffManager(self, self.battle_dashboard)
        self._monsterstatusmanager = MonsterStatusManager(self, self.battle_dashboard)
        self.pausecontroller = PauseController()
        self.turn = -1

    def clear_cache(self) -> None:
        # 重新解析戰鬥儀表板以獲取最新的怪物狀態
        self.battle_dashboard.refresh()
        self._monsterstatusmanager.clear_cache()

    def set_battle_parameters(
        self, statthreshold: StatThreshold, forbidden_skills: list[list]
    ) -> None:
        self.statthreshold = statthreshold
        self.forbidden_skills = forbidden_skills

    def click_skill(self, key: str, iswait=True) -> bool:
        if key in self.forbidden_skills:
            return False
        return self._skillmanager.cast(key, iswait=iswait)

    def get_stat_percent(self, stat: str) -> float:
        match stat.lower():
            case "hp":
                value = self.battle_dashboard.character.vitals.hp
            case "mp":
                value = self.battle_dashboard.character.vitals.mp
            case "sp":
                value = self.battle_dashboard.character.vitals.sp
            case "overcharge":
                value = self.battle_dashboard.character.vitals.overcharge
            case _:
                raise ValueError(f"Unknown stat: {stat}")
        return value

    @property
    def new_logs(self) -> list[str]:
        new_logs = self._logprovider.get_new_logs()
        # 固定寬度，假設最大 3 位數
        turn_str = f"Turn {self.turn:>5}"
        round_str = f"Round {self._logprovider.current_round:>3} / {self._logprovider.total_round:<3}"
        return [f"{turn_str} {round_str} {line}" for line in new_logs]

    def use_item(self, key: str) -> bool:
        return self._itemprovider.use(key)

    def apply_buff(self, key: str, force: bool = False) -> bool:
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

    @property
    def system_monster_alive_ids(self) -> list[int]:
        return self._monsterstatusmanager.alive_system_monster_ids

    @return_false_on_nosuch
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

    @return_false_on_nosuch
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

    @return_false_on_nosuch
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

    @return_false_on_nosuch
    def check_overcharge(self) -> bool:
        if self._buffmanager.has_buff("Spirit Stance"):
            # If Spirit Stance is active, check if Overcharge and SP are below thresholds
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

    @return_false_on_nosuch
    def go_next_floor(self) -> bool:
        continue_images = [
            "/y/battle/arenacontinue.png",
            "/y/battle/grindfestcontinue.png",
            "/y/battle/itemworldcontinue.png",
        ]
        continue_elements = self.driver.find_elements(
            By.XPATH, searchxpath_fun(continue_images)
        )

        if continue_elements:
            self.element_action_manager.click_and_wait_log(continue_elements[0])
            self._create_last_debuff_monster_id()
            return True
        else:
            return False

    def attack_monster(self, n: int) -> bool:
        elements = self.driver.find_elements(
            By.XPATH, '//div[@id="mkey_{n}"]'.format(n=n)
        )

        if not elements:
            return False

        self.element_action_manager.click_and_wait_log(elements[0])
        return True

    def attack(self) -> bool:
        base_monster_ids = [1, 3, 5, 7, 9, 2, 4, 6, 8, 0]

        def base_monster_ids_starting_with(n: int) -> list[int]:
            """Returns a list of monster IDs starting with the given number."""
            return (
                base_monster_ids[base_monster_ids.index(n) :]
                + base_monster_ids[: base_monster_ids.index(n)]
            )

        def resort_monster_alive_ids(bmlist: list[int]) -> list[int]:
            """Returns a list of monster IDs starting with the given number."""
            return [
                id
                for id in bmlist
                if id in self._monsterstatusmanager.alive_monster_ids
            ]

        # Check if Orbital Friendship Cannon can be used
        if all(
            [
                self.with_ofc,
                self.get_stat_percent("overcharge") > 220,
                self._buffmanager.has_buff("Spirit Stance"),
                self._monsterstatusmanager.alive_count
                >= self.statthreshold.countmonster[1],
                self._skillmanager.get_skill_status("Orbital Friendship Cannon")
                == "available",
            ]
        ):
            self.click_skill("Orbital Friendship Cannon", iswait=False)
            self.attack_monster(self._monsterstatusmanager.alive_monster_ids[0])
            return True

        # Get the list of alive monster IDs
        monster_alive_ids = resort_monster_alive_ids(base_monster_ids)

        # Get the list of monster IDs that are not debuffed with the specified debuffs
        if all(
            [
                len(monster_alive_ids) > 3,
                self.get_stat_percent("mp") > self.statthreshold.mp[1],
            ]
        ):
            monster_debuff_list = [
                "Weakened",
                "Slowed",
                "Blinded",
                "Magically Snared",
                "Silenced",
                "Vital Theft",
            ]
            for debuff in monster_debuff_list:
                if debuff in self.forbidden_skills:
                    continue
                monster_without_debuff = [
                    n
                    for n in monster_alive_ids
                    if n
                    not in self._monsterstatusmanager.get_monster_ids_with_debuff(
                        debuff
                    )
                ]
                if len(monster_without_debuff) / len(monster_alive_ids) > 0.3:
                    for n in monster_without_debuff:
                        if n != self.last_debuff_monster_id[debuff]:
                            self.click_skill(
                                monster_debuff_to_character_skill[debuff], iswait=False
                            )
                            self.attack_monster(n)
                            self.last_debuff_monster_id[debuff] = n
                            return True

        # Get the list of monster IDs that are not debuffed with Imperil
        if self.get_stat_percent("mp") > self.statthreshold.mp[1]:
            monster_with_imperil = (
                self._monsterstatusmanager.get_monster_ids_with_debuff("Imperiled")
            )
        else:
            monster_with_imperil = monster_alive_ids

        if len(self.system_monster_alive_ids):
            monster_alive_ids = resort_monster_alive_ids(
                base_monster_ids_starting_with(self.system_monster_alive_ids[0])
            )
        for monster_name in ["Yggdrasil", "Skuld", "Urd", "Verdandi"][::-1]:
            monster_id = self._monsterstatusmanager.get_monster_id_by_name(monster_name)
            if monster_id in monster_alive_ids:
                monster_alive_ids = resort_monster_alive_ids(
                    base_monster_ids_starting_with(monster_id)
                )

        n = monster_alive_ids[0]
        if n in monster_with_imperil:
            self.last_debuff_monster_id["Imperiled"] = -1
            self.attack_monster(n)
        else:
            if n == self.last_debuff_monster_id["Imperiled"]:
                # If the last debuffed monster is the same, attack it directly
                if random() < 0.5:
                    self.click_skill("Imperil", iswait=False)
                self.attack_monster(n)
            else:
                self.click_skill("Imperil", iswait=False)
                self.attack_monster(n)
                self.last_debuff_monster_id["Imperiled"] = n
        return True

    def finish_battle(self) -> bool:
        elements = self.driver.find_elements(
            By.XPATH, searchxpath_fun(["/y/battle/finishbattle.png"])
        )

        if not elements:
            return False

        ActionChains(self.driver).move_to_element(elements[0]).click().perform()
        return True

    def use_channeling(self) -> bool:
        if "Channeling" in self.battle_dashboard.character.buffs:
            skill_names = ["Regen", "Heartseeker"]
            skill2remaining: dict[str, float] = dict()
            for skill_name in skill_names:
                remaining_turns = self._buffmanager.get_buff_remaining_turns(skill_name)
                refresh_turns = self._buffmanager.skill2turn[skill_name]
                skill_cost = self._skillmanager.get_skill_mp_cost_by_name(skill_name)
                skill2remaining[skill_name] = (
                    (refresh_turns - remaining_turns) * refresh_turns / skill_cost
                )
            to_use_skill_name = max(skill2remaining, key=lambda k: skill2remaining[k])

            self.apply_buff(to_use_skill_name, force=True)
            return True

        return False

    def battle_in_turn(self) -> str:
        self.turn += 1
        self.clear_cache()
        # Print the current round logs
        print("\n".join(self.new_logs))

        if self.finish_battle():
            return "break"

        for fun in [
            self.go_next_floor,
            PonyChart(self).check,
            self.check_hp,
            self.check_mp,
            self.check_sp,
            self.check_overcharge,
            partial(self.apply_buff, "Health Draught"),
            partial(self.apply_buff, "Mana Draught"),
            partial(self.apply_buff, "Spirit Draught"),
            partial(self.apply_buff, "Regen"),
            partial(self.apply_buff, "Scroll of Life"),
            partial(self.apply_buff, "Absorb"),
            partial(self.apply_buff, "Heartseeker"),
        ]:
            if fun():
                return "continue"

        if self.use_channeling():
            return "continue"

        if self.attack():
            return "continue"

        return "break"

    def _create_last_debuff_monster_id(self) -> None:
        self.last_debuff_monster_id: dict[str, int] = defaultdict(lambda: -1)

    def battle(self) -> None:
        self._create_last_debuff_monster_id()
        partial_click_skill_menu = partial(
            self.element_action_manager.click,
            self.driver.find_element(By.ID, "ckey_skill"),
        )
        partial_click_skill_menu()
        partial_click_skill_menu()

        while True:
            match self.pausecontroller.pauseable(self.battle_in_turn)():
                case "break":
                    break
                case "continue":
                    continue
                case _:
                    raise ValueError("Unexpected return value from battle_in_turn.")
