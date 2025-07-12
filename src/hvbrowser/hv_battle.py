from functools import partial
from collections import defaultdict
from random import random

from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.by import By

from .hv import HVDriver, searchxpath_fun
from .hv_battle_stat_provider import (
    StatProviderHP,
    StatProviderMP,
    StatProviderSP,
    StatProviderOvercharge,
)
from .hv_battle_ponychart import PonyChart
from .hv_battle_item_provider import ItemProvider
from .hv_battle_action_manager import ElementActionManager
from .hv_battle_skill_manager import SkillManager
from .hv_battle_buff_manager import BuffManager
from .hv_battle_monster_status_manager import MonsterStatusManager
from .hv_battle_log import LogProvider
from .pause_controller import PauseController


def interleave_even_odd(nums):
    if 0 in nums:
        nums = sorted(nums[:-1]) + [0]  # 0在最後
    else:
        nums = sorted(nums)
    even = nums[::2]
    odd = nums[1::2]
    result = []
    i = j = 0
    for k in range(len(nums)):
        if k % 2 == 0 and i < len(even):
            result.append(even[i])
            i += 1
        elif j < len(odd):
            result.append(odd[j])
            j += 1
    return result


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
        self.with_ofc = "isekai" not in self.driver.current_url
        self._logprovider = LogProvider(self)
        self._itemprovider = ItemProvider(self)
        self._skillmanager = SkillManager(self)
        self._buffmanager = BuffManager(self)
        self._monsterstatusmanager = MonsterStatusManager(self)
        self.pausecontroller = PauseController()
        self.turn = -1

    def set_battle_parameters(self, statthreshold: StatThreshold) -> None:
        self.statthreshold = statthreshold

    def click_skill(self, key: str, iswait=True) -> bool:
        return self._skillmanager.cast(key, iswait=iswait)

    def get_stat_percent(self, stat: str) -> float:
        match stat.lower():
            case "hp":
                value = StatProviderHP(self).get_percent()
            case "mp":
                value = StatProviderMP(self).get_percent()
            case "sp":
                value = StatProviderSP(self).get_percent()
            case "overcharge":
                value = StatProviderOvercharge(self).get_percent()
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

    @property
    def is_with_spirit_stance(self) -> bool:
        return StatProviderOvercharge(self).get_spirit_stance_status() == "activated"

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
    def monster_alive_count(self) -> int:
        return self._monsterstatusmanager.alive_count

    @property
    def monster_alive_ids(self) -> list[int]:
        return self._monsterstatusmanager.alive_monster_ids

    @property
    def system_monster_alive_ids(self) -> list[int]:
        return self._monsterstatusmanager.alive_system_monster_ids

    def get_monster_id_by_name(self, name: str) -> int:
        """
        根據怪物名稱取得對應的 monster id（如 mkey_0 會回傳 0）。
        """
        return self._monsterstatusmanager.get_monster_id_by_name(name)

    @return_false_on_nosuch
    def check_hp(self) -> bool:
        if self.get_stat_percent("hp") < self.statthreshold.hp[0]:
            if any(
                [
                    self.use_item("Health Gem"),
                    self.click_skill("Full-Cure"),
                    self.use_item("Health Potion"),
                    self.use_item("Health Elixir"),
                    self.use_item("Last Elixir"),
                    self.click_skill("Cure"),
                ]
            ):
                return True

        if self.get_stat_percent("hp") < self.statthreshold.hp[1]:
            if any(
                [
                    self.use_item("Health Gem"),
                    self.click_skill("Cure"),
                    self.use_item("Health Potion"),
                ]
            ):
                return True

        return False

    @return_false_on_nosuch
    def check_mp(self) -> bool:
        if self.get_stat_percent("mp") < self.statthreshold.mp[0]:
            if any(
                [
                    self.use_item("Mana Gem"),
                    self.use_item("Mana Potion"),
                    self.use_item("Mana Elixir"),
                    self.use_item("Last Elixir"),
                ]
            ):
                return True

        if self.get_stat_percent("mp") < self.statthreshold.mp[1]:
            if any(
                [
                    self.use_item("Mana Gem"),
                    self.use_item("Mana Potion"),
                ]
            ):
                return True

        return False

    @return_false_on_nosuch
    def check_sp(self) -> bool:
        if self.get_stat_percent("sp") < self.statthreshold.sp[0]:
            if any(
                [
                    self.use_item("Spirit Gem"),
                    self.use_item("Spirit Potion"),
                    self.use_item("Spirit Elixir"),
                    self.use_item("Last Elixir"),
                ]
            ):
                return True

        if self.get_stat_percent("sp") < self.statthreshold.sp[1]:
            if any(
                [
                    self.use_item("Spirit Gem"),
                    self.use_item("Spirit Potion"),
                ]
            ):
                return True

        return False

    @return_false_on_nosuch
    def check_overcharge(self) -> bool:
        if self.is_with_spirit_stance:
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
                not self.is_with_spirit_stance,
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
            ElementActionManager(self).click_and_wait_log(continue_elements[0])
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

        ElementActionManager(self).click_and_wait_log(elements[0])
        return True

    def attack(self) -> bool:
        # Check if Orbital Friendship Cannon can be used
        if all(
            [
                self.with_ofc,
                self.get_stat_percent("overcharge") > 220,
                self.is_with_spirit_stance,
                self.monster_alive_count >= self.statthreshold.countmonster[1],
            ]
        ):
            self.click_skill("Orbital Friendship Cannon", iswait=False)

        # Get the list of alive monster IDs
        monster_alive_ids = interleave_even_odd(self.monster_alive_ids)
        if len(self.system_monster_alive_ids):
            monster_id = self.system_monster_alive_ids[0]
            monster_alive_ids = (
                monster_alive_ids[monster_alive_ids.index(monster_id) :]
                + monster_alive_ids[: monster_alive_ids.index(monster_id)]
            )
        for monster_name in ["Yggdrasil", "Skuld", "Urd", "Verdandi"][-1::-1]:
            monster_id = self.get_monster_id_by_name(monster_name)
            if monster_id in monster_alive_ids:
                monster_alive_ids = (
                    monster_alive_ids[monster_alive_ids.index(monster_id) :]
                    + monster_alive_ids[: monster_alive_ids.index(monster_id)]
                )

        # Get the list of monster IDs that are not debuffed with the specified debuffs
        for debuff in ["Weaken", "Slow", "Blind", "MagNet", "Silence", "Drain"]:
            monster_with_debuff = (
                self._monsterstatusmanager.get_monster_ids_with_debuff(debuff)
            )
            if all(
                [
                    monster_alive_ids,
                    len(monster_alive_ids) > 3,
                    len(monster_with_debuff) / len(monster_alive_ids) < 0.7,
                ]
            ):
                for n in monster_alive_ids:
                    if all(
                        [
                            n not in monster_with_debuff,
                            n != self.last_debuff_monster_id[debuff],
                        ]
                    ):
                        self.click_skill(debuff, iswait=False)
                        self.attack_monster(n)
                        self.last_debuff_monster_id[debuff] = n
                        return True

        # Get the list of monster IDs that are not debuffed with Imperil
        if self.get_stat_percent("mp") > self.statthreshold.mp[1]:
            monster_with_imperil = (
                self._monsterstatusmanager.get_monster_ids_with_debuff("Imperil")
            )
        else:
            monster_with_imperil = monster_alive_ids
        for n in monster_alive_ids:
            if n not in monster_with_imperil:
                if n == self.last_debuff_monster_id["Imperil"]:
                    # If the last debuffed monster is the same, attack it directly
                    if random() < 0.5:
                        self.click_skill("Imperil", iswait=False)
                    self.attack_monster(n)
                else:
                    self.click_skill("Imperil", iswait=False)
                    self.attack_monster(n)
                    self.last_debuff_monster_id["Imperil"] = n
            else:
                self.last_debuff_monster_id["Imperil"] = -1
                self.attack_monster(n)
            return True
        return False

    def finish_battle(self) -> bool:
        elements = self.driver.find_elements(
            By.XPATH, searchxpath_fun(["/y/battle/finishbattle.png"])
        )

        if not elements:
            return False

        ActionChains(self.driver).move_to_element(elements[0]).click().perform()
        return True

    def use_channeling(self) -> bool:
        channeling_elements = self.driver.find_elements(
            By.XPATH, searchxpath_fun(["/y/e/channeling.png"])
        )
        if channeling_elements:
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
        # Print the current round logs
        print("\n".join(self.new_logs))

        if self.finish_battle():
            return "break"

        if any(
            fun()
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
            ]
        ):
            return "continue"

        if self.use_channeling():
            return "continue"

        if self.attack():
            return "continue"

        return "continue"

    def _create_last_debuff_monster_id(self) -> None:
        self.last_debuff_monster_id: dict[str, int] = defaultdict(lambda: -1)

    def battle(self) -> None:
        self._create_last_debuff_monster_id()

        while True:
            match self.pausecontroller.pauseable(self.battle_in_turn)():
                case "break":
                    break
                case "continue":
                    continue
                case _:
                    raise ValueError("Unexpected return value from battle_in_turn.")
