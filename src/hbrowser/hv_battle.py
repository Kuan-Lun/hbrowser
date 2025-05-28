import time
from functools import partial

from selenium.common.exceptions import (
    NoSuchElementException,
    ElementNotInteractableException,
)
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.by import By
from selenium.webdriver.remote.webelement import WebElement

from .hv import HVDriver, searchxpath_fun
from .hv_battle_statprovider import (
    StatProviderHP,
    StatProviderMP,
    StatProviderSP,
    StatProviderOvercharge,
)
from .hv_battle_ponychart import PonyChart
from .hv_battle_item import ItemProvider
from .hv_battle_actionmanager import ElementActionManager
from .hv_battle_skillmanager import SkillManager


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
    def set_battle_parameters(self, statthreshold: StatThreshold) -> None:
        self.statthreshold = statthreshold
        self.with_ofc = "isekai" not in self.driver.current_url

    @property
    def _skillmanager(self) -> SkillManager:
        return SkillManager(self)

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
    def _itemprovider(self) -> ItemProvider:
        return ItemProvider(self)

    def use_item(self, key: str) -> bool:
        return self._itemprovider.use(key)

    @return_false_on_nosuch
    def check_hp(self) -> bool:
        if self.get_stat_percent("hp") < self.statthreshold.hp[0]:
            for fun in [
                partial(self.click_skill, "Full-Cure"),
                partial(self.use_item, "Health Potion"),
                partial(self.use_item, "Health Elixir"),
                partial(self.use_item, "Last Elixir"),
                partial(self.click_skill, "Cure"),
            ]:
                if self.get_stat_percent("hp") < self.statthreshold.hp[0]:
                    if not fun():
                        continue
                    return True

        if self.get_stat_percent("hp") < self.statthreshold.hp[1]:
            for fun in [
                partial(self.click_skill, "Cure"),
                partial(self.click_skill, "Full-Cure"),
                partial(self.use_item, "Health Potion"),
                partial(self.use_item, "Health Elixir"),
                partial(self.use_item, "Last Elixir"),
            ]:
                if self.get_stat_percent("hp") < self.statthreshold.hp[1]:
                    if not fun():
                        continue
                    return True
        try:
            self.driver.find_element(By.XPATH, searchxpath_fun(["/y/e/healthpot.png"]))
        except NoSuchElementException:
            return self.use_item("Health Draught")
        return False

    @return_false_on_nosuch
    def check_mp(self) -> bool:
        if self.get_stat_percent("mp") < self.statthreshold.mp[0]:
            for key in ["Mana Potion", "Mana Elixir", "Last Elixir"]:
                if self.use_item(key):
                    return True
        try:
            self.driver.find_element(By.XPATH, searchxpath_fun(["/y/e/manapot.png"]))
        except NoSuchElementException:
            return self.use_item("Mana Draught")
        return False

    @return_false_on_nosuch
    def check_sp(self) -> bool:
        if self.get_stat_percent("sp") < self.statthreshold.sp[0]:
            for key in ["Spirit Potion", "Spirit Elixir", "Last Elixir"]:
                if self.use_item(key):
                    return True
        try:
            self.driver.find_element(By.XPATH, searchxpath_fun(["/y/e/spiritpot.png"]))
        except NoSuchElementException:
            return self.use_item("Spirit Draught")
        return False

    @return_false_on_nosuch
    def check_overcharge(self) -> bool:
        clickspirit = partial(
            ElementActionManager(self).click_and_wait_log,
            self.driver.find_element(By.ID, "ckey_spirit"),
        )
        if (
            self.count_monster() >= self.statthreshold.countmonster[1]
            and self.get_stat_percent("overcharge") < self.statthreshold.overcharge[0]
        ):
            try:
                self.driver.find_element(
                    By.XPATH, searchxpath_fun(["/y/battle/spirit_a.png"])
                )
                clickspirit()
                return True
            except NoSuchElementException:
                return False
        if (
            self.get_stat_percent("overcharge") > self.statthreshold.overcharge[1]
            and self.get_stat_percent("sp") > self.statthreshold.sp[0]
        ):
            try:
                self.driver.find_element(
                    By.XPATH, searchxpath_fun(["/y/battle/spirit_a.png"])
                )
            except NoSuchElementException:
                clickspirit()
                return True
        return False

    def count_monster(self) -> int:
        count = 0
        for n in range(10):
            count += (
                len(
                    self.driver.find_elements(
                        By.XPATH,
                        '//div[@id="mkey_{n}" and not(.//img[@src="/y/s/nbardead.png"]) and not(.//img[@src="/isekai/y/s/nbardead.png"])]'.format(
                            n=n
                        ),
                    )
                )
                > 0
            )
        return count

    def go_next_floor(self) -> bool:
        try:
            ElementActionManager(self).click_and_wait_log(
                self.driver.find_element(
                    By.XPATH,
                    searchxpath_fun(
                        [
                            "/y/battle/arenacontinue.png",
                            "/y/battle/grindfestcontinue.png",
                            "/y/battle/itemworldcontinue.png",
                        ]
                    ),
                )
            )
            return True
        except NoSuchElementException:
            return False

    def click_ofc(self) -> None:
        if self.with_ofc and (self.get_stat_percent("overcharge") > 220):
            try:
                self.driver.find_element(
                    By.XPATH, searchxpath_fun(["/y/battle/spirit_a.png"])
                )
                if self.count_monster() >= self.statthreshold.countmonster[1]:
                    self.click_skill("Orbital Friendship Cannon", iswait=False)
            except NoSuchElementException:
                pass

    def attack(self) -> bool:
        self.click_ofc()
        for n in [2, 1, 3, 5, 4, 6, 8, 7, 9, 0]:
            try:
                self.driver.find_element(
                    By.XPATH,
                    '//div[@id="mkey_{n}" and not(.//img[@src="/y/s/nbardead.png"]) and not(.//img[@src="/isekai/y/s/nbardead.png"])]'.format(
                        n=n
                    ),
                )
                if self.get_stat_percent("mp") > self.statthreshold.mp[1]:
                    try:
                        self.driver.find_element(
                            By.XPATH,
                            '//div[@id="mkey_{n}" and not(.//img[@src="/y/e/imperil.png"]) and not(.//img[@src="/isekai/y/e/imperil.png"])]'.format(
                                n=n
                            ),
                        )
                        self.click_skill("Imperil", iswait=False)
                    except NoSuchElementException:
                        pass
                ElementActionManager(self).click_and_wait_log(
                    self.driver.find_element(
                        By.XPATH, '//div[@id="mkey_{n}"]'.format(n=n)
                    )
                )
                return True
            except NoSuchElementException:
                pass
        return False

    def finish_battle(self) -> bool:
        try:
            ending = self.driver.find_element(
                By.XPATH, searchxpath_fun(["/y/battle/finishbattle.png"])
            )
            actions = ActionChains(self.driver)
            actions.move_to_element(ending).click().perform()
            return True
        except NoSuchElementException:
            return False

    def battle(self) -> None:
        while True:
            if self.go_next_floor():
                continue

            if PonyChart(self).check():
                continue

            if self.finish_battle():
                break

            iscontinue = False
            for fun in [
                self.check_hp,
                self.check_mp,
                self.check_sp,
                self.check_overcharge,
            ]:
                iscontinue |= fun()
                if iscontinue:
                    break
            if iscontinue:
                continue

            try:
                self.driver.find_element(By.XPATH, searchxpath_fun(["/y/e/regen.png"]))
            except NoSuchElementException:
                self.click_skill("Regen")
                continue

            try:
                self.driver.find_element(
                    By.XPATH, searchxpath_fun(["/y/e/heartseeker.png"])
                )
            except NoSuchElementException:
                self.click_skill("Heartseeker")
                continue

            try:
                self.driver.find_element(
                    By.XPATH, searchxpath_fun(["/y/e/channeling.png"])
                )
                self.click_skill("Heartseeker")
                continue
            except NoSuchElementException:
                pass

            if self.attack():
                continue
