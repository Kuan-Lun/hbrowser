import time
import re
from functools import partial

from selenium.common.exceptions import NoSuchElementException, JavascriptException
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.by import By
from selenium.webdriver.remote.webelement import WebElement

from .beep import beep_os_independent
from .hv import HVDriver, searchxpath_fun


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

    def check_ponychart(self) -> bool:
        try:
            self.driver.find_element(By.ID, "riddlesubmit")
        except NoSuchElementException:
            return False
        beep_os_independent()
        time.sleep(20)
        return True

    def _get_stat_rate(self, key: str) -> float:
        match key:
            case "HP":
                searchxpath = searchxpath_fun(
                    ["/y/bar_bgreen.png", "/y/bar_dgreen.png"]
                )
                factor = 100
            case "MP":
                searchxpath = searchxpath_fun(["/y/bar_blue.png"])
                factor = 100
            case "SP":
                searchxpath = searchxpath_fun(["/y/bar_red.png"])
                factor = 100
            case "Overcharge":
                searchxpath = searchxpath_fun(["/y/bar_orange.png"])
                factor = 250

        img_element = self.driver.find_element(By.XPATH, searchxpath)
        style_attribute = img_element.get_attribute("style")
        width_value_match = re.search(r"width:\s*(\d+)px", style_attribute)
        if width_value_match is None:
            raise ValueError("width_value_match is None")
        width_value_match = width_value_match.group(1)  # type: ignore
        return factor * (int(width_value_match) - 1) / (414 - 1)  # type: ignore

    def get_hp(self) -> float:
        return self._get_stat_rate("HP")

    def get_mp(self) -> float:
        return self._get_stat_rate("MP")

    def get_sp(self) -> float:
        return self._get_stat_rate("SP")

    def get_overcharge(self) -> float:
        return self._get_stat_rate("Overcharge")

    def _click2newlog(self, element: WebElement) -> None:
        html = self.driver.find_element(By.ID, "textlog").get_attribute("outerHTML")
        actions = ActionChains(self.driver)
        actions.move_to_element(element).click().perform()
        time.sleep(0.01)
        n: float = 0
        while html == self.driver.find_element(By.ID, "textlog").get_attribute(
            "outerHTML"
        ):
            time.sleep(0.01)
            n += 0.01
            if n == 10:
                raise TimeoutError("I don't know what happened.")

    def click_item(self, key: str) -> bool:
        try:
            element = self.driver.find_element(
                By.XPATH,
                searchxpath_fun(["/y/battle/items_n.png"]),
            )
            element.click()
        except NoSuchElementException:
            return False

        try:
            self._click2newlog(
                self.driver.find_element(
                    By.XPATH,
                    "//div[@class=\"fc2 fal fcb\"]/div[contains(text(), '{key}')]".format(
                        key=key
                    ),
                )
            )
            return True
        except NoSuchElementException:
            return False

    def click_skill(self, key: str, iswait=True) -> bool:
        def click_this_skill(skillstring: str) -> None:
            element = self.driver.find_element(By.XPATH, skillstring)
            if iswait:
                self._click2newlog(element)
            else:
                actions = ActionChains(self.driver)
                actions.move_to_element(element).click().perform()
                time.sleep(0.01)

        skillstring = "//div[not(@style)]/div/div[contains(text(), '{key}')]".format(
            key=key
        )
        try:
            click_this_skill(skillstring)
        except JavascriptException:
            element = self.driver.find_element(
                By.XPATH, searchxpath_fun(["/y/battle/skill_n.png"])
            )
            element.click()
            try:
                click_this_skill(skillstring)
            except JavascriptException:
                element = self.driver.find_element(
                    By.XPATH, searchxpath_fun(["/y/battle/sbsel_spells_n.png"])
                )
                element.click()
                click_this_skill(skillstring)
        except NoSuchElementException:
            return False
        return True

    def check_hp(self) -> bool:
        if self.get_hp() < self.statthreshold.hp[0]:
            for fun in [
                partial(self.click_skill, "Full-Cure"),
                partial(self.click_item, "Health Potion"),
                partial(self.click_item, "Health Elixir"),
                partial(self.click_item, "Last Elixir"),
                partial(self.click_skill, "Cure"),
            ]:
                if self.get_hp() < self.statthreshold.hp[0]:
                    if not fun():
                        continue
                    return True

        if self.get_hp() < self.statthreshold.hp[1]:
            for fun in [
                partial(self.click_skill, "Cure"),
                partial(self.click_skill, "Full-Cure"),
                partial(self.click_item, "Health Potion"),
                partial(self.click_item, "Health Elixir"),
                partial(self.click_item, "Last Elixir"),
            ]:
                if self.get_hp() < self.statthreshold.hp[1]:
                    if not fun():
                        continue
                    return True
        try:
            self.driver.find_element(By.XPATH, searchxpath_fun(["/y/e/healthpot.png"]))
        except NoSuchElementException:
            return self.click_item("Health Draught")
        return False

    def check_mp(self) -> bool:
        if self.get_mp() < self.statthreshold.mp[0]:
            for key in ["Mana Potion", "Mana Elixir", "Last Elixir"]:
                if self.click_item(key):
                    return True
        try:
            self.driver.find_element(By.XPATH, searchxpath_fun(["/y/e/manapot.png"]))
        except NoSuchElementException:
            return self.click_item("Mana Draught")
        return False

    def check_sp(self) -> bool:
        if self.get_sp() < self.statthreshold.sp[0]:
            for key in ["Spirit Potion", "Spirit Elixir", "Last Elixir"]:
                if self.click_item(key):
                    return True
        try:
            self.driver.find_element(By.XPATH, searchxpath_fun(["/y/e/spiritpot.png"]))
        except NoSuchElementException:
            return self.click_item("Spirit Draught")
        return False

    def check_overcharge(self) -> bool:
        clickspirit = partial(
            self._click2newlog, self.driver.find_element(By.ID, "ckey_spirit")
        )
        if (
            self.count_monster() >= self.statthreshold.countmonster[1]
            and self.get_overcharge() < self.statthreshold.overcharge[0]
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
            self.get_overcharge() > self.statthreshold.overcharge[1]
            and self.get_sp() > self.statthreshold.sp[0]
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
            self._click2newlog(
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
        if self.with_ofc and (self.get_overcharge() > 220):
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
                if self.get_mp() > self.statthreshold.mp[1]:
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
                self._click2newlog(
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

            if self.check_ponychart():
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
