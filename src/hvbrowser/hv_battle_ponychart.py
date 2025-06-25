import time
from selenium.webdriver.common.by import By
from selenium.webdriver.remote.webdriver import WebDriver
from selenium.common.exceptions import NoSuchElementException

from hbrowser.beep import beep_os_independent

from .hv import HVDriver
from .hv_battle_stat_provider import StatProviderHP


class PonyChart:
    def __init__(self, driver: HVDriver) -> None:
        self.hvdriver: HVDriver = driver

    @property
    def driver(self) -> WebDriver:
        return self.hvdriver.driver

    def _check(self) -> bool:
        return self.driver.find_elements(By.ID, "riddlesubmit") != []

    def check(self) -> bool:
        isponychart: bool = self._check()
        if not isponychart:
            return isponychart

        beep_os_independent()

        waitlimit: float = 100
        while waitlimit > 0 and self._check():
            time.sleep(0.1)
            beep_os_independent()
            waitlimit -= 0.1

        if waitlimit <= 0:
            print("PonyChart check timeout, please check your network connection.")

        time.sleep(1)

        return isponychart
