import time
from selenium.webdriver.common.by import By
from selenium.webdriver.remote.webdriver import WebDriver

from hbrowser.beep import beep_os_independent

from .hv import HVDriver


class PonyChart:
    def __init__(self, driver: HVDriver) -> None:
        self.hvdriver: HVDriver = driver

    @property
    def driver(self) -> WebDriver:
        return self.hvdriver.driver

    def _check(self) -> bool:
        return self.driver.find_elements(By.ID, "ponychart") != []

    def check(self) -> bool:
        isponychart: bool = self._check()
        if not isponychart:
            return isponychart

        beep_os_independent()
        time.sleep(5)

        waitlimit: float = 100
        while waitlimit > 0 and self._check():
            time.sleep(0.1)
            waitlimit -= 0.1

        time.sleep(1)

        return isponychart
