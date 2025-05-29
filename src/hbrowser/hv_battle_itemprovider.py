from selenium.webdriver.common.by import By
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.remote.webdriver import WebDriver

from .hv import HVDriver, searchxpath_fun
from .hv_battle_actionmanager import ElementActionManager


class ItemProvider:
    def __init__(self, driver: HVDriver) -> None:
        self.hvdriver: HVDriver = driver

    @property
    def driver(self) -> WebDriver:
        return self.hvdriver.driver

    def use(self, item: str) -> bool:
        try:
            ElementActionManager(self.hvdriver).click(
                self.driver.find_element(
                    By.XPATH,
                    searchxpath_fun(["/y/battle/items_n.png"]),
                )
            )
        except NoSuchElementException:
            return False

        try:
            ElementActionManager(self.hvdriver).click_and_wait_log(
                self.driver.find_element(
                    By.XPATH,
                    "//div[@class=\"fc2 fal fcb\"]/div[contains(text(), '{key}')]".format(
                        key=item
                    ),
                )
            )
            return True
        except NoSuchElementException:
            return False
