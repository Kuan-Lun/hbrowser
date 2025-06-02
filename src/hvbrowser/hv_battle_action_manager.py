import time

from selenium.webdriver.remote.webelement import WebElement
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.remote.webdriver import WebDriver

from .hv import HVDriver
from .hv_battle_log import LogProvider


class ElementActionManager:
    def __init__(self, driver: HVDriver) -> None:
        self.hvdriver: HVDriver = driver

    @property
    def driver(self) -> WebDriver:
        return self.hvdriver.driver

    def click(self, element: WebElement) -> None:
        actions = ActionChains(self.driver)
        actions.move_to_element(element).click().perform()

    def click_and_wait_log(self, element: WebElement) -> None:
        html = LogProvider(self.hvdriver).get()
        self.click(element)
        time.sleep(0.01)
        n: float = 0
        while html == LogProvider(self.hvdriver).get():
            time.sleep(0.01)
            n += 0.01
            if n == 10:
                raise TimeoutError("I don't know what happened.")
