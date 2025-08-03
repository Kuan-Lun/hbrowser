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
        log_provider = LogProvider(self.hvdriver)
        html = log_provider.get()
        self.click(element)
        time.sleep(0.001)
        n: float = 0
        while html == log_provider.get():
            time.sleep(0.001)
            n += 0.001
            if n == 1000:
                raise TimeoutError("I don't know what happened.")
