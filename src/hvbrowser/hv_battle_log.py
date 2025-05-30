from selenium.webdriver.common.by import By
from selenium.webdriver.remote.webdriver import WebDriver

from .hv import HVDriver


class LogProvider:
    def __init__(self, driver: HVDriver) -> None:
        self.hvdriver: HVDriver = driver

    @property
    def driver(self) -> WebDriver:
        return self.hvdriver.driver

    def get(self) -> str:
        result = self.driver.find_element(By.ID, "textlog").get_attribute("outerHTML")
        if result is None:
            return ""
        return result
