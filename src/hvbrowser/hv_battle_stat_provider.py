import re
from abc import ABC, abstractmethod

from selenium.webdriver.common.by import By
from selenium.webdriver.remote.webdriver import WebDriver

from .hv import HVDriver, searchxpath_fun


class StatProvider(ABC):
    def __init__(self, driver: HVDriver) -> None:
        self.hvdriver = driver

    @property
    def driver(self) -> WebDriver:
        return self.hvdriver.driver

    @property
    @abstractmethod
    def _factor(self) -> int:
        pass

    @property
    @abstractmethod
    def searchxpath(self) -> str:
        pass

    def get_percent(self) -> float:
        img_element = self.hvdriver.find_element_chain(
            (By.ID, "pane_vitals"),
            (By.XPATH, self.searchxpath),
        )
        style_attribute = str(img_element.get_attribute("style"))
        width_value_match = re.search(r"width:\s*(\d+)px", style_attribute)
        if width_value_match is None:
            raise ValueError("width_value_match is None")
        width_value_match = width_value_match.group(1)  # type: ignore
        return self._factor * (int(width_value_match) - 1) / (414 - 1)  # type: ignore


class StatProviderHP(StatProvider):
    @property
    def _factor(self) -> int:
        return 100

    @property
    def searchxpath(self) -> str:
        return searchxpath_fun(["/y/bar_bgreen.png", "/y/bar_dgreen.png"])


class StatProviderMP(StatProvider):
    @property
    def _factor(self) -> int:
        return 100

    @property
    def searchxpath(self) -> str:
        return searchxpath_fun(["/y/bar_blue.png"])


class StatProviderSP(StatProvider):
    @property
    def _factor(self) -> int:
        return 100

    @property
    def searchxpath(self) -> str:
        return searchxpath_fun(["/y/bar_red.png"])


class StatProviderOvercharge(StatProvider):
    @property
    def _factor(self) -> int:
        return 250

    @property
    def searchxpath(self) -> str:
        return searchxpath_fun(["/y/bar_orange.png"])

    def get_spirit_stance_status(self) -> str:
        if self.driver.find_elements(
            By.XPATH, searchxpath_fun(["/y/battle/spirit_a.png"])
        ):
            return "activated"

        if self.driver.find_elements(
            By.XPATH, searchxpath_fun(["/y/battle/spirit_n.png"])
        ):
            return "inactive"

        return "charging"
