import time

from selenium.webdriver.remote.webelement import WebElement
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.remote.webdriver import WebDriver

from .hv import HVDriver
from .hv_battle_dashboard import BattleDashBoard


class ElementActionManager:
    def __init__(self, driver: HVDriver, battle_dashboard: BattleDashBoard) -> None:
        self.hvdriver = driver
        self.battle_dashboard = battle_dashboard

    @property
    def driver(self) -> WebDriver:
        return self.hvdriver.driver

    def click(self, element: WebElement) -> None:
        actions = ActionChains(self.driver)
        actions.move_to_element(element).click().perform()

    def click_and_wait_log(self, element: WebElement) -> None:
        html = self.battle_dashboard.get_current_log()
        self.click(element)

        # 優化 1: 減少初始等待時間
        time.sleep(0.05)  # 從 0.1 減少到 0.05

        n: float = 0
        check_interval = 0.05  # 優化 2: 減少檢查間隔
        max_wait_time = 5.0  # 優化 3: 減少最大等待時間

        while html == self.battle_dashboard.get_current_log():
            time.sleep(check_interval)
            n += check_interval
            if n >= max_wait_time:
                raise TimeoutError("I don't know what happened.")

        # 優化 4: 確保至少有一些變化後再繼續
        time.sleep(0.01)  # 很短的穩定等待
