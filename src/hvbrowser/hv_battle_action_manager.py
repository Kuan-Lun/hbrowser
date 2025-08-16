import time

from hv_bie import parse_snapshot
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.remote.webdriver import WebDriver
from selenium.webdriver.remote.webelement import WebElement

from .hv import HVDriver
from .hv_battle_observer_pattern import BattleDashboard


class ElementActionManager:
    def __init__(self, driver: HVDriver, battle_dashboard: BattleDashboard) -> None:
        self.hvdriver = driver
        self.battle_dashboard = battle_dashboard

    @property
    def driver(self) -> WebDriver:
        return self.hvdriver.driver

    def click(self, element: WebElement) -> None:
        actions = ActionChains(self.driver)
        actions.move_to_element(element).click().perform()

    def click_and_wait_log(self, element: WebElement, is_retry=True) -> None:
        html = self.battle_dashboard.log_entries.get_new_lines(
            parse_snapshot(self.hvdriver.driver.page_source)
        )
        self.click(element)

        # 優化 1: 減少初始等待時間
        time.sleep(0.05)  # 從 0.1 減少到 0.05

        n: float = 0
        check_interval = 0.05  # 優化 2: 減少檢查間隔
        max_wait_time = 5.0  # 優化 3: 減少最大等待時間

        while html == self.battle_dashboard.log_entries.get_new_lines(
            parse_snapshot(self.hvdriver.driver.page_source)
        ):
            time.sleep(check_interval)
            n += check_interval
            if n >= max_wait_time:
                if is_retry:
                    self.hvdriver.driver.refresh()
                    return self.click_and_wait_log(element, is_retry=False)
                else:
                    raise TimeoutError("I don't know what happened.")

        # 優化 4: 確保至少有一些變化後再繼續
        time.sleep(0.01)  # 很短的穩定等待
