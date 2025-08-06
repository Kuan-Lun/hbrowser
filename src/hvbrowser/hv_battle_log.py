import re
from collections import deque

from bs4 import BeautifulSoup
from selenium.webdriver.remote.webdriver import WebDriver

from .hv import HVDriver
from .hv_battle_dashboard import BattleDashBoard


class LogProvider:
    def __init__(self, driver: HVDriver, battle_dashboard: BattleDashBoard) -> None:
        self.hvdriver: HVDriver = driver
        self.battle_dashboard: BattleDashBoard = battle_dashboard
        self._reset_prev_lines()
        self.current_round = 0
        self.prev_round = 0
        self.total_round = 0

    def _reset_prev_lines(self) -> None:
        self.prev_lines: deque[str] = deque(maxlen=1000)

    @property
    def driver(self) -> WebDriver:
        return self.hvdriver.driver

    def _parse_round_info(self, lines: list[str]) -> None:
        for line in lines:
            if "Round" in line:
                match = re.search(r"Round (\d+) / (\d+)", line)
                if match:
                    self.current_round = int(match.group(1))
                    if self.prev_round != self.current_round:
                        self.prev_round = self.current_round
                        self._reset_prev_lines()
                    self.total_round = int(match.group(2))

    def get_new_logs(self) -> list[str]:
        soup = BeautifulSoup(self.battle_dashboard.log, "html.parser")
        lines = [td.text for td in soup.find_all("td", class_="tl")][-1::-1]
        new_lines = [line for line in lines if line not in self.prev_lines]
        self._parse_round_info(new_lines)
        self.prev_lines.extend(new_lines)
        return new_lines
