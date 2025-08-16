import re
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field

from hv_bie import parse_snapshot
from hv_bie.types import BattleSnapshot

from .hv import HVDriver


class Observer(ABC):
    @abstractmethod
    def update(self, snap: BattleSnapshot) -> None:
        """更新物件狀態，就地修改而非建立新物件"""
        pass


class BattleSubject:
    def __init__(self, driver: HVDriver):
        self._observers: list[Observer] = list()
        self._hvdriver = driver
        self.snap = parse_snapshot(driver.driver.page_source)

    def attach(self, observer: Observer):
        self._observers.append(observer)

    def detach(self, observer: Observer):
        self._observers.remove(observer)

    def notify(self):
        self.snap = parse_snapshot(self._hvdriver.driver.page_source)
        for observer in self._observers:
            observer.update(self.snap)


@dataclass
class LogEntry(Observer):
    current_round: int = 0
    prev_round: int = 0
    total_round: int = 0
    prev_lines: deque[str] = field(default_factory=lambda: deque(maxlen=1000))
    current_lines: list[str] = field(default_factory=list)

    def _parse_round_info(self, lines: list[str]) -> None:
        for line in lines:
            if "Round" in line:
                match = re.search(r"Round (\d+) / (\d+)", line)
                if match:
                    self.current_round = int(match.group(1))
                    if self.prev_round != self.current_round:
                        self.prev_round = self.current_round
                        self.prev_lines = deque(maxlen=1000)
                    self.total_round = int(match.group(2))

    def get_new_lines(self, snap: BattleSnapshot) -> list[str]:
        textlog = snap.log.lines
        return textlog

    def update(self, snap: BattleSnapshot):
        lines = self.get_new_lines(snap)
        if lines:
            self.current_lines = [line for line in lines if line not in self.prev_lines]
            self._parse_round_info(self.current_lines)
            self.prev_lines.extend(self.current_lines)


class BattleDashboard:
    def __init__(self, driver: HVDriver):
        self._hvdriver = driver
        self.battle_subject = BattleSubject(driver)
        self.snap = self.battle_subject.snap
        self.log_entries: LogEntry = LogEntry()
        self.battle_subject.attach(self.log_entries)
        self.update()

    def update(self):
        self.battle_subject.notify()
        self.snap = self.battle_subject.snap
