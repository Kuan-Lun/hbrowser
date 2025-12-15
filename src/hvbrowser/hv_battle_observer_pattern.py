import re
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from typing import Any, TypeVar, overload

from hv_bie import parse_snapshot
from hv_bie.types import BattleSnapshot

from .hv import HVDriver

_T = TypeVar("_T")


class DefaultMinusOneDict(dict[Any, int]):
    """A dict that returns -1 for missing keys without inserting them."""

    def __missing__(self, key: Any) -> int:
        return -1

    @overload
    def get(self, key: Any, /) -> int: ...

    @overload
    def get(self, key: Any, /, default: _T) -> int | _T: ...

    def get(self, key: Any, /, default: Any = None) -> Any:
        # By default, behave like __getitem__ and return -1 for missing keys
        # This intentionally changes dict's behavior to return -1 instead of None
        if default is None:
            return self[key]
        return super().get(key, default)


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

    def attach(self, observer: Observer) -> None:
        self._observers.append(observer)

    def detach(self, observer: Observer) -> None:
        self._observers.remove(observer)

    def notify(self) -> None:
        self.snap = parse_snapshot(self._hvdriver.driver.page_source)
        for observer in self._observers:
            observer.update(self.snap)


@dataclass
class OverviewMonsters:
    alive_monster: list[int] = field(default_factory=list)
    alive_system_monster: list[int] = field(default_factory=list)
    alive_monster_with_buff: dict[str, list[int]] = field(default_factory=dict)
    alive_monster_name: dict[str, int] = field(default_factory=DefaultMinusOneDict)


class ExtendedBattleSnapshot(BattleSnapshot):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)


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
        return snap.log.lines

    def update(self, snap: BattleSnapshot) -> None:
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
        self.overview_monsters = OverviewMonsters()
        self.log_entries: LogEntry = LogEntry()
        self.battle_subject.attach(self.log_entries)
        self.update()

    def update(self) -> None:
        self.battle_subject.notify()
        self.snap = self.battle_subject.snap
        self.update_overview_monsters()

    def update_overview_monsters(self) -> None:
        self.overview_monsters.alive_monster = [
            monster.slot_index
            for monster in self.snap.monsters.values()
            if monster.alive
        ]
        self.overview_monsters.alive_system_monster = [
            monster.slot_index
            for monster in self.snap.monsters.values()
            if monster.alive and monster.system_monster_type
        ]
        self.overview_monsters.alive_monster_with_buff = {
            buff: [
                monster.slot_index
                for monster in self.snap.monsters.values()
                if monster.alive and buff in monster.buffs
            ]
            for buff in set(
                buff
                for monster in self.snap.monsters.values()
                for buff in monster.buffs
            )
        }
        self.overview_monsters.alive_monster_name = {
            monster.name: monster.slot_index
            for monster in self.snap.monsters.values()
            if monster.alive
        }
