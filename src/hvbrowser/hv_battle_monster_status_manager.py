from selenium.webdriver.common.by import By
from selenium.webdriver.remote.webdriver import WebDriver, WebElement

from .hv import HVDriver
from .hv_battle_dashboard import BattleDashBoard


# Debuff 名稱對應圖示檔名
BUFF_ICON_MAP = {
    "Imperil": ["imperil.png"],
    "Weaken": ["weaken.png"],
    "Blind": ["blind.png"],
    "Slow": ["slow.png"],
    "MagNet": ["magnet.png"],
    "Silence": ["silence.png"],
    "Drain": ["drainhp.png"],
    # 你可以繼續擴充
}


class MonsterStatusCache:
    """
    用於緩存怪物狀態的類別。
    這樣可以避免每次都從網頁重新獲取怪物狀態，提高性能。
    """

    def __init__(self) -> None:
        self.buff2ids: dict[str, list[int]] = dict()
        self.name2id: dict[str, int] = dict()

    def clear(self) -> None:
        self.buff2ids = dict()
        self.name2id = dict()


class MonsterStatusManager:
    def __init__(self, driver: HVDriver, battle_dashboard: BattleDashBoard) -> None:
        self.battle_dashboard = battle_dashboard
        self.cache = MonsterStatusCache()

    def clear_cache(self) -> None:
        self.cache.clear()

    @property
    def alive_count(self) -> int:
        """Returns the number of alive monsters in the battle."""
        return len(self.alive_monster_ids)

    @property
    def alive_monster_ids(self) -> list[int]:
        """Returns a list of IDs of alive monsters in the battle."""
        return [
            monster_id
            for monster_id, monster in self.battle_dashboard.monster_list.items()
            if monster.is_alive
        ]

    @property
    def alive_system_monster_ids(self) -> list[int]:
        """Returns a list of system monster IDs in the battle that have style attribute and are alive."""
        return [
            monster_id
            for monster_id, monster in self.battle_dashboard.monster_list.items()
            if monster.is_alive and monster.is_system
        ]

    def get_monster_ids_with_debuff(self, debuff: str) -> list[int]:
        """Returns a list of alive monster IDs that have the specified debuff."""

        if debuff not in self.cache.buff2ids:
            self.cache.buff2ids[debuff] = []
            for monster_id, monster in self.battle_dashboard.monster_list.items():
                if monster.is_alive and debuff in monster.buffs:
                    self.cache.buff2ids[debuff].append(monster_id)

        return self.cache.buff2ids[debuff]

    def get_monster_id_by_name(self, name: str) -> int:
        """
        根據怪物名稱取得對應的 monster id（如 mkey_0 會回傳 0）。
        """
        # 使用原始 XPath 邏輯確保正確性
        if name not in self.cache.name2id:
            for monster_id, monster in self.battle_dashboard.monster_list.items():
                if monster.name == name:
                    self.cache.name2id[name] = monster_id
                    return self.cache.name2id[name]
            return -1
        return self.cache.name2id[name]
