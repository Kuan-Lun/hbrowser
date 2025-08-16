from .hv import HVDriver
from .hv_battle_observer_pattern import BattleDashboard


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


class MonsterStatusManager:
    def __init__(self, driver: HVDriver, battle_dashboard: BattleDashboard) -> None:
        self.battle_dashboard = battle_dashboard

    @property
    def alive_count(self) -> int:
        """Returns the number of alive monsters in the battle."""
        return len(self.alive_monster_ids)

    @property
    def alive_monster_ids(self) -> list[int]:
        """Returns a list of IDs of alive monsters in the battle."""
        return [
            monster.slot_index
            for monster in self.battle_dashboard.snap.monsters.values()
            if monster.alive
        ]

    @property
    def alive_system_monster_ids(self) -> list[int]:
        """Returns a list of system monster IDs in the battle that have style attribute and are alive."""
        return [
            monster.slot_index
            for monster in self.battle_dashboard.snap.monsters.values()
            if monster.alive and monster.system_monster_type is not None
        ]

    def get_monster_ids_with_debuff(self, debuff: str) -> list[int]:
        """Returns a list of alive monster IDs that have the specified debuff."""

        result = list()
        for monster in self.battle_dashboard.snap.monsters.values():
            if monster.alive and debuff in monster.buffs:
                result.append(monster.slot_index)

        return result

    def get_monster_id_by_name(self, name: str) -> int:
        """
        根據怪物名稱取得對應的 monster id（如 mkey_0 會回傳 0）。
        """
        # 使用原始 XPath 邏輯確保正確性
        for monster in self.battle_dashboard.snap.monsters.values():
            if monster.name == name:
                return monster.slot_index
        return -1
