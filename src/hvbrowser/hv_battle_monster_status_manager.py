from selenium.webdriver.common.by import By
from selenium.webdriver.remote.webdriver import WebDriver, WebElement

from .hv import HVDriver


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
        self.pane_monster: WebElement | None = None
        self.monsters_elements: list[WebElement] = list()
        self.alive_monsters_elements: list[WebElement] = list()
        self.alive_monster_ids: list[int] = list()
        self.alive_system_monster_ids: list[int] = list()
        self.buff2ids: dict[str, list[int]] = dict()
        self.name2id: dict[str, int] = dict()

    def clear(self) -> None:
        self.pane_monster = None
        self.monsters_elements = list()
        self.alive_monsters_elements = list()
        self.alive_monster_ids = list()
        self.alive_system_monster_ids = list()
        self.buff2ids = dict()
        self.name2id = dict()


class MonsterStatusManager:
    # 保留原始的 XPath 邏輯以確保正確性，但進行性能優化
    def __init__(self, driver: HVDriver) -> None:
        self.hvdriver: HVDriver = driver
        self.cache = MonsterStatusCache()

    def clear_cache(self) -> None:
        self.cache.clear()

    @property
    def driver(self) -> WebDriver:
        return self.hvdriver.driver

    def get_pane_monster(self) -> WebElement:
        if self.cache.pane_monster is None:
            self.cache.pane_monster = self.driver.find_element(By.ID, "pane_monster")
        return self.cache.pane_monster

    def get_monsters_elements(self) -> list[WebElement]:
        """
        Returns a list of WebElement representing all monsters in the battle.
        """
        if not bool(self.cache.monsters_elements):
            monsters: list[WebElement] = list()
            pane = self.get_pane_monster()

            for i in range(10):
                monsters += pane.find_elements(By.ID, f"mkey_{i}")

            self.cache.monsters_elements = monsters

        return self.cache.monsters_elements

    def get_alive_monsters_elements(self) -> list[WebElement]:
        """返回所有活着的怪物元素"""
        if not bool(self.cache.alive_monsters_elements):
            self.cache.alive_monsters_elements = [
                el
                for el in self.get_monsters_elements()
                if el.get_attribute("onclick") is not None
            ]
        return self.cache.alive_monsters_elements

    @property
    def alive_count(self) -> int:
        """Returns the number of alive monsters in the battle."""
        return len(self.cache.alive_monster_ids)

    @property
    def alive_monster_ids(self) -> list[int]:
        """Returns a list of IDs of alive monsters in the battle."""
        if not bool(self.cache.alive_monster_ids):
            self.cache.alive_monster_ids = [
                int(id_.removeprefix("mkey_"))
                for el in self.get_alive_monsters_elements()
                if (id_ := el.get_attribute("id")) is not None
            ]
        return self.cache.alive_monster_ids

    @property
    def alive_system_monster_ids(self) -> list[int]:
        """Returns a list of system monster IDs in the battle that have style attribute and are alive."""
        if not bool(self.cache.alive_system_monster_ids):
            self.cache.alive_system_monster_ids = [
                int(id_.removeprefix("mkey_"))
                for el in self.get_alive_monsters_elements()
                if (id_ := el.get_attribute("id")) is not None
                and el.get_attribute("style") is not None
            ]
        return self.cache.alive_system_monster_ids

    def get_monster_ids_with_debuff(self, debuff: str) -> list[int]:
        """Returns a list of alive monster IDs that have the specified debuff."""

        if debuff not in self.cache.buff2ids:

            icons = BUFF_ICON_MAP.get(debuff, [f"{debuff}.png"])
            result = []

            # 只检查活着的怪物
            for monster_el in self.get_alive_monsters_elements():
                if (id_ := monster_el.get_attribute("id")) is not None:
                    # 检查怪物元素内是否有包含指定图标的图片
                    for icon in icons:
                        imgs = monster_el.find_elements(By.TAG_NAME, "img")
                        for img in imgs:
                            src = img.get_attribute("src")
                            if src and icon in src:
                                result.append(int(id_.removeprefix("mkey_")))
                                break
                        else:
                            continue
                        break
            self.cache.buff2ids[debuff] = result

        return self.cache.buff2ids[debuff]

    def get_monster_id_by_name(self, name: str) -> int:
        """
        根據怪物名稱取得對應的 monster id（如 mkey_0 會回傳 0）。
        """
        # 使用原始 XPath 邏輯確保正確性
        if name not in self.cache.name2id:
            xpath = f'/div[starts-with(@id, "mkey_")][.//div[text()="{name}"]]'
            elements = self.get_pane_monster().find_elements(By.XPATH, xpath)
            if elements:
                id_ = elements[0].get_attribute("id")
                if id_ and id_.startswith("mkey_"):
                    self.cache.name2id[name] = int(id_.removeprefix("mkey_"))
                    return self.cache.name2id[name]
            return -1
        return self.cache.name2id[name]
