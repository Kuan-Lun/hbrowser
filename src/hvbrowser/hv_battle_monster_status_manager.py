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


class MonsterStatusManager:
    # 保留原始的 XPath 邏輯以確保正確性，但進行性能優化
    ALIVE_MONSTER_XPATH = f'/div[starts-with(@id, "mkey_") and not(.//img[@src="/y/s/nbardead.png"]) and not(.//img[@src="/isekai/y/s/nbardead.png"])]'
    MONSTER_CSS_SELECTOR = '#csp #mainpane #battle_right #pane_monster div[id^="mkey_"]'  # 用於快速查找所有怪物
    DEAD_IMG_SRCS = ["/y/s/nbardead.png", "/isekai/y/s/nbardead.png"]

    def __init__(self, driver: HVDriver) -> None:
        self.hvdriver: HVDriver = driver

    @property
    def driver(self) -> WebDriver:
        return self.hvdriver.driver

    def get_pane_monster(self) -> WebElement:
        return self.driver.find_element(By.ID, "pane_monster")

    def get_monsters_elements(self) -> list[WebElement]:
        """
        Returns a list of WebElement representing all monsters in the battle.
        """
        return self.get_pane_monster().find_elements(
            By.CSS_SELECTOR, 'div[id^="mkey_"]'
        )

    def _is_monster_alive(self, monster_element: WebElement) -> bool:
        """检查怪物是否活着（没有死亡图标）"""
        imgs = monster_element.find_elements(By.TAG_NAME, "img")
        for img in imgs:
            src = img.get_attribute("src")
            if src and "nbardead.png" in src:
                return False
        return True

    def get_alive_monsters_elements(self) -> list[WebElement]:
        """返回所有活着的怪物元素"""
        return [el for el in self.get_monsters_elements() if self._is_monster_alive(el)]

    @property
    def alive_count(self) -> int:
        """Returns the number of alive monsters in the battle."""
        return len(self.alive_monster_ids)

    @property
    def alive_monster_ids(self) -> list[int]:
        """Returns a list of IDs of alive monsters in the battle."""
        return [
            int(id_.removeprefix("mkey_"))
            for el in self.get_alive_monsters_elements()
            if (id_ := el.get_attribute("id")) is not None
        ]

    @property
    def alive_system_monster_ids(self) -> list[int]:
        """Returns a list of system monster IDs in the battle that have style attribute and are alive."""
        return [
            int(id_.removeprefix("mkey_"))
            for el in self.get_alive_monsters_elements()
            if (id_ := el.get_attribute("id")) is not None
            and el.get_attribute("style") is not None
        ]

    def get_monster_ids_with_debuff(self, debuff: str) -> list[int]:
        """Returns a list of alive monster IDs that have the specified debuff."""
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

        return result

    def get_monster_id_by_name(self, name: str) -> int:
        """
        根據怪物名稱取得對應的 monster id（如 mkey_0 會回傳 0）。
        """
        # 使用原始 XPath 邏輯確保正確性
        xpath = f'/div[starts-with(@id, "mkey_")][.//div[text()="{name}"]]'
        elements = self.get_pane_monster().find_elements(By.XPATH, xpath)
        if elements:
            id_ = elements[0].get_attribute("id")
            if id_ and id_.startswith("mkey_"):
                return int(id_.removeprefix("mkey_"))
        return -1
