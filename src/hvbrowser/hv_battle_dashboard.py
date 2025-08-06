import re
from typing import Dict, List, Optional, Union

from bs4 import BeautifulSoup

from .hv import HVDriver

BUFF2ICONS = {
    # Item icons
    "Health Draught": {"/y/e/healthpot.png"},
    "Mana Draught": {"/y/e/manapot.png"},
    "Spirit Draught": {"/y/e/spiritpot.png"},
    "Scroll of Life": {"/y/e/sparklife_scroll.png"},
    # Skill icons
    "Absorb": {"/y/e/absorb.png", "/y/e/absorb_scroll.png"},
    "Heartseeker": {"/y/e/heartseeker.png"},
    "Regen": {"/y/e/regen.png"},
    "Shadow Veil": {"/y/e/shadowveil.png"},
    "Spark of Life": {"/y/e/sparklife.png", "/y/e/sparklife_scroll.png"},
    # Spirit icon
    "Spirit Stance": {"/y/battle/spirit_a.png"},
}

ICON2BUFF: Dict[str, List[str]] = {}
for buff_name, icon_paths in BUFF2ICONS.items():
    for icon_path in icon_paths:
        filename = icon_path.split("/")[-1]
        if filename not in ICON2BUFF:
            ICON2BUFF[filename] = []
        ICON2BUFF[filename].append(buff_name)

ADDITIONAL_ICON_MAPPINGS = {
    "protection.png": "Protection",
    "haste.png": "Haste",
    "shadowveil.png": "Shadow Veil",
    "spiritshield.png": "Spirit Shield",
    "absorb.png": "Absorb",
    "overwhelming.png": "Overwhelming Strikes",
    "healthpot.png": "Health Draught",
    "manapot.png": "Mana Draught",
    "regen.png": "Regen",
    "sparklife_scroll.png": "Scroll of Life",
    "heartseeker.png": "Heartseeker",
    "spiritpot.png": "Spirit Draught",
    "wpn_ap.png": "Penetrated Armor",
    "firedot.png": "Searing Skin",
    "wpn_stun.png": "Stunned",
}

for icon, buff in ADDITIONAL_ICON_MAPPINGS.items():
    if icon not in ICON2BUFF:
        ICON2BUFF[icon] = [buff]
    elif buff not in ICON2BUFF[icon]:
        ICON2BUFF[icon].append(buff)


class SpiritData:
    def __init__(self, status, cost_type, cost, cooldown):
        self.status = status
        self.cost_type = cost_type
        self.cost = cost
        self.cooldown = cooldown


class SkillBook:
    def __init__(self):
        self.skills: Dict[str, SpiritData] = {}
        self.spells: Dict[str, SpiritData] = {}


class CharacterVitals:
    def __init__(self, soup: BeautifulSoup):
        self.hp: float = 0.0
        self.mp: float = 0.0
        self.sp: float = 0.0
        self.overcharge: float = 0.0
        self._parse_vitals(soup)

    def _parse_vitals(self, soup: BeautifulSoup) -> None:
        """解析角色的生命值、法力值等信息"""
        pane_vitals = soup.find("div", id="pane_vitals")
        if not pane_vitals or not hasattr(pane_vitals, "find"):
            return

        # 查找生命值条 (bar_dgreen.png)
        health_bar = pane_vitals.find("img", src=re.compile(r"bar_dgreen\.png"))
        if health_bar:
            style = health_bar.get("style", "")
            width_match = re.search(r"width:(\d+)px", style)
            if width_match:
                width = int(width_match.group(1))
                self.hp = width / 414 * 100

        # 查找法力值条 (bar_blue.png)
        mana_bar = pane_vitals.find("img", src=re.compile(r"bar_blue\.png"))
        if mana_bar:
            style = mana_bar.get("style", "")
            width_match = re.search(r"width:(\d+)px", style)
            if width_match:
                width = int(width_match.group(1))
                self.mp = width / 414 * 100

        # 查找精神值条 (bar_red.png)
        spirit_bar = pane_vitals.find("img", src=re.compile(r"bar_red\.png"))
        if spirit_bar:
            style = spirit_bar.get("style", "")
            width_match = re.search(r"width:(\d+)px", style)
            if width_match:
                width = int(width_match.group(1))
                self.sp = width / 414 * 100

        # 查找过充值条 (bar_orange.png)
        overcharge_bar = pane_vitals.find("img", src=re.compile(r"bar_orange\.png"))
        if overcharge_bar:
            style = overcharge_bar.get("style", "")
            width_match = re.search(r"width:(\d+)px", style)
            if width_match:
                width = int(width_match.group(1))
                self.overcharge = width / 414 * 250


class Character:
    """角色类，解析并存储角色的所有信息"""

    def __init__(self, page_source: str):
        """
        初始化角色，解析页面源码

        Args:
            page_source: HTML 页面源码
        """
        self.soup = BeautifulSoup(page_source, "html.parser")
        self.buffs: Dict[str, Union[float, int]] = {}
        self.vitals: CharacterVitals = CharacterVitals(self.soup)
        self.skillbook = SkillBook()

        self._parse_buffs()
        self._parse_skills_and_spells()

    def _parse_buffs(self):
        """解析角色的 buff 效果"""
        # <img id="ckey_spirit" onclick="battle.lock_action(this,0,'spirit')" onmouseover="battle.set_infopane('Spirit')" src="/isekai/y/battle/spirit_n.png">
        spirit_img = self.soup.find("img", id="ckey_spirit")
        if spirit_img and "spirit_a.png" in spirit_img.get("src", ""):
            self.buffs["Spirit Stance"] = float("inf")

        pane_effects = self.soup.find("div", id="pane_effects")
        if not pane_effects:
            return

        # 查找所有 buff 图标
        buff_imgs = pane_effects.find_all("img")

        for img in buff_imgs:
            src = img.get("src", "")

            # 获取 buff 名称和持续时间
            onmouseover = img.get("onmouseover", "")

            # 从 onmouseover 属性中提取 buff 信息
            if "battle.set_infopane_effect" in onmouseover:
                # 解析格式：battle.set_infopane_effect('Buff Name', '...', duration)
                # 使用更精确的正则表达式
                match = re.search(
                    r"battle\.set_infopane_effect\('([^']+)',\s*'[^']*',\s*([^)]+)\)",
                    onmouseover,
                )
                if match:
                    buff_name = match.group(1)
                    duration_str = match.group(2).strip().strip("'\"")

                    # 处理特殊的 buff 名称格式
                    if "(x" in buff_name and ")" in buff_name:
                        # 保持堆叠数量格式，如 "Overwhelming Strikes (x3)"
                        pass

                    # 解析持续时间
                    if duration_str in ["autocast", "permanent"]:
                        duration = float("inf")
                    else:
                        try:
                            duration = int(duration_str)
                        except ValueError:
                            duration = float("inf")

                    # 检查是否即将过期（透明度 < 1）
                    style = img.get("style", "")
                    if "opacity:" in style and "0.485" in style:
                        # 即将过期的 buff，保持解析出的持续时间
                        pass

                    # 映射某些特殊名称
                    if buff_name == "Regeneration":
                        buff_name = "Health Draught"
                    elif buff_name == "Replenishment":
                        buff_name = "Mana Draught"
                    elif buff_name == "Absorbing Ward":
                        buff_name = "Absorb"
                    elif buff_name == "Hastened":
                        buff_name = "Haste"
                    elif buff_name == "Spark of Life" and "scroll" in src.lower():
                        buff_name = "Scroll of Life"

                    self.buffs[buff_name] = duration

    def _parse_skills_and_spells(self):
        """解析角色的技能和法术"""
        # 解析技能 (skills)
        table_skills = self.soup.find("table", id="table_skills")
        if table_skills:
            skill_divs = table_skills.find_all("div", class_="btsd")
            for skill_div in skill_divs:
                self._parse_skill_or_spell(skill_div, is_skill=True)

        # 解析法术 (spells)
        table_magic = self.soup.find("table", id="table_magic")
        if table_magic:
            spell_divs = table_magic.find_all("div", class_="btsd")
            for spell_div in spell_divs:
                self._parse_skill_or_spell(spell_div, is_skill=False)

    def _parse_skill_or_spell(self, element, is_skill: bool):
        """解析单个技能或法术"""
        # 获取技能/法术名称
        name_div = element.find("div", class_="fc2 fal fcb")
        if not name_div or not name_div.find("div"):
            return

        name = name_div.find("div").get_text(strip=True)

        # 检查是否可用
        style = element.get("style", "")
        is_available = "opacity:0.5" not in style
        status = "available" if is_available else "unavailable"

        # 从 onmouseover 属性解析详细信息
        onmouseover = element.get("onmouseover", "")
        cost_type = ""
        cost = 0
        cooldown = 0

        if "battle.set_infopane_spell" in onmouseover:
            # 使用更简单的方法：提取最后三个数字
            numbers = re.findall(r"\b(\d+)\b", onmouseover)
            if len(numbers) >= 3:
                # 取最后三个数字
                param1 = int(numbers[-3])
                param2 = int(numbers[-2])
                param3 = int(numbers[-1])

                if is_skill:
                    # 对于技能，第二个参数是 overcharge 消耗，第三个是冷却
                    cost = param2
                    cooldown = param3
                    if cost > 0:
                        cost_type = "Charge"
                    else:
                        cost_type = ""
                else:
                    # 对于法术，第一个参数是魔法点消耗，第三个是冷却
                    cost = param1
                    cooldown = param3
                    if cost > 0:
                        cost_type = "Magic Point"
                    else:
                        cost_type = ""

        # 存储到相应的字典
        skill_data = SpiritData(
            status=status,
            cost_type=cost_type,
            cost=cost,
            cooldown=cooldown,
        )
        if is_skill:
            self.skillbook.skills[name] = skill_data
        else:
            self.skillbook.spells[name] = skill_data


class Monster:
    """怪物类"""

    def __init__(self):
        self.id: int = 0
        self.name: str = ""
        self.vitals: Dict[str, float] = {}
        self.buffs: Dict[str, int] = {}
        self.is_system: bool = False

    @property
    def is_alive(self) -> bool:
        """检查怪物是否存活"""
        return self.vitals.get("Health", -1) != -1


class MonsterList:
    """怪物列表类，解析并存储所有怪物信息"""

    def __init__(self, page_source: str):
        """
        初始化怪物列表，解析页面源码

        Args:
            page_source: HTML 页面源码
        """
        self.soup = BeautifulSoup(page_source, "html.parser")
        self.monsters: Dict[int, Monster] = {}

        self._parse_monsters()

    def _parse_monsters(self):
        """解析所有怪物信息"""
        pane_monster = self.soup.find("div", id="pane_monster")
        if not pane_monster:
            return

        # 查找所有怪物容器
        monster_divs = pane_monster.find_all("div", id=re.compile(r"mkey_\d+"))

        for monster_div in monster_divs:
            monster = self._parse_single_monster(monster_div)
            if monster:
                self.monsters[monster.id] = monster

    def _parse_single_monster(self, monster_div) -> Optional[Monster]:
        """解析单个怪物信息"""
        monster = Monster()

        # 获取怪物 ID
        mkey_id = monster_div.get("id", "")
        match = re.search(r"mkey_(\d+)", mkey_id)
        if not match:
            return None

        monster.id = int(match.group(1))

        # 检查是否为系统怪物（根据 style 属性判断）
        monster_style = monster_div.get("style", "")
        monster.is_system = bool(monster_style.strip())

        # 获取怪物名称
        name_div = monster_div.find("div", class_="btm3")
        if name_div:
            name_element = name_div.find("div", class_="fc2 fal fcb")
            if name_element and name_element.find("div"):
                monster.name = name_element.find("div").get_text(strip=True)

        # 检查怪物是否死亡（透明度为 0.3）
        style = monster_div.get("style", "")
        is_dead = "opacity:0.3" in style

        # 解析生命值条
        health_bars = monster_div.find_all(
            "img", src=re.compile(r"nbar(green|dead)\.png")
        )
        if is_dead:
            health_value = -1.0  # 死亡怪物設為 -1
        else:
            health_value = 0.0  # 活著的怪物默認為 0

            for bar in health_bars:
                src = bar.get("src", "")
                if "nbardead.png" in src:
                    health_value = -1
                    break
                elif "nbargreen.png" in src:
                    bar_style = bar.get("style", "")
                    width_match = re.search(r"width:(\d+)px", bar_style)
                    if width_match:
                        width = int(width_match.group(1))
                        health_value = width / 120 * 100  # 怪物血条最大宽度为 120
                    break

        monster.vitals["Health"] = health_value

        # 解析法力值条
        mana_bars = monster_div.find_all("img", src=re.compile(r"nbarblue\.png"))
        mana_value: float = -1

        if is_dead:
            mana_value = -1  # 死亡怪物的法力值为 -1
        else:
            for bar in mana_bars:
                if bar.get("alt") == "magic":
                    style = bar.get("style", "")
                    width_match = re.search(r"width:(\d+)px", style)
                    if width_match:
                        width = int(width_match.group(1))
                        mana_value = width / 120 * 100
                    break

            if mana_value == -1:
                mana_value = 0  # 活着的怪物如果没有法力条，设为 0

        monster.vitals["Mana"] = mana_value

        # 解析精神值条
        spirit_bars = monster_div.find_all("img", src=re.compile(r"nbarred\.png"))
        spirit_value: float = -1

        if is_dead:
            spirit_value = -1  # 死亡怪物的精神值为 -1
        else:
            for bar in spirit_bars:
                if bar.get("alt") == "spirit":
                    style = bar.get("style", "")
                    width_match = re.search(r"width:(\d+)px", style)
                    if width_match:
                        width = int(width_match.group(1))
                        spirit_value = width / 120 * 100
                    break

            if spirit_value == -1:
                spirit_value = 0  # 活着的怪物如果没有精神条，设为 0

        monster.vitals["Spirit"] = spirit_value

        # 解析 buff 效果
        buff_container = monster_div.find("div", class_="btm6")
        if buff_container:
            buff_imgs = buff_container.find_all("img")
            for img in buff_imgs:
                onmouseover = img.get("onmouseover", "")
                if "battle.set_infopane_effect" in onmouseover:
                    # 解析 buff 名称和持续时间
                    match = re.search(
                        r"battle\.set_infopane_effect\('([^']+)'.*?,\s*(\d+)\)",
                        onmouseover,
                    )
                    if match:
                        buff_name = match.group(1)
                        duration = int(match.group(2))
                        monster.buffs[buff_name] = duration

        return monster

    def __getitem__(self, monster_id: int) -> Optional[Monster]:
        """通过 ID 获取怪物"""
        return self.monsters.get(monster_id)

    def __iter__(self):
        """迭代所有怪物"""
        return iter(self.monsters.values())

    def keys(self):
        """获取所有怪物 ID"""
        return self.monsters.keys()

    def values(self):
        """获取所有怪物对象"""
        return self.monsters.values()

    def items(self):
        """获取所有 (ID, 怪物) 对"""
        return self.monsters.items()


class BattleDashBoard:
    def __init__(self, driver: HVDriver) -> None:
        self.hvdriver: HVDriver = driver
        self.refresh()

    def refresh(self):
        self.character = Character(self.hvdriver.driver.page_source)
        self.monster_list = MonsterList(self.hvdriver.driver.page_source)
        self.log = self.get_current_log()

    def get_current_log(self) -> str:
        soup = BeautifulSoup(self.hvdriver.driver.page_source, "html.parser")
        textlog_element = soup.find(id="textlog")
        return str(textlog_element) if textlog_element else ""
