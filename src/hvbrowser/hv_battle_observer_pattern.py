import re
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

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

ICON2BUFF: dict[str, list[str]] = {}
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


class Observer(ABC):
    @abstractmethod
    def update(self, soup: BeautifulSoup) -> None:
        """更新物件狀態，就地修改而非建立新物件"""
        pass


class BattleSubject:
    def __init__(self, driver: HVDriver):
        self._observers: list[Observer] = list()
        self._hvdriver = driver

    def attach(self, observer: Observer):
        self._observers.append(observer)

    def detach(self, observer: Observer):
        self._observers.remove(observer)

    def notify(self):
        soup = BeautifulSoup(self._hvdriver.driver.page_source, "html.parser")

        # 使用執行緒池並行更新所有觀察者
        with ThreadPoolExecutor(max_workers=min(len(self._observers), 4)) as executor:
            # 提交所有觀察者更新任務
            futures = [
                executor.submit(observer.update, soup) for observer in self._observers
            ]

            # 等待所有任務完成
            for future in futures:
                future.result()


@dataclass
class CharacterVitals(Observer):
    hp: float = 0.0
    mp: float = 0.0
    sp: float = 0.0
    overcharge: float = 0.0

    def update(self, soup: BeautifulSoup) -> None:
        """就地更新角色生命值等資訊"""
        new_data = parse_vitals(soup)
        self.hp = new_data.hp
        self.mp = new_data.mp
        self.sp = new_data.sp
        self.overcharge = new_data.overcharge


@dataclass
class SpiritData:
    name: str = ""
    status: str = "unavailable"
    cost_type: str = ""
    cost: int = 0
    cooldown: int = 0


@dataclass
class CharacterBuffs(Observer):
    buffs: dict[str, float] = field(default_factory=dict)

    def update(self, soup: BeautifulSoup) -> None:
        """就地更新角色 buff 資訊"""
        new_data = parse_buffs(soup)
        self.buffs.clear()
        self.buffs.update(new_data.buffs)


@dataclass
class CharacterSkillBook(Observer):
    skills: dict[str, SpiritData] = field(default_factory=dict)
    spells: dict[str, SpiritData] = field(default_factory=dict)

    @property
    def skills_and_spells(self) -> dict[str, SpiritData]:
        return {**self.skills, **self.spells}

    def update(self, soup: BeautifulSoup) -> None:
        """就地更新技能與法術資訊"""
        # 使用執行緒池並行解析技能和法術
        with ThreadPoolExecutor(max_workers=2) as executor:
            skills_future = executor.submit(parse_skills, soup)
            spells_future = executor.submit(parse_spells, soup)

            skills_data = skills_future.result()
            spells_data = spells_future.result()

        self.skills.clear()
        self.skills.update(skills_data.skills)

        self.spells.clear()
        self.spells.update(spells_data.spells)

    def __or__(self, other: "CharacterSkillBook") -> "CharacterSkillBook":
        combined = CharacterSkillBook()
        combined.skills = {**self.skills, **other.skills}
        combined.spells = {**self.spells, **other.spells}
        return combined


def parse_vitals(soup: BeautifulSoup) -> CharacterVitals:
    """解析角色的生命值、法力值等資訊"""
    pane_vitals = soup.find("div", id="pane_vitals")
    if not pane_vitals or not hasattr(pane_vitals, "find"):
        return CharacterVitals()

    hp = 0.0
    mp = 0.0
    sp = 0.0
    overcharge = 0.0

    # 查找生命值條 (bar_bgreen.png 或 bar_dgreen.png)
    health_bar = pane_vitals.find("img", src=re.compile(r"bar_[bd]green\.png"))
    if health_bar:
        style = health_bar.get("style", "")
        width_match = re.search(r"width:(\d+)px", style)
        if width_match:
            width = int(width_match.group(1))
            hp = width / 414 * 100

    # 查找法力值條 (bar_blue.png)
    mana_bar = pane_vitals.find("img", src=re.compile(r"bar_blue\.png"))
    if mana_bar:
        style = mana_bar.get("style", "")
        width_match = re.search(r"width:(\d+)px", style)
        if width_match:
            width = int(width_match.group(1))
            mp = width / 414 * 100

    # 查找精神值條 (bar_red.png)
    spirit_bar = pane_vitals.find("img", src=re.compile(r"bar_red\.png"))
    if spirit_bar:
        style = spirit_bar.get("style", "")
        width_match = re.search(r"width:(\d+)px", style)
        if width_match:
            width = int(width_match.group(1))
            sp = width / 414 * 100

    # 查找過充值條 (bar_orange.png)
    overcharge_bar = pane_vitals.find("img", src=re.compile(r"bar_orange\.png"))
    if overcharge_bar:
        style = overcharge_bar.get("style", "")
        width_match = re.search(r"width:(\d+)px", style)
        if width_match:
            width = int(width_match.group(1))
            overcharge = width / 414 * 250

    return CharacterVitals(
        hp=hp,
        mp=mp,
        sp=sp,
        overcharge=overcharge,
    )


def parse_skills(soup: BeautifulSoup) -> CharacterSkillBook:
    skill_book = CharacterSkillBook()
    table_skills = soup.find("table", id="table_skills")
    if table_skills and hasattr(table_skills, "find_all"):
        skill_divs = table_skills.find_all("div", class_="btsd")
        for skill_div in skill_divs:
            skill = parse_skill_or_spell(skill_div, is_skill=True)
            skill_book.skills[skill.name] = skill
    return skill_book


def parse_spells(soup: BeautifulSoup) -> CharacterSkillBook:
    skill_book = CharacterSkillBook()
    table_magic = soup.find("table", id="table_magic")
    if table_magic and hasattr(table_magic, "find_all"):
        spell_divs = table_magic.find_all("div", class_="btsd")
        for spell_div in spell_divs:
            spell = parse_skill_or_spell(spell_div, is_skill=False)
            skill_book.spells[spell.name] = spell
    return skill_book


def parse_skill_or_spell(element, is_skill: bool) -> SpiritData:
    """解析單個技能或法術"""
    # 獲取技能/法術名稱
    name_div = element.find("div", class_="fc2 fal fcb")
    if not name_div or not name_div.find("div"):
        return SpiritData()

    name = name_div.find("div").get_text(strip=True)

    # 檢查是否可用
    style = element.get("style", "")
    is_available = "opacity:0.5" not in style
    status = "available" if is_available else "unavailable"

    # 從 onmouseover 屬性解析詳細資訊
    onmouseover = element.get("onmouseover", "")
    cost_type = ""
    cost = 0
    cooldown = 0

    if "battle.set_infopane_spell" in onmouseover:
        # 使用更簡單的方法：提取最後三個數字
        numbers = re.findall(r"\b(\d+)\b", onmouseover)
        if len(numbers) >= 3:
            # 取最後三個數字
            param1 = int(numbers[-3])
            param2 = int(numbers[-2])
            param3 = int(numbers[-1])

            if is_skill:
                # 對於技能，第二個參數是 overcharge 消耗，第三個是冷卻
                cost = param2
                cooldown = param3
                if cost > 0:
                    cost_type = "Charge"
                else:
                    cost_type = ""
            else:
                # 對於法術，第一個參數是魔法點消耗，第三個是冷卻
                cost = param1
                cooldown = param3
                if cost > 0:
                    cost_type = "Magic Point"
                else:
                    cost_type = ""

    return SpiritData(
        name=name,
        status=status,
        cost_type=cost_type,
        cost=cost,
        cooldown=cooldown,
    )


def parse_buffs(soup: BeautifulSoup) -> CharacterBuffs:
    """解析角色的 buff 效果"""
    # <img id="ckey_spirit" onclick="battle.lock_action(this,0,'spirit')" onmouseover="battle.set_infopane('Spirit')" src="/isekai/y/battle/spirit_n.png">

    character_buffs = CharacterBuffs()

    spirit_img = soup.find("img", id="ckey_spirit")
    if (
        spirit_img
        and hasattr(spirit_img, "get")
        and "spirit_a.png" in spirit_img.get("src", "")
    ):
        character_buffs.buffs["Spirit Stance"] = float("inf")

    pane_effects = soup.find("div", id="pane_effects")
    if not pane_effects or not hasattr(pane_effects, "find_all"):
        return character_buffs

    # 查找所有 buff 圖標
    buff_imgs = pane_effects.find_all("img")

    for img in buff_imgs:
        src = img.get("src", "")

        # 獲取 buff 名稱和持續時間
        onmouseover = img.get("onmouseover", "")

        # 從 onmouseover 屬性中提取 buff 資訊
        if "battle.set_infopane_effect" in onmouseover:
            # 解析格式：battle.set_infopane_effect('Buff Name', '...', duration)
            # 使用更精確的正則表達式
            match = re.search(
                r"battle\.set_infopane_effect\('([^']+)',\s*'[^']*',\s*([^)]+)\)",
                onmouseover,
            )
            if match:
                buff_name = match.group(1)
                duration_str = match.group(2).strip().strip("'\"")

                # 處理特殊的 buff 名稱格式
                if "(x" in buff_name and ")" in buff_name:
                    # 保持堆疊數量格式，如 "Overwhelming Strikes (x3)"
                    pass

                # 解析持續時間
                if duration_str in ["autocast", "permanent"]:
                    duration = float("inf")
                else:
                    try:
                        duration = int(duration_str)
                    except ValueError:
                        duration = float("inf")

                # 檢查是否即將過期（透明度 < 1）
                style = img.get("style", "")
                if "opacity:" in style and "0.485" in style:
                    # 即將過期的 buff，保持解析出的持續時間
                    pass

                # 映射某些特殊名稱
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

                character_buffs.buffs[buff_name] = duration

    return character_buffs


@dataclass
class MonsterVitals:
    health: float = 0
    mana: float = 0
    spirit: float = 0


@dataclass
class MonsterBuffs:
    buffs: dict[str, float] = field(default_factory=dict)


@dataclass
class Monster:
    id: int = 0
    name: str = ""
    vitals: MonsterVitals = field(default_factory=MonsterVitals)
    buffs: MonsterBuffs = field(default_factory=MonsterBuffs)
    is_system: bool = False

    @property
    def is_alive(self) -> bool:
        return self.vitals.health != -1


@dataclass
class MonsterList(Observer):
    monsters: dict[int, Monster] = field(default_factory=dict)

    def update(self, soup: BeautifulSoup) -> None:
        """就地更新怪物列表資訊"""
        new_data = parse_monsters(soup)
        self.monsters.clear()
        self.monsters.update(new_data.monsters)

    def __getitem__(self, monster_id: int) -> Optional[Monster]:
        """通過 ID 獲取怪物"""
        return self.monsters.get(monster_id)

    def __iter__(self):
        """迭代所有怪物"""
        return iter(self.monsters.values())

    def keys(self):
        """獲取所有怪物 ID"""
        return self.monsters.keys()

    def values(self):
        """獲取所有怪物對象"""
        return self.monsters.values()

    def items(self):
        """獲取所有 (ID, 怪物) 對"""
        return self.monsters.items()


def parse_monsters(soup: BeautifulSoup) -> MonsterList:
    def parse_single_monster(monster_div) -> Monster:
        monster = Monster()

        # 獲取怪物 ID
        mkey_id = monster_div.get("id", "")
        match = re.search(r"mkey_(\d+)", mkey_id)
        if not match:
            return monster

        monster.id = int(match.group(1))

        # 檢查是否為系統怪物（根據 style 屬性判斷）
        monster_style = monster_div.get("style", "")
        monster.is_system = bool(monster_style.strip())

        # 獲取怪物名稱
        name_div = monster_div.find("div", class_="btm3")
        if name_div:
            name_element = name_div.find("div", class_="fc2 fal fcb")
            if name_element and name_element.find("div"):
                monster.name = name_element.find("div").get_text(strip=True)

        # 檢查怪物是否死亡（透明度為 0.3）
        style = monster_div.get("style", "")
        is_dead = "opacity:0.3" in style

        # 解析生命值條
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
                        health_value = width / 120 * 100  # 怪物血條最大寬度為 120
                    break

        # 解析法力值條
        mana_bars = monster_div.find_all("img", src=re.compile(r"nbarblue\.png"))
        mana_value: float = -1

        if is_dead:
            mana_value = -1  # 死亡怪物的法力值為 -1
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
                mana_value = 0  # 活著的怪物如果沒有法力條，設為 0

        # 解析精神值條
        spirit_bars = monster_div.find_all("img", src=re.compile(r"nbarred\.png"))
        spirit_value: float = -1

        if is_dead:
            spirit_value = -1  # 死亡怪物的精神值為 -1
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
                spirit_value = 0  # 活著的怪物如果沒有精神條，設為 0

        monster.vitals = MonsterVitals(
            health=health_value, mana=mana_value, spirit=spirit_value
        )

        # 解析 buff 效果
        buff_container = monster_div.find("div", class_="btm6")
        if buff_container:
            buff_imgs = buff_container.find_all("img")
            for img in buff_imgs:
                onmouseover = img.get("onmouseover", "")
                if "battle.set_infopane_effect" in onmouseover:
                    # 解析 buff 名稱和持續時間
                    match = re.search(
                        r"battle\.set_infopane_effect\('([^']+)'.*?,\s*(\d+)\)",
                        onmouseover,
                    )
                    if match:
                        buff_name = match.group(1)
                        duration = int(match.group(2))
                        monster.buffs.buffs[buff_name] = duration

        return monster

    """解析所有怪物資訊"""
    monster_list = MonsterList()
    pane_monster = soup.find("div", id="pane_monster")
    if not pane_monster or not hasattr(pane_monster, "find_all"):
        return monster_list

    # 查找所有怪物容器
    monster_divs = pane_monster.find_all("div", id=re.compile(r"mkey_\d+"))

    # 使用執行緒池並行解析所有怪物
    with ThreadPoolExecutor(max_workers=min(len(monster_divs), 8)) as executor:
        # 提交所有解析任務
        futures = [
            executor.submit(parse_single_monster, monster_div)
            for monster_div in monster_divs
        ]

        # 收集結果
        for future in futures:
            monster = future.result()
            if monster and monster.id >= 0:  # 確保解析成功且有有效ID（包含ID為0的怪物）
                monster_list.monsters[monster.id] = monster

    return monster_list


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

    def get_new_lines(self, soup: BeautifulSoup) -> list[str]:
        textlog = soup.find(id="textlog")
        if textlog and hasattr(textlog, "find_all"):
            lines = [
                td.text.strip() for td in textlog.find_all("td") if td.text.strip()
            ][-1::-1]
            return lines
        else:
            return []

    def update(self, soup: BeautifulSoup):
        lines = self.get_new_lines(soup)
        if lines:
            self.current_lines = [line for line in lines if line not in self.prev_lines]
            self._parse_round_info(self.current_lines)
            self.prev_lines.extend(self.current_lines)


class BattleDashboard:
    def __init__(self, driver: HVDriver):
        self._hvdriver = driver
        self.battle_subject = BattleSubject(driver)
        self.character_vitals = CharacterVitals()
        self.character_buffs = CharacterBuffs()
        self.character_skillbook = CharacterSkillBook()
        self.monster_list = MonsterList()
        self.log_entries: LogEntry = LogEntry()
        self.battle_subject.attach(self.character_vitals)
        self.battle_subject.attach(self.character_buffs)
        self.battle_subject.attach(self.character_skillbook)
        self.battle_subject.attach(self.monster_list)
        self.battle_subject.attach(self.log_entries)
        self.update()

    def update(self):
        self.battle_subject.notify()
