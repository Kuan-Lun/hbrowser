import asyncio
from collections import defaultdict
from collections.abc import Callable
from functools import partial, wraps
from random import random
from typing import Any, TypeVar

from ponychart_classifier import update as update_ponychart_classifier
from zendriver import cdp

from hbrowser.gallery.utils import setup_logger
from hbrowser.notify import notify

from .control_panel import ControlPanel
from .hv import HVDriver
from .hv_battle_action_manager import ElementActionManager
from .hv_battle_buff_manager import BuffManager
from .hv_battle_defaults import (
    DEFAULT_FORBIDDEN_SKILLS,
    DEFAULT_STATTHRESHOLD,
    StatThreshold,
)
from .hv_battle_item_provider import ItemProvider
from .hv_battle_observer_pattern import BattleDashboard
from .hv_battle_ponychart import PonyChart
from .hv_battle_skill_manager import SkillManager

logger = setup_logger(__name__)

_F = TypeVar("_F", bound=Callable[..., Any])

MONSTER_DEBUFF_TO_CHARACTER_SKILL = {
    "imperiled": "imperil",
    "weakened": "weaken",
    "slowed": "slow",
    "asleep": "sleep",
    "confused": "confuse",
    "magically snared": "magnet",
    "blinded": "blind",
    "vital theft": "drain",
    "silenced": "silence",
}


def update_ponychart_on(expected: bool) -> Callable[[_F], _F]:
    """當被修飾的函數回傳值等於 expected 時，呼叫 update_ponychart_classifier()"""

    def decorator(func: _F) -> _F:
        @wraps(func)
        async def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
            result = await func(self, *args, **kwargs)
            if result is expected:
                update_ponychart_classifier()
            return result

        return wrapper  # type: ignore[return-value]

    return decorator


def retry_on_server_fail(func: _F) -> _F:
    """在出現 Server communication failed alert ��，自動刷新頁面並重試一次"""

    @wraps(func)
    async def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        try:
            return await func(self, *args, **kwargs)
        except Exception as e:
            # 檢查是否為 alert 相關錯誤
            if "alert" in str(e).lower() or "dialog" in str(e).lower():
                try:
                    # 嘗試接受 alert via CDP
                    await self.hvdriver.page.send(
                        cdp.page.handle_javascript_dialog(accept=True)
                    )
                    logger.warning(
                        "Server communication failed detected, "
                        "retrying after refresh..."
                    )
                    await asyncio.sleep(5)
                    await self.hvdriver.page.reload()
                    return await func(self, *args, **kwargs)
                except Exception as inner_e:
                    logger.error(f"Failed to handle alert or refresh: {inner_e}")
            raise

    return wrapper  # type: ignore[return-value]


class BattleDriver(HVDriver):
    def __init__(
        self,
        *args: Any,
        statthreshold: StatThreshold | None = None,
        forbidden_skills: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.statthreshold = statthreshold or DEFAULT_STATTHRESHOLD
        self.battle_dashboard: BattleDashboard = None  # type: ignore[assignment]
        self.element_action_manager: ElementActionManager = None  # type: ignore[assignment]

        self.with_ofc: bool = True
        self._itemprovider: ItemProvider = None  # type: ignore[assignment]
        self._skillmanager: SkillManager = None  # type: ignore[assignment]
        self._buffmanager: BuffManager = None  # type: ignore[assignment]
        self.control_panel = ControlPanel()
        self.control_panel.register_toggle("auto_next_battle")

        forbidden_lower = [
            s.lower() for s in (forbidden_skills or DEFAULT_FORBIDDEN_SKILLS)
        ]
        skill_groups = self._build_skill_groups()
        extra_buff_skills = sorted(
            s
            for s in forbidden_lower
            if s not in skill_groups["Debuff Skills"]
            and s not in skill_groups["Buff Skills"]
        )
        if extra_buff_skills:
            skill_groups["Buff Skills"] = sorted(
                set(skill_groups["Buff Skills"]) | set(extra_buff_skills)
            )
        self.control_panel.set_skills(skill_groups, forbidden_lower)

        self.turn = -1
        self.round = -1
        self.pround = -1

        update_ponychart_classifier()

    async def _init_browser(self) -> None:
        """Override: 初始化瀏覽器後也初始化戰鬥組件"""
        await super()._init_browser()
        await self._init_battle_components()

    async def _init_battle_components(self) -> None:
        """初始化戰鬥相關組件（需要 page 已就緒）"""
        self.battle_dashboard = BattleDashboard(self)
        await self.battle_dashboard.init()
        self.element_action_manager = ElementActionManager(self, self.battle_dashboard)

        self.with_ofc = not await self.is_isekai
        self._itemprovider = ItemProvider(self, self.battle_dashboard)
        self._skillmanager = SkillManager(self, self.battle_dashboard)
        self._buffmanager = BuffManager(self, self.battle_dashboard)

    async def _setup_alert_handler(self) -> None:
        """設置 JavaScript dialog 自動處理"""

        async def dialog_handler(
            event: cdp.page.JavascriptDialogOpening,
        ) -> None:
            await self.page.send(cdp.page.handle_javascript_dialog(accept=True))

        self.page.add_handler(cdp.page.JavascriptDialogOpening, dialog_handler)

    @property
    def auto_next_battle(self) -> bool:
        return self.control_panel.get_toggle("auto_next_battle")

    async def clear_cache(self) -> None:
        # 重新解析戰鬥儀表板以獲取最新的怪物狀態
        self.round = self.battle_dashboard.log_entries.current_round
        await self.battle_dashboard.update()

    def reset_pround(self) -> None:
        self.pround = self.round

    def _build_skill_groups(self) -> dict[str, list[str]]:
        debuff_skills = sorted(MONSTER_DEBUFF_TO_CHARACTER_SKILL.values())
        buff_skills = sorted(
            {
                "health draught",
                "mana draught",
                "spirit draught",
                "regen",
                "scroll of life",
                "scroll of absorption",
                "absorb",
                "scroll of protection",
                "heartseeker",
            }
        )
        return {"Debuff Skills": debuff_skills, "Buff Skills": buff_skills}

    @property
    def forbidden_skills(self) -> list[str]:
        return self.control_panel.get_forbidden_skills()

    async def click_skill(self, key: str, iswait: bool = True) -> bool:
        if key in self.forbidden_skills:
            return False
        result = await self._skillmanager.cast(key, iswait=iswait)
        return result

    def get_stat_percent(self, stat: str) -> float:
        match stat.lower():
            case "hp":
                value = self.battle_dashboard.snap.player.hp_percent
            case "mp":
                value = self.battle_dashboard.snap.player.mp_percent
            case "sp":
                value = self.battle_dashboard.snap.player.sp_percent
            case "overcharge":
                value = self.battle_dashboard.snap.player.overcharge_value
            case _:
                raise ValueError(f"Unknown stat: {stat}")
        return float(value)

    @property
    def new_logs(self) -> list[str]:
        new_logs = self.battle_dashboard.log_entries.current_lines
        # 固定寬度，假設最大 3 位數
        turn_str = f"Turn {self.turn:>5}"
        current = self.battle_dashboard.log_entries.current_round
        total = self.battle_dashboard.log_entries.total_round
        round_str = f"Round {current:>3} / {total:<3}"
        return [f"{turn_str} {round_str} {line}" for line in new_logs]

    async def use_item(self, key: str) -> bool:
        return await self._itemprovider.use(key)

    async def apply_buff(self, key: str, force: bool = False) -> bool:
        if key in self.forbidden_skills:
            return False
        apply_buff = partial(self._buffmanager.apply_buff, key=key, force=force)
        if not force:
            match key:
                case "health draught":
                    if self.get_stat_percent("hp") < 90:
                        return await apply_buff()
                    else:
                        return False
                case "mana draught":
                    if self.get_stat_percent("mp") < 90:
                        return await apply_buff()
                    else:
                        return False
                case "spirit draught":
                    if self.get_stat_percent("sp") < 90:
                        return await apply_buff()
                    else:
                        return False
        return await apply_buff()

    async def check_hp(self) -> bool:
        if self.get_stat_percent("hp") < self.statthreshold.hp_low:
            for fun in [
                partial(self.use_item, "health gem"),
                partial(self.click_skill, "full-cure"),
                partial(self.use_item, "health potion"),
                partial(self.use_item, "health elixir"),
                partial(self.use_item, "last elixir"),
                partial(self.click_skill, "cure"),
            ]:
                if await fun():
                    return True

        if self.get_stat_percent("hp") < self.statthreshold.hp_high:
            for fun in [
                partial(self.use_item, "health gem"),
                partial(self.click_skill, "cure"),
                partial(self.use_item, "health potion"),
            ]:
                if await fun():
                    return True

        return False

    async def check_mp(self) -> bool:
        if self.get_stat_percent("mp") < self.statthreshold.mp_low:
            for fun in [
                partial(self.use_item, "mana gem"),
                partial(self.use_item, "mana potion"),
                partial(self.use_item, "mana elixir"),
                partial(self.use_item, "last elixir"),
            ]:
                if await fun():
                    return True

        if self.get_stat_percent("mp") < self.statthreshold.mp_high:
            for fun in [
                partial(self.use_item, "mana gem"),
                partial(self.use_item, "mana potion"),
            ]:
                if await fun():
                    return True

        return False

    async def check_sp(self) -> bool:
        if self.get_stat_percent("sp") < self.statthreshold.sp_low:
            for fun in [
                partial(self.use_item, "spirit gem"),
                partial(self.use_item, "spirit potion"),
                partial(self.use_item, "spirit elixir"),
                partial(self.use_item, "last elixir"),
            ]:
                if await fun():
                    return True

        if self.get_stat_percent("sp") < self.statthreshold.sp_high:
            for fun in [
                partial(self.use_item, "spirit gem"),
                partial(self.use_item, "spirit potion"),
            ]:
                if await fun():
                    return True

        return False

    async def check_overcharge(self) -> bool:
        if self._buffmanager.has_buff("spirit stance"):
            if any(
                [
                    self.get_stat_percent("overcharge")
                    < self.statthreshold.overcharge_low,
                    self.get_stat_percent("sp") < self.statthreshold.sp_low,
                ]
            ):
                return await self.apply_buff("spirit stance", force=True)

        if all(
            [
                self.get_stat_percent("overcharge")
                > self.statthreshold.overcharge_high,
                self.get_stat_percent("sp") > self.statthreshold.sp_low,
                not self._buffmanager.has_buff("spirit stance"),
            ]
        ):
            return await self.apply_buff("spirit stance")
        return False

    @update_ponychart_on(True)
    async def go_next_floor(self) -> bool:
        elements = await self.page.query_selector_all("#btcp")
        if elements:
            await self.element_action_manager.click_and_wait_log_locator("#btcp")
            self._create_last_debuff_monster_id()
            return True
        return False

    @update_ponychart_on(True)
    async def go_next_battle(self) -> bool:
        path_prefix = await self._get_path_prefix()
        arena_url = f'{self.url["HentaiVerse"]}{path_prefix}/?s=Battle&ss=ar'
        current_url = await self.page.evaluate("window.location.href")
        if current_url != arena_url:
            return False
        elements = await self.page.select_all(
            f'img[src="{path_prefix}/y/arena/startchallenge.png"]', timeout=2
        )
        if elements:
            # 先覆寫 window.confirm 讓它自動回傳 true，避免確認對話框阻塞
            await self.page.evaluate("window.confirm = function() { return true; };")
            await elements[-1].click()
            return True
        return False

    async def debuff_monster(self, debuff: str, nums: list[int]) -> bool:
        debuff_skill = MONSTER_DEBUFF_TO_CHARACTER_SKILL[debuff]
        if debuff_skill in self.forbidden_skills:
            return False

        monster_ids_with_debuff = (
            self.battle_dashboard.overview_monsters.alive_monster_with_buff.get(
                debuff, []
            )
        ) + [self.last_debuff_monster_id[debuff]]
        for num in nums:
            if num not in monster_ids_with_debuff:
                await self.attack_monster_by_skill(
                    num, MONSTER_DEBUFF_TO_CHARACTER_SKILL[debuff]
                )
                self.last_debuff_monster_id[debuff] = num
                return True
        return False

    async def attack_monster(self, n: int) -> bool:
        selector = f'[id="mkey_{n}"]'
        elements = await self.page.query_selector_all(selector)
        if not elements:
            return False
        await self.element_action_manager.click_and_wait_log_locator(selector)
        return True

    async def attack_monster_by_skill(self, n: int, skill_name: str) -> bool:
        await self.click_skill(skill_name, iswait=False)
        return await self.attack_monster(n)

    async def attack(self) -> bool:
        base_monster_ids: list[int] = [1, 3, 5, 7, 9, 2, 4, 6, 8, 0]

        def monster_ids_starting_with(ids: list[int], n: int) -> list[int]:
            return ids[ids.index(n) :] + ids[: ids.index(n)]

        def resort_monster_alive_ids(bmlist: list[int]) -> list[int]:
            monster_alive_ids: list[int] = [
                id
                for id in bmlist
                if id in self.battle_dashboard.overview_monsters.alive_monster
            ]
            if len(self.battle_dashboard.overview_monsters.alive_monster):
                monster_alive_ids = monster_ids_starting_with(
                    monster_alive_ids,
                    self.battle_dashboard.overview_monsters.alive_monster[0],
                )
            for monster_name in ["Yggdrasil", "Skuld", "Urd", "Verdandi"][::-1]:
                if (
                    monster_name
                    not in self.battle_dashboard.overview_monsters.alive_monster_name
                ):
                    continue
                monster_id = self.battle_dashboard.overview_monsters.alive_monster_name[
                    monster_name
                ]
                if monster_id in monster_alive_ids:
                    monster_alive_ids = monster_ids_starting_with(
                        monster_alive_ids, monster_id
                    )
            return monster_alive_ids

        # Check if Orbital Friendship Cannon can be used
        if (
            self.with_ofc
            and self.get_stat_percent("overcharge") > 220
            and self._buffmanager.has_buff("spirit stance")
            and len(self.battle_dashboard.overview_monsters.alive_monster)
            >= self.statthreshold.countmonster_high
            and "Orbital Friendship Cannon"
            in self.battle_dashboard.snap.abilities.skills
            and self.battle_dashboard.snap.abilities.skills[
                "Orbital Friendship Cannon"
            ].available
        ):
            await self.attack_monster_by_skill(
                self.battle_dashboard.overview_monsters.alive_monster[0],
                "Orbital Friendship Cannon",
            )
            return True

        monster_alive_ids: list[int] = resort_monster_alive_ids(base_monster_ids)

        if (
            len(monster_alive_ids) > 3
            and self.get_stat_percent("mp") > self.statthreshold.mp_high
        ):
            for debuff in MONSTER_DEBUFF_TO_CHARACTER_SKILL:
                if debuff in ["imperiled"]:
                    continue
                debuffed_monsters = (
                    self.battle_dashboard.overview_monsters.alive_monster_with_buff.get(
                        debuff, []
                    )
                )
                if len(monster_alive_ids) - len(debuffed_monsters) < 3:
                    continue
                if await self.debuff_monster(debuff, monster_alive_ids):
                    return True

        monster_with_imperil: list[int]
        if (
            "imperil" not in self.forbidden_skills
            and self.get_stat_percent("mp") > self.statthreshold.mp_high
        ):
            monster_with_imperil = (
                self.battle_dashboard.overview_monsters.alive_monster_with_buff.get(
                    "imperiled", []
                )
            )
        else:
            monster_with_imperil = monster_alive_ids

        if monster_alive_ids:
            n = monster_alive_ids[0]
            if n in monster_with_imperil:
                if self.get_stat_percent(
                    "overcharge"
                ) > 200 and self._buffmanager.has_buff("spirit stance"):
                    monster_health = self.battle_dashboard.snap.monsters[n].hp_percent
                    if (
                        monster_health < 25
                        and "merciful blow"
                        in self.battle_dashboard.snap.abilities.skills
                        and self.battle_dashboard.snap.abilities.skills[
                            "merciful blow"
                        ].available
                    ):
                        await self.attack_monster_by_skill(n, "merciful blow")
                    elif (
                        monster_health > 5
                        and "vital strike"
                        in self.battle_dashboard.snap.abilities.skills
                        and self.battle_dashboard.snap.abilities.skills[
                            "vital strike"
                        ].available
                    ):
                        await self.attack_monster_by_skill(n, "vital strike")
                    else:
                        await self.attack_monster(n)
                else:
                    await self.attack_monster(n)
                self.last_debuff_monster_id["imperiled"] = -1
            else:
                if n == self.last_debuff_monster_id["imperiled"]:
                    if random() < 0.5:
                        await self.attack_monster_by_skill(n, "imperil")
                    else:
                        await self.attack_monster(n)
                else:
                    await self.attack_monster_by_skill(
                        n, MONSTER_DEBUFF_TO_CHARACTER_SKILL["imperiled"]
                    )
                    self.last_debuff_monster_id["imperiled"] = n
            return True
        else:
            return False

    async def use_channeling(self) -> bool:
        if "channeling" in self.battle_dashboard.snap.player.buffs:
            skill_names = ["regen", "heartseeker"]
            skill2remaining: dict[str, float] = dict()
            for skill_name in skill_names:
                remaining_turns = self._buffmanager.get_buff_remaining_turns(skill_name)
                refresh_turns = self._buffmanager.skill2turn[skill_name]
                skill_cost = self._skillmanager.get_max_skill_mp_cost_by_name(
                    skill_name
                )
                skill2remaining[skill_name] = (
                    (refresh_turns - remaining_turns) * refresh_turns / skill_cost
                )
            if max(skill2remaining.values()) < 0:
                return False

            to_use_skill_name = max(skill2remaining, key=lambda k: skill2remaining[k])

            await self.apply_buff(to_use_skill_name, force=True)
            return True

        return False

    @retry_on_server_fail
    async def battle_in_turn(self) -> bool:
        if self.turn == -1:
            is_isekai = await self.is_isekai
            mode = "Isekai" if is_isekai else "Persistent"
            self.control_panel.set_title(f"Battle Control Panel ({mode})")
        self.turn += 1
        await self.clear_cache()
        # Log the current round logs
        if self.new_logs:
            for log_line in self.new_logs:
                logger.info(log_line)

        for fun in [
            *([] if not self.auto_next_battle else [self.go_next_battle]),
            self.go_next_floor,
            PonyChart(self).check,
            self.check_hp,
            self.check_mp,
            self.check_sp,
            self.check_overcharge,
            lambda: self.apply_buff("health draught"),
            lambda: self.apply_buff("mana draught"),
            lambda: self.apply_buff("spirit draught"),
            lambda: self.apply_buff("regen"),
            lambda: self.apply_buff("scroll of life"),
            lambda: self.apply_buff("scroll of absorption"),
            lambda: self.apply_buff("absorb"),
            lambda: self.apply_buff("scroll of protection"),
            lambda: self.apply_buff("heartseeker"),
            self.use_channeling,
            self.attack,
        ]:
            if await fun():
                return True

        return False

    def _create_last_debuff_monster_id(self) -> None:
        self.last_debuff_monster_id: dict[str, int] = defaultdict(lambda: -1)

    async def _is_in_battle(self) -> bool:
        try:
            await self.battle_dashboard.update()
            return (
                bool(self.battle_dashboard.overview_monsters.alive_monster_name)
                or await PonyChart(self).check()
            )
        except Exception:
            logger.info("Alert or error detected, attempting to handle it.")
            try:
                await self.page.send(cdp.page.handle_javascript_dialog(accept=True))
            except Exception:
                logger.debug("No dialog to accept or already dismissed.")
            return False

    def _wait_if_paused(self) -> None:
        self.control_panel.wait_if_paused()

    async def _wait_for_page_recovery(
        self, timeout: int = 300, poll_interval: int = 5, log_interval: int = 30
    ) -> bool:
        """Poll the page periodically to detect early recovery."""
        for i in range(timeout // poll_interval):
            await asyncio.sleep(poll_interval)
            elapsed = (i + 1) * poll_interval
            if elapsed % log_interval == 0:
                logger.info(f"Waiting for recovery... ({elapsed}/{timeout}s)")
            try:
                current_url = await self.page.evaluate("window.location.href")
                await self.page.get(current_url)
                if await self._is_in_battle():
                    logger.info(f"Page recovered after {elapsed}s")
                    return True
            except TimeoutError:
                pass
        logger.warning(f"Page did not recover within {timeout}s")
        return False

    async def _wait_for_battle(self, timeout: int = 300, interval: int = 1) -> bool:
        if await self._is_in_battle():
            return True
        logger.info(f"Waiting up to {timeout}s for user to start a battle...")
        for _ in range(timeout // interval):
            await asyncio.sleep(interval)
            self._wait_if_paused()
            if await self._is_in_battle():
                return True
        return False

    async def battle(self) -> None:
        if not await self._wait_for_battle():
            logger.info("No battle detected after waiting, exiting.")
            return

        self._create_last_debuff_monster_id()

        max_retries = 3
        retry_count = 0
        while True:
            self._wait_if_paused()
            try:
                if not await self.battle_in_turn():
                    break
                retry_count = 0
            except TimeoutError:
                retry_count += 1
                if retry_count >= max_retries:
                    logger.error(
                        "TimeoutError caught, max retries reached "
                        f"({max_retries}/{max_retries})"
                    )
                    raise
                logger.warning(
                    "TimeoutError caught, reloading page "
                    f"(attempt {retry_count}/{max_retries})"
                )
                if await self._wait_for_page_recovery():
                    retry_count = 0

        notify("HBrowser", "Battle complete")
        logger.info("Battle complete, waiting 300s for user to start next battle...")

        await self.battle()
