"""代理輪換策略"""

from abc import ABC, abstractmethod
from typing import Any

import zendriver as zd

from ..utils import setup_logger
from .factory import create_browser, stop_browser

logger = setup_logger(__name__)


class ProxyRotator(ABC):
    """代理輪換策略抽象介面

    子類實作不同的 IP 輪換方式：
    - DriverRestartRotator: 重啟瀏覽器取得新 proxy（方案 B）
    - 未來可新增 TorCircuitRotator: 透過 Tor NEWNYM 換 circuit（方案 A）
    """

    @abstractmethod
    async def rotate(
        self, current_browser: Any, headless: bool
    ) -> tuple[zd.Browser, zd.Tab]:
        """輪換代理並回傳新的 (browser, page) tuple

        Args:
            current_browser: 目前的 zendriver Browser 實例
            headless: 是否使用無頭模式

        Returns:
            新的 (browser, page) tuple
        """
        pass


class DriverRestartRotator(ProxyRotator):
    """透過重啟瀏覽器來取得新的代理連線"""

    async def rotate(
        self, current_browser: Any, headless: bool
    ) -> tuple[zd.Browser, zd.Tab]:
        logger.warning("Rotating proxy by restarting browser...")
        try:
            await stop_browser(current_browser)
        except Exception:
            logger.debug("Failed to stop current browser (non-fatal)")

        browser, page = await create_browser(headless=headless)
        logger.info("Browser restarted with new proxy connection")
        return browser, page
