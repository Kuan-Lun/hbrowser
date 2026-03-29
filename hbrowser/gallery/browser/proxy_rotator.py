"""代理輪換策略"""

from abc import ABC, abstractmethod
from typing import Any

from ..utils import setup_logger
from .factory import create_driver

logger = setup_logger(__name__)


class ProxyRotator(ABC):
    """代理輪換策略抽象介面

    子類實作不同的 IP 輪換方式：
    - DriverRestartRotator: 重啟瀏覽器取得新 proxy（方案 B）
    - 未來可新增 TorCircuitRotator: 透過 Tor NEWNYM 換 circuit（方案 A）
    """

    @abstractmethod
    def rotate(self, current_driver: Any, headless: bool) -> Any:
        """輪換代理並回傳新的（或同一個）driver

        Args:
            current_driver: 目前的 WebDriver 實例
            headless: 是否使用無頭模式

        Returns:
            新的或重置過的 WebDriver 實例
        """
        pass


class DriverRestartRotator(ProxyRotator):
    """透過重啟瀏覽器來取得新的代理連線"""

    def rotate(self, current_driver: Any, headless: bool) -> Any:
        logger.warning("Rotating proxy by restarting browser...")
        try:
            current_driver.quit()
        except Exception:
            logger.debug("Failed to quit current driver (non-fatal)")

        new_driver = create_driver(headless=headless)
        logger.info("Browser restarted with new proxy connection")
        return new_driver
