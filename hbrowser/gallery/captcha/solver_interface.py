"""驗證碼解決器抽象接口"""

from abc import ABC, abstractmethod
from typing import Any

from .models import ChallengeDetection, SolveResult


class CaptchaSolver(ABC):
    """
    驗證碼解決器抽象接口

    擴展套件可以實現此接口來提供自動化驗證碼解決功能。
    """

    @abstractmethod
    def solve(self, challenge: ChallengeDetection, driver: Any) -> SolveResult:
        """
        解決驗證碼

        Args:
            challenge: 檢測到的驗證信息
            driver: Selenium WebDriver 實例（用於注入 token）

        Returns:
            SolveResult: 解決結果（包含 token 或錯誤信息）
        """
        pass
