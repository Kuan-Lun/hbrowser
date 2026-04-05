"""驗證碼管理器 - 協調檢測和解決"""

from typing import Any

from .detector import CaptchaDetector
from .models import ChallengeDetection
from .solver_interface import CaptchaSolver


class CaptchaManager:
    """驗證碼管理器 - 核心協調邏輯"""

    def __init__(self, solver: CaptchaSolver) -> None:
        """
        初始化驗證碼管理器

        Args:
            solver: 驗證碼解決器實例
        """
        self.solver = solver
        self.detector = CaptchaDetector()

    async def detect(self, page: Any, timeout: float = 2.0) -> ChallengeDetection:
        """
        檢測驗證碼

        Args:
            page: zendriver Tab 實例
            timeout: 檢測超時時間（秒）

        Returns:
            ChallengeDetection: 檢測結果
        """
        return await self.detector.detect(page, timeout)

    async def solve(self, challenge: ChallengeDetection, page: Any) -> bool:
        """
        解決驗證碼

        Args:
            challenge: 檢測到的驗證信息
            page: zendriver Tab 實例

        Returns:
            bool: 是否成功解決
        """
        if challenge.kind == "none":
            return True

        result = await self.solver.solve(challenge, page)
        return result.success

    async def detect_and_solve(self, page: Any, timeout: float = 2.0) -> bool:
        """
        檢測並解決驗證碼

        Args:
            page: zendriver Tab 實例
            timeout: 檢測超時時間（秒）

        Returns:
            bool: 是否成功解決
        """
        challenge = await self.detect(page, timeout)

        if challenge.kind == "none":
            return True

        return await self.solve(challenge, page)
