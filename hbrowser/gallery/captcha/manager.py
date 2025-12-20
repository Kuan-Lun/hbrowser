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

    def detect(self, driver: Any, timeout: float = 2.0) -> ChallengeDetection:
        """
        檢測驗證碼

        Args:
            driver: Selenium WebDriver 實例
            timeout: 檢測超時時間（秒）

        Returns:
            ChallengeDetection: 檢測結果
        """
        return self.detector.detect(driver, timeout)

    def solve(self, challenge: ChallengeDetection, driver: Any) -> bool:
        """
        解決驗證碼

        Args:
            challenge: 檢測到的驗證信息
            driver: Selenium WebDriver 實例

        Returns:
            bool: 是否成功解決
        """
        if challenge.kind == "none":
            return True

        result = self.solver.solve(challenge, driver)
        return result.success

    def detect_and_solve(self, driver: Any, timeout: float = 2.0) -> bool:
        """
        檢測並解決驗證碼

        Args:
            driver: Selenium WebDriver 實例
            timeout: 檢測超時時間（秒）

        Returns:
            bool: 是否成功解決
        """
        challenge = self.detect(driver, timeout)

        if challenge.kind == "none":
            return True

        return self.solve(challenge, driver)
