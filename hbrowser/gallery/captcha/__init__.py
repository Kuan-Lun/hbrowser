"""
驗證碼處理模組

提供驗證碼檢測、解決和管理功能
"""

from .adapters import TwoCaptchaAdapter
from .detector import CaptchaDetector
from .manager import CaptchaManager
from .models import ChallengeDetection, Kind, SolveResult
from .solver_interface import CaptchaSolver

__all__ = [
    "ChallengeDetection",
    "SolveResult",
    "Kind",
    "CaptchaSolver",
    "CaptchaDetector",
    "CaptchaManager",
    "TwoCaptchaAdapter",
]
