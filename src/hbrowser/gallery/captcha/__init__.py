"""
驗證碼處理模組

提供驗證碼檢測、解決和管理功能
"""
from .models import ChallengeDetection, SolveResult, Kind
from .solver_interface import CaptchaSolver
from .detector import CaptchaDetector
from .manager import CaptchaManager
from .adapters import TwoCaptchaAdapter

__all__ = [
    "ChallengeDetection",
    "SolveResult",
    "Kind",
    "CaptchaSolver",
    "CaptchaDetector",
    "CaptchaManager",
    "TwoCaptchaAdapter",
]
